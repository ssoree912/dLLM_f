import types
from typing import Optional, Tuple

import torch
import torch.nn as nn

from dllm_cache.cache import dLLMCache
from dllm_cache.maskkv import (
    MaskKVAttentionRequest,
    maskkv_scaled_dot_product_attention,
)

from .cache_hook_LLaDA import (
    logout_cache_LLaDA,
    register_cache_LLaDA as register_base_cache_LLaDA,
)


def register_cache_LLaDA(model: nn.Module, tf_block_module_key_name: str) -> None:
    register_base_cache_LLaDA(model, tf_block_module_key_name)
    feature_cache = dLLMCache()
    if not feature_cache.maskkv_enabled:
        return

    target_module: Optional[nn.ModuleList] = None
    for name, module in model.named_modules():
        if name == tf_block_module_key_name:
            target_module = module
            break
    if target_module is None:
        return
    for tf_block in target_module:
        tf_block.attention = types.MethodType(_maskkv_attention, tf_block)


def _maskkv_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    q_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    batch_size, q_len, channels = q.size()
    _, k_len, _ = k.size()
    dtype = k.dtype
    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)
    q = q.view(batch_size, q_len, self.config.n_heads, channels // self.config.n_heads)
    q = q.transpose(1, 2)
    k = k.view(
        batch_size,
        k_len,
        self.config.effective_n_kv_heads,
        channels // self.config.n_heads,
    ).transpose(1, 2)
    v = v.view(
        batch_size,
        k_len,
        self.config.effective_n_kv_heads,
        channels // self.config.n_heads,
    ).transpose(1, 2)
    if layer_past is not None:
        past_key, past_value = layer_past
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)
    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]
    if self.config.rope:
        q, k = self.rotary_emb(q, k, q_index=q_index)
    if attention_bias is not None:
        attention_bias = self._cast_attn_bias(
            attention_bias[:, :, key_len - query_len : key_len, :key_len],
            dtype,
        )

    feature_cache = dLLMCache()
    att = None
    if attention_bias is None:
        request = MaskKVAttentionRequest(
            q=q,
            k=k,
            v=v,
            mask_index=feature_cache.get_mask_index(),
            prompt_length=feature_cache.prompt_length,
            layer_id=self.layer_id,
            layer_count=self.config.n_layers,
            budget=feature_cache.maskkv_budget,
            layer_base_rate=feature_cache.maskkv_layer_base_rate,
            head_base_rate=feature_cache.maskkv_head_base_rate,
        )
        att = maskkv_scaled_dot_product_attention(request)
    if att is None:
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )
    att = att.transpose(1, 2).contiguous().view(batch_size, q_len, channels)
    return self.attn_out(att), present
