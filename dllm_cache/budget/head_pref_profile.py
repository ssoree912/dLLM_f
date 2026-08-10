"""Offline calibration profiles for the two-stage layer+head prompt-KV budget split.

  I(l)   -- layer importance: 1 - mean_i cos(h_in_i, h_out_i) over prompt-token
            positions, measured at prefill (the block's residual input vs. its
            output after attention+FFN).
  P_h(l) -- head prompt-preference: S(mask->prompt) / (S(mask->prompt)+S(mask->mask))
            per head, evaluated at diffusion step 0 (the whole response region is
            still the mask token) -- no denoising trajectory needed to compute it.

Both come from a single forward pass per calibration sample: prompt tokens followed
by `mask_id`-filled response positions, no KV cache, no generation. They're averaged
over a calibration set on the premise the two-stage split relies on: layer/head
importance ranking is consistent across samples, so a profile computed once offline
can drive per-sample budget allocation at inference time.

Not named `attention_*` on purpose -- that pattern is gitignored in this tree
(see .gitignore:191) and silently drops the file from disk.
"""

from __future__ import annotations

import inspect
import math
import types
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_cache.budget.attention_teacher import (
    NamedModuleModel,
    find_transformer_blocks,
    project_attention_heads,
    repeat_kv_heads,
)


class LayerHeadProfileError(RuntimeError):
    pass


@dataclass(slots=True)
class LayerHeadProfileCollector:
    prompt_length: int
    mask_length: int
    layer_count: int
    importance_sum: list[float] = field(default_factory=list)
    preference_sum: list[list[float]] = field(default_factory=list)
    sample_count: int = 0
    pending_preference: dict[int, list[float]] = field(default_factory=dict)
    attn_originals: list[tuple[nn.Module, types.MethodType]] = field(default_factory=list)
    hidden_handles: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.importance_sum = [0.0] * self.layer_count
        self.preference_sum = [[] for _ in range(self.layer_count)]

    @property
    def sequence_length(self) -> int:
        return self.prompt_length + self.mask_length

    def start_sample(self) -> None:
        self.pending_preference = {}

    def hidden_hook(self, layer_id: int):
        def hook(module: nn.Module, args: tuple, output: object) -> None:
            x_in = args[0]
            x_out = output[0] if isinstance(output, tuple) else output
            if x_in.shape[1] != self.sequence_length:
                raise LayerHeadProfileError(
                    f"layer {layer_id}: expected seq_len {self.sequence_length}, "
                    f"got {x_in.shape[1]}"
                )
            h_in = x_in[:, : self.prompt_length, :].float()
            h_out = x_out[:, : self.prompt_length, :].float()
            cos = F.cosine_similarity(h_in, h_out, dim=-1)
            self.importance_sum[layer_id] += float((1.0 - cos).mean().item())

        return hook

    def attention_capture(
        self,
        block: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_bias: torch.Tensor | None,
    ) -> None:
        if q.shape[0] != 1 or q.shape[1] != self.sequence_length:
            raise LayerHeadProfileError(
                "head-preference collector expects batch size 1 and "
                f"sequence length {self.sequence_length}"
            )
        q_heads, k_heads = project_attention_heads(block, q, k)
        if q_heads.shape[1] != k_heads.shape[1]:
            k_heads = repeat_kv_heads(k_heads, q_heads.shape[1])
        raw_scores = torch.matmul(
            q_heads.float(), k_heads.float().transpose(-1, -2)
        ) / math.sqrt(float(q_heads.shape[-1]))
        if attention_bias is not None:
            raw_scores = raw_scores + attention_bias[
                :, :, : self.sequence_length, : self.sequence_length
            ].float()
        attention = torch.softmax(raw_scores, dim=-1)
        mask_to_prompt = attention[
            :, :, self.prompt_length : self.sequence_length, : self.prompt_length
        ]
        mask_to_mask = attention[
            :,
            :,
            self.prompt_length : self.sequence_length,
            self.prompt_length : self.sequence_length,
        ]
        s_prompt = mask_to_prompt.sum(dim=(0, 2, 3))
        s_mask = mask_to_mask.sum(dim=(0, 2, 3))
        preference = (s_prompt / (s_prompt + s_mask).clamp_min(1e-12)).detach().cpu().tolist()
        layer_id = int(getattr(block, "layer_id"))
        self.pending_preference[layer_id] = preference

    def finish_sample(self) -> None:
        missing = sorted(set(range(self.layer_count)) - set(self.pending_preference))
        if missing:
            raise LayerHeadProfileError(f"head preference missing layers: {missing}")
        for layer_id, preference in self.pending_preference.items():
            if not self.preference_sum[layer_id]:
                self.preference_sum[layer_id] = [0.0] * len(preference)
            for head_id, value in enumerate(preference):
                self.preference_sum[layer_id][head_id] += value
        self.sample_count += 1
        self.pending_preference = {}

    def finalize(self) -> dict[str, object]:
        if self.sample_count == 0:
            raise LayerHeadProfileError("no calibration samples were accumulated")
        importance = [total / self.sample_count for total in self.importance_sum]
        preference = [
            [total / self.sample_count for total in layer] for layer in self.preference_sum
        ]
        return {
            "layer_importance": importance,
            "head_preference": preference,
            "sample_count": self.sample_count,
        }

    def restore(self) -> None:
        for module, original in self.attn_originals:
            module.attention = original
        self.attn_originals.clear()
        for handle in self.hidden_handles:
            handle.remove()
        self.hidden_handles.clear()


def install_layer_head_profile_collector(
    model: NamedModuleModel,
    prompt_length: int,
    mask_length: int,
) -> LayerHeadProfileCollector:
    blocks = find_transformer_blocks(model)
    collector = LayerHeadProfileCollector(
        prompt_length=prompt_length,
        mask_length=mask_length,
        layer_count=len(blocks),
    )
    for layer_id, block in enumerate(blocks):
        collector.hidden_handles.append(block.register_forward_hook(collector.hidden_hook(layer_id)))

        original = block.attention
        accepts_block_mask = "block_mask" in inspect.signature(original).parameters
        wrapped = _make_wrapped_attention(collector, original, accepts_block_mask)
        block.attention = types.MethodType(wrapped, block)
        collector.attn_originals.append((block, original))
    return collector


def _make_wrapped_attention(
    collector: LayerHeadProfileCollector,
    original,
    accepts_block_mask: bool,
):
    def wrapped_attention(
        block_self: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        block_mask: torch.Tensor | None = None,
    ):
        collector.attention_capture(block_self, q, k, attention_bias)
        if accepts_block_mask:
            return original(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                block_mask=block_mask,
            )
        return original(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

    return wrapped_attention
