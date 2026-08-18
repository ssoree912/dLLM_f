from __future__ import annotations

import inspect
import types
from dataclasses import dataclass, field
from math import sqrt
from typing import Protocol

import torch
import torch.nn as nn


class NamedModuleModel(Protocol):
    def named_modules(self) -> list[tuple[str, nn.Module]]:
        ...


@dataclass(slots=True)
class AnswerAttentionCollector:
    prompt_length: int
    answer_start: int
    answer_end: int
    layer_count: int
    scores_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    originals: list[tuple[nn.Module, types.MethodType]] = field(default_factory=list)

    def capture(
        self,
        block: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_bias: torch.Tensor | None,
    ) -> None:
        if q.shape[0] != 1:
            raise RuntimeError("revealed-answer teacher extraction expects batch size 1")
        q_heads, k_heads = project_attention_heads(block, q, k)
        if q_heads.shape[1] != k_heads.shape[1]:
            k_heads = repeat_kv_heads(k_heads, q_heads.shape[1])
        answer_q = q_heads[:, :, self.answer_start : self.answer_end, :]
        prompt_k = k_heads[:, :, : self.prompt_length, :]
        if answer_q.shape[-2] == 0 or prompt_k.shape[-2] == 0:
            raise RuntimeError("empty answer or prompt span while collecting teacher score")
        raw_scores = torch.matmul(
            answer_q.float(),
            prompt_k.float().transpose(-1, -2),
        ) / sqrt(float(answer_q.shape[-1]))
        if attention_bias is not None:
            raw_scores = raw_scores + attention_bias[
                :, :, self.answer_start : self.answer_end, : self.prompt_length
            ].float()
        attention = torch.softmax(raw_scores, dim=-1)
        layer_score = attention.sum(dim=-2).mean(dim=1).squeeze(0)
        layer_id = int(getattr(block, "layer_id"))
        self.scores_by_layer[layer_id] = layer_score.detach().cpu()

    def scores_tensor(self) -> torch.Tensor:
        missing = sorted(set(range(self.layer_count)) - set(self.scores_by_layer))
        if missing:
            raise RuntimeError(f"teacher scores missing layers: {missing}")
        return torch.stack(
            [self.scores_by_layer[layer_id] for layer_id in range(self.layer_count)],
            dim=0,
        )

    def restore(self) -> None:
        for module, original in self.originals:
            module.attention = original


def project_attention_heads(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, q_len, channels = q.size()
    _, k_len, _ = k.size()
    dtype = k.dtype
    q_norm = getattr(block, "q_norm", None)
    k_norm = getattr(block, "k_norm", None)
    if q_norm is not None and k_norm is not None:
        q = q_norm(q).to(dtype=dtype)
        k = k_norm(k).to(dtype=dtype)
    config = getattr(block, "config")
    head_dim = channels // int(config.n_heads)
    q_heads = q.view(batch_size, q_len, int(config.n_heads), head_dim).transpose(1, 2)
    k_heads = k.view(
        batch_size,
        k_len,
        int(config.effective_n_kv_heads),
        head_dim,
    ).transpose(1, 2)
    if bool(config.rope):
        q_heads, k_heads = block.rotary_emb(q_heads, k_heads)
    return q_heads, k_heads


def repeat_kv_heads(k_heads: torch.Tensor, q_head_count: int) -> torch.Tensor:
    kv_head_count = k_heads.shape[1]
    if q_head_count % kv_head_count != 0:
        raise RuntimeError("query head count must be divisible by KV head count")
    return k_heads.repeat_interleave(q_head_count // kv_head_count, dim=1)


def install_answer_attention_collector(
    model: NamedModuleModel,
    prompt_length: int,
    answer_length: int,
) -> AnswerAttentionCollector:
    blocks = find_transformer_blocks(model)
    collector = AnswerAttentionCollector(
        prompt_length=prompt_length,
        answer_start=prompt_length,
        answer_end=prompt_length + answer_length,
        layer_count=len(blocks),
    )
    for block in blocks:
        original = block.attention
        accepts_block_mask = "block_mask" in inspect.signature(original).parameters
        block.attention = types.MethodType(
            make_wrapped_attention(collector, original, accepts_block_mask),
            block,
        )
        collector.originals.append((block, original))
    return collector


def make_wrapped_attention(
    collector: AnswerAttentionCollector,
    original: types.MethodType,
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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        collector.capture(block_self, q, k, attention_bias)
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


def find_transformer_blocks(model: NamedModuleModel) -> nn.ModuleList:
    for name, module in model.named_modules():
        if name == "model.transformer.blocks":
            if not isinstance(module, nn.ModuleList):
                raise RuntimeError("model.transformer.blocks is not a ModuleList")
            return module
    raise RuntimeError("could not find model.transformer.blocks")
