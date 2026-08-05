from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .distribution_pruning import DistributionPruningController
from .refresh_cache_forward import _prepare_embeddings, _suffix_logits


@dataclass(frozen=True, slots=True)
class DecoderParts:
    decoder: nn.Module
    embedding: nn.Embedding
    blocks: nn.ModuleList


def _decoder_parts(model: nn.Module) -> DecoderParts:
    decoder = getattr(model, "model", None)
    if not isinstance(decoder, nn.Module):
        raise TypeError("expected a Hugging Face LLaDA model with .model decoder")
    transformer = getattr(decoder, "transformer", None)
    if not isinstance(transformer, nn.ModuleDict):
        raise TypeError("LLaDA decoder transformer must be a ModuleDict")
    embedding = transformer["wte"]
    blocks = transformer["blocks"]
    if not isinstance(embedding, nn.Embedding):
        raise TypeError("LLaDA wte must be an Embedding")
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("distribution distillation requires ungrouped LLaDA blocks")
    return DecoderParts(decoder, embedding, blocks)


def decoder_embedding(model: nn.Module) -> nn.Embedding:
    return _decoder_parts(model).embedding


@torch.no_grad()
def extract_prompt_features(
    model: nn.Module,
    prompt_ids: torch.Tensor,
) -> torch.Tensor:
    """Return prompt-only layer inputs with shape [layer, prompt, hidden]."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("prompt ids must have shape [1, prompt]")
    parts = _decoder_parts(model)
    hidden = _prepare_embeddings(parts.decoder, parts.embedding(prompt_ids))
    features: list[torch.Tensor] = []
    for block in parts.blocks:
        features.append(hidden.squeeze(0).detach())
        hidden = _run_block(block, hidden)
    return torch.stack(features)


def run_suffix_logits(
    model: nn.Module,
    sequence_ids: torch.Tensor,
    *,
    prompt_length: int,
    controller: DistributionPruningController | None = None,
    layer_scores: torch.Tensor | None = None,
    checkpoint_blocks: bool = False,
) -> torch.Tensor:
    """Run a full-length state but materialize vocabulary logits for suffix only."""
    if sequence_ids.ndim != 2 or sequence_ids.shape[0] != 1:
        raise ValueError("sequence ids must have shape [1, sequence]")
    if not 0 < prompt_length < sequence_ids.shape[1]:
        raise ValueError("prompt length must leave a non-empty suffix")
    if (controller is None) != (layer_scores is None):
        raise ValueError("controller and layer scores must be provided together")
    parts = _decoder_parts(model)
    blocks = parts.blocks
    if layer_scores is not None and layer_scores.shape[0] != len(blocks):
        raise ValueError("selector score layers do not match decoder")

    hidden = _prepare_embeddings(parts.decoder, parts.embedding(sequence_ids))
    for layer_id, block in enumerate(blocks):
        if controller is None or layer_scores is None:
            hidden = _run_block(block, hidden)
            continue
        score = layer_scores[layer_id]
        if checkpoint_blocks:
            hidden = _checkpointed_block(
                block,
                controller,
                layer_id,
                hidden,
                score,
            )
        else:
            controller.set_layer_score(layer_id, score)
            hidden = _run_block(block, hidden)
    return _suffix_logits(parts.decoder, hidden[:, prompt_length:])


def _run_block(block: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    result = block(
        hidden,
        attention_bias=None,
        layer_past=None,
        use_cache=False,
    )
    if not isinstance(result, tuple) or not isinstance(result[0], torch.Tensor):
        raise TypeError("LLaDA block must return (hidden, cache)")
    return result[0]


def _checkpointed_block(
    block: nn.Module,
    controller: DistributionPruningController,
    layer_id: int,
    hidden: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    def forward(current_hidden: torch.Tensor, current_score: torch.Tensor) -> torch.Tensor:
        controller.set_layer_score(layer_id, current_score)
        return _run_block(block, current_hidden)

    result = checkpoint(
        forward,
        hidden,
        score,
        use_reentrant=False,
        preserve_rng_state=False,
    )
    if not isinstance(result, torch.Tensor):
        raise TypeError("checkpointed LLaDA block must return hidden tensor")
    return result


__all__ = ["decoder_embedding", "extract_prompt_features", "run_suffix_logits"]
