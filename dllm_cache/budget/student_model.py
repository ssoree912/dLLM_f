from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class StudentConfig:
    layer_count: int = 32
    hidden_dim: int = 4096
    proj_dim: int = 256
    mlp_dim: int = 512
    heads: tuple[str, ...] = ("score",)


class PromptUtilityStudentLayer(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.heads = normalize_heads(config.heads)
        self.token_proj = nn.Linear(config.hidden_dim, config.proj_dim)
        self.question_proj = nn.Linear(config.hidden_dim, config.proj_dim)
        if self.heads == ("score",):
            self.score_head = build_score_head(config)
        else:
            self.score_heads = nn.ModuleDict(
                {head: build_score_head(config) for head in self.heads}
            )

    def select_head(self, head: str | None) -> nn.Module:
        if hasattr(self, "score_heads"):
            selected = head or self.heads[0]
            if selected not in self.score_heads:
                raise RuntimeError(f"student head {selected!r} not found in {self.heads}")
            return self.score_heads[selected]
        return self.score_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_indices: torch.Tensor,
        question_indices: torch.Tensor,
        head: str | None = None,
    ) -> torch.Tensor:
        prompt_hidden = hidden_states.index_select(dim=1, index=prompt_indices)
        question_hidden = hidden_states.index_select(dim=1, index=question_indices)
        question_pool = question_hidden.mean(dim=1)
        token_proj = self.token_proj(prompt_hidden)
        question_proj = self.question_proj(question_pool).unsqueeze(1)
        question_proj = question_proj.expand(-1, token_proj.shape[1], -1)
        fused = torch.cat(
            [token_proj, question_proj, token_proj * question_proj],
            dim=-1,
        )
        return self.select_head(head)(fused).squeeze(-1)


def normalize_heads(heads: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(heads, str):
        normalized = (heads,)
    else:
        normalized = tuple(heads)
    if not normalized:
        raise RuntimeError("student must define at least one head")
    if len(set(normalized)) != len(normalized):
        raise RuntimeError(f"duplicate student heads: {normalized}")
    return normalized


def build_score_head(config: StudentConfig) -> nn.Sequential:
    return nn.Sequential(
            nn.Linear(config.proj_dim * 3, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, 1),
    )


class PromptUtilityStudent(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        heads = normalize_heads(config.heads)
        if heads != config.heads:
            config = StudentConfig(
                layer_count=config.layer_count,
                hidden_dim=config.hidden_dim,
                proj_dim=config.proj_dim,
                mlp_dim=config.mlp_dim,
                heads=heads,
            )
        self.config = config
        self.layer_indices = tuple(range(config.layer_count))
        self.layers = nn.ModuleDict(
            {
                str(layer_id): PromptUtilityStudentLayer(config)
                for layer_id in self.layer_indices
            }
        )

    def forward_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        prompt_indices: torch.Tensor,
        question_indices: torch.Tensor,
        head: str | None = None,
    ) -> torch.Tensor:
        return self.layers[str(layer_id)](
            hidden_states,
            prompt_indices,
            question_indices,
            head=head,
        )

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "layer_count": self.config.layer_count,
            "hidden_dim": self.config.hidden_dim,
            "proj_dim": self.config.proj_dim,
            "mlp_dim": self.config.mlp_dim,
            "heads": list(normalize_heads(self.config.heads)),
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float,
    top_ratio: float,
    bottom_ratio: float,
) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 2:
        raise RuntimeError("pred and target must have matching [batch, prompt] shape")
    losses: list[torch.Tensor] = []
    for batch_index in range(pred.shape[0]):
        target_row = target[batch_index]
        pred_row = pred[batch_index]
        top_count = max(1, int(round(target_row.numel() * top_ratio)))
        bottom_count = max(1, int(round(target_row.numel() * bottom_ratio)))
        top_idx = torch.topk(target_row, top_count, largest=True).indices
        bottom_idx = torch.topk(target_row, bottom_count, largest=False).indices
        pred_top = pred_row.index_select(0, top_idx).unsqueeze(1)
        pred_bottom = pred_row.index_select(0, bottom_idx).unsqueeze(0)
        losses.append(torch.clamp(pred_bottom - pred_top + margin, min=0.0).mean())
    return torch.stack(losses).mean()


def student_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    rank_weight: float,
    rank_margin: float,
    rank_top_ratio: float,
    rank_bottom_ratio: float,
    topk_weight: float = 0.0,
    topk_k: int = 128,
    topk_positive_weight: float = 8.0,
    loss_mode: str = "mse",
    bce_positive_weight: float = 1.0,
    rank_input: str = "prob",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = scores.float()
    target = target.float()
    match loss_mode:
        case "mse":
            pred = F.softmax(logits, dim=-1)
            primary = F.mse_loss(pred, target)
        case "bce":
            pred = torch.sigmoid(logits)
            primary = soft_label_bce_loss(logits, target, bce_positive_weight)
        case _:
            raise RuntimeError(f"unsupported student loss_mode: {loss_mode}")
    match rank_input:
        case "prob":
            rank_pred = pred
        case "logit":
            rank_pred = logits
        case _:
            raise RuntimeError(f"unsupported rank_input: {rank_input}")
    rank = pairwise_ranking_loss(
        rank_pred,
        target,
        margin=rank_margin,
        top_ratio=rank_top_ratio,
        bottom_ratio=rank_bottom_ratio,
    )
    topk = topk_bce_loss(logits, target, topk_k, topk_positive_weight)
    return primary + rank_weight * rank + topk_weight * topk, primary, rank, topk


def soft_label_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 2:
        raise RuntimeError("logits and target must have matching [batch, prompt] shape")
    pos_weight = torch.full((), positive_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits.float(),
        target.float().clamp(0.0, 1.0),
        pos_weight=pos_weight,
    )


def topk_bce_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    k: int,
    positive_weight: float,
) -> torch.Tensor:
    if scores.shape != target.shape or scores.ndim != 2:
        raise RuntimeError("scores and target must have matching [batch, prompt] shape")
    labels = torch.zeros_like(target.float())
    top_count = min(max(1, k), target.shape[-1])
    top_idx = torch.topk(target.float(), top_count, dim=-1, largest=True).indices
    labels.scatter_(dim=-1, index=top_idx, value=1.0)
    pos_weight = torch.full((), positive_weight, dtype=scores.dtype, device=scores.device)
    return F.binary_cross_entropy_with_logits(scores.float(), labels, pos_weight=pos_weight)
