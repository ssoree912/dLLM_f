from __future__ import annotations

import inspect
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from dllm_cache.budget.drift_refresh_kv import generate_with_drift_refresh
from dllm_cache.budget.extract_prune_cache_teacher import validate_no_chat_sources
from dllm_cache.budget.offline_hybrid_teacher import OfflineHybridTeacherConfig
from dllm_cache.budget.prune_cache import (
    PRUNE_CACHE_HEADS,
    resolve_generation_kwargs,
    split_prune_cache_budget,
    validate_prune_cache_heads,
)
from dllm_cache.budget.student_model import PromptUtilityStudent, StudentConfig
from dllm_cache.budget.train_prune_cache_student import fixed_train_config


@pytest.mark.parametrize(
    ("gen_length", "prompt_tokens", "kept_tokens", "updated_tokens"),
    [
        (32, 2016, 1008, 504),
        (64, 1984, 992, 496),
        (128, 1920, 960, 480),
        (512, 1536, 768, 384),
    ],
)
def test_budget_uses_prompt_after_generation_reservation(
    gen_length: int,
    prompt_tokens: int,
    kept_tokens: int,
    updated_tokens: int,
) -> None:
    assert prompt_tokens == 2048 - gen_length
    budget = split_prune_cache_budget(prompt_tokens)
    assert budget.kept_tokens == kept_tokens
    assert budget.updated_tokens == updated_tokens


def test_budget_rounds_up_for_odd_nonempty_prompts() -> None:
    assert split_prune_cache_budget(1).kept_tokens == 1
    assert split_prune_cache_budget(1).updated_tokens == 1
    assert split_prune_cache_budget(5).kept_tokens == 3
    assert split_prune_cache_budget(5).updated_tokens == 2
    with pytest.raises(RuntimeError, match="positive"):
        split_prune_cache_budget(0)


@pytest.mark.parametrize("gen_length", [32, 64, 128, 512])
def test_official_task_output_length_drives_generation(gen_length: int) -> None:
    resolved = resolve_generation_kwargs({"max_gen_toks": gen_length})
    assert resolved["gen_length"] == gen_length
    assert resolved["steps"] == gen_length
    assert resolved["block_length"] == 32


def test_prune_cache_requires_the_joint_two_head_student() -> None:
    validate_prune_cache_heads(PRUNE_CACHE_HEADS)
    with pytest.raises(RuntimeError, match="attention and delta"):
        validate_prune_cache_heads(("score",))


def test_teacher_defaults_are_no_topk_confidence_weighted_max() -> None:
    config = OfflineHybridTeacherConfig()
    assert config.active_top_k == 0
    assert config.confidence_weight is True
    assert config.target_aggregation == "max"


def test_delta_update_set_is_frozen_by_default() -> None:
    default = inspect.signature(generate_with_drift_refresh).parameters[
        "delta_select_once"
    ].default
    assert default is True


def test_prune_cache_teacher_rejects_chat_wrapped_source(tmp_path: Path) -> None:
    dataset = tmp_path / "samsum"
    dataset.mkdir()
    torch.save({"apply_chat_template": 1}, dataset / "chat.pt")
    with pytest.raises(RuntimeError, match="must be no-chat"):
        validate_no_chat_sources(tmp_path, ["samsum"], n_samples=0)


def test_prune_cache_training_is_always_joint_two_head() -> None:
    config = fixed_train_config(
        Namespace(
            teacher_root=Path("teacher"),
            output_dir=Path("output"),
            model=Path("model"),
            datasets=["samsum"],
            n_samples=300,
            device="cuda:0",
            dtype="bfloat16",
        )
    )
    assert config.target_mode == "attention_delta"
    assert config.loss_mode == "mse"
    assert config.rank_weight == 0.1
    assert config.topk_weight == 0.0


def test_joint_student_exposes_both_scorer_heads() -> None:
    student = PromptUtilityStudent(
        StudentConfig(
            layer_count=1,
            hidden_dim=4,
            proj_dim=2,
            mlp_dim=3,
            heads=PRUNE_CACHE_HEADS,
        )
    )
    hidden = torch.randn(1, 5, 4)
    prompt_indices = torch.arange(5)
    question_indices = torch.tensor([3, 4])
    attention = student.forward_layer(
        0,
        hidden,
        prompt_indices,
        question_indices,
        head="attention",
    )
    delta = student.forward_layer(
        0,
        hidden,
        prompt_indices,
        question_indices,
        head="delta",
    )
    assert attention.shape == (1, 5)
    assert delta.shape == (1, 5)
    assert not torch.equal(attention, delta)
