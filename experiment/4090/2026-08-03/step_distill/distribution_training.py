from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoModel, AutoTokenizer

from .distribution_artifacts import (
    MetricsWriter,
    result_record,
    save_checkpoint,
    write_record,
)
from .distribution_rollout import (
    DistributionRolloutConfig,
    DistributionRolloutResult,
    run_distribution_rollout,
)
from .distribution_student import SelectorConfig, StateConditionedSelector
from .generation_output import LLadaDecodeTokenizer
from .task_data import (
    OffsetTokenizer,
    SamsumStateSpan,
    TeacherSample,
    load_teacher_samples,
    tokenize_samsum_prompt,
)


class SamsumTokenizer(LLadaDecodeTokenizer, OffsetTokenizer, Protocol): ...


@dataclass(frozen=True, slots=True)
class OnlineTrainConfig:
    model_path: Path
    train_data: Path
    validation_data: Path
    output_dir: Path
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    max_length: int = 2048
    train_limit: int = 8
    validation_limit: int = 8
    epochs: int = 1
    max_rollout_steps: int = 0
    learning_rate: float = 1e-4
    projection_dim: int = 256
    mlp_dim: int = 512
    seed: int = 4090

    def __post_init__(self) -> None:
        if min(
            self.max_length,
            self.train_limit,
            self.validation_limit,
            self.epochs,
            self.projection_dim,
            self.mlp_dim,
        ) <= 0:
            raise ValueError("training counts and dimensions must be positive")
        if self.max_rollout_steps < 0 or self.learning_rate <= 0.0:
            raise ValueError("rollout limit must be non-negative and LR positive")


def run_training(config: OnlineTrainConfig) -> Path:
    torch.manual_seed(config.seed)
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = cast(
        SamsumTokenizer,
        AutoTokenizer.from_pretrained(
            str(config.model_path),
            trust_remote_code=True,
        ),
    )
    model_config = model.config
    selector_config = SelectorConfig(
        layer_count=int(model_config.n_layers),
        hidden_dim=int(model_config.d_model),
        projection_dim=config.projection_dim,
        mlp_dim=config.mlp_dim,
    )
    selector = StateConditionedSelector(selector_config).to(config.device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=config.learning_rate)
    rollout_config = DistributionRolloutConfig()
    train_samples = load_teacher_samples(config.train_data, limit=config.train_limit)
    validation_samples = load_teacher_samples(
        config.validation_data,
        limit=config.validation_limit,
    )
    _assert_disjoint(train_samples, validation_samples)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "metrics.jsonl").open(
        "w",
        encoding="utf-8",
    ) as metrics:
        _train_epochs(
            model,
            tokenizer,
            selector,
            optimizer,
            train_samples,
            config,
            rollout_config,
            metrics,
        )
        _validate(
            model,
            tokenizer,
            selector,
            validation_samples,
            config,
            rollout_config,
            metrics,
        )
    checkpoint_path = config.output_dir / "checkpoint.pt"
    train_config_record: dict[str, object] = {
        **asdict(config),
        "dtype": str(config.dtype),
    }
    save_checkpoint(
        checkpoint_path,
        selector,
        optimizer,
        selector_config,
        rollout_config,
        train_config_record,
        [sample.sample_id for sample in train_samples],
        [sample.sample_id for sample in validation_samples],
    )
    return checkpoint_path


def _train_epochs(
    model: torch.nn.Module,
    tokenizer: SamsumTokenizer,
    selector: StateConditionedSelector,
    optimizer: torch.optim.Optimizer,
    samples: list[TeacherSample],
    config: OnlineTrainConfig,
    rollout_config: DistributionRolloutConfig,
    metrics: MetricsWriter,
) -> None:
    for epoch in range(1, config.epochs + 1):
        selector.train()
        for sample_index, sample in enumerate(samples, start=1):
            result = _run_sample(
                model,
                tokenizer,
                selector,
                optimizer,
                sample,
                config,
                rollout_config,
            )
            record = result_record(
                result,
                split="train",
                sample_id=sample.sample_id,
                epoch=epoch,
                gold=sample.answer,
                tokenizer=tokenizer,
            )
            write_record(metrics, record)
            print(
                f"[train {sample_index}/{len(samples)}] "
                f"loss={record['loss']:.6f} grad={record['selector_grad_norm']:.6f}",
                flush=True,
            )
            _release_cuda()


def _validate(
    model: torch.nn.Module,
    tokenizer: SamsumTokenizer,
    selector: StateConditionedSelector,
    samples: list[TeacherSample],
    config: OnlineTrainConfig,
    rollout_config: DistributionRolloutConfig,
    metrics: MetricsWriter,
) -> None:
    selector.eval()
    for sample_index, sample in enumerate(samples, start=1):
        result = _run_sample(
            model,
            tokenizer,
            selector,
            None,
            sample,
            config,
            rollout_config,
        )
        record = result_record(
            result,
            split="validation",
            sample_id=sample.sample_id,
            epoch=config.epochs,
            gold=sample.answer,
            tokenizer=tokenizer,
        )
        write_record(metrics, record)
        print(
            f"[validation {sample_index}/{len(samples)}] "
            f"kl={record['kd_loss']:.6f} top1={record['token_top1_agreement']:.4f}",
            flush=True,
        )
        _release_cuda()


def _run_sample(
    model: torch.nn.Module,
    tokenizer: SamsumTokenizer,
    selector: StateConditionedSelector,
    optimizer: torch.optim.Optimizer | None,
    sample: TeacherSample,
    config: OnlineTrainConfig,
    rollout_config: DistributionRolloutConfig,
) -> DistributionRolloutResult:
    prompt = tokenize_samsum_prompt(
        tokenizer,
        sample,
        max_length=config.max_length,
        reserve_length=rollout_config.gen_length,
        state_span=SamsumStateSpan.TARGET_REQUEST,
    )
    prompt_ids = torch.tensor(
        [prompt.prompt_ids],
        dtype=torch.long,
        device=config.device,
    )
    return run_distribution_rollout(
        model,
        selector,
        prompt_ids,
        prompt.question_indices.to(config.device),
        rollout_config,
        optimizer=optimizer,
        max_steps=config.max_rollout_steps,
    )


def _assert_disjoint(
    train_samples: list[TeacherSample],
    validation_samples: list[TeacherSample],
) -> None:
    overlap = {sample.sample_id for sample in train_samples} & {
        sample.sample_id for sample in validation_samples
    }
    if overlap:
        raise ValueError(f"train/validation sample IDs overlap: {sorted(overlap)[:3]}")


def _release_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
__all__ = ["OnlineTrainConfig", "run_training"]
