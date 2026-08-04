from __future__ import annotations

import json
from pathlib import Path

import pytest

from step_distill.prepare_samsum_fewshot import PrepareFewshotConfig, run_prepare
from step_distill.samsum_fewshot_data import (
    FewshotBuildSpec,
    FewshotDataError,
    FewshotPackRequest,
    pack_samsum_target,
    select_samsum_targets,
)
from step_distill.task_data import (
    TeacherSample,
    load_teacher_samples,
    tokenize_samsum_prompt,
)


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _samples(count: int) -> tuple[TeacherSample, ...]:
    return tuple(
        TeacherSample(
            sample_id=f"sample-{index}",
            dataset="samsum",
            question="Summarize.",
            context=f"Dialogue: speaker-{index}: " + "x" * 80,
            answer=f"summary-{index}",
        )
        for index in range(count)
    )


def test_select_samsum_targets_keeps_all_targets_out_of_demo_pool() -> None:
    # Given
    samples = _samples(8)
    spec = FewshotBuildSpec(
        max_length=128,
        reserve_length=16,
        train_count=3,
        validation_count=2,
        seed=4090,
    )

    # When
    selected = select_samsum_targets(samples, spec)

    # Then
    target_ids = {
        sample.sample_id for sample in (*selected.train, *selected.validation)
    }
    demo_ids = {sample.sample_id for sample in selected.demo_pool}
    assert target_ids.isdisjoint(demo_ids)
    assert len(selected.train) == 3
    assert len(selected.validation) == 2


def test_pack_samsum_target_fills_prompt_cap_without_target_as_demo() -> None:
    # Given
    tokenizer = CharacterTokenizer()
    spec = FewshotBuildSpec(
        max_length=128,
        reserve_length=16,
        train_count=1,
        validation_count=0,
        seed=4090,
    )
    target, *demos = _samples(5)

    # When
    packed = pack_samsum_target(
        FewshotPackRequest(
            tokenizer=tokenizer,
            target=target,
            demo_pool=tuple(demos),
            spec=spec,
        )
    )
    tokenized = tokenize_samsum_prompt(
        tokenizer,
        packed.teacher_sample,
        max_length=spec.max_length,
        reserve_length=spec.reserve_length,
    )

    # Then
    assert len(tokenized.prompt_ids) == spec.prompt_cap
    assert tokenized.truncation_offset > 0
    assert target.sample_id not in packed.demo_ids
    assert packed.prompt_length == spec.prompt_cap


def test_pack_samsum_target_is_deterministic_for_seed_and_target() -> None:
    # Given
    tokenizer = CharacterTokenizer()
    spec = FewshotBuildSpec(
        max_length=160,
        reserve_length=16,
        train_count=1,
        validation_count=0,
        seed=7,
    )
    target, *demos = _samples(7)
    request = FewshotPackRequest(tokenizer, target, tuple(demos), spec)

    # When
    first = pack_samsum_target(request)
    second = pack_samsum_target(request)

    # Then
    assert first == second


def test_pack_samsum_target_requires_actual_left_truncation_at_exact_cap() -> None:
    # Given
    tokenizer = CharacterTokenizer()
    target, demo = _samples(2)
    candidate = TeacherSample(
        sample_id=f"samsum-fewshot-{target.sample_id}",
        dataset="samsum",
        question=f"{target.context.strip()}\nSummary: ",
        context=f"{demo.context.strip()}\nSummary: {demo.answer.strip()}",
        answer=target.answer,
    )
    full_length = len(
        tokenize_samsum_prompt(
            tokenizer,
            candidate,
            max_length=10_000,
            reserve_length=16,
        ).prompt_ids
    )
    spec = FewshotBuildSpec(full_length + 16, 16, 1, 0, 4090)

    # When / Then
    with pytest.raises(FewshotDataError, match="could not fill prompt cap"):
        pack_samsum_target(FewshotPackRequest(tokenizer, target, (demo,), spec))


def test_run_prepare_writes_disjoint_loader_compatible_splits(
    tmp_path: Path,
) -> None:
    # Given
    source_path = tmp_path / "source.jsonl"
    rows = [
        {
            "_id": sample.sample_id,
            "dataset": sample.dataset,
            "context": sample.context,
            "question": sample.question,
            "answers": [sample.answer],
        }
        for sample in _samples(8)
    ]
    source_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    spec = FewshotBuildSpec(128, 16, 3, 2, 4090)
    config = PrepareFewshotConfig(
        source_path=source_path,
        eval_path=None,
        output_root=tmp_path / "out",
        tokenizer_id="character-test",
        spec=spec,
    )

    # When
    outputs = run_prepare(config, CharacterTokenizer())

    # Then
    assert outputs.train_path.suffix == ".xz"
    assert outputs.train_path.read_bytes().startswith(b"\xfd7zXZ\x00")
    train = load_teacher_samples(outputs.train_path)
    validation = load_teacher_samples(outputs.validation_path)
    assert len(train) == 3
    assert len(validation) == 2
    assert {row.sample_id for row in train}.isdisjoint(
        row.sample_id for row in validation
    )
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["prompt_cap"] == spec.prompt_cap
    assert manifest["train_count"] == 3
    assert manifest["validation_count"] == 2
