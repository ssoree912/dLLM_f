from __future__ import annotations

import json
from pathlib import Path

from step_distill.schema import teacher_artifact_path
from step_distill.task_data import (
    SAMSUM_INSTRUCTION,
    SamsumStateSpan,
    TeacherSample,
    load_teacher_samples,
    tokenize_2wiki_prompt,
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


def test_tokenize_2wiki_prompt_tracks_the_real_question_after_left_truncation() -> None:
    # Given
    sample = TeacherSample(
        sample_id="s1",
        dataset="2wikimultihopqa_train",
        question="Question: Who won?\n",
        context="x" * 100,
        answer="Ada",
    )

    # When
    tokenized = tokenize_2wiki_prompt(
        CharacterTokenizer(),
        sample,
        max_length=48,
        reserve_length=8,
    )
    question_text = "".join(
        chr(tokenized.prompt_ids[index]) for index in tokenized.question_indices
    )

    # Then
    assert question_text == "Who won?"
    assert len(tokenized.prompt_ids) == 40
    assert tokenized.truncation_offset > 0


def test_teacher_artifact_path_is_collision_safe_and_stays_under_root(
    tmp_path: Path,
) -> None:
    # Given / When
    first = teacher_artifact_path(tmp_path, "2wikimqa", "a/b")
    second = teacher_artifact_path(tmp_path, "2wikimqa", "a_b")

    # Then
    assert first != second
    assert first.parent == tmp_path / "2wikimqa"
    assert second.parent == tmp_path / "2wikimqa"


def test_tokenize_samsum_prompt_preserves_legacy_instruction_feature() -> None:
    # Given
    sample = TeacherSample(
        sample_id="dialogue-1",
        dataset="samsum",
        question="Summarize the dialogue into a short summary.",
        context="Dialogue: A: Hello\nB: Hi",
        answer="A and B greet each other.",
    )

    # When
    tokenized = tokenize_samsum_prompt(
        CharacterTokenizer(),
        sample,
        max_length=256,
        reserve_length=32,
    )
    query_feature = "".join(
        chr(tokenized.prompt_ids[index]) for index in tokenized.question_indices
    )

    # Then
    assert tokenized.prompt_text == (
        f"{SAMSUM_INSTRUCTION}\n\n{sample.context}\n\n{sample.question}"
    )
    assert query_feature == SAMSUM_INSTRUCTION


def test_samsum_initial_state_never_uses_retained_instruction_fragment() -> None:
    # Given
    sample = TeacherSample(
        sample_id="dialogue-2",
        dataset="samsum",
        question="Dialogue: A: Ready?\nB: Yes.\nSummary: ",
        context="x" * 80,
        answer="A asks whether B is ready.",
    )

    # When
    tokenized = tokenize_samsum_prompt(
        CharacterTokenizer(),
        sample,
        max_length=24,
        reserve_length=8,
        state_span=SamsumStateSpan.TARGET_REQUEST,
    )
    legacy = tokenize_samsum_prompt(
        CharacterTokenizer(),
        sample,
        max_length=24,
        reserve_length=8,
    )
    initial_state = "".join(
        chr(tokenized.prompt_ids[index]) for index in tokenized.question_indices
    )

    # Then
    assert tokenized.prompt_ids == legacy.prompt_ids
    assert initial_state == sample.question.strip()[-16:]


def test_load_teacher_samples_accepts_longbench_eval_input_field(
    tmp_path: Path,
) -> None:
    # Given
    path = tmp_path / "samsum.jsonl"
    path.write_text(
        json.dumps(
            {
                "_id": "eval-1",
                "dataset": "samsum",
                "context": "Dialogue: Example\nSummary: Example summary.",
                "input": "Dialogue: A: Hello\nB: Hi\nSummary: ",
                "answers": ["A and B greet each other."],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # When
    sample = load_teacher_samples(path)[0]

    # Then
    assert sample.question == "Dialogue: A: Hello\nB: Hi\nSummary: "
    assert sample.answer == "A and B greet each other."
