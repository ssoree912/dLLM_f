from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def teacher_artifact_path(output_root: Path, dataset: str, sample_id: str) -> Path:
    """Build a collision-resistant path without trusting dataset or sample identifiers."""
    dataset_component = _safe_component(dataset, fallback="dataset")
    sample_component = _safe_component(sample_id, fallback="sample")[:80]
    digest = sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return output_root / dataset_component / f"{sample_component}-{digest}.pt"


def _safe_component(value: str, *, fallback: str) -> str:
    component = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    component = component.strip("._") or fallback
    if component != value:
        digest = sha256(value.encode("utf-8")).hexdigest()[:8]
        return f"{component[:64]}-{digest}"
    return component[:80]
