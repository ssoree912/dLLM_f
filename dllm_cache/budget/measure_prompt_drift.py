"""Run full LongBench generations and dump per-prompt-token drift statistics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_cache.budget.prompt_drift import install_prompt_drift_collector
from utils.generate_function import generate


@dataclass(frozen=True, slots=True)
class DriftConfig:
    model_path: Path
    data_path: Path
    output_dir: Path
    device: str
    dtype: torch.dtype
    limit: int
    max_length: int
    gen_length: int
    steps: int
    block_length: int
    repeats: int


SAMSUM_INSTRUCTION = (
    "Summarize the dialogue into a few short sentences. "
    "The following are some examples."
)


def build_prompt(row: dict) -> str:
    context = str(row["context"]).strip()
    request = str(row.get("input", row.get("question", ""))).strip()
    return f"{SAMSUM_INSTRUCTION}\n\n{context}\n{request}"


def load_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit > 0 and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def run(config: DriftConfig) -> Path:
    model = AutoModel.from_pretrained(
        str(config.model_path), trust_remote_code=True, torch_dtype=config.dtype
    ).to(config.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model_path), trust_remote_code=True
    )
    rows = load_rows(config.data_path, config.limit)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_cap = config.max_length - config.gen_length

    for index, row in enumerate(rows, start=1):
        encoded = tokenizer(build_prompt(row), add_special_tokens=False)["input_ids"]
        prompt_ids = torch.tensor(
            [[int(token) for token in encoded][-prompt_cap:]],
            dtype=torch.long,
            device=config.device,
        )
        prompt_length = int(prompt_ids.shape[1])
        for repeat in range(config.repeats):
            # Re-seeding per repeat lets us ask whether drift is a property of the
            # prompt or of whatever happened to be generated.
            torch.manual_seed(1234 + repeat)
            collector = install_prompt_drift_collector(model, prompt_length)
            try:
                with torch.inference_mode():
                    generate(
                        input_ids=prompt_ids,
                        attention_mask=torch.ones_like(prompt_ids),
                        model=model,
                        steps=config.steps,
                        gen_length=config.gen_length,
                        block_length=config.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                    )
                stats = collector.result()
            finally:
                collector.restore()
            payload = {
                "sample_id": row.get("_id", f"sample_{index}"),
                "repeat": repeat,
                "prompt_length": prompt_length,
                "prompt_input_ids": prompt_ids.squeeze(0).cpu(),
                **{name: tensor.cpu() for name, tensor in stats.items()},
            }
            path = config.output_dir / f"drift_{index:04d}_r{repeat}.pt"
            torch.save(payload, path)
            print(
                f"[drift {index}/{len(rows)} r{repeat}] prompt={prompt_length} "
                f"cum={stats['cumulative'].mean():.4f} "
                f"step={stats['stepwise'].mean():.4f}",
                flush=True,
            )
            torch.cuda.empty_cache()
    return config.output_dir


def parse_args(argv: Sequence[str] | None = None) -> DriftConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="generations per prompt; >1 tests whether drift reproduces",
    )
    args = parser.parse_args(argv)
    return DriftConfig(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],
        limit=args.limit,
        max_length=args.max_length,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        repeats=args.repeats,
    )


def main(argv: Sequence[str] | None = None) -> int:
    print(f"[done] {run(parse_args(argv))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
