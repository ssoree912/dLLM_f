"""CLI: compute the offline I(l)/P_h(l) calibration profile for layer+head budgets.

Runs one no-cache forward pass per calibration sample (prompt + mask_id-filled
response, step-0 convention -- see head_pref_profile.py) and averages the two
collectors' output into a JSON profile consumed by layer_head_budget.py.

Example:
  python -m dllm_cache.budget.compute_layer_head_profile \
    --data /workspace/dllm/data/longbench/2wikimqa.jsonl \
    --n-samples 64 --mask-length 128 \
    --output results/budget/layer_head_profile_2wikimqa.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_cache.budget.common import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    LongBenchSample,
    build_2wikimqa_prompt,
    encode_text,
    load_longbench_samples,
)
from dllm_cache.budget.head_pref_profile import install_layer_head_profile_collector


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    model_path: Path
    data_path: Path
    output_path: Path
    max_length: int
    mask_length: int
    mask_id: int
    n_samples: int
    device: str
    dtype: torch.dtype


def parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output", type=Path, default=Path("results/budget/layer_head_profile.json")
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--mask-length", type=int, default=128)
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()
    return ProfileConfig(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        max_length=args.max_length,
        mask_length=args.mask_length,
        mask_id=args.mask_id,
        n_samples=args.n_samples,
        device=args.device,
        dtype=parse_dtype(args.dtype),
    )


def parse_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case _:
            raise argparse.ArgumentTypeError(f"unsupported dtype: {dtype}")


def tokenize_prompt_for_profile(
    tokenizer,
    sample: LongBenchSample,
    max_length: int,
    mask_length: int,
) -> list[int]:
    prompt_text = build_2wikimqa_prompt(sample)
    prompt_ids_full = encode_text(tokenizer, prompt_text)
    prompt_cap = max(1, max_length - mask_length)
    truncation_offset = max(0, len(prompt_ids_full) - prompt_cap)
    return prompt_ids_full[truncation_offset:]


def load_model_and_tokenizer(config: ProfileConfig):
    print(f"[load] model={config.model_path} dtype={config.dtype} device={config.device}", flush=True)
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path), trust_remote_code=True)
    return model, tokenizer


@torch.inference_mode()
def accumulate_sample(
    model: torch.nn.Module,
    prompt_ids: list[int],
    config: ProfileConfig,
) -> tuple[list[float], list[list[float]]]:
    input_ids = torch.tensor(
        [prompt_ids + [config.mask_id] * config.mask_length],
        dtype=torch.long,
        device=config.device,
    )
    attention_mask = torch.ones_like(input_ids)
    collector = install_layer_head_profile_collector(
        model,
        prompt_length=len(prompt_ids),
        mask_length=config.mask_length,
    )
    collector.start_sample()
    try:
        model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        collector.finish_sample()
        result = collector.finalize()
    finally:
        collector.restore()
    return result["layer_importance"], result["head_preference"]


def main() -> int:
    config = parse_args()
    model, tokenizer = load_model_and_tokenizer(config)
    samples = load_longbench_samples(config.data_path, config.n_samples)
    if not samples:
        raise RuntimeError(f"no calibration samples loaded from {config.data_path}")

    layer_count: int | None = None
    importance_sum: list[float] | None = None
    preference_sum: list[list[float]] | None = None
    used = 0
    t0 = time.time()
    for index, sample in enumerate(samples, start=1):
        prompt_ids = tokenize_prompt_for_profile(
            tokenizer, sample, config.max_length, config.mask_length
        )
        if not prompt_ids:
            continue
        importance, preference = accumulate_sample(model, prompt_ids, config)
        if importance_sum is None:
            layer_count = len(importance)
            importance_sum = [0.0] * layer_count
            preference_sum = [[0.0] * len(preference[l]) for l in range(layer_count)]
        for l, value in enumerate(importance):
            importance_sum[l] += value
        for l, layer_pref in enumerate(preference):
            for h, value in enumerate(layer_pref):
                preference_sum[l][h] += value
        used += 1
        print(
            f"[profile {index}/{len(samples)}] prompt_len={len(prompt_ids)} "
            f"used={used} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    if importance_sum is None or preference_sum is None or layer_count is None:
        raise RuntimeError("no usable calibration samples (all prompts were empty)")

    profile = {
        "layer_importance": [total / used for total in importance_sum],
        "head_preference": [[total / used for total in layer] for layer in preference_sum],
        "meta": {
            "model_path": str(config.model_path),
            "data_path": str(config.data_path),
            "sample_count": used,
            "mask_length": config.mask_length,
            "mask_id": config.mask_id,
            "max_length": config.max_length,
            "convention": (
                "P_h(l) measured at diffusion step 0 (response fully masked); "
                "I(l) measured over prompt-token positions only"
            ),
        },
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(profile, indent=2))
    print(f"[done] samples={used} layers={layer_count} saved={config.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
