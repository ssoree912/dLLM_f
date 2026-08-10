"""Time prefill and decoding separately for each prompt-budget method.

End-to-end wall clock divided by sample count folds in model loading, harness
setup and dataset preparation. That overhead is a rounding error for a 27s/sample
method and 5-7% for a 4.5s one, which is enough to reorder the fast end of the
table. This times only the two phases that the methods actually change, with a
warmup and explicit CUDA synchronisation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_cache.budget.layer_split_prompt_kv import (
    build_layer_split_prompt_cache,
    generate_with_layer_split_prompt_kv,
)
from dllm_cache.budget.prompt_kv_cache import build_prompt_kv_cache
from dllm_cache.budget.prompt_kv_generate import generate_with_prompt_kv
from dllm_cache.budget.dynamic_prompt_kv import (
    build_dynamic_prompt_kv_cache,
    generate_with_dynamic_prompt_kv,
)
from dllm_cache.budget.student_model import PromptUtilityStudent, StudentConfig
from dllm_cache.cache import dLLMCache
from dllm_cache.hooks import register_cache_LLaDA
from utils.generate_function import generate

SAMSUM_INSTRUCTION = (
    "Summarize the dialogue into a few short sentences. "
    "The following are some examples."
)


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    model_path: Path
    data_path: Path
    student_path: Path
    output_path: Path
    methods: tuple[str, ...]
    samples: int
    warmup: int
    budget: int
    frozen_layers: int
    refresh_tokens: int
    measured_tokens: int
    refresh_interval: int
    gen_length: int
    steps: int
    block_length: int
    max_length: int
    device: str


def load_student(directory: Path, device: str) -> PromptUtilityStudent:
    config = StudentConfig(**json.loads((directory / "config.json").read_text()))
    student = PromptUtilityStudent(config)
    student.load_state_dict(
        torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    )
    return student.to(device).eval()


@torch.no_grad()
def student_scores(model, student, prompt_ids: torch.Tensor) -> torch.Tensor:
    output = model(
        prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    length = int(prompt_ids.shape[1])
    positions = torch.arange(length, dtype=torch.long, device=prompt_ids.device)
    window = min(128, length)
    question = torch.arange(
        length - window, length, dtype=torch.long, device=prompt_ids.device
    )
    return torch.stack(
        [
            torch.softmax(
                student.forward_layer(
                    layer_id, output.hidden_states[layer_id].float(), positions, question
                ).float(),
                dim=-1,
            ).squeeze(0)
            for layer_id in student.layer_indices
        ]
    )


def elapsed(function) -> tuple[float, object]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return time.perf_counter() - started, result


def run_method(
    name: str,
    model,
    student,
    prompt_ids: torch.Tensor,
    config: LatencyConfig,
) -> tuple[float, float]:
    """Return (prefill seconds, decode seconds) for one sample."""
    gen = dict(
        steps=config.steps,
        gen_length=config.gen_length,
        block_length=config.block_length,
        temperature=0.0,
        cfg_scale=0.0,
    )
    if name in {"origin", "dllm_cache"}:
        # dllm_cache uses the same generate() path; the difference is that the
        # feature cache config + hooks are installed once in main() when
        # "dllm_cache" is requested. origin runs without hooks -> full compute.
        decode, _ = elapsed(
            lambda: generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                model=model,
                **gen,
            )
        )
        return 0.0, decode

    prefill, scores = elapsed(lambda: student_scores(model, student, prompt_ids))

    if name == "kv_cache":
        build, cache = elapsed(
            lambda: build_prompt_kv_cache(
                model, prompt_ids, budget=config.budget, teacher_scores=scores
            )
        )
        decode, _ = elapsed(
            lambda: generate_with_prompt_kv(
                input_ids=prompt_ids, model=model, prompt_cache=cache, **gen
            )
        )
    elif name == "dynkv":
        build, cache = elapsed(
            lambda: build_dynamic_prompt_kv_cache(
                model,
                prompt_ids,
                budget=config.budget,
                teacher_scores=scores,
                selection_mode="global",
            )
        )
        decode, _ = elapsed(
            lambda: generate_with_dynamic_prompt_kv(
                input_ids=prompt_ids,
                model=model,
                prompt_cache=cache,
                refresh_interval=config.refresh_interval,
                **gen,
            )
        )
    elif name in {"layersplit", "measured"}:
        measured = config.measured_tokens if name == "measured" else 0
        refresh = config.refresh_tokens if name == "layersplit" else 0
        build, cache = elapsed(
            lambda: build_layer_split_prompt_cache(
                model,
                prompt_ids,
                budget=config.budget,
                teacher_scores=scores,
                frozen_layers=config.frozen_layers,
                refresh_tokens=refresh,
                measured_tokens=measured,
            )
        )
        decode, _ = elapsed(
            lambda: generate_with_layer_split_prompt_kv(
                input_ids=prompt_ids, model=model, prompt_cache=cache, **gen
            )
        )
    else:
        raise RuntimeError(f"unsupported method: {name}")
    return prefill + build, decode


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    model = AutoModel.from_pretrained(
        str(config.model_path), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(config.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model_path), trust_remote_code=True
    )
    student = load_student(config.student_path, config.device)

    # dLLM-Cache (feature cache) needs its config + forward hooks installed once.
    # Only do this when requested, and run "dllm_cache" as its own invocation so
    # the hooks never contaminate origin/dynkv/layersplit measurements.
    if "dllm_cache" in config.methods:
        dLLMCache.new_instance(
            prompt_interval_steps=100,
            gen_interval_steps=8,
            cfg_interval_steps=1,
            transfer_ratio=0.25,
        )
        register_cache_LLaDA(model, "model.transformer.blocks")

    rows: list[dict] = []
    with config.data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= config.samples + config.warmup:
                break
            rows.append(json.loads(line))

    cap = config.max_length - config.gen_length
    prompts = []
    for row in rows:
        text = (
            f"{SAMSUM_INSTRUCTION}\n\n{str(row['context']).strip()}\n"
            f"{str(row.get('input', row.get('question', ''))).strip()}"
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][-cap:]
        prompts.append(
            torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=config.device)
        )

    summary: dict[str, dict] = {}
    for name in config.methods:
        prefills: list[float] = []
        decodes: list[float] = []
        torch.cuda.reset_peak_memory_stats()
        for index, prompt_ids in enumerate(prompts):
            prefill, decode = run_method(name, model, student, prompt_ids, config)
            if index >= config.warmup:
                prefills.append(prefill)
                decodes.append(decode)
            torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 2**30
        summary[name] = {
            "samples": len(decodes),
            "prefill_s": sum(prefills) / len(prefills),
            "decode_s": sum(decodes) / len(decodes),
            "total_s": (sum(prefills) + sum(decodes)) / len(decodes),
            "decode_ms_per_step": sum(decodes) / len(decodes) / config.steps * 1000,
            "peak_memory_gib": peak,
        }
        row = summary[name]
        print(
            f"{name:>12}  prefill {row['prefill_s']:6.2f}s  decode {row['decode_s']:6.2f}s"
            f"  ({row['decode_ms_per_step']:6.1f} ms/step)  peak {row['peak_memory_gib']:.1f} GiB",
            flush=True,
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {config.output_path}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> LatencyConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["origin"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--budget", type=int, default=960)
    parser.add_argument("--frozen-layers", type=int, default=16)
    parser.add_argument("--refresh-tokens", type=int, default=0)
    parser.add_argument("--measured-tokens", type=int, default=480)
    parser.add_argument("--refresh-interval", type=int, default=1)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    return LatencyConfig(
        model_path=args.model,
        data_path=args.data,
        student_path=args.student,
        output_path=args.output,
        methods=tuple(args.methods),
        samples=args.samples,
        warmup=args.warmup,
        budget=args.budget,
        frozen_layers=args.frozen_layers,
        refresh_tokens=args.refresh_tokens,
        measured_tokens=args.measured_tokens,
        refresh_interval=args.refresh_interval,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
