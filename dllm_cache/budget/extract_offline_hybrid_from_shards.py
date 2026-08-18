# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers"]
# ///

"""Extract Attention+Delta teacher shards from reusable prompt-source shards.

The source may be a legacy teacher root or a prompt-only root produced by
``export_prompt_source_from_teacher``.  Only prompt tokens and sample metadata are
required.  This lets a replacement teacher preserve the exact balanced sample
population without retaining obsolete teacher labels.

Output shards follow extract_offline_hybrid_teacher.py's schema (teacher_raw /
teacher_norm / ref_union_mask / delta_raw / delta_norm / delta_step_max).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from dllm_cache.budget.extract_teacher import load_model_and_tokenizer, parse_dtype
from dllm_cache.budget.offline_hybrid_teacher import (
    OfflineHybridTeacherConfig,
    generate_with_offline_hybrid_teacher,
)


@dataclass(frozen=True, slots=True)
class ExtractFromShardsConfig:
    model_path: Path
    source_root: Path
    output_root: Path
    datasets: tuple[str, ...]
    n_samples: int
    device: str
    dtype: torch.dtype
    gen_length: int
    block_length: int
    steps: int
    active_top_k: int
    temperature: float
    confidence_weight: bool
    target_aggregation: str
    apply_chat_template: bool
    max_length: int
    question_window: int


def parse_args(argv: Sequence[str] | None = None) -> ExtractFromShardsConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="existing teacher root whose shards supply prompt_input_ids")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="dataset subdirectories to process; default: all")
    parser.add_argument("--n-samples", type=int, default=0, help="per dataset; 0 = all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--active-top-k",
        type=int,
        default=0,
        help="per-step attention support budget; 0 keeps all prompt positions",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--confidence-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight committed-token attention by generation confidence (default: true)",
    )
    parser.add_argument("--target-aggregation", choices=["max", "sum"], default="max")
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="decode the stored prompt tokens, wrap them the way the inference harness "
        "does, and re-encode -- delta is sensitive to the inference-time distribution "
        "(matched-sample samsum: templated 0.3876 vs template-free 0.3522)",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--question-window", type=int, default=128)
    args = parser.parse_args(argv)
    return ExtractFromShardsConfig(
        model_path=args.model,
        source_root=args.source_root,
        output_root=args.output_root,
        datasets=tuple(args.datasets) if args.datasets else (),
        n_samples=args.n_samples,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        active_top_k=args.active_top_k,
        temperature=args.temperature,
        confidence_weight=args.confidence_weight,
        target_aggregation=args.target_aggregation,
        apply_chat_template=args.apply_chat_template,
        max_length=args.max_length,
        question_window=args.question_window,
    )


def list_source_files(config: ExtractFromShardsConfig) -> list[Path]:
    if not config.source_root.is_dir():
        raise RuntimeError(f"source root is not a directory: {config.source_root}")
    files: list[Path] = []
    for dataset_dir in sorted(p for p in config.source_root.iterdir() if p.is_dir()):
        if config.datasets and dataset_dir.name not in config.datasets:
            continue
        dataset_files = sorted(dataset_dir.glob("*.pt"))
        if config.n_samples > 0:
            dataset_files = dataset_files[: config.n_samples]
        files.extend(dataset_files)
    if not files:
        raise RuntimeError(f"no source shards found under {config.source_root}")
    return files


def build_prompt_tensor(tokenizer, src: dict, config: ExtractFromShardsConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt tokens (optionally re-wrapped with the chat template) and question indices."""
    prompt_tensor = src["prompt_input_ids"].to(torch.long)
    if not config.apply_chat_template:
        return prompt_tensor, src["question_token_indices"].to(torch.long)
    text = tokenizer.decode(prompt_tensor, skip_special_tokens=True)
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = [int(t) for t in tokenizer(templated, add_special_tokens=False)["input_ids"]]
    prompt_cap = max(1, config.max_length - config.gen_length)
    ids = ids[-prompt_cap:]
    prompt_tensor = torch.tensor(ids, dtype=torch.long)
    window = min(max(1, config.question_window), len(ids))
    question_indices = torch.arange(len(ids) - window, len(ids), dtype=torch.long)
    return prompt_tensor, question_indices


@torch.inference_mode()
def extract_one(model: torch.nn.Module, tokenizer, src: dict, config: ExtractFromShardsConfig) -> dict:
    prompt_tensor, question_indices = build_prompt_tensor(tokenizer, src, config)
    prompt_ids = prompt_tensor.unsqueeze(0).to(config.device)
    teacher_config = OfflineHybridTeacherConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        active_top_k=config.active_top_k,
        temperature=config.temperature,
        confidence_weight=config.confidence_weight,
        target_aggregation=config.target_aggregation,
    )
    result = generate_with_offline_hybrid_teacher(model, prompt_ids, teacher_config)
    generated_answer = tokenizer.batch_decode(
        result.generated_ids.unsqueeze(0), skip_special_tokens=True
    )[0].strip()
    prompt_length = int(prompt_tensor.numel())
    reference_support = (
        "all_prompt_positions"
        if config.active_top_k == 0
        else "temporal_union_topk_support"
    )
    return {
        "teacher_kind": "offline_hybrid_ref_delta",
        "teacher_formula": (
            f"reference={config.target_aggregation}_step_newly_committed_queries_"
            f"suffix_to_prompt_attention_{reference_support}; "
            "delta=sum_t mean(K_relative_stepwise,V_relative_stepwise)"
        ),
        "teacher_graph": "full_sequence_prompt_suffix_single_forward_ref_and_delta",
        "prompt_source": src.get(
            "prompt_source_kind", "reused_from_existing_teacher_shard"
        ),
        "source_teacher_kind": src.get("source_teacher_kind", ""),
        "sample_id": src["sample_id"],
        "dataset": src["dataset"],
        "task": src.get("task", ""),
        "question": src.get("question", ""),
        "answers": src.get("answers", ""),
        "generated_answer": generated_answer,
        "prompt_input_ids": prompt_tensor,
        "answer_input_ids": result.generated_ids.to(torch.long),
        "generated_answer_input_ids": result.generated_ids.to(torch.long),
        "prompt_token_indices": torch.arange(prompt_length, dtype=torch.long),
        "question_token_indices": question_indices,
        "teacher_raw": result.teacher_raw.to(torch.float16),
        "teacher_norm": result.teacher_norm.to(torch.float16),
        "ref_union_mask": result.ref_union_mask,
        "ref_union_size_by_layer": result.ref_union_size_by_layer,
        "ref_union_size_mean": float(result.ref_union_size_by_layer.float().mean().item()),
        "delta_raw": result.delta_raw.to(torch.float16),
        "delta_norm": result.delta_norm.to(torch.float16),
        "delta_step_max": result.delta_step_max.to(torch.float16),
        "delta_observation_count": result.delta_observation_count,
        "prompt_length": prompt_length,
        "generated_length": int(result.generated_ids.numel()),
        "sequence_length": prompt_length + int(result.generated_ids.numel()),
        "gen_length": config.gen_length,
        "block_length": config.block_length,
        "steps": config.steps,
        "active_top_k": config.active_top_k,
        "target_aggregation": config.target_aggregation,
        "reference_query_mode": "commit",
        "prompt_format": src.get("prompt_format", ""),
        "apply_chat_template": int(config.apply_chat_template),
        "reference_step_count": result.reference_step_count,
        "commit_count": result.commit_count,
        "confidence_weight_sum": result.weight_sum,
        "confidence_weight": int(config.confidence_weight),
    }


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    model, tokenizer = load_model_and_tokenizer(config)
    files = list_source_files(config)
    started = time.time()
    saved = 0
    for index, path in enumerate(files, start=1):
        out_path = config.output_root / path.relative_to(config.source_root)
        if out_path.exists():
            saved += 1
            continue
        src = torch.load(path, map_location="cpu", weights_only=False)
        rec = extract_one(model, tokenizer, src, config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rec, out_path)
        saved += 1
        print(
            f"[hybrid-from-shards {index}/{len(files)}] saved={out_path} "
            f"dataset={rec['dataset']} prompt={rec['prompt_length']} "
            f"delta_mean={rec['delta_raw'].float().mean().item():.4f} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
    print(f"[done] saved={saved} elapsed={time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
