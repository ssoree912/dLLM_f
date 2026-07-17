from __future__ import annotations

import argparse
import inspect
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for import_root in (SCRIPT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.attention_teacher import find_transformer_blocks
from revealed_answer.prompt_kv_cache import project_heads, repeat_heads
from utils.generate_function import get_num_transfer_tokens


class TokenizerLike(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]: ...


@dataclass(frozen=True, slots=True)
class DriftConfig:
    model_path: Path
    data_path: Path
    output_dir: Path
    device: str
    dtype: torch.dtype
    prompt_cap: int
    gen_length: int
    steps: int
    mask_id: int
    layers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DriftStats:
    max_abs: float
    mean_abs: float
    rel_l2: float
    mean_cosine: float


def parse_args() -> DriftConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/home/M2026107/dllm/model/LLaDA-8B-Instruct"))
    parser.add_argument("--data", type=Path, default=Path("/home/M2026107/dllm/data/longbench/2wikimqa.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/2026-07-14/results/prompt_kv_drift_2wikimqa_sample0"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--prompt-cap", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--layers", default="0,1,8,16,24,31")
    args = parser.parse_args()
    layers = tuple(int(item) for item in args.layers.split(",") if item)
    return DriftConfig(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        prompt_cap=args.prompt_cap,
        gen_length=args.gen_length,
        steps=args.steps,
        mask_id=args.mask_id,
        layers=layers,
    )


def parse_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case unreachable:
            raise argparse.ArgumentTypeError(f"unsupported dtype: {unreachable}")


def build_2wikimqa_prompt(data_path: Path) -> str:
    row = json.loads(data_path.read_text(encoding="utf-8").splitlines()[0])
    instruction = "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    return (
        f"{instruction}\n\n"
        f"The following are given passages.\n{row['context']}\n\n"
        f"{instruction}\n\n"
        f"Question: {row['question']}\n"
        "Answer:"
    )


def encode_prompt(tokenizer: TokenizerLike, prompt: str, prompt_cap: int) -> list[int]:
    encoded = tokenizer(prompt, add_special_tokens=False)
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    return token_ids[-prompt_cap:]


def install_kv_recorder(
    model: torch.nn.Module,
    prompt_length: int,
    layers: tuple[int, ...],
    records: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.nn.Module, types.MethodType]]:
    originals: list[tuple[torch.nn.Module, types.MethodType]] = []
    selected_layers = set(layers)
    for block in find_transformer_blocks(model):
        original = block.attention
        accepts_block_mask = "block_mask" in inspect.signature(original).parameters

        def wrapped_attention(
            block_self: torch.nn.Module,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attention_bias: torch.Tensor | None = None,
            layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
            use_cache: bool = False,
            block_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
            layer_id = int(getattr(block_self, "layer_id"))
            if layer_id in selected_layers:
                q_heads, k_heads, v_heads = project_heads(block_self, q, k, v, position_offset=0)
                if q_heads.shape[1] != k_heads.shape[1]:
                    k_heads = repeat_heads(k_heads, q_heads.shape[1])
                    v_heads = repeat_heads(v_heads, q_heads.shape[1])
                records[layer_id] = (
                    k_heads[:, :, :prompt_length, :].detach().float().cpu(),
                    v_heads[:, :, :prompt_length, :].detach().float().cpu(),
                )
            if accepts_block_mask:
                return original(
                    q,
                    k,
                    v,
                    attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    block_mask=block_mask,
                )
            return original(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        block.attention = types.MethodType(wrapped_attention, block)
        originals.append((block, original))
    return originals


def restore_attention(originals: list[tuple[torch.nn.Module, types.MethodType]]) -> None:
    for block, original in originals:
        block.attention = original


@torch.inference_mode()
def capture_prompt_kv(
    model: torch.nn.Module,
    x: torch.Tensor,
    prompt_length: int,
    layers: tuple[int, ...],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    records: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    originals = install_kv_recorder(model, prompt_length, layers, records)
    try:
        model(x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True)
    finally:
        restore_attention(originals)
    return records


@torch.inference_mode()
def denoise_one_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    prompt_length: int,
    transfer_counts: torch.Tensor,
    step_index: int,
    mask_id: int,
) -> None:
    suffix_mask = x[:, prompt_length:] == mask_id
    logits = model(x, attention_mask=torch.ones_like(x)).logits[:, prompt_length:]
    probs = F.softmax(logits, dim=-1)
    x0 = torch.argmax(logits, dim=-1)
    confidence = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    confidence = torch.where(suffix_mask, confidence, torch.full_like(confidence, -float("inf")))
    for batch_index in range(confidence.shape[0]):
        select_index = torch.topk(confidence[batch_index], k=transfer_counts[batch_index, step_index]).indices
        x[batch_index, prompt_length + select_index] = x0[batch_index, select_index]


def drift_stats(base: torch.Tensor, current: torch.Tensor) -> DriftStats:
    delta = current - base
    base_flat = base.flatten(start_dim=2)
    current_flat = current.flatten(start_dim=2)
    cosine = F.cosine_similarity(base_flat, current_flat, dim=-1)
    return DriftStats(
        max_abs=float(delta.abs().max().item()),
        mean_abs=float(delta.abs().mean().item()),
        rel_l2=float(delta.norm().item() / base.norm().clamp_min(1e-12).item()),
        mean_cosine=float(cosine.mean().item()),
    )


def print_stats(
    label: str,
    baseline: dict[int, tuple[torch.Tensor, torch.Tensor]],
    current: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    print(f"\n[state] {label}", flush=True)
    for layer_id in sorted(baseline):
        base_k, base_v = baseline[layer_id]
        current_k, current_v = current[layer_id]
        k_stats = drift_stats(base_k, current_k)
        v_stats = drift_stats(base_v, current_v)
        print(
            f"layer={layer_id:02d} "
            f"K max={k_stats.max_abs:.6g} mean={k_stats.mean_abs:.6g} rel_l2={k_stats.rel_l2:.6g} cos={k_stats.mean_cosine:.8f} | "
            f"V max={v_stats.max_abs:.6g} mean={v_stats.mean_abs:.6g} rel_l2={v_stats.rel_l2:.6g} cos={v_stats.mean_cosine:.8f}",
            flush=True,
        )
        rows.append({"state": label, "layer": layer_id, "kind": "k", **asdict(k_stats)})
        rows.append({"state": label, "layer": layer_id, "kind": "v", **asdict(v_stats)})
    return rows


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] model={config.model_path} dtype={config.dtype} device={config.device}", flush=True)
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path), trust_remote_code=True)
    prompt_ids = encode_prompt(tokenizer, build_2wikimqa_prompt(config.data_path), config.prompt_cap)
    prompt_length = len(prompt_ids)
    x = torch.full(
        (1, prompt_length + config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=config.device,
    )
    x[:, :prompt_length] = torch.tensor(prompt_ids, dtype=torch.long, device=config.device)
    print(
        f"[setup] prompt_length={prompt_length} gen_length={config.gen_length} steps={config.steps} layers={config.layers}",
        flush=True,
    )
    baseline = capture_prompt_kv(model, x, prompt_length, config.layers)
    rows = print_stats("step=0 all_suffix_mask baseline", baseline, baseline)
    transfer_counts = get_num_transfer_tokens(x[:, prompt_length:] == config.mask_id, config.steps)
    for step_index in range(config.steps):
        denoise_one_step(model, x, prompt_length, transfer_counts, step_index, config.mask_id)
        filled = int((x[:, prompt_length:] != config.mask_id).sum().item())
        current = capture_prompt_kv(model, x, prompt_length, config.layers)
        rows.extend(print_stats(f"step={step_index + 1} filled_suffix={filled}/{config.gen_length}", baseline, current))
    out_path = config.output_dir / "prompt_kv_drift.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[done] wrote={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
