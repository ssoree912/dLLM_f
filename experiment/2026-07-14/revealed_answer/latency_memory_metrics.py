from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MemorySample:
    peak_mib: float
    delta_mib: float


@dataclass(frozen=True, slots=True)
class MemoryStart:
    device: str
    allocated: int


def aggregate(rows: list[dict[str, str | int | float]], key: str) -> dict[str, int | float]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {"count": 0, "total": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def mean(rows: list[dict[str, str | int | float]], key: str) -> float:
    if not rows:
        return 0.0
    return statistics.fmean(float(row[key]) for row in rows)


def sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def start_memory_sample(device: str) -> MemoryStart:
    sync_if_cuda(device)
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return MemoryStart(device=device, allocated=0)
    torch_device = torch.device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch_device)
    return MemoryStart(device=device, allocated=torch.cuda.memory_allocated(torch_device))


def finish_memory_sample(start: MemoryStart) -> MemorySample:
    sync_if_cuda(start.device)
    if not (start.device.startswith("cuda") and torch.cuda.is_available()):
        return MemorySample(peak_mib=0.0, delta_mib=0.0)
    torch_device = torch.device(start.device)
    peak = torch.cuda.max_memory_allocated(torch_device)
    delta = max(0, peak - start.allocated)
    return MemorySample(peak_mib=bytes_to_mib(peak), delta_mib=bytes_to_mib(delta))


def bytes_to_mib(value: int) -> float:
    return float(value) / 1024.0 / 1024.0


def estimate_attention_work(
    prompt_length: int,
    budget: int,
    gen_length: int,
    steps: int,
    layer_count: int,
) -> dict[str, float | int]:
    kept_prompt = min(max(1, budget), prompt_length)
    full_key_tokens = prompt_length + gen_length
    pruned_key_tokens = kept_prompt + gen_length
    sequence_length = prompt_length + gen_length
    full_qk = layer_count * steps * sequence_length * full_key_tokens
    pruned_qk = layer_count * steps * sequence_length * pruned_key_tokens
    return {
        "full_key_tokens": full_key_tokens,
        "student_key_tokens": pruned_key_tokens,
        "prompt_kv_keep_ratio": kept_prompt / prompt_length,
        "attention_qk_keep_ratio": pruned_qk / full_qk,
        "attention_qk_reduction_ratio": 1.0 - (pruned_qk / full_qk),
        "full_attention_qk_elements": full_qk,
        "student_attention_qk_elements": pruned_qk,
    }
