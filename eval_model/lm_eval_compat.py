import torch


def get_dtype(dtype: str | torch.dtype | None) -> str | torch.dtype:
    if dtype is None or dtype == "auto":
        return "auto"
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype in dtype_map:
        return dtype_map[dtype]
    raise ValueError(f"Unsupported dtype: {dtype}")
