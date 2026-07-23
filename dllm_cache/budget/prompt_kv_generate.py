from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_cache.budget.prompt_kv_cache import PromptKVCache
from dllm_cache.budget.prompt_kv_forward import prompt_kv_suffix_logits
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


@torch.inference_mode()
def generate_with_prompt_kv(
    input_ids: torch.Tensor,
    model: nn.Module,
    prompt_cache: PromptKVCache,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> torch.Tensor:
    if cfg_scale > 0.0:
        raise RuntimeError("prompt KV cache generation does not support cfg_scale")
    batch_size, prompt_length = input_ids.shape
    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    x[:, :prompt_length] = input_ids
    if gen_length % block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise RuntimeError("steps must be divisible by number of blocks")
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        start_idx = prompt_length + num_block * block_length
        end_idx = prompt_length + (num_block + 1) * block_length
        block_mask_index = x[:, start_idx:end_idx] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_idx in range(steps_per_block):
            mask_index = x == mask_id
            logits = prompt_kv_suffix_logits(model, x, prompt_cache)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)),
                    -1,
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise RuntimeError(f"unsupported remasking: {remasking}")
            x0_p[:, (num_block + 1) * block_length :] = -float("inf")
            x0 = torch.where(mask_index[:, prompt_length:], x0, x[:, prompt_length:])
            confidence = torch.where(mask_index[:, prompt_length:], x0_p, -float("inf"))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                select_index = torch.topk(
                    confidence[batch_idx],
                    k=num_transfer_tokens[batch_idx, step_idx],
                ).indices
                transfer_index[batch_idx, select_index] = True
            x[:, prompt_length:][transfer_index] = x0[transfer_index]
    return x[:, prompt_length:]
