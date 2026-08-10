"""Dynamic oracle refresh (Phase 0 gate) — self-contained, core files untouched.

Per denoising step, two passes:

  Pass A (shadow, fresh)  : forward [kept ; suffix] fresh -> per-layer fresh K*/V* at
                            kept positions, suffix->kept attention A*, and suffix logits
                            (-> S_t, the positions committing this step).
  refresh decision        : d^V = ||V* - V_hat|| / ||V*||   (V_hat = served stale value)
                            s*  = mean_{j in S_t} mean_h A*(j->i)
                            r*  = s* . d^V   ->   TopR per layer within the kept set.
  Pass B (selective)      : suffix attends [ served K/V (R refreshed + rest stale) ; suffix ]
                            -> selective logits, which drive the commit (this is the model
                            that actually generates under the cache).

Baseline = "kept + suffix" fresh forward (the method's own full-refresh ceiling), so:
  R >= budget  == dynkv refresh=1   (full refresh every step)
  R == 0       == frozen cache      (never refresh after prefill)
These two identities are the correctness sanity checks.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from dllm_cache.budget.dynamic_prompt_kv import (
    project_qkv,
    project_heads_at_positions,
    repeat_heads,
    run_block_mlp,
    suffix_logits_from_hidden,
)
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


def _embed(decoder, ids):
    config = decoder.config
    x = decoder.transformer.wte(ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    return decoder.transformer.emb_drop(x)


@torch.inference_mode()
def _shadow_forward(model, kept_ids, suffix_ids, keep_positions, suffix_positions):
    """Fresh forward of [kept ; suffix]. Returns per-layer (K*_kept, V*_kept, A*), logits.

    K*/V*_kept are head tensors [1, H, nkeep, hd] (RoPE applied, repeated to query heads).
    A* is suffix->kept attention weights [1, H, gen, nkeep].
    """
    decoder = getattr(model, "model")
    nkeep = int(kept_ids.shape[1])
    x = torch.cat([_embed(decoder, kept_ids), _embed(decoder, suffix_ids)], dim=1)
    positions = torch.cat([keep_positions, suffix_positions], dim=0)

    fresh_k, fresh_v, attn_w = [], [], []
    for block in decoder.transformer.blocks:
        q, k, v = project_qkv(block, x)
        qh, kh, vh = project_heads_at_positions(block, q, k, v, positions)
        if qh.shape[1] != kh.shape[1]:
            kh = repeat_heads(kh, qh.shape[1])
            vh = repeat_heads(vh, qh.shape[1])
        fresh_k.append(kh[:, :, :nkeep, :].detach().clone())
        fresh_v.append(vh[:, :, :nkeep, :].detach().clone())
        # suffix -> kept attention weights (softmax over full [kept ; suffix] keys)
        q_suffix = qh[:, :, nkeep:, :]
        scores = (q_suffix.float() @ kh.float().transpose(-1, -2)) / math.sqrt(qh.shape[-1])
        attn_w.append(F.softmax(scores, dim=-1)[..., :nkeep].detach())
        att = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
        att = att.transpose(1, 2).contiguous().view_as(x)
        x = x + block.dropout(block.attn_out(att))
        x = run_block_mlp(block, x)
    logits = suffix_logits_from_hidden(decoder, x[:, nkeep:, :])
    return fresh_k, fresh_v, attn_w, logits


@torch.inference_mode()
def _selective_forward(model, suffix_ids, suffix_positions, served_k, served_v):
    """Suffix-only forward; each block attends [served kept K/V ; fresh suffix K/V]."""
    decoder = getattr(model, "model")
    x = _embed(decoder, suffix_ids)
    for layer_id, block in enumerate(decoder.transformer.blocks):
        q, k, v = project_qkv(block, x)
        qh, kh, vh = project_heads_at_positions(block, q, k, v, suffix_positions)
        if qh.shape[1] != kh.shape[1]:
            kh = repeat_heads(kh, qh.shape[1])
            vh = repeat_heads(vh, qh.shape[1])
        key = torch.cat([served_k[layer_id], kh], dim=2)
        value = torch.cat([served_v[layer_id], vh], dim=2)
        att = F.scaled_dot_product_attention(qh, key, value, dropout_p=0.0, is_causal=False)
        att = att.transpose(1, 2).contiguous().view_as(x)
        x = x + block.dropout(block.attn_out(att))
        x = run_block_mlp(block, x)
    return suffix_logits_from_hidden(decoder, x)


@torch.inference_mode()
def generate_with_oracle_refresh(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    keep_indices: torch.Tensor,
    refresh_tokens: int,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int = 126336,
) -> torch.Tensor:
    device = input_ids.device
    P = int(input_ids.shape[1])
    L = len(getattr(model, "model").transformer.blocks)
    keep = keep_indices.to(device=device, dtype=torch.long).sort().values
    nkeep = int(keep.numel())
    R = int(refresh_tokens)

    kept_ids = input_ids.index_select(dim=1, index=keep)
    keep_positions = keep.clone()
    suffix_positions = torch.arange(P, P + gen_length, device=device, dtype=torch.long)

    x = torch.full((1, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = input_ids

    served_k: list[torch.Tensor | None] = [None] * L
    served_v: list[torch.Tensor | None] = [None] * L

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    for nb in range(num_blocks):
        start = P + nb * block_length
        end = P + (nb + 1) * block_length
        block_mask = x[:, start:end] == mask_id
        ntt = get_num_transfer_tokens(block_mask, steps_per_block)
        for si in range(steps_per_block):
            suffix_ids = x[:, P:]
            fresh_k, fresh_v, attn_w, shadow_logits = _shadow_forward(
                model, kept_ids, suffix_ids, keep_positions, suffix_positions
            )

            # S_t: positions committing this step (current block, top-ntt by confidence)
            probs = F.softmax(shadow_logits, dim=-1)
            conf = probs.max(dim=-1).values.squeeze(0)               # [gen]
            gen_mask = (suffix_ids == mask_id).squeeze(0)
            cand = torch.full_like(conf, float("-inf"))
            blk = slice(nb * block_length, (nb + 1) * block_length)
            cand[blk] = torch.where(gen_mask[blk], conf[blk], torch.full_like(conf[blk], float("-inf")))
            k_commit = int(ntt[0, si].item())
            if k_commit > 0:
                s_idx = torch.topk(cand, k=min(k_commit, int(torch.isfinite(cand).sum()))).indices
            else:
                s_idx = torch.arange(0, device=device)

            for l in range(L):
                if served_v[l] is None:                              # prefill: init fresh
                    served_k[l] = fresh_k[l].clone()
                    served_v[l] = fresh_v[l].clone()
                    continue
                if R >= nkeep:                                       # full refresh
                    served_k[l] = fresh_k[l].clone()
                    served_v[l] = fresh_v[l].clone()
                    continue
                if R <= 0 or s_idx.numel() == 0:                     # no refresh
                    continue
                dV = ((fresh_v[l] - served_v[l]).norm(dim=-1)
                      / fresh_v[l].norm(dim=-1).clamp_min(1e-6)).mean(dim=1).squeeze(0)  # [nkeep]
                # attn_w[l]: [1, H, gen, nkeep] -> select S_t queries, mean over heads & S_t
                sstar = attn_w[l][0][:, s_idx, :].mean(dim=0).mean(dim=0)  # [nkeep]
                r = sstar * dV
                due = torch.topk(r, k=R).indices
                served_k[l][:, :, due, :] = fresh_k[l][:, :, due, :]
                served_v[l][:, :, due, :] = fresh_v[l][:, :, due, :]

            logits = _selective_forward(model, suffix_ids, suffix_positions, served_k, served_v)
            logits_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noise, dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, (nb + 1) * block_length:] = -float("inf")
            mask_index = (x[:, P:] == mask_id)
            x0 = torch.where(mask_index, x0, x[:, P:])
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            cnt = int(ntt[0, si].item())
            if cnt > 0:
                sel = torch.topk(confidence[0], k=cnt).indices
                transfer[0, sel] = True
            x[:, P:][transfer] = x0[transfer]
    return x[:, P:]
