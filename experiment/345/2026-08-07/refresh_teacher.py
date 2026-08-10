"""Refresh (drift) teacher collection — our method, §5/§9.

Per denoising step, a shadow (fresh) forward gives fresh K*/V* and the suffix->kept
attention; comparing V* to the served (stale) cache gives value staleness d^V, and the
attention over the committing set S_t gives read-weight s*. The refresh utility is

    r*_{t,l,i} = s*_{t,l,i} . d^V_{t,l,i}      (per step, per layer, per kept token)

The behavior policy refreshes the Top-R by r* each step (on-policy for the oracle), so the
recorded cache age / staleness match what a student trained to imitate this policy will see.

Only LABELS + trajectory are dumped (r*, cache_age, commit schedule, prompt/keep ids); the
student features (prefill hidden u/q, current-gen context c_t) are recomputed at train time
by re-running the frozen model, exactly like the importance-student trainer -- dumping
per-token hidden would be ~240MB/sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from refresh_oracle import _shadow_forward, _selective_forward
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


@torch.inference_mode()
def collect_refresh_teacher(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    keep_indices: torch.Tensor,
    refresh_tokens: int,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    mask_id: int = 126336,
) -> dict:
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
    age = torch.zeros(L, nkeep, dtype=torch.long, device=device)      # steps since last refresh

    r_star = torch.zeros(steps, L, nkeep, dtype=torch.float32)
    age_before = torch.zeros(steps, L, nkeep, dtype=torch.int16)
    commit_step = torch.full((gen_length,), -1, dtype=torch.long)     # step each gen pos committed

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    gstep = 0
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
            conf = probs.max(dim=-1).values.squeeze(0)
            gen_mask = (suffix_ids == mask_id).squeeze(0)
            cand = torch.full_like(conf, float("-inf"))
            blk = slice(nb * block_length, (nb + 1) * block_length)
            cand[blk] = torch.where(gen_mask[blk], conf[blk], torch.full_like(conf[blk], float("-inf")))
            k_commit = int(ntt[0, si].item())
            s_idx = (
                torch.topk(cand, k=min(k_commit, int(torch.isfinite(cand).sum()))).indices
                if k_commit > 0 else torch.arange(0, device=device)
            )

            age_before[gstep] = age.to(torch.int16).cpu()
            for l in range(L):
                if served_v[l] is None:                      # prefill: init fresh, r*=0
                    served_k[l] = fresh_k[l].clone()
                    served_v[l] = fresh_v[l].clone()
                    continue
                dV = ((fresh_v[l] - served_v[l]).norm(dim=-1)
                      / fresh_v[l].norm(dim=-1).clamp_min(1e-6)).mean(dim=1).squeeze(0)  # [nkeep]
                if s_idx.numel() > 0:
                    sstar = attn_w[l][0][:, s_idx, :].mean(dim=0).mean(dim=0)             # [nkeep]
                else:
                    sstar = torch.zeros(nkeep, device=device)
                r = sstar * dV
                r_star[gstep, l] = r.float().cpu()
                # behavior policy: refresh Top-R by r*
                if 0 < R < nkeep:
                    due = torch.topk(r, k=R).indices
                elif R >= nkeep:
                    due = torch.arange(nkeep, device=device)
                else:
                    due = torch.arange(0, device=device)
                if due.numel() > 0:
                    served_k[l][:, :, due, :] = fresh_k[l][:, :, due, :]
                    served_v[l][:, :, due, :] = fresh_v[l][:, :, due, :]
                    age[l] += 1
                    age[l, due] = 0
                else:
                    age[l] += 1

            logits = _selective_forward(model, suffix_ids, suffix_positions, served_k, served_v)
            x0 = torch.argmax(add_gumbel_noise(logits, temperature=0.0), dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, (nb + 1) * block_length:] = -float("inf")
            mask_index = (x[:, P:] == mask_id)
            x0 = torch.where(mask_index, x0, x[:, P:])
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            if k_commit > 0:
                sel = torch.topk(confidence[0], k=k_commit).indices
                transfer[0, sel] = True
                commit_step[sel.cpu()] = gstep
            x[:, P:][transfer] = x0[transfer]
            gstep += 1

    return {
        "schema_version": "refresh_teacher_v1",
        "prompt_input_ids": input_ids.squeeze(0).cpu().to(torch.long),
        "keep_indices": keep.cpu().to(torch.long),
        "refresh_tokens": int(R),
        "steps": int(steps),
        "gen_length": int(gen_length),
        "block_length": int(block_length),
        "generated_input_ids": x[:, P:].squeeze(0).cpu().to(torch.long),
        "commit_step": commit_step.cpu(),                 # [gen] step each pos committed
        "r_star": r_star.to(torch.float16),               # [T, L, nkeep]
        "cache_age_before": age_before,                   # [T, L, nkeep] int16
    }
