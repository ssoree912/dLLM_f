"""Drift-based prompt-KV refresh: pick which kept tokens to recompute each step.

The keep set is fixed by the importance student (top-`budget` prompt tokens, shared
across layers). What this module decides is the *time* axis: at every denoising step and
every layer, which R of the kept tokens get their K/V recomputed and which keep serving
the stale cache.

Selectors:

  oracle   r*_{t,l,i} = s*_{t,l,i} . d^V_{t,l,i}
           s* = suffix->kept attention mass over the positions committing this step
           d^V = ||V_fresh - V_served|| / ||V_fresh||   (how wrong the served value is)
           This needs the fresh values, so it is an upper bound, not a deployable policy.

  student  z = f(prompt/state features) from a trained RefreshStudent -- no fresh values
           needed, so this is the deployable version the oracle bounds.

  delta_student  re-applies the offline delta PromptUtilityStudent to the currently
           served prompt hidden states at every denoising step.  Layer probabilities
           are pooled before a global top-R is selected, so all layers update the same
           tokens while the selected set can change from one step to the next.

The oracle uses a shadow forward over every kept token.  The deployable selectors instead
re-score cached inference-time state and forward only the selected R prompt tokens plus the
suffix.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_cache.budget.dynamic_prompt_kv import (
    project_qkv,
    project_heads_at_positions,
    repeat_heads,
    run_block_mlp,
    suffix_logits_from_hidden,
)
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens

__all__ = [
    "RefreshStudent",
    "load_refresh_student",
    "generate_with_drift_refresh",
    "select_delta_student_topk",
]


class RefreshStudent(nn.Module):
    """Scores kept prompt tokens for refresh from inference-time features only."""

    def __init__(self, hidden=4096, proj=256, mlp=512, n_layers=32, age_dim=16, layer_dim=16):
        super().__init__()
        self.p_tok = nn.Linear(hidden, proj)
        self.p_q = nn.Linear(hidden, proj)
        self.p_c = nn.Linear(hidden, proj)
        self.layer_emb = nn.Embedding(n_layers, layer_dim)
        self.age_proj = nn.Linear(1, age_dim)
        feat = proj * 5 + age_dim + layer_dim + 1
        self.mlp = nn.Sequential(nn.Linear(feat, mlp), nn.GELU(), nn.Linear(mlp, 1))

    def forward(self, u, q, c, age, step_frac, layer_id):
        u = self.p_tok(u)
        q = self.p_q(q).unsqueeze(0)
        c = self.p_c(c).unsqueeze(0)
        n = u.shape[0]
        qexp = q.expand(n, -1)
        cexp = c.expand(n, -1)
        lay = self.layer_emb(torch.tensor(layer_id, device=u.device)).unsqueeze(0).expand(n, -1)
        agef = self.age_proj(age.float().unsqueeze(-1) / 128.0)
        stepf = torch.full((n, 1), float(step_frac), device=u.device)
        phi = torch.cat([u, qexp, u * qexp, cexp, u * cexp, agef, lay, stepf], dim=-1)
        return self.mlp(phi).squeeze(-1)


def load_refresh_student(ckpt_path, device: str) -> RefreshStudent:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    student = RefreshStudent(proj=cfg.get("proj_dim", 256), mlp=cfg.get("mlp_dim", 512))
    student.load_state_dict(ckpt["state_dict"])
    return student.to(device).eval()


@torch.inference_mode()
def select_delta_student_topk(
    student: nn.Module,
    hidden: list[torch.Tensor],
    keep_positions: torch.Tensor,
    prompt_length: int,
    question_window: int,
    refresh_tokens: int,
) -> torch.Tensor:
    """Re-score the current served prompt state and return one global refresh set."""
    if not hidden:
        raise RuntimeError("delta student refresh needs cached prompt hidden states")
    keep_count = int(keep_positions.numel())
    if not 0 < refresh_tokens <= keep_count:
        raise RuntimeError("refresh_tokens must fall inside the kept prompt")
    prompt_indices = torch.arange(keep_count, device=keep_positions.device)
    question_start = max(0, prompt_length - max(1, question_window))
    question_indices = torch.nonzero(
        keep_positions >= question_start, as_tuple=False
    ).squeeze(-1)
    if question_indices.numel() == 0:
        question_indices = prompt_indices[-1:]

    probabilities = []
    for layer_id, layer_hidden in enumerate(hidden):
        scores = student.forward_layer(
            layer_id,
            layer_hidden.float(),
            prompt_indices,
            question_indices,
        )
        probabilities.append(torch.softmax(scores.float(), dim=-1).squeeze(0))
    pooled = torch.stack(probabilities).mean(dim=0)
    return torch.topk(pooled, k=refresh_tokens, largest=True).indices.sort().values


def _embed(decoder, ids):
    config = decoder.config
    x = decoder.transformer.wte(ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    return decoder.transformer.emb_drop(x)


@torch.inference_mode()
def _shadow_forward(model, kept_ids, suffix_ids, keep_positions, suffix_positions, want_attn):
    """Fresh forward over [kept ; suffix]; returns per-layer fresh K/V (and attention)."""
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
        if want_attn:
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
def _prefill_kept_states(model, kept_ids, suffix_ids, keep_positions, suffix_positions):
    """One full forward over [kept ; suffix] that also keeps each block's input hidden.

    The cheap student path needs those hidden states: to recompute a kept token's K/V at
    layer l it re-projects that token's cached hidden[l] instead of forwarding the whole
    prompt again. Positions that are never chosen keep serving their prefill values.
    """
    decoder = getattr(model, "model")
    nkeep = int(kept_ids.shape[1])
    x = torch.cat([_embed(decoder, kept_ids), _embed(decoder, suffix_ids)], dim=1)
    positions = torch.cat([keep_positions, suffix_positions], dim=0)

    hidden, served_k, served_v = [], [], []
    for block in decoder.transformer.blocks:
        hidden.append(x[:, :nkeep, :].detach().clone())
        q, k, v = project_qkv(block, x)
        qh, kh, vh = project_heads_at_positions(block, q, k, v, positions)
        if qh.shape[1] != kh.shape[1]:
            kh = repeat_heads(kh, qh.shape[1])
            vh = repeat_heads(vh, qh.shape[1])
        served_k.append(kh[:, :, :nkeep, :].detach().clone())
        served_v.append(vh[:, :, :nkeep, :].detach().clone())
        att = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
        att = att.transpose(1, 2).contiguous().view_as(x)
        x = x + block.dropout(block.attn_out(att))
        x = run_block_mlp(block, x)
    return hidden, served_k, served_v


@torch.inference_mode()
def _student_step_forward(
    model, suffix_ids, suffix_positions, keep_positions,
    hidden, served_k, served_v, student, u_all, q_all, c_t, age, step_frac, R,
    frozen_layers=0,
    due_indices=None,
):
    """One denoising step where only the R student-chosen kept tokens are recomputed.

    Cost per layer is R + |suffix| tokens instead of the full kept set: the selected
    positions re-project from their cached hidden, refresh their K/V in the served cache
    and advance their own hidden state; everyone else is read from cache only.

    Below `frozen_layers` nothing is refreshed at all -- only the suffix flows and the
    prompt serves its prefill K/V. Measured drift says those layers barely move (0.00 at
    layer 0, 0.04 by 8, 0.13 by 16), so refreshing them buys nothing and costs half the
    stack.
    """
    decoder = getattr(model, "model")
    x = _embed(decoder, suffix_ids)
    L = len(decoder.transformer.blocks)
    for layer_id, block in enumerate(decoder.transformer.blocks):
        if layer_id < frozen_layers:
            qs, ks, vs = project_qkv(block, x)
            qsh, ksh, vsh = project_heads_at_positions(block, qs, ks, vs, suffix_positions)
            if qsh.shape[1] != ksh.shape[1]:
                ksh = repeat_heads(ksh, qsh.shape[1])
                vsh = repeat_heads(vsh, qsh.shape[1])
            key = torch.cat([served_k[layer_id], ksh], dim=2)
            value = torch.cat([served_v[layer_id], vsh], dim=2)
            att_s = F.scaled_dot_product_attention(qsh, key, value, dropout_p=0.0, is_causal=False)
            att_s = att_s.transpose(1, 2).contiguous().view_as(x)
            x = x + block.dropout(block.attn_out(att_s))
            x = run_block_mlp(block, x)
            continue

        if due_indices is not None:
            due = due_indices
        elif student is None:  # random control: same budget, no learned ranking
            due = torch.randperm(hidden[layer_id].shape[1], device=x.device)[:R]
        else:
            score = student(u_all[layer_id], q_all[layer_id], c_t, age[layer_id], step_frac, layer_id)
            due = torch.topk(score, k=R).indices

        h_sel = hidden[layer_id].index_select(1, due)
        qk, kk, vk = project_qkv(block, h_sel)
        qkh, kkh, vkh = project_heads_at_positions(
            block, qk, kk, vk, keep_positions.index_select(0, due)
        )
        qs, ks, vs = project_qkv(block, x)
        qsh, ksh, vsh = project_heads_at_positions(block, qs, ks, vs, suffix_positions)
        if qsh.shape[1] != ksh.shape[1]:
            kkh = repeat_heads(kkh, qsh.shape[1])
            vkh = repeat_heads(vkh, qsh.shape[1])
            ksh = repeat_heads(ksh, qsh.shape[1])
            vsh = repeat_heads(vsh, qsh.shape[1])

        served_k[layer_id][:, :, due, :] = kkh
        served_v[layer_id][:, :, due, :] = vkh
        key = torch.cat([served_k[layer_id], ksh], dim=2)
        value = torch.cat([served_v[layer_id], vsh], dim=2)

        att_sel = F.scaled_dot_product_attention(qkh, key, value, dropout_p=0.0, is_causal=False)
        att_sel = att_sel.transpose(1, 2).contiguous().view_as(h_sel)
        h_new = h_sel + block.dropout(block.attn_out(att_sel))
        h_new = run_block_mlp(block, h_new)
        if layer_id + 1 < L:
            hidden[layer_id + 1].index_copy_(1, due, h_new.detach())

        att_s = F.scaled_dot_product_attention(qsh, key, value, dropout_p=0.0, is_causal=False)
        att_s = att_s.transpose(1, 2).contiguous().view_as(x)
        x = x + block.dropout(block.attn_out(att_s))
        x = run_block_mlp(block, x)

        age[layer_id] += 1
        age[layer_id, due] = 0
    return suffix_logits_from_hidden(decoder, x)


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
def generate_with_drift_refresh(
    input_ids: torch.Tensor,
    model: nn.Module,
    keep_indices: torch.Tensor,
    refresh_tokens: int,
    mode: str = "oracle",
    refresh_student: RefreshStudent | None = None,
    delta_student: nn.Module | None = None,
    frozen_layers: int = 0,
    question_window: int = 128,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> torch.Tensor:
    if cfg_scale and float(cfg_scale) > 0.0:
        raise RuntimeError("drift refresh generation does not support cfg_scale")
    if mode not in {"oracle", "student", "random", "delta_student"}:
        raise RuntimeError(f"unsupported drift refresh mode: {mode}")
    if mode == "student" and refresh_student is None:
        raise RuntimeError("student mode requires a trained refresh student")
    if mode == "delta_student" and delta_student is None:
        raise RuntimeError("delta_student mode requires the offline delta student")

    device = input_ids.device
    P = int(input_ids.shape[1])
    decoder = getattr(model, "model")
    L = len(decoder.transformer.blocks)
    keep = keep_indices.to(device=device, dtype=torch.long).sort().values
    nkeep = int(keep.numel())
    R = int(refresh_tokens)

    kept_ids = input_ids.index_select(dim=1, index=keep)
    keep_positions = keep.clone()
    suffix_positions = torch.arange(P, P + gen_length, device=device, dtype=torch.long)

    u_all = q_all = None
    if mode == "student":
        hs = model(input_ids, attention_mask=torch.ones_like(input_ids),
                   output_hidden_states=True, use_cache=False, return_dict=True).hidden_states
        question = torch.arange(max(0, P - 128), P, device=device)
        u_all = [hs[l][0].index_select(0, keep).float() for l in range(L)]
        q_all = [hs[l][0].index_select(0, question).float().mean(0) for l in range(L)]
        del hs

    x = torch.full((1, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = input_ids
    served_k: list[torch.Tensor | None] = [None] * L
    served_v: list[torch.Tensor | None] = [None] * L
    age = torch.zeros(L, nkeep, dtype=torch.long, device=device)
    hidden = None
    if mode in {"student", "random", "delta_student"} and 0 < R < nkeep:
        # Cheap path: prefill once, then only the R chosen tokens are recomputed per step.
        hidden, served_k, served_v = _prefill_kept_states(
            model, kept_ids, x[:, P:], keep_positions, suffix_positions
        )

    if gen_length % block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise RuntimeError("steps must be divisible by number of blocks")
    steps_per_block = steps // num_blocks

    gstep = 0
    previous_due_mask = None
    due_jaccard_sum = 0.0
    due_comparisons = 0
    changed_due_steps = 0
    ever_due = torch.zeros(nkeep, dtype=torch.bool, device=device)
    for nb in range(num_blocks):
        start = P + nb * block_length
        end = P + (nb + 1) * block_length
        block_mask = x[:, start:end] == mask_id
        ntt = get_num_transfer_tokens(block_mask, steps_per_block)
        for si in range(steps_per_block):
            suffix_ids = x[:, P:]
            k_commit = int(ntt[0, si].item())

            if hidden is not None:
                # cheap path: no shadow forward, only R tokens recomputed
                c_t = None
                if refresh_student is not None:
                    committed = x[0, P:] != mask_id
                    c_t = (decoder.transformer.wte(x[0, P:][committed]).float().mean(0)
                           if committed.any() else q_all[0])
                due_indices = None
                if mode == "delta_student":
                    due_indices = select_delta_student_topk(
                        delta_student,
                        hidden,
                        keep_positions,
                        prompt_length=P,
                        question_window=question_window,
                        refresh_tokens=R,
                    )
                    due_mask = torch.zeros_like(ever_due)
                    due_mask[due_indices] = True
                    ever_due |= due_mask
                    if previous_due_mask is not None:
                        intersection = int((due_mask & previous_due_mask).sum().item())
                        union = int((due_mask | previous_due_mask).sum().item())
                        due_jaccard_sum += intersection / max(1, union)
                        due_comparisons += 1
                        changed_due_steps += int(not torch.equal(due_mask, previous_due_mask))
                    previous_due_mask = due_mask
                logits = _student_step_forward(
                    model, suffix_ids, suffix_positions, keep_positions,
                    hidden, served_k, served_v, refresh_student,
                    u_all, q_all, c_t, age, gstep / steps, R,
                    frozen_layers=frozen_layers,
                    due_indices=due_indices,
                )
                x, gstep = _commit(
                    x, P, logits, nb, block_length, k_commit, temperature, remasking,
                    mask_id, gstep,
                )
                continue

            fresh_k, fresh_v, attn_w, shadow_logits = _shadow_forward(
                model, kept_ids, suffix_ids, keep_positions, suffix_positions,
                want_attn=(mode == "oracle"),
            )

            s_idx = None
            c_t = None
            if mode == "oracle":
                probs = F.softmax(shadow_logits, dim=-1)
                conf = probs.max(dim=-1).values.squeeze(0)
                gen_mask = (suffix_ids == mask_id).squeeze(0)
                cand = torch.full_like(conf, float("-inf"))
                blk = slice(nb * block_length, (nb + 1) * block_length)
                cand[blk] = torch.where(gen_mask[blk], conf[blk],
                                        torch.full_like(conf[blk], float("-inf")))
                s_idx = (torch.topk(cand, k=min(k_commit, int(torch.isfinite(cand).sum()))).indices
                         if k_commit > 0 else torch.arange(0, device=device))
            elif mode == "student":
                committed = x[0, P:] != mask_id
                c_t = (decoder.transformer.wte(x[0, P:][committed]).float().mean(0)
                       if committed.any() else q_all[0])

            for l in range(L):
                if served_v[l] is None:
                    served_k[l] = fresh_k[l].clone()
                    served_v[l] = fresh_v[l].clone()
                    continue
                if R >= nkeep:
                    served_k[l] = fresh_k[l].clone()
                    served_v[l] = fresh_v[l].clone()
                    continue
                if R <= 0:
                    age[l] += 1
                    continue
                if mode == "oracle":
                    dV = ((fresh_v[l] - served_v[l]).norm(dim=-1)
                          / fresh_v[l].norm(dim=-1).clamp_min(1e-6)).mean(dim=1).squeeze(0)
                    sstar = (attn_w[l][0][:, s_idx, :].mean(dim=0).mean(dim=0)
                             if s_idx.numel() > 0 else torch.zeros(nkeep, device=device))
                    score = sstar * dV
                else:
                    score = refresh_student(u_all[l], q_all[l], c_t, age[l], gstep / steps, l)
                due = torch.topk(score, k=R).indices
                served_k[l][:, :, due, :] = fresh_k[l][:, :, due, :]
                served_v[l][:, :, due, :] = fresh_v[l][:, :, due, :]
                age[l] += 1
                age[l, due] = 0

            logits = _selective_forward(model, suffix_ids, suffix_positions, served_k, served_v)
            x, gstep = _commit(
                x, P, logits, nb, block_length, k_commit, temperature, remasking,
                mask_id, gstep,
            )
    if mode == "delta_student":
        mean_jaccard = due_jaccard_sum / max(1, due_comparisons)
        print(
            f"[delta-student-refresh] steps={gstep} changed_steps={changed_due_steps}/"
            f"{due_comparisons} mean_adjacent_jaccard={mean_jaccard:.4f} "
            f"unique_refreshed={int(ever_due.sum().item())}/{nkeep}",
            flush=True,
        )
    return x[:, P:]


def _commit(x, P, logits, nb, block_length, k_commit, temperature, remasking, mask_id, gstep):
    """Standard low-confidence commit, shared by both refresh paths."""
    x0 = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise RuntimeError(f"unsupported remasking: {remasking}")
    x0_p[:, (nb + 1) * block_length:] = -float("inf")
    mask_index = (x[:, P:] == mask_id)
    x0 = torch.where(mask_index, x0, x[:, P:])
    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))
    transfer = torch.zeros_like(x0, dtype=torch.bool)
    if k_commit > 0:
        transfer[0, torch.topk(confidence[0], k=k_commit).indices] = True
    x[:, P:][transfer] = x0[transfer]
    return x, gstep + 1
