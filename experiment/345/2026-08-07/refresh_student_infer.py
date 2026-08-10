"""Inference with the trained refresh (drift) student -- our method at test time.

Same generation loop as the oracle, but the per-step, per-layer refresh set is chosen by
the student's predicted score z (from cheap prompt/state features) instead of the oracle
r* (which needs the gold fresh values). This measures whether the student learned to pick
the right tokens to refresh. (Fresh K/V still come from a shadow forward here -- that is
the accuracy harness; the efficiency win of forwarding only the R chosen tokens is a
separate optimization and does not change which tokens get refreshed.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from refresh_oracle import _shadow_forward, _selective_forward
from refresh_student_train import RefreshStudent
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


def load_refresh_student(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    student = RefreshStudent(proj=cfg.get("proj_dim", 256), mlp=cfg.get("mlp_dim", 512))
    student.load_state_dict(ckpt["state_dict"])
    return student.to(device).eval()


@torch.inference_mode()
def generate_with_student_refresh(
    input_ids, model, keep_indices, student, refresh_tokens,
    steps=128, gen_length=128, block_length=32, mask_id=126336, min_layer=0,
):
    device = input_ids.device
    P = int(input_ids.shape[1])
    L = len(model.model.transformer.blocks)
    wte = model.model.transformer.wte
    keep = keep_indices.to(device=device, dtype=torch.long).sort().values
    nkeep = int(keep.numel())
    R = int(refresh_tokens)

    kept_ids = input_ids.index_select(dim=1, index=keep)
    keep_positions = keep.clone()
    suffix_positions = torch.arange(P, P + gen_length, device=device, dtype=torch.long)

    # prefill hidden for student features u_l, q_l
    hs = model(input_ids, attention_mask=torch.ones_like(input_ids),
               output_hidden_states=True, use_cache=False, return_dict=True).hidden_states
    question = torch.arange(max(0, P - 128), P, device=device)
    u_all = [hs[l][0].index_select(0, keep).float() for l in range(L)]
    q_all = [hs[l][0].index_select(0, question).float().mean(0) for l in range(L)]

    x = torch.full((1, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = input_ids
    served_k = [None] * L
    served_v = [None] * L
    age = torch.zeros(L, nkeep, dtype=torch.long, device=device)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    for nb in range(num_blocks):
        for si in range(steps_per_block):
            suffix_ids = x[:, P:]
            fresh_k, fresh_v, _, _ = _shadow_forward(
                model, kept_ids, suffix_ids, keep_positions, suffix_positions
            )
            committed = x[0, P:] != mask_id
            c_t = wte(x[0, P:][committed]).float().mean(0) if committed.any() else q_all[0]

            for l in range(L):
                if served_v[l] is None:
                    served_k[l] = fresh_k[l].clone()
                    served_v[l] = fresh_v[l].clone()
                    continue
                if R >= nkeep:
                    served_k[l] = fresh_k[l].clone(); served_v[l] = fresh_v[l].clone(); continue
                if R <= 0 or l < min_layer:
                    age[l] += 1
                    continue
                z = student(u_all[l], q_all[l], c_t, age[l], (nb * steps_per_block + si) / steps, l)
                due = torch.topk(z, k=R).indices
                served_k[l][:, :, due, :] = fresh_k[l][:, :, due, :]
                served_v[l][:, :, due, :] = fresh_v[l][:, :, due, :]
                age[l] += 1
                age[l, due] = 0

            logits = _selective_forward(model, suffix_ids, suffix_positions, served_k, served_v)
            x0 = torch.argmax(add_gumbel_noise(logits, temperature=0.0), dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, (nb + 1) * block_length:] = -float("inf")
            mask_index = (x[:, P:] == mask_id)
            x0 = torch.where(mask_index, x0, x[:, P:])
            conf = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))
            block_mask = x[:, P + nb * block_length: P + (nb + 1) * block_length] == mask_id
            ntt = get_num_transfer_tokens(block_mask, steps_per_block)
            cnt = int(ntt[0, si].item())
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            if cnt > 0:
                transfer[0, torch.topk(conf[0], k=cnt).indices] = True
            x[:, P:][transfer] = x0[transfer]
    return x[:, P:]
