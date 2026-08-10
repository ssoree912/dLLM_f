"""Train the refresh (drift) student -- our method, §6/§8.

Per (step t, layer l, kept token i) the student predicts a refresh score z from features
that are available at inference time only:

    phi = [ u_{l,i} ; q_l ; u.q ; c_t ; u.c_t ; E_age(age) ; step/T ; E_layer(l) ]

    u_{l,i} = proj(prefill_hidden[l, i])         q_l = proj(pool(prefill_hidden[l, question]))
    c_t     = proj(mean wte(committed tokens up to t))   age = steps since last refresh

Labels are the oracle refresh utility r*_{t,l,i} from refresh_teacher. Loss (§8):
    L = KL(softmax(r*/tau) || softmax(z/tau))  +  lambda * rank-hinge(Top-R vs rest)

Prefill hidden is recomputed per shard with one frozen-model forward (cheap: 1 pass, not
128 steps), so nothing per-token is stored on disk. Model stays frozen; only the small
student head trains.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class RefreshStudent(nn.Module):
    def __init__(self, hidden=4096, proj=256, mlp=512, n_layers=32, age_dim=16, layer_dim=16):
        super().__init__()
        self.p_tok = nn.Linear(hidden, proj)
        self.p_q = nn.Linear(hidden, proj)
        self.p_c = nn.Linear(hidden, proj)
        self.layer_emb = nn.Embedding(n_layers, layer_dim)
        self.age_proj = nn.Linear(1, age_dim)
        feat = proj * 5 + age_dim + layer_dim + 1  # u, q, u*q, c, u*c + age + layer + step
        self.mlp = nn.Sequential(nn.Linear(feat, mlp), nn.GELU(), nn.Linear(mlp, 1))

    def forward(self, u, q, c, age, step_frac, layer_id):
        # u:[N,H] q:[H] c:[H] age:[N] step_frac:scalar layer_id:int
        u = self.p_tok(u)                       # [N,proj]
        q = self.p_q(q).unsqueeze(0)            # [1,proj]
        c = self.p_c(c).unsqueeze(0)            # [1,proj]
        n = u.shape[0]
        qexp = q.expand(n, -1)
        cexp = c.expand(n, -1)
        lay = self.layer_emb(torch.tensor(layer_id, device=u.device)).unsqueeze(0).expand(n, -1)
        agef = self.age_proj(age.float().unsqueeze(-1) / 128.0)
        stepf = torch.full((n, 1), float(step_frac), device=u.device)
        phi = torch.cat([u, qexp, u * qexp, cexp, u * cexp, agef, lay, stepf], dim=-1)
        return self.mlp(phi).squeeze(-1)        # [N]


def listwise_rank_loss(z, r, top_r, tau=1.0, rank_w=0.1, margin=0.05):
    # z, r: [N]
    pi_star = F.softmax(r / tau, dim=-1)
    log_pi = F.log_softmax(z / tau, dim=-1)
    kl = F.kl_div(log_pi, pi_star, reduction="sum")
    # rank hinge: Top-R by r* should score above the rest
    k = min(top_r, r.numel() - 1)
    if k <= 0:
        return kl
    pos_idx = torch.topk(r, k=k).indices
    pos_mask = torch.zeros_like(r, dtype=torch.bool)
    pos_mask[pos_idx] = True
    zp = z[pos_mask]
    zn = z[~pos_mask]
    if zp.numel() == 0 or zn.numel() == 0:
        return kl
    # sampled pairwise margin
    m = min(zp.numel(), zn.numel(), 64)
    hinge = F.relu(margin - zp[:m].unsqueeze(1) + zn[:m].unsqueeze(0)).mean()
    return kl + rank_w * hinge


@torch.no_grad()
def prefill_hidden(model, prompt_ids):
    out = model(prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                output_hidden_states=True, use_cache=False, return_dict=True)
    return out.hidden_states  # tuple[L+1] each [1,P,H]


def train(args):
    device = args.device
    model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    wte = model.model.transformer.wte
    files = sorted(glob.glob(str(args.teacher_root / "**" / "*.pt"), recursive=True))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"teacher shards: {len(files)}", flush=True)

    student = RefreshStudent(proj=args.proj_dim, mlp=args.mlp_dim).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)

    for epoch in range(args.epochs):
        total, count = 0.0, 0
        for path in files:
            rec = torch.load(path, map_location="cpu", weights_only=False)
            prompt_ids = rec["prompt_input_ids"].unsqueeze(0).to(device)
            keep = rec["keep_indices"].to(device)
            r_star = rec["r_star"].float()                    # [T,L,nkeep] cpu
            age = rec["cache_age_before"].long()              # [T,L,nkeep] cpu
            commit_step = rec["commit_step"]                  # [gen]
            gen_ids = rec["generated_input_ids"].to(device)   # [gen]
            T, L, nkeep = r_star.shape
            question = torch.arange(max(0, prompt_ids.shape[1] - 128), prompt_ids.shape[1], device=device)

            hs = prefill_hidden(model, prompt_ids)            # tuple, each [1,P,H]
            u_all = [hs[l][0].index_select(0, keep).float() for l in range(L)]       # per layer [nkeep,H]
            q_all = [hs[l][0].index_select(0, question).float().mean(0) for l in range(L)]

            # precompute c_t per step (mean wte of committed-before-t tokens)
            with torch.no_grad():
                emb = wte(gen_ids).float()                    # [gen,H] (feature, not trained through)
            steps_per_pos = commit_step.to(device)
            shard_loss = 0.0
            for t in range(0, T, args.step_stride):
                committed = (steps_per_pos >= 0) & (steps_per_pos < t)
                c_t = emb[committed].mean(0) if committed.any() else q_all[0]
                for l in range(args.min_layer, L):
                    r = r_star[t, l].to(device)
                    if r.sum() <= 0:
                        continue
                    z = student(u_all[l], q_all[l], c_t, age[t, l].to(device), t / T, l)
                    loss = listwise_rank_loss(z, r, args.refresh_tokens,
                                              rank_w=args.rank_weight, margin=args.rank_margin)
                    opt.zero_grad(); loss.backward(); opt.step()
                    shard_loss += float(loss); count += 1
            total += shard_loss
        print(f"[epoch {epoch+1}/{args.epochs}] mean_loss={total/max(1,count):.4f} updates={count}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": student.state_dict(),
                    "config": {"proj_dim": args.proj_dim, "mlp_dim": args.mlp_dim}},
                   args.output_dir / "checkpoint-last.pt")
    print("[done]", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--mlp-dim", type=int, default=512)
    p.add_argument("--refresh-tokens", type=int, default=240)
    p.add_argument("--rank-weight", type=float, default=0.1)
    p.add_argument("--rank-margin", type=float, default=0.05)
    p.add_argument("--min-layer", type=int, default=0)
    p.add_argument("--step-stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    train(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
