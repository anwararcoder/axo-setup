"""Character language model from scratch as an ordering signal. For each poet fold: train a small causal
transformer LM on the training-fold stanzas (windows: first line, consecutive line pairs, last line + end), score
every hold-out stanza's start / transition / end log-likelihoods, combine with the saved v1 log-probs
(out/lp_v1_f<fold>.npz) and decode exactly. Usage: python -I exp_lm.py <solution module> <fold> [key=value]"""
import importlib.util
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from dev_eval import author_folds, row_scores, summary

spec = importlib.util.spec_from_file_location("s", sys.argv[1]); s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
fold = int(sys.argv[2])
opts = dict(a.split("=", 1) for a in sys.argv[3:])
CFG = dict(layers=3, dim=192, heads=4, epochs=6, batch=128, lr=1e-3, max_line=100, dropout=0.1)
CFG.update({k: json.loads(v) for k, v in opts.items() if k != "tag"})
TAG = opts.get("tag", f"lm_l{CFG['layers']}_d{CFG['dim']}_e{CFG['epochs']}")
DEV = "cuda:0"
BOS, MID, EOS, NL = "\x02", "\x01", "\x03", "\n"


class Block(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, causal):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        att = att.masked_fill(causal[:T, :T], float("-inf")).softmax(-1)
        x = x + self.drop(self.proj((att @ v).transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class CharLM(nn.Module):
    def __init__(self, V, cfg, max_len):
        super().__init__()
        d = cfg["dim"]
        self.emb = nn.Parameter(torch.randn(V, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg["heads"], cfg["dropout"]) for _ in range(cfg["layers"])])
        self.ln = nn.LayerNorm(d)
        self.V = V
        self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1))

    def forward(self, ids):
        x = F.one_hot(ids, self.V).float() @ self.emb + self.pos[: ids.shape[1]]
        for b in self.blocks:
            x = b(x, self.causal)
        return self.ln(x) @ self.emb.T                                  # tied output layer


def windows_of(lines, order, ml):
    L = [s.clean(x)[:ml] for x in lines]
    o = order
    out = [(BOS, L[o[0]] + NL)]
    out += [(MID + L[a] + NL, L[b] + NL) for a, b in zip(o[:-1], o[1:])]
    out.append((MID + L[o[-1]] + NL, EOS))
    return out


def encode(pairs, chars, max_len):
    ids = np.zeros((len(pairs), max_len), np.int64)
    tmask = np.zeros((len(pairs), max_len), bool)
    for i, (ctx, tgt) in enumerate(pairs):
        seq = [chars.get(c, 1) for c in ctx + tgt][:max_len]
        ids[i, :len(seq)] = seq
        tmask[i, len(ctx):len(seq)] = True                             # positions whose char is scored
    return ids, tmask


def main():
    torch.manual_seed(0); np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    table = pd.read_csv("data/train.csv"); folds = author_folds(table)
    tr, ho = table[folds != fold].reset_index(drop=True), table[folds == fold].reset_index(drop=True)
    ml = CFG["max_line"]; max_len = 2 * ml + 4
    from collections import Counter
    cnt = Counter()
    for lj in tr["lines"]:
        for x in json.loads(lj):
            cnt.update(s.clean(x)[:ml])
    chars = {c: i + 2 for i, (c, _) in enumerate(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))}
    for c in (BOS, MID, EOS, NL):
        chars.setdefault(c, len(chars) + 2)
    V = len(chars) + 2
    train_pairs = []
    for lj, od in zip(tr["lines"], tr["order"]):
        train_pairs += windows_of(json.loads(lj), [int(x[1:]) - 1 for x in od.split()], ml)
    ids, tmask = encode(train_pairs, chars, max_len)
    # train on every char after the window's first marker (context chars are text too)
    lmask = ids[:, 1:] > 0
    t0 = time.time()
    model = CharLM(V, CFG, max_len).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.05)
    N = len(ids); spe = (N + CFG["batch"] - 1) // CFG["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=CFG["epochs"] * spe, pct_start=0.05)
    gen = torch.Generator().manual_seed(0)
    model.train()
    for ep in range(CFG["epochs"]):
        perm = torch.randperm(N, generator=gen).numpy()
        tot = 0.0
        for k in range(spe):
            r = np.sort(perm[k * CFG["batch"]:(k + 1) * CFG["batch"]])
            T = int((ids[r] > 0).sum(1).max())
            x = torch.tensor(ids[r, :T], device=DEV)
            m = torch.tensor(lmask[r, :T - 1], device=DEV)
            logits = model(x[:, :-1])
            loss = (F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1), reduction="none").reshape(m.shape) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1}: train loss {tot / spe:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()

    # score the hold-out: start / transitions / end per stanza, same layout as v1's log-probs
    S = len(ho)
    lm_start = np.full((S, s.MAX_LINES), -1e9); lm_succ = np.full((S, s.MAX_LINES, s.MAX_LINES + 1), -1e9)
    pairs, where = [], []
    for si, lj in enumerate(ho["lines"]):
        L = [s.clean(x)[:ml] for x in json.loads(lj)]; n = len(L)
        for j in range(n):
            pairs.append((BOS, L[j] + NL)); where.append((si, -1, j))
            pairs.append((MID + L[j] + NL, EOS)); where.append((si, j, s.MAX_LINES))
            for b in range(n):
                if b != j:
                    pairs.append((MID + L[j] + NL, L[b] + NL)); where.append((si, j, b))
    ids, tmask = encode(pairs, chars, max_len)
    vals = np.zeros(len(pairs))
    with torch.no_grad():
        for k in range(0, len(pairs), 256):
            r = slice(k, k + 256)
            T = int((ids[r] > 0).sum(1).max())
            x = torch.tensor(ids[r, :T], device=DEV)
            lp = F.log_softmax(model(x[:, :-1]), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
            vals[r] = (lp * torch.tensor(tmask[r, 1:T], device=DEV)).sum(1).double().cpu().numpy()
    for (si, a, b), v in zip(where, vals):
        if a == -1:
            lm_start[si, b] = v
        else:
            lm_succ[si, a, b] = v
    np.savez_compressed(f"out/{TAG}_f{fold}.npz", start=lm_start, succ=lm_succ)
    print(f"scored {len(pairs)} windows ({time.time() - t0:.0f}s)", flush=True)

    v1 = np.load(f"out/lp_v1_f{fold}.npz")
    n_arr = ho["n_lines"].to_numpy()
    for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 1e3]:
        orders = [None] * S
        for n in np.unique(n_arr).tolist():
            rows = np.flatnonzero(n_arr == n)
            st = v1["start"][rows, :n] + lam * lm_start[rows, :n]
            su = v1["succ"][rows] + lam * lm_succ[rows]
            if lam == 1e3:
                st, su = lm_start[rows, :n], lm_succ[rows]
            best = s.decode(st, su, n)
            for r, o in zip(rows, best): orders[r] = o.tolist()
        sc = row_scores(ho, orders)
        print(f"RESULT {TAG} fold {fold} lambda {lam if lam < 1e3 else 'LM only'}: mean {sc.mean():.4f}, "
              f"test-size weighted {summary(ho, sc)[1]:.4f}", flush=True)
        np.save(f"out/scores_{TAG}_lam{lam}_f{fold}.npy", sc)


if __name__ == "__main__":
    main()
