"""Development only: graph transformer from scratch (CPU). Full attention over a graph's papers with learned biases
from the links, the surviving directions, shortest-path distance and (round 2) the predicted directions.
Same folds, canonical masks and output format as exp_gnn.py.

    python -I exp_gt.py <public_dir> [fold=0] [epochs=30] [d=64] [heads=4] [layers=4] [rounds=2] [seed=0] [tag=name]
"""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "4"

import ast
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef

HERE = Path(__file__).parent
torch.set_num_threads(4)
MAXD = 7  # distances 0..6, 7 = farther or unreachable


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = load("exp_gnn")


def distances(n, a, b):
    U = np.zeros((n, n), bool)
    U[a, b] = U[b, a] = True
    dist = np.full((n, n), MAXD, np.int64)
    np.fill_diagonal(dist, 0)
    reached = np.eye(n, dtype=bool)
    frontier = np.eye(n, dtype=bool)
    for k in range(1, MAXD):
        frontier = ((frontier.astype(np.float64) @ U) > 0) & ~reached
        dist[frontier] = k
        reached |= frontier
    return dist


class Block(nn.Module):
    def __init__(self, d, heads, n_bias):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.bias = nn.Linear(n_bias, heads)          # per-head weights of the bias channels
        self.dist = nn.Embedding(MAXD + 1, heads)      # per-head bias by shortest-path distance
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, h, channels, dist, key_mask, node_mask):
        B, N, d = h.shape
        q, k, v = self.qkv(self.n1(h)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / self.dh ** 0.5                       # B, H, N, N
        scores = scores + self.bias(channels).permute(0, 3, 1, 2) + self.dist(dist).permute(0, 3, 1, 2)
        scores = scores.masked_fill(key_mask, -1e9)
        att = torch.softmax(scores, -1)
        z = (att @ v).transpose(1, 2).reshape(B, N, d)
        h = h + self.out(z)
        h = h + self.ff(self.n2(h))
        return h * node_mask


class Net(nn.Module):
    def __init__(self, d_in, d, heads, layers, rounds):
        super().__init__()
        self.inp = nn.Linear(d_in + 2, d)
        # channels: link, known cites, known cited-by, predicted cites, predicted cited-by
        self.rounds = nn.ModuleList(nn.ModuleList(Block(d, heads, 5) for _ in range(layers)) for _ in range(rounds))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(4 * d + 1, d), nn.ReLU(), nn.Linear(d, 1)) for _ in range(rounds))

    def forward(self, X, U, K, node_mask, dist, bi, p, q, all_rounds=False):
        kout, kin = K.sum(2, keepdim=True), K.sum(1).unsqueeze(2)
        h = self.inp(torch.cat([X, torch.log1p(kout), torch.log1p(kin)], 2)) * node_mask
        key_mask = (node_mask.squeeze(2) == 0)[:, None, None, :]
        cn = torch.bmm(U, U)[bi, p, q].unsqueeze(1)
        pred = torch.zeros_like(K)
        outputs = []
        for blocks, head in zip(self.rounds, self.heads):
            channels = torch.stack([U, K, K.transpose(1, 2), pred, pred.transpose(1, 2)], -1)
            for block in blocks:
                h = block(h, channels, dist, key_mask, node_mask)
            logit = G.edge_logit(head, h, bi, p, q, cn)
            outputs.append(logit)
            prob = torch.sigmoid(logit)
            pred = torch.zeros_like(K).index_put((bi, p, q), prob).index_put((bi, q, p), 1.0 - prob)
        return outputs if all_rounds else outputs[-1]


def dist_batch(items):
    N = max(it["n"] for it in items)
    out = np.full((len(items), N, N), MAXD, np.int64)
    for i, it in enumerate(items):
        out[i, :it["n"], :it["n"]] = it["dist"]
    return torch.from_numpy(out)


def main():
    opts = dict(fold=0, epochs=30, d=64, heads=4, layers=4, rounds=2, lr=1e-3, wd=0.01, bs=32, seed=0, tag="gt")
    for arg in sys.argv[2:]:
        k, v = arg.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        opts[k] = v
    t0 = time.time()
    graphs = G.S.load_train_graphs(Path(sys.argv[1]))
    fold = G.D.make_folds(G.D.graph_groups(graphs, 12), 4)
    tr_idx = [i for i in range(len(graphs)) if fold[i] != opts["fold"]]
    va_idx = [i for i in range(len(graphs)) if fold[i] == opts["fold"]]
    prep = {i: G.prepare(graphs[i]) for i in tr_idx + va_idx}
    for it in prep.values():
        it["dist"] = distances(it["n"], it["a"], it["b"])
    allX = np.vstack([prep[i]["X"] for i in tr_idx])
    mean, std = allX.mean(0), allX.std(0) + 1e-6

    G.make_deterministic(opts["seed"])
    net = Net(prep[tr_idx[0]]["X"].shape[1], opts["d"], opts["heads"], opts["layers"], opts["rounds"])
    optim = torch.optim.AdamW(net.parameters(), lr=opts["lr"], weight_decay=opts["wd"])
    steps = opts["epochs"] * ((len(tr_idx) + opts["bs"] - 1) // opts["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=opts["lr"], total_steps=steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(opts["epochs"]):
        order = np.random.RandomState(opts["seed"] * 7919 + epoch).permutation(len(tr_idx))
        net.train()
        for s in range(0, len(order), opts["bs"]):
            ids = [tr_idx[j] for j in order[s:s + opts["bs"]]]
            keeps = [G.mask_of(graphs[i], 1000 + epoch, i) for i in ids]
            items = [prep[i] for i in ids]
            X, U, K, nm, bi, p, q, y, _, _ = G.batch(items, keeps, mean, std)
            loss = sum(loss_fn(lg, y) for lg in net(X, U, K, nm, dist_batch(items), bi, p, q, all_rounds=True))
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    net.eval()
    probs = []
    va_keep = [G.mask_of(graphs[i], 777, i) for i in va_idx]
    with torch.no_grad():
        for s in range(0, len(va_idx), 64):
            items = [prep[i] for i in va_idx[s:s + 64]]
            X, U, K, nm, bi, p, q, y, _, _ = G.batch(items, va_keep[s:s + 64], mean, std)
            probs.append(torch.sigmoid(net(X, U, K, nm, dist_batch(items), bi, p, q)).numpy())
    pg = np.concatenate(probs)
    yv = np.concatenate([graphs[i]["y"][~k] for i, k in zip(va_idx, va_keep)])
    print(f"fold {opts['fold']}: transformer MCC {matthews_corrcoef(yv, (pg > 0.5).astype(int)):.4f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez_compressed(f"out/gt_{opts['tag']}_f{opts['fold']}.npz", y=yv, net=pg)


if __name__ == "__main__":
    main()
