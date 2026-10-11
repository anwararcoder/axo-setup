"""Development only: comparison-message network (each message is an MLP of sender, receiver, their difference and the
link's features), flat edge lists, from scratch on CPU. Same folds and canonical masks as exp_gnn.py.

    python -I exp_mpnn.py <public_dir> [fold=0] [epochs=30] [d=64] [layers=4] [rounds=2] [seed=0] [tag=name]
"""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "4"

import ast
import importlib.util
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef

HERE = Path(__file__).parent
torch.set_num_threads(4)


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = load("exp_gnn")  # prepare(), mask_of(), make_deterministic()
D = G.D
S = G.S


def flat_batch(items, keeps, mean, std):
    """Concatenate graphs: node features, directed message list (sender -> receiver) with link features, queries."""
    xs, snd, rcv, ef, qa, qb, ys, qcn = [], [], [], [], [], [], [], []
    off = 0
    for it, keep in zip(items, keeps):
        n, a, b, y = it["n"], it["a"], it["b"], it["y"]
        U = np.zeros((n, n))
        U[a, b] = U[b, a] = 1.0
        cn = (U @ U)[a, b]
        src, dst = np.where(y == 1, a, b), np.where(y == 1, b, a)
        kout = np.bincount(src[keep], minlength=n).astype(np.float32)
        kin = np.bincount(dst[keep], minlength=n).astype(np.float32)
        x = (it["X"] - mean) / std
        xs.append(np.hstack([x, np.log1p(kout)[:, None], np.log1p(kin)[:, None]]))
        # known direction seen from the sender: +1 sender cites receiver, -1 receiver cites sender, 0 unknown
        known_ab = np.where(keep, np.where(y == 1, 1.0, -1.0), 0.0)  # for message a -> b
        snd += [a + off, b + off]
        rcv += [b + off, a + off]
        ef.append(np.column_stack([np.concatenate([known_ab, -known_ab]), np.log1p(np.concatenate([cn, cn]))]))
        qa.append(a[~keep] + off)
        qb.append(b[~keep] + off)
        ys.append(y[~keep])
        qcn.append(np.log1p(cn[~keep]))
        off += n
    t = lambda v, dt=np.float32: torch.from_numpy(np.asarray(v, dtype=dt))
    return dict(x=t(np.vstack(xs)), snd=t(np.concatenate(snd), np.int64), rcv=t(np.concatenate(rcv), np.int64),
                ef=t(np.vstack(ef)), qa=t(np.concatenate(qa), np.int64), qb=t(np.concatenate(qb), np.int64),
                y=t(np.concatenate(ys)), qcn=t(np.concatenate(qcn))[:, None], n=off)


class Layer(nn.Module):
    def __init__(self, d, e):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(3 * d + e, d), nn.ReLU(), nn.Linear(d, d))
        self.upd = nn.Linear(3 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, snd, rcv, ef, deg):
        hs, hr = h[snd], h[rcv]
        m = self.msg(torch.cat([hs, hr, hs - hr, ef], 1))
        total = torch.zeros_like(h).index_add_(0, rcv, m)
        z = self.upd(torch.cat([h, total / deg, total / 10.0], 1))
        return h + torch.relu(self.norm(z))


class Net(nn.Module):
    def __init__(self, d_in, d, layers, rounds):
        super().__init__()
        e = 3  # known direction, shared neighbours, predicted direction (0 in round 1)
        self.inp = nn.Linear(d_in, d)
        self.rounds = nn.ModuleList(nn.ModuleList(Layer(d, e) for _ in range(layers)) for _ in range(rounds))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(4 * d + 1, d), nn.ReLU(), nn.Linear(d, 1)) for _ in range(rounds))

    def forward(self, b):
        h = torch.relu(self.inp(b["x"]))
        deg = torch.zeros(b["n"], 1).index_add_(0, b["rcv"], torch.ones(len(b["rcv"]), 1)).clamp(min=1.0)
        # predicted direction per message (sender cites receiver, in [-1, 1]); 0 before the first prediction
        pred = torch.zeros(len(b["snd"]), 1)
        # position of each query's two messages in the message list (a -> b and b -> a)
        outputs = []
        for layers, head in zip(self.rounds, self.heads):
            ef = torch.cat([b["ef"], pred], 1)
            for layer in layers:
                h = layer(h, b["snd"], b["rcv"], ef, deg)
            hp, hq = h[b["qa"]], h[b["qb"]]
            fw = head(torch.cat([hp, hq, hp - hq, hp * hq, b["qcn"]], 1))
            bw = head(torch.cat([hq, hp, hq - hp, hp * hq, b["qcn"]], 1))
            logit = (fw - bw).squeeze(1)
            outputs.append(logit)
            prob = torch.sigmoid(logit)
            pred = torch.zeros(len(b["snd"]), 1).index_put((b["qmsg_ab"],), (2 * prob - 1)[:, None]) \
                .index_put((b["qmsg_ba"],), (1 - 2 * prob)[:, None])
        return outputs


def add_query_messages(b):
    """Index of each query's a->b and b->a message (queries are the unknown links)."""
    key = b["snd"] * (b["n"] + 1) + b["rcv"]
    order = torch.argsort(key)
    sorted_key = key[order]
    find = lambda s, r: order[torch.searchsorted(sorted_key, s * (b["n"] + 1) + r)]
    b["qmsg_ab"], b["qmsg_ba"] = find(b["qa"], b["qb"]), find(b["qb"], b["qa"])
    return b


def main():
    opts = dict(fold=0, epochs=30, d=64, layers=4, rounds=2, lr=2e-3, wd=0.01, bs=32, seed=0, tag="mpnn")
    for arg in sys.argv[2:]:
        k, v = arg.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        opts[k] = v
    t0 = time.time()
    graphs = S.load_train_graphs(Path(sys.argv[1]))
    fold = D.make_folds(D.graph_groups(graphs, 12), 4)
    tr_idx = [i for i in range(len(graphs)) if fold[i] != opts["fold"]]
    va_idx = [i for i in range(len(graphs)) if fold[i] == opts["fold"]]
    prep = {i: G.prepare(graphs[i]) for i in tr_idx + va_idx}
    allX = np.vstack([prep[i]["X"] for i in tr_idx])
    mean, std = allX.mean(0), allX.std(0) + 1e-6

    G.make_deterministic(opts["seed"])
    net = Net(prep[tr_idx[0]]["X"].shape[1] + 2, opts["d"], opts["layers"], opts["rounds"])
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
            b = add_query_messages(flat_batch([prep[i] for i in ids], keeps, mean, std))
            loss = sum(loss_fn(lg, b["y"]) for lg in net(b))
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    net.eval()
    probs, ys = [], []
    with torch.no_grad():
        for i in va_idx:  # one held-out graph at a time
            keep = G.mask_of(graphs[i], 777, i)
            b = add_query_messages(flat_batch([prep[i]], [keep], mean, std))
            probs.append(torch.sigmoid(net(b)[-1]).numpy())
            ys.append(b["y"].numpy())
    p, y = np.concatenate(probs), np.concatenate(ys).astype(int)
    print(f"fold {opts['fold']}: mpnn MCC {matthews_corrcoef(y, (p > 0.5).astype(int)):.4f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez_compressed(f"out/mpnn_{opts['tag']}_f{opts['fold']}.npz", y=y, net=p)


if __name__ == "__main__":
    main()
