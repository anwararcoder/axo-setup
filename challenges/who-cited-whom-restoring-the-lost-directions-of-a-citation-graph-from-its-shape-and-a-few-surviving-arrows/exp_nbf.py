"""Development only: pair-conditioned message passing (NBFNet / labeling-trick family), from scratch on CPU.

For every source paper p of a graph the network runs message passing with p marked, so each paper q ends with a
vector describing q AS SEEN FROM p (the paths between them, the surviving arrows along those paths, the papers they
pass through). A link (p, q) is scored from H[p -> q] and H[q -> p] by an antisymmetric head.
Same folds, canonical masks and output format as exp_gnn.py.

    python -I exp_nbf.py <public_dir> [fold=0] [epochs=20] [d=32] [layers=4] [rounds=1] [bs=8] [seed=0] [tag=name]

rounds > 1: as in exp_gnn.py, the predicted directions of the answered links are written into a soft 'who cites
whom' matrix next to the surviving ones and the next round passes messages along it (training loss summed).
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


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = load("exp_gnn")


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_, self.und, self.out, self.inn = (nn.Linear(d, d) for _ in range(4))
        self.norm = nn.LayerNorm(d)

    def forward(self, H, U, K, deg, mask):
        # H: [B, S, N, d] (source, paper); U, K: [B, 1, N, N] shared by all sources of a graph
        m_und = torch.matmul(U, H) / deg
        m_out = torch.matmul(K, H)
        m_in = torch.matmul(K.transpose(-1, -2), H)
        z = self.self_(H) + self.und(m_und) + self.out(m_out) + self.inn(m_in)
        return (H + torch.relu(self.norm(z))) * mask


class Net(nn.Module):
    def __init__(self, d_in, d, layers, rounds=1):
        super().__init__()
        self.inp = nn.Linear(d_in + 2, d)
        self.boundary = nn.Parameter(torch.randn(d) * 0.1)  # marks the source paper
        self.rounds = nn.ModuleList(nn.ModuleList(Layer(d) for _ in range(layers)) for _ in range(rounds))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(3 * d + 1, d), nn.ReLU(), nn.Linear(d, 1)) for _ in range(rounds))

    def forward(self, X, U, K, node_mask, bi, p, q, all_rounds=False):
        B, N, _ = X.shape
        kout, kin = K.sum(2, keepdim=True), K.sum(1).unsqueeze(2)
        x = self.inp(torch.cat([X, torch.log1p(kout), torch.log1p(kin)], 2)) * node_mask      # B, N, d
        eye = torch.eye(N).view(1, N, N, 1)
        H = x[:, None, :, :] + eye * self.boundary                                               # B, S, N, d
        mask = node_mask[:, None, :, :] * node_mask[:, :, None, :]                               # B, S, N, 1
        H = H * mask
        U4 = U[:, None]
        deg = U.sum(2).view(B, 1, N, 1).clamp(min=1.0)
        cn = torch.bmm(U, U)[bi, p, q].unsqueeze(1)
        directed, outputs = K, []
        for layers, head in zip(self.rounds, self.heads):
            for layer in layers:
                H = layer(H, U4, directed[:, None], deg, mask)
            a, b = H[bi, p, q], H[bi, q, p]   # q seen from p, p seen from q
            fw = head(torch.cat([a, b, a * b, cn], 1))
            bw = head(torch.cat([b, a, a * b, cn], 1))
            logit = (fw - bw).squeeze(1)      # antisymmetric logit of "p cites q"
            outputs.append(logit)
            prob = torch.sigmoid(logit)
            directed = K.index_put((bi, p, q), prob).index_put((bi, q, p), 1.0 - prob)
        return outputs if all_rounds else outputs[-1]


def main():
    opts = dict(fold=0, epochs=20, d=32, layers=4, rounds=1, lr=2e-3, wd=0.01, bs=8, seed=0, tag="nbf")
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
    allX = np.vstack([prep[i]["X"] for i in tr_idx])
    mean, std = allX.mean(0), allX.std(0) + 1e-6

    G.make_deterministic(opts["seed"])
    net = Net(prep[tr_idx[0]]["X"].shape[1], opts["d"], opts["layers"], opts["rounds"])
    optim = torch.optim.AdamW(net.parameters(), lr=opts["lr"], weight_decay=opts["wd"])
    steps = opts["epochs"] * ((len(tr_idx) + opts["bs"] - 1) // opts["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=opts["lr"], total_steps=steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(opts["epochs"]):
        # graphs of similar size together keeps padding (and the S x N x N tensors) small
        rng = np.random.RandomState(opts["seed"] * 7919 + epoch)
        order = sorted(tr_idx, key=lambda i: (prep[i]["n"], rng.rand()))
        chunks = [order[s:s + opts["bs"]] for s in range(0, len(order), opts["bs"])]
        net.train()
        for c in rng.permutation(len(chunks)):
            ids = chunks[c]
            keeps = [G.mask_of(graphs[i], 1000 + epoch, i) for i in ids]
            X, U, K, nm, bi, p, q, y, _, _ = G.batch([prep[i] for i in ids], keeps, mean, std)
            loss = sum(loss_fn(lg, y) for lg in net(X, U, K, nm, bi, p, q, all_rounds=True))
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    net.eval()
    va_keep = [G.mask_of(graphs[i], 777, i) for i in va_idx]
    probs = []
    with torch.no_grad():
        for i, keep in zip(va_idx, va_keep):  # one held-out graph at a time
            X, U, K, nm, bi, p, q, y, _, _ = G.batch([prep[i]], [keep], mean, std)
            probs.append(torch.sigmoid(net(X, U, K, nm, bi, p, q)).numpy())
    pg = np.concatenate(probs)
    yv = np.concatenate([graphs[i]["y"][~k] for i, k in zip(va_idx, va_keep)])
    print(f"fold {opts['fold']}: pair-conditioned MCC {matthews_corrcoef(yv, (pg > 0.5).astype(int)):.4f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez_compressed(f"out/nbf_{opts['tag']}_f{opts['fold']}.npz", y=yv, net=pg)


if __name__ == "__main__":
    main()
