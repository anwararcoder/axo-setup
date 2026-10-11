"""Development only: message-passing network from scratch (CPU) against the probe's tree model, same masks.

    python -I exp_gnn.py <public_dir> [fold=0] [epochs=60] [d=64] [layers=4] [lr=2e-3] [tag=name] ...

Outer folds = dev_eval.py's fixed 4 folds of whole shape clusters. Held-out graphs get the canonical evaluation mask
(RandomState(777 * 1_000_003 + global graph index)), the same as exp_ica.py. Prints the network's MCC, the tree
model's MCC on the same links, and blends; saves both probability vectors to out/gnn_<tag>_f<fold>.npz.
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
DEVICE = "cpu"
torch.set_num_threads(4)


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = load("solution_v1")
D = load("dev_eval")


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def mask_of(g, seed, gi, share=0.1):
    rng = np.random.RandomState(seed * 1_000_003 + gi)
    return rng.random_sample(len(g["a"])) < share


def prepare(g):
    """Mask-independent per-graph data: size, links, structural node features (the probe's first 12 columns)."""
    n = int(max(g["a"].max(), g["b"].max())) + 1
    empty = np.zeros(0, np.int64)
    t = S.graph_tables(n, g["a"], g["b"], empty, empty)
    m = len(g["a"])
    graph = np.array([np.log(n), np.log(m), m / (n * (n - 1) / 2.0)])
    X = np.hstack([t["F"][:, :12], np.repeat(graph[None, :], n, axis=0)])
    X[:, [0, 3, 5, 6, 8]] = np.log1p(X[:, [0, 3, 5, 6, 8]])  # counts on a log scale
    T, acyclic = node_targets(n, g["a"], g["b"], g["y"])
    return dict(n=n, a=g["a"], b=g["b"], y=g["y"], X=X, T=T, acyclic=acyclic)


def node_targets(n, a, b, y):
    """Per-paper age targets from the TRUE directions of a training graph: position depth/(depth+height)
    (0 = newest, 1 = oldest; strictly increasing along every citation), cited share, height / max height.
    depth = longest citation chain above the paper (papers citing it, transitively), height = longest chain below."""
    src, dst = np.where(y == 1, a, b), np.where(y == 1, b, a)
    A = np.zeros((n, n), bool)
    A[src, dst] = True  # src cites dst
    indeg = A.sum(0).astype(int)
    order, ready, left = [], [v for v in range(n) if indeg[v] == 0], indeg.copy()
    while ready:
        v = ready.pop()
        order.append(v)
        for w in np.nonzero(A[v])[0]:
            left[w] -= 1
            if left[w] == 0:
                ready.append(w)
    if len(order) < n:
        return np.zeros((n, 3)), False
    depth = np.zeros(n)
    for v in order:  # parents (citers) come first
        for w in np.nonzero(A[v])[0]:
            depth[w] = max(depth[w], depth[v] + 1)
    height = np.zeros(n)
    for v in order[::-1]:
        succ = np.nonzero(A[v])[0]
        if len(succ):
            height[v] = height[succ].max() + 1
    pos = depth / np.maximum(depth + height, 1)
    share = A.sum(0) / np.maximum(A.sum(0) + A.sum(1), 1)
    return np.column_stack([pos, share, height / max(height.max(), 1)]), True


def distance_features(n, a, b):
    """Shortest-path centralities per paper (old foundational papers sit at the centre of their area's graph)."""
    U = np.zeros((n, n), bool)
    U[a, b] = U[b, a] = True
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    reached = np.eye(n, dtype=bool)
    frontier = np.eye(n, dtype=bool)
    for k in range(1, n):
        frontier = ((frontier.astype(np.float64) @ U) > 0) & ~reached
        if not frontier.any():
            break
        dist[frontier] = k
        reached |= frontier
    finite = np.where(np.isfinite(dist), dist, n)
    ecc = finite.max(1)
    closeness = (n - 1) / np.maximum(finite.sum(1), 1)
    harmonic = np.where(finite > 0, 1.0 / np.maximum(finite, 1), 0).sum(1) / (n - 1)
    reach3 = (finite <= 3).sum(1) / n
    # peeling: repeatedly remove the smallest-degree paper; removed late = central/old
    deg = U.sum(1).astype(float)
    alive, peel = np.ones(n, bool), np.zeros(n)
    for t in range(n):
        c = np.nonzero(alive)[0]
        v = c[np.argmin(deg[c])]
        peel[v] = t / (n - 1)
        alive[v] = False
        deg[U[v]] -= 1
    return np.column_stack([ecc / ecc.max(), ecc - ecc.min(), closeness, harmonic, reach3, peel])


def augment(it, rng, min_papers=25):
    """A random induced sub-graph of a prepared training graph: all citations among a subset of its papers, with the
    true directions. Kinds: drop the newest papers (an earlier snapshot of the area), drop the oldest (a later
    window), or drop random papers. Keeps the largest connected piece; None if it is too small."""
    n, a, b, y = it["n"], it["a"], it["b"], it["y"]
    kind = rng.randint(3) if it["acyclic"] else 2
    pos = it["T"][:, 0]  # 0 = newest .. 1 = oldest (exact on acyclic graphs)
    if kind == 0:
        keep = pos >= np.quantile(pos, rng.uniform(0.05, 0.3))
    elif kind == 1:
        keep = pos <= np.quantile(pos, 1.0 - rng.uniform(0.05, 0.3))
    else:
        keep = rng.random_sample(n) < rng.uniform(0.7, 0.95)
    inside = keep[a] & keep[b]
    a2, b2, y2 = a[inside], b[inside], y[inside]
    if len(a2) < min_papers:
        return None
    # largest connected piece
    adj = [[] for _ in range(n)]
    for u, v in zip(a2, b2):
        adj[u].append(v)
        adj[v].append(u)
    comp, best = -np.ones(n, int), (0, -1)
    for s in range(n):
        if comp[s] >= 0 or not keep[s] or not adj[s]:
            continue
        stack, comp[s], size = [s], s, 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = s
                    stack.append(v)
        best = max(best, (size, s))
    if best[0] < min_papers:
        return None
    in_comp = comp == best[1]
    sel = in_comp[a2]
    old = np.nonzero(in_comp)[0]
    new_id = -np.ones(n, int)
    new_id[old] = np.arange(len(old))
    return dict(a=new_id[a2[sel]], b=new_id[b2[sel]], y=y2[sel])


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_, self.und, self.out, self.inn = (nn.Linear(d, d) for _ in range(4))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, U, K, deg, node_mask):
        m_und = torch.bmm(U, h) / deg            # mean over all linked papers
        m_out = torch.bmm(K, h)                  # papers this one is known to cite
        m_in = torch.bmm(K.transpose(1, 2), h)   # papers known to cite this one
        z = self.self_(h) + self.und(m_und) + self.out(m_out) + self.inn(m_in)
        return (h + torch.relu(self.norm(z))) * node_mask


def edge_logit(head, h, bi, p, q, cn):
    hp, hq = h[bi, p], h[bi, q]
    forward = head(torch.cat([hp, hq, hp - hq, hp * hq, cn], 1))
    backward = head(torch.cat([hq, hp, hq - hp, hp * hq, cn], 1))
    return (forward - backward).squeeze(1)  # antisymmetric: logit of "p cites q"


class Net(nn.Module):
    """rounds > 1: after each round the predicted directions of the answered links are written into a soft
    'who cites whom' matrix next to the surviving ones, and the next round passes messages along it."""

    def __init__(self, d_in, d, layers, rounds=1, pos_term=0):
        super().__init__()
        self.inp = nn.Linear(d_in + 2, d)
        self.rounds = nn.ModuleList(nn.ModuleList(Layer(d) for _ in range(layers)) for _ in range(rounds))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(4 * d + 1, d), nn.ReLU(), nn.Linear(d, 1)) for _ in range(rounds))
        self.node_heads = nn.ModuleList(nn.Linear(d, 3) for _ in range(rounds))  # position, cited share, height
        self.pos_term = pos_term
        self.pos_weight = nn.Parameter(torch.zeros(rounds))

    def forward(self, X, U, K, node_mask, bi, p, q, all_rounds=False, with_nodes=False):
        kout, kin = K.sum(2, keepdim=True), K.sum(1).unsqueeze(2)
        h = torch.relu(self.inp(torch.cat([X, torch.log1p(kout), torch.log1p(kin)], 2))) * node_mask
        deg = U.sum(2, keepdim=True).clamp(min=1.0)
        cn = torch.bmm(U, U)[bi, p, q].unsqueeze(1)
        directed, outputs, nodes = K, [], []
        for r, (layers, head) in enumerate(zip(self.rounds, self.heads)):
            for layer in layers:
                h = layer(h, U, directed, deg, node_mask)
            node_out = self.node_heads[r](h)  # logits of the three per-paper targets
            nodes.append(node_out)
            logit = edge_logit(head, h, bi, p, q, cn)
            if self.pos_term:
                # q older than p (higher position) means p cites q
                logit = logit + self.pos_weight[r] * (node_out[bi, q, 0] - node_out[bi, p, 0])
            outputs.append(logit)
            prob = torch.sigmoid(logit)
            directed = K.index_put((bi, p, q), prob).index_put((bi, q, p), 1.0 - prob)
        out = outputs if all_rounds else outputs[-1]
        return (out, nodes) if with_nodes else out


def batch(items, keeps, mean, std):
    """Padded tensors for a list of prepared graphs and their surviving-direction masks."""
    N = max(it["n"] for it in items)
    B = len(items)
    X = np.zeros((B, N, items[0]["X"].shape[1]), np.float32)
    U = np.zeros((B, N, N), np.float32)
    K = np.zeros((B, N, N), np.float32)
    node_mask = np.zeros((B, N, 1), np.float32)
    T = np.zeros((B, N, 3), np.float32)
    aux_mask = np.zeros((B, N, 1), np.float32)
    bi, p, q, y = [], [], [], []
    for i, (it, keep) in enumerate(zip(items, keeps)):
        n = it["n"]
        X[i, :n] = (it["X"] - mean) / std
        T[i, :n] = it["T"]
        aux_mask[i, :n] = float(it["acyclic"])
        U[i, it["a"], it["b"]] = U[i, it["b"], it["a"]] = 1.0
        src = np.where(it["y"] == 1, it["a"], it["b"])
        dst = np.where(it["y"] == 1, it["b"], it["a"])
        K[i, src[keep], dst[keep]] = 1.0
        node_mask[i, :n] = 1.0
        bi.append(np.full((~keep).sum(), i))
        p.append(it["a"][~keep])
        q.append(it["b"][~keep])
        y.append(it["y"][~keep])
    t = lambda a: torch.from_numpy(a)
    return (t(X), t(U), t(K), t(node_mask), t(np.concatenate(bi)), t(np.concatenate(p)), t(np.concatenate(q)),
            t(np.concatenate(y).astype(np.float32)), t(T), t(aux_mask))


def main():
    opts = dict(fold=0, epochs=60, d=64, layers=4, rounds=1, lr=2e-3, wd=0.01, bs=32, seed=0, tag="gnn", tree=1,
                extra=0, aux=0.0, pos_term=0, train_frac=1.0, aug=0.0)
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
    if opts["train_frac"] < 1:  # learning curve: a fixed subset of the training graphs
        keep_n = int(round(len(tr_idx) * opts["train_frac"]))
        tr_idx = sorted(np.random.RandomState(5).permutation(tr_idx)[:keep_n].tolist())
    prep = {i: prepare(graphs[i]) for i in tr_idx + va_idx}
    if opts["extra"]:
        for i, it in prep.items():
            it["X"] = np.hstack([it["X"], distance_features(it["n"], it["a"], it["b"])])
    allX = np.vstack([prep[i]["X"] for i in tr_idx])
    mean, std = allX.mean(0), allX.std(0) + 1e-6  # fitted on training graphs only
    va_keep = [mask_of(graphs[i], 777, i) for i in va_idx]

    make_deterministic(opts["seed"])
    net = Net(prep[tr_idx[0]]["X"].shape[1], opts["d"], opts["layers"], opts["rounds"], opts["pos_term"])
    optim = torch.optim.AdamW(net.parameters(), lr=opts["lr"], weight_decay=opts["wd"])
    steps = opts["epochs"] * ((len(tr_idx) + opts["bs"] - 1) // opts["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=opts["lr"], total_steps=steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(opts["epochs"]):
        rng = np.random.RandomState(opts["seed"] * 7919 + epoch)
        order = rng.permutation(len(tr_idx))
        net.train()
        total = 0.0
        for s in range(0, len(order), opts["bs"]):
            ids = [tr_idx[j] for j in order[s:s + opts["bs"]]]
            items, keeps = [], []
            for i in ids:
                arng = np.random.RandomState((3000 + epoch) * 1_000_003 + i)
                sub = augment(prep[i], arng) if arng.rand() < opts["aug"] else None
                if sub is None:
                    items.append(prep[i])
                    keeps.append(mask_of(graphs[i], 1000 + epoch, i))  # a fresh surviving-direction draw per epoch
                else:
                    items.append(prepare(sub))
                    keeps.append(arng.random_sample(len(sub["a"])) < 0.1)
            X, U, K, nm, bi, p, q, y, T, am = batch(items, keeps, mean, std)
            logits, nodes = net(X, U, K, nm, bi, p, q, all_rounds=True, with_nodes=True)
            loss = sum(loss_fn(logit, y) for logit in logits)
            if opts["aux"]:
                # per-paper age targets (soft BCE on [0, 1] targets), acyclic training graphs only
                for node_out in nodes:
                    per = nn.functional.binary_cross_entropy_with_logits(node_out, T, reduction="none")
                    loss = loss + opts["aux"] * (per * am).sum() / (am.sum() * 3).clamp(min=1.0)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            total += loss.item() * len(y)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: loss {total / 1e5:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    net.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(va_idx), 64):
            ids = va_idx[s:s + 64]
            X, U, K, nm, bi, p, q, y, _, _ = batch([prep[i] for i in ids], va_keep[s:s + 64], mean, std)
            probs.append(torch.sigmoid(net(X, U, K, nm, bi, p, q)).numpy())
    pg = np.concatenate(probs)
    yv = np.concatenate([graphs[i]["y"][~k] for i, k in zip(va_idx, va_keep)])
    # the same score on 180 TRAINING graphs with a mask draw never used in training (structure seen, mask new)
    fit_ids = tr_idx[::max(1, len(tr_idx) // 180)]
    fit_keep = [mask_of(graphs[i], 555, i) for i in fit_ids]
    with torch.no_grad():
        X, U, K, nm, bi, p, q, y, _, _ = batch([prep[i] for i in fit_ids], fit_keep, mean, std)
        pf = torch.sigmoid(net(X, U, K, nm, bi, p, q)).numpy()
    fit_mcc = matthews_corrcoef(y.numpy().astype(int), (pf > 0.5).astype(int))
    print(f"fold {opts['fold']}: network MCC {matthews_corrcoef(yv, (pg > 0.5).astype(int)):.4f} "
          f"(training graphs, new mask: {fit_mcc:.4f}) ({(time.time() - t0) / 60:.1f} min)", flush=True)

    pt = np.full_like(pg, np.nan)
    if opts["tree"]:
        # the probe's tree model under the same masks: training draw = seed 0, global index
        xs, ys = [], []
        for i in tr_idx:
            g, keep = graphs[i], mask_of(graphs[i], 0, i)
            src, dst = np.where(g["y"] == 1, g["a"], g["b"]), np.where(g["y"] == 1, g["b"], g["a"])
            t = S.graph_tables(prep[i]["n"], g["a"], g["b"], src[keep], dst[keep])
            xs += [S.pair_features(t, g["a"][~keep], g["b"][~keep]), S.pair_features(t, g["b"][~keep], g["a"][~keep])]
            ys += [g["y"][~keep], 1 - g["y"][~keep]]
        model = S.fit(np.vstack(xs), np.concatenate(ys), S.CONFIG)
        out = []
        for i, keep in zip(va_idx, va_keep):
            g = graphs[i]
            src, dst = np.where(g["y"] == 1, g["a"], g["b"]), np.where(g["y"] == 1, g["b"], g["a"])
            t = S.graph_tables(prep[i]["n"], g["a"], g["b"], src[keep], dst[keep])
            out.append(S.predict_pairs(model, t, g["a"][~keep], g["b"][~keep]))
        pt = np.concatenate(out)
        line = [f"tree {matthews_corrcoef(yv, (pt > 0.5).astype(int)):.4f}"]
        for w in (0.3, 0.5, 0.7):
            pb = w * pg + (1 - w) * pt
            line.append(f"blend w_net={w} {matthews_corrcoef(yv, (pb > 0.5).astype(int)):.4f}")
        print(f"fold {opts['fold']}: " + ", ".join(line), flush=True)
    Path("out").mkdir(exist_ok=True)
    np.savez_compressed(f"out/gnn_{opts['tag']}_f{opts['fold']}.npz", y=yv, net=pg, tree=pt)


if __name__ == "__main__":
    main()
