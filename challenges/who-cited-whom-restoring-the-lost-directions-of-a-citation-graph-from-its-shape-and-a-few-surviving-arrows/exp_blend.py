"""Development only: blends of saved held-out probabilities and acyclic decoding, pooled over the 4 folds.

    python -I exp_blend.py <public_dir> <name=file_pattern:key> ...
    e.g.  ica3=out/ica_f{f}.npz:s3  gnn=out/gnn_r2_f{f}.npz:net

Acyclic decoding: per graph, one order of the papers (newest first) that maximises the summed log-probability of
the orientations it implies, with the surviving directions as near-hard constraints; every answered link is then
oriented by the order (no cycles possible). Found by repeated best-insertion moves from a score-sorted start.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = load("solution_v1")
D = load("dev_eval")
EPS = 1e-4


def order_decode(n, a, b, p, ks, kd, sweeps=6):
    """a, b, p: answered links with P(a cites b); ks cites kd: surviving directions. Returns oriented 0/1."""
    # log-weights per directed pair: W[i, j] = log P(i cites j) for every linked pair
    W = np.full((n, n), np.nan)
    lp = np.log(np.clip(p, EPS, 1 - EPS))
    lq = np.log(np.clip(1 - p, EPS, 1 - EPS))
    W[a, b], W[b, a] = lp, lq
    W[ks, kd], W[kd, ks] = 0.0, np.log(EPS) * 10  # a surviving direction is (almost) never reversed
    linked = ~np.isnan(W)
    Wz = np.where(linked, W, 0.0)
    # start: newest first by the summed log-odds of citing over being cited
    score = (Wz - Wz.T).sum(1)
    order = list(np.argsort(-score, kind="stable"))
    for _ in range(sweeps):
        moved = False
        for v in range(n):
            rest = [u for u in order if u != v]
            nb = np.array([linked[v, u] for u in rest])
            # placing v at slot k: nodes rest[:k] are newer (they cite v), rest[k:] are older (v cites them)
            cited_by = np.concatenate([[0.0], np.cumsum(np.where(nb, Wz[rest, v], 0.0))])
            cites = np.concatenate([np.cumsum(np.where(nb, Wz[v, rest], 0.0)[::-1])[::-1], [0.0]])
            total = cited_by + cites
            k = int(np.argmax(total))
            current = order.index(v)  # v's slot among the other nodes
            if total[k] > total[current] + 1e-9:
                order = rest[:k] + [v] + rest[k:]
                moved = True
        if not moved:
            break
    pos = np.empty(n, int)
    pos[order] = np.arange(n)
    return (pos[a] < pos[b]).astype(int)  # a earlier in the order = newer = a cites b


def main():
    graphs = S.load_train_graphs(Path(sys.argv[1]))
    fold = D.make_folds(D.graph_groups(graphs, 12), 4)
    specs = [arg.split("=", 1) for arg in sys.argv[2:]]
    ys, probs, struct = [], {name: [] for name, _ in specs}, []
    for f in range(4):
        va_idx = [i for i in range(len(graphs)) if fold[i] == f]
        y = None
        for name, spec in specs:
            path, key = spec.split(":")
            data = np.load(path.format(f=f))
            probs[name].append(data[key])
            y = data["y"] if y is None else y
            assert np.array_equal(y, data["y"]), f"{name} fold {f}: links are not aligned"
        ys.append(y)
        for i in va_idx:
            g = graphs[i]
            keep = np.random.RandomState(777 * 1_000_003 + i).random_sample(len(g["a"])) < 0.1
            struct.append((g, keep))
    y = np.concatenate(ys)
    P = {name: np.concatenate(v) for name, v in probs.items()}
    mcc = lambda p: matthews_corrcoef(y, (p > 0.5).astype(int))
    for name, p in P.items():
        print(f"{name:8s} pooled MCC {mcc(p):.4f}")
    names = list(P)
    if len(names) == 2:
        logit = {k: np.log(np.clip(v, EPS, 1 - EPS) / np.clip(1 - v, EPS, 1 - EPS)) for k, v in P.items()}
        for w in (0.2, 0.35, 0.5, 0.65, 0.8):
            z = w * logit[names[0]] + (1 - w) * logit[names[1]]
            print(f"blend {w:.2f} {names[0]} + {1 - w:.2f} {names[1]}: {mcc(1 / (1 + np.exp(-z))):.4f}")
        best = 0.5 * logit[names[0]] + 0.5 * logit[names[1]]
        P["blend50"] = 1 / (1 + np.exp(-best))
    # acyclic decoding of each candidate
    for name, p in P.items():
        out, start = [], 0
        for g, keep in struct:
            k = int((~keep).sum())
            src = np.where(g["y"] == 1, g["a"], g["b"])
            dst = np.where(g["y"] == 1, g["b"], g["a"])
            n = int(max(g["a"].max(), g["b"].max())) + 1
            out.append(order_decode(n, g["a"][~keep], g["b"][~keep], p[start:start + k], src[keep], dst[keep]))
            start += k
        print(f"{name:8s} acyclic decoding pooled MCC {matthews_corrcoef(y, np.concatenate(out)):.4f}", flush=True)


if __name__ == "__main__":
    main()
