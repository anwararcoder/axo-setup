"""Development only: compare saved held-out predictions (same 4 folds, canonical masks) against v2 and the current
nested blend. Per fold, pooled, nested blends (weights chosen on the other three folds), and a paired bootstrap over
whole held-out graphs (the unit the test is drawn in).

    python -I exp_compare.py <public_dir>
"""
import importlib.util
import itertools
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


G = load("exp_gnn")
mcc = lambda y, p: matthews_corrcoef(y, (p > 0.5).astype(int))


def member(paths, f):
    arrs = [np.load(p.format(f=f)) for p in paths]
    return arrs[0]["y"], np.mean([a["net"] for a in arrs], 0)


def nested(P, names, Y):
    grid = [w for w in itertools.product(*[np.arange(0, 1.01, 0.1)] * len(names)) if abs(sum(w) - 1) < 1e-9]
    blend = lambda w, f: sum(wi * P[n][f] for wi, n in zip(w, names))
    out, chosen = [], []
    for f in range(4):
        others = [k for k in range(4) if k != f]
        y_o = np.concatenate([Y[k] for k in others])
        best = max(grid, key=lambda w: mcc(y_o, np.concatenate([blend(w, k) for k in others])))
        chosen.append(dict(zip(names, np.round(best, 1))))
        out.append(blend(best, f))
    return out, chosen


def main():
    graphs = G.S.load_train_graphs(Path(sys.argv[1]))
    fold = G.D.make_folds(G.D.graph_groups(graphs, 12), 4)
    # links per held-out graph, in the order the experiment scripts write them
    sizes = [[int((~G.mask_of(graphs[i], 777, i)).sum()) for i in range(len(graphs)) if fold[i] == f]
             for f in range(4)]
    members = {
        "gnn3": ["out/gnn_r2_f{f}.npz", "out/gnn_r2s1_f{f}.npz", "out/gnn_r2s2_f{f}.npz"],  # = v2
        "aug": ["out/gnn_aug60_f{f}.npz"],
        "aug2": ["out/gnn_aug60_f{f}.npz", "out/gnn_aug60s1_f{f}.npz"],
        "nbf": ["out/nbf_base_f{f}.npz"],
        "nbf2": ["out/nbf_base_f{f}.npz", "out/nbf_s1_f{f}.npz"],
        "nbfr2": ["out/nbf_r2_f{f}.npz"],
        "gt": ["out/gt_base_f{f}.npz"],
    }
    P, Y = {}, None
    for name, paths in members.items():
        if not all(Path(p.format(f=f)).exists() for p in paths for f in range(4)):
            print(f"{name}: missing files, skipped")
            continue
        ys, ps = zip(*[member(paths, f) for f in range(4)])
        if Y is None:
            Y = list(ys)
        assert all(np.array_equal(a, b) for a, b in zip(Y, ys)), name
        assert all(len(Y[f]) == sum(sizes[f]) for f in range(4))
        P[name] = list(ps)
    print("member     per fold                          pooled")
    for name in P:
        print(f"{name:8s} " + " ".join(f"{mcc(Y[f], P[name][f]):.4f}" for f in range(4))
              + f"   {mcc(np.concatenate(Y), np.concatenate(P[name])):.4f}")
    combos = {
        "blend (current)": ["gnn3", "nbf", "gt", "aug"],
        "+ seeds": ["gnn3", "nbf2", "gt", "aug2"],
        "nbf -> nbfr2": ["gnn3", "nbfr2", "gt", "aug"],
        "seeds + nbfr2": ["gnn3", "nbf2", "nbfr2", "aug2"],
        "aug2 + nbf2": ["aug2", "nbf2"],
    }
    res = {"v2": P["gnn3"]}
    if "aug2" in P:
        res["aug2 alone"] = P["aug2"]  # one family, two seeds averaged: no blend weights to choose
    for label, names in combos.items():
        if all(n in P for n in names):
            res[label], chosen = nested(P, names, Y)
            print(f"\nNESTED {label}: weights per fold {chosen}")
    y = np.concatenate(Y)
    print("\ncandidate          per fold                          pooled")
    for label, pr in res.items():
        print(f"{label:18s} " + " ".join(f"{mcc(Y[f], pr[f]):.4f}" for f in range(4))
              + f"   {mcc(y, np.concatenate(pr)):.4f}")
    # paired bootstrap over whole held-out graphs
    starts = np.concatenate([[0], np.cumsum(np.concatenate(sizes))])
    blocks = [np.arange(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]
    rng = np.random.RandomState(0)
    picks = [np.concatenate([blocks[j] for j in rng.randint(0, len(blocks), len(blocks))]) for _ in range(500)]
    flat = {k: np.concatenate(v) > 0.5 for k, v in res.items()}
    print("\npaired bootstrap over graphs (500 draws): difference, sd, share of draws <= 0")
    for ref in ("v2", "blend (current)"):
        for label in res:
            if label in ("v2", ref):
                continue
            d = np.array([matthews_corrcoef(y[pk], flat[label][pk]) - matthews_corrcoef(y[pk], flat[ref][pk])
                          for pk in picks])
            print(f"{label:18s} vs {ref:16s} {d.mean():+.4f}  sd {d.std():.4f}  P<=0 {np.mean(d <= 0):.3f}  "
                  f"agree {np.mean(flat[label] == flat[ref]):.4f}")


if __name__ == "__main__":
    main()
