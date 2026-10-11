"""Development only: honest blend estimate. For each fold, blend weights (grid on the simplex, step 0.1) are chosen
on the OTHER three folds' held-out predictions and applied to this fold; the pooled score of those predictions is
the estimate. Members are saved held-out probabilities on the canonical masks.

    python -I exp_nested_blend.py name=pattern:key ...    e.g. gnn=out/gnn_r2_f{f}.npz:net
"""
import itertools
import sys

import numpy as np
from sklearn.metrics import matthews_corrcoef


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    names = [n for n, _ in specs]
    P = {n: [] for n in names}
    Y = []
    for f in range(4):
        y = None
        for n, spec in specs:
            parts = spec.split(":")
            paths, key = parts[0].split("+"), parts[1]
            arrs = [np.load(p.format(f=f)) for p in paths]  # "+" averages several files (seeds)
            P[n].append(np.mean([a[key] for a in arrs], 0))
            y = arrs[0]["y"] if y is None else y
            assert all(np.array_equal(y, a["y"]) for a in arrs), f"{n} fold {f} not aligned"
        Y.append(y)
    grid = [w for w in itertools.product(*[np.arange(0, 1.01, 0.1)] * len(names)) if abs(sum(w) - 1) < 1e-9]
    mcc = lambda y, p: matthews_corrcoef(y, (p > 0.5).astype(int))
    blend = lambda w, f: sum(wi * P[n][f] for wi, n in zip(w, names))
    for n in names:
        print(f"{n:8s} alone pooled {mcc(np.concatenate(Y), np.concatenate(P[n])):.4f}")
    nested, chosen = [], []
    for f in range(4):
        others = [k for k in range(4) if k != f]
        y_o = np.concatenate([Y[k] for k in others])
        best = max(grid, key=lambda w: mcc(y_o, np.concatenate([blend(w, k) for k in others])))
        chosen.append(best)
        nested.append(blend(best, f))
    y = np.concatenate(Y)
    print("weights chosen per fold:", [dict(zip(names, np.round(w, 1))) for w in chosen])
    print("nested per-fold:", " ".join(f"{mcc(Y[f], nested[f]):.4f}" for f in range(4)))
    print(f"NESTED blend pooled {mcc(y, np.concatenate(nested)):.4f}")


if __name__ == "__main__":
    main()
