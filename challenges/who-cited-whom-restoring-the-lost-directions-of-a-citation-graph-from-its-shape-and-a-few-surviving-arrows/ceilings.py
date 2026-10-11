"""Development only: what each idea could score if its part were perfect (no training).

    python -I ceilings.py <public_dir>

Each line orients every training link from one per-paper number (the paper with the larger "age" value is cited)
and reports MCC over all training links; ties are split by a fair coin with a fixed seed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

public_dir = Path(sys.argv[1])
links = pd.read_csv(public_dir / "train.csv").merge(pd.read_csv(public_dir / "train_labels.csv"), on="link_id")
rng = np.random.RandomState(0)

results = {}
link_share_source = []
for gid, df in links.groupby("graph_id", sort=True):
    a = df.paper_a.to_numpy() - 1
    b = df.paper_b.to_numpy() - 1
    y = df.a_cites_b.to_numpy()
    n = int(max(a.max(), b.max())) + 1
    src, dst = np.where(y == 1, a, b), np.where(y == 1, b, a)
    indeg = np.bincount(dst, minlength=n).astype(float)    # times cited inside the graph
    outdeg = np.bincount(src, minlength=n).astype(float)   # references inside the graph
    deg = indeg + outdeg
    U = np.zeros((n, n))
    U[a, b] = U[b, a] = 1
    # peeling: repeatedly remove the smallest-degree node; removed early = newest
    d, alive, peel = deg.copy(), np.ones(n, bool), np.zeros(n)
    for t in range(n):
        c = np.nonzero(alive)[0]
        v = c[np.argmin(d[c])]
        peel[v] = t
        alive[v] = False
        d[U[v] > 0] -= 1
    # each "age" score: larger = older = cited
    scores = {
        "oracle in-degree (times cited)": indeg,
        "oracle in - out": indeg - outdeg,
        "oracle cited share in/(in+out)": indeg / np.maximum(deg, 1),
        "oracle source flag (never cited -> newest)": (indeg > 0).astype(float),
        "degree": deg,
        "peeling order (removed later = older)": peel,
    }
    for name, s in scores.items():
        diff = s[b] - s[a]  # > 0: b is older, so a cites b
        pred = np.where(diff > 0, 1, np.where(diff < 0, 0, rng.randint(0, 2, len(a))))
        results.setdefault(name, ([], []))
        results[name][0].append(y)
        results[name][1].append(pred)
    link_share_source.append((indeg[src] == 0).mean())

for name, (ys, ps) in results.items():
    print(f"{matthews_corrcoef(np.concatenate(ys), np.concatenate(ps)):.4f}  {name}")
print(f"share of links whose citing paper is never cited (newest papers): median {np.median(link_share_source):.3f}")
