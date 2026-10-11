"""Development only: facts about the graphs. Usage: python -I inspect_data.py <public_dir>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

public_dir = Path(sys.argv[1])
train = pd.read_csv(public_dir / "train.csv").merge(pd.read_csv(public_dir / "train_labels.csv"), on="link_id")
test = pd.read_csv(public_dir / "test.csv")
known = pd.read_csv(public_dir / "test_known.csv")

print("label mean", train.a_cites_b.mean().round(4))
for name, df in (("train", train), ("test+known", pd.concat([test, known]))):
    g = df.groupby("graph_id")
    nodes = g[["paper_a", "paper_b"]].max().max(axis=1)
    edges = g.size()
    dens = edges / (nodes * (nodes - 1) / 2)
    print(f"{name}: graphs {g.ngroups}, nodes {nodes.min()}-{nodes.max()} median {nodes.median()}, "
          f"edges {edges.min()}-{edges.max()} median {edges.median()}, density median {dens.median():.3f}")

kf = known.groupby("graph_id").size().reindex(test.graph_id.unique(), fill_value=0)
tf = test.groupby("graph_id").size()
print("test known share", (kf / (kf + tf)).describe().round(3).to_dict(), "graphs with 0 known", int((kf == 0).sum()))


def orient(df):
    """Edges as (src cites dst)."""
    a, b, y = df.paper_a.to_numpy(), df.paper_b.to_numpy(), df.a_cites_b.to_numpy()
    return np.where(y == 1, a, b), np.where(y == 1, b, a)


# acyclicity and how much of the order the true directions fix
cyc, closure_share, src_counts, sink_counts, hamil = 0, [], [], [], 0
deg_out_corr = []
for gid, df in train.groupby("graph_id", sort=True):
    n = int(max(df.paper_a.max(), df.paper_b.max()))
    s, d = orient(df)
    A = np.zeros((n + 1, n + 1), bool)
    A[s, d] = True
    indeg = A.sum(0)[1:]
    outdeg = A.sum(1)[1:]
    # Kahn topological sort
    ind = A.sum(0).copy()
    ind[0] = 0
    order, ready = [], [v for v in range(1, n + 1) if ind[v] == 0]
    src_counts.append(len(ready))
    sink_counts.append(int((outdeg == 0).sum()))
    while ready:
        v = ready.pop()
        order.append(v)
        for w in np.nonzero(A[v])[0]:
            ind[w] -= 1
            if ind[w] == 0:
                ready.append(w)
    if len(order) < n:
        cyc += 1
        continue
    # transitive closure: share of node pairs whose order is fixed by the DAG
    R = A.copy()
    for k in order[::-1]:
        R[k] |= R[np.nonzero(A[k])[0]].any(0) if A[k].any() else R[k]
    closure_share.append(R[1:, 1:].sum() / (n * (n - 1) / 2))
    # unique topological order <=> consecutive nodes in order are linked
    hamil += all(A[order[i], order[i + 1]] for i in range(n - 1))
    deg = indeg + outdeg
    deg_out_corr.append(np.corrcoef(deg, outdeg - indeg)[0, 1])

print(f"train graphs with a cycle: {cyc}")
print(f"share of node pairs ordered by the closure: median {np.median(closure_share):.3f}")
print(f"graphs whose order is unique (Hamiltonian in time order): {hamil}")
print(f"sources per graph median {np.median(src_counts)}, sinks per graph median {np.median(sink_counts)}")
print(f"corr(degree, out-in) median {np.nanmedian(deg_out_corr):.3f}")
