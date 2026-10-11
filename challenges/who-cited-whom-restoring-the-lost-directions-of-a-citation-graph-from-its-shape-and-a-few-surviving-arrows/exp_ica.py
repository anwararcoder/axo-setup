"""Development only: iterative classification (stage 2 learns from stage-1 orientations summed per paper).

    python -I exp_ica.py <public_dir> [fold=0] [stages=2] [inner=4] [key=value ...]

Outer split = dev_eval.py's fixed 4 folds of whole shape clusters. Inside the training part, stage-1 predictions
for each training graph come from a model that did not see that graph (inner folds), so stage 2 learns from
predictions of the same quality as on unseen graphs.
"""
import ast
import importlib.util
import sys
import time
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


def masked(g, seed, gi, share):
    """The graph with a fixed surviving-direction draw: arrays for the answered links and the known ones."""
    rng = np.random.RandomState(seed * 1_000_003 + gi)
    keep = rng.random_sample(len(g["a"])) < share
    src = np.where(g["y"] == 1, g["a"], g["b"])
    dst = np.where(g["y"] == 1, g["b"], g["a"])
    n = int(max(g["a"].max(), g["b"].max())) + 1
    return dict(n=n, a=g["a"][~keep], b=g["b"][~keep], y=g["y"][~keep], all_a=g["a"], all_b=g["b"],
                ks=src[keep], kd=dst[keep])


def agg_tables(m, p_ab):
    """Stage-1 probabilities summed per paper. P[i, j] = P(i cites j) on linked pairs, known directions exact."""
    n = m["n"]
    U = np.zeros((n, n))
    U[m["all_a"], m["all_b"]] = U[m["all_b"], m["all_a"]] = 1.0
    P = np.zeros((n, n))
    P[m["a"], m["b"]] = p_ab
    P[m["b"], m["a"]] = 1.0 - p_ab
    P[m["ks"], m["kd"]] = 1.0
    P[m["kd"], m["ks"]] = 0.0
    deg = U.sum(1)
    cited, cites = P.sum(0), P.sum(1)
    share = cited / np.maximum(deg, 1.0)
    nbr_share = (U @ share) / np.maximum(deg, 1.0)
    nbr_share_max = (U * share[None, :]).max(1)
    return dict(P=P, deg=deg, cited=cited, cites=cites, share=share, nbr_share=nbr_share,
                nbr_share_max=nbr_share_max, chains=P @ P)


def agg_pair(A, p, q):
    deg, P = A["deg"], A["P"]
    # leave-one-out shares: the link (p, q) itself is left out of its own papers' sums
    loo_p = np.where(deg[p] > 1, (A["cited"][p] - P[q, p]) / np.maximum(deg[p] - 1, 1), 0.5)
    loo_q = np.where(deg[q] > 1, (A["cited"][q] - P[p, q]) / np.maximum(deg[q] - 1, 1), 0.5)
    return np.column_stack([
        P[p, q], A["share"][p], A["share"][q], loo_p, loo_q, loo_q - loo_p,
        A["nbr_share"][p], A["nbr_share"][q], A["nbr_share_max"][p], A["nbr_share_max"][q],
        A["cited"][p], A["cited"][q], A["cites"][p], A["cites"][q],
        A["chains"][p, q], A["chains"][q, p], A["chains"][p, q] - A["chains"][q, p],
    ])


def stage_rows(ms, tables, aggs, both=True):
    xs, ys = [], []
    for m, t, A in zip(ms, tables, aggs):
        base = S.pair_features(t, m["a"], m["b"])
        extra = [] if A is None else [agg_pair(A, m["a"], m["b"])]
        xs.append(np.hstack([base] + extra))
        ys.append(m["y"])
        if both:
            base = S.pair_features(t, m["b"], m["a"])
            extra = [] if A is None else [agg_pair(A, m["b"], m["a"])]
            xs.append(np.hstack([base] + extra))
            ys.append(1 - m["y"])
    return np.vstack(xs), np.concatenate(ys)


def predict(model, m, t, A):
    """Symmetric P(a cites b) for the answered links of one graph."""
    def x(p, q):
        return np.hstack([S.pair_features(t, p, q)] + ([] if A is None else [agg_pair(A, p, q)]))
    f = model.predict_proba(x(m["a"], m["b"]))[:, 1]
    r = model.predict_proba(x(m["b"], m["a"]))[:, 1]
    return 0.5 * (f + 1.0 - r)


def main():
    opts = dict(fold=0, stages=2, inner=4, eval_seed=777, train_seed=0)
    cfg = dict(S.CONFIG)
    for arg in sys.argv[2:]:
        k, v = arg.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        (opts if k in opts else cfg)[k] = v
    t0 = time.time()
    graphs = S.load_train_graphs(Path(sys.argv[1]))
    fold = D.make_folds(D.graph_groups(graphs, 12), 4)
    tr_idx = [i for i in range(len(graphs)) if fold[i] != opts["fold"]]
    va_idx = [i for i in range(len(graphs)) if fold[i] == opts["fold"]]
    tr = [masked(graphs[i], opts["train_seed"], i, cfg["known_share"]) for i in tr_idx]
    va = [masked(graphs[i], opts["eval_seed"], i, cfg["known_share"]) for i in va_idx]
    t_tr = [S.graph_tables(m["n"], m["all_a"], m["all_b"], m["ks"], m["kd"]) for m in tr]
    t_va = [S.graph_tables(m["n"], m["all_a"], m["all_b"], m["ks"], m["kd"]) for m in va]
    inner = np.random.RandomState(1).permutation(len(tr)) % opts["inner"]

    agg_tr, agg_va = [None] * len(tr), [None] * len(va)
    saved = {}
    for stage in range(1, opts["stages"] + 1):
        # out-of-fold predictions on the training graphs for the next stage
        new_tr = [None] * len(tr)
        if stage < opts["stages"]:
            for k in range(opts["inner"]):
                fit_ids = [i for i in range(len(tr)) if inner[i] != k]
                X, y = stage_rows([tr[i] for i in fit_ids], [t_tr[i] for i in fit_ids], [agg_tr[i] for i in fit_ids])
                model = S.fit(X, y, cfg)
                for i in np.nonzero(inner == k)[0]:
                    new_tr[i] = agg_tables(tr[i], predict(model, tr[i], t_tr[i], agg_tr[i]))
        X, y = stage_rows(tr, t_tr, agg_tr)
        model = S.fit(X, y, cfg)
        probs = [predict(model, m, t, A) for m, t, A in zip(va, t_va, agg_va)]
        yv, pv = np.concatenate([m["y"] for m in va]), np.concatenate(probs)
        print(f"stage {stage}: held-out MCC {matthews_corrcoef(yv, (pv > 0.5).astype(int)):.4f} "
              f"({X.shape[1]} features, {(time.time() - t0) / 60:.1f} min)", flush=True)
        # orientation by comparing leave-one-out cited shares alone
        aggs = [agg_tables(m, p) for m, p in zip(va, probs)]
        cmp = np.concatenate([agg_pair(A, m["a"], m["b"])[:, 5] for A, m in zip(aggs, va)])
        print(f"         orient by cited-share comparison: MCC {matthews_corrcoef(yv, (cmp > 0).astype(int)):.4f}",
              flush=True)
        agg_tr, agg_va = new_tr, aggs
        saved[f"s{stage}"] = pv
    # same order as exp_gnn.py: held-out graphs in index order, answered links in file order
    np.savez_compressed(f"out/ica_f{opts['fold']}.npz", y=yv, **saved)


if __name__ == "__main__":
    main()
