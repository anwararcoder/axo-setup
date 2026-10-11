"""Development only, never submitted: grouped hold-out of a candidate solution file.

    python -I dev_eval.py <solution_file.py> <public_dir> [tag=name] [run_folds=1] [folds=4] [key=value ...]

Test graphs come from research areas with no training graph, and the area is not in the data. So training graphs
are clustered on simple shape statistics into 12 large groups and whole groups are held out (many small groups
overstate the score, lessons.md). Held-out graphs get their own simulated surviving directions (a different seed
from training), and the score is MCC over their answered links pooled, as on the board.
Prints one line per fold (held-out MCC, fitted-rows MCC, per-graph spread) and `RESULT <tag>: ...`; saves the
held-out probabilities to out/oof_<tag>.npz.
"""
import ast
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef

OPTIONS = dict(tag="run", folds=4, run_folds=1, n_groups=12, eval_seed=777, fit_score_graphs=100)


def load_module(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_groups(graphs, n_groups):
    """12 large groups of graphs with similar shape (size, density, degree spread)."""
    stats = []
    for g in graphs:
        n = int(max(g["a"].max(), g["b"].max())) + 1
        deg = np.bincount(np.concatenate([g["a"], g["b"]]), minlength=n).astype(float)
        m = len(g["a"])
        stats.append([np.log(n), np.log(m), m / (n * (n - 1) / 2), deg.std() / deg.mean(), deg.max() / n])
    z = np.asarray(stats)
    z = (z - z.mean(0)) / z.std(0)
    return KMeans(n_groups, n_init=10, random_state=0).fit_predict(z)


def make_folds(groups, folds):
    names, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    fold_of_group, load = np.zeros(len(names), int), np.zeros(folds, int)
    for g in np.argsort(-sizes, kind="stable"):
        f = int(load.argmin())
        fold_of_group[g], load[f] = f, load[f] + sizes[g]
    return fold_of_group[inverse]


def evaluate(S, model, graphs, cfg, seed):
    """Held-out probabilities for the unknown links of `graphs` under a simulated mask; returns y, prob, graph index."""
    ys, ps, gis = [], [], []
    for gi, g in enumerate(graphs):
        n = int(max(g["a"].max(), g["b"].max())) + 1
        rng = np.random.RandomState(seed * 1_000_003 + gi)
        keep = rng.random_sample(len(g["a"])) < cfg["known_share"]
        src = np.where(g["y"] == 1, g["a"], g["b"])
        dst = np.where(g["y"] == 1, g["b"], g["a"])
        t = S.graph_tables(n, g["a"], g["b"], src[keep], dst[keep])
        ps.append(S.predict_pairs(model, t, g["a"][~keep], g["b"][~keep]))
        ys.append(g["y"][~keep])
        gis.append(np.full((~keep).sum(), gi))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(gis)


def main():
    S = load_module(sys.argv[1])
    public_dir = Path(sys.argv[2])
    cfg, opts = dict(S.CONFIG), dict(OPTIONS)
    for arg in sys.argv[3:]:
        key, value = arg.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        (opts if key in opts else cfg)[key] = value
    print("config changes", {k: v for k, v in cfg.items() if S.CONFIG.get(k) != v},
          "| options", {k: v for k, v in opts.items() if OPTIONS[k] != v}, flush=True)

    graphs = S.load_train_graphs(public_dir)
    # n_groups=0: every graph is its own group (a random-graph split, for comparison only)
    groups = np.arange(len(graphs)) if opts["n_groups"] == 0 else graph_groups(graphs, opts["n_groups"])
    fold = make_folds(groups, opts["folds"])
    print("graphs per fold", np.bincount(fold).tolist(), flush=True)

    scores, all_y, all_p, all_f = [], [], [], []
    t_all = time.time()
    for f in range(opts["run_folds"]):
        t0 = time.time()
        tr = [g for g, k in zip(graphs, fold) if k != f]
        va = [g for g, k in zip(graphs, fold) if k == f]
        X, y = S.training_rows(tr, cfg)
        model = S.fit(X, y, cfg)
        yv, pv, gv = evaluate(S, model, va, cfg, opts["eval_seed"])
        score = matthews_corrcoef(yv, (pv > 0.5).astype(int))
        per_graph = [matthews_corrcoef(yv[gv == i], (pv[gv == i] > 0.5).astype(int)) for i in np.unique(gv)]
        sub = tr[:: max(1, len(tr) // opts["fit_score_graphs"])]
        yt, pt, _ = evaluate(S, model, sub, cfg, cfg["seed"])  # the training draw: rows the model was fitted on
        fit_score = matthews_corrcoef(yt, (pt > 0.5).astype(int))
        acc = ((pv > 0.5) == yv).mean()
        print(f"fold {f}: {len(va)} graphs, {len(yv)} links, MCC {score:.4f} (acc {acc:.4f}), fitted-rows MCC "
              f"{fit_score:.4f}, per-graph MCC sd {np.std(per_graph):.3f} ({time.time() - t0:.0f}s)", flush=True)
        scores.append(score)
        all_y.append(yv), all_p.append(pv), all_f.append(np.full(len(yv), f))
    Path("out").mkdir(exist_ok=True)
    yv, pv = np.concatenate(all_y), np.concatenate(all_p)
    np.savez_compressed(f"out/oof_{opts['tag']}.npz", y=yv, p=pv, fold=np.concatenate(all_f))
    pooled = matthews_corrcoef(yv, (pv > 0.5).astype(int))
    print(f"RESULT {opts['tag']}: folds {[round(s, 4) for s in scores]} mean {np.mean(scores):.4f} pooled {pooled:.4f} "
          f"({(time.time() - t_all) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
