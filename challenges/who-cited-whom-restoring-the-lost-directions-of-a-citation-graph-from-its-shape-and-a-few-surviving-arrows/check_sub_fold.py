"""Development only: run a submission file's OWN functions on one dev fold (canonical masks), seeds as given.

    python -I check_sub_fold.py <solution_file.py> <public_dir> [fold=0] [seeds=[0]]
"""
import ast
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

HERE = Path(__file__).parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    V = load(sys.argv[1], "candidate")
    D = load(HERE / "dev_eval.py", "dev_eval")
    S1 = load(HERE / "solution_v1.py", "solution_v1")
    opts = dict(fold=0, seeds=[0])
    for arg in sys.argv[3:]:
        k, v = arg.split("=", 1)
        opts[k] = ast.literal_eval(v)
    t0 = time.time()
    V.torch.set_num_threads(V.CPU_THREADS)
    graphs = V.load_train_graphs(Path(sys.argv[2]))
    fold = D.make_folds(D.graph_groups(S1.load_train_graphs(Path(sys.argv[2])), 12), 4)
    tr = [g for g, f in zip(graphs, fold) if f != opts["fold"]]
    va_idx = [i for i in range(len(graphs)) if fold[i] == opts["fold"]]
    cfg = dict(V.CONFIG, seeds=opts["seeds"])
    prepared = [V.prepare(int(max(g["a"].max(), g["b"].max())) + 1, g["a"], g["b"], None, None) for g in tr]
    stacked = np.vstack([pr["X"] for pr in prepared])
    mean, std = stacked.mean(0), stacked.std(0) + 1e-6
    nets = [V.train_net(tr, prepared, mean, std, cfg, s, lambda f: None) for s in cfg["seeds"]]
    ys, ps = [], []
    for i in va_idx:
        g = graphs[i]
        keep = np.random.RandomState(777 * 1_000_003 + i).random_sample(len(g["a"])) < 0.1
        src, dst = np.where(g["y"] == 1, g["a"], g["b"]), np.where(g["y"] == 1, g["b"], g["a"])
        n = int(max(g["a"].max(), g["b"].max())) + 1
        item = V.prepare(n, g["a"], g["b"], src[keep], dst[keep])
        item["ask_a"], item["ask_b"] = g["a"][~keep], g["b"][~keep]
        ps.append(V.predict_graph(nets, item, mean, std))
        ys.append(g["y"][~keep])
    y, p = np.concatenate(ys), np.concatenate(ps)
    print(f"{Path(sys.argv[1]).name} fold {opts['fold']} seeds {opts['seeds']}: MCC "
          f"{matthews_corrcoef(y, (p > 0.5).astype(int)):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez_compressed(f"out/sub_{Path(sys.argv[1]).stem}_f{opts['fold']}.npz", y=y, p=p)


if __name__ == "__main__":
    main()
