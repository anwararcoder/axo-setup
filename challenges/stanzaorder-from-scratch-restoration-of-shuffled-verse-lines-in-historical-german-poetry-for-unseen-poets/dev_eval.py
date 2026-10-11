"""Development hold-out for StanzaOrder: whole poets held out (about 20 % of training stanzas), vocabulary and
model fitted on the other poets only. Prints the mean score, the score reweighted to the test size mix, and the
chance level. Usage: python -I dev_eval.py <public_dir> <solution module> [fold=0] [key=value config overrides]"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TEST_SIZES = {4: 1103, 5: 201, 6: 683, 7: 108, 8: 626, 9: 74, 10: 190}


def load_module(path):
    spec = importlib.util.spec_from_file_location("sol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def author_folds(table, k=5, seed=0):
    """k folds of whole poets, poets assigned greedily (largest first, random tie order) to the lightest fold."""
    counts = table["author_id"].value_counts()
    rng = np.random.RandomState(seed)
    authors = sorted(counts.index.tolist())
    rng.shuffle(authors)
    authors = sorted(authors, key=lambda a: -counts[a])
    load = [0] * k
    fold_of = {}
    for a in authors:
        f = int(np.argmin(load))
        fold_of[a] = f
        load[f] += counts[a]
    return table["author_id"].map(fold_of).to_numpy()


def row_scores(table, orders):
    out = []
    for gold, pred in zip(table["order"], orders):
        g = [int(x[1:]) - 1 for x in gold.split()]
        nxt = {a: b for a, b in zip(pred[:-1], pred[1:])}
        out.append(sum(nxt.get(a) == b for a, b in zip(g[:-1], g[1:])) / (len(g) - 1))
    return np.array(out)


def summary(table, scores):
    n = table["n_lines"].to_numpy()
    by_n = {int(v): float(scores[n == v].mean()) for v in sorted(np.unique(n))}
    tot = sum(TEST_SIZES.values())
    weighted = sum(TEST_SIZES[v] / tot * by_n.get(v, 0.0) for v in TEST_SIZES)
    chance = sum(TEST_SIZES[v] / tot / v for v in TEST_SIZES)
    return float(scores.mean()), weighted, by_n, chance


def main():
    public_dir = Path(sys.argv[1])
    mod = load_module(sys.argv[2])
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    fold = int(opts.pop("fold", 0))
    tag = opts.pop("tag", Path(sys.argv[2]).stem)
    cfg = dict(mod.CONFIG)
    for k, v in opts.items():
        cfg[k] = json.loads(v)
    table = pd.read_csv(public_dir / "train.csv")
    folds = author_folds(table)
    tr, ho = table[folds != fold].reset_index(drop=True), table[folds == fold].reset_index(drop=True)
    t0 = time.time()
    vocab = mod.build_vocab(tr, cfg["first_vocab"])
    dtr, dho = mod.featurise(tr, vocab), mod.featurise(ho, vocab)
    first, succ = mod.targets(tr)
    print(f"fold {fold}: train {len(tr)} stanzas, hold-out {len(ho)} stanzas / {ho.author_id.nunique()} poets, "
          f"features {time.time() - t0:.0f}s", flush=True)
    models = [mod.fit(dtr, first, succ, cfg, s, on_epoch=mod.progress_logger()) for s in cfg["seeds"]]
    if "w_pos" in cfg:
        lp = mod.log_probs(models, dho)
        np.savez_compressed(Path("out") / f"lp_{tag}_f{fold}.npz", **lp)
        for wp, wb in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.25), (0.5, 0.25), (1.0, 0.25), (1.0, 0.5)]:
            g = dict(cfg, w_pos=wp, w_before=wb)
            s_ = row_scores(ho, mod.predict_orders(models, dho, g, lp=lp))
            print(f"  decode w_pos={wp} w_before={wb}: mean {s_.mean():.4f}, test-size weighted {summary(ho, s_)[1]:.4f}")
        orders = mod.predict_orders(models, dho, cfg, lp=lp)
        tr_orders = mod.predict_orders(models, dtr, cfg)
    else:
        orders = mod.predict_orders(models, dho)
        tr_orders = mod.predict_orders(models, dtr)
    sc = row_scores(ho, orders)
    mean, weighted, by_n, chance = summary(ho, sc)
    ident = row_scores(ho, [list(range(n)) for n in ho["n_lines"]])
    tr_sc = row_scores(tr, tr_orders)
    out = Path("out")
    out.mkdir(exist_ok=True)
    np.save(out / f"scores_{tag}_f{fold}.npy", sc)
    print(f"RESULT {tag} fold {fold}: hold-out mean {mean:.4f}, test-size weighted {weighted:.4f}, "
          f"train fit {tr_sc.mean():.4f}, identity order {ident.mean():.4f}, chance (test mix) {chance:.4f}, "
          f"per-row sd {sc.std():.3f} -> SE {sc.std() / np.sqrt(len(sc)):.4f}, {time.time() - t0:.0f}s total")
    print("by size:", {k: round(v, 3) for k, v in by_n.items()})


if __name__ == "__main__":
    main()
