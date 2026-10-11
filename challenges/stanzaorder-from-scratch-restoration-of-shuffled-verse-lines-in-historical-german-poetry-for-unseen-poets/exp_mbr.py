"""v1 on all 5 poet folds: MAP decoding (sum of log-probs, as v1) vs expected-adjacency decoding (sum of successor
probabilities, the metric's own objective). Saves hold-out log-probs per fold."""
import importlib.util, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from dev_eval import author_folds, row_scores, summary
spec = importlib.util.spec_from_file_location("s", sys.argv[1]); s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
tag = sys.argv[2]
table = pd.read_csv("data/train.csv"); folds = author_folds(table)
allsc = {"map": [], "mbr": [], "mix": []}; allho = []
for fold in range(5):
    tr, ho = table[folds != fold].reset_index(drop=True), table[folds == fold].reset_index(drop=True)
    vocab = s.build_vocab(tr, s.CONFIG["first_vocab"]); dtr, dho = s.featurise(tr, vocab), s.featurise(ho, vocab)
    first, succ = s.targets(tr)
    model = s.fit(dtr, first, succ, s.CONFIG, 0)
    start, sc = s.log_probs([model], dho)
    np.savez_compressed(f"out/lp_{tag}_f{fold}.npz", start=start, succ=sc)
    res = {}
    for name in allsc:
        orders = [None] * len(ho)
        for n in np.unique(dho["n"]).tolist():
            rows = np.flatnonzero(dho["n"] == n)
            if name == "map":
                st, su = start[rows, :n], sc[rows]
            elif name == "mbr":
                st, su = np.zeros((len(rows), n)), np.exp(sc[rows]); su[:, :, s.MAX_LINES] = 0.0
            else:   # expected adjacencies plus a small share of the first/last evidence
                st, su = 0.5 * np.exp(start[rows, :n]), np.exp(sc[rows]); su[:, :, s.MAX_LINES] *= 0.5
            best = s.decode(st, su, n)
            for r, o in zip(rows, best): orders[r] = o.tolist()
        scr = row_scores(ho, orders); allsc[name].append(scr); res[name] = summary(ho, scr)[1]
    allho.append(ho)
    print(f"fold {fold} ({len(ho)} stanzas, {ho.author_id.nunique()} poets): " + " ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
H = pd.concat(allho, ignore_index=True)
for name, v in allsc.items():
    v = np.concatenate(v); m, w, byn, ch = summary(H, v)
    print(f"POOLED {name}: mean {m:.4f}, test-size weighted {w:.4f}")
d = np.concatenate(allsc["mbr"]) - np.concatenate(allsc["map"])
print(f"mbr - map paired: {d.mean():+.4f} (SE {d.std() / np.sqrt(len(d)):.4f})")
