"""Development harness for an Axo solution. NOT submitted.

Copy next to solution.py as solution_dev.py and fill in the two functions marked below.

    python solution_dev.py <public_dir> [tag=name] [run_folds=1] [folds=4] [prune=0.0] [key=value ...]

- A key that is a harness option (see OPTIONS) sets it; every other key=value
  overrides solution.CONFIG for this run, e.g. `epochs=20 dropout=0.4`.
- Version 1 needs one hold-out: run_folds=1 (the default). Use more folds only
  after the first submission.
- prune=X stops after the first fold if its score is below X (the reference
  configuration's score on that fold), so a losing idea costs one fold, not all.
- Prints one line per fold with the held-out score and the score on lines the
  model was trained on (a large gap means it memorises), then one line
  `RESULT <tag>: ...`. Held-out predictions are saved to out/oof_<tag>.npz when
  they are arrays, so decision rules can be tested later without retraining.
- A fold that runs far longer than its first pass projected is aborted.

It relies on solution.py exposing: CONFIG, load_train(public_dir) -> dict of
row-aligned values, fit(train, cfg, seed, on_epoch=None), predict(models, data, cfg).
"""
import ast
import inspect
import sys
import time
from pathlib import Path

import numpy as np

import solution as S

OPTIONS = dict(tag="run", folds=4, run_folds=1, prune=0.0, train_score=1000, slow_factor=3.0, split_seed=0)


# ---- challenge-specific: fill in these two ------------------------------------------------
def metric(predictions, data):
    """The challenge metric (higher is better) for `predictions` against the labelled
    rows in `data`. Check it once against the anchor scores stated in the description."""
    raise NotImplementedError("write the challenge metric here")


def group_ids(train):
    """One group id per training row; rows of a group are held out together. Choose the
    grouping that imitates how the test set differs from training (new stations, pages,
    documents, operators ...). When the data has no such id, use a FEW LARGE groups built
    from simple input statistics: many small groups share sources with training and
    overstate the score. Return None for a plain random split."""
    return None
# -------------------------------------------------------------------------------------------


def n_rows(data):
    return len(next(iter(data.values())))


def take(obj, idx):
    """Rows `idx` of a dict of row-aligned values (tensors, arrays, tables or lists)."""
    if isinstance(obj, dict):
        return {k: take(v, idx) for k, v in obj.items()}
    if hasattr(obj, "iloc"):
        return obj.iloc[idx].reset_index(drop=True)
    if isinstance(obj, list):
        return [obj[i] for i in idx]
    if type(obj).__module__.startswith("torch"):
        import torch
        return obj[torch.as_tensor(np.asarray(idx), device=obj.device)]
    return obj[idx]


def make_folds(train, folds, seed):
    n = n_rows(train)
    groups = group_ids(train)
    rng = np.random.RandomState(seed)
    if groups is None:
        fold = np.empty(n, int)
        fold[rng.permutation(n)] = np.arange(n) % folds
        return fold
    groups = np.asarray(groups)
    names, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    # balance fold sizes: largest group first, each into the currently smallest fold
    fold_of_group, load = np.zeros(len(names), int), np.zeros(folds, int)
    for g in np.argsort(-sizes, kind="stable"):
        f = int(load.argmin())
        fold_of_group[g], load[f] = f, load[f] + sizes[g]
    return fold_of_group[inverse]


def to_numpy(pred):
    """Predictions as one array if they are array-like (for saving); otherwise None."""
    if type(pred).__module__.startswith("torch"):
        return pred.detach().cpu().numpy()
    if isinstance(pred, (np.ndarray, list)):
        try:
            arr = np.asarray(pred)
        except ValueError:          # ragged
            return None
        return arr if arr.dtype != object else None
    return None


class SlowRun(Exception):
    pass


def main():
    public_dir = Path(sys.argv[1])
    cfg, opts = dict(S.CONFIG), dict(OPTIONS)
    for arg in sys.argv[2:]:
        key, value = arg.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        (opts if key in opts else cfg)[key] = value
    changed = {k: v for k, v in cfg.items() if S.CONFIG.get(k) != v}
    print("config changes", changed, "| options", {k: v for k, v in opts.items() if OPTIONS[k] != v}, flush=True)

    train = S.load_train(public_dir)
    fold = make_folds(train, opts["folds"], opts["split_seed"])
    print("fold sizes", np.bincount(fold, minlength=opts["folds"]).tolist(), flush=True)
    takes_progress = "on_epoch" in inspect.signature(S.fit).parameters
    seeds = cfg.get("seeds", [cfg.get("seed", 0)])

    scores, held_idx, held_pred = [], [], []
    t_all = time.time()
    for f in range(opts["run_folds"]):
        t0 = time.time()
        tr_idx, va_idx = np.where(fold != f)[0], np.where(fold == f)[0]
        tr, va = take(train, tr_idx), take(train, va_idx)

        state = {}

        def on_epoch(fraction, t0=t0, state=state, f=f):
            elapsed = time.time() - t0
            if "projected" not in state:
                state["projected"] = elapsed / fraction * len(seeds)
                print(f"PROJECTED fold {f}: {state['projected'] / 60:.1f} min of training", flush=True)
            elif elapsed > opts["slow_factor"] * state["projected"]:
                raise SlowRun(f"fold {f} at {elapsed / 60:.1f} min, over {opts['slow_factor']:g} x its projection")

        try:
            models = [S.fit(tr, cfg, seed, on_epoch=on_epoch) if takes_progress else S.fit(tr, cfg, seed)
                      for seed in seeds]
        except SlowRun as e:
            print(f"RESULT {opts['tag']}: ABORTED - {e}. Check power and other GPU jobs.", flush=True)
            sys.exit(3)
        pred = S.predict(models, va, cfg)
        scores.append(float(metric(pred, va)))
        sub = np.arange(0, len(tr_idx), max(1, len(tr_idx) // max(1, opts["train_score"])))
        fit_score = float(metric(S.predict(models, take(tr, sub), cfg), take(tr, sub))) if opts["train_score"] else float("nan")
        print(f"fold {f}: n={len(va_idx)} score {scores[-1]:.4f} train-score {fit_score:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        arr = to_numpy(pred)
        if arr is not None and len(arr) == len(va_idx):
            held_idx.append(va_idx)
            held_pred.append(arr)
        if f == 0 and scores[0] < opts["prune"]:
            print(f"RESULT {opts['tag']}: PRUNED after fold 0 ({scores[0]:.4f} < {opts['prune']})", flush=True)
            return

    pooled = ""
    if len(held_pred) == opts["run_folds"]:
        idx, pred = np.concatenate(held_idx), np.concatenate(held_pred)
        Path("out").mkdir(exist_ok=True)
        np.savez_compressed(f"out/oof_{opts['tag']}.npz", pred=pred, idx=idx, fold=fold)
        if opts["run_folds"] > 1:
            try:
                pooled = f" pooled {float(metric(pred, take(train, idx))):.4f}"
            except Exception:
                pooled = ""
    print(f"RESULT {opts['tag']}: folds {[round(s, 4) for s in scores]} mean {np.mean(scores):.4f}{pooled} "
          f"({(time.time() - t_all) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
