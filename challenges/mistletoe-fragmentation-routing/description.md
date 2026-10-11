# Mistletoe Fragmentation Routing

- Domain: Computer Vision
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

Mistletoe Fragmentation Routing

Overview

You are building a visual routing model for an urban-forest inspection workflow. A UAV tile has already been selected because it contains visible Struthanthus interruptus. Predict the fragmentation band of the infestation geometry so an operator can choose a compact local treatment view or a wider distributed-inspection view.

The source release is a real two-date UAV collection with specialist-created masks. The benchmark hides the masks and removes source names from solver-facing rows; it does not ask you to reproduce the original segmentation benchmark. The training target is a deterministic geometry label derived from the real mask, while every solver input remains a real RGB tile.

The holdout is grouped by the source DJI image identifier, including all tiles from a group, to prevent adjacent-tile memorization. The metric is ordinal and asymmetric: calling a highly scattered infestation compact can leave parts of the host canopy untreated, while calling a compact patch scattered mainly spends additional inspection capacity.

Dataset

File descriptions

* `dataset/public/train.csv`: feature rows with `id` and `image_path`; labels are in the public `train_targets.csv` sidecar.
* `dataset/public/train_targets.csv`: one public target row per training id with the integer `target` column.
* `dataset/public/test.csv`: feature rows with the same `id,image_path` schema and no target values.
* `dataset/public/images/`: real RGB JPEG tile views copied from the source release and renamed to opaque ids.
* `dataset/public/sample_submission.csv`: correctly formatted non-constant example predictions for every test id.

Column descriptions

* `id`: opaque string identifier; copy it unchanged into the submission.
* `image_path`: path relative to `dataset/public/` for one real 250 by 250 RGB tile.
* `target`: integer fragmentation class: `0` compact, `1` split, or `2` scattered.
* `prediction`: one integer in `0`, `1`, or `2`.

Evaluation

The hidden target is computed from the real specialist mask. Let `largest_component_share` be the area of the largest 8-connected foreground component divided by all foreground area. The target is `0` when this share is at least `0.85`, `1` when it is at least `0.55` and below `0.85`, and `2` otherwise. This makes the target a coarse geometry band rather than a direct species label or a pixel-mask recreation.

The evaluator maximizes the following complete code. `macro_f1` protects all three geometry bands. `ordered_utility` gives partial credit to adjacent bands and assigns a larger cost to undercalling fragmentation (`prediction < truth`) than to overcalling it. The score is bounded in `[0, 1]`; missing classes, malformed predictions, and non-finite values are rejected before scoring.

```
import numpy as np

def score(y_true, y_pred):
    classes = np.array([0, 1, 2], dtype=int)
    max_undercall_cost = 1.4 * 2.0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    if len(y_true) == 0 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must be finite and non-empty")
    if not np.equal(y_true, np.floor(y_true)).all() or not np.equal(y_pred, np.floor(y_pred)).all():
        raise ValueError("Targets and predictions must be integer class ids")
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    if not np.isin(y_true, classes).all() or not np.isin(y_pred, classes).all():
        raise ValueError("Targets and predictions must be 0, 1, or 2")
    if set(y_true.tolist()) != set(classes.tolist()):
        raise ValueError("Targets must contain all three fragmentation classes")

    f1_values = []
    for label in classes:
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    macro_f1 = float(np.mean(f1_values))

    distance = np.abs(y_pred - y_true).astype(float)
    cost = distance * 1.4
    overcall = y_pred >= y_true
    cost[overcall] = distance[overcall] * 0.8
    ordered_utility = float(np.mean(1.0 - cost / max_undercall_cost))
    value = 0.65 * macro_f1 + 0.35 * ordered_utility
    if not 0.0 <= value <= 1.0:
        raise ValueError("Score is outside [0, 1]")
    return float(value)
```

Submission

Submit one CSV with this exact column order:

* `id`
* `prediction`

Example:

```
id,prediction
tile_767bb7852983fe49,1
tile_d06d624056f5f1ee,0
```

Requirements

* Include exactly one row for every id in `dataset/public/test.csv`.
* Keep every prediction as a finite integer from `0` through `2`.
* Preserve ids exactly; do not replace them with image filenames, source DJI names, or row numbers.
* Use only the public images and public training-target sidecar; no external source lookup or hidden masks are permitted.

What Not To Use

* Do not use `segmentation_masks.zip`, author manifests, raw source filenames, GPS fields, acquisition folders, or source-group identifiers to recover hidden targets.
* Do not reverse-map opaque ids to the original Figshare tile names or retrieve the original archive, mask table, or cached target geometry.
* Do not add generated, synthetic, pseudo-labeled, or externally collected images or labels; the benchmark contains only the released real RGB tiles and real source-derived training targets.
* Do not split adjacent tiles from one source DJI image across an internal validation fold when claiming a generalization result.
* Do not optimize or report plain accuracy alone; the official score includes per-class F1 and the asymmetric cost of undercalling scattered infestation.
