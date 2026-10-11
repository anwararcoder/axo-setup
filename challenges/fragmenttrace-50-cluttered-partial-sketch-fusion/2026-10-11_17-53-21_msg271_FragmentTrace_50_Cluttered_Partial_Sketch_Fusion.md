# FragmentTrace-50: Cluttered Partial Sketch Fusion

- Domain: Computer Vision
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: computer-vision, multi-view, fine-grained-classification, clutter-robustness, partial-observation, from-scratch

---

FragmentTrace-50: Cluttered Partial Sketch Fusion

Overview

**Domain: Computer Vision.** Each example contains two independently posed 28×28 monochrome composites. Both contain complementary fragments of the same target sketch, and each also contains a different target-independent clutter fragment. Predict the target sketch's opaque semantic class.

The benchmark tests whether a model can fuse incomplete evidence across views while suppressing plausible sketch clutter. Neither view contains the full target, and nuisance poses are unrelated to the label.

Objective

For every row in `test.csv`, predict `prediction` as one integer in `[0,49]`. Labels are an opaque, deterministic mapping of 50 target sketch classes. Independent right-angle rotations, reflections, translations, fragmentation masks, and clutter poses are nuisance variables. One row is one independent three-drawing bundle: one target drawing and two unique clutter drawings. Ties do not occur.

Why the task is difficult

Each target view receives a disjoint 30% partition of the target ink pixels; the remaining 40% is omitted. Each composite also receives 50% of one unrelated clutter drawing. Target and clutter fragments receive independent right-angle poses, reflections, and translations. The two target fragments do not overlap and the clutter differs, so useful evidence is distributed across both views.

Construction controls include:

* all 50 target classes have exactly 200 training and 50 test examples;
* a separate organizer-only reservoir selects a visually dense source vocabulary and never enters train or test;
* exact source duplicates and ambiguous cross-category duplicates are excluded before selection;
* the split is assigned before any public transformation;
* clutter drawings are target-independent and are never reused across rows or splits;
* class mapping, identifiers, row order, fragment layout, and nuisance transforms use independent randomness namespaces;
* paired transformed views are globally unique and contain a minimum amount of visible ink.

Research distinction

Unlike ordinary single-image classification, each view is incomplete, independently cluttered, and posed, and the two target fragments contain different ink pixels. Successful systems must learn local topology, suppress view-specific clutter, and combine complementary evidence. Single-view ablations and no-training probes are kept reviewer-side to measure whether a method is exploiting marginal artifacts instead of multi-view content.

Dataset

The prepared public package contains:

* `train.csv`: 10,000 labelled independent bundles;
* `test.csv`: 2,500 unlabelled independent bundles;
* `train_views.npz`: aligned `view_a` and `view_b` uint8 arrays with shape `(10000,28,28)`;
* `test_views.npz`: aligned `view_a` and `view_b` uint8 arrays with shape `(2500,28,28)`;
* `sample_submission.csv`: a schema-valid constant baseline.

One scored row is one independent target-plus-clutter bundle and produces exactly one answer row. No target or clutter drawing is reused across bundles or splits. Array rows follow the corresponding CSV row order. `image_index` is a package-global locator: training values are `0`–`9999` and test values are `10000`–`12499`, so the locator is never reused across splits.

Columns

`train.csv` contains:

* `id`: opaque bundle identifier;
* `image_index`: package-global bundle locator; array rows follow CSV row order;
* `target`: opaque semantic class in `[0,49]`.

`test.csv` contains the same solver-visible feature columns except `target`. Its rows align in order with `test_views.npz`.

Modelling guidance

A useful conventional baseline extracts HOG or pooled-gradient features from both views and trains a regularized multiclass classifier. Stronger systems can use a shared convolutional encoder with symmetric cross-view fusion and train the 50-way objective on an A10G.

Use a fixed seed, epoch count, augmentation policy, and checkpoint rule. The target is invariant to the supplied right-angle pose nuisances, so matching train-time augmentation is appropriate.

Evaluation

The metric is top-1 multiclass accuracy:

`score = correctly classified test bundles / all evaluated test bundles`

The score is bounded in `[0,1]` and is maximized. A perfect submission scores `1.0`; uniform random guessing has expected accuracy `1 / 50 = 0.02`. The grader aligns by `id`, so submission row order does not affect the score.

Submission

Write `submission.csv` with exactly these columns in this order:

```
id,prediction
ft_opaque_identifier,0
```

Requirements:

* one row for every test ID;
* no duplicate, missing, or unknown IDs;
* `prediction` must be a finite base-10 integer in `[0,49]`;
* no extra or reordered columns.

Structural violations raise a clear validation error.

What not to use

* Do not use external datasets, APIs, web search, hosted inference, or hidden/private files.
* Do not identify or reconstruct upstream drawings, category names, source positions, or organizer-only mappings.
* Do not exploit IDs, row order, archive bytes, filenames, compression details, or leaderboard probing.
* Generate predictions automatically from the supplied labelled training examples and public view arrays.

Intended use and limitations

This benchmark studies fine-grained partial-shape recognition, clutter suppression, and cross-view fusion for small monochrome sketches. It does not measure natural-image perception, artistic quality, human intent, or safety-critical perception. The drawing styles reflect a crowdsourced drawing game and are not representative of the visual world.
