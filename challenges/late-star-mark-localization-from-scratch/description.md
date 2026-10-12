# Late Star-Mark Localization from Scratch

- Domain: From Scratch
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

Overview

This computer-vision challenge asks you to score each eligible cell in a galaxy image for whether it will receive repeated human star marks in the next ten classifications, using the full image and the first five classifications. The output is a 14×14 spatial score map reconstructed from one probability per eligible image-cell row; it can highlight regions for human review but does not verify astronomical objects.

Dataset

The prepared public data contains 4,000 training images with 771,902 eligible cell rows and 1,999 evaluation images with 385,913 eligible cell rows. Images are 525×525 RGB PNGs. The task grid has 14 rows and 14 columns, stored in row-major order.

* `train.csv` has `id` (string), `image_id` (string), `cell_row` (integer), and `cell_col` (integer) columns; each row identifies one training image-cell pair.
* `train_labels.csv` has `id` (string) and `p_late_star` (binary integer) columns, with one label for each `train.csv` ID; 2,976 training labels are positive.
* `test.csv` has the same ordered columns and types as `train.csv`, with one row for each eligible evaluation image-cell pair.
* `image_context.csv` has `image_id` (string) and `image_path` (relative path string) columns; join it by `image_id` to locate each image under `images/`.
* `panels.jsonl` has one object per image, with an `image_id` and a chronological `classifications` list containing five visible entries; each entry has `centers` and `stars` arrays of native `[x,y]` image-pixel coordinates, and an empty array means no marks of that type.
* `images/` contains the RGB images referenced by `image_context.csv`.
* `sample_submission.csv` contains the ordered answer columns and example probability values.

**Eligibility and target:** A cell is eligible when none of the first five visible classifications marks a star there. Its label is 1 when at least two distinct classifications among the next ten mark a star in that cell, and 0 otherwise; one classification contributes at most one vote to a cell even if it contains repeated points there. Classifications after this fixed 15-entry window do not affect the label. Coordinates use zero-based pixels with x as column and y as row; an in-frame point maps to `cell_col=floor(14*x/525)` and `cell_row=floor(14*y/525)` for `0<=x<525` and `0<=y<525`. The image rectangle is half-open, so out-of-frame points remain visible in the panel input but map to no cell.

**Independent units:** All cells and visible panels belonging to one image stay in one partition. Images with the same catalog identifier or exact RGB pixels stay together, so the training and evaluation image groups are disjoint. The 4,000 training images provide 771,902 labeled cell observations; these correlated cells are not independent images. The 1,999 evaluation images provide 385,913 scored rows, and evaluation is limited to new image groups from the same collection.

Evaluation

The score is pooled average precision over all eligible evaluation image-cell rows, with every row receiving equal weight. Let P be the number of positive rows. For each distinct submitted score threshold t, let `precision_t` be the fraction of rows with scores at least t that are positive, and let `recall_t` be the fraction of all P positive rows with scores at least t; with `recall_prev` set to zero before the first threshold, average precision is `AP = Σ_t (recall_t - recall_prev) × precision_t`. Tied scores are grouped at one threshold without interpolation. The score is higher-is-better on [0,1], a perfect ranking scores 1, and a subset with no positive targets scores 0. Predictions must be finite probabilities in [0,1]; the metric does not clip them.

The binary target records repeated marks in the released sequence, not verified astronomical truth or agreement among independent experts. Binary cross-entropy supplies a per-cell training signal for these binary labels, while average precision evaluates their ranking rather than probability calibration. The metric pools cell rows, but cells within an image are correlated; estimate validation uncertainty by resampling whole images. This split supports generalization to held-out images within the same collection, not transfer across surveys, populations, sensors, or acquisition programs.

Submission

Submit a UTF-8 CSV with exactly the ordered columns `id,p_late_star`, one row for every `test.csv` ID exactly once; row order may vary. Copy the 12-character IDs exactly from `test.csv`. Each `p_late_star` value must be a finite number in [0,1]. The `cell_row` and `cell_col` rows reconstruct each image's partial 14×14 probability map over eligible cells.

```
id,p_late_star  
cA1b2C3d4E5F,0.370000  
cG6h7I8j9K0L,0.080000
```

Rules

Train a neural model from scratch on the supplied training data during the current run, using available libraries; do not install packages. Do not use pretrained models, weights or embeddings, external data or answer lookup, synthetic training data, or assets carried across runs.
