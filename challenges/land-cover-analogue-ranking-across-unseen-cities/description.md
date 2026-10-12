# Land-Cover Analogue Ranking Across Unseen Cities

- Domain: Computer Vision
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

Overview

Rank five satellite patches, A through E, from most to least similar to a query patch Q in **land-cover arrangement**. Each case supplies six aligned 12-band satellite patches and an RGB preview board. The goal is to retrieve places with a similar spatial pattern of vegetation, developed area, water, and other land cover, even when their color or local imaging conditions differ.

Environmental analysts often look for a comparable landscape window before transferring a mapping method to a new municipality. Matching average greenness is not enough: two windows can contain similar amounts of vegetation and buildings arranged in different places. This benchmark ranks candidates using both class composition and their pixel-level spatial agreement. Its test municipalities are absent from training, so a model must transfer its multispectral representation instead of memorizing a city's imagery. Training rankings are derived from land-cover masks associated with the patches; those masks are not included in prepared inputs.

The competition environment provides access to a single NVIDIA A10G GPU.

The entire pipeline must finish within 1.5 hours, including data loading, training or adaptation, inference, decoding, validation, and submission generation.

Dataset

There are 1,100 training ranking cases from ten municipalities and 400 test cases from five different municipalities. Each case has one query and five candidates from the same local region. A patch is 64 by 64 pixels and has twelve spectral channels. The preview is a 420 by 300 JPEG board with Q, A, B, C, D, and E in reading order; it is only a visual rendering of the packet, not a substitute for the twelve channels.

```
File                         Contents
train.csv                    Input columns plus analogue_order
test.csv                     Input columns only
sample_submission.csv        Valid-format constant-rank example, not reference labels
packets/<case_id>.npz        Six-patch spectral packet referenced by packet_path
previews/<case_id>.jpg       Six-panel RGB board referenced by preview_path
```

The training CSV columns, in order, are `case_id`, `packet_path`, `preview_path`, and `analogue_order`. The test CSV has exactly the first three columns. Paths are relative to the public data directory. No municipality, date, or original source filename is a feature column.

```
Column             Type      Meaning
case_id            string    Opaque case identifier
packet_path        string    Relative path to a compressed NumPy packet
preview_path       string    Relative path to the six-panel JPEG preview
analogue_order     string    Five candidate letters in best-to-worst order
```

Each `.npz` file has one array named `cube`, with shape `(6, 12, 64, 64)` and dtype `uint16`. Axis zero is Q, A, B, C, D, E. Axis one follows the twelve-band order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12. Values were formed by multiplying source band measurements by 100,000,000, rounding, and clipping to the unsigned 16-bit range. Divide by 100,000,000 to recover their approximate measurement scale; a model may instead normalize channels from training data. The preview board is generated from the source's RGB visualization and enlarged only for inspection.

The reference ranking compares Q with each candidate's land-cover mask. Let `h(X)` be the twelve-class pixel-frequency vector of patch X and let `M(X)` be its 64 by 64 class-code mask. The candidate utility is `U(Q,X) = 0.35 * (1 - ||h(Q)-h(X)||_1 / 2) + 0.65 * mean(M(Q) = M(X))`. Higher utility means a closer analogue. The first term rewards similar class amounts; the second rewards classes appearing in the same image locations. Candidate positions A-E are randomized within each case. The utility itself and the reference masks are not submitted; only the resulting order is predicted.

A real labeled training row has `case_id=case_00642e12f89bc50d0274053c`, paths `packets/case_00642e12f89bc50d0274053c.npz` and `previews/case_00642e12f89bc50d0274053c.jpg`, and `analogue_order=B>C>D>A>E`. Thus B is the closest spatial analogue and E is the least similar among the five candidates.

Submission Format

Write the final file to `./working/submission.csv`. It must have exactly one row for each of the 400 test IDs, with exactly these two string columns in this order:

```
case_id,analogue_order
```

`analogue_order` has exactly five distinct letters A-E joined by `>` and no spaces. It is nine characters long. A format-valid example is:

```
case_id,analogue_order
case_00642e12f89bc50d0274053c,B>C>D>A>E
```

The example shows the grammar; use the actual test IDs rather than this training ID. Duplicate columns or IDs, extra columns or IDs, and missing IDs invalidate a submission. A malformed order receives zero for its case.

Evaluation

The **Landscape Analogue Ranking Score** is the arithmetic mean of per-case scores. Higher is better. Minimum score: 0.0. Maximum score: 1.0.

For a case, `Top1` equals 1 when the submitted first letter is the reference's first letter and 0 otherwise. Among five candidates there are ten unordered pairs. A pair is concordant if the submitted order puts the same candidate first as the reference order. Let `P = concordant_pairs / 10`.

`CaseScore = 0.45 * Top1 + 0.55 * P^2`.

The top candidate carries 45% because an analyst usually inspects the first retrieved analogue. The remaining 55% checks whether the entire list is useful, not just its first entry. Squaring the pairwise fraction makes a nearly correct list noticeably more valuable than a half-random list; it also limits credit from an arbitrary constant permutation. An exact order scores 1.0, a reverse order scores 0.0, and an invalid order scores 0.0. The grader checks the complete ID set and exact column schema before averaging.

What Not To Use

Do not infer rankings from case IDs, filename bytes, row order, packet size, or split artifacts. Do not recover evaluation targets from original annotation archives or source-identifier lookups. Do not tune on hidden results or the unlabeled test distribution, alter test inputs, or exploit malformed submissions. Test-time training, adaptation, and train-test pooling are prohibited; use training-only validation. Learned multispectral encoders, segmentation-assisted ranking, contrastive methods, and classical image features are legitimate approaches.
