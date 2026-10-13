# AfterPhase-SEQ: Censored Cascade Regime Forecasting

- Domain: From Scratch
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: text

---

AfterPhase-SEQ: Censored Cascade Regime Forecasting

Overview

A cascade is observed only during its first few hours. From that irregular event prefix, predict how the hidden continuation will change in both activity rate and spatial spread.

Each case is centered on a dominant seismic event and contains a variable-length sequence of smaller events recorded during a fixed early observation window. The continuation is withheld. The target is one of six joint regimes formed by three rate-transition states and two spatial-transition states.

This is a from-scratch structured-sequence learning benchmark, not next-event regression or static-row prediction. A correct solution must learn from the released training rows how an aggregate hidden continuation relates to a censored, irregular point sequence.

Data

The public package contains only:

* `train.csv`: 751 labeled sequences;
* `test.csv`: 400 unlabeled sequences;
* `sample_submission.csv`: a valid submission template.

Every row is one independent cascade episode. Episode construction prevents temporally and spatially overlapping footprints from being reused across rows.

Columns

`id` is an opaque case identifier.

`mainshock` is a JSON array with three values:

1. main-event magnitude;
2. main-event depth in kilometres;
3. total number of observed prefix events before deterministic sequence subsampling.

`sequence` is a JSON array ordered by observation time. Every event is represented by five values:

1. elapsed hours after the main event;
2. local east offset in kilometres;
3. local north offset in kilometres;
4. depth difference from the main event in kilometres;
5. magnitude difference from the main event.

At most 48 observations are released. Longer prefixes are sampled at fixed, evenly spaced ranks so that both early and late portions of the observation window remain represented. There is no padding inside the CSV value.

`target` appears only in `train.csv`. It is an integer from 0 through 5.

Target semantics

The hidden continuation begins when the public observation window ends. Define the rate statistic as:

```
rate_log_ratio = log((future events/hour + 0.05) / (prefix events/hour + 0.05))
```

Training-set tertiles divide this statistic into three relative groups:

* `0`: low, where `rate_log_ratio < -1.9044008540`;
* `1`: middle, where `-1.9044008540 <= rate_log_ratio < -1.3793579171`;
* `2`: high, where `rate_log_ratio >= -1.3793579171`.

These labels denote only low, middle, and high future-to-prefix rate-ratio quantile groups; they are not absolute physical-state labels.

Define the spread statistic as:

```
spread_log_ratio = log((future median radius km + 2) / (prefix median radius km + 2))
```

The training-set median divides this statistic into two relative groups:

* `0`: lower spread ratio, where `spread_log_ratio < 0.1715133409`;
* `1`: higher spread ratio, where `spread_log_ratio >= 0.1715133409`.

These labels denote relative spread-ratio groups and do not imply an absolute change direction.

The released class is:

```
target = 2 * rate_state + spread_state
```

All three cut points were fitted only on training episodes. Test targets reuse the frozen values shown above.

Why the cut points are frozen

The train/test assignment is made by a salted hash of the independent episode identifier before any cut point is fitted. Both sides use the same event eligibility rules, observation and continuation windows, censoring procedure, and temporal/spatial independence filter. The cut points are then estimated once from training continuations and frozen before test targets are derived.

This makes the six regimes a training-reference definition: a held-out continuation is measured against the reference population learned from training, even if its marginal distribution differs. Re-estimating thresholds on test continuations would leak held-out target distribution information and would change the meaning of the labels at evaluation time. Organizer-side audits verify zero dependency crossings, non-empty support for every regime, and stable rankings under independent-episode perturbations; exact held-out class quotas are intentionally not disclosed.

Submission

Submit a UTF-8 CSV with exactly these columns in this order:

```
id,prediction

ap_...,3
```

Requirements:

* include every test ID exactly once;
* do not add, remove, or reorder columns;
* `prediction` must be a finite integer in `0..5`;
* row order does not affect scoring.

Evaluation

The metric is exact regime accuracy:

```
accuracy = number of correct predictions / number of evaluated cases
```

Higher is better. The theoretical range is 0 to 1. The evaluator aligns predictions by `id`.

Intended modeling approach

Useful models can combine:

* temporal count and cadence features over several prefix intervals;
* magnitude-trajectory summaries;
* radial-spread and anisotropy summaries;
* a tree ensemble over sequence statistics; or
* a masked temporal encoder over the ordered five-dimensional events.

Train every learned parameter from scratch on `train.csv`. The task is small enough for a deterministic CPU workflow. A fixed feature-extraction, training, and inference plan should be used. Training more models conditionally on elapsed time is not allowed.

Allowed and prohibited resources

Pretrained model weights are not allowed. External task data, external labels, hosted prediction APIs, web search for case identities, and files outside the public package are prohibited. All fitted parameters, feature normalizers, thresholds, and model-selection decisions must be learned from the released training rows only. A solution may use only the released inputs to identify a test case.

The public observations are real measurements, while the six labels are deterministic derived regimes. The integer output is only a compact code for a future temporal-sequence regime. These codes are not manually assigned hazard assessments and must not be interpreted as operational forecasts.
