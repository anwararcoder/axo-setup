# Availability-Aware Model Candidate Recommendation

- Domain: Recommendation
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

Overview

Model routers need to choose a small set of candidates that are likely to matter in the next observed usage window. A candidate with high current share may be fading, while a lower-share candidate with improving momentum may be both useful and more likely to remain available. This challenge asks you to rank model candidates from only the information available at the current snapshot.

This is a recommendation-ranking task, not a time-series forecasting task: the submission is a within-snapshot priority order for candidates that a router should keep available for the next decision. It does not ask for an absolute future usage forecast or a reconstructed future series. The hidden evaluation uses the next observed snapshot only as delayed operational feedback for judging whether today's recommendation list retained useful candidates.

The public split contains 45 training snapshots (indices 0–44), followed by one withheld bridge snapshot (index 45), and 12 test snapshots (indices 46–57). The bridge snapshot is used only to construct the next-window label for the final training snapshot and is not included in either public file, so no training target is computed from a public test snapshot. Candidate codes may recur across the training and test periods because availability persistence is part of the recommendation problem; row IDs and target rows remain disjoint, and each row's features use only observations at or before that row's own snapshot.

Dataset

File descriptions

* `train.csv` — feature-only recommendation snapshots from 45 chronological groups (indices 0–44), with one row per candidate and an opaque `id`.
* `train_targets.csv` — public training targets with columns `id,target`; each target is the observed next-window growth gain for the matching training row.
* `test.csv` — feature-only recommendation candidates from 12 later chronological groups (indices 46–57); it has the same feature columns as `train.csv` and contains no target column.
* `sample_submission.csv` — correctly shaped example submission with columns `id,prediction`.

Column descriptions

* `id` — opaque 12-character row identifier; use it to join `train_targets.csv` and to form the submission.
* `snapshot_index` — chronological snapshot group identifier; rows with the same value are ranked together.
* `candidate_code` — opaque 12-character candidate identifier; it is stable for a candidate within this dataset but does not reveal its source name.
* `rank_pct` — current within-snapshot rank normalized to the interval [0,1], where 0 is the best observed rank.
* `share_log` — natural-log transform of the current observed usage share.
* `requests_log` — natural-log transform of the current observed request count.
* `tokens_log` — natural-log transform of the current observed token count.
* `prompt_fraction` — current prompt-token fraction of prompt plus completion tokens.
* `completion_fraction` — current completion-token fraction of prompt plus completion tokens.
* `group_size` — number of candidates in the current snapshot.
* `share_entropy` — entropy of the current snapshot's normalized share distribution.
* `candidate_seen_count` — number of earlier snapshots in which the candidate was observed.
* `candidate_active_rate` — earlier-snapshot presence rate for the candidate.
* `candidate_age` — number of snapshot steps since the candidate first appeared.
* `days_since_seen` — snapshot-step gap since the previous observation of the candidate.
* `previous_presence` — whether the candidate was present in the immediately preceding snapshot.
* `prev_rank_pct` — candidate's previous observed normalized rank, with a neutral cold-start value.
* `rank_delta_1` — current normalized rank minus the candidate's previous normalized rank; for a cold-start candidate with no history this is set to 0.
* `rank_delta_3` — current normalized rank minus the candidate's recent normalized-rank mean; for a cold-start candidate with no history this is set to 0.
* `share_delta_1` — current log share minus the candidate's previous log share.
* `share_delta_3` — current log share minus the candidate's recent log-share mean.
* `rank_mean_3` — recent mean normalized rank for the candidate.
* `share_mean_3` — recent mean log share for the candidate.
* `share_std_3` — recent standard deviation of the candidate's log share.

Evaluation

The submission value is interpreted only as a ranking score. For each `snapshot_index`, rows are sorted by descending `prediction`, with ascending `id` as the deterministic tie-breaker. The hidden future target contains a non-negative growth gain and a binary stability indicator for each row. The growth gain is `0.8 * min(max(log1p(next_share) - log1p(current_share), 0), 2.0) + 0.2 * stable`, where `stable` is one when the candidate appears in the next snapshot and its share is at least 90% of the current share.

The score is computed as follows:

```
import numpy as np

def stability_weighted_ndcg_at_10(gain, stable, prediction, row_id, snapshot):
    snapshot_scores = []
    snapshot_persistence = []
    for group in sorted(set(snapshot)):
        positions = [i for i, value in enumerate(snapshot) if value == group]
        order = sorted(positions, key=lambda i: (-prediction[i], row_id[i]))[:10]
        discounts = 1.0 / np.log2(np.arange(2, len(order) + 2, dtype=float))
        dcg = sum(gain[i] * discounts[j] for j, i in enumerate(order))
        ideal_gain = sorted((gain[i] for i in positions), reverse=True)[:10]
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_gain) + 2, dtype=float))
        idcg = sum(value * ideal_discounts[j] for j, value in enumerate(ideal_gain))
        snapshot_scores.append(dcg / idcg if idcg > 0 else 0.0)
        snapshot_persistence.append(np.mean([stable[i] for i in order]) if order else 0.0)
    return 0.8 * float(np.mean(snapshot_scores)) + 0.2 * float(np.mean(snapshot_persistence))
```

The final score is bounded to [0,1] and higher is better. `train_targets.csv` exposes the complete composite gain for training rows, including its small stability component; the same hidden stability indicator is also scored as a separate top-10 operational term. This intentionally rewards both row-level future opportunity and group-level continuity. The withheld bridge snapshot prevents the final training label from reading any public test row. The test rows are later recommendation snapshots, and `snapshot_index` is the only grouping key needed to reproduce the per-snapshot ranking structure.

Submission

Format

* `id` — one exact test-row identifier from `test.csv`.
* `prediction` — a finite numeric ranking score in the inclusive range [0,1].

Example:

```
id,prediction
85da4f4eff29,0.81250000
3ba55189286a,0.43750000
```

Requirements

* Include every test `id` exactly once and no other IDs.
* Keep the two columns in the order `id,prediction`.
* Provide finite numeric predictions in [0,1].
* Produce the file at `submission.csv` when submitting through the platform.

Allowed methods

Use any reproducible ranking or regression method that trains from the provided public files. Chronological validation, feature engineering, group-aware ranking, and models that handle opaque candidate codes are allowed.

What Not To Use

* Do not reverse-map `candidate_code` or `id` to recover the source model identity or future observations.
* Do not use measurements from a later capture window to construct features for an earlier snapshot.
* Do not query a live routing or usage feed to recover the hidden next-snapshot outcome.
* Do not replace learning with a fixed list of candidates, a hand-coded future lookup, or a submission copied from an external ranking.
* Do not create synthetic observations or synthetic labels to enlarge the public training data.
