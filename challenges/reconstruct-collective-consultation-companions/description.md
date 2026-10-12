# Reconstruct collective consultation companions

- Domain: Recommendation
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: text

---

Reconstruct collective consultation companions

Background

Related legislative procedures can be considered together while retaining separate procedural roles. Recovering these groups requires connecting policy meaning, procedural context and timing. An administrative link can also describe agenda placement or removal rather than substantive debate.

Overview

For each focal procedure, predict the complete set of companion procedures and their recorded roles on its first recorded joint-consultation day. The target unions all focal positions on that day. Pair order is irrelevant; a companion may appear with several distinct roles.

This is retrospective reconstruction from cached records. Earlier-dated titles may have been revised later. The task does not establish causal forecastability or historically versioned information availability. Exact agenda choices are partly unobserved; meaningful text and earlier procedure context provide evidence without uniquely determining every decision.

The catalog is shared and includes all 89,798 observed procedures, rather than an answer-dependent shortlist. A candidate is eligible when its introduced\_day is strictly less than cutoff\_day and it is not focal\_id. Only earlier focal observations are supplied. Complete linked families, identical-title families, named-law revisions and conservative near-title families are separated between 2,360 training queries and 669 test queries. Named-law revision groups match the same substantive instrument after normalizing its ordinal edition. Procedural-template groups also combine exact complete budget titles differing only by year and safe-origin-country designation laws differing in their country lists. These groups limit memorization of repeated legislative templates; they do not assert that separate annual laws or proposals are the same native procedure. Generic legislative wording and shared institution names do not create a family.

The complete original cohort of 3,029 native queries is retained. One fixed lowest-hash family per observed target role is reserved in training; remaining complete families receive a fixed 75/25 hash assignment. This coverage-aware split combines periods and is not a chronological future-period generalization test. Training covers all 23 observed target roles; all 13 held-out target roles have native training examples. Six roles occur in only one independent family and consequently appear in training only; evaluation does not establish generalization for those roles.

Public files

```
+-----------------------+-------------------------------------------+
| File                  | Contents                                  |
+-----------------------+-------------------------------------------+
| candidate_catalog.csv | 89,798 candidate records                  |
| train.csv             | 2,360 input records with native targets   |
| test.csv              | 669 input records without targets         |
| train_groups.csv      | 2,360 rows; 583 validation families       |
| role_vocabulary.json  | List of 151 permitted native role strings |
| sample_submission.csv | 669 empty-set example predictions         |
+-----------------------+-------------------------------------------+
```

Input schema

candidate\_catalog.csv:

```
+----------------+---------+-----------------------------------------+
| Column         | Type    | Meaning                                 |
+----------------+---------+-----------------------------------------+
| candidate_id   | string  | Keyed opaque c_ plus 16 hex digits      |
| title          | string  | Earliest-dated cached procedure title   |
| procedure_type | string  | Native administrative procedure type    |
| initial_stage  | string  | Stage of the earliest observed position |
| introduced_day | integer | Earliest observed day; common origin    |
+----------------+---------+-----------------------------------------+
```

Where earliest dates tie, public title, stage and type determine canonical selection. introduced\_day denotes the earliest observed cached position, not an independently verified legal creation or publication date.

train.csv and test.csv:

```
+-------------------------+---------+-------------------------------+
| Column                  | Type    | Meaning                       |
+-------------------------+---------+-------------------------------+
| query_id                | string  | Keyed q_ plus 16 hex digits   |
| focal_id                | string  | Focal catalog candidate_id    |
| cutoff_day              | integer | Exact consultation-day offset |
| history_json            | JSON    | Earlier focal observations    |
| consultation_pairs_json | JSON    | Native target; training only  |
+-------------------------+---------+-------------------------------+
```

Join focal\_id to candidate\_catalog.csv candidate\_id for the focal policy title. Policy text is stored once in the catalog. History observations contain structured earlier procedural evidence; they do not repeat the catalog title. Each history\_json is a list with these fields:

```
+----------------+-------------+---------------------------------------+
| Field          | Type        | Meaning                               |
+----------------+-------------+---------------------------------------+
| relative_day   | number      | Approximate day offset; strictly < 0  |
| stage          | string      | Native earlier procedural stage       |
| procedure_type | string      | Native procedure type                 |
| authors        | string list | Supplied institutional authors        |
+----------------+-------------+---------------------------------------+
```

An empty authors list means no authors were supplied. History list order follows native day/title/stage/type order. Timing disturbance does not imply within-day chronology. Deterministic label-independent Gaussian noise has standard deviation 0.2 day and is clipped to +/-0.45 day. An additional sinusoid has amplitude 0.08 day and period 29 days with an observation-specific phase. Total timing disturbance is bounded by 0.53 day and rounded to six decimals. Cutoffs, candidate introduction days and all targets remain exact. Titles retain policy words, years and legal article references; explicit lookup citations are masked and whitespace/dash typography is normalized.

train\_groups.csv:

```
+----------+--------+------------------------------------------------+
| Column   | Type   | Meaning                                        |
+----------+--------+------------------------------------------------+
| query_id | string | Training query identifier                      |
| group_id | string | Opaque family; keep whole groups in validation |
+----------+--------+------------------------------------------------+
```

Role strings preserve their native language and spelling. They describe the companion's recorded role, not the focal procedure's role. The vocabulary JSON is a list of strings, not a mapping or target table.

Evaluation metric and formulas

For a query, let P be the predicted set and G the native set of (candidate\_id, role) pairs. Define TP = |P intersect G|.

```
precision = TP / |P|
recall    = TP / |G|
row_score = sqrt(precision * recall)
          = TP / sqrt(|P| * |G|)
score     = mean(row_score over all supplied evaluation queries)
```

One empty set against a nonempty set scores 0; empty against empty scores 1. Actual native targets are nonempty. A perfect prediction scores exactly 1. Extra pairs reduce precision, missing pairs reduce recall, and an incorrect role does not match a correct companion. Unknown, self or ineligible aliases cannot match gold and count as false positives; the grader does not read the public catalog separately to reject them.

Malformed individual JSON predictions score 0 for their row. Duplicate pairs, invalid roles, invalid aliases or incorrect pair types are malformed. Wrong, missing, extra or reordered participant columns; duplicate or mismatched query IDs; malformed physical CSV; and empty or oversized tables cause an entire-submission score of 0. Participant tables have at most 20,000 rows and 8,000,000 serialized UTF-8 CSV bytes. Each payload has at most 131,072 characters and 1,024 pairs. A pair is exactly a two-string JSON list.

Sample submission format

Use exactly these columns, in this order, with every test ID exactly once:

```
+-------------------------+--------+-----------------------------------+
| Column                  | Type   | Meaning                           |
+-------------------------+--------+-----------------------------------+
| query_id                | string | Exact test query identifier       |
| consultation_pairs_json | string | JSON list of candidate-role pairs |
+-------------------------+--------+-----------------------------------+
```

Example payload:

```
[["c_0123456789abcdef","Beratung"]]
```

This alias is illustrative. Use actual catalog aliases and vocabulary strings. Empty arrays in sample\_submission.csv are format placeholders, not labels. Private answers have the same two columns. The evaluation system may attach visibility metadata to trusted answers; participants must not add that column to predictions.

Execution and compute

Selected compute: 10 CPU cores 62 GB system RAM, and a runtime budget of 90 minutes. Generic pretrained weights and fine-tuning are permitted. Any training and prediction must run in the submitted end-to-end script. No measured end-to-end runtime or score guarantee is implied. Use train\_groups.csv to hold out complete families when validating.

What not to use

Use the provided public files and permitted generic pretrained weights. Do not use raw snapshots, organizer files, private answers, source websites, mirrors, extra annotations, external task datasets or task-specific checkpoints. Do not invert opaque aliases, look up titles in external source records, hard-code answers, or train on test targets or test predictions. Do not fit calibration or learned transformations on test statistics, or perform test-time adaptation. Source access must be isolated during blind evaluation because externally available original records can reveal labels.
