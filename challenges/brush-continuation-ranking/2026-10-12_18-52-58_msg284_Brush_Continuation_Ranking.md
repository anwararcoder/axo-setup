# Brush Continuation Ranking

- Domain: Recommendation
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: small-data, feature-engineering

---

Brush Continuation Ranking

Task

Given an earlier wrist-motion clip and a shortlist of 16 toothbrush-motion clips,

rank the candidates by how likely each is to be the recorded future continuation.

Exactly one candidate is the brush interval that actually followed the query.

The wrist context occupies [t, t+2) seconds. The matching brush candidate occupies

[t+5, t+7) seconds. There is a three-second blind gap between the end of the context

and the beginning of the target. This is future-item ranking, not simultaneous

sensor synchronization or prediction of a numerical health outcome.

The shortlists contain only clips from the SAME participant, sampled across that

person's recordings. Participant identity therefore cannot identify the match.

Training and test participants are disjoint. Learn how a person's current wrist

movement relates to the brush motion that follows, while transferring to unseen

people and handling device-local coordinate frames.

This is an offline candidate-ranking benchmark. The candidate library is supplied

at inference time, but which candidate followed each query is hidden. The task

does not claim that a recorded movement is healthy, effective or recommended

dental treatment. It does not measure plaque removal or diagnose oral disease.

Input data

Every clip has 64 ordered samples at 32 Hz, with step values 0 through 63.

Both views contain ax, ay, az (acceleration in g, including gravity) and gx, gy, gz

(angular velocity in degrees per second). Axes are local to each device, so their

directions are not guaranteed to match between wrist and brush.

Original measurements were block-averaged and resampled. Query starts are 7.5

seconds apart within a recording, so no public wrist context overlaps any target

interval in that recording. Future wrist recordings and contemporaneous brush

recordings are not supplied. The nominal final sample of a two-second clip is at

1.96875 seconds; the interval definitions above are half-open time windows.

Public files:

* train\_queries.csv: 582 rows with query\_id, participant\_group,

  session\_group, candidate\_ids. candidate\_ids is a JSON array of 16 distinct IDs.
* test\_queries.csv: 200 rows with the same columns.
* train\_wrist.csv and test\_wrist.csv: query\_id, step, ax, ay, az, gx, gy, gz.

  Each query has exactly 64 rows. Sort by step before using its signal.
* train\_brush.csv and test\_brush.csv: candidate\_id, step, ax, ay, az, gx, gy, gz.

  Each brush candidate has exactly 64 rows. Join IDs to each query's shortlist.
* train\_candidate\_pairs.csv: 9,312 rows with row\_id, query\_id, candidate\_id.
* test\_candidate\_pairs.csv: 3,200 rows with the same columns. Each query has 16

  rows, one per supplied candidate. row\_id joins query\_id and candidate\_id using

  two underscores; this combines public identifiers and does not encode relevance.
* train\_labels.csv: row\_id, continuation\_score. The value is 1 for the observed

  future candidate and 0 for the other 15 candidates in each training query.
* sample\_submission.csv: row\_id, continuation\_score. One row per test candidate

  pair, with constant placeholder score 0.5. These are not answer labels.

There are 11 training participants and 5 test

participants, in 37 and 15 recordings respectively.

The participant and session group fields allow grouped validation. Keep whole

participants out of model fitting when building your validation set.

Candidate IDs are shared when a clip appears in multiple shortlists. Each brush

item is the true continuation for exactly one query in its partition. Solvers may

use this one-to-one structure, although the evaluator independently scores the

ranking for each query and does not require top choices to form an assignment.

IDs, row order and candidate-list order have no intended temporal meaning.

Evaluation

The score is ordinary mean reciprocal rank (MRR), averaged equally over queries.

Within each query, candidates are ranked by descending continuation\_score.

If the matching candidate is at one-based position r, that query earns 1/r.

Only within-query ordering matters; scores need not sum to one or be calibrated.

Exact score ties receive the expected reciprocal rank under uniform ordering

within that tied block. If g candidates have strictly higher scores and e have

the same score as the true candidate, the query score is:

`sum(1/r for r in range(g+1, g+e+1)) / e`

An all-equal shortlist therefore earns approximately 0.211296, not an ID-dependent

rank. With distinct scores this is ordinary MRR without any score transformation.

* Maximize the score.
* Maximum: 1.0, when all matching candidates are first.
* Minimum valid score: 1/16 = 0.0625, when all matching candidates are last.
* Expected score of uniformly random rankings: approximately 0.211296.
* Only the position of the unique recorded continuation matters. There is no

  additional reward or penalty for the order of the remaining candidates.
* Every evaluated query must receive scores for all 16 candidates; no abstention.

Leaderboard scoring uses fixed public and private query partitions. One complete

session per test participant is assigned to public scoring by a seeded hash;

the remaining sessions form private scoring. Each score is ordinary MRR within

its own partition. Submit ALL 3,200 candidate rows across the 200 test queries. The platform

selects the answer partition; solvers do not supply a visibility column.

Submission format

Write a UTF-8 CSV with these two prediction columns:

row\_id,continuation\_score

continuation\_score must be a finite numeric value in [0,1]; higher means more

likely to be the observed future continuation. Do not submit JSON arrays, candidate

names or query-level rows as predictions. Copy row\_id from test\_candidate\_pairs.csv

and use sample\_submission.csv as the format template.

Include exactly one row per required candidate pair. Row and column order do not

affect scoring. Missing or extra row IDs, duplicates, invalid identifiers, missing

or non-finite scores, scores outside [0,1], booleans and extra columns are rejected.

Only the reserved public/private visibility metadata is tolerated by the grader

for platform-generated fixtures; solvers should omit it. Internal partial scoring

accepts the full submission or the exact complete-query answer subset. No query

may be split between visibility partitions.

Rules and limitations

Use only the supplied training data and test inputs. External lookup of the

original recordings, recovery of withheld pairing/time keys, and external answer

mappings are prohibited. General-purpose software libraries are allowed.

The recordings follow a prescribed brushing sequence rather than unsupervised

daily-life behavior. The small adult cohort and repeated recordings limit broad

generalization; there are only five independent test participants. Brushing may

be partly unpredictable across the gap, so similar-looking candidates need not

be interchangeable labels. Ground truth identifies the observed continuation,

not an optimal action. Device-clock alignment can contain residual error, and

quality exclusions may introduce selection bias. No medical use is established.

Sensor channels are not normally distributed. In the full training brush file,

about 7.12% of gx samples lie beyond three interquartile ranges from the quartiles.

The observed range is approximately -1214.13 to 1079.79 degrees/second. These

source-derived values are retained, not clipped to remove a statistical warning.

Their presence does not establish that every sensor peak is artifact-free.
