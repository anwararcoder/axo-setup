# Underwater Clip Prioritization Across Recording Locations

- Domain: Computer Vision
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: image, video

---

Overview

Predict which of two underwater clips deserves earlier human review, and whether the difference in human anomaly votes is zero, small, medium, or large. Each case contains eight clips; submit these two predictions for all 28 clip pairs.

Fixed underwater cameras record fish, crabs, moving organisms, and changing visibility against harbour and seafloor backgrounds at approximately 11 and 9 metres depth. Sixteen people annotated each frame for visually anomalous activity. A clip's review score is the mean of its eight frame-level vote proportions, not an ecological diagnosis or an animal identity. A review queue needs both an ordering and a measure of separation: a barely different pair should not be treated like one with strongly different evidence.

Training uses one acquisition location; evaluation uses the other location, with no recording shared between them. The aim is to learn visual evidence of human review priority that transfers beyond one camera background. Background subtraction alone can miss stationary activity, and a ranking alone does not specify vote separation. The finite source collection also limits how broadly these results can be interpreted.

The competition environment provides access to a single NVIDIA A10G GPU. The entire pipeline must finish within 1.5 hours, including data loading, training or adaptation, inference, decoding, validation, and submission generation.

Dataset

There are 1,182 training cases from 10 recordings and 268 test cases from 15 different recordings. Each recording supplies nonoverlapping eight-frame windows. Every window occurs in exactly eight cases from that recording, with shuffled candidate positions. Cases sharing a recording are correlated and are not independent events. The 25 recordings contain one annotated event each; the row count must not be interpreted as 1,450 independently collected events.

The supplied files are:

```
File | Contents
train.csv | id, 64 frame-path inputs, preferences, vote_gap_bands; 1,182 rows
test.csv | id and the same 64 frame-path inputs; 268 rows
sample_submission.csv | id, preferences, vote_gap_bands; 268 all-tie example predictions
frames/ | 11,600 RGB JPEG frames, each 640 by 360 pixels, grouped into opaque clip directories
```

All input columns and target columns are defined below. Paths are relative to the directory containing train.csv and test.csv.

```
Column | Type | Meaning
id | string | case_ followed by 20 lowercase hexadecimal characters; join key only
c{a}_f{t}_path | string | JPEG path for candidate a in 0..7 and temporal frame t in 0..7; all 64 combinations occur
preferences | string | p: followed by 28 L/T/R symbols in the pair order below
vote_gap_bands | string | g: followed by 28 digits in 0..3 in the same pair order
```

Each candidate is an eight-frame temporal clip, not eight unrelated still images. Frames appear as full underwater views with variable illumination and visibility; small foreground organisms can be subtle. Public frame names do not encode votes. The source has 16 votes per frame; clip means are therefore multiples of 1/128. Unannotated video tails are excluded rather than treated as normal footage. No synthetic anomaly labels or label-selected test cases are introduced.

The pair order is lexicographic: (0,1) through (0,7), then (1,2) through (1,7), continuing through (6,7):

```
(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
(2,3),(2,4),(2,5),(2,6),(2,7),
(3,4),(3,5),(3,6),(3,7),
(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)
```

For pair (a,b), let m\_a and m\_b be the mean anomaly vote proportions. L means m\_a is greater; R means m\_b is greater; T means exact equality. Let d be the absolute difference between these means. Gap digit 0 means d = 0; 1 means 0 < d <= 0.125; 2 means 0.125 < d <= 0.5; 3 means 0.5 < d <= 1. A tie must use gap 0, and a strict preference must use gap 1, 2, or 3. These are discrete predictions; no arbitrary floating-point output is required.

The thresholds distinguish a small separation of at most two of sixteen votes on average, an intermediate separation, and a separation greater than half of all votes. They are fixed across both locations. Human disagreement is retained rather than replaced by a unanimous binary label.

The comparison distributions are:

```
Outcome | Train comparisons | Test comparisons
L | 11,521 | 3,432
T | 9,827 | 948
R | 11,748 | 3,124
Gap 0 | 9,827 | 948
Gap 1 | 6,830 | 1,762
Gap 2 | 6,494 | 1,585
Gap 3 | 9,945 | 3,209
```

A real labeled target record is shown below. In train.csv, the 64 path columns occur between id and the targets; this excerpt omits them for readability. A path has the form frames/<24 lowercase hexadecimal characters>/f0.jpg, with f0 through f7 in temporal order.

```
id,preferences,vote_gap_bands
case_a0c41c9c1361fce0efe1,p:LLRLLLLRRLRLLRLRLLLLLLRTTLLT,g:3313233132222322223233300330
```

Submission Format

Write the final CSV to ./working/submission.csv. Its columns must be exactly id,preferences,vote\_gap\_bands, in that order. All three are strings. Submit exactly one row for every test ID; row order may differ from test.csv. Each target string is exactly 30 characters including its two-character prefix.

```
id,preferences,vote_gap_bands
case_a0c41c9c1361fce0efe1,p:LLRLLLLRRLRLLRLRLLLLLLRTTLLT,g:3313233132222322223233300330
```

This grammar example uses a training ID; use only test IDs in the actual submission. Duplicate columns, extra or reordered columns, duplicate/missing/extra IDs, malformed values, wrong lengths, invalid symbols, and a tie paired with a nonzero gap (or the reverse) are rejected. The evaluator does not silently remove duplicates or repair invalid strings. Grammatically valid but nontransitive preference predictions are accepted with the coherence reduction specified below.

Evaluation

The Cross-Location Review Separation Score ranges from 0 to 1; higher is better. It combines relation macro F1 (55%), gap-band macro F1 (35%), and joint pair accuracy (10%), multiplied by preference coherence. Relation receives the largest weight because queue order is primary. Gap-band F1 measures how strongly the evidence separates the clips; its equal class treatment prevents large gaps or ties from dominating. The small joint term checks that the two predictions are correct together without replacing their diagnostic partial credit.

For a symbol c, let TP\_c count comparisons correctly assigned c, P\_c count predictions of c, and Y\_c count reference comparisons labeled c. Class F1\_c = 2 times TP\_c divided by (P\_c + Y\_c). RelationScore is the mean of F1\_c over L, T, R; GapScore is its counterpart over 0, 1, 2, 3. Counts pool all 28 comparisons across all evaluation cases. If a subset contains a class in neither predictions nor references, omit that class from that subset's macro average; a predicted class absent from references receives zero. All documented classes occur in the complete evaluation split.

JointScore is the number of comparisons for which both relation and gap are exactly correct divided by 28 times N, where N is the number of evaluation cases. Coherence is 1 minus V divided by 56 times N. V counts inconsistent candidate triples across cases. A triple is inconsistent when its three pair relations cannot be generated by any three numerical scores, allowing ties; this includes contradictory strict order and equality transitivity. All 56 triples of eight candidates are checked per case.

The final score is (0.55 times RelationScore + 0.35 times GapScore + 0.10 times JointScore) times Coherence. Equivalently:

```
F1_c = 2 * TP_c / (P_c + Y_c)
RelationScore = mean(F1_c for active classes c in L,T,R)
GapScore = mean(F1_c for active classes c in 0,1,2,3)
JointScore = jointly_correct_comparisons / (28 * N)
Coherence = 1 - V / (56 * N)
Score = (0.55 * RelationScore + 0.35 * GapScore + 0.10 * JointScore) * Coherence
```

The factor never adds credit: coherent preferences have factor 1. Exact valid targets score 1. Ties and narrow separations are genuine annotated outcomes, not evidence of invalid or missing data. The small collection and subjective votes mean that perfect prediction is not assumed possible from pixels alone.

What Not To Use

Learn from supplied training labels and visible frame evidence. Do not obtain evaluation targets through source-annotation lookup, external footage matching, ID hashes, path tokens, file sizes, row ordering, or repeated-path frequency. Do not propagate labels from duplicate detection or exploit evaluator behavior. Test-to-train adaptation is not allowed: do not fit or update models, background templates, clustering thresholds, normalization statistics, or calibration using the unlabeled test collection. Training-only validation and fixed inference on each individual test case are allowed, including comparing its eight candidates. Do not pool held-out cases to infer a recording-specific background or tune thresholds from test predictions. Do not tune on repeated score feedback or use hidden-answer information.
