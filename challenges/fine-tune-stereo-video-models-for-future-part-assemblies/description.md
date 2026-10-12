# Fine-tune stereo video models for future part assemblies

- Domain: Fine-Tuning
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: image, video

---

Fine-tune stereo video models for future part assemblies

Background

Advance annotation of an articulated subject requires the future locations of its visible parts and the body's motion phase. A useful forecast must cover the complete requested assembly, including small parts. Independent credit for one easy part does not establish a complete annotation.

Overview

Fine-tune an approved generic pretrained visual backbone on the supplied training pairs to predict complete future part assemblies from stereo observations. Fine-tuning must occur in the submitted script within the stated GPU session; a frozen pretrained model used only for inference does not meet the execution requirements.

Given four consecutive stereo observation pairs and current left-view boxes for an eligible roster of bird parts, predict four future left-view assemblies. Each future assembly contains a box for every requested part and exactly one shared native body phase. The five possible parts are body, head, tail, left\_wing and right\_wing; phases are flying, takeoff, landing and sitting. The current body phase is flying or takeoff.

All boxes and phases are native annotations, including native interpolation where present. No 3D calibration, joint angles, physical articulation constraints or independent manual keyframes are claimed. A part is eligible only when it has one valid current anchor and one valid annotation in each of the four future frames. This roster condition uses annotation availability; it does not disclose future coordinates or establish that unlisted parts are physically absent.

There are 815 non-overlapping eight-frame blocks: 512 training and 303 test blocks. They retain all 2,553 training and 1,513 test native part tubes. Training has 508 five-part, three three-part and one four-part assemblies; test has 301 five-part and two four-part assemblies. Every supplied roster must be predicted exactly.

The first four frames are inputs; the next four are targets. Target images are never public observations. Whole acquisition clips define the split: 184 training and 114 test clips. The same small set of birds and environment recurs across splits; unseen-bird, species or camera generalization is not established. Keep each supplied training group together in validation.

Images preserve aspect ratio at width 1280. Deterministic label-independent disturbances include Gaussian pixel noise with standard deviation 6, a sinusoidal component with amplitude 3, gain from 0.97 to 1.03, and additional bounded disturbances of magnitude at most 18 at about 0.2 percent of pixel sites. Native geometry and semantic targets are unchanged.

Public files

```
+-----------------------+------+------------------------------------+
| File                  | Rows | Contents                           |
+-----------------------+------+------------------------------------+
| train.csv             | 512  | Inputs and native future assemblies|
| test.csv              | 303  | Held-out assembly forecast inputs  |
| sample_submission.csv | 303  | Format placeholders                |
| train_groups.csv      | 512  | Whole-clip validation groups       |
| images/               | 6520 | Observed RGB JPEG images           |
+-----------------------+------+------------------------------------+
```

All image paths are relative to the public directory. The 6,520 files are the eight observed camera images per block. Sample predictions are placeholders, not training labels.

Input schema

```
+---------------+--------+-----------------------------------------+
| Column        | Type   | Meaning                                 |
+---------------+--------+-----------------------------------------+
| id            | string | q followed by 24 opaque hex digits      |
| anchors_json  | JSON   | Part-name to current left-view box      |
| frames_json   | JSON   | Four chronological [left,right] paths   |
| assembly_json | JSON   | Four future assemblies; training only   |
+---------------+--------+-----------------------------------------+
```

train.csv contains these four columns in order. test.csv contains the first three. train\_groups.csv contains exactly id,group; group is an opaque whole-clip identifier. sample\_submission.csv contains exactly id,assembly\_json.

anchors\_json is an object mapping each eligible part name to [x,y,width,height]. All rectangles are normalized to image width and height after clipping native boxes to the visible image. Values must be finite numbers in [0,1], with positive width and height, x+width<=1 and y+height<=1; the parser allows 1e-12 numerical boundary tolerance. Booleans are invalid. There is no rounding of native target coordinates.

assembly\_json is a four-element array in chronological future-frame order. Each element has exactly motion and boxes. motion is one of the four listed phases; boxes maps exactly the anchor roster's part names to rectangles. The roster must remain identical across all four frames. Unknown or duplicate JSON keys are invalid. A missing, extra or malformed part invalidates the row. The five-part maximum is fixed; no unannotated part is invented.

Evaluation metric

Maximize score in [0,1]. For each future frame, compute ordinary rectangle IoU for every eligible part. The geometric mean measures whole-assembly coverage; the shared body phase gates that frame's credit.

```
I[p,t] = area(predicted_box[p,t] intersect native_box[p,t])
U[p,t] = area(predicted_box[p,t]) + area(native_box[p,t]) - I[p,t]
IoU[p,t] = I[p,t] / U[p,t]
G[t] = product(p in roster, IoU[p,t]) ** (1 / roster_size)
Q[t] = G[t] * I(predicted_motion[t] == native_motion[t])
row_score = sum(t=1..4, Q[t]) / 4
score = mean(row_score over all test blocks)
```

Every block has equal weight; its parts jointly determine its frame quality. A nonoverlapping part makes that frame's geometric mean zero, while a wrong body phase gives no frame credit. Exact valid native assemblies score one. There is no arbitrary score cap, fabricated kinematic penalty or format-only reward. All native rectangles have positive area.

Malformed payloads score zero for their row. Incorrect or reordered CSV columns, extra data fields, duplicate/missing/unknown IDs, wrong row counts, invalid CSV or empty tables score zero globally. CSV files are limited to 64 MiB and 20,000 rows; JSON payloads to 16,000 characters. Nonfinite numbers, Boolean coordinates, wrong frame counts, unknown phases and changing or incorrect rosters are invalid. Submission row order is immaterial.

Sample submission format

Submit exactly id,assembly\_json, including every test ID once. For a one-part grammar illustration:

```
id,assembly_json
q000000000000000000000000,"[{""motion"":""flying"",""boxes"":{""body"":[0.4,0.4,0.05,0.05]}},{""motion"":""flying"",""boxes"":{""body"":[0.4,0.4,0.05,0.05]}},{""motion"":""flying"",""boxes"":{""body"":[0.4,0.4,0.05,0.05]}},{""motion"":""flying"",""boxes"":{""body"":[0.4,0.4,0.05,0.05]}}]"
```

This ID and one-part roster are illustrative. Use actual sample IDs and exactly each supplied test anchor roster. The released sample already includes all required part keys and four frames with placeholder boxes and phases.

Execution constraints

Use one NVIDIA A10G with 24 GB GPU memory in a 90-minute session, including loading, fine-tuning, inference and serialization. Fine-tune an approved generic pretrained visual backbone and assembly prediction heads using public training data. Neural training and inference use the GPU; CPU loading, preprocessing and serialization are permitted. Generic initial checkpoints must be supplied by the runtime. No online access is required or permitted.

What not to use

Do not use external data or annotation lookup, source archive matching, private answers, manual test labeling, hard-coded test predictions, hosted model APIs, challenge-specific pretrained models or previously fine-tuned challenge weights. All fine-tuning must occur in the submitted script. Do not tune parameters or preprocessing using held-out inputs or aggregate held-out predictions. Use public training supervision, training-only validation and permitted generic initial weights.
