# Fine-tune muscle-signal models for cross-writer bank matching

- Domain: Fine-Tuning
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

Fine-tune muscle-signal models for cross-writer bank matching

Background

A drawing interface may receive a batch of wrist muscle recordings and a separate, unordered collection of editable pen programs. Associating the two modalities requires more than choosing a character name: several trials can depict the same character with different native curves. Each matched curve must also be assigned to the appropriate segment of its recording.

Overview

Fine-tune a generic pretrained sequence encoder on banks from two writers, then perform cross-modal matching for an entirely held-out third writer. For each evaluation bank, predict a one-to-one association between its muscle recordings and supplied native stroke programs, together with the native muscle-sample interval belonging to every selected curve. Use every program exactly once. All candidate coordinates are supplied input; the learned decisions are their recording associations and curve-level temporal ownership. Fine-tuning must occur in the submitted script within the stated GPU session.

A full bank has 16 trials and 16 programs, drawn from four native character classes with four actual trials per class. Nine evaluation tail banks have eight trials and programs, with two trials per class. Native class labels are used only for organizer batching and are not public features. Multiple trials can share a character identity; exact trial identity is not the scoring target. The metric gives credit according to the selected native geometry and its predicted interval ownership.

Programs contain the actual ordered pen-down curves. Each curve has exactly 32 equally spaced arclength points. Drawings are centered on their bounding box and uniformly scaled so their longer side has length one. Native axes, aspect ratio, direction and curve order remain unchanged. Arclength progress is not a time axis. No measured point-by-point synchronization, pressure, force, pen-up duration or invented kinematics is supplied.

The collection preserves all 45,000 selected native pairs: 36,000 training trials and 9,000 evaluation trials. Training contains two writers; the evaluation writer is excluded from training. Banks never mix writers. Training has 2,250 full banks; evaluation has 558 full banks and nine eight-trial tails. Every trial and its native program occurs exactly once in its split. Correct associations and native intervals are public only for training. Evaluation geometry is intentionally supplied as an unordered candidate bank for this matching task; evaluation intervals and associations are private.

Trial aliases, candidate aliases, list orders and bank row order are independently keyed. A program cannot be associated with a recording by matching alias strings or list positions. Training validation groups contain complete writer-and-character-quartet strata: 18 groups of 125 banks. Keep each group together during internal validation rather than splitting its repeated native classes across training and validation.

Recordings contain eight electrode channels sampled at 200 Hz. Deterministic, label-independent disturbances include gains from 0.97 to 1.03, Gaussian noise with standard deviation 8% of channel RMS, sinusoidal interference at 0.5-2 Hz with amplitude 2% of RMS, drift bounded by 1% of RMS and two short pulses bounded by 4% of RMS. RMS has a one-unit minimum. Sensor sample timing, native geometry and native interval targets remain unchanged.

Writer-exclusive protocol and validation

All 18,000 selected trials from each of the two training writers remain in training. All 9,000 selected trials from the third writer remain in evaluation. No recording, repeated trial or bank from that evaluation writer enters training. The same 36 native character classes occur in every writer: each training writer supplies 500 trials per class, and the held-out writer supplies 250. The generalization axis is writer identity, with character coverage retained; it is not unseen-character recognition or a measured device shift.

Banks preserve four actual same-class repeats per character in the usual sixteen-trial case. Character recognition alone therefore leaves four candidate programs per class to distinguish. Association choices must satisfy a complete bank-wide bijection, while interval predictions must explain the selected curves using each trial's own visible sample timeline. This protocol requires both cross-writer association and sample ownership; copying program coordinates or a character template supplies neither association nor native timing.

The 18 public validation groups are complete writer-and-character-quartet strata, each containing 125 banks. Keep each group together in training-only development. These strata are not separate devices or independent writer-level evaluation folds. The official evaluation boundary remains the entire third writer. Native trace and muscle intervals can contain the documented order or unassigned-span exceptions; those are retained, not repaired into synthetic timing labels.

Public files

```
+-----------------------+--------------+----------------------------+
| File                  | Rows / files | Contents                   |
+-----------------------+--------------+----------------------------+
| train.csv             | 2250         | Native banks and matches   |
| test.csv              | 567          | Inputs without matches     |
| sample_submission.csv | 567          | Legal format placeholders  |
| train_groups.csv      | 2250         | Group IDs for validation   |
| signals/*.npy         | 45000        | T x 8 float32 recordings   |
+-----------------------+--------------+----------------------------+
```

Signal paths are relative to the public directory. The arrays contain electrode values, not native stroke-index annotations or character labels. Recording length T varies. There are no additional required public files.

Input schema

```
+-------------------+--------+--------------------------------------+
| Column            | Type   | Meaning                              |
+-------------------+--------+--------------------------------------+
| id                | string | Bank ID: b and 24 hexadecimal digits  |
| trials_json       | JSON   | Independently ordered trial list     |
| program_bank_json | JSON   | Independently ordered native programs|
| assignment_json   | JSON   | Native matches; training only        |
+-------------------+--------+--------------------------------------+
```

train.csv contains these four columns in order; test.csv contains the first three. train\_groups.csv contains exactly id,group\_id. sample\_submission.csv contains exactly id,assignment\_json, matching the required submission columns.

trials\_json is a list of eight or sixteen objects with exactly trial\_id,signal\_file,n\_samples. trial\_id is t followed by 24 hexadecimal digits; signal\_file is the corresponding relative NumPy path; n\_samples is the positive integer T visible in that array.

program\_bank\_json is a list with the same bank size. Each object has exactly candidate\_id and strokes. candidate\_id is c followed by 24 hexadecimal digits. strokes contains one to eight objects with exactly points; each points value is a list of exactly 32 [x,y] pairs. The released native programs retain their actual curve counts. Geometry is public candidate context, not a linked evaluation answer.

assignment\_json is a list with exactly one object per trial. Each object has exactly trial\_id,candidate\_id,strokes. Every trial alias and candidate alias from that bank must occur exactly once. Each stroke has exactly points and samples. Copy points exactly from the selected public candidate program, preserving its curve and point order. The grader validates this copy and obtains scoring geometry from its trusted native candidate lookup; supplied points cannot fabricate a better candidate.

Coordinates are finite numbers in [-0.5,0.5], with numerical boundary tolerance 0.000001. Boolean coordinates are invalid. samples is an integer pair [start,end] defining the half-open native EMG interval [start,end), in zero-based indices. Require 0<=start<end<=n\_samples. One sample is 5 ms. Booleans and fractional indices are invalid.

Different curves assigned to one trial cannot overlap in sample time. Adjacent intervals and gaps are allowed. Intervals need not follow geometric curve order: native trace order is retained even when native interval order differs. Some muscle annotation spans have no retained curve and remain unassigned. The complete native census verifies that each trial's maximum native target end equals its visible recording length; the grader uses that trusted length to reject out-of-record intervals. No native pair or exception is dropped.

Evaluation metric

Maximize a trial-weighted score in [0,1]. For each trial, resolve the chosen candidate's actual native curves and combine them with the predicted intervals. Compare them with the trial's native ordered curves and intervals using ordered edit distance. For predicted curve i and target curve j:

```
d(i,j) = sqrt(sum(k=1..32, ||p[i,k] - t[j,k]||^2) / 32)
J(i,j) = length(predicted_interval_i intersect native_interval_j)
         / length(predicted_interval_i union native_interval_j)
c(i,j) = 1 - exp(-d(i,j) / 0.05) * J(i,j)
D(0,j) = j; D(i,0) = i
D(i,j) = min(D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+c(i,j))
trial_score = max(0, 1 - D(m,n)/max(m,n))
score = sum(all evaluated trial_score) / number_of_evaluated_trials
```

Here m and n are the selected and native program curve counts. Insertion and deletion cost one. The 0.05 scale represents 5% of the normalized drawing's longer side. Near native geometry receives gradual credit; correct interval ownership is still required. Choosing the wrong native program is penalized by its actual geometry difference, not by an arbitrary exact-trial-identity gate. Identical native geometries can therefore be interchangeable when intervals are correct. Exact native matches and intervals attain one; there is no arbitrary score ceiling or score rescaling.

Every native trial has equal weight. An eight-trial tail bank has half the weight of a sixteen-trial bank. A malformed bank prediction contributes zero for all its trials; it is never removed from the denominator. Incorrect, extra or reordered CSV columns, extra row fields, duplicate/missing/unknown bank IDs, invalid CSV or wrong row counts score zero globally. Submission row order is immaterial.

A bank payload must be at most 512 KiB in UTF-8; tables at most 512 MiB and 100,000 rows. Unknown or duplicate JSON keys, nonfinite values, Boolean numbers, wrong geometry copies, invalid aliases, repeated or missing candidate use, wrong point or curve counts, invalid intervals and overlapping intervals invalidate that bank. Class labels and original source identifiers are not prediction fields.

Sample submission format

Submit a UTF-8 CSV with exactly this header:

```
id,assignment_json
```

The released sample includes every test bank ID and a complete legal assignment using each listed trial and candidate exactly once. It copies each selected program's full 32-point curves and partitions the visible recording into placeholder intervals. These are format examples, not native matches or intervals. Replace them with learned predictions. Do not submit ellipses, incomplete points, empty lists or character names.

Execution constraints

Use one NVIDIA A10G with 24 GB GPU memory in a 90-minute offline session with 62 GB system RAM. The session includes loading, fine-tuning, inference, matching and serialization. Fine-tune a generic pretrained sequence encoder and train an association-and-interval prediction head using the supplied training banks. Generic pretrained encoders are permitted; initial checkpoints must be available in the runtime. No particular checkpoint is promised by this description. Neural fine-tuning and inference use the GPU; CPU preprocessing, assignment optimization and serialization are permitted. Checkpoints trained on these recordings or their copies are prohibited.

What not to use

Do not use external recordings, online or hosted APIs, source archive or identifier lookup, private associations or intervals, manual evaluation labeling, hard-coded evaluation assignments, or challenge-specific pretrained weights. Do not tune on evaluation labels or aggregate evaluation outcomes. Use public training supervision and complete held-out training groups for development. Candidate geometry is intentionally public and may be used; it does not provide the correct recording association or native sample ownership.
