# Reading the Storm From the Sky

- Domain: Computer Vision
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: image, large-scale

---

Reading the Storm From the Sky

Overview

Cameras at high-latitude stations photograph the whole night sky every few seconds. When Earth's magnetic field is disturbed, the aurora brightens, moves and changes shape. This challenge asks whether a short sequence of sky images can tell how disturbed the field is now and how disturbed it will get in the next half hour.

The images are messy on purpose. Clouds, moonlight, sensor noise, hot pixels, missing frames and changing exposure are all present. Each sequence is also rotated and sometimes mirrored, so the direction of north is unknown.

Task

Each sample is a sequence of nine frames taken 15 seconds apart. The last frame is "now". For every test sequence, predict three values in nanotesla (nT).

* ae\_now

  + The auroral electrojet (AE) index at the last frame. It measures how strong the auroral currents are across the whole northern auroral zone.
* ae\_next30

  + The highest AE index during the 30 minutes after the last frame.
* local\_range10

  + The range (largest minus smallest value) of the horizontal magnetic field measured at the camera's station during the 10 minutes ending at the last frame.

Data

* train.csv: 12,511 sequences with id, frames\_present and the three targets.
* test.csv: 4,335 sequences with id and frames\_present.
* train\_aux.csv: training only. night\_group and aurora\_type for each training sequence.
* sample\_submission.csv: a valid submission filled with random values.
* images/train/ and images/test/: one JPEG per sequence named [id].jpg.

Each JPEG is a 3 x 3 grid of 160 x 160 grayscale frames. Read the grid left to right, then top to bottom. The ninth frame is the last one. A black tile means that frame is missing.

Column details:

* id

  + Type: string
  + Opaque sequence id.
* frames\_present

  + Type: integer
  + Number of frames that are not blank, from 4 to 9.
* ae\_now, ae\_next30, local\_range10 (train.csv only)

  + Type: float
  + The targets described above, in nT. All are zero or positive.
* night\_group (train\_aux.csv)

  + Type: integer
  + The night a training sequence comes from. Use it for grouped validation.
* aurora\_type (train\_aux.csv)

  + Type: string
  + For about 5,400 training sequences, an expert label: arc, diffuse, discrete, cloudy, moon or noaurora. Other sequences are unlabelled. It is not given for test sequences and is not scored.

Training and test sequences come from different nights. Test nights are quieter on average than training nights. No timestamps or station names are given.

What Makes This Challenge Different

* The input is a short time series of noisy sky images, not one clean picture. Motion and brightening between frames carry information.
* The targets are physical measurements from instruments other than the camera. A model has to learn how what the sky looks like relates to currents and magnetic fields.
* One target looks 30 minutes ahead, so part of the task is forecasting, not just describing.
* Moonlight and clouds can look bright without any magnetic activity. Brightness alone is misleading, so the model must understand what it sees.
* Random rotation, mirroring, exposure changes and dropped frames remove easy shortcuts tied to a particular camera.
* Expert aurora labels are available for part of the training set. They can be used as an extra training signal but are never scored.
* Scoring removes any gain from guessing the typical test level. All credit comes from ranking and estimating sequences correctly.

Compute

Solutions run on one NVIDIA A10G GPU with 24 GB of memory and CUDA available. Training and inference together must finish within 1.5 hours.

Submission Format

Submit a CSV with columns id, ae\_now, ae\_next30 and local\_range10. Include one row per test id. Values are in nT. Negative values are treated as 0.

Evaluation

Each target is compared on the scale log(1 + value), because activity is heavy-tailed.

For each target:

* rank part: Spearman rank correlation between your predictions and the truth. Values below 0 count as 0.
* error part: 1 minus your mean absolute error divided by the mean absolute error of the best constant guess. Values below 0 count as 0.
* target score: the average of the two parts.

Final score = 0.45 x local\_range10 + 0.30 x ae\_now + 0.25 x ae\_next30. Scores lie between 0 and 1. Higher is better.

Missing or non-numeric predictions are replaced by a fixed typical value. Missing columns, missing test ids or duplicate ids cause the submission to be rejected.

Rules

* Use only the files provided. Do not use any external sky camera images, aurora datasets, magnetometer data, space weather indices or solar wind data.
* Do not try to find the time or place of a sequence in any outside source.
* General-purpose pretrained weights are allowed, such as ImageNet, CLIP, DINOv2 or SigLIP models. Weights trained on auroral or sky camera data are not allowed.
* Predict each test sequence from its own frames only. Do not fit, pseudo-label, cluster or normalise using the test set as a whole. Test-time augmentation within one sequence is allowed.
