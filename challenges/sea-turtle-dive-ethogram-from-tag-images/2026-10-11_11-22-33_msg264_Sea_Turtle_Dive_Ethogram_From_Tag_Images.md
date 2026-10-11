# Sea Turtle Dive Ethogram From Tag Images

- Domain: Computer Vision
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

Sea Turtle Dive Ethogram From Tag Images

Task Type

This is a computer-vision task on time-frequency images: weakly supervised segmentation of each image along its horizontal (time) axis. Every image shows one minute of the movement of a wild sea turtle, recorded by a motion tag glued to its shell, and it has to be divided, column by column, into stretches of the turtle's behaviours (swimming, gliding, resting on the bottom, breathing at the surface and so on). The result is assessed through three structural statistics of that segmentation: how much of the minute the turtle spent swimming, where the first change of behaviour lies, and how long the longest uninterrupted stretch of one behaviour lasts. None of the three statistics is a brightness, a texture or any other property that the image carries as a whole; each one exists only once the image has been cut into stretches. The segmentation itself is never given for training; only its three statistics are.

Overview

Biologists who put camera tags on sea turtles watch the video afterwards and write down, moment by moment, what the animal was doing. That video labelling takes hours per deployment, and the cameras run for a few hours only, while the motion sensors of the same tag can record for weeks. What behavioural studies then report are summary numbers per stretch of recording: how much time the turtle spent actively swimming, when it switched from one activity to the next, and how long its activity bouts lasted. The aim here is to read those numbers directly from a picture of the tag's motion signals, so that the long sensor records without video can be summarised in the same terms.

Each item is one grayscale image, 264 pixels high and 240 pixels wide, made from a 60-second window of the tag's 20 Hz three-axis accelerometer and three-axis gyroscope. Column k is centred on second 0.25 x k of the window, so time runs from left to right. The image has two parts:

* **Rows 0 to 191: six time-frequency panels** of 32 rows each, from top to bottom for the accelerometer X, Y and Z axes and the gyroscope X, Y and Z axes. Within a panel, the top row is 9.6875 Hz and the bottom row is 0 Hz, in steps of 0.3125 Hz; each column is the power spectrum of a 3.2-second Hann-windowed stretch centred on the column, with its mean removed. Brighter is more power, on a fixed logarithmic scale that is the same for every image (seven decades from black to white, one scale for the accelerometer panels and one for the gyroscope panels), so equal grey levels mean equal power in every image.
* **Rows 192 to 263: six trace bands** of 12 rows each, in the same channel order, whose grey level is the mean of the channel over the quarter second around the column on a fixed linear scale: from -1.25 g (black) to +1.25 g (white) for the accelerometer axes, which shows how the shell is tilted relative to gravity and the slow rocking of flipper strokes, and from -300 to +300 gyroscope units for the gyroscope axes.

Segment each image into behaviour stretches along its width and report three statistics of the segmentation:

* `**swim_share**`: the share of the window's samples labelled as swimming (active flipper stroking), between 0 and 1.
* `**first_change_s**`: the time in seconds from the left edge of the window (its start) to the first moment at which the labelled behaviour changes, between 0.05 and 59.95.
* `**longest_bout_s**`: the duration in seconds of the longest uninterrupted run of a single behaviour inside the window. A run cut by the start or the end of the window counts with the part that lies inside it.

The behaviours were labelled by hand from the tag's video, at the 20 Hz rate of the sensors, as one of seven behaviours: breathing, feeding, gliding, resting, scratching, staying at the surface and swimming. A window is used only when at least 95 % of its samples carry a label and the labelled behaviour changes at least once inside it; the few unlabelled samples take the label of the preceding labelled sample (or of the first labelled sample, when the window starts unlabelled). Every window therefore holds at least two bouts.

Over the training windows the three statistics are close to independent: their pairwise rank correlations are 0.003 (swim share vs first change), -0.026 (swim share vs longest bout) and 0.173 (first change vs longest bout). Their training medians are 0.482, 17.1 s and 36.0 s.

What sets this task apart

* **The behaviour sequence is never shown.** Training supervision is three numbers per minute, never the moment-by-moment labels they were computed from. Where a bout starts and ends, and which behaviour it is, has to be discovered from those window-level statistics alone.
* **A location, not an amount.** The first-change time asks where along the image the first boundary between two behaviours lies, and it is scored with a tolerance of one second (four image columns). Getting the overall mix of behaviours right says nothing about it.
* **Structure, not totals.** The longest bout depends on where all the boundaries lie, not on how much of each behaviour there is: one missed boundary can double it, one invented boundary can halve it.
* **Unseen turtles.** Every test window comes from a turtle that contributes nothing to training, with its own tag placement on the shell, body size, habitat and habits.

Why it is difficult

* **Some boundaries are subtle.** Swimming and gliding differ only in whether the flippers beat; a turtle may glide between strokes for a few seconds, and the labelled change can lie a second or two away from where the signal first changes.
* **Tag placement differs.** Each tag sits at a slightly different position and angle on the shell, so the same posture tilts the accelerometer axes differently from one turtle to the next, and the trace bands shift accordingly.
* **Short events.** Breathing at the surface and scratching can last only a few seconds, but each one creates boundaries that decide the first-change time and cut the bouts around it.
* **Uneven animals.** Training turtles contribute between 15 and 367 windows each, so a few animals dominate the training set.
* **The left edge matters.** The window start is fixed, and the first-change time depends on the direction of time, so the image cannot be mirrored left to right without changing the answer.

Data

* `train_images/<id>.png`: the 1,240 training images, 264 x 240 grayscale.
* `train_labels.csv`: `id, swim_share, first_change_s, longest_bout_s`, the truth for every training window, in submission format.
* `train_groups.csv`: `id, turtle`. The turtle label (`t` plus two digits) is shared by all windows of one animal.
* `test_images/<id>.png`: the 392 test images to read.
* `test_queries.csv`: `id` of every test window.
* `sample_submission.csv`: a valid submission holding constants chosen on the training labels, the training means: `swim_share` 0.49569, `first_change_s` 21.45085 and `longest_bout_s` 37.10403.

Windows are consecutive, non-overlapping one-minute blocks of each recording, counted from its first sample. Neighbouring windows of one turtle can be similar, so the turtle is the unit that must be kept together.

Evaluation protocol: whole turtles held out

Three of the thirteen turtles are held out: every window of a test turtle is in the test set (392 windows, 24.0 % of all windows), and no turtle has windows on both sides. Windows of one turtle share the tag position, the animal's body and its habits, so a random split of the training windows overstates what transfers to a new animal; validate by holding out whole `turtle` groups.

Scoring

Each statistic has its own measure. Every sub-score is clipped to [0, 1].

* **SwimScore** (root-mean-square-error skill) = 1 - sqrt( sum (sub - true)^2 / sum (0.48208 - true)^2 ), where sub is the submitted value. Submitted values are clipped to [0, 1].
* **ChangeScore** (tolerance hit-rate skill). A window is a hit when the submitted first-change time is within 1.0 second of the true one, |sub - true| <= 1.0. ChangeScore = (H - H0) / (N - H0), where H is the number of hits of the submission, N the number of test windows and H0 the number of hits that the constant 17.1 s would make. Submitted values are clipped to [0.05, 60].
* **BoutScore** (log-ratio skill) = 1 - sum |ln(sub / true)| / sum |ln(36.0 / true)|. Submitted values are clipped to [0.05, 60].

**DiveScore** = 0.20 x SwimScore + 0.40 x ChangeScore + 0.40 x BoutScore, higher is better, in [0, 1].

The constants 0.48208, 17.1 and 36.0 are the training medians of the three statistics. They are fixed, so submitting the same constant for every window scores about zero, and the exact statistics of the labelled behaviour sequence score 1.

Submission

A CSV file with exactly the columns `id, swim_share, first_change_s, longest_bout_s` and one row per test window, as in `sample_submission.csv`.

The submission is rejected with no score if a column is missing, if any other column is present (including a saved index column), if an id is blank, if an id appears twice, or if a test window has no row. Rows whose id is not a test window are ignored. A value that is empty, malformed or not finite is read as the fixed training median of that statistic (swim\_share 0.48208, first\_change\_s 17.1, longest\_bout\_s 36.0); the rest of the submission is scored normally.

Rules

* Allowed: any method trained on the provided training images and labels; general-purpose pretrained weights, such as backbones trained on ImageNet photographs, not trained on animal-tag, accelerometer or other motion-sensor recordings.
* Not allowed: any use of the test images beyond submitting one row of values for each of them: no training, fine-tuning, pseudo-labelling, self-supervised or unsupervised fitting, and no normalisation or other statistics computed across the test set; every test item must be read on its own by a model fitted on the training data only; external bio-logger or motion-sensor recordings, behaviour labels or behaviour models, or any model pretrained on accelerometer, gyroscope or other inertial signals; attempts to locate the source recordings or their labels; hand-entered or hard-coded test answers, or constants tuned on the test set.
