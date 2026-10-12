# Complete 32-second rhythmic loop editing

- Domain: From Scratch
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: audio

---

Background

A rhythmic editor needs a consistent beat-position schedule over an entire audio region and a loop that follows that schedule. Correctly locating a few strong attacks is insufficient when missed beats or phase slips corrupt the rest of the edit.

Overview

For each 32-second excerpt, recover every annotated beat with its native within-bar position and select a one- or two-bar loop that best satisfies the supplied duration request. Every excerpt is mono 16 kHz PCM16 with 512,000 samples. Audio outside a source recording is zero padded before time warping and noise are applied. There are no annotated beats in padded regions.

Audio and native beat timestamps undergo the same deterministic, smooth, strictly increasing time map. The map spans 0–32,000 ms. Independent positive rates are sampled on a 500 ms grid from 0.60–1.40, smoothed with a reflected [0.25,0.5,0.25] kernel and normalized to preserve total duration. Audio uses the inverse map; labels use the forward map. Warp generation does not inspect labels. Gain and 30 dB broadband noise are independent of the labels. These local tempo variations discourage a single constant-period grid.

Native recording families stay together. Evaluation recordings are held out from training. Training excerpts overlap and are correlated; use train\_groups.csv for grouped validation. Native timing and beat-position annotations remain the supervision source; no pseudo-labels are added.

Public files

* train.csv: excerpt\_id, audio\_path, preferred\_ms, max\_cycles, loop\_plan.
* test.csv: the same features, omitting loop\_plan.
* training\_beat\_supervision.csv: excerpt\_id and beats, reproducing the complete training schedules.
* train\_groups.csv: excerpt\_id, train\_group, train\_split, with fit and validation recording families.
* sample\_submission.csv: every test ID with a legal no-loop/empty-schedule placeholder.
* dataset\_counts.json: actual counts, sample rate, sample count and schedule bounds.
* audio/\*.wav: paths relative to the public directory.

Target schema

Submit exactly excerpt\_id,loop\_plan in that order, with every test ID once. loop\_plan is a JSON object containing cuts and beats.

cuts is [start\_ms,end\_ms,cycles]. Legal candidates start at a native downbeat whose unrounded warped time lies in [2000,4000) ms. They end at the next downbeat after exactly one or two complete native bars, with cycles no greater than max\_cycles and end no later than 32,000 ms. Select the candidate whose duration is closest to preferred\_ms; break ties by earliest unrounded start, earliest end, then smallest cycles. Round the selected boundaries to integer milliseconds. If there is no candidate, use [-1,-1,0]. preferred\_ms ranges from 1,500 to 4,500; max\_cycles is 1 or 2.

beats contains the entire 32-second schedule as integer [timestamp\_ms,position] pairs, in strictly increasing timestamp order. Timestamps range from 0 through 31,999. Positions are positive native within-bar indices. At most 256 beats may be submitted. Do not reset the phase at the selected loop or truncate the schedule to the loop.

An optional reference field may be null or contain native\_cuts and native\_beats with the same grammar. It is ignored when scoring predictions. It does not replace cuts or beats.

Evaluation metric

Each excerpt scores 1 only when both conditions hold:

* Loop success: the predicted cycle count equals the native target; for a nonempty loop, both boundaries are within 70 ms of the selected native boundaries. A native no-loop case requires the no-loop sentinel.
* Complete schedule: the predicted and native schedules have equal beat counts and admit a monotone one-to-one matching of every beat, with exact within-bar position and timing difference at most 50 ms. No missed or extra beats are allowed.

Otherwise the excerpt scores 0. Final score is the arithmetic mean over all test excerpts. Scores range from 0 to 1, higher is better, and perfect predictions score 1. Beat F1 alone is not the metric: one missed beat, extra beat or incorrect phase can invalidate a complete editing schedule.

Invalid row payloads score zero for that excerpt. Missing, duplicate or unknown IDs, incorrect columns, oversized or malformed tables invalidate the submission. Booleans and non-integers are rejected. Each JSON cell is limited to 32,768 characters.

Sample submission

A legal placeholder loop\_plan is {"cuts":[-1,-1,0],"beats":[]}. Quote this JSON cell according to CSV rules. Placeholder values are not expected answers. Submit every provided test ID exactly once; row ordering does not affect scoring.

What Not To Use

Do not use pretrained beat trackers, pretrained weights, external audio or annotations, hosted APIs, internet access, manual test labels, source lookup or hidden evaluation labels. Do not use opaque IDs or row order as label proxies.

Execution constraints

Finish preprocessing, training, inference and submission generation offline within 90 minutes using one NVIDIA A10G GPU with 24 GB VRAM. Initialize all learned parameters randomly and train on the supplied data. Deterministic signal processing and handcrafted audio features are allowed. CPU preprocessing and serialization are allowed.
