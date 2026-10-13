# Never-Heard Story: Matching Brain Recordings to Speech from Scratch

- Domain: From Scratch
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

Never-Heard Story: Matching Brain Recordings to Speech from Scratch

**Domain:** From Scratch. You design and train your own models of brain signals and of speech audio, from random initialisation, on the released recordings, on one GPU. No pretrained component of any kind is allowed.

Overview

Magnetoencephalography (MEG) measures the tiny magnetic fields produced by the brain, with a helmet of sensors around the head, many times per second. When a person listens to speech, the auditory cortex follows the sound: its activity rises at syllable and word onsets, tracks loudness and pitch changes, and reacts to pauses. Part of this response is visible in MEG, but it is buried in much larger background activity and environmental noise, and it differs between people and between sessions.

In this challenge, people listened to narrated stories while their MEG was recorded. Training data cover three stories. **The test story is a fourth story that never appears in the training data**, and some test listeners are also new.

The test set is organised in **pools**. A pool comes from one listener in one session, while that listener heard the test story. It holds:

* about 159 **probes**: 3-second MEG windows of that listener;
* about 318 **candidates**: 2-second audio clips cut from the test story.

Every probe was recorded while the listener heard exactly one of the pool's candidate clips. The other half of the candidates are distractors: clips from the same story that none of the pool's probes was recorded during.

**For every probe, find the clip the listener was hearing.**

Why this is hard

* **A story nobody trained on.** No audio, MEG or annotation of the test story is in the training data. The model must link brain activity to speech it has never heard, not recognise memorised passages.
* **Two seconds of speech, many look-alikes.** All candidates of a pool come from the same story, read by the same voices at similar rates. A pool has about 318 of them, so guessing is right about once in 318 tries.
* **Weak, noisy signal.** The speech-locked part of the MEG is small compared with spontaneous brain activity, eye and muscle artefacts, and environmental magnetic noise. A few seconds give little evidence.
* **Different heads.** Each person's brain, head shape and position in the helmet differ, and the head is placed again at every session. Seven test listeners have no training data at all.
* **Limited speech in training.** Every training listener heard the same three stories, so the training data contain many brain recordings but only about 44 minutes of distinct speech. Models that memorise the training audio do not transfer.

Why this is not plain classification

* **Open set on both sides.** The answer for a probe is another test item, a clip of unseen audio, not a label from a fixed list. No class of the test story exists in training.
* **Cross-modal matching.** The model must learn, from the training stories, a shared representation of brain signals and speech in which matching pairs are close, and then compare new brain windows with new clips.
* **Structured output.** Within a pool, every clip is the true match of at most one probe. The answer for a pool is a partial one-to-one matching between 159 probes and 318 clips.

Data

All files are in the public folder.

MEG signals

* **224 columns** per time sample, stored as int16:

  + columns 0 to 207: sensors around the head (axial gradiometers), in femtotesla (1 unit = 1 fT);
  + columns 208 to 223: reference sensors far from the head that mostly pick up environmental noise, in units of 10 fT.
* **50 samples per second.** The signals were recorded at 1,000 Hz and resampled to 50 Hz with an anti-aliasing filter; nothing else was removed or corrected.
* Sensor positions and orientations, in metres in a coordinate frame fixed to the helmet, are in `sensors.csv` `channel`, `kind`, `x`, `y`, `z`, `nx`, `ny`, `nz`). Reference sensors have no meaningful position.

Audio

* int16 mono waveform, **16,000 samples per second**.
* The stories are read by three voices whose speaking rate changes every few sentences, with short pauses between sentences. Story passages alternate with lists of words taken from the passage, read one by one.

Training files: 20 listeners, three stories

* `train_meg.npy`: int16 array of shape (5,475,600, 224), every training recording back to back.
* `train_recordings.csv`, one row per recording (one listener, one session, one story):

  + `recording_id` (int)
  + `listener_id` (int, 1 to 20): the same person across sessions and stories
  + `session` (int, 0 or 1): most listeners came for two sessions, on different occasions, and heard all stories in each
  + `story` (string): `A`, `B` or `C`
  + `row_offset` (int): first row of the recording in `train_meg.npy`
  + `n_rows` (int): length in rows (50 rows per second). Recordings are complete, including the time between audio parts.
* `train_audio.npy`: int16 array, every training audio part back to back.
* `train_sounds.csv`, one row per audio part (a story is played as several parts of 1 to 3 minutes):

  + `sound_id` (string), `story`, `part` (int, playing order within the story)
  + `sample_offset` (int), `n_samples` (int): position in `train_audio.npy`
* `train_sound_onsets.csv`: `recording_id`, `sound_id`, `onset_s` (float). The audio part started playing `onset_s` seconds after the first row of the recording, so audio time `t` of that part corresponds to row `round((onset_s + t) * 50)` of the recording, to within one row.
* `train_words.csv`: `sound_id`, `word_index`, `onset_s`, `duration_s` (seconds from the start of the audio part), `condition` `sentence`, `word_list` or `pseudo_words`), `voice` `v1`, `v2`, `v3`), `speech_rate` (words per minute). The words themselves are not given.
* `train_phonemes.csv`: `sound_id`, `word_index`, `onset_s`, `duration_s`, `phoneme`. Phoneme labels carry a position suffix `_B` first phoneme of the word, `_I` inside, `_E` last, `_S` a one-phoneme word).

Word and phoneme timings come from automatic alignment and can be off by a few tens of milliseconds.

Test files: 49 pools

* `test_meg.npy`: int16 array of shape (7,783, 150, 224), one probe per row.
* `test_probes.csv`: `probe_id` (int), `pool_id` (int), `listener_id` (int), `session` (int), `row` (int, row in `test_meg.npy`).

  + `listener_id` is the training listener id for the 37 pools of listeners who are in the training data.
  + `listener_id` is `-1` for the 12 pools of the seven held-out listeners.
* `test_audio.npy`: int16 array of shape (15,572, 32,000), one 2-second candidate clip per row.
* `test_candidates.csv`: `candidate_id` (int), `pool_id` (int), `row` (int, row in `test_audio.npy`).

**How the pools were built.**

* There is one pool per recording of the test story: 49 pools. All probes of a pool come from that one recording.
* Inside every audio part of the test story, slots start at a random phase and repeat every 3.5 seconds. A slot is the 2-second stretch of audio from its start time `t`. The random phase is drawn anew for every part and every recording, so the clips of different pools usually start at different times.
* Every slot of the recording is a candidate of the pool: its 2-second clip.
* Half of the slots, drawn at random, are also probes. A probe's MEG window runs from 0.5 seconds before `t` to 0.5 seconds after the end of the clip: 150 rows, with row 25 aligned with the start of the clip to within one row.
* Slots, probes and distractors were drawn without looking at the MEG or the audio.
* Probe ids, candidate ids, pool ids, the storage rows and the order of the csv files are random and carry no information.

`sample_submission.csv`

A valid submission in the required format, with a random candidate of the probe's pool for every probe.

Evaluation

The score is **matching accuracy**: the fraction of probes whose submitted `candidate_id` is the clip the listener was hearing.

```
import numpy as np

def evaluate(true_candidate_id, predicted_candidate_id):

    # one entry per probe

    t = np.asarray(true_candidate_id)

    p = np.asarray(predicted_candidate_id)

    return float(np.mean(t == p))
```

The score ranges from 0 to 1, and higher is better. A candidate from another pool always counts as wrong. You may give the same candidate to several probes, but at most one of them can be right. Each pool is scored entirely on either the public or the private leaderboard.

From-Scratch Requirement

Every learned component must be trained from random initialisation, using only the released training files. Every fitted state your solution uses must be created from the released training data during the graded run. That covers network weights, tree structures and split points, embeddings, scalers, lookup tables, calibration maps, and any threshold or weight chosen by looking at data. Loading weights or any other fitted state produced beforehand, or shipping a model with the submission, does not satisfy this.

The component that compares brain windows with audio clips must be fitted to the training data, whether a neural network, a linear model or anything else fitted. Hand-written signal processing (filtering, spectrograms, envelopes, re-referencing with the reference sensors) and post-processing (such as solving the one-to-one matching within a pool) are fine, provided any parameter they use is fitted on training data only, or is computed from the probes and clips of the single pool being processed.

Within a pool you may use all of its probes and candidates together, for example to normalise the MEG with the pool's own statistics or to solve the matching jointly.

What not to use

* No pretrained weights of any kind, including speech, audio, language or brain-signal models, and weights reached indirectly through a library or a cached model package. No features or embeddings produced by a pretrained model.
* No external data: no other brain recordings, speech corpora, text, pronunciation dictionaries or audio. **Do not attempt to obtain the source dataset**, the stories, their text or any other copy of these recordings.
* **Solve each pool independently.** The prediction for a probe may depend only on the trained models and on the probes and candidates of its own pool. Do not compare, cluster, link, normalise, fit or pseudo-label across pools. In particular, do not try to recognise the same listener or the same stretch of story in different pools, and do not reassemble the test story from clips.
* No training or adaptation on test MEG or test audio, including self-supervised or unsupervised training. No use of test-set statistics to choose features, thresholds or weights.
* Do not transcribe or otherwise decode the words of the test audio with any tool that was not trained by your solution on the released training files.
* No hard-coded answers, and no use of `probe_id`, `candidate_id`, `pool_id`, `row` or file order as information.
* Do not search the web, or use any external source, to identify this dataset, its publication, its authors, the stories or the collection setup. Do not use such information in your solution.

Compute

One GPU (A10G class). Your solution must finish end to end, including loading, training and prediction, within **1.5 hours**.

* The limit applies to the **complete pipeline that produces the submission**, run from scratch on one GPU. That includes:

  + every training stage, including stages warm-started from an earlier stage;
  + auxiliary models such as re-rankers, calibrators or scorers;
  + candidate generation on training data;
  + decoding and post-processing of the test items.
* If the solution is split over several scripts, the times of all stages add up.
* The time available in your working session is longer than this limit and does not extend it. A submission whose pipeline needs more than 1.5 hours does not satisfy the challenge, whatever its score.

Submission

Write your predictions to `submission.csv`, with one row per probe and these columns:

* `probe_id` (int): a `probe_id` from `test_probes.csv`.
* `candidate_id` (int): the `candidate_id` of the clip you match to it.

```
probe_id,candidate_id

100076,3055623

100083,8308134

...
```

**Requirements**

* Exactly 7,783 rows, one for each probe in `test_probes.csv`, with each `probe_id` exactly once.
* Include a header row.
* A missing probe, a duplicate `probe_id`, an empty or non-integer `candidate_id` makes the submission invalid.
