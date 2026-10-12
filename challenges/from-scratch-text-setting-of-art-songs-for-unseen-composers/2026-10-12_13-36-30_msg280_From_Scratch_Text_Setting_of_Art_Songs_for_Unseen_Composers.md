# From-Scratch Text Setting of Art Songs for Unseen Composers

- Domain: From Scratch
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

From-Scratch Text Setting of Art Songs for Unseen Composers

1. Overview

A song composer "sets" a text: each syllable gets a pitch and a duration.  
Stressed syllables fall on strong beats, phrase endings are lengthened, a  
single vowel may be stretched over several notes (melisma), and the melody  
moves within the harmony and the singer's range. Every composer does this in a  
personal style. Modelling text setting is useful for music generation,  
score completion and editorial reconstruction of damaged sources.

**Train a model from scratch on art songs and generate the vocal notes of the  
next bar from the three preceding bars and the lyric syllables to be sung in  
that bar, for composers whose songs never appear in training.**

2. Objective

For each test row, generate the complete vocal content of the target bar: a  
sequence of events `pitch/duration`, where `pitch` is a MIDI note number or `r`  
(rest) and `duration` is a fraction of a whole note (`1/4` = crotchet, `3/8` =  
dotted crotchet, `1/12` = triplet quaver). Every target bar is complete: its  
durations add up to the time signature, so pickup and incomplete bars are not  
targets. It contains at least one sung syllable.

3. Public Data

| File | Description |
| --- | --- |
| `train.csv` | 35,803 training rows (every eligible bar of the training songs). |
| `test.csv` | 2,009 rows from 290 songs by 43 composers who have no song in training. |
| `sample_submission.csv` | Valid submission with two crotchets on G4 for every row. |

| Column | Type | Meaning |
| --- | --- | --- |
| `example_id` | string | `TR_#####` / `TE_#####`. |
| `composer_id`, `song_id` | string | Opaque IDs. |
| `time_signature` | string | E.g. `3/4`, `6/8`. |
| `key_signature` | int | Number of sharps (positive) or flats (negative). |
| `context` | string | The three preceding bars, separated by `|`. Events are `pitch/duration`, followed by `[syllable]` when a syllable starts on that note; `empty` marks a bar with no events. |
| `lyrics` | string | Syllables sung in the target bar, in order, space-separated (verse 1). A leading `-` continues a word, a trailing `-` means the word continues, spaces inside a syllable are written `_`. |
| `notes` | string (train only) | The target bar's events, e.g. `67/1/8 r/1/8 70/3/16 68/1/16`. |

Context and target use the first (voice) staff and the first verse of the  
lyrics. For chords, the highest note is taken; grace notes are omitted.

4. Submission Format

CSV with exactly the columns `example_id,notes`, in that order, one row per  
test `example_id`. `notes` is 1–64 space-separated events matching  
`(r|<0–127>)/<fraction>`, with durations greater than 0 and at most 2.

```
example_id,notes  
TE_00001,67/1/8 r/1/8 r/1/8 67/1/8 70/3/16 68/1/16  
TE_00002,72/1/2 66/1/2
```

5. Evaluation

For each row, two normalized edit similarities are computed, each  
`1 − Levenshtein(pred, gold) / max(len(pred), len(gold))`:

* **rhythm:** over the sequences of durations;
* **pitch:** over the sequences of pitches (rests as `r`).

`row_score = 0.5 × rhythm + 0.5 × pitch`. The final score is the mean over all  
test rows. Range `[0, 1]`; higher is better.

A submission is rejected with an explicit validation error (no score) if the  
columns are wrong or out of order, any ID is missing, duplicated or unknown,  
any value is missing, or any `notes` value is malformed.

6. Split and Leakage Controls

* **Split unit: composer.** No song of a test composer appears in any public training file.
* Test windows do not overlap, so no test target appears in another test row's context.
* Song titles, composers and lyric texts beyond the shown syllables are not published.

7. Not Allowed

* **No pretrained weights:** no pretrained music or language models; all learned components are trained  
  from scratch on the provided files.
* No external scores: no score archives, music databases or lookups of these songs.
* Each test row must be predicted on its own: do not use other rows of `test.csv` (their inputs may contain material from the same source text).
