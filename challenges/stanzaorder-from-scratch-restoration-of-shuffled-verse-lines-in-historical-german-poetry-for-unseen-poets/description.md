# StanzaOrder: From-Scratch Restoration of Shuffled Verse Lines in Historical German Poetry for Unseen Poets

- Domain: From Scratch
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

StanzaOrder: Reassembling Scrambled Stanzas of Early-Modern German Verse from Rhyme, Metre and Period Punctuation

1. Overview

Printed German poetry from the Baroque to the late nineteenth century is  
highly structured. Rhyme schemes such as `abab`, `aabb`, `abba` and the  
Baroque `aabccb` tie specific lines together. Alternating masculine and  
feminine endings follow fixed patterns. Iambic and trochaic metre sets the  
syllable count of each line. Period punctuation marks the sense units: the  
virgule `/` within a line and the colon or full stop that closes a stanza.  
None of this behaves like modern prose. The spelling of these prints varies  
(long s `ſ`, `vnd`, `seyn`, `Hertz`), so the same rhyme can be spelled in  
several ways, and a model has to discover the sound patterns from the text  
itself.

This challenge treats stanza reconstruction as a problem of **verse  
structure**, not of general text coherence. Each example is one stanza whose  
lines have been scrambled. Restoring the printed order means inferring which  
lines rhyme, which metrical and ending pattern the poet used, and where the  
syntax and the punctuation close. All of this has to be done for poets the  
model has never seen, whose stanza forms and spelling habits differ from those  
in training.

**Restore the printed line order of scrambled 4–10 line stanzas of German  
verse (1600–1900) by poets who are not in the training data.** Models must be  
trained from scratch.

2. Objective

For each test stanza, output its line labels in the printed order.

3. Public Data

| File | Description |
| --- | --- |
| `train.csv` | 10,374 stanzas by 112 poets, lines scrambled, with the printed order. |
| `test.csv` | 2,985 stanzas by 32 other poets, lines scrambled. |
| `sample_submission.csv` | Valid submission listing each stanza's labels as `L1 L2 … Ln`. |

| Column | Type | Meaning |
| --- | --- | --- |
| `example_id` | string | `TR_#####` / `TE_#####`. |
| `author_id` | string | Opaque poet ID (every test poet is absent from training). |
| `n_lines` | int | Number of lines `n` (4–10). |
| `lines` | string | JSON array of the stanza's `n` lines in scrambled order, exactly as printed (original spelling and punctuation). The k-th element has label `Lk`; the label number carries no order information. |
| `order` | string (train only) | Line labels in printed order, for example `L3 L1 L4 L2`. |

Stanza sizes in test: 4 lines 1,103; 5 lines 201; 6 lines 683; 7 lines 108;  
8 lines 626; 9 lines 74; 10 lines 190. Stanzas come from lyric collections,  
songs and hymns, verse epics and verse inserted in novels. Each stanza  
appears once. Stanzas are excluded if they repeat an identical line or contain  
print-wrapped fragments (a line ending in a hyphen, a one-word line, or a line  
starting in lowercase).

4. Submission Format

CSV with exactly the columns `example_id,order`, in that order, one row per  
test `example_id`. `order` is a permutation of that stanza's line labels,  
separated by single spaces.

```
example_id,order  
TE_00001,L2 L4 L1 L3  
TE_00002,L5 L1 L3 L6 L2 L4
```

5. Evaluation

For a stanza with printed order `g1 g2 … gn`, the row score is the fraction of  
the `n − 1` consecutive printed pairs `(gk, gk+1)` for which `gk+1`  
immediately follows `gk` in the predicted order:

`row_score = #{k : pred_next(gk) = gk+1} / (n − 1)`

A value that is not a permutation of that stanza's labels scores 0 for that  
stanza. The final score is the mean over all test stanzas. Range `[0, 1]`;  
higher is better. A random order scores about `1/n`.

A submission is rejected with an explicit validation error (no score) if the  
columns are wrong or out of order, any ID is missing, duplicated or unknown,  
or any value is missing or empty.

6. Split and Leakage Controls

* **Split unit: poet.** No text of a test poet appears in any public file of this or related challenges built  
  from the same archive.
* Test stanzas share no line with any training stanza.

7. Not Allowed

* **From scratch:** no pretrained weights of any kind (language models, embeddings), and no external text,  
  poetry collections, rhyme or pronunciation dictionaries. Train only on the files provided.
* No use of any edition of the poems, and no data from other challenges.
* Each test row must be predicted on its own: do not use other rows of `test.csv` (their inputs may contain material from the same source text).
