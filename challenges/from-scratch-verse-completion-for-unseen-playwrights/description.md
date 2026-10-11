# From-Scratch Verse Completion for Unseen Playwrights

- Domain: From Scratch
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: -

---

From-Scratch Verse Completion for Unseen Playwrights

1. Overview

German drama from the 16th to the 19th century is largely written in verse:  
blank verse, rhymed couplets, alexandrines. The last word of a verse line  
carries the most constraint, because it must complete the sentence, fit the  
metre and often rhyme. Predicting it requires a model of the language, of  
metre and of rhyme at once.

**Train a language model from scratch on verse drama and predict the masked  
last word of verse lines in plays by playwrights who never appear in the  
training data, given the four preceding lines and the following line.**

2. Objective

Each test row shows one verse line whose final word has been replaced by  
`<mask>`. Return up to five candidate words, ranked from most to least likely.

Masked words are always:

* the last word of a complete verse line (a line not split between speakers);
* alphabetic, at least 3 characters long, compared case-insensitively;
* not among the 100 most frequent words of `train_corpus.csv`;
* seen at least 3 times in `train_corpus.csv`.

3. Public Data

| File | Description |
| --- | --- |
| `train_corpus.csv` | The pretraining corpus: all 316,197 verse lines of the 205 training plays (99 playwrights), in order. |
| `train.csv` | 8,005 masked-line examples from the training plays, built exactly like the test rows, with answers. |
| `test.csv` | 2,811 masked-line examples from 71 plays by 33 playwrights who have no play in training. |
| `sample_submission.csv` | Valid submission predicting the five most frequent eligible words for every row. |

`train_corpus.csv` columns: `play_id` (string, opaque play ID `PL_` + 8 hex  
characters), `line_index` (int, position of the verse line in the play, from 0),  
`text` (string, the verse line; stage directions inside the line removed).

`test.csv` / `train.csv` columns (`train.csv` adds `predictions`):

| Column | Type | Meaning |
| --- | --- | --- |
| `example_id` | string | `TE_######` or `TR_######`. |
| `play_id` | string | Opaque play ID. |
| `context` | string | The four verse lines before the masked line, joined by `/`. |
| `masked_line` | string | The verse line with its last word replaced by `<mask>` (punctuation kept). |
| `next_line` | string | The verse line after the masked line. |
| `predictions` | string (train only) | JSON list holding the masked word, lowercased, e.g. `["besitzen"]`. |

Verse lines from all speakers of a play are taken in reading order, so the  
context and the next line may belong to different speakers.

4. Submission Format

CSV with exactly the columns `example_id,predictions`, in that order, with one  
row per test `example_id` and no duplicates or extra IDs. `predictions` is a  
JSON list of 1–5 distinct non-empty strings of at most 60 characters, ranked  
best first.

```
example_id,predictions  
TE_000001,"[""besitzen"", ""gewinnen"", ""erretten""]"  
TE_000002,"[""herz""]"
```

5. Evaluation

**Mean Reciprocal Rank at 5 (MRR@5).** Predictions and answers are trimmed and  
lowercased. A row scores `1/r` if the masked word is at rank `r` (1–5) and `0`  
otherwise. The final score is the mean over all test rows. Range `[0, 1]`;  
higher is better.

A submission is rejected with an explicit validation error (no score) if the  
columns are wrong or out of order, any ID is missing, duplicated or unknown,  
any value is missing, or any `predictions` cell is not a JSON list of 1–5  
distinct non-empty strings of at most 60 characters. Only `example_id` and  
`predictions` are read from the answers, so platform split columns (for  
example `visibility`) are ignored.

6. Split and Leakage Controls

* **Split unit: playwright.** Each play's author (Wikidata ID) is assigned to  
  train or test by a salted hash; 25% of the 132 playwrights are test. No test  
  playwright has any play in `train_corpus.csv` or `train.csv`.
* Only plays written (or else printed or premiered) in 1900 or earlier, with at  
  least 200 verse lines, are used.
* **Row construction:** at most 40 rows per play. Masked lines in the same  
  play are at least 6 lines apart, so no row's context or next line contains  
  another row's masked line.
* **Overlap guard:** a test row is dropped if its full masked line (normalized)  
  occurs anywhere in `train_corpus.csv`.
* Play IDs and example IDs are salted hashes or hash-sorted sequence numbers;  
  author, title and year are not published.

7. Not Allowed

* **No pretrained weights:** no pretrained language models, embeddings,  
  tokenizers trained elsewhere, rhyme dictionaries or morphological  
  analyzers. All learned components must be trained from scratch on the  
  provided files. Hand-written rules (for example rhyme matching on word  
  endings) are allowed.
* No external text: no other German literature, the GerDraCor corpus or  
  dracor.org, or any other edition of these plays.
