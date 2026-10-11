# Wiktionary Relation Closure: Typing Hidden Synonym, Antonym and Hypernym Links in French

- Domain: From Scratch
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: text, large-scale

---

Wiktionary Relation Closure

Objective

For each test pair of French dictionary entries (a query entry and a candidate entry), predict how the candidate relates to the query: synonym (SYN), antonym (ANT), broader term (HYPER), narrower term (HYPO), or no listed relation (NOREL). The query entries come from word families (stems) that never appear as queries in training, and the predictions should stay logically coherent with each other and with the known relation graph.

Overview

Wiktionary editors keep lists of synonyms, antonyms, hypernyms and hyponyms for each word, but the lists are patchy. Many words have none, and many relations are listed on one side only. A system that can read a dictionary and complete these lists for words it has never seen labelled could densify lexical resources for French and for other under-resourced languages.

You receive the complete French Wiktionary as text: about 405,000 entries with every numbered definition and up to three usage examples. You also receive every relation among entries of the training families. For the test families, all relations are removed from the public files. The only exception is one revealed relation per query (the anchor) for queries that had at least two.

Three things make this a reasoning task and not only text similarity:

* Hold-out by word family. Test queries belong to stem families (first five letters of the first content word, accents removed) that are disjoint from every training query family. Morphological neighbours such as "solide", "solidité" and "solidifier" are therefore all in test or all in train.
* Hard negatives. NOREL candidates are cohyponyms of the query (words under the same hypernym), definition look-alikes with the same part of speech, and frequency-matched words. None of them is linked to the query in Wiktionary, directly or through a synonym or antonym chain.
* Closure. A query's anchor, and the known relations of train-side entries, constrain the answer through composition. For example, a synonym of a synonym is not an antonym, and a narrower term of a narrower term is not a broader term. Incoherent predictions are penalised (see Evaluation).

Data

Files in `public/`:

* `lexicon.csv.gz` — 404,740 rows, one per entry (nouns, adjectives, verbs, adverbs)
  + `entry_id` (string) — random entry id, for example `w0c445152fca3034`
  + `headword` (string) — the written form, possibly multiword ("barbe à papa")
  + `pos` (string) — `noun`, `adj`, `verb` or `adv`
  + `n_senses` (int) — number of definitions
  + `definitions` (string) — all definitions in sense order, formatted `1. text || 2. text || ...`
  + `examples` (string) — up to three usage examples joined by `||`, empty for 175,720 entries
* `train_relations.csv` — 122,701 rows, every labelled relation whose two entries are both outside the test families
  + `head_id`, `tail_id` (string) — entry ids
  + `relation` (string) — how the tail relates to the head: `SYN`, `ANT`, `HYPER` (tail is broader than head) or `HYPO` (tail is narrower than head). Each unordered pair appears once, in a random orientation.
* `train.csv` — 273,956 rows of labelled candidate bundles for 60,000 training queries, built exactly like the test bundles
  + `id` (string) — row id
  + `query_id`, `candidate_id` (string) — entry ids
  + `relation` (string) — `SYN`, `ANT`, `HYPER`, `HYPO` or `NOREL`, read as "the candidate is a \_\_\_ of the query"
* `train_anchors.csv` — 28,305 rows (`query_id`, `anchor_id`, `relation`): the revealed relation of each training query that has one. These pairs are not repeated in `train.csv`.
* `test.csv` — 66,980 rows for 14,125 test queries (3 to 15 candidates each)
  + `id`, `query_id`, `candidate_id` (string)
* `test_anchors.csv` — 6,904 rows (`query_id`, `anchor_id`, `relation`): one revealed relation per test query that had at least two. Anchor pairs are never scored.
* `sample_submission.csv` — every test id with the constant label `SYN`

How the files link:

* Every `query_id`, `candidate_id`, `anchor_id`, `head_id` and `tail_id` is an `entry_id` in `lexicon.csv.gz`.
* Test queries have no rows in `train_relations.csv`. Their only revealed relation is the anchor.
* Test candidates can be training-family entries, whose relations to other training-family entries are visible in `train_relations.csv`, or test-family entries, whose relations are all hidden.
* When two test queries are related, both orientations appear in `test.csv`, as (Q, C) and (C, Q).
* In `train.csv` the training queries' own relations are also present in `train_relations.csv`. To mimic test conditions when you build features, mask a training query's edges except its anchor.

Label shares in `train.csv`: NOREL 63.6%, SYN 22.5%, HYPER 6.9%, ANT 3.6%, HYPO 3.5%. The test set has the same construction.

Evaluation

Submissions are scored in [0, 1], higher is better. A perfect submission scores 1.0 and the constant `SYN` sample scores about 0.14.

score = 0.15 × D + 0.85 × T × C

* Row weight w — 2 if the pair has no lexical overlap, otherwise 1. A pair has lexical overlap when the candidate's headword appears in the query's `definitions` text, or the query's headword appears in the candidate's. The test is: both strings lowercased, accents removed (Unicode NFKD, combining marks dropped), whole-word match. About 88% of test rows have weight 2.
* D — weighted F1 of the binary decision "related vs NOREL". Positive means any of SYN, ANT, HYPER or HYPO.
* T — the unweighted mean of the four weighted per-class F1 scores for SYN, ANT, HYPER and HYPO. NOREL predictions count as misses for the gold class and as nothing else.
* C — coherence factor, C = 1 − 0.5 × v. Here v = violated checks / active checks, and v = 0 when no check is active. A violated check must also contain at least one wrong prediction, so the perfect submission always has C = 1.

Notation. r(X, Y) is the relation of Y with respect to X, and inverse swaps HYPER and HYPO and leaves SYN, ANT and NOREL unchanged. The forbidden table gives the relations that may not hold between X and Z, given a = r(X, Y) and b = r(Y, Z):

* a = SYN, b = SYN: ANT is forbidden
* a = SYN, b = ANT, or a = ANT, b = SYN: SYN is forbidden
* a = SYN, b = HYPER, or a = HYPER, b = SYN: HYPO and ANT are forbidden
* a = SYN, b = HYPO, or a = HYPO, b = SYN: HYPER and ANT are forbidden
* a = HYPER, b = HYPER: HYPO and ANT are forbidden
* a = HYPO, b = HYPO: HYPER and ANT are forbidden
* every other combination has no forbidden relation

The checks, all derivable from the public files:

* Inverse pair — test rows (Q, C) and (C, Q). Active unless both predictions are NOREL. Violated when the second prediction is not the inverse of the first.
* Shared-query triangle — test rows (Q, C1) and (Q, C2), where `train_relations.csv` holds an edge between C1 and C2 giving L = r(C1, C2). Active when both predictions are relations and (inverse of the prediction for (Q, C1), prediction for (Q, C2)) appears in the table. Violated when L is forbidden for that combination.
* Anchor triangle — test row (Q, C), where Q has anchor A with a = r(Q, A), and `train_relations.csv` holds an edge between A and C giving b = r(A, C). Active when the prediction is a relation and (a, b) appears in the table. Violated when the prediction is forbidden for (a, b).

Reference implementation of the D and T terms (C as described above):

```
import numpy as np

REL = ("SYN", "ANT", "HYPER", "HYPO")

def wf1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 0.0 if d <= 0 else 2 * tp / d

def evaluate(y_true, y_pred, weight, coherence=1.0):
    """y_true, y_pred: arrays of labels; weight: array of 1/2 row weights; coherence: C."""
    y_true, y_pred, w = np.asarray(y_true), np.asarray(y_pred), np.asarray(weight, float)
    valid = np.isin(y_pred, REL + ("NOREL",))
    g = y_true != "NOREL"
    p = np.where(valid, y_pred != "NOREL", ~g)          # invalid labels count as the wrong side
    D = wf1(w[g & p].sum(), w[~g & p].sum(), w[g & ~p].sum())
    share = np.where(g, 1 / 3, 1 / 4)                  # invalid rows: one FP spread over other classes
    T = np.mean([wf1(w[(y_true == c) & (y_pred == c)].sum(),
                     w[(y_true != c) & (y_pred == c)].sum() + (w * share)[~valid & (y_true != c)].sum(),
                     w[(y_true == c) & (y_pred != c)].sum()) for c in REL])
    return 0.15 * D + 0.85 * T * coherence
```

Submission

Submit one CSV with exactly two columns, `id` and `relation`, and exactly one row for each of the 66,980 test ids, in any order. Rows are matched by `id`. Example using real test ids:

```
id,relation
t00010fc118ae503,NOREL
t000386b18194ea6,SYN
t00040f647d7f676,HYPER
```

Requirements

* The `id` set must equal the `test.csv` id set exactly. A missing, duplicated or unknown id raises an error, and so does a missing, renamed or extra column.
* `relation` must be one of `SYN`, `ANT`, `HYPER`, `HYPO`, `NOREL`. Matching ignores case and surrounding spaces. The no-relation label is `NOREL`, not `NONE`, because common CSV readers turn the string `NONE` into a missing value.
* An empty, NaN or unrecognised `relation` is scored as wrong on every term. It is a miss for its gold class, the wrong side of the related-vs-NOREL decision, and one false positive spread evenly over the relation classes other than the gold class. A file in which every `relation` is unusable raises an error.
* The notebook must run end to end on one GPU (A10G 24 GB class, 64 GB RAM) in under 90 minutes, read only `./dataset/public/` and write `./working/submission.csv`.

What not to use

* No pretrained weights of any kind: no pretrained language models, sentence encoders, tokenizers trained elsewhere, or word vectors. Every learned component must be trained from scratch on the provided files.
* No external lexical data: no Wiktionary or dumps thesauri, or dictionaries. No network access at run time.
* Do not try to recover hidden relations by reverse-engineering the split, by probing the scorer, or from anything other than the public files.

Extra Modelling Information

The usual work on this topic are lexical-relation classification. Like this task, they label word pairs as synonym, antonym, hypernym or random. They differ in three ways. They are English word pairs scored with distributional vectors. Their pairs are split so that the same words recur across train and test, which lets a model memorise prototypical hypernyms. And each pair is judged in isolation.

This task works differently:

* Each word comes with its full dictionary definitions and nothing else.
* Whole stem families are held out, so no query word or close morphological relative was ever labelled.
* Every prediction is embedded in a partially revealed relation graph (anchors and the training graph), so the answer has to be composed by logical closure.
* The metric rewards coherence of the whole labelling across inverse pairs and triangles, not only per-pair accuracy.

Here pretrained models are banned, so the relational knowledge must be learned from scratch from the dictionary itself.
