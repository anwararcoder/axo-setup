# Syllabus Holding Page-Span Grounding

- Domain: RAG
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: text, feature-engineering

---

Syllabus Holding Page-Span Grounding

Overview

Each item is one argued case of the U.S. Supreme Court, given as two linked documents:

* **The syllabus**: the reporter's summary of the decision. It states the holding (after `Held:`) and then the supporting points, usually lettered `(a)`, `(b)`, `(c)`.
* **The opinion of the Court**: the majority opinion, split into its printed pages, given as `pages_json`. Below, `n_pages` means the number of pages in `pages_json`, so page indices run from `0` to `n_pages - 1`.

In the published syllabus, each supporting point ends with a page citation telling the reader which opinion pages support it. For every case, one supporting point is the **target**. Its citation is replaced by the marker `[PP]`, and every other page citation in the syllabus is deleted. The target paragraph is also given on its own as `target_text`.

Predict the **inclusive interval of opinion pages** `[start, end]` that the target's removed citation pointed to.

This is **summary-to-source page grounding**: find where in a long opinion a summarized holding is argued. It is not summarization or syllabus generation, not SuperPAL-style proposition alignment (no sentence pairs are scored), not LePaRD/CLERC-style precedent retrieval across different decisions, not checking whether a citation is "on point", not picking one of k candidates, and not span extraction inside a single passage.

Format-only example, invented text, with `n_pages = 8`:

* `syllabus_text`: `Held: A permit fee is subject to the same test as any other condition. (a) The test asks whether the condition is tied to the project's impact. (b) Nothing in the text exempts legislative fees. [PP] (c) The county's remaining arguments were not raised below.`
* `target_text`: `(b) Nothing in the text exempts legislative fees.`
* `span_json`: `[5,7]`

Why this is hard

* An opinion runs 5–40 pages. The syllabus paraphrases it rather than quoting it, and the target is the supporting point whose wording singles out a page **least** clearly (see below).
* Section headings such as `I`, `II`, `A`, `B` are removed from the pages, so the opinion's own outline cannot be read off to find where a line of reasoning starts and stops.
* Only one point per case has a citation, so the answer cannot be inferred by fitting the other points' known page ranges around it.
* The answer is an interval. Half of the credit needs both edge pages exactly right. The other half is measured **relative to the answer "every page"**: covering the whole opinion scores exactly 0, and a wide hedge earns little.
* Test cases come from source volumes that never appear in training.

How items and splits were built

* **Target choice.** The candidates are a case's supporting points (not the `Held:` paragraph). A point is dropped if a trivial predictor already gets it nearly right, meaning any of the four position-only or single-feature predictors in the reference list below scores 0.70 or more on it. Among the remaining points, the target is the one whose best-matching opinion page stands out least from the second best by TF-IDF similarity: the point with no close lexical match. A case with no remaining point is dropped. The same rule is applied to every case, in train and in test.
* **Pages.** Lines that consist only of a section label, such as `I`, `II`, `A`, `1`, or a row of asterisks, are deleted. All other text is unchanged.
* **Split by source volume.** Each volume collects one period of the Court's decisions. Volumes are held out in a seeded draw stratified across three eras, so test spans the same decades as training. No volume is on both sides; the build asserts `volume_overlap = 0`.
* **Sizes.** **938** training items from **59** volumes. **320** test items from **27** volumes: a seeded random sample with no selection by difficulty, at most 18 per volume. One item per case. Near-duplicate syllabi (companion cases) are removed.
* `group_id` is an opaque key for the source volume. Use it with `GroupKFold` for honest validation. It is grouping metadata, not a feature, and test volumes have their own keys.

Input and output

Columns in `train.csv` and `test.csv`:

* `item_id` (string): opaque id that starts with `SPG_`
* `group_id` (string): opaque source-volume key, for grouped validation only
* `syllabus_text` (string): the full syllabus. The target's citation is replaced by `[PP]`, which occurs exactly once, and all other page citations are deleted.
* `target_text` (string): the target supporting point, which is the syllabus text that runs up to `[PP]`, starting after the previous point's deleted citation
* `pages_json` (string): a JSON array of strings, one per printed page of the opinion of the Court, in order. Its length is `n_pages`. Running heads, page numbers, and section-label lines are removed.

Training-only target:

* `span_json` (string): the target's page interval as a JSON array `[start, end]` of two integers

Text notes: text was extracted from the official printed pages. Footnotes stay on the page where they were printed. Words hyphenated across lines were rejoined where a dictionary confirmed the join, and dropped fi and fl ligatures were repaired with a dictionary pass. Some extraction noise remains, consistently across train and test. The first page begins with the line naming the opinion's author, and the last page ends where the Court's opinion ends.

Evaluation

Metric: **pin-cite score**, in `[0, 1]`, higher is better.

For each item, with gold interval `G` and predicted interval `P` (both treated as sets of page indices) in an opinion of `n` pages:

```
exact = 1.0 if P == G else 0.0                  # both edge pages right

dice  = 2 * len(P & G) / (len(P) + len(G))

base  = 2 * len(G) / (n + len(G))               # dice of the cover-everything answer [0, n-1]

skill = max(0.0, (dice - base) / (1 - base))

item_score  = (exact + skill) / 2           # half exact match, half overlap skill

final_score = mean(item_score over test items)  # clamped to [0, 1]
```

**Why this metric.** A page citation is useful only if it sends the reader to the right pages. Half the credit is for the exact interval, the citation as the reporter wrote it. The other half rewards near misses by page overlap, measured as skill over the trivial answer. Plain Dice or IoU would reward hedging, because predicting the whole opinion earns a lot of overlap on every wide citation. Subtracting the cover-everything baseline makes the whole opinion score exactly 0. An exact interval scores 1. One edge off by a page typically earns about 0.4.

Reference points, measured on a **validation split of the training data**: 11 training volumes (189 items) held out, everything fit on the other 48 volumes (749 items):

* Cover-all, as in `sample_submission.csv`: **0.000**
* Constant `[0, 0]`: **0.000**
* Uniform random interval (chance level): **0.067**
* Position only, even tiling of the syllabus points over the pages: **0.118**
* Position only, tiling of the pages after the opening ~15%: **0.169**
* TF-IDF best single page for `target_text`: **0.060**
* TF-IDF best contiguous window for `target_text`: **0.284**
* CPU reference (TF-IDF window plus a position prior, tuned by `GroupKFold`): **0.329**

With 320 test items, the standard error of a typical score is about **0.015**, so differences smaller than roughly **0.03** should not be read as a ranking.

Scoring joins on `item_id`. Row order does not matter.

Submission

Write `./working/submission.csv` with columns exactly:

`item_id,span_json`

* Header row required
* Exactly one row per test `item_id`, with no duplicates and no extra ids
* `span_json` is a JSON array of two integers `[start, end]` with `0 <= start <= end <= n_pages - 1`

Edge cases

* Single-page citations are intervals with `start == end`.
* The `Held:` paragraph and points that are not the target carry no marker and are not scored.
* A `span_json` that cannot be parsed, is not exactly two integers, has `start > end`, or falls outside `[0, n_pages - 1]` scores 0 for that item. Floats and numeric strings are not coerced.
* A missing column, a duplicate `item_id`, an empty value, or an `item_id` set that differs from `test.csv` raises an error.

Dataset

Public files under `./dataset/public/`: `train.csv`, `test.csv`, `sample_submission.csv`.

Source note: built from the U.S. Supreme Court's official published decisions, which are US public-domain government works. Target intervals were derived offline by mapping each removed in-syllabus page citation onto the opinion's page sequence. They are not an upstream metadata field. Use only `./dataset/public/`.

Environment

CPU, within the platform time limit. Packages are limited to the Kaggle Python Docker image. Resources already present in that image may be used. Runtime downloads, pretrained-weight downloads, and hosted inference APIs are not allowed. TF-IDF, gradient boosting, and dynamic programming over intervals all run comfortably on CPU at this data size.

What not to use

1. **Looking up the published syllabi or opinions.** That includes the United States Reports (print or PDF), the Supreme Court's website, and case-law archives or legal databases such as CourtListener, Justia, the Caselaw Access Project, Westlaw, or LexisNexis, in any form, to recover the removed page citations. The answer must come from `./dataset/public/` alone.
2. External datasets, APIs, model or weight downloads, or web scraping of any kind
3. Anything under `./dataset/private/`
4. Hardcoded `item_id` lookup tables
5. Packages beyond the Kaggle Python Docker image
6. `item_id`, `group_id`, or row order as prediction signals
