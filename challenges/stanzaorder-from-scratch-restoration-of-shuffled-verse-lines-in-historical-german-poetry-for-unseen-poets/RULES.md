# RULES — StanzaOrder (From Scratch, Medium, Compute A10G, baseline 0.4846)

| # | Source | Rule (verbatim) | How we comply | Our margin | How it is checked | Status |
|---|---|---|---|---|---|---|
| 1 | Challenge › Not Allowed | "**From scratch:** no pretrained weights of any kind (language models, embeddings), and no external text, poetry collections, rhyme or pronunciation dictionaries. Train only on the files provided." | Randomly initialised PyTorch net trained in the script on train.csv; rhyme by hand-written ending rules; first-word vocabulary counted from train.csv | — | gate `--forbid 'from_pretrained|torch\.hub|load_state_dict|urllib|requests'`; read imports | pass (2026-10-11, gate v1) |
| 2 | Challenge › Not Allowed | "No use of any edition of the poems, and no data from other challenges." | Only train.csv / test.csv / sample_submission.csv are read | — | grep `read_csv` / `open(` in the solution | pass (2026-10-11, read) |
| 3 | Challenge › Not Allowed | "Each test row must be predicted on its own: do not use other rows of `test.csv` (their inputs may contain material from the same source text)." | `stanza_features` uses only the stanza's own lines + training vocab; the DP solves each stanza independently (grouped by size only for vectorisation) | stated in code docstring and description | read `featurise`, `stanza_features`, `decode` | pass (2026-10-11, read) |
| 4 | Challenge › Objective | "For each test stanza, output its line labels in the printed order." | `order` = decoded permutation | — | gate stage 4 + assert permutation in script | pass (2026-10-11, gate v1) |
| 5 | Challenge › Submission Format | "CSV with exactly the columns `example_id,order`, in that order, one row per test `example_id`. `order` is a permutation of that stanza's line labels, separated by single spaces." | sample columns/ids copied; labels joined with one space | — | gate stage 4 `--regex order`; script asserts | pass (2026-10-11, gate v1) |
| 6 | Challenge › Evaluation | "row_score = #{k : pred_next(gk) = gk+1} / (n − 1)" … "A value that is not a permutation of that stanza's labels scores 0 for that stanza. The final score is the mean over all test stanzas." | dev_eval.py `row_scores` implements it | — | identity order on hold-out 0.2124; chance (test mix) 0.1845 = Σ share_n / n | pass (2026-10-11) |
| 7 | Challenge › Evaluation | "A submission is rejected with an explicit validation error (no score) if the columns are wrong or out of order, any ID is missing, duplicated or unknown, or any value is missing or empty." | as row 5 | — | gate stage 4 | pass (2026-10-11, gate v1) |
| 8 | Challenge › Split | "**Split unit: poet.**" / "every test poet is absent from training" | validation holds out whole poets (author_id folds) | — | dev_eval.py `author_folds` | pass (2026-10-11) |
| 9 | Challenge › Data | "The k-th element has label `Lk`; the label number carries no order information." | model is permutation-equivariant; label index never a feature | — | read features: no index feature | pass (2026-10-11, read) |
| 10 | Challenge › header | "Compute A10G" | DEVICE = "cuda:0" literal | — | static check | pass (2026-10-11, gate v1) |
| 11 | Platform › Script | "Targets about 1 hour of runtime on the configured environment" | ~0.5 min locally | ours: < 5 min | gate stage 3 timing | pass (2026-10-11, gate v1) |
| 12 | Platform › Script | "Only the standard libraries of the configured Kaggle Docker runtime." | numpy, pandas, torch, stdlib | — | imports read | pass (2026-10-11, read) |
| 13 | Platform › Script | public_dir / submission_out argv pattern, fallback paths | template pattern, one spelling | — | gate stage 1 | pass (2026-10-11, gate v1) |
| 14 | Platform › Script | "Comments explain the reasoning behind important steps." / no download, no extraction, no experiments | docstring + comments; no dev code in solution | — | gate stage 1 | pass (2026-10-11, gate v1) |
| 15 | Platform (bot 2026-10-10) | platform name must not appear in the solution file | not used | — | gate stage 1 | pass (2026-10-11, gate v1) |
| 16 | Determinism (ours, always) | fixed plan, two byte-identical runs under different hash seeds | skeleton of determinism.md §8 | — | gate stage 3 | pass (2026-10-11, gate v1) |

## Six questions (compliance firewall)

1. Training rows: train.csv only (all 10,374 stanzas in the submission; hold-out folds in development).
2. At inference: the stanza's own lines, the training vocabulary and the trained weights.
3. Does any feature depend on another evaluation example? No. `stanza_features` takes one stanza; the DP runs per stanza.
4. Does the model or preprocessing see the test population before prediction? No; test.csv is read after training.
5. Generated data: none (rules do not forbid it, but none is used).
6. External resource as annotation source: none; rhyme rules are hand-written string rules (allowed in spirit: "no … rhyme or pronunciation dictionaries" — we use none).

## Metric facts

- Chance (random order) on the test size mix: Σ_n share_n / n = 0.1845.
- Test size mix: 4:1103, 5:201, 6:683, 7:108, 8:626, 9:74, 10:190 (train is heavier on 4-line stanzas: 53 % vs 37 %), so local scores are also reported reweighted to the test mix.
- Board noise: per-row sd ≈ 0.37 → SE ≈ 0.37 / sqrt(2985) ≈ 0.007.

## Approach hints in the description (verbatim, not rules)

- "Rhyme schemes such as `abab`, `aabb`, `abba` and the Baroque `aabccb` tie specific lines together. Alternating masculine and feminine endings follow fixed patterns. Iambic and trochaic metre sets the syllable count of each line. Period punctuation marks the sense units: the virgule `/` within a line and the colon or full stop that closes a stanza."
- "the same rhyme can be spelled in several ways, and a model has to discover the sound patterns from the text itself."
- "Restoring the printed order means inferring which lines rhyme, which metrical and ending pattern the poet used, and where the syntax and the punctuation close."
