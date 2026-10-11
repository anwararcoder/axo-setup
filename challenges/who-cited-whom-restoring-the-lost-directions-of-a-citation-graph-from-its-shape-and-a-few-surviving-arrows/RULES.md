# RULES — Who Cited Whom?

Source: `description.md` (the challenge markdown, unmodified) and `.claude/skills/axo/references/platform-rules.md`.
Every rule quoted verbatim. "Our margin" entries are our own targets, not the rule.

| # | Source | Rule (verbatim) | How we comply | Our margin | How it is checked | Status |
|---|---|---|---|---|---|---|
| 1 | Challenge › Rules | "Only the five released files may be used: no other citation data, no pretrained weights, and no attempt to find out which papers these are." | Reads only the five CSVs from `public_dir`; no model weights loaded; no lookup of papers | | grep for file reads other than the five names; no `load`/`from_pretrained`/network | pass (2026-10-11, gate v1 + code read) |
| 2 | Challenge › Rules | "Everything a solution learns or tunes (weights, settings) comes from the training graphs alone." | Model fitted on training graphs only; every setting chosen on training-graph folds; no fitting of any kind on `test_known.csv` | Strict reading: no per-test-graph fitting of weights either (surviving directions are features/constraints only) | read code: `fit` sees only train graphs; test read after training | pass (2026-10-11, gate v1 + code read) |
| 3 | Challenge › Rules | "The surviving directions of a test graph are inputs for that graph only: the answer for a test graph is computed from its own links and its own surviving directions, and may not depend on any other test graph." | Prediction loops over test graphs one at a time; each graph's features use only its own links and own known directions; no statistic across test graphs | No normalisation across test rows; per-graph normalisation only within the graph | read the inference loop against this sentence; change one test graph's input and confirm others' outputs unchanged | pass (2026-10-11, gate v1 + code read) |
| 4 | Challenge › Dataset | "`paper_a` and `paper_b` are numbers from 1 to the number of papers of the graph; they were given out at random and mean something only inside their graph. Which paper of a link is listed first was decided by a fair coin." / "`graph_id` names the graph and `link_id` the link; both are random codes, and the rows are sorted by `link_id`." | No feature from paper numbers, ids, row order or list position; features are symmetric functions of the structure (a/b swap averaged) | | grep features for `link_id`, `paper_a` used as value | pass (2026-10-11, gate v1 + code read) |
| 5 | Challenge › Evaluation | "The Matthews correlation coefficient between the true and the submitted value of `a_cites_b`, over the links to answer of all test graphs taken together (the value `sklearn.metrics.matthews_corrcoef` returns)." | Local metric = `sklearn.metrics.matthews_corrcoef` on pooled held-out links | | dev script uses it directly | pass (2026-10-11, gate v1 + code read) |
| 6 | Challenge › Evaluation | "A file with a missing or repeated `link_id`, or with a value other than 0 or 1, is rejected with an error." | Output built from the sample's ids, values cast to int 0/1 | | `validate_submission.py` + `--regex a_cites_b '[01]'` in the gate | pass (2026-10-11, gate v1 + code read) |
| 7 | Challenge › Submission | "A CSV file with a header row (`link_id`, `a_cites_b`) and one row for each of the 25,257 links of `test.csv`, in any order, where `a_cites_b` is 0 or 1." | Same header and ids as `sample_submission.csv` | | gate stage 4 | pass (2026-10-11, gate v1 + code read) |
| 8 | Challenge › header | "Compute: CPU" | One fixed CPU plan, no torch/CUDA, fixed thread count | | builder static check; no `cuda` in file | pass (2026-10-11, gate v1 + code read) |
| 9 | Platform › Script | "only standard Kaggle Docker libraries" | numpy, pandas, scikit-learn, threadpoolctl (confirmed on the platform, `target-env.md`) | No LightGBM (not confirmed on the platform) | import list printed by gate stage 1 | pass (2026-10-11, gate v1 + code read) |
| 10 | Platform › Script | "targets about 1 hour of runtime on the configured environment" | Probe sized at about 2 minutes | ours: probe ≤ 2 min, later versions ≤ 20 min | gate stage 3 timing | pass (2026-10-11, gate v1 + code read) |
| 11 | Platform › Script | public_dir / submission_out argv pattern, "use one variable name consistently", no dataset download or archive extraction | Template pattern, fallback `/kaggle/working/extracted_dataset` and `/kaggle/working/submission.csv` | | gate stage 1 | pass (2026-10-11, gate v1 + code read) |
| 12 | Platform (bot, 2026-10-10) | The bot refuses a solution file that contains the platform name in any form | Name absent from code and description | | gate stage 1 | pass (2026-10-11, gate v1 + code read) |
| 13 | Platform (user's workspace notes) | Approach text "at most 1500 characters, enforced by the bot" | Description file kept under 1,500 characters | | `wc -c` on the description | pass (2026-10-11, gate v1 + code read) |
| 14 | Platform › Determinism | "a fixed, deterministic training and inference plan" (checked by the platform) | Fixed seeds, fixed iterations, no early stopping, fixed thread scope, sorted iteration | | gate stage 3 (two runs, two hash seeds) → IDENTICAL | pass (2026-10-11, gate v1 + code read) |
| 15 | Platform › Confidentiality | Private data never shared or published | Data stays in this folder | | — | pass (by construction) |

## Metric

MCC over all answered test links pooled: `matthews_corrcoef(y_true, y_pred)`. Constant prediction → 0.0 (sklearn returns 0
when a column is constant). **Chance level: 0.0.** With balanced labels (coin-flip a/b order) MCC ≈ 2·accuracy − 1, so the
baseline 0.5478 ≈ 77.4 % of links right.

**Baseline: 0.5478.**

## Approach hints in the description (not rules)

- "What is left to go on is the shape of the graph and the few directions that survive."
- "the answer for a test graph is one orientation of the whole graph, written out link by link."
- "A citation points from one paper to another, usually from the newer to the older." (a time order: the true orientation is acyclic if it holds strictly)
- Related work named: MagNet (direction prediction on one graph), direction recovery from topology, network archaeology (node-arrival order from one snapshot, preferential attachment).

## Six questions (compliance-firewall.md)

1. Rows for training: the training graphs (`train.csv` + `train_labels.csv`) only.
2. At inference: a test graph's own links (`test.csv` + `test_known.csv` rows of that `graph_id`) and its own surviving directions.
3. Can a feature depend on another evaluation example? Yes within the same test graph (the rule allows the graph's own links); never across test graphs.
4. Can the model or preprocessing see the target population before prediction? No: nothing is fitted on test graphs; per-graph features are computed inside the per-graph loop.
5. Generated data forbidden? Not stated. We simulate the 10 % surviving-direction mask on training graphs (a resampling of training labels, not new labelled examples); stated in the description.
6. External annotation source? None used.
