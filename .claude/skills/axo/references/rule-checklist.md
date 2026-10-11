# Rule checklist

Fill this in during intake and keep it next to the solution as `RULES.md`. It is the gate before every submission: a row counts only when its check has actually been run on the exact files being submitted.

## How to fill it in

1. Go through the challenge markdown section by section (Overview, Dataset, Submission, Evaluation, Expected Approach, Methods/Rules, Compute Environment, Baselines) and through `platform-rules.md`.
2. Every sentence that constrains what may be done, used or output becomes one row. Small details count: column order, a file encoding, "exactly", "at most", "one row per id", a variable-name convention.
3. **Quote the rule verbatim.** Paraphrasing is where limits drift. If the document says 90 minutes, the row says 90 minutes.
4. Any tighter target chosen for safety goes in the "Our margin" column and is labelled as ours. Never present it to the user as the rule.
5. "How it is checked" must be something runnable or inspectable: a command, an assert in the script, a grep, a timed run. "Reviewed" alone is not a check.
6. Status is one of: `not checked`, `pass (date, what was run)`, `fail`.
7. Below the table, copy the description's own **approach hints** word for word (an "Expected Approach" section, or sentences that name a formulation). They are not rules, but they say what the baseline most likely does and decide the recipe for version 1.

Quote the rules at intake (10 minutes). Fill the "how we comply" and "how it is checked" columns while the first training run is going, not before it.

## Template

| # | Source | Rule (verbatim) | How we comply | Our margin | How it is checked | Status |
|---|---|---|---|---|---|---|
| 1 | Challenge › Compute | "…" | | | | not checked |
| 2 | Challenge › Methods | "…" | | | | not checked |
| 3 | Challenge › Submission | "…" | | | | not checked |
| 4 | Platform › Script | "only standard Kaggle Docker libraries" | | | list of imports vs. Kaggle image | not checked |

## Rows that apply to almost every challenge

Add these unless the document clearly makes one irrelevant.

**Output format**
- Exact header and column order.
- One row per test id; no missing, extra or duplicate ids. → `scripts/validate_submission.py`
- Encoding, value syntax (valid JSON, number ranges, non-empty fields), per-row limits.

**Methods**
- Allowed and forbidden model sources (pretrained weights, embeddings, named models).
- External data, external word lists, hosted inference.
- Use of test rows: no manual labelling, no lookup by id or row order, no hard-coded answers; whether statistics across test rows are allowed.
- Rules "learned from supplied examples" must be computed in code from the labels. → grep the script for hand-typed domain vocabulary.

**Evaluation integrity** (the firewall, `compliance-firewall.md`; answer the six questions there and record the answers)
- Only labelled training rows are training inputs; test rows are never used for fitting, selection, vocabulary, thresholds or calibration. → grep the script for any use of the test table before the final prediction pass.
- No test-to-test information flow: no pooled context, sibling or equivalent cases, retrieval among test cases or statistics across test rows, unless this challenge's markdown allows it; quote the sentence that settles it (The Enclosure Mention was scored and then rejected by human reviewers for one cross-row adjustment its markdown had ruled out). → read the inference code against question 3.
- No pseudo-labels, self-training or test-time augmentation unless allowed; no synthetic labelled data if prohibited.
- Everything fitted from labels (lexicons, thresholds, calibration, encoders, indexes) is fitted on training rows, inside each validation fold.
- Ids, row order, released group aliases and the public/private split are not signals.
- No external resource acts as an annotation source.

**Runtime and determinism** (always included, even if the document does not mention determinism; details and the four standard rows are in `determinism.md`)
- Hard time limit and hardware, as written.
- Fixed seeds and a fixed number of passes; no early stopping or time-based stopping. → grep for `patience`, `early_stop`, clock-based conditions.
- Deterministic kernels enabled, not just seeds. → setting present in the script, no `warn_only`.
- Two full runs, separate processes, byte-identical output. → `scripts/check_determinism.py` prints `IDENTICAL`.
- A valid submission exists before the limit. → fallback file written at start; kill the run mid-way and confirm.

**Script contract (platform)**
- `public_dir` / `submission_out` argv pattern, one consistent variable name, fallback paths `/kaggle/working/extracted_dataset` and `/kaggle/working/submission.csv`; runs when pasted into one notebook cell (arguments tolerated, never required).
- One fixed hardware plan and no run-time branching on device, paths, imports or retries. → the builder's static check; a `fixed-plan-ok` line has a stated reason.
- Runs end to end from a clean folder with only the public directory as input.
- No download/unzip steps, no leftover experiments, comments explain the reasoning.
- Standard Kaggle libraries only.

**Confidentiality**
- Private data is not shared or published anywhere.
