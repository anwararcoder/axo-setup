# Version 1 is a probe: clock, sizing, gate; picking a challenge

Version 1 is a **probe**: the fastest logical solution that follows every rule, passes the full gate and scores above chance on one hold-out. It does not have to beat the baseline, and it is not the place for the strongest approach. The strong solution is version 2, started the moment the probe is handed over.

What the probe buys is a real board score while slots are still open and credits remain. That score checks three things no local run can:

- that our copy of the metric agrees with the platform's;
- that the test answers behave like the training answers;
- that the script and its output file behave on the platform as they do here.

Why this order is fixed (measured):

- **Tray Elytra Metrology, 2026-10-05.** The user asked for more than 50 on the first submission, so the probe was skipped and a strong version was built first. It took 2.5 hours, validated at about 54 locally (four held-out folds: 40, 61, 54, 61) and scored 9.43 against a baseline of 18.98. The challenge closed before the gap could be traced with further submissions. A probe in the first half hour would have shown the gap with five credits and the whole evening left.
- **Soundtrack Shortlists.** The board went from 6/10 to closed in about 40 minutes; local validation read 0.02 to 0.03 above the board.

This rule was replaced on 2026-10-05 by "the first solution is the strongest credible approach" (the challenge-engine pack) and restored the same day by the user after the result above. A request for a high first score does not change the order: say that the probe goes first, why, and that the strong version follows as version 2.

What the probe does not buy (`platform-rules.md`): a place or continued eligibility. The latest recorded bot workflow opens grace to everyone, then locks submissions while selecting the top10; above-baseline status alone does not secure admission. The board keeps a solver's best version, and reviewers accept below-baseline calibration attempts. Use the current bot cutoff and top-10 margin.

You run everything here (the user's decision, 2026-10-05): training, validation, the determinism proof and the gate all happen on the user's PC.

## Contents
- The 20-minute probe clock
- What the probe is, and what it is not
- Sizing the plan
- One batch of questions
- Working in parallel
- When a stage overruns
- The gate and the hand-over
- After the hand-over: version 2
- Picking a challenge

## The 20-minute probe clock

It starts when the data is on disk. Write the start time at the top of `NOTES.md`. The boxes are our targets, never quoted as the platform's.

| Stage | Box | Done when |
|---|---|---|
| Intake | 6 min | `RULES.md` has every rule quoted, the metric as a formula, the baseline, the chance level, and the description's own approach hints |
| Data and machine check | 2 min | Files and columns match the description; `scripts/preflight.py` says READY |
| Pick a recipe | 1 min | One line in `NOTES.md`: which recipe, why |
| Build and one hold-out | 6 min | `solution.py` runs end to end; the metric reproduces the stated anchors; one hold-out score is above chance |
| Gate and package | 5 min | `scripts/gate.py --probe --chance <value>` has built `submission/` |

Measured: in the soundtrack challenge the folder was built 10 minutes after the data was on disk.

If the data makes 20 minutes impossible (very large images, a long mandatory training), say so at intake with the time you expect instead, and keep the same order of work. A probe that needs an hour is still sent before any strong version is built.

## What the probe is, and what it is not

**In:** the method the description itself points to, or the recipe default, at its default setting; one model and one seed; augmentation only where the effect on the label is certain; one hold-out; the determinism settings from the first line of training code.

**Not in, these belong to later versions:** any settings grid; any fix aimed at the score; several models or seeds; multi-fold validation; a new architecture; any part that rests on a new finding about the data (an oddity in the labels, a quirk of the files). Such a part goes in alone, in its own later submission, so the board shows what it is worth.

**Never relaxed for a probe:** the quoted rules, the fixed deterministic plan, the two byte-identical runs, the format check, the documented script and the separate description. A probe is early, not sloppy.

**The chance level** is the metric's value for a random or constant prediction, worked out at intake. It is the probe's bar and goes to the gate as `--chance`. At or below chance, do not upload: the pipeline is broken (a label mapping, the hold-out design, the metric). Fix that, not the model.

## Sizing the plan

Size the probe so that **one full run takes at most 2 minutes**. The reproducibility proof needs two full runs, so this keeps the whole gate under 5 minutes. Later versions may take up to about 20 minutes per run (`climb.md`).

- The template prints the projected time per model after the first pass. Set the number of passes from it.
- Prefer cutting passes or model width over cutting data.
- If the smallest plan that scores above chance needs longer than 2 minutes, use it and tell the user the gate will take twice that.
- The challenge's own runtime limit is far above this; these sizes are our choice for speed, never quoted as the rule.

## One batch of questions

Collect what only the user can answer and ask once, with up to four questions: the closing time and the current status block; any rule with two readings (state your stricter default); spent time (default: the tracked time in `NOTES.md`). Ask at a moment when waiting costs nothing, right after the first training run has started, and keep working on the stricter reading meanwhile.

## Working in parallel

While a run goes, fill `RULES.md`'s "how it is checked" column and draft `SOLUTION_DESCRIPTION.md`. While the gate runs its two full runs, finish the description's validation section; when the measured run time is printed put it in the Runtime section. Start nothing else heavy on the machine while the gate's timed runs go: side experiments slow them and move the hand-over.

## When a stage overruns

If a stage is half over its box, cut scope (fewer passes, a narrower model, a simpler hold-out) instead of extending it, and note the cut in `NOTES.md`. Do not start investigating why something is slightly worse than hoped. Two things are never cut: quoting the rules and the checks.

## The gate and the hand-over

- Run `scripts/gate.py` with `--probe --chance <value>`: the chance comparison replaces the baseline comparison and nothing else is relaxed. Later versions run without `--probe` and must be above the baseline locally or be handed over with that said plainly.
- Hand over in a few lines before any note-keeping: where the folder is, that it is a probe, the local score against chance and against the baseline, what the score rests on (how many held-out items), and the request for the status block and the score.
- Call the local score "local, not yet checked against the board". Promise nothing about the baseline or a rank.
- When you give an arrival time, include the two full runs of the gate.

## After the hand-over: version 2

Start the strong version at once; do not wait for the probe's score. "Strong" means the largest expected jump, not the probe plus refinements: begin with the 30-minute look at raw examples and the ceiling measurements of `climb.md`, "Biggest step first", plus the short methods-only research pass of `research-gate.md` (the user's instruction, 2026-10-09), and build the top item of that ranked list. When the score arrives, read it before anything else (`climb.md`):

- **Board close to local** (inside the fold-to-fold spread and the board's noise): local validation can be trusted; continue with version 2.
- **Board far from local:** stop building. Keep version 2 unsubmitted and find the cause first, starting with what local runs cannot show: the platform's run of the script, our copy of the metric, and whether the test answers behave like the training answers.

A second clock runs from the same start: the first version that beats the baseline locally, target 60 minutes. If nothing is above the baseline locally by then, tell the user the numbers (local, the probe's board score, baseline) and let them choose between continuing and switching challenge.

## Picking a challenge

**First filter: how fast can this PC show whether an approach works?** Rank by time to the first honest signal before anything else: prefer challenges that run fast enough on the current machine, not ones that need large models and hours of training just to find out whether an approach is good or bad. The PC: RTX 3060 Laptop (6 GB, sometimes held by another job), 16 GB RAM usually nearly full, 12 CPU cores, a CUDA PyTorch environment with scikit-learn and the gradient-boosting libraries but no `transformers`. Run `scripts/preflight.py` for the state right now. Also check that a credible approach fits the challenge's declared compute (`Compute: A10G` or `CPU`) and runtime, and that it does not depend on resources the platform may not give (`target-env.md`: libraries yes, pretrained weights and network no unless provided).

Estimate it in five minutes from the markdown and the archive size (`curl -sIL <dataset link>` returns the size without downloading):

| Signal | Reading for this PC |
|---|---|
| `Compute: CPU`; tabular, text or small arrays; thousands to tens of thousands of rows | Best: a held-out score in seconds to minutes |
| `Compute: A10G`, but the description's own method is small (up to about 60 M parameters from scratch) and one pass over the data takes under a minute on a 3060 | Fits; time the first pass before committing |
| A large pretrained model, a long fine-tune, a stated training budget of an hour or more, or images, video, audio or 3D data in the GBs | Skip: hours per experiment, and nothing pretrained is available on the platform |
| `Hard` with `multimodal` or `small-data` tags and no recipe here | Slow to learn anything; below the others |

An A10G label, a large archive or a multimodal tag alone does not rule a challenge out: check for an affordable legal method (fixed descriptors, sparse retrieval, a compact network, supplied weights). Targets (ours, not platform limits): the probe's folder within about 20 minutes of the data being on disk and no experiment longer than about 8 minutes.

**Second filter: the status block.** Ask for the status block of every remaining candidate (solvers who beat the AI, submissions, top score, closing time) before recommending one. Someone above the baseline is evidence the challenge is beatable by some approach, not by ours; many submissions with nobody above it suggest a plateau just under it; 8/10 or more means only a first package that fits the remaining window is worth starting.

Say the pick in a few lines: recipe, expected time per experiment on this PC, what was dropped and why, the main uncertainty. The user decides.
