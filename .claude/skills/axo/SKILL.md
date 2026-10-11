---
name: axo
description: >
  End-to-end workflow for Axo AI/ML challenges (the private Telegram
  group with an "AI Baseline" to beat), built on the trusted challenge-engine
  pack: select a challenge, lock the rules and data, ship a fast rule-compliant
  probe that beats chance as version 1, build the strong solution while it is
  scored, improve from scores and logs, finalize a deterministic submission,
  and run a postmortem (subcommands: select, solve, improve, finalize, audit,
  postmortem). Use this skill whenever the user mentions Axo,
  an Axo challenge, the AXO bot, "challenge-engine", "beat the
  baseline", a markdown with Domain/Level/Compute/Baseline fields, a
  `curl ... dataset.zip` line for a Kaggle notebook, submission credits, a
  leaderboard, status block or "Submission scored" block from the bot, a
  traceback or logs from a challenge run, or asks which challenge to pick,
  whether one is doable, or how to plan, solve, improve, finalize, check or
  submit one — even if they only paste the challenge post or the markdown
  without saying "Axo".
---

# Axo challenge workflow

## Source of truth

The challenge-engine pack the user sent on 2026-10-05 (`references/pack/`, condensed into `compliance-firewall.md`, `research-gate.md`, `postmortem.md`, `commands.md` and the phases below) is **more trusted than the older text of this skill**: where they disagree, follow the pack. What the pack does not cover (Axo's platform rules, credits, closing time, packaging, the checking scripts) stays as written. The challenge description and a live bot message outrank both.

| Older rule | Now |
|---|---|
| (Pack) Start with the strongest credible approach; no baseline first | **Not adopted for version 1**, by the user's decision (2026-10-05, after Tray Elytra Metrology scored 9.43 against a local estimate of 54): version 1 is a fast probe that follows every rule and beats chance; the pack's strong solution is version 2, built while the probe is scored (Phase 1) |
| (Pack) Do not train locally; the user runs on Kaggle and sends logs | **Not adopted**, by the user's decision (2026-10-05): the agent is the primary runner and trains, validates, checks and packages on the user's PC; the user only uploads the finished folder and reports the board score and status block |
| Search for a missing resource or dataset, fall back, pick the device at run time | One fixed plan: fixed device, fixed paths, no discovery or fallback branches |
| Determinism proven on every version as part of the build | Fixed plan written in from the first file; the finalize phase freezes the winner into a separate deterministic file and proves it |
| Every close-out appends a lesson | Postmortem: challenge facts kept apart from reusable principles, at most 3 lessons, only with evidence |
| Kaggle fallback path `/kaggle/input/datasets/<owner>/...` | `/kaggle/working/extracted_dataset` (the user's own pattern) |

## Goal, in this order

1. Maximize expected hidden-test performance.
2. Follow the full challenge description and the platform rules exactly. A rule is never traded for score; when it is ambiguous take the stricter reading and tell the user. The rules are the ones in the markdown of the challenge in hand: each challenge has its own, and nothing carries over from another. A score from the bot is not approval: people review the code afterwards (The Enclosure Mention's 0.835 was scored, then rejected for a cross-row inference step its markdown had ruled out; `references/compliance-firewall.md`).
3. Protect against leakage, public-leaderboard overfitting and brittle hacks.
4. Improve the current best solution iteratively instead of rewriting blindly; protect the best.
5. Research only when the evidence justifies it.
6. Finalize determinism after the best valid solution is selected.
7. Save tokens by removing repeated work, not by weakening the main solver.

Axo adds: capacity is 10 solvers, 6 credits per challenge, the board keeps a solver's best version, and the live bot defines grace, lockdown and top-10 admission; an above-baseline version alone does not secure continued eligibility. A scored probe comes before any strong version (Phase 1): its board score is the only check that local numbers mean anything on the platform. A late "improvement" that is inside the noise is a lottery ticket and is named as one (`lessons.md`).

**Continue improving the work smartly.** Judge each next step by its likely contribution to reaching and retaining admission, and by its complete delivery cost. Re-rank after material board updates; test a different model's complementary value where rules and time allow; freeze from timings measured under the actual deterministic runtime plan. A real local improvement can still be too small or too late for admission. The decision procedure is in `references/climb.md`, "Continue improving the work smartly".

## Who runs what

You are the primary runner: you train, validate, check determinism and build the package on the user's PC (`scripts/preflight.py`, `assets/dev_harness.py`, `scripts/run_queue.py`, `scripts/check_determinism.py`, `scripts/gate.py`). The user never runs code on Kaggle or in a notebook and sends results back; they upload the finished folder to Axo and report the board score and the status block. Facts about the platform's runtime therefore come from Axo's own scored results and messages (`target-env.md`), not from probes the user runs.

## Workflow and commands

`solve` (build the probe here and hand it over, then start the strong version) -> the user uploads and reports the board score -> `improve` -> run here -> ... -> choose winner -> `finalize` -> deterministic submission -> `postmortem`; plus `select` (choose a challenge) and `audit`. The event router, the prompts and the golden rules are in `references/commands.md`. Research is never a first step.

## Reference files

- `references/commands.md` — event router, prompts for each situation, golden rules. Read when a score, error or instruction arrives. It holds the user's standing "continue improving smartly" instruction (biggest step first, five steps).
- `references/compliance-firewall.md` — leakage patterns, six questions, audit pass. Read at intake and before every submission.
- `references/research-gate.md` — when and how to research: a short methods-only pass right after the probe, deeper at a plateau. Read when the probe is handed over.
- `references/postmortem.md` — close-out and lesson promotion. Read when a challenge ends.
- `references/pack/` — the trusted originals (pack, prompts). Read when a detail matters or to settle a conflict.
- `references/platform-rules.md` — Axo's own rules: capacity, closing, credits, what to submit, script requirements. Read at intake.
- `references/rule-checklist.md` — the checklist template; becomes `RULES.md`. Read at intake.
- `references/recipes.md` — the default first approach per task type, with what was measured. Read when choosing the approach.
- `references/target-env.md` — what the platform's runtime contains and cannot give; confirmed against stated. Read before choosing the approach.
- `references/determinism.md` — fixed plan, reproducible output, finalization and its retry protocol. Read before writing training code and at `finalize`.
- `references/climb.md` — biggest step first (the order of work after the probe), diagnosis order, evidence labels, protect-the-best, calibration, experiments, stop-loss. Read when the probe is handed over and again when the first score arrives.
- `references/fast-v1.md` — the probe (version 1): its 20-minute clock, sizing, what is deferred, the gate and hand-over; picking a challenge (fit to this PC). Read at the start.
- `references/submission-package.md` — the upload folder and checks. Read before the first submission.
- `references/lessons.md` — past mistakes and the habit that prevents each. Skim at intake.
- `assets/solution_template.py`, `assets/dev_harness.py` — solution skeleton (fixed plan, argv pattern); fold runner.
- `scripts/` — `preflight.py`, `env_probe.py`, `gate.py`, `run_queue.py`, `check_determinism.py`, `validate_submission.py`, `build_submission_folder.py` (static source checks included).

## Working folder

```
<challenge-name>/
  <challenge>.md            the description, unmodified
  data/                     the extracted public files
  solution_dev.py           development version (validation loops, diagnostics)
  solution_v1.py ...        candidate versions; a measured candidate is never edited
  solution.py               the finalized, deterministic submission file built from the chosen candidate
  SOLUTION_DESCRIPTION.md   separate description for the reviewers
  RULES.md                  the filled rule checklist
  NOTES.md                  clock, spent time, decisions, the BEST line, every measured result
  out/                      logs, saved predictions, gate work files
  submission/               the ready-to-upload folder
```

`NOTES.md` keeps one line per version (version, score, key changes, run time, validation behaviour, known failure modes) and a `BEST:` line naming the protected version. A change that is not measured yet is a candidate, never the best.

## Deadline feasibility and admission cutoff

Before committing to a user deadline, budget selection/repair, two full exact-script runs, final checks, packaging and manual upload from measured timings. A promised handover must leave upload time. If the critical path cannot fit, say so immediately; do not spend the remaining window on an unmeasured model or imply that a new checked version is assured. Set a hard freeze point and report the actual result before the deadline, even when no supported improvement exists. Preserve all compliance checks.

Beating the baseline and retaining submission eligibility are separate objectives. In the recorded grace/lockdown workflow, the admission target is the current top-10 cutoff, with uncertainty as other scores arrive. Record baseline margin, top-10 margin and the freshest cutoff information. A forwarded countdown may already be old; do not treat its receipt time as a verified start of the remaining interval. The live bot remains authoritative.

Protect the best files and official score, but do not restrict candidates to that model family. The user explicitly permits different models when justified by rules, measurements and the remaining submission budget. Stop repeated noise-level extensions before they consume the packaging window.

## Before picking a challenge

Rank candidates first by fit to this PC (`references/fast-v1.md`, `scripts/preflight.py`): prefer those that give a held-out score within about 30 minutes and experiments of a few minutes, and drop those that need large models or hours of training to learn whether an approach works. Then by whether the strongest credible approach fits the challenge's declared compute (the `Compute` header) and runtime, by dependence on resources the platform may not give (`target-env.md`), and by live opportunity (baseline, solvers above it, submissions, top score, deadline, credits). Ask for the status block of every candidate. Say the pick, the recipe, the expected time per experiment on this PC, what was dropped and why, why the others rank lower, and what is uncertain.

## Phase 0 — Rule and data lock

- Read the whole challenge markdown and `references/platform-rules.md`; inspect the actual files. Download links expire hourly and are per challenge: compare file names, row counts and columns with the description; a mismatch means the wrong link, stop and ask.
- Extract: task and target; metric and direction; train/test/evaluation structure; allowed and forbidden information; compute and runtime constraints; model restrictions; package and environment constraints; submission format; reproducibility requirements.
- Write `RULES.md` from `references/rule-checklist.md`: one row per constraint **quoted verbatim**; tighter targets in the "our margin" column and named as ours to the user; the metric as an exact formula with edge cases; the baseline or **Not announced**; the description's approach hints as starting hypotheses, not proof of how the baseline was built.
- Run the evaluation-integrity firewall and the six questions (`compliance-firewall.md`); record the answers.
- Write the clock at the top of `NOTES.md`: start, closing time if known, solver count, credits. Ask what only the user can answer in one batch (closing time and status block, ambiguous rules with your default, spent time) and keep working on the stricter reading meanwhile.
- Run `scripts/preflight.py` (power, GPU, other jobs, libraries) and use an existing environment before any install; never modify the user's environments.

## Phase 1 — Probe first (version 1), then the strong solution (version 2)

**Version 1 is a probe**: the fastest logical solution that follows every rule, passes the full gate and scores above chance on one hold-out. It does not have to beat the baseline and it is not sized for score. Target (ours): the upload folder ready within 20 minutes of the data being on disk, one full run of at most 2 minutes (`references/fast-v1.md`).

The probe buys a real board score while there is still time to use it. That score checks three things no local run can: that our copy of the metric agrees with the platform's, that the test answers behave like the training answers, and that the script and its output file behave on the platform as they do here. Measured: Tray Elytra Metrology (2026-10-05) went straight to a strong first version. It took 2.5 hours, validated at about 54 locally and scored 9.43 against a baseline of 18.98, and the challenge closed before the gap could be traced with further submissions.

- Take the description's own method or the recipe default (`references/recipes.md`) at its default setting: one model, one seed, no settings grid, and no part that rests on a new finding about the data.
- A request for a high first score does not change the order. Say that the probe goes first and why, and that the strong version follows as version 2.
- At or below chance the pipeline is broken (a wrong mapping, a faulty hold-out). Fix that before uploading: such an upload costs a credit and teaches nothing.
- Until the probe's board score exists, report every local number as "local, not yet checked against the board" and promise nothing about the baseline or a rank.

**Version 2 is the strong solution**, started as soon as the probe is handed over, not when its score arrives: the strongest credible approach that fits the rules and runtime, built as one coherent plan. Its first half hour is the look at raw examples, the ceiling measurements and the short methods-only research pass (`references/research-gate.md`); candidate model families are then tested in one parallel wave, never one after another, and a candidate that shares 97% or more of its test predictions with an uploaded version is not built (`references/climb.md`). When the probe's score arrives it decides what happens to this work (`references/climb.md`): board close to local, continue; board far from local, stop building and find the cause first.

Every version, the probe included:

- uses only legal information and the runtime the challenge describes (`target-env.md`: libraries yes, pretrained weights and network no unless provided);
- runs when pasted into one notebook cell; the argv pattern is tolerated and never required;
- reads the data from the provided directory structure and does not depend on local packages, files or a local GPU;
- has a simple configuration block at the top and logs progress with an estimated completion time;
- keeps its split logic deterministic and entity-aware when the task groups rows, with validation that reflects the hidden-test structure (`climb.md`);
- has no dynamic model, device, cache or fallback logic (`determinism.md`): one fixed hardware plan;
- keeps training practical: the stated runtime is about one hour on the configured environment, or the challenge's own stricter limit; our own shorter targets are named as ours.

Build and validate each version here and hand the folder over before any note-keeping.

## Phase 2 — Score-driven improvement (`improve`)

**Biggest step first (the user's decision, 2026-10-08, after Anchored Layout Alignment was lost).** After the probe, version 2 is not the next logical refinement of the probe. It is the single change with the largest expected jump, found and built as early as possible. Places are rank-limited: on Anchored Layout Alignment three careful versions moved 0.239 -> 0.345 -> 0.338 on the board while the tenth place rose to 0.39, and the challenge was over before the approach was questioned. The order of work after the probe is in `references/climb.md`, "Biggest step first": look at raw examples and measure upper bounds before adding anything, rank ideas by expected gain per hour, and leave every idea worth less than about a tenth of the gap to the tenth place until the place is safe.

When a score, logs, a traceback or failure examples arrive (`references/climb.md`, `references/commands.md`):

1. Record the score; calibrate it against local validation (three numbers: board minus local, the board's own noise, the distance to the baseline and the top). If the board is far from local, by more than the fold-to-fold spread, stop building and find the cause before anything else.
2. Diagnose in order: runtime correctness, data and split correctness, leakage and compliance, validation quality, optimisation and stability, architecture and features, post-processing. Do not change the architecture while the real problem is upstream.
3. Label every proposed change CONFIRMED, STRONG HYPOTHESIS or SPECULATIVE; prefer confirmed over speculative.
4. Take the highest-impact change first (`climb.md`, "Biggest step first"): the one that can close the largest part of the gap to the top ten, even when it replaces the model. Small additions to a working model come after the place is safe. Keep what already works; deliver the revised solution as a new candidate version.
5. Protect the best: never replace a higher-scoring valid solution with a lower one because the new method is nicer. A worse result is discarded, analysed and recorded, not kept.
6. Research only through the gate (`research-gate.md`): a short methods-only pass right after the probe (the user's instruction, 2026-10-09), a deeper one at a plateau.
7. Submit an increment only when its gain exceeds the measured noise (second seed, or grouped K-fold over all rows); otherwise say it is a lottery ticket. At 8/10 solvers or on "now": hand over the best measured candidate first. Stop-loss and board watching are in `climb.md`.

## Phase 3 — Runtime error repair

For `RuntimeError`, CUDA, multiprocessing, device, memory, package or environment failures, and for the bot's "Your solution failed with <Exception>": read the traceback before touching the algorithm; decide from the rules whether the challenge is CPU-only, GPU, single or multi-device; keep no CUDA code in a CPU-only challenge; do not fall back silently from GPU to CPU in the final submission; avoid try/except chains that change device count, model, workers, scaler, precision or path; fix the smallest cause and keep the winning algorithm (`commands.md`).

## Phase 4 — Deterministic finalization (`finalize`)

Only after the best valid solution is selected. Freeze it, create the separate deterministic file `solution.py`, remove runtime-dependent execution paths, and prove it with `scripts/check_determinism.py` / `scripts/gate.py`. Follow `references/determinism.md` (freeze list, bad patterns, explicit plan, retry protocol). Then audit (`compliance-firewall.md`), build the package (`submission-package.md`) and hand it over with no caveats. Warnings are secondary; submission-blocking errors and rule violations are primary.

## Rules that hold in every phase

**Deterministic plan.** A fixed number of passes, no early or time-based stopping, one fixed device plan, fixed seeds, the same file on every run — whether or not the challenge asks for it. `DIFFERENT` on two runs is a failed rule, fixed before anything else. Every solution, the probe included, starts from the deterministic skeleton of `references/determinism.md`, section 8 (the user's instruction, 2026-10-08: determinism must never go wrong again): assigned cuBLAS setting, all seven backend lines, a literal device, nothing iterated in set order, no hardware-reading calls, and a "Deterministic runtime plan" section in the docstring. The gate enforces it and runs its two runs under different Python hash seeds. In a From Scratch or GPU challenge the probe is already a randomly initialised PyTorch model on the stated device.

**Solution script contract.** The submitted script uses the `public_dir` / `submission_out` argv pattern with one spelling and the fallback paths `/kaggle/working/extracted_dataset` and `/kaggle/working/submission.csv` (argv tolerated, never required); runs end to end from the public directory alone; uses one fixed hardware plan (GPU challenge: one fixed `cuda:0`; CPU challenge: `cpu`; several GPUs only when the rules allow and the plan stays fixed); takes no decision from CUDA availability, GPU or CPU count, `CUDA_VISIBLE_DEVICES`, files found under `/kaggle/input`, or an optional import, and has no out-of-memory retry; writes a safety file at start and replaces it in a final inference-and-CSV stage (a safety net, never a prediction method); logs progress, an ETA and wall time per stage; imports only libraries that `references/target-env.md` lists as present; explains its reasoning in a top docstring and comments; contains no download or unzip code, no experiments and no development switches. `scripts/build_submission_folder.py` flags the run-time branching patterns; a line that is genuinely a fixed plan carries `# fixed-plan-ok: <reason>`.

**Submission folder.** Every submission is one folder built by script:

```
submission/
  1_submission.csv             prediction file                   -> upload
  2_solution.py                well-documented solution code     -> upload
  3_SOLUTION_DESCRIPTION.md    separate solution description     -> upload / paste
  HOW_TO_SUBMIT.md             the user's guide                  -> not uploaded
```

Always both a documented solution and a separate description. `HOW_TO_SUBMIT.md` is for the user: which file at which prompt, the challenge name, the spent time as the exact text to type (the tracked time by default; the user may change it). The folder is handed over **with no caveats**: if anything is unresolved — a placeholder path, development code, an unchecked rule — fix it or ask the one question that settles it before building. Never assemble the folder by hand around a failing check. Record each submission (version, time, score, key changes) in `NOTES.md`.

## Phase 5 — Postmortem

When the challenge ends or the user says it is finished: `references/postmortem.md`. Record the final result and what worked and failed (list A), promote at most 3 evidence-backed reusable lessons (list B), update the recipe that was used, and do what the user asked with the files (archive or remove). Do not rewrite this skill after one noisy result.

## Operating style and reporting

- Think deeply, keep the user-facing reply compact; do not narrate small checks; implementation first when a file is asked for.
- Lead with the number that matters (local estimate against baseline against leaderboard), then what it rests on. Say what has and has not been run; an untested script is described as untested.
- A local number is not a board estimate until one scored version has shown that the two agree. Before that, say "local, not yet checked against the board".
- Give times and limits as the document states them; name our own targets as ours. Do not claim "best possible" without evidence.
- When behind other solvers, say so, what is left to try and what you expect from it. Short updates; a status question gets the current state in a few lines.
