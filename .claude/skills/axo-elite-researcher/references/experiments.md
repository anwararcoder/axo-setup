# Experiments: pipeline, running on this PC, evidence, score loop

## Contents
1. The pipeline (solve)
2. Running things on this PC
3. Evidence and logging
4. The score loop (improve)

## 1. The pipeline (solve)

1. **Data.** After selection only. Download with the challenge's own `curl` line (in `kaggle-notebook.md`; the
   link expires every hour, refreshing it needs `fetch_challenges.py` and the user's approval) into `$C/dl/`.
   Check that the file is a real zip and not an "expired" XML page; extract into `$C/data/`; compare file names,
   row counts and columns with the description and stop on a mismatch (a link from another post downloads
   another challenge). Check free disk and RAM before extracting a large archive. Data never leaves the machine.
2. **Metric.** Implement it exactly from the description, every reading of an ambiguous sentence. Test it on
   hand-made cases and the description's examples; compute the chance, constant and sample-submission scores.
3. **Validation.** Mirror the hidden split: hold out whole groups (authors, conversation groups, countries,
   scaffolds); when the held-out factor is not in the data, use a few large clusters of simple input statistics.
   Grouped K-fold with out-of-fold predictions for every comparison; the same folds all challenge long.
4. **Data and leakage checks.** Exact and near duplicates across folds and between train and test inputs;
   group overlap; time order; no signal in ids, row order or file layout; class balance; train-to-test shift
   (test-input statistics are computed descriptively, never fitted on, unless the rules allow it).
5. **Probe, version 1.** The fastest rule-compliant solution that beats chance, written in the final framework
   from the Axo deterministic skeleton (`$AXO/assets/solution_template.py`, `$AXO/references/determinism.md`
   section 8). Package it with the gate (`--probe --chance <value>`) and offer it for upload: its board score is
   the only check that local validation agrees with the platform (Tray Elytra: about 54 locally, 9.43 on the
   board, found too late). The user decides whether to spend the credit.
6. **Strong candidates.** The research plan's top items, tested as one wave of structurally different families,
   each in its own file; one change per experiment against the same reference on the same folds; fixed seeds;
   then ablations of what the winner adds. A candidate whose test predictions agree with an uploaded version on
   97% of rows or more is the same bet: do not package it unless that small change is the point.
7. **Clean-state test.** The final file runs from a fresh process with only the public data folder; the gate's
   two full runs are this test.

## 2. Running things on this PC

- Wrap every command that loads challenge data or trains, run from `$C`:
  `python3 $S/guarded_run.py --tag <tag> [--gpu] [--mem 2G] [--cpus 4] [--timeout-min M] --note "<what changes>" -- <ML python> <script> <args>`.
  The ledger is `$C/experiments.jsonl`, logs go to `$C/out/runs/`. Without `--gpu` the run cannot see the GPU.
- **Size first.** Run one epoch or ten steps under the same wrapper, read the peak RAM and VRAM in the ledger,
  then set the plan. Keep peak VRAM at or below about 3.2 GB on the 3.68 GiB card (our margin): smaller batches
  with gradient accumulation, release a model before building the next one, inference at the training batch size.
- **Caps.** Start at 2 GiB (the practice on this machine). Ask for more only when `hw_profile.py` shows the room
  and no other heavy job runs; the wrapper refuses any cap that would leave less than 1.5 GiB for the desktop.
  Status "killed at the memory cap" (exit 137): make the job lighter (load in chunks, store float16, fewer
  copies) or move it to Kaggle. Never rerun it unwrapped.
- **Parallel waves.** CPU jobs with one thread and small caps, for example `--cpus 1 --mem 400M --max-cpu-jobs 6`,
  as many as the wrapper accepts. One GPU job at a time; the wrapper refuses a GPU job while another process
  computes on the GPU (the parallel session may): do not override that without the user.
- **Logs.** Output goes to the log file. Never pipe a long run through `tail` or `head`; start long runs in the
  background (Bash `run_in_background`) with a timeout and wait for the notification instead of polling.
- **Kaggle.** When a run needs more than about 3 GB of VRAM or many GPU hours, or the grader's package versions
  matter (local pandas 3.0 and numpy 2.5 against pandas 2.3 and numpy 2.1 in the Kaggle image of October 2026),
  use the parallel workspace's private-kernel runner under its `references/kaggle_execution.md`: private kernels
  only, data only through the challenge's own cells, quota checked, the user asked before any run over 3 GPU
  hours. Report remote runs as remote, with the kernel's `summary.json`.

## 3. Evidence and logging

- `experiments.jsonl` (written by `guarded_run.py`): one line per run with tag, note, command, script sha256,
  times, status, caps, peaks, and the metrics the script printed.
- Scripts end with one line `METRIC {"cv_mean": ..., "cv_std": ..., "folds": [...], "seed": ..., "config": "..."}`.
- `LOG.md` (from `assets/challenge_log.md`): clock, decisions with reasons, versions, submissions with the
  verbatim bot messages, the `BEST:` line, failures, lesson candidates.
- **Noise.** Measure the seed spread of the reference once (the same configuration, two seeds). A gain under
  about twice that spread, or under one board standard error, is not evidence. Compare candidates by paired
  per-fold or per-group differences. A configuration chosen on the folds that score it needs nested validation,
  or is reported at the unselected number (Station Siting version 4 lost 0.05 on the board).
- Call a noise-level upload a lottery ticket, and say it before the credit is spent.

## 4. The score loop (improve)

1. Record the complete bot message verbatim, with the version and the file sha256.
2. Calibrate: board minus local; the board's noise, about the per-row score spread over the square root of the
   test rows (or the between-group spread over the square root of the groups for group-averaged metrics); the
   distance to the baseline, the 10th, the 3rd and the top.
3. Board far from local (more than the fold spread plus the board's noise): stop building and find the cause
   first: the platform's run of the script, our copy of the metric, the difference between training and test.
   Replaying failure modes on held-out predictions (drop a component, break a column) can locate it.
4. Diagnose in this order: runtime, data and split, leakage and compliance, validation, optimisation,
   architecture and features, post-processing. Label each proposed change CONFIRMED, STRONG HYPOTHESIS or
   SPECULATIVE; never build a speculative one while a confirmed one is open.
5. Re-rank by expected gain per hour against the moving cutoff (the 10th place once seats fill, the top 3 for
   prizes). The biggest step first; two additions to one model inside the noise close that family; when the top
   is far above the current formulation's ceiling, change the formulation.
6. Build, validate, audit, package as a new version, hand over, and ask before the upload.

- **Credits:** 6 per challenge on the shared account, one back every 4 hours. Recommend an upload only when the
  gain clears the noise or the upload buys calibration; never to test code.
- **Stop-loss:** 90 minutes after a score without a gain on the calibrated split, or every planned family inside
  the noise: report the realistic outcome and recommend continuing or switching. The user decides.
- **Deadlines:** set the last experiment start and the freeze point backwards from the cutoff using the measured
  gate time (two full runs) plus the upload. At 8/10 seats, or when the user says "now", hand over the best
  measured package before anything else.
