# Submission package

Every submission is handed to the user as one folder, `submission/`, that holds everything to upload plus a guide, **with nothing left open**. The user uploads from that folder without having to decide, fix or check anything.

```
submission/
  1_submission.csv             prediction file                      → upload
  2_solution.py                well-documented solution code        → upload
  3_SOLUTION_DESCRIPTION.md    separate solution description        → upload / paste
  HOW_TO_SUBMIT.md             the user's step-by-step guide        → NOT uploaded
```

The user always submits **both** a well-documented solution **and** a separate solution description. The bot says a description is optional when the code explains itself; that option is not used.

## No caveats

"Ready" means there is nothing to add when handing the folder over. If something is unresolved — a guessed path, development code still in the script, a check not run, a score not above the baseline, a rule with a doubtful reading — the folder is not built yet. Resolve it first:

- Something you can fix or check yourself: do it.
- Something only the user can settle (the Kaggle dataset path, how to read an ambiguous rule): it was asked in the single batch of questions at intake; if it is still open, ask now, then build.

`scripts/gate.py` enforces this: it runs every generic check in order, stops at the first failure with the reason, and builds the folder only when all of them pass. Do not assemble the folder by hand to get around a failing check, and do not hand over a folder together with a list of reservations.

**Only a recommended candidate becomes a `submission_vN` folder (the user's anger, 2026-10-09).** The user uploads whatever sits among the upload folders, so a candidate that scores below the best version locally, shares 97% or more of its test predictions with an uploaded version, or is only an experiment stays in `out/not_for_upload/` (or is never built) and is not described as ready. Decide whether to recommend it before the gate builds anything, and give the reason with the hand-over.

## 1. Prediction file

The file written by the final run of the exact script being sent. If the code changed after the file was written, re-run.

## 2. Solution code — well documented

One self-contained script. Before building:

- **Docstring at the top**: the task in one line, the approach step by step, why each step fits this task, the training plan, and the rules the solution observes.
- **Comments explain reasoning**, not syntax: why a threshold, why a decoding constraint, why a rule is derived the way it is. Every non-obvious constant has a comment.
- Function docstrings for anything a reviewer would need to follow the pipeline.
- **Remove all development-only code**: cross-validation loops, error dumps, development switches and environment flags, caching, alternative experiments, debug prints. The submitted file contains only what produces the submission. Keep the development version separately in the challenge folder.
- No dataset download or unzip code.
- The `public_dir` / `submission_out` pattern, one spelling, with the fallback paths `/kaggle/working/extracted_dataset` and `/kaggle/working/submission.csv` (the user's own pattern; argv is tolerated, never required, so the file runs pasted into one notebook cell).
- Imports limited to the standard Kaggle image. One fixed hardware plan and no run-time branching (`determinism.md`); `scripts/build_submission_folder.py` flags the bad patterns, and a line that really is a fixed plan carries `# fixed-plan-ok: <reason>`.
- After the clean-up, run the gate on the cleaned file. Its two full runs are the timed run and the proof of identical output, and run A's prediction file is the one that goes in the folder.


## 3. Solution description — always a separate file

Write `SOLUTION_DESCRIPTION.md` for the Axo reviewers. They have not seen the conversation; write it to be read on its own. Use these sections:

```markdown
# Solution description — <challenge name>

## Approach
<What the solution does, step by step, and why each step fits this task.>

## Validation
<How the score was estimated locally, on what data, and the number. Say plainly that it is a local estimate.>

## Rule compliance
<One line per notable rule: model/weight sources, data sources, use of test rows, determinism.>

## Runtime
<Measured wall time of the full script and the hardware it was measured on.>
```

Only what was actually done and measured. No score that was not computed, no claim about hardware it was not run on.

## 4. Spent time

Track it from the start of the challenge in `NOTES.md` (start time at the top, pauses noted). The default figure is that tracked time, rounded up to the half hour; it is written into `HOW_TO_SUBMIT.md` as the exact text to type and mentioned at hand-over, so the user can change it before uploading. Do not stop to ask for it, and do not invent a figure that was not tracked.

## 5. Build the folder: one command

```bash
python <skill>/scripts/gate.py \
    --challenge "<exact name as shown in the bot>" \
    --solution solution.py --public-dir data --sample data/sample_submission.csv \
    --description SOLUTION_DESCRIPTION.md \
    --spent-time "<tracked time, e.g. 1.5 hours>" \
    --local-score <estimate> --baseline <AI Baseline> \
    --expect-minutes <projected full-run time> --max-minutes <the challenge's limit> \
    [--json-column <column>] [--regex <column> '<pattern every value must match>'] \
    [--forbid '<pattern that must not appear in the code>'] [--dev-marker <development switch name>] \
    [--probe --chance <score of a random or constant prediction>]
```

`--probe` is for version 1, the probe (`fast-v1.md`). It replaces "local score above the baseline" with "local score above `--chance`" and adds a line to `HOW_TO_SUBMIT.md` saying the version is a first scored version and whether it is above the baseline locally. No other check changes. A probe that is under the baseline locally is not a caveat on the folder: it is what a probe is, and it is stated. Later versions run without `--probe`.

The gate runs five stages and stops at the first failure, naming it:

| Stage | What it checks | Time |
|---|---|---|
| 1 static | argv pattern with one spelling; no placeholder path; top docstring; no download, extraction, debug or development code; `--forbid` patterns absent; local score above the baseline | seconds |
| 2 kill | the solution is started and killed after 30 s: a valid fallback submission must already be on disk | half a minute |
| 3 runs | two full runs as separate processes, timed, byte-identical; a crawling run is stopped (`--expect-minutes`); the limit is respected (`--max-minutes`) | two full runs |
| 4 format | header, ids, and each value against `--json-column` / `--regex` | seconds |
| 5 build | description has all four sections and no placeholders; copies the three files; writes `HOW_TO_SUBMIT.md` with file names, bot steps, spent time, fingerprints and the checks passed | seconds |

**Use the run time.** Write the description while stage 3 runs. If it is not complete when stage 5 is reached, the gate stops there and keeps the runs; finish it (the measured run time is printed for the Runtime section) and repeat the same command with `--resume`, which goes straight to stages 4 and 5 as long as the solution file is unchanged.

Because stage 1 comes first, a forgotten placeholder or a development switch costs seconds, not two full runs.

The gate covers the rules every challenge shares. The rule checklist (`RULES.md`) covers the challenge-specific ones; every row must be `pass` before the folder is handed over. `scripts/check_determinism.py`, `scripts/validate_submission.py` and `scripts/build_submission_folder.py` remain available on their own with their original arguments.

## 6. Hand-over

Tell the user where the folder is and that it is ready. Point them to `HOW_TO_SUBMIT.md`. Report the local score against the baseline and what that score rests on, and the spent time written in the guide. Ask for the status block now and the score when it arrives. Nothing else is needed — if there is something unresolved to say about the folder, it was not ready.

## After the user submits

- Record version, time, local score and what changed in `NOTES.md`.
- When the leaderboard score arrives, record it next to the local estimate; the gap calibrates local validation for the next version.
- Rejected: get the exact message, fix that point, add a row for it to `RULES.md`, rebuild the folder.
- For the next version, rebuild the folder from scratch with the script; never edit files inside `submission/`.
