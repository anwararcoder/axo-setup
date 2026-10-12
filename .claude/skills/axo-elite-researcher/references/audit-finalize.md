# Audit and finalize

## Contents
- A. Audit checklist
- B. Finalize
- C. Limits of the reused Axo scripts on this PC

## A. Audit checklist

Run it on the exact files to be uploaded. For each item record what was run and the result.

**Rules**
1. Every `RULES.md` row is quoted verbatim and marked pass with the command or inspection that checked it.
2. The model policy (A, B or C) holds: imports, downloads, weights, tokenizers. Class B: every learned parameter
   is initialised randomly in the file and trained on the provided files.
3. Data use: only permitted files; test rows only in the single prediction pass unless the text allows more; no
   test-to-test information flow unless allowed; no external data.
4. No leakage: nothing fitted on test inputs (vocabulary, scaler, threshold, index); ids and row order unused.

**Output**
5. Schema: exact columns in order, the sample's ids with none missing, extra or duplicated, row count, types,
   JSON cells that parse, value ranges and per-row limits, no NaN or Inf.
6. The prediction file was written by the exact script being sent (gate run A; sha256 in `HOW_TO_SUBMIT.md`).

**Runtime**
7. Runs end to end from the public folder alone; the `public_dir` / `submission_out` argv pattern with one
   spelling and the Kaggle fallback paths; safe to paste into one notebook cell.
8. Only libraries of the Kaggle image (its versions differ from the local ones); no network, no download or
   extraction code, no hardware-reading calls, one fixed device plan.
9. Deterministic: two full runs byte-identical under different hash seeds (gate stage 3); the Axo deterministic
   skeleton present (`$AXO/references/determinism.md`, section 8).
10. Runtime inside the challenge's limit with margin, measured here; measured on Kaggle too when the grader's
    hardware matters.

**Platform**
11. The platform's name appears nowhere in the solution or the description, in any case: the bot refuses it.
12. The description has Approach, Validation, Rule compliance and Runtime, real measured numbers only, and stays
    within 1500 characters when it has to be pasted as text (the bot's limit as recorded in the parallel
    workspace, 2026-10).
13. Score and artifact match: the reported local score and its ledger tag belong to these files' sha256.

Report findings ranked by severity (rule violation, invalid submission, reproducibility, risk), each with its
evidence, and list what stays uncertain.

## B. Finalize

1. Choose the winner by validation evidence, not by a public-score difference inside the noise; write why.
2. Freeze it: copy to a new file (`solution_vN_final.py`) and change only runtime behaviour
   (`$AXO/references/determinism.md`, sections 5 and 8). Keep the measured candidate unchanged.
3. Gate, run from `$C` inside the wrapper so a timeout or interruption stops the whole process tree:

   ```bash
   python3 $S/guarded_run.py --tag gate-vN [--gpu] --mem <solution peak + margin> --timeout-min <4 x run> -- \
     python3 $AXO/scripts/gate.py --python <ML python> --challenge "<exact name in the bot>" \
       --solution solution_vN_final.py --public-dir data --sample data/sample_submission.csv \
       --description SOLUTION_DESCRIPTION_vN.md --spent-time "<tracked time>" \
       --local-score <local> --baseline <baseline> --expect-minutes <measured run> --max-minutes <limit> \
       --work out/gate_vN --out out/_building_vN [--probe --chance <chance>] [--json-column C] [--regex C 'P'] [--forbid 'P']
   ```

   The gate checks the source, kills a run after 30 s to find the fallback file, makes two full runs and
   compares them byte for byte, checks the format and builds the folder with `HOW_TO_SUBMIT.md`. A failure
   names its stage; fix the cause, never assemble the folder by hand.
4. Rename `out/_building_vN` to `submission_vN/` only when everything has passed. Only a recommended candidate
   becomes a `submission_vN` folder: the user uploads whatever sits there.
5. Run checklist A on the folder's files. Hand over: where the folder is, which version, the local score with
   its label against the baseline, the 10th place and the top, what it rests on, the spent time written in the
   guide, and the request for the status block. Then stop and wait for the user.

## C. Limits of the reused Axo scripts on this PC

- `keep_awake` and the power check in the Axo scripts work on Windows only; `hw_profile.py` reports the power
  source here.
- The Axo `kill_tree` stops only the direct child process on Linux, hence the gate inside `guarded_run.py`.
- The Axo `preflight.py` treats more than 600 MiB of GPU memory in use as busy; `hw_profile.py` lists the
  actual compute processes instead.
