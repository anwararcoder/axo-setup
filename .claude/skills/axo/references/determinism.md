# Deterministic plan

Axo challenges ask for "a fixed, deterministic training and inference plan". Treat that as two separate requirements, both set up **before the first training run**, not checked at the end:

1. **Fixed plan** — the amount of work never depends on the clock or on a validation score.
2. **Reproducible output** — two runs of the same script on the same machine write byte-identical submission files.

Setting seeds is not enough. In the negation challenge the script had fixed seeds and `cudnn.deterministic = True`, and two runs still differed (2,412 vs 2,413 predicted frames) because GPU embedding gradients are accumulated in a non-deterministic order. It was only caught by comparing two full runs — after hours of experiments had already been run on a non-reproducible script.

## 1. Fixed plan

- Fixed number of epochs / passes / boosting rounds, written as constants at the top of the script.
- No early stopping, no "train until N minutes have passed", no patience counters in the submitted run.
- No hyper-parameter search inside the submitted run. Search during development; hard-code the chosen values.
- Any quantity fitted from data (thresholds, lexicons, calibration) comes from a fixed procedure on fixed data splits.
- Runtime safety comes from sizing the fixed plan with a wide margin, plus the fallback submission written at start — never from time checks that change what the script does.

## 2. Reproducible output — settings to put in the script from day one

**PyTorch**
```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # assigned, never setdefault; before importing torch
import torch

torch.manual_seed(SEED)                      # seeds CPU and all GPUs
torch.use_deterministic_algorithms(True)     # errors on any non-deterministic op: fix it, do not silence it
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
- Re-seed immediately before each model is created and before each training stage, so one stage's randomness does not depend on how much the previous one consumed.
- `DataLoader`: `num_workers=0`, or pass a seeded `generator` and a `worker_init_fn`; shuffle with your own seeded `numpy.random.RandomState` when possible.
- Mixed precision and `torch.compile` are fine only if the two-run check still passes.
- If `use_deterministic_algorithms(True)` raises for an operation, replace the operation (or move that step to CPU). Do not switch to `warn_only=True`.

**NumPy / Python**
- Use a local `np.random.RandomState(seed)` / `random.Random(seed)` per purpose rather than the global state.
- Never iterate over a `set` (or `dict` built from one) when the order affects the result — sort first. `PYTHONHASHSEED` is not under the script's control on the platform.
- `sorted(...)`/`argsort` with ties: use `kind="stable"` and a full tie-break key.
- File listings: `sorted(glob(...))`, never raw `os.listdir` order.

**Tree models / scikit-learn**
- Set `random_state` / `seed` on every estimator and splitter.
- LightGBM: `deterministic=True`, `force_row_wise=True`, fixed `num_threads`. XGBoost: fixed `seed`, `tree_method="hist"`; GPU histogram methods need the two-run check.
- CatBoost: fixed `random_seed`; GPU training is not guaranteed bit-reproducible — verify or use CPU.

**CPU backend limits, including preloaded notebooks**
- Environment assignments before imports may not change BLAS/OpenMP pools if a notebook already loaded NumPy, SciPy or scikit-learn. Enforce the chosen fixed CPU thread count during fitting and inference, using the framework's supported controls. For the confirmed scikit-learn stack, `threadpoolctl.threadpool_limits(limits=CPU_THREADS)` supplies an active scope; `CPU_THREADS` is a constant chosen for the plan, not detected from hardware. It is a required scikit-learn dependency (`target-env.md`). Keep applicable environment limits fixed as well.
- Seed every random source actually used, including estimator/splitter seeds. Verify the real execution scope with numerical libraries preloaded as well as in fresh processes. Use controls appropriate to the challenge's device/framework; CUDA backend settings do not repair a NumPy/scikit-learn CPU plan.
- Measure the full runtime after enforcing these controls. Earlier timings under uncontrolled worker/backend settings cannot support a reliable handover promise.

**Libraries that train for you** (Ultralytics, Lightning, Hugging Face Trainer): pass their seed and deterministic flags (`seed=`, `deterministic=True`, `workers` fixed) and still run the two-run check; their defaults often include time- or patience-based stopping that must be turned off.

**Output writing**
- Sort or keep a fixed row order; format floats with an explicit format string.
- Round in a way that cannot flip a thresholded value.

## 3. The check

Two moments, both with `scripts/check_determinism.py` (two separate processes, byte comparison, wall time of each run):

1. **As soon as the script first runs end to end**, on a reduced setting that takes a minute or two (two passes, the same code path). This catches a non-deterministic operation before any score is measured on it. A small wrapper that imports the solution, shrinks `CONFIG` and calls `main()` is enough.
2. **On every version that will be submitted**, on the full script. `scripts/gate.py` does this as its third stage.

```bash
python <skill>/scripts/check_determinism.py solution.py <public_dir> out/determinism --expect-minutes 8
```

- It must be the **full** script, started twice as separate processes. Checking one function, or one process run twice in a loop, misses process-level sources.
- `--expect-minutes` stops a run that passes twice the expected time. A full check once crawled for 71 minutes on a throttled laptop before anyone looked; with the limit it would have stopped after 48.
- `DIFFERENT` is a failed rule, not a curiosity. Do not submit until it is `IDENTICAL`; bisect by stage (compare saved intermediate predictions) to find the source.

### Interrupted verification is not a completed proof

Preserve completed outputs and logs when orchestration is interrupted. Reuse a completed run only when its successful exit and unchanged source, inputs and execution plan can be established. Start the missing full run as a separate process and compare the completed CSVs byte for byte, recording the actual processes, source fingerprint and timings. Exclude an interrupted run's safety/partial file. Use the standard validation and folder builder only after genuine completed-run evidence exists; never fabricate a successful gate state or weaken a failed check. If the interruption has no explanatory traceback, leave its cause unknown. Independent commands may avoid repeating a long orchestration command, but do not change the submitted plan to fit the clock.

### Keeping the proof cheap

The proof costs two full runs, so the size of the plan decides how long a submission takes to prepare.

- Size the plan so two full runs fit with margin; at most 20 minutes per run keeps the gate inside the one-hour window that remains if the challenge fills. In the transcript challenge version 1 used three models at 24 minutes per run, the proof took 49 minutes and the extra models were worth about 0.003, so the plan is sized to the budget, not to the maximum.
- These are our sizes for speed. The challenge's runtime limit is quoted separately and is never replaced by them.

## 4. What determinism does not promise

Identical output is expected on the same hardware and library versions. The platform's GPU differs from the development machine, so its numbers can differ slightly from the local run; that is normal and not a violation. What the rule protects is that the organisers get the same file every time they run the script.

## 5. Finalization: the winner becomes a separate deterministic file

Source: the challenge-engine pack (`references/pack/`, `deterministic_finalization.md` and phase 4 of its skill). The fixed-plan habits above are in every candidate from the first file; finalization is the explicit step after the best valid solution has been selected. Create `solution.py` as a separate copy and leave the measured candidate untouched.

**Freeze first; change none of these:** model family, preprocessing, feature construction, split logic, hyper-parameters, checkpoint selection, inference logic, submission column and order logic.

**Remove run-time branching.** Bad patterns:

- choosing the GPU count dynamically; a CPU fallback when CUDA is missing; the device decided by `torch.cuda.is_available()`;
- trying one scaler or backend and then another; retrying with fewer workers after a failure; changing the batch size after an out-of-memory error;
- loading whichever checkpoint happens to exist; opportunistic caches;
- `os.cpu_count()` or `CUDA_VISIBLE_DEVICES` deciding the plan; behaviour that depends on files found under `/kaggle/input`;
- changing the model because an optional dependency is missing.

`scripts/build_submission_folder.py` flags these patterns in the source. A line that genuinely is a fixed plan carries `# fixed-plan-ok: <reason>`, and the reason goes into the description.

**Make the plan explicit:** one device strategy (GPU challenge: a single fixed `cuda:0`; CPU challenge: `cpu` explicitly; several GPUs only when the rules and the runtime check allow it and the plan stays fixed); one model name or path; fixed workers, fixed order, fixed seeds, fixed backend settings; no recovery path that changes execution semantics. PyTorch: fixed seeds, `torch.use_deterministic_algorithms(True)`, deterministic cuDNN, `benchmark=False`, fixed worker seeds, a fixed `CUBLAS_WORKSPACE_CONFIG`, using only flags the installed version supports. TensorFlow: the framework's deterministic and seed controls that match the installed Kaggle version; never assume the local version. Warnings are secondary; submission-blocking errors and rule violations are primary.

Determinism can change speed and sometimes the score. The deterministic copy is an evaluation-safe artifact, not an opportunity to redesign the model.

**Proof:** `scripts/check_determinism.py` or `scripts/gate.py` (two full runs, separate processes, byte comparison).

**Retry after a failed check:** inspect the exact checker failure and the runtime-dependent pattern that remains; keep the algorithm and the compliant behaviour; prefer style and control-flow-only edits; create `solution_deterministic_retry.py` and keep the winner and the earlier deterministic file; list every modification as stylistic only or runtime-behavioural, and say plainly when style-only edits cannot fix the check (`commands.md`).

## 6. Checklist rows to add to RULES.md

| Rule | How it is checked |
|---|---|
| Fixed number of passes; no early or time-based stopping | grep the script for `time.time()` used in a condition, `patience`, `early_stop`; constants at top |
| Seeds fixed for every random source | grep for unseeded `random`, `np.random.<fn>` on global state, `shuffle=True` without a generator |
| Deterministic kernels enabled | `torch.use_deterministic_algorithms(True)` present and no `warn_only` |
| Byte-identical output across two full runs | `scripts/check_determinism.py` (stage 3 of `scripts/gate.py`) → IDENTICAL |
| No run-time branching on device, paths, imports or retries | the builder's static check passes; every `fixed-plan-ok` line has a stated reason |

## 7. The bot's runtime-plan check and the fix that usually clears it

Source: a screenshot the user shared on 2026-10-07 from another session, where a scored solution was held back by this message from the platform: "We couldn't confirm that your solution uses a fixed, deterministic runtime plan. Review runtime-dependent training, inference, fallback, worker, seed, and backend logic, then retry the checks. If this persists, contact support." The user says the changes below usually clear it.

| What triggered it | Fix |
|---|---|
| Training stops early when the clock passes a limit | Remove it: a fixed number of epochs; the time is only logged |
| `cuda` if available, otherwise CPU | One fixed `cuda:0` |
| `DataParallel` when more than one GPU is present | Remove it: always one GPU, so the output is the same on one A10G or two T4s |
| `try/except` that picks the `GradScaler` or mixed-precision mode by device | One fixed path |
| No determinism settings for the backend | `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, `torch.use_deterministic_algorithms(True)`, TF32 off (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`), `CUBLAS_WORKSPACE_CONFIG` set before torch is imported |

After the fix, run a short version twice and compare losses, metrics and the submission file; they must match exactly, and strict mode must raise no error. Two things to tell the user with such a fix: the score can move a little when training goes from two GPUs to one, and with no time-based stop the run must be timed again against the challenge's limit (reduce folds or epochs by a fixed amount if it is close).

## 8. The deterministic skeleton: mandatory from the probe, never added later

The user's instruction, 2026-10-08, after three platform rejections in one evening cost the grace hour of Conversational Functional Span Decomposition: determinism must never go wrong again. It is no longer a set of habits; it is a skeleton that every solution starts from, and the gate now refuses a file that lacks any part of it.

**What happened (all on 2026-10-08, one challenge):**

| Upload | Platform result | Cause found |
|---|---|---|
| Version 1, scikit-learn logistic regression on CPU | "Prompt Compliance ... couldn't confirm that your solution follows the challenge's explicit implementation requirements" | A From Scratch challenge ("All learned model parameters must start from random initialization"); a zero-initialised CPU model cannot be confirmed as that |
| Version 3, PyTorch on GPU, scored 0.4726 | "Deterministic Execution ... couldn't confirm ... a fixed, deterministic runtime plan" | `for w in set(words)` decided the order in which features were summed; no TF32 flags; `os.environ.setdefault` for cuBLAS; a `torch.cuda.get_device_name` call |
| Version 4, the same with those four fixed | accepted, 0.4607 | - |
| Version 5, identical file with `nn_seeds=3` instead of 1 | "Deterministic Execution" again | not found; the file differed only in that constant. The check is not fully predictable, so the file must give it nothing to doubt |

The local two-run check said IDENTICAL every time. It could not see the hash-order dependence because both runs had the same environment.

**The skeleton (copy from `assets/solution_template.py`; never write these lines from memory):**

1. `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` assigned before torch is imported. Never `setdefault`: a value that depends on the environment is a runtime dependence.
2. `make_deterministic(seed)` with all seven lines: `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True`, `cudnn.benchmark = False`, `cuda.matmul.allow_tf32 = False`, `cudnn.allow_tf32 = False`. Call it before every model is created.
3. One literal device constant. No call that reads the hardware: no `is_available`, `device_count`, `get_device_name`, `get_device_properties`, memory queries. Print the constant, not the hardware.
4. Sorted everything: never iterate over a `set`, a set comprehension or a dict built from one; write `sorted(...)`. Sort feature ids, vocabularies, class lists and file lists. When features are summed, the order of the terms is part of the result.
5. No workers (`num_workers=0` or no DataLoader), no mixed precision, no `try/except` that changes the path, no clock in any condition, no early stopping.
6. A docstring section headed "Deterministic runtime plan" that states points 1 to 5 in words, and the same list in the description's rule section. The platform's check reads the code; say plainly what it should find.
7. Files written as plain UTF-8 with LF line ends. PowerShell `Set-Content -Encoding UTF8` adds a byte-order mark and CRLF; never rewrite a solution file with it.

**What the tools now enforce (both copies of the skill):**

- `check_determinism.py` (stage 3 of the gate) runs A and B under two different `PYTHONHASHSEED` values, so a hash-order dependence prints DIFFERENT locally.
- `build_submission_folder.py` (stage 1 of the gate) stops on: `environ.setdefault`, `for ... in set(...)`, hardware-reading calls, `warn_only=True`, `num_workers>0`, mixed precision, the clock in a condition, a byte-order mark or CRLF, any of the seven backend settings missing from a file that imports torch, and a top docstring that does not state the deterministic plan.

**Process rules that go with it:**

- The probe is written in the final framework. In a From Scratch or GPU challenge the probe is a PyTorch model on the stated device with explicit random initialisation, even when a scikit-learn model would score the same in less time. A rejected probe costs the whole first window.
- At intake, quote the sentence about initialisation and the Compute line into `RULES.md` and answer in writing: would a reviewer reading only the code be able to confirm it? If the answer needs an argument, change the code.
- Once a file has passed the platform's checks, the next version changes only what was measured. Keep the passed skeleton byte for byte, and prefer the smallest diff. If the same check fails on a near-identical file, do not guess a third time during a deadline: send the passed version's settings and report the message to support.
- A platform check failure is recorded in `NOTES.md` with the exact message and the diff against the last accepted file.

