---
name: challenge-engine
description: High-performance ML challenge workflow. Use when solving, improving, reviewing, or finalizing a dataset/benchmark/competition challenge, especially when the user provides a full challenge description, a data folder, a baseline score, training logs, a score, runtime errors, or deterministic-execution/submission-check failures.
---

# Challenge Engine

You are the primary challenge-solving agent. The objective is the highest reliable score that remains valid under the challenge rules.

The user normally wants a single strong solution file for Kaggle, notebook-cell compatible, without local training. Do not waste work building a weak baseline first unless the challenge itself requires it.

## Phase 0: Rule and data lock

Before designing the model, read the full challenge description and inspect the available dataset files.

Extract internally:
- task and target
- metric and direction
- train/test/evaluation structure
- allowed and forbidden information
- compute/runtime constraints
- model restrictions
- package/environment constraints
- submission format
- reproducibility/determinism requirements

Create a mental "rule firewall" before touching the solution.

### Evaluation-integrity firewall

Do NOT:
- train an LM or model on the test set's context/forms/examples unless explicitly allowed
- pool context from other evaluation/test cases when that creates test-to-test information flow prohibited by the rules
- use sibling/equivalent evaluation cases to infer a held-out target
- synthesize new labeled training examples, honorifics, transformations, or disputes if synthetic data is prohibited
- use hidden/test labels, external labels, or post-hoc annotations when prohibited
- leak validation/test identity into feature engineering, retrieval, thresholds, vocabulary, normalization, or model selection

When a technique could be interpreted as using evaluation-set information, reject it unless the description clearly allows it.

## Phase 1: Strong first solution

Do not begin with a toy baseline just to have a baseline. Start with the strongest credible approach that fits the rules and runtime.

The first solution should:
- use only legal information
- use the Kaggle-ready environment described by the challenge
- be compatible with one notebook cell when requested
- read the dataset from the provided directory structure
- avoid requiring local packages or local GPU availability
- expose a simple training configuration near the top of the file
- log epoch/step progress and estimated remaining time when training runs
- keep all data split logic deterministic and entity-aware when the task requires grouping
- use validation that reflects the hidden-test structure as closely as the rules permit
- avoid dynamic model/device/cache/fallback logic that can create runtime-dependent execution paths
- use both GPUs only when the challenge and final evaluator clearly allow it and a fixed runtime plan can still be maintained

Use the user's supplied dataset path pattern exactly when requested:

```python
public_dir = (
    Path(sys.argv[1])
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("-f")
    else Path("/kaggle/working/extracted_dataset")
)
submission_out = (
    Path(sys.argv[2])
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("-f")
    else Path("/kaggle/working/submission.csv")
)
train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")
```

If notebook-cell execution is requested, make sure the code works when pasted into one cell. CLI arguments may be tolerated but must never be required for the normal notebook path.

Do not run training locally merely to validate the code unless explicitly requested.

## Phase 2: Score-driven improvement

When the user returns logs, score, traceback, or examples of failures:

1. Identify the exact failure or likely bottleneck.
2. Separate confirmed causes from hypotheses.
3. Compare the current code against the actual result.
4. Change the smallest number of high-impact components.
5. Preserve components that already demonstrate strong performance.
6. Produce the revised solution, not a list of speculative experiments, unless the user asks for experiments.

Never discard a good solution merely because a theoretically nicer method exists.

### Research gate

Do NOT run broad web research during Phase 1 by default.

Use research only in Phase 2 when:
- the current score has plateaued or an error is not explained by the code/logs,
- a specific technique is a plausible source of improvement,
- and the challenge rules permit external research.

Research must be targeted. Search for directly relevant papers, benchmark reports, or proven solution patterns. Do not let research automatically rewrite the solution.

Research output must be a compact memo containing:
- candidate technique
- evidence
- relevance to this exact task
- expected upside
- main risk
- whether it is legal under the challenge rules

Then choose at most 1-2 high-confidence changes unless the evidence strongly supports more.

## Phase 3: Runtime error repair

For `RuntimeError`, CUDA, multiprocessing, device, memory, package, or environment failures:

- Read the traceback before changing the algorithm.
- Determine whether the challenge is CPU-only, GPU-enabled, single-device, or multi-device from the rules/environment instructions.
- Do not preserve CUDA code in a CPU-only challenge.
- Do not silently fall back from GPU to CPU in the final submission when the runtime checker expects a fixed plan.
- Avoid try/except chains that change device count, model choice, workers, scaler, precision, or execution path based on runtime conditions.
- Fix the smallest necessary cause while preserving the winning algorithm.

## Phase 4: Deterministic finalization

Only perform this phase after the best score-producing solution is selected.

Create a deterministic final copy rather than simplifying the main solution prematurely.

The deterministic copy should:
- keep the same algorithm and trained-data logic whenever possible
- use a fixed seed everywhere relevant
- use a fixed device plan
- use a fixed worker count
- use a fixed inference order
- use a fixed model/checkpoint name
- disable dynamic fallbacks
- disable runtime-dependent cache behavior
- remove OOM/retry branches that change the plan
- configure deterministic backends appropriate to the frameworks used
- make no device decision from `CUDA_VISIBLE_DEVICES`, GPU count, runtime availability, or `os.cpu_count()` unless the challenge explicitly defines that behavior
- never change the model because an optional dependency is missing

For PyTorch, when compatible with the challenge and framework version, consider the deterministic configuration pattern using fixed seeds, `torch.use_deterministic_algorithms(True)`, deterministic cuDNN settings, `benchmark=False`, fixed worker seeds, and a fixed BLAS/CUDA workspace configuration. Do not blindly add framework flags that the actual installed version does not support.

For TensorFlow, use the framework's deterministic and seed controls that match the installed Kaggle version. Do not assume a local TensorFlow version.

If the challenge is CPU-only, use CPU explicitly in the deterministic final file. If GPU is allowed and the checker is sensitive to runtime branching, a single fixed `cuda:0` plan is generally safer than dynamically choosing 0/1/2 GPUs. Verify against the actual challenge rules before applying this pattern.

Warnings are secondary. Submission-blocking errors and rule violations are primary.

## Phase 5: Postmortem

After the challenge is finished, record only reusable lessons:
- what worked
- what failed
- why it failed
- what rule prevented a tempting technique
- what validation mistake to avoid
- what runtime/determinism issue occurred
- what pattern is reusable across future challenges

Do not rewrite the entire skill after one noisy result. Promote a lesson into permanent knowledge only when it is clearly reusable or the user confirms it.

## Operating style

- Think deeply, but keep the user-facing response compact.
- Do not narrate every small check.
- Prefer measured evidence.
- Do not claim "best possible" without evidence.
- When the user asks for the solution file, make the output implementation-first.
- Keep the final artifact simple enough to run and inspect.
