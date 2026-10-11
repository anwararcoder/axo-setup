# Claude Challenge Engine: Commands, Prompts, and Workflow

## Goal

Use `challenge-engine` as a disciplined competition workflow, not as a giant prompt.
The priorities are:

1. Maximize expected hidden-test performance.
2. Follow the full challenge description and rules exactly.
3. Protect against leakage, public-leaderboard overfitting, and brittle hacks.
4. Improve the current best solution iteratively instead of rewriting blindly.
5. Use targeted research only when it is justified by evidence.
6. Finalize determinism only after the best valid solution is selected.
7. Save tokens by removing repeated/wasted work, not by unnecessarily weakening the main solver.

Your normal main solver can remain your strongest model at your highest useful effort.

---

# 1. The exact workflow

```text
NEW CHALLENGE
    |
    v
/challenge-engine solve
    |
    v
Kaggle training
    |
    v
Score + logs + errors
    |
    v
/challenge-engine improve
    |
    v
Kaggle training
    |
    +------> repeat while improvements are credible
    |
    v
Plateau / unclear bottleneck?
    |
    +---- no ----> improve again
    |
    +---- yes ---> targeted research
                       |
                       v
                    improve
                       |
                       v
                 choose winner
                       |
                       v
              /challenge-engine finalize
                       |
                       v
              deterministic submission
                       |
                       v
             /challenge-engine postmortem
```

Do not make broad research a mandatory first step. Start from the task, data, rules, and strong existing knowledge. Research later when a real information bottleneck appears.

---

# 2. New challenge: `solve`

## Command

```text
/challenge-engine solve
```

## Paste this after it

```text
[PASTE THE FULL CHALLENGE DESCRIPTION HERE]

The dataset is available in the current working directory. Inspect the workspace and understand the actual dataset before implementing the solution.

Use the prepared Kaggle environment. Do not install packages unless the challenge explicitly requires it and the environment genuinely does not provide what is needed.

Do not train locally. I will run the solution in Kaggle.

Produce one complete solution.py.

The solution must be notebook-compatible: if I paste/run the code in one notebook cell, it should work. Do not design it as a CLI-only program.

Use the exact challenge-provided input/output contract.

Prioritize the strongest credible approach you can implement under the available compute and time constraints. Do not intentionally give me a toy baseline unless the challenge requires one.

The primary objective is strong hidden-test generalization, not merely public leaderboard optimization.

Read and follow every rule in the challenge description before coding.

Do not use prohibited evaluation information. Do not train on test/evaluation targets, create prohibited synthetic evaluation data, pool information across evaluation cases when the rules do not allow it, or introduce any other leakage.

Respect all from-scratch restrictions. If the challenge is from-scratch, do not use pretrained models, checkpoints, embeddings, or representations unless explicitly allowed.

Respect compute and training-time limits. Keep training practical and do not make it unnecessarily long.

Add useful training logs so I can monitor progress and see an estimated completion time.

Use the strongest allowed hardware configuration described by the challenge. If multiple GPUs are explicitly allowed, use them when genuinely beneficial. If the challenge is CPU-only, do not use CUDA.

Do not rely on packages, files, or GPUs from my local machine. The code must target the Kaggle environment where I will run it.

Do not make broad research yet unless a specific missing piece of knowledge is blocking a strong solution.

Return the complete solution.py.
```

## If no baseline score exists

Use:

```text
Baseline score: Not available yet. No submissions have been made.

Do not invent or assume a baseline. Optimize for the strongest expected generalization performance allowed by the challenge.
```

---

# 3. Challenge-specific additions

## CPU-only

```text
The challenge is CPU-only.

Do not use CUDA, GPU execution, GPU-specific libraries, or runtime device fallbacks that may select a GPU.

Use an explicitly CPU-only execution plan.
```

## GPU allowed / multi-GPU

```text
The challenge allows GPU execution and Kaggle provides multiple GPUs.

Use the available GPUs when this is compatible with the rules and provides a real training-time benefit. Keep the training/inference plan compatible with the final challenge requirements.
```

## From-scratch

```text
This is a from-scratch challenge.

Strictly obey the from-scratch restriction. Do not use pretrained models, pretrained checkpoints, pretrained embeddings, external learned representations, or transferred weights unless the challenge explicitly permits them.
```

---

# 4. After the first Kaggle run: `improve`

Do NOT paste the full challenge description again if Claude is still in the same session and already has the context.

## Command

```text
/challenge-engine improve
```

## Send only the new evidence

```text
New score: 0.XXXX

Training logs:
[PASTE LOGS]


Improve the current solution further.

Use the actual score, logs, code, and observed behavior to identify the highest-value bottleneck.

Keep the current best approach unless there is strong evidence that a change should improve generalization or solve a real limitation.

Do not optimize only for the public leaderboard.

Do not violate any rule in the full challenge description.
Do not use prohibited test/evaluation information.
Do not introduce synthetic data when prohibited.
Do not create cross-test-case features or pooled evaluation context unless explicitly allowed.

Keep the current best solution protected. Create a new candidate version for any significant change rather than destroying the best version.
```

Run this again for every real improvement round. You only need the new score, logs, errors, and important observations.

---

# 5. If the new experiment is worse

```text
The new experiment reduced performance.

Previous best score: X.XXXX
New score: Y.YYYY

Do not keep the weaker version.

Analyze why the change failed using the actual logs and results.

Return to the strongest validated solution. Do not make another random change.

If the failure reveals a reusable lesson, record it for the postmortem.
```

---

# 6. Anti-overfitting / hidden-test protection

Use this whenever you are worried Claude is chasing a public score:

```text
Do not optimize for the current public leaderboard score as if it were the true objective.

Treat public leaderboard performance as noisy evidence, not as the target distribution itself.

The primary objective is strong generalization to the hidden evaluation set.

Before accepting a change, ask whether there is a plausible mechanism explaining the improvement and whether it is likely to generalize.

Do not add brittle heuristics whose only justification is a small public-score increase.

Do not tune logic to individual public-test examples unless the challenge explicitly permits that information.

Do not exploit accidental properties of the current public split.

Avoid increasing complexity merely to chase small leaderboard gains.

Prefer robust improvements supported by validation, ablation, data analysis, error analysis, or strong task-specific reasoning.

Keep separate:
1. validation evidence,
2. public leaderboard evidence,
3. hidden-test expectation.

If a change improves the public score while making the method less defensible or more brittle, flag the risk instead of automatically accepting it.
```

---

# 7. If someone is #1 on the public leaderboard

Knowing the top score is useful, but it must not become a reason to overfit.

Use:

```text
There is currently a top-ranked public solution with a score of: X.XXXX.

Our current best score is: Y.YYYY.

Treat the top score as an external signal, not as proof that the same method will generalize to the hidden set.

Analyze the likely reasons for the performance gap.

Do not use prohibited evaluation information or reverse-engineer hidden labels.
Do not copy any technique that conflicts with the challenge rules.

Determine whether the gap is more likely caused by:
- representation,
- model capacity,
- optimization,
- preprocessing,
- feature engineering,
- validation design,
- inference,
- metric alignment,
- data usage,
- or leaderboard overfitting.

Any adopted improvement must have a defensible reason and remain fully compliant with the challenge rules.
```

## If the top solution code is public and the rules allow analysis

```text
A public top-ranked solution is available.

First verify that its methods are allowed by the challenge rules.

Analyze the generalizable ideas and adapt only the allowed ideas that fit our solution and compute budget.

Do not import evaluation leakage, prohibited external information, or test-derived tricks.

Keep our current best solution untouched and test the adaptation as a separate candidate.
```

---

# 8. When to use research

Do NOT research just because the challenge is difficult.

Recommended sequence:

```text
solve
-> train
-> measure
-> improve
-> measure
-> improve
-> plateau / unclear bottleneck
-> targeted research
-> improve
```

Research is most justified when:

- Claude cannot identify a credible next improvement.
- Two or more strong improvement attempts fail.
- There is a large validation/public gap that needs explanation.
- The task requires a specialized method Claude is not confident about.
- Evidence suggests a relevant method family is missing from the current solution.

---

# 9. Targeted research prompt

Use the same `improve` command, then send:

```text
We have reached a plateau.

Current best score: X.XXXX

Previous attempts:
[brief summary of the strongest attempts]

Current suspected bottleneck:
[describe it]

Review the current solution, logs, observed behavior, and challenge rules first.

If the bottleneck cannot be solved confidently from the existing evidence, perform targeted research on this exact bottleneck.

Research must be focused and evidence-based.

Do not perform a broad literature review.

Look for at most 3 high-value, plausible approaches.

For each candidate, report:
- why it is relevant,
- supporting evidence,
- expected upside,
- implementation cost,
- overfitting risk,
- rule-compliance risk,
- and how it fits the current best solution.

Do not rewrite the winning solution just because research found a new idea.

Keep the current best solution untouched and create a separate candidate for testing.
```

The rule is: research should answer a specific question, not search for a magical "best solution."

---

# 10. If Claude cannot improve further

```text
We have attempted multiple credible improvements and there is no clear score increase.

Do not invent random changes.

Perform a structured post-analysis of:
- validation strategy,
- data distribution,
- target/feature behavior,
- error patterns,
- metric optimization,
- inference,
- and generalization risks.

Then perform targeted research only on the unresolved bottleneck.

I want evidence-backed options, not a long list of ideas.

Keep the current best solution as the protected fallback.
```

---

# 11. Runtime errors

Use:

```text
/challenge-engine improve
```

Then:

```text
The Kaggle run failed with RuntimeError.

Exact traceback:
[PASTE TRACEBACK]

Identify the root cause before modifying the solution.

Fix only what is necessary.

Preserve the intended algorithm, challenge compliance, compute plan, and expected model behavior.

Do not remove useful functionality just to hide the error.

Return the corrected solution as a separate version when practical.
```

## CPU/GPU mismatch

```text
The challenge explicitly requires CPU execution.

The RuntimeError appears related to CUDA/GPU usage.

Make the execution plan explicitly CPU-only.

Do not introduce runtime device fallbacks.

Do not change the model strategy unless the CPU requirement genuinely makes the current method impossible.
```

---

# 12. Deterministic finalization

Do this only after selecting the best valid solution.

## Command

```text
/challenge-engine finalize
```

## Prompt

```text
This is the final selected winning solution.

Create a deterministic final version for submission.

Preserve the exact winning algorithm and intended behavior as much as possible.

The goal is to eliminate runtime-dependent execution paths that can prevent deterministic-execution verification.

Review:
- device selection,
- worker counts,
- random seeds,
- backend flags,
- execution ordering,
- fallback logic,
- cache-dependent behavior,
- environment-dependent branching,
- nondeterministic inference,
- and runtime-dependent training plans.

Use a single fixed execution plan where the challenge requires it.

Do not violate any challenge rule.
Do not use prohibited evaluation information.

Do not change the data, labels, feature engineering, task formulation, or winning method unless deterministic verification genuinely requires a runtime-behavior change.

Keep the winning solution untouched and create a separate deterministic file.
```

---

# 13. Determinism retry: if the check still fails

Use only after the first deterministic version is rejected.

```text
The deterministic execution check still fails.

First inspect the exact checker failure and identify the runtime-dependent pattern that remains.

Preserve the winning algorithm and challenge-compliant behavior.

Do not change:
- data,
- labels,
- feature engineering,
- preprocessing,
- model architecture,
- loss,
- optimizer,
- hyperparameters,
- validation split,
- inference strategy,
- submission format,

unless the deterministic checker genuinely requires a behavior change.

If possible, prefer a style/control-flow-only revision: reorganize code, remove unnecessary stylistic complexity, make equivalent operations explicit, and eliminate ambiguous runtime branching without changing intended functionality.

Do not introduce new functionality.
Do not violate any challenge rule.
Do not use prohibited evaluation information.

Create:
solution_deterministic_retry.py

Keep the original winning solution and previous deterministic version untouched.

At the end, list every modification and classify it as:
- stylistic only, or
- runtime-behavioral.

If the checker cannot be fixed by style-only changes, say so clearly instead of pretending that a behavioral change was purely stylistic.
```

Important: style-only edits cannot guarantee a determinism fix. If the checker is reacting to GPU selection, workers, fallbacks, seeds, backend behavior, or another runtime-dependent plan, the necessary fix may genuinely affect execution behavior.

---

# 14. Final submission check

Before uploading the final file:

```text
Review the final submission one last time.

Do not optimize the algorithm further.

Verify:
1. challenge rules,
2. data usage,
3. no prohibited leakage,
4. no prohibited synthetic evaluation data,
5. no test-target learning,
6. correct input/output contract,
7. deterministic runtime plan if required,
8. submission format,
9. fixed device/runtime assumptions,
10. reproducibility.

Do not make speculative performance changes.

If everything is valid, return the exact final submission file and a concise summary of the runtime plan.
```

---

# 15. Postmortem

After the challenge is finished:

```text
/challenge-engine postmortem
```

Then:

```text
Create a concise postmortem for this challenge.

Record:
- final score,
- best score,
- strongest successful idea,
- failed ideas,
- important errors,
- rule/compliance lessons,
- validation lessons,
- overfitting risks discovered,
- deterministic-execution lessons,
- and reusable lessons for future challenges.

Separate:
A. challenge-specific facts
B. reusable general principles

Do not rewrite the active solution.
Do not invent lessons that are not supported by the actual results.
```

---

# 16. `/loop`

Do NOT use `/loop` while you are actively waiting for a Kaggle run.

Do NOT think of `/loop` as training Claude.

Use it for periodic maintenance after you have accumulated several challenge postmortems.

Example:

```text
/loop 3d
```

Prompt:

```text
Review completed challenge postmortems and challenge-related knowledge.

Identify repeated, evidence-supported failure patterns and reusable improvements.

Do not modify active challenge solutions.
Do not perform broad research.
Do not add a rule merely because it appeared once.

Only update reusable knowledge when there is strong evidence that the lesson generalizes across tasks.

Keep permanent instructions concise and avoid knowledge-file bloat.
```

---

# 17. Token-saving rules

The safest token savings come from reducing repeated work, not from forcing the main solver to reason less.

## Good

After the first run:

```text
Use the existing challenge context and current solution.

Focus only on the new score, logs, errors, and the unresolved bottleneck.

Do not repeat analyses that have already been established.
```

Use targeted research instead of broad research.

Keep the best solution protected.

Use `/compact` when the context is large:

```text
/compact Preserve the complete challenge rules, current best solution, measured scores, strongest experiments, important failures, current bottleneck, and unresolved errors.
```

## Bad

Avoid repeatedly asking for:

```text
Do everything again from scratch.
```

Avoid broad research on every improvement round.

Avoid creating many subagents for tasks that can be completed in the main session.

Avoid reducing the main model's reasoning effort only for token savings when the task genuinely needs deep reasoning.

---

# 18. How to use the public leaderboard correctly

Keep this mental model:

```text
Challenge rules = authority
Validation = evidence
Public leaderboard = evidence
Hidden-test generalization = objective
```

If the public score rises for no defensible reason:

```text
The public score increased, but I am concerned about public-set overfitting.

Analyze whether the improvement is robust and mechanistically justified.

Do not keep the change merely because the public score increased.

Prefer the candidate with stronger expected hidden-test generalization.
```

Never turn public test behavior into hidden target labels or a prohibited training signal.

---

# 19. Comparing candidate solutions

```text
Compare solution_A and solution_B.

Consider:
- validation evidence,
- public score,
- stability,
- complexity,
- training cost,
- inference cost,
- rule compliance,
- leakage risk,
- overfitting risk,
- deterministic-execution risk,
- and expected hidden-test generalization.

Choose the strongest overall candidate, not simply the one with the smallest public-score advantage.

Keep both candidates intact.
```

---

# 20. Quick reference

| Situation | Command |
|---|---|
| New challenge | `/challenge-engine solve` |
| New Kaggle result | `/challenge-engine improve` |
| Plateau | `/challenge-engine improve` + targeted research prompt |
| Final winning solution | `/challenge-engine finalize` |
| Determinism still failing | Determinism Retry prompt |
| Finished challenge | `/challenge-engine postmortem` |
| Periodic knowledge cleanup | `/loop 3d` |
| Huge context | `/compact ...` |
| Check installed skills | `/skills` |
| Check memory | `/memory` |
| Inspect context usage | `/context` |

---

# 21. Golden rules

1. **Protect the best validated solution.** New ideas are candidates until tested.
2. **Do not confuse public score with hidden-test performance.**
3. **Research a bottleneck, not the entire universe.**
4. **Research after evidence of a plateau, not automatically before the first solution.**
5. **Deterministic finalization comes after winning-solution selection.**
6. **Challenge rules always outrank leaderboard score.**
7. **Do not claim an improvement without a measured result.**
8. **Keep separate files/versions for strong candidates so a regression can be discarded.**
9. **A top-1 public score is a signal to investigate, not permission to copy or overfit.**
10. **Use the main high-effort solver for the hard decisions; use research/auxiliary agents only when they add clear value.**

