# Commands, event router and prompts

Source: `references/pack/Claude_Challenge_Commands_and_Prompts.md` (the trusted original, with the full prompt texts). This page is the condensed working version.

## Workflow

```
solve -> run here -> hand over -> board score -> improve -> run here -> ... -> (plateau? targeted research) -> choose winner
      -> finalize -> deterministic submission -> postmortem
```

Subcommands of this skill: `select` (choose a challenge), `solve` (new challenge), `improve` (new score, logs or error), `finalize`, `audit`, `postmortem`. Never make broad research a first step.

## Event router: what arrives and what to do

| The user sends | Do |
|---|---|
| Challenge posts, status blocks | `select`: `SKILL.md` "Before picking a challenge" |
| A challenge description and data | `solve`: rule and data lock, then the probe as version 1 (hand it over), then the strong version; one complete `solution.py` per version |
| "continue improving", "improve", "keep going" | The "Continue improving smartly" section below: biggest step first, in its five-step order |
| "Submission scored" block, status block | `improve`: record the score, then Phase 2; do not repeat analysis already established |
| A score lower than the best | Worse-experiment flow below |
| Traceback, "Your solution failed with X" | Phase 3: root cause before any change |
| A deterministic-check failure | Phase 4 retry protocol |
| "now", "stop", 8/10 or more, a deadline | Hand over the best measured candidate; nothing else first; stop experiments when told |
| Challenge closed, "remove it" | `postmortem`, then do what they asked with the files |

After the first scored upload, do not ask for the description again if the session still has it: work from the new board score, the status block, your own logs, errors and observations.

## solve

Extract and obey every rule before coding. One complete `solution.py` that runs when pasted into one notebook cell and honours the exact input/output contract (`platform-rules.md`); version 1 is the fast probe that only has to beat chance and is uploaded first, the strongest credible approach under the stated compute and time is version 2 (the user's decision, 2026-10-05; `SKILL.md`); useful progress logs with an ETA; no installs unless the challenge requires them and the environment lacks them; you train, validate and check here, as the primary runner (`SKILL.md`). CPU-only challenge: explicit CPU plan, no CUDA. GPU challenge: one fixed device; several GPUs only when the rules allow and a fixed plan can be kept. From-scratch challenge: no pretrained models, checkpoints, embeddings or transferred weights unless explicitly allowed. No baseline score yet: write "Not announced"; do not invent one.

## Continue improving smartly

The user's standing instruction (added 2026-10-08, in their words). It applies every time they say "continue improving" or the work goes on after a submission:

> Continue improving the work smartly.
>
> - The biggest step first is the rule for everything after the first fast submission: prioritize Version 2, the single change with the largest expected jump, rather than the next refinement.
> - Order of work after the probe:
>   1. Spend 30 minutes reading the complete raw examples and checking the training objective by hand.
>   2. Measure what each part of the pipeline would score if it were perfect.
>   3. Rank ideas by expected gain per hour and share the list in that order.
>   4. Build the top idea alone and hand it over.
>   5. Re-rank after every board update.

How to follow it: `climb.md`, "Biggest step first", has the detail, the measured cases behind it and the list of what is deferred until the place is safe. When the instruction arrives and steps 1 to 3 have already been done for this challenge, do not repeat them: say the current ranked list in one short table, then carry on with step 4 or 5. Every rule check and the deterministic skeleton (`determinism.md`, section 8) still apply to whatever is built.

## Worse experiment

State previous best and new score; do not keep the weaker version; analyse the failure from the actual logs; return to the strongest validated solution; make no other random change; record a reusable lesson only if the failure reveals one (`postmortem.md`).

## Anti-overfitting

Keep three kinds of evidence apart: validation, public leaderboard, hidden-test expectation. The public score is noisy evidence, not the target. Before accepting a change ask: is there a plausible mechanism, will it generalise? Do not add brittle heuristics for a small public gain; do not tune to individual public examples; flag a gain that makes the method more brittle instead of accepting it automatically. Rules outrank score; do not claim an improvement without a measured result.

## Someone is first on the board

Treat the top score as an external signal, not proof that their method generalises. Do not use prohibited evaluation information or copy a technique that conflicts with the rules. Say which of these most likely explains the gap: representation, model capacity, optimisation, preprocessing, feature engineering, validation design, inference, metric alignment, data usage, leaderboard overfitting. Any adopted idea needs a defensible reason, full rule compliance and a separate candidate version.

## Cannot improve further

Do not invent random changes. Review validation strategy, data distribution, target and feature behaviour, error patterns, metric optimisation, inference and generalisation risks; then research only the unresolved bottleneck (`research-gate.md`) and give evidence-backed options, not a long list. The current best stays the protected fallback.

## Candidate comparison

Compare A and B on validation evidence, public score, stability, complexity, training and inference cost, rule compliance, leakage risk, overfitting risk, determinism risk and expected hidden-test generalisation. Choose the strongest overall candidate, not the one with the smallest public-score advantage. Keep both intact.

## Runtime errors

Read the traceback and find the root cause before modifying anything. Fix only what is necessary; keep the algorithm, compliance, compute plan and expected behaviour; do not remove useful functionality to hide the error; return the corrected solution as a separate version. CPU/GPU mismatch: if the challenge is CPU-only make the plan explicitly CPU-only with no device fallback; otherwise keep the model strategy.

## Finalize and determinism retry

`finalize`: freeze the winner, create a separate deterministic file, change nothing but runtime behaviour (`determinism.md`). Retry after a failed check: inspect the exact checker failure first; do not change data, labels, features, preprocessing, model, loss, optimiser, hyper-parameters, split, inference or submission format unless the check genuinely requires it; prefer style and control-flow-only edits (reorganise, make equivalent operations explicit, remove ambiguous runtime branching); create `solution_deterministic_retry.py`; keep the winner and the earlier deterministic file; list every modification as **stylistic only** or **runtime-behavioural**, and say plainly when style-only edits cannot fix the check.

## Final submission check

Do not optimise further and make no speculative change. Verify: 1 challenge rules, 2 data usage, 3 no prohibited leakage, 4 no prohibited synthetic evaluation data, 5 no test-target learning, 6 input/output contract, 7 deterministic plan if required, 8 submission format, 9 fixed device and runtime assumptions, 10 reproducibility. If everything is valid, hand over the exact final file and a short summary of the runtime plan.

## Context hygiene

Reuse the session context; do not redo established analyses; research only a bottleneck; keep the best solution protected; avoid spawning agents for what the main session can do. When the context is large:

```
/compact Preserve the complete challenge rules, current best solution, measured scores, strongest experiments, important failures, current bottleneck, and unresolved errors.
```

## Golden rules (the pack's)

1 Protect the best validated solution; new ideas are candidates until tested. 2 Public score is not hidden-test performance. 3 Research a bottleneck, not the universe. 4 Research after evidence of a plateau. 5 Finalize determinism after the winner is selected. 6 Rules outrank score. 7 No claimed improvement without a measured result. 8 Keep separate files for strong candidates so a regression can be discarded. 9 A top public score is a signal to investigate, not permission to copy or overfit. 10 Use the main solver for hard decisions; use helpers only when they add clear value.
