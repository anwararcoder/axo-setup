# Targeted research gate

Source: the challenge-engine pack (`references/pack/`). Research is an optimisation tool, not a first step.

## Trigger

Two cases. **Default, right after the probe is handed over (the user's instruction, 2026-10-09):** one short methods-only pass, about 3 searches and 10 minutes, before version 2 is designed, for any task type or domain that a recipe does not already cover; its one or two best-supported ideas enter the ranked list of `climb.md`. The user's words after three versions stuck at 0.385 to 0.39: "you didn't research what approaches work best on this task from scratch and that really is hurting my standing". **Otherwise** only after a valid solution has a measured score, and when one of these holds:

- a plateau: two strong attempts failed, or no credible next improvement is visible;
- an error the code and logs do not explain;
- weak validation evidence, or a large gap between validation and the board;
- a specialised method family that the current solution lacks;
- one technical fact is needed for compliance or implementation.

Never before the probe exists (the probe goes first), and never because a challenge is merely hard.

## Limits

- The rules must permit it. Research methods only: never search for challenge content (texts, fragments, ids), other solvers' code or leaderboards, or datasets, and honour any "searching ..." prohibition in the description.
- One focused pass: a few high-signal sources, at most 3 candidates, at most 1-2 changes adopted unless the evidence is unusually strong. Our margin: about 10-15 minutes. Stop when the evidence becomes repetitive; do not repeat a search; do not research generic ML the agent already knows.
- Research the bottleneck the score and logs revealed, not "the best solution".

## Memo (compact)

For each candidate: technique; source and evidence; why it maps to this exact task; implementation cost; expected upside; failure and overfitting risk; rule-compliance status. A paper's gain on another dataset is not evidence for this one.

## After the memo

Keep the current best untouched and implement the candidate as a separate version. Adopt it only on a measured result. Do not rewrite the winner because research found a new idea.

## Researcher role (subagent, when the user asks)

"Research one concrete bottleneck. No literature review. Do not rewrite the solution. Do not invent benchmark results. Do not recommend anything that violates the challenge description. Return at most 3 candidate interventions with evidence, exact relevance, expected upside, risk, implementation cost and a rule-compliance assessment. Stop when evidence becomes repetitive."
