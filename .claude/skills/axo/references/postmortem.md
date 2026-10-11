# Postmortem and maintenance

Source: the challenge-engine pack (`references/pack/`). Record only what the results support; keep permanent files short.

## When

After the challenge closes, or when the user says the challenge is finished or to remove it. Do it briefly; the user may have said no more work on the challenge, and a postmortem is not more work on it.

## Two lists, kept apart

**A. Challenge-specific facts** (stay with the challenge and its recipe entry, never in permanent instructions): final score and rank, the best version and what made it, failed ideas, errors, rule and compliance lessons, validation lessons, overfitting risks, determinism lessons, time to first package, full-run time, experiment throughput, local against official results.

**B. Reusable principles:** only those clearly transferable across challenges. Mark each CONFIRMED (results show it) or HYPOTHESIS. Promote into `lessons.md` only when it is clearly reusable or the user confirms.

## Rules

- At most 3 lessons per review, each as "what happened -> what it cost -> the habit that prevents it".
- Do not add a rule because it appeared once; do not invent lessons the results do not support; do not rewrite the skill after one noisy result.
- Do not rewrite the active solution.
- Where a lesson belongs: recipe entry (method facts), `lessons.md` (habits), memory (user preferences), or nowhere.
- Keep the files concise. If `lessons.md` grows, merge duplicates instead of appending.

## Maintenance review (the pack's loop prompt, run by hand)

Review the finished postmortems only when there is new evidence since the last review. Extract at most 3 genuinely reusable lessons; for each, state the evidence, say whether it is confirmed or a hypothesis, and decide where it belongs. Do not change working rules for style, do not add challenge-specific detail to global instructions, do not create new solution ideas unless a failure pattern repeats. Keep the pass small. A recurring `/loop` is optional and only for a dedicated maintenance session; never keep one running during development. A manual postmortem per finished challenge is cheaper and more predictable.

## Postmortem role (subagent, when the user asks)

"Review the completed challenge result and extract reusable lessons. Only record lessons that are clearly transferable. Separate confirmed facts, hypotheses, and challenge-specific details that should not become global rules. Focus on rule and compliance lessons, leakage patterns, validation patterns, runtime and determinism failures, model-selection mistakes and useful solution patterns. Do not rewrite the main skill."
