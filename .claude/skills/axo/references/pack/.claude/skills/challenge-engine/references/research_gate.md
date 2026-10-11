# Targeted Research Gate

Research is an optimization tool, not the default first step.

## Trigger

Enter this gate only after a valid first solution has produced a measured score and one of the following is true:
- score plateau
- unexplained error
- weak validation/generalization evidence
- known task-specific technique likely to improve the current approach

## Budget

Use one focused research pass. Prefer a small number of high-signal sources over broad browsing.

Return no more than 3 candidate changes unless the evidence is unusually strong.

For each candidate record:
1. technique
2. source/evidence
3. why it maps to this task
4. implementation cost
5. expected upside
6. failure risk
7. rule-compliance status

Do not modify the solution merely because a paper reports an improvement on another dataset.

## Token discipline

- Do not repeat searches for the same question.
- Stop when the evidence becomes repetitive.
- Do not research generic ML concepts the agent already knows.
- Research the bottleneck revealed by the score/logs.
