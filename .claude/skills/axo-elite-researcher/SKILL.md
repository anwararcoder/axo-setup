---
name: axo-elite-researcher
description: >
  Research-first workflow for Axo AI/ML challenges fetched into challenges/: scan every challenge folder and
  this PC, rank the open challenges with a transparent risk-adjusted framework, classify each challenge's
  pretrained-model policy (A allowed, B from scratch, C unclear), research the selected task deeply before
  coding, run controlled resource-capped experiments, improve from bot scores, audit, and build a verified
  upload folder. Operations: scan, select, research, solve, improve, audit, finalize, status. Use when the
  user invokes axo-elite-researcher or asks for this research-first workflow (scan or rank all challenges,
  deep research before solving, an evidence-labelled status). The probe-first /axo skill stays separate.
argument-hint: "scan | select | research [slug] | solve [slug] | improve [bot message] | audit | finalize | status"
---

# Axo Elite Researcher

Request: `$ARGUMENTS`

The first word is the operation, the rest is its input (a challenge slug, a pasted bot message, notes). No
operation means `status`. A plain-language request maps to the closest operation ("which one should I do?" is
`select`, a pasted "Submission scored" block is `improve`); say which operation you are running.

## Objective

Maximize the expected score on the hidden (private) evaluation, risk-adjusted, inside the challenge's rules,
with every claim backed by evidence. Speed matters through the clock only (closing time, free seats, credits):
plan by it, never trade a rule or the evidence for it.

## Non-negotiable rules

1. **No submission and no irreversible or outward action without the user's explicit approval.** The user
   uploads; you build folders and say when one is ready. Never spend a credit to test what a local check can
   show. Never push, publish or upload challenge files, data, descriptions or dataset links; never print a
   dataset link in a report; never open `.env`, `*.session`, `~/.kaggle/*` or other credential files. Before
   writing challenge material (reports, data, solutions, logs) inside a git work tree, run `git remote -v` and
   `git check-ignore <path>`. If a remote exists and the path is not ignored, add the path to
   `.git/info/exclude` (local only, never pushed) and tell the user; never edit `.gitignore` or tracked files
   for this without asking. Treat every remote as public until the user says otherwise.
2. **No fabrication.** Every number carries its kind: `measured` (ledger tag or log line), `published` (URL
   and date), `estimate` (its basis) or `bot` (the verbatim message). Code that has not run is "untested". A
   local number is not a board estimate until a scored version has calibrated it.
3. **Rules outrank score.** The challenge's own `description.md` is the law for that challenge and nothing
   carries over from another one. An ambiguous sentence gets the strictest reading; tell the user and ask when
   it matters. A bot score is not approval: people review the code afterwards.
4. **Model policy before model choice.** Classify the challenge A, B or C from quoted sentences
   (`references/select.md`, section 2). Never use a model because it is installed, popular or allowed elsewhere.
5. **This PC freezes when RAM runs out (no swap).** Every local process that loads challenge data or trains
   runs through `scripts/guarded_run.py`: memory cap, refusal when free RAM is short, one GPU job at a time,
   a ledger line per run. Size with a short timed run first. Never raise a cap silently, never run unwrapped.
6. **Protect the best.** A measured candidate file is never edited; changes become new files. `LOG.md` keeps
   one `BEST:` line. A worse result is recorded, analysed and discarded.
7. **Never delete or overwrite user files.** New outputs go to new paths. `rm -rf`, `git reset`, `git clean`,
   force-pushes and history rewrites happen only on the user's explicit request.

## Where things are (verify each path before using it)

| What | Path |
|---|---|
| Workspace | the folder that holds `fetch_challenges.py` and `challenges/` (2026-10-12: `/mnt/work/shipd-ai-eris/axo-setup`) |
| This skill's scripts, `$S` | `.claude/skills/axo-elite-researcher/scripts/` |
| The Axo skill, reused read-only, `$AXO` | `.claude/skills/axo/`: `scripts/gate.py`, `scripts/check_determinism.py`, `scripts/validate_submission.py`, `assets/solution_template.py`, `assets/dev_harness.py`, `references/` |
| One challenge, `$C` | `challenges/<slug>/`. Fetched: `description.md`, `baseline.md`, `public-leaderboard.md`, `kaggle-notebook.md`. Ours: `RULES.md`, `RESEARCH.md`, `LOG.md`, `experiments.jsonl`, `data/`, `solution_vN.py`, `out/`, `submission_vN/` |
| Scan reports | `challenges/_scan/` |
| Parallel workspace, may be active | `/mnt/work/shipd-ai-eris/axo-eris`: `challenges/<slug>/WORKLOG.md`; private Kaggle kernel runner `.claude/skills/challenge-engine/scripts/kaggle_kernel.py` with `references/kaggle_execution.md` |
| ML Python | `/mnt/work/shipd-ai-eris/axo-eris/.venv-ml/bin/python` (torch with CUDA, scikit-learn, lightgbm); confirm with `hw_profile.py` |

## Operations

### scan: inventory challenges and resources (read-only)

1. `python3 $S/hw_profile.py --python <ML python> --torch --disk .` for the machine profile. Compare it with
   the user's stated profile and report every difference.
2. `python3 $S/scan_challenges.py challenges --also <parallel workspace>/challenges --md challenges/_scan/scan_<UTC>.md --json challenges/_scan/scan_<UTC>.json`.
3. Read `description.md` in full for every selectable challenge, the quoted sentences for the rest, and
   confirm or correct each heuristic policy class. Status values are as old as their files: give their age.
   Refreshing them needs `fetch_challenges.py`, which uses the user's Telegram login: only with approval.
4. Report the inventory, the selectable challenges with their gates, the policy class of each (confirmed or
   open), the environment and the missing information. Nothing is downloaded.

### select: rank and recommend one

Follow `references/select.md`: gates, the weighted criteria with one line of evidence per score, the
uncertainty, and the check for work by the parallel session and for submissions on the shared account. Give one
recommendation, the strongest alternative, what would change the decision, and the questions only the user can
answer (live status block, closing time, who made an existing account submission). Download nothing and start
no experiment until the user has picked.

### research: deep research before the main solution

`references/research.md`. First the rule lock (`$C/RULES.md` from `$AXO/references/rule-checklist.md`, every
constraint quoted verbatim, the six questions of `$AXO/references/compliance-firewall.md` answered), then the
data look once the data is on disk, then the research pass, time-boxed by the clock and only where the rules
permit research. Write `$C/RESEARCH.md` from `assets/research_report.md`: sources with URL and date, findings
labelled published, measured or estimate, why each applies here, and a ranked implementation plan.

### solve: build, validate, package

`references/experiments.md`, sections 1 to 3: data verified against the description, the metric
re-implemented and tested, a validation split that mirrors the hidden split, leakage checks, a fast
rule-compliant probe that beats chance (packaged and offered for upload, because only its board score can show
that local validation agrees with the platform), then the research plan's top candidates as one controlled
wave. Every run goes through `guarded_run.py` into `$C/experiments.jsonl`.

### improve: score-driven iteration

`references/experiments.md`, section 4. Record the complete bot message verbatim in `LOG.md`, calibrate
(board minus local, the board's noise, gaps to the baseline, the 10th place and the top), diagnose in order,
re-rank the plan against the moving cutoff, build the top item as a new candidate, validate, audit, package,
and ask before any upload.

### audit: rules and submission integrity

`references/audit-finalize.md`, checklist A, on the exact files to be uploaded. Findings ranked by severity,
each with the rule or code line it rests on; "no findings" states what was checked.

### finalize: the verified upload folder

`references/audit-finalize.md`, checklist B: freeze the winner, make the deterministic copy, run
`$AXO/scripts/gate.py` (five stages) inside `guarded_run.py`, build `$C/submission_vN/` with its
`HOW_TO_SUBMIT.md`, audit it. Then stop: say it is ready and wait for the user's decision.

### status: where we are

From `LOG.md`, `experiments.jsonl`, the latest scan and the running processes: the challenge, the clock, the
`BEST:` line with its label, the last submission and its score, what is running, the next action and what is
needed from the user. At most ten lines.

## Reference files

- `references/select.md`: inventory fields, policy classes A/B/C, gates, the ranking criteria, hardware fit.
- `references/research.md`: when and how long, what may be researched, the questions, sources, A and B specifics.
- `references/experiments.md`: the pipeline, running on this PC, evidence and logging, the score loop, credits.
- `references/audit-finalize.md`: the audit checklist, finalization with the Axo gate, limits of reused scripts.
- `references/prior-knowledge.md`: lessons, platform facts and machine facts reused from the Axo skill, with
  the inconsistencies found in it. Read at `select` and `research`, and verify each item before applying it.
- `assets/research_report.md`, `assets/challenge_log.md`: templates for `RESEARCH.md` and `LOG.md`.

## Every reply

Short: the user's limit is 1500 characters. Lead with the decision-relevant number and its label, say what was
and was not run, and put the details in the challenge's files.
