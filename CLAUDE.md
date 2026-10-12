# Axo challenge workspace (axo-setup)

Workspace for Axo AI/ML challenges: fetch the challenge posts, pick one, solve it on this PC, and hand the user a
ready-to-upload folder. The user uploads through the Axo Telegram bot; Claude never submits or spends a credit.

## Layout

- `fetch_challenges.py`: pulls the latest N posts of the "AXO Challanges" Telegram channel into
  `challenges/<slug>/` (`.venv/bin/python fetch_challenges.py N`). It logs into the user's own Telegram account and
  may ask for a login code: run it only when the user asks.
- `challenges/<slug>/`, written by the fetcher: `description.md` (the challenge's rules, authoritative),
  `baseline.md` (domain, level, compute, baseline), `public-leaderboard.md` (state, credits, closing countdown,
  board; as old as the file), `kaggle-notebook.md` (download cells; the link expires every hour) and the archived
  post `<date>_msg<id>.md`. Our files go in the same folder: `RULES.md`, `RESEARCH.md`, `LOG.md`, `data/`,
  `solution_vN.py`, `out/`, `submission_vN/`.
- `challenges/_scan/`: scan and ranking reports.
- `00_start.md` to `08_news.md`: the bot's pages as captured on 2026-10-06. Lifecycle in `01_instructions.md`,
  confidentiality and AI use in `02_rules.md`, requirements and credits in `04_submit.md`, prizes in
  `05_prizes.md`. A live bot message wins over them.

## Skills

| Situation | Use |
|---|---|
| New work: scan and rank every challenge, research first, guarded local runs, audit, final folder | `/axo-elite-researcher <scan, select, research, solve, improve, audit, finalize, status>` |
| The probe-first workflow, its recipes, lessons and postmortems | `/axo`. Its machine specs (RTX 3060) and Windows helpers are stale: see `.claude/skills/axo-elite-researcher/references/prior-knowledge.md` |

Both build upload folders with `.claude/skills/axo/scripts/gate.py`. Do not edit the `axo` skill unless the user asks.

## This machine

- Ryzen 5 5600H (6 cores, 12 threads), 15 GiB RAM and no swap, RTX 3050 Laptop GPU with 4 GB (3.7 GiB usable),
  Zorin OS 18.1. When RAM runs out the desktop freezes.
- Every local command that loads challenge data or trains runs through the guarded runner, including the `axo`
  skill's `gate.py` and `run_queue.py`:
  `python3 .claude/skills/axo-elite-researcher/scripts/guarded_run.py --tag <name> [--gpu] [--mem 2G] -- <command>`.
  One GPU job at a time; never raise a cap without telling the user; never run such a command unwrapped.
- ML Python: `/mnt/work/shipd-ai-eris/axo-eris/.venv-ml/bin/python` (torch with CUDA, scikit-learn, lightgbm,
  transformers). `.venv/` here only serves the fetcher. Install nothing without asking.
- Current state: `python3 .claude/skills/axo-elite-researcher/scripts/hw_profile.py --python <ML Python> --torch`.

## The parallel workspace

Another Claude session works in `/mnt/work/shipd-ai-eris/axo-eris` at the same time, on the same GPU, RAM and
Axo account, so "Submitted: Yes", an "Axo solver" row and the credits may belong to other work. Before working on a
challenge, read `axo-eris/challenges/<slug>/WORKLOG.md` and ask the user when it overlaps. Its private Kaggle
kernel runner (`.claude/skills/challenge-engine/scripts/kaggle_kernel.py`, rules in
`references/kaggle_execution.md`) is the remote GPU option; ask before using it. Do not edit files there.

## Secrets, confidentiality and git

- Never open or print `.env`, `*.session` or `~/.kaggle/*`.
- Challenge data, descriptions, dataset links, submissions and solution code are confidential (`02_rules.md`):
  never upload, publish or paste them outside this machine, and never put a dataset link in a report.
- This folder is a git repository with a GitHub remote. Commit or push only when the user asks, never with
  `git add -A`, and never commit credentials, `.venv/` or challenge material. Keep new outputs that hold challenge
  material out of git by adding their paths to `.git/info/exclude`.

## Platform facts not in the bot pages (2026-10; the live bot wins)

- The bot takes the solution description as text of at most 1500 characters (recorded in the axo-eris workspace).
- The solution file must not contain the platform's name in any form; the bot refuses it.
- Before scoring, automated checks run: Prompt Compliance, Deterministic Execution, CSV Score Validation. A generic
  block returns the credit and allows a retry after five minutes.
- The fallback data path in solutions is `/kaggle/working/extracted_dataset` (the user's pattern), the output
  `/kaggle/working/submission.csv`; arguments are accepted, never required.

## Working with the user

- The user pastes bot messages as they are; record them verbatim in the challenge's `LOG.md`.
- Keep replies under 1500 characters; details go into the challenge's files.
- Ask before: running the fetcher, downloading a dataset before a challenge is chosen, any Kaggle run, anything
  that touches git history or the remote.
