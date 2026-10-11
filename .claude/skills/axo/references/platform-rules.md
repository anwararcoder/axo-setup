# Axo platform rules

Captured from the Axo Telegram bot (`/instructions`, `/submit`, `/rules`) at version 1.7.1. The bot is the authority: if a live message says something different, the live message wins. Re-check with `/news` when a challenge starts.

## Contents
- How challenges work
- Capacity, states and closing time
- What to submit
- Script requirements
- Submission credits
- After submitting

## How challenges work
- Each challenge has a dataset, a description (markdown), evaluation requirements and an **AI Baseline** score.
- The goal is a solution that beats the AI Baseline while following the challenge rules.
- AI tools (Claude, ChatGPT, Gemini, Cursor, Copilot) are allowed in every challenge.
- "The objective is an effective, clean, reasonable solution, not unnecessary complexity."
- Public portion: for development. Private portion: hidden, used for the final result. Beating the public score does not guarantee the private one; build for generalisation.
- Private data must never be shared, published, redistributed or exposed.

## Capacity, states and closing time
- Solver capacity is **10**. A challenge can become full, change state, or close; do not assume it stays open.
- After a successful scored submission you can keep improving. If capacity fills later, an already-submitted participant gets a limited extra window — **1 hour** after the full/closing state is triggered in the current workflow.
- Every challenge has a **Closing Time**. After it, the private leaderboard is announced and nothing more can be submitted.
- You only get the private/final result if you made a submission whose score was successfully calculated. An upload attempt is not a successful submission.

## What to submit
1. **Submission file**: the prediction/output file used for evaluation.
2. **Solution/code file**: the code that produced the submission.
3. **Solution description**: complete and accurate. Not required separately if the approach and reasoning are clearly explained inside the code.
4. **Spent Time**: approximate time spent on the challenge.

The challenge is picked by name from a list in the bot; there is no ID to copy. Send the files for the challenge you picked.

## Script requirements
- Self-contained; runs end to end with no manual intervention.
- Only the standard libraries of the configured Kaggle Docker runtime. What that image contains and what it lacks (no pretrained weights, no network): `target-env.md`.
- Targets **about 1 hour** of runtime on the configured environment (the challenge markdown may state its own hard limit; honour the stricter one and record both).
- Comments explain the reasoning behind important steps.
- No unnecessary cells or setup: no dataset download, no archive extraction, no repeated setup, no unrelated experiments.
- Required input/output pattern (adapt path and filenames; use one variable name consistently — if it is `public_dir`, never later write `PUBLIC_DIR`):

```python
import sys
from pathlib import Path
import pandas as pd

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

# Train and predict...

submission_out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_out, index=False)
```

The three requirements behind the pattern: accept the runtime input path, keep a safe Kaggle fallback, write the final submission to the expected output location.

Path note (2026-10-05): the fallback `/kaggle/working/extracted_dataset` is the user's own pattern (the challenge-engine pack, `references/pack/`, which the user declared more trusted); the bot's example showed the placeholder `/kaggle/input/datasets/<owner>/<challenge-dataset>`. The platform passes both paths as arguments when it runs the script, so the fallback serves notebook runs. Never discover the dataset by searching `/kaggle/input` (that makes the plan depend on the environment); a live bot message that shows another path wins. The script must also run when pasted into one notebook cell: arguments are tolerated, never required.

## Submission credits

| Item | Value |
|---|---|
| Maximum credits per problem | 6 |
| Refill rate | 1 credit every 4 hours |
| Full refill | 24 hours |
| Scope | per problem |
| Consumed by | submit only |
| Draft / check / edit cost | 0 credits |

Each version (1, 2, 3…) is a separate submission. Whether another is possible depends on the challenge rules, remaining credits, challenge state and capacity.

## After submitting
- Scoring takes roughly 15–30 minutes; no exact time is promised.
- Below the baseline: improve and resubmit if credits and state allow, or move on.
- The team can review and reject a submission (rule violation, failed required check, invalid structure, attempt to manipulate scoring). Fix and resubmit when possible.
- Score calculation is not guaranteed to succeed; if it cannot be fixed within the window, move to another challenge.

## First versions, the grace hour and the board (answered by the user, 2026-10-04)

1. **Grace hour:** a solver who has **beaten the baseline** gets one hour to keep submitting after the slots are filled. A scored version below the baseline does not earn it. So a first version under the baseline buys feedback only; the place and the grace hour come from the first baseline-beating version, which stays urgent.
2. **The board keeps a solver's best version.** A new version can never lower the standing, so every version that is better locally is worth a credit.
3. **Reviewers do not mind a below-baseline first version.**

One observation not explained by rule 1: on 2026-10-04, with version 1 scored above the baseline, version 2 was refused with "This challenge is closed. Nothing was sent." a few minutes after the board showed 9/10. Treat the grace hour as likely, not guaranteed, and still hand over at once from 8/10.

## Dataset download
The challenge post gives a `curl` line with a short link. **The link refreshes every hour**; an old one returns an "expired" error. Links are per challenge — a link copied from a different post downloads a different dataset.

## When the slots fill (bot messages, 2026-10-06, The Enclosure Mention)

- "All solver slots are taken. For one hour, anyone can still submit and edit: you do not need an earlier submission, and it does not matter whether you are above or below the AI baseline."
- "The cutoff is the end of the hour. A submission sent before it gets up to one more hour to finish scoring, but scoring takes about 15-30 minutes."
- "Then comes the lockdown hour: nobody can submit or edit while the top 10 are picked." "Only those top 10 keep submitting until the challenge closes."
- So a solver safely inside the top 10 does not need a rushed version in that hour; the better version can go in after the lockdown. This is newer than the 2026-10-04 note that only baseline-beating solvers get the extra hour.

## Grace/lockdown reconfirmation (user-forwarded bot, 2026-10-07)

FlockSight explicitly advertised a grace hour open to everyone, regardless of prior submission or baseline status. Submissions/edits stop at its end; a submission sent before the cutoff may get up to one additional hour to finish scoring. Then comes a lockdown hour while the top10 are picked, and only those10 keep submitting until final closing. This reconfirms the2026-10-06 bot evidence and supersedes the older baseline-only grace account. Live announcements and their actual cutoff timestamps remain authoritative; a forwarded countdown's age may be unknown. Above-baseline score is not a guaranteed place.

## Platform name in the solution file (bot, 2026-10-10, Station Siting)

The bot refuses a solution file that contains the platform name in any form, upper or lower case: a comment, a variable or function name, a path. It names the line and asks for a corrected file. `scripts/build_submission_folder.py` fails the static stage on it; keep the name out of the description file as well.

## The automatic "Prompt Compliance" check (bot, 2026-10-10 and 11, Station Siting)

It runs before scoring and answers with one generic text ("could not confirm that your solution follows the explicit implementation requirements"); the credit is returned and a retry is allowed after five minutes. It stopped unchanged, compliant files several times in a row and accepted them later. Write per-test-row rules visibly in the code (one row at a time, test data read after training) and allow for several retries before the closing time.
