# How to submit — Who Cited Whom? Restoring the Lost Directions of a Citation Graph from Its Shape and a Few Surviving Arrows

This file is your guide. **Do not upload it.** Everything to upload is in this folder and is final: all checks passed when the folder was built (2026-10-11 23:01).

## What to upload

| # | File in this folder | What the bot calls it |
|---|---|---|
| 1 | `1_submission.csv` | Submission file (the predictions) |
| 2 | `2_solution_v1.py` | Solution / code file |
| 3 | `3_SOLUTION_DESCRIPTION.md` | Solution description |

And one value to type:

| Item | Value to enter |
|---|---|
| Spent time | **0.5 hours** |

## Steps

1. Open the Axo bot in Telegram and send `/submit`.
2. Tap the button to start. You can stop at any time with `/cancel`; nothing is used until the final submit.
3. Pick the challenge from the list by name: **Who Cited Whom? Restoring the Lost Directions of a Citation Graph from Its Shape and a Few Surviving Arrows**. Check the name before continuing — files sent to the wrong challenge waste a credit.
4. When the bot asks for the **submission file**, upload `1_submission.csv`.
5. When the bot asks for the **solution / code file**, upload `2_solution_v1.py`.
6. When the bot asks for the **solution description**, send `3_SOLUTION_DESCRIPTION.md` — upload the file, or if the bot only accepts text, open it and paste its contents.
7. When the bot asks for the **spent time**, type: `0.5 hours`
8. Review what the bot shows, then confirm the submit. This is the step that uses one credit.

The bot leads the conversation, so if it asks for these in a different order, follow its order and send the matching item.

## After submitting

- Scoring usually takes 15–30 minutes. The result arrives as a message from the bot.
- Local score of this version: **0.4077** (AI Baseline: 0.5478). The local score is an estimate; the leaderboard score can differ.
- **This is version 1, a probe**: a fast first version sent to check local validation against the board. Its local score is not yet checked against the board. Locally it is NOT above the baseline; beating it is the job of the next version.
- Credits: 6 maximum per challenge, 1 refilled every 4 hours. Only the final submit uses one.
- If the bot rejects the submission, send me the exact message.

## File fingerprints

So you can confirm you are uploading the files that were checked:

| File | Size | SHA-256 |
|---|---|---|
| `1_submission.csv` | 353,616 bytes | `0de13c112ed2e3e582d47c55f913331be5a2c8069f79a2f34adf0190073ceadb` |
| `2_solution_v1.py` | 14,621 bytes | `bbbee576f38b950beee4fa08cae49d46e0a587fff83a9c3e818cf06a36a46174` |
| `3_SOLUTION_DESCRIPTION.md` | 1,433 bytes | `bfe23742e6061cc966ac5ec6ea3b2e2fbd5d936f89a4a9fe030eed46f0afdfd2` |

## Checks passed when this folder was built

- Submission file matches the sample: same header, same ids, none missing or duplicated.
- Solution uses the required `public_dir` / `submission_out` pattern with one spelling.
- Solution opens with a docstring explaining the approach.
- Solution has no run-time-dependent device, path, import or retry branching.
- Solution has no download, extraction, debug or development-only code.
- Two full runs of the solution produced byte-identical submissions.
- Separate solution description present with Approach, Validation, Rule compliance and Runtime.
- First scored version: local score 0.4077 is above the chance level 0.0.
