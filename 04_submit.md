How to submit

What to submit
1. Submission file: the prediction/output file used for evaluation.
2. Solution/code file: the code used to produce the submission.
You pick the challenge by its name from a list, so there is no ID to copy. Always send the files for the challenge you picked.

Solution description
Describe your approach completely and accurately. The exception: if the important approach and reasoning are already clearly explained inside the code itself, a separate description is not required. The work should be understandable and reproducible from what you send.
Also tell us the Spent Time: the approximate time you spent on the challenge.

Notebook / script quality
Remove unnecessary cells and irrelevant setup: downloading the dataset, extracting archives, repeated setup steps, unrelated experiments, or anything not needed to reproduce the actual solution. Keep it focused on the problem.

Dataset access pattern
The solution must be self-contained and run end-to-end without manual intervention, using only the standard libraries of the configured Kaggle Docker runtime. A common expected pattern:
import sys
from pathlib import Path
import pandas as pd

public_dir = (
    Path(sys.argv[1])
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("-f")
    else Path("/kaggle/input/datasets/<owner>/<challenge-dataset>")
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
Adapt the dataset path and filenames to the actual challenge, and use one variable name consistently (if it is public_dir, never later write PUBLIC_DIR). The requirements: accept the runtime input path, keep a safe Kaggle fallback, and write the final submission to the expected output location.

Runtime expectations
• runs end-to-end with no manual steps,
• only standard Kaggle Docker libraries,
• targets about 1 hour of runtime on the configured environment,
• comments explain the reasoning behind important steps.
Avoid large unnecessary dependencies and unnecessary complexity.

After you submit
• Score processing may take about 15-30 minutes. No exact time is promised. You will get the status or a review message here when it is available.
• Below the AI Baseline? That only means this version has not beaten the benchmark. Improve it and submit again if credits and the challenge state allow, or move to another challenge if more work is not practical. One unsuccessful submission does not end your participation in Axo.
• Review / rejection: the team can review a submission. If it has an error or fails a requirement you get a message with the reason (for example: not following the challenge rules, failing a required check, an invalid submission structure, or an attempt to manipulate the scoring). Fix it and submit again when possible. Axo does not guarantee that every submitted solution is accepted.
• Score problems: Axo cannot guarantee that every score calculation succeeds immediately or that every platform-side issue can be resolved. Use the message you receive to understand the issue. If it cannot be fixed within the challenge window, moving to another challenge may be the practical option.


Repeated submissions
You may be able to submit several versions of the same challenge (Version 1, Version 2, Version 3...). Each one is treated as a separate submission. Submissions are not unlimited: whether another one is possible depends on the challenge rules, your remaining submission credits, the challenge state and submission capacity.

Submission credits
Maximum credits per problem   6
Refill rate                   1 credit every 4 hours
Full refill                   24 hours
Scope                         per problem
Consumed by                   submit only
Draft/check/edit cost         0 credits
A credit is used only when a submission is actually submitted. Drafts, checks and edits cost nothing. When the live value is known, it is shown to you instead of these defaults.

Ready? Tap the button below and I will guide you step by step. You can stop at any time with /cancel.