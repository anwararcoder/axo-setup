"""Assemble the ready-to-upload submission folder for an Axo challenge.

    python build_submission_folder.py \
        --challenge "Exact challenge name as shown in the bot" \
        --solution solution.py \
        --submission out/submission.csv \
        --sample data/sample_submission.csv \
        --description SOLUTION_DESCRIPTION.md \
        --spent-time "6 hours" \
        --local-score 0.7259 --baseline 0.5412 \
        [--json-column frames_json] [--out submission] [--dev-marker NEG_DEV ...]

Creates <out>/ containing exactly:
    1_<submission file>          prediction file          (upload)
    2_<solution file>            documented solution code (upload)
    3_SOLUTION_DESCRIPTION.md    separate description     (upload / paste)
    HOW_TO_SUBMIT.md             step-by-step guide for the user (NOT uploaded)

The folder is only built when every check passes. Any failure stops with a list
of what to fix and leaves no folder behind, so a folder that exists is ready to
send with nothing left open.
"""
import argparse
import hashlib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_submission import validate  # noqa: E402

GUIDE = """# How to submit — {challenge}

This file is your guide. **Do not upload it.** Everything to upload is in this folder and is final: all checks passed when the folder was built ({built}).

## What to upload

| # | File in this folder | What the bot calls it |
|---|---|---|
| 1 | `{submission_name}` | Submission file (the predictions) |
| 2 | `{solution_name}` | Solution / code file |
| 3 | `{description_name}` | Solution description |

And one value to type:

| Item | Value to enter |
|---|---|
| Spent time | **{spent_time}** |

## Steps

1. Open the Axo bot in Telegram and send `/submit`.
2. Tap the button to start. You can stop at any time with `/cancel`; nothing is used until the final submit.
3. Pick the challenge from the list by name: **{challenge}**. Check the name before continuing — files sent to the wrong challenge waste a credit.
4. When the bot asks for the **submission file**, upload `{submission_name}`.
5. When the bot asks for the **solution / code file**, upload `{solution_name}`.
6. When the bot asks for the **solution description**, send `{description_name}` — upload the file, or if the bot only accepts text, open it and paste its contents.
7. When the bot asks for the **spent time**, type: `{spent_time}`
8. Review what the bot shows, then confirm the submit. This is the step that uses one credit.

The bot leads the conversation, so if it asks for these in a different order, follow its order and send the matching item.

## After submitting

- Scoring usually takes 15–30 minutes. The result arrives as a message from the bot.
- Local score of this version: **{local_score}** (AI Baseline: {baseline}). The local score is an estimate; the leaderboard score can differ.{probe_note}
- Credits: 6 maximum per challenge, 1 refilled every 4 hours. Only the final submit uses one.
- If the bot rejects the submission, send me the exact message.

## File fingerprints

So you can confirm you are uploading the files that were checked:

| File | Size | SHA-256 |
|---|---|---|
{fingerprints}

## Checks passed when this folder was built

{checks}
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Run-time-dependent execution paths (the challenge-engine pack, deterministic_finalization.md): a submitted
# script follows one fixed plan. A line that really is a fixed plan carries the marker `fixed-plan-ok`
# with a reason, e.g. `# fixed-plan-ok: both branches build the same model`.
RUNTIME_BRANCH_PATTERNS = (
    (r"\.cuda\.is_available\s*\(", "the device is chosen from CUDA availability; use one fixed device"),
    (r"device_count\s*\(", "the plan depends on the GPU count"),
    (r"(getenv|environ\.get)\(\s*['\"]CUDA_VISIBLE_DEVICES", "the plan depends on CUDA_VISIBLE_DEVICES"),
    (r"\bcpu_count\s*\(", "the plan depends on the CPU count"),
    (r"(rglob|glob|walk|listdir|iterdir|scandir)\s*\([^)\n]*kaggle/input|kaggle/input[^\n]*\.(rglob|glob|iterdir|scandir)\s*\(",
     "behaviour depends on files discovered under /kaggle/input"),
    (r"except\s*\(?\s*(ImportError|ModuleNotFoundError)", "behaviour depends on an optional import"),
    (r"OutOfMemoryError|out of memory", "an out-of-memory handler that changes the plan"),
    (r"\bargparse\b|\bArgumentParser\b", "command-line parsing; the notebook path must not need arguments"),
    (r"environ\.setdefault\s*\(", "an environment setting applied only when it is not already set; assign it"),
    (r"\bfor\s+[\w, ]+\s+in\s+set\s*\(|\bfor\s+[\w, ]+\s+in\s+\{[^}:]*\bfor\b", "iteration over a set; iterate over sorted(...)"),
    (r"get_device_name|get_device_properties|current_device\s*\(|mem_get_info", "reads the hardware at run time"),
    (r"warn_only\s*=\s*True", "deterministic algorithms in warn-only mode"),
    (r"num_workers\s*=\s*[1-9]", "data-loader workers; use num_workers=0"),
    (r"\bautocast\b|GradScaler", "mixed precision; one fixed full-precision path unless proven and stated"),
    (r"\b(if|elif|while)\b[^\n]*time\.(time|perf_counter|monotonic)\(\)", "the clock is used in a condition"),
)
# Settings a script that imports torch must contain (determinism.md, section 8). Missing ones stop the gate.
TORCH_REQUIRED = (
    ("CUBLAS_WORKSPACE_CONFIG", 'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" before torch is imported'),
    ("manual_seed", "torch.manual_seed(seed)"),
    ("use_deterministic_algorithms(True)", "torch.use_deterministic_algorithms(True)"),
    ("cudnn.deterministic = True", "torch.backends.cudnn.deterministic = True"),
    ("cudnn.benchmark = False", "torch.backends.cudnn.benchmark = False"),
    ("matmul.allow_tf32 = False", "torch.backends.cuda.matmul.allow_tf32 = False"),
    ("cudnn.allow_tf32 = False", "torch.backends.cudnn.allow_tf32 = False"),
)


def code_problems(code, dev_markers=()):
    """Static checks of the solution source. Returns (problems, passed_checks).
    Also used by gate.py so these fail in seconds, before any long run."""
    problems, checks = [], []
    if "public_dir" not in code or "submission_out" not in code or "sys.argv" not in code:
        problems.append("solution: the public_dir / submission_out argv pattern is missing")
    elif re.search(r"\b(PUBLIC_DIR|SUBMISSION_OUT|Public_Dir|publicDir)\b", code):
        problems.append("solution: the path variable is spelled in more than one way")
    else:
        checks.append("Solution uses the required `public_dir` / `submission_out` pattern with one spelling.")
    # The bot refuses a file that holds the platform's name in any form (Station Siting, 2026-10-10: a comment).
    named = [str(n) for n, l in enumerate(code.splitlines(), 1) if "axo" in l.lower()]
    if named:
        problems.append("solution: the platform name appears (any case, any form) on line(s) " + ", ".join(named[:8]))
    if "<owner>" in code or "<challenge-dataset>" in code:
        problems.append("solution: the Kaggle fallback path still contains template placeholders")
    if code.startswith("\ufeff"):
        problems.append("solution: the file starts with a byte-order mark (written by PowerShell Set-Content?); "
                        "rewrite it as plain UTF-8 with LF line ends")
    if "\r\n" in code:
        problems.append("solution: CRLF line ends; rewrite with LF")
    if re.search(r"^\s*(import|from)\s+torch\b", code, re.M):
        missing = [how for needle, how in TORCH_REQUIRED if needle not in code]
        if missing:
            problems.append("solution: determinism settings missing (determinism.md, section 8): " + "; ".join(missing))
        else:
            checks.append("Solution contains the full set of deterministic backend settings.")
    if "eterministic" not in code.split('"""')[1] if code.lstrip("\ufeff").lstrip().startswith('"""') and code.count('"""') >= 2 else False:
        problems.append("solution: the top docstring does not state the fixed, deterministic runtime plan "
                        "(device, seeds, passes, backends, no hash-order dependence)")
    if not code.lstrip().startswith(('"""', "'''")):
        problems.append("solution: no top-of-file docstring explaining the approach")
    else:
        checks.append("Solution opens with a docstring explaining the approach.")
    for pattern, what in ((r"curl |wget |urlretrieve|requests\.get", "download code"),
                          (r"\bunzip\b|zipfile|extractall", "archive extraction code"),
                          (r"TODO|PLACEHOLDER|\bFIXME\b|\bXXX\b|FILL IN", "TODO/FIXME/FILL IN markers"),
                          (r"breakpoint\(\)|pdb\.set_trace", "debugger calls")):
        # A dataset whose inputs are supplied inside an archive (Pottery Sherds: images in inputs.zip) has to be
        # read from it in memory: such lines carry `# supplied-archive-ok: <reason>` and nothing is extracted to disk.
        scanned = "\n".join(l for l in code.splitlines() if "supplied-archive-ok" not in l) if "archive" in what else code
        if re.search(pattern, scanned):
            problems.append(f"solution: contains {what}")
    branching = []
    for number, line in enumerate(code.splitlines(), 1):
        if "fixed-plan-ok" in line:
            continue
        for pattern, what in RUNTIME_BRANCH_PATTERNS:
            if re.search(pattern, line):
                branching.append(f"line {number}: {what}")
    if branching:
        problems.append("solution: run-time-dependent execution path (mark a genuinely fixed line with "
                        "'# fixed-plan-ok: <reason>'): " + "; ".join(branching[:6]))
    else:
        checks.append("Solution has no run-time-dependent device, path, import or retry branching.")
    for marker in dev_markers:
        if marker in code:
            problems.append(f"solution: still contains development code ({marker!r})")
    if not any(p.startswith("solution: contains") or "development code" in p for p in problems):
        checks.append("Solution has no download, extraction, debug or development-only code.")
    return problems, checks


def description_problems(text):
    """Checks of the separate solution description. Returns (problems, passed_checks)."""
    problems, checks = [], []
    missing = [h for h in ("Approach", "Validation", "Rule compliance", "Runtime") if h.lower() not in text.lower()]
    if missing:
        problems.append(f"description: missing section(s) {missing}")
    if re.search(r"<[^>\n]{2,40}>|\bTBD\b|TODO|PLACEHOLDER|FILL IN", text):
        problems.append("description: contains unfilled placeholders")
    if len(text.split()) < 80:
        problems.append("description: too short to be a complete account (under 80 words)")
    if not missing:
        checks.append("Separate solution description present with Approach, Validation, Rule compliance and Runtime.")
    return problems, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--spent-time", required=True)
    ap.add_argument("--local-score", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--json-column")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--dev-marker", action="append", default=[],
                    help="text that must NOT appear in the solution (development switches, debug code)")
    ap.add_argument("--determinism-log",
                    help="file holding the output of check_determinism.py for this solution")
    ap.add_argument("--probe", action="store_true",
                    help="version 1, the fast probe: the local score must beat --chance, not the baseline")
    ap.add_argument("--chance")
    ap.add_argument("--below-baseline-ok", action="store_true",
                    help="a later version handed over below the baseline at the user's request")
    a = ap.parse_args()
    if (a.probe or a.below_baseline_ok) and a.chance is None:
        ap.error("--probe / --below-baseline-ok need --chance")

    solution, submission, description = Path(a.solution), Path(a.submission), Path(a.description)
    problems, checks = [], []

    for label, path in (("solution", solution), ("submission", submission),
                        ("sample", Path(a.sample)), ("description", description)):
        if not path.is_file():
            problems.append(f"{label} file not found: {path}")
    if problems:
        sys.exit("NOT BUILT\n - " + "\n - ".join(problems))

    # 1. submission structure
    found = validate(str(submission), a.sample, a.json_column)
    if found:
        problems += [f"submission: {p}" for p in found]
    else:
        checks.append("Submission file matches the sample: same header, same ids, none missing or duplicated"
                      + (f", every `{a.json_column}` value is valid JSON." if a.json_column else "."))

    # 2. solution code
    found, passed = code_problems(solution.read_text(encoding="utf-8"), a.dev_marker)
    problems += found
    checks += passed

    # 3. determinism evidence
    if not a.determinism_log:
        problems.append("no --determinism-log: run check_determinism.py on this solution and pass its output file")
    else:
        log = Path(a.determinism_log)
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        if "IDENTICAL" not in text or "DIFFERENT" in text:
            problems.append("determinism log does not show IDENTICAL")
        else:
            checks.append("Two full runs of the solution produced byte-identical submissions.")

    # 4. description
    found, passed = description_problems(description.read_text(encoding="utf-8"))
    problems += found
    checks += passed

    # 5. score
    probe_note = ""
    try:
        if a.probe:
            beats = float(a.local_score) > float(a.baseline)
            if float(a.local_score) <= float(a.chance):
                problems.append(f"local score {a.local_score} is not above the chance level {a.chance}")
            else:
                checks.append(f"First scored version: local score {a.local_score} is above the chance level {a.chance}.")
            probe_note = ("\n- **This is version 1, a probe**: a fast first version sent to check local validation "
                          "against the board. Its local score is not yet checked against the board. Locally it is "
                          + ("above" if beats else "NOT above") + " the baseline; beating it is the job of the next version.")
        elif a.below_baseline_ok:
            if float(a.local_score) <= float(a.chance):
                problems.append(f"local score {a.local_score} is not above the chance level {a.chance}")
            else:
                checks.append(f"Local score {a.local_score} is above the chance level {a.chance}.")
            if float(a.local_score) <= float(a.baseline):
                probe_note = ("\n- **Below the AI Baseline locally**: this version was built at your request although "
                              f"its local score ({a.local_score}) is under the baseline ({a.baseline}). The board "
                              "keeps your best version, so uploading it costs one credit and can only help if it "
                              "scores above your current best.")
        elif float(a.local_score) <= float(a.baseline):
            problems.append(f"local score {a.local_score} is not above the baseline {a.baseline}")
        else:
            checks.append(f"Local score {a.local_score} is above the AI Baseline {a.baseline}.")
    except ValueError:
        problems.append("--local-score, --baseline and --chance must be numbers")

    if problems:
        sys.exit("NOT BUILT - fix these first:\n - " + "\n - ".join(problems))

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = {
        "submission_name": f"1_{submission.name}",
        "solution_name": f"2_{solution.name}",
        "description_name": "3_SOLUTION_DESCRIPTION.md",
    }
    shutil.copyfile(submission, out / names["submission_name"])
    shutil.copyfile(solution, out / names["solution_name"])
    shutil.copyfile(description, out / names["description_name"])
    fingerprints = "\n".join(
        f"| `{n}` | {(out / n).stat().st_size:,} bytes | `{sha256(out / n)}` |" for n in names.values()
    )
    (out / "HOW_TO_SUBMIT.md").write_text(GUIDE.format(
        challenge=a.challenge, built=datetime.now().strftime("%Y-%m-%d %H:%M"), spent_time=a.spent_time,
        local_score=a.local_score, baseline=a.baseline, fingerprints=fingerprints, probe_note=probe_note,
        checks="\n".join(f"- {c}" for c in checks), **names,
    ), encoding="utf-8")
    print(f"BUILT {out.resolve()}")
    for p in sorted(out.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
