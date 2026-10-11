r"""One command from a finished solution to the ready-to-upload folder.

    python gate.py --challenge "Name exactly as in the bot" --solution solution.py --public-dir data \
        --sample data/sample_submission.csv --description SOLUTION_DESCRIPTION.md \
        --spent-time "2 hours" --local-score 0.34 --baseline 0.30 \
        [--expect-minutes 8] [--max-minutes 90] [--kill-after 30]
        [--json-column COL] [--regex COL PATTERN]... [--forbid REGEX]... [--dev-marker TEXT]...
        [--work out/gate] [--out submission] [--python EXE] [--resume]
        [--probe --chance 0.32]

--probe is for version 1, the fast probe that is uploaded before any strong version
(references/fast-v1.md): the local score must be above --chance instead of above the
baseline. Every other stage is unchanged.

Stages. It stops at the first failure and says which one.
  1 static   solution source checks, and the local score against the baseline (seconds,
             before any long run). --forbid adds patterns that must not appear, e.g.
             "from_pretrained|torch\.hub|load_state_dict" in a from-scratch challenge.
  2 kill     start the solution and kill it after --kill-after seconds: a valid
             fallback submission must already be on disk.
  3 runs     two full runs as separate processes, timed and compared byte for byte.
             --expect-minutes stops a crawling run; --max-minutes is the challenge's limit.
  4 format   run A's submission against the sample (ids, header; JSON / regex per value).
  5 build    the submission/ folder with the upload guide.

Write the description WHILE stage 3 runs. If it is missing or incomplete when stage 5
is reached, the runs are kept: finish the description (the measured run time is printed)
and run the same command again with --resume, which skips stages 2 and 3 as long as the
solution file has not changed.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from axo_common import keep_awake, kill_tree, sha256  # noqa: E402
from build_submission_folder import code_problems, description_problems  # noqa: E402
from validate_submission import validate  # noqa: E402


def stop(stage, problems):
    print(f"\nGATE STOPPED at stage {stage}:")
    for p in problems:
        print(" -", p)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("challenge", "solution", "public-dir", "sample", "description", "spent-time", "local-score", "baseline"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--expect-minutes", type=float)
    ap.add_argument("--max-minutes", type=float, help="the challenge's runtime limit")
    ap.add_argument("--kill-after", type=float, default=30.0, help="seconds before the kill test stops the run")
    ap.add_argument("--json-column")
    ap.add_argument("--regex", nargs=2, action="append", default=[], metavar=("COLUMN", "PATTERN"))
    ap.add_argument("--forbid", action="append", default=[])
    ap.add_argument("--dev-marker", action="append", default=[])
    ap.add_argument("--work", default="out/gate")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="version 1, the fast probe: the local score must beat --chance, not the baseline")
    ap.add_argument("--chance", help="score of a random or constant prediction (required with --probe)")
    ap.add_argument("--below-baseline-ok", action="store_true",
                    help="a later version handed over below the baseline at the user's request: the local score "
                         "must beat --chance, and the guide says plainly that it is below the baseline")
    a = ap.parse_args()
    if (a.probe or a.below_baseline_ok) and a.chance is None:
        ap.error("--probe / --below-baseline-ok need --chance")
    if a.probe and a.below_baseline_ok:
        ap.error("--probe and --below-baseline-ok are exclusive")
    keep_awake()
    t_start = time.time()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    state_file = work / "gate_state.json"
    solution = Path(a.solution)

    # ---- 1 static
    if not solution.is_file():
        stop("1 static", [f"solution not found: {solution}"])
    code = solution.read_text(encoding="utf-8")
    problems, _ = code_problems(code, a.dev_marker)
    for pattern in a.forbid:
        hits = [f"line {i + 1}: {line.strip()[:90]}" for i, line in enumerate(code.splitlines()) if re.search(pattern, line)]
        if hits:
            problems.append(f"solution: forbidden pattern {pattern!r} at " + "; ".join(hits[:3]))
    try:
        # A first scored version below the baseline only has to show the pipeline learns something; later versions must beat the baseline.
        bar, name = (a.chance, "chance level") if a.probe or a.below_baseline_ok else (a.baseline, "baseline")
        if float(a.local_score) <= float(bar):
            problems.append(f"local score {a.local_score} is not above the {name} {bar}")
        float(a.baseline)
    except ValueError:
        problems.append("--local-score, --baseline and --chance must be numbers")
    imports = sorted(set(re.findall(r"^(?:import|from)\s+([A-Za-z_][\w]*)", code, flags=re.M)))
    if problems:
        stop("1 static", problems)
    print(f"1 static   passed. Imports: {', '.join(imports)}", flush=True)

    digest = sha256(solution)
    run_a = work / "run_a" / "submission.csv"
    resumed = False
    if a.resume and state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        resumed = state.get("solution_sha") == digest and state.get("identical") and run_a.is_file()
        if not resumed:
            print("   --resume ignored: the solution changed or the earlier runs did not pass", flush=True)
    if resumed:
        minutes = state["minutes"]
        print(f"2 kill     kept from the earlier pass\n3 runs     kept: IDENTICAL, {minutes[0]:.1f} and {minutes[1]:.1f} min", flush=True)
    else:
        # ---- 2 kill
        kill_out = work / "kill" / "submission.csv"
        kill_out.parent.mkdir(parents=True, exist_ok=True)
        if kill_out.exists():
            kill_out.unlink()
        with open(work / "kill.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen([a.python, "-u", str(solution), a.public_dir, str(kill_out)],
                                    stdout=log, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=a.kill_after)
                how = "the run finished on its own before the kill time"
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                how = f"killed after {a.kill_after:g} s"
        if not kill_out.is_file():
            stop("2 kill", [f"no fallback submission on disk ({how}); the script must write one before training. Log {work / 'kill.log'}"])
        found = validate(str(kill_out), a.sample, a.json_column)
        if found:
            stop("2 kill", [f"fallback submission invalid ({how}): {p}" for p in found])
        print(f"2 kill     passed: valid fallback file present ({how})", flush=True)

        # ---- 3 runs
        cmd = [a.python, str(HERE / "check_determinism.py"), str(solution), a.public_dir, str(work), "--python", a.python]
        if a.expect_minutes:
            cmd += ["--expect-minutes", str(a.expect_minutes)]
        det = subprocess.run(cmd, capture_output=True, text=True)
        (work / "determinism.txt").write_text(det.stdout + det.stderr, encoding="utf-8")
        minutes = [float(m) for m in re.findall(r"run_[ab]: exit 0, ([\d.]+) min", det.stdout)]
        minutes = (minutes + [0.0, 0.0])[:2]
        identical = det.returncode == 0 and "IDENTICAL" in det.stdout
        state_file.write_text(json.dumps(dict(solution_sha=digest, identical=identical, minutes=minutes)), encoding="utf-8")
        if not identical:
            lines = (det.stdout + det.stderr).strip().splitlines()
            first = next((i for i, line in enumerate(lines)
                          if re.search(r"DIFFERENT|STOPPED|failed|Error", line)), max(0, len(lines) - 8))
            stop("3 runs", lines[first:first + 8] + [f"full output: {work / 'determinism.txt'}"])
        print(f"3 runs     passed: IDENTICAL, {minutes[0]:.1f} and {minutes[1]:.1f} min", flush=True)
    if a.max_minutes and max(minutes) >= a.max_minutes:
        stop("3 runs", [f"a run took {max(minutes):.1f} min, over the limit of {a.max_minutes:g} min"])
    print(f"           measured full-run time: {max(minutes):.1f} min  <- the Runtime section of the description", flush=True)

    # ---- 4 format
    problems = [f"submission: {p}" for p in validate(str(run_a), a.sample, a.json_column)]
    if a.regex and not problems:
        with open(run_a, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for column, pattern in a.regex:
            if rows and column not in rows[0]:
                problems.append(f"submission: no column {column!r} for --regex")
                continue
            bad = [r[column] for r in rows if not re.fullmatch(pattern, r[column])]
            if bad:
                problems.append(f"submission: {len(bad)} values of {column!r} do not match {pattern!r}, e.g. {bad[0][:60]!r}")
    if problems:
        stop("4 format", problems)
    print("4 format   passed", flush=True)

    # ---- 5 build
    description = Path(a.description)
    if not description.is_file():
        stop("5 build", [f"description not found: {description}. Runs are kept - write it, then repeat with --resume"])
    problems, _ = description_problems(description.read_text(encoding="utf-8"))
    if problems:
        stop("5 build", problems + ["Runs are kept - fix the description, then repeat with --resume"])
    cmd = [a.python, str(HERE / "build_submission_folder.py"), "--challenge", a.challenge, "--solution", str(solution),
           "--submission", str(run_a), "--sample", a.sample, "--description", str(description),
           "--spent-time", a.spent_time, "--local-score", a.local_score, "--baseline", a.baseline,
           "--determinism-log", str(work / "determinism.txt"), "--out", a.out]
    if a.json_column:
        cmd += ["--json-column", a.json_column]
    if a.probe:
        cmd += ["--probe", "--chance", a.chance]
    if a.below_baseline_ok:
        cmd += ["--below-baseline-ok", "--chance", a.chance]
    for marker in a.dev_marker:
        cmd += ["--dev-marker", marker]
    built =subprocess.run(cmd, capture_output=True, text=True)
    if built.returncode != 0:
        stop("5 build", (built.stdout + built.stderr).strip().splitlines())
    print("5 build    " + built.stdout.strip().replace("\n", "\n           "), flush=True)
    print(f"\nGATE PASSED in {(time.time() - t_start) / 60:.1f} min. Folder: {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
