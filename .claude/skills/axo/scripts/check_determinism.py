"""Run a solution script twice as separate processes and compare the submissions.

    python check_determinism.py <solution.py> <public_dir> [work_dir] [--python EXE]
                                [--expect-minutes M] [--limit-factor 2]

Each run is started exactly as the platform starts it:
    <python> <solution.py> <public_dir> <work_dir>/run_x/submission.csv
Prints IDENTICAL (exit 0) or DIFFERENT with the first differing lines (exit 1).
Wall time of each run is printed, so this doubles as the timed full run.

--expect-minutes M   the run time you expect. A run that passes M x limit-factor is
                     stopped and reported as SLOW (exit 2), instead of crawling for
                     an hour on a throttled machine before anyone notices.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from axo_common import keep_awake, run_limited  # noqa: E402


def run_twice(solution, public_dir, work, python=sys.executable, expect_minutes=None, limit_factor=2.0):
    """Returns dict(status, minutes=[a, b], outputs=[path_a, path_b], differing=[...])."""
    work = Path(work)
    limit = expect_minutes * limit_factor * 60 if expect_minutes else None
    result = dict(status="IDENTICAL", minutes=[], outputs=[], differing=[])
    # The two runs get DIFFERENT Python hash seeds on purpose: a result that depends on the iteration order of a
    # set (or of a dict built from one) then shows up as DIFFERENT here instead of on the platform
    # (Conversational Functional Span Decomposition, 2026-10-08).
    hash_seed = {"run_a": "1", "run_b": "20261008"}
    for name in ("run_a", "run_b"):
        out = work / name / "submission.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()  # a stale file must not be mistaken for this run's output
        log = work / f"{name}.log"
        code, seconds, timed_out = run_limited([python, "-u", solution, public_dir, out], log, limit,
                                               env={"PYTHONHASHSEED": hash_seed[name]})
        result["minutes"].append(seconds / 60)
        if timed_out:
            print(f"{name}: STOPPED after {seconds / 60:.1f} min - over {limit_factor:g} x the expected "
                  f"{expect_minutes:g} min. Check power, other GPU jobs, sleep. Log {log}")
            result["status"] = "SLOW"
            return result
        print(f"{name}: exit {code}, {seconds / 60:.1f} min, log {log}")
        if code != 0 or not out.exists():
            print(f"{name} failed; see its log")
            result["status"] = "FAILED"
            return result
        result["outputs"].append(out)

    a, b = (o.read_bytes() for o in result["outputs"])
    if a != b:
        result["status"] = "DIFFERENT"
        la, lb = a.decode("utf-8", "replace").splitlines(), b.decode("utf-8", "replace").splitlines()
        result["differing"] = [i for i in range(min(len(la), len(lb))) if la[i] != lb[i]]
        print("DIFFERENT")
        print(f"  {len(result['differing'])} differing lines (of {len(la)} / {len(lb)})")
        for i in result["differing"][:3]:
            print(f"  line {i + 1}:\n    a: {la[i][:160]}\n    b: {lb[i][:160]}")
    else:
        print("IDENTICAL")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solution")
    ap.add_argument("public_dir")
    ap.add_argument("work_dir", nargs="?", default="out/determinism")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--expect-minutes", type=float)
    ap.add_argument("--limit-factor", type=float, default=2.0)
    a = ap.parse_args()
    keep_awake()
    start = time.time()
    result = run_twice(a.solution, a.public_dir, a.work_dir, a.python, a.expect_minutes, a.limit_factor)
    if result["status"] != "IDENTICAL":
        sys.exit({"DIFFERENT": 1, "SLOW": 2}.get(result["status"], 1))
    print(f"both runs: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
