"""Run experiments one after another from a queue file.

    python run_queue.py queue.txt --public-dir data [--harness solution_dev.py] [--log out/queue.log]
                        [--common "run_folds=2 prune=0.2060"] [--max-minutes 40] [--python EXE]

Queue file: one experiment per line, a tag followed by key=value overrides.
    drop      dropout=0.4
    wd        weight_decay=0.2
    # blank lines and lines starting with # are ignored

The file is read again before every experiment. To change priorities, EDIT THE
PENDING LINES (add, remove, reorder) while a run is in progress; the running
experiment is never interrupted and nothing already computed is thrown away.
A tag that has been started is not run again. A line `STOP` ends the queue after
the current run (use it to free the GPU for the submission checks).

Each experiment runs as
    <python> -u <harness> <public_dir> tag=<tag> <common> <overrides>
Its full output goes to out/runs/<tag>.log; fold lines, the RESULT line and any
error are copied to the queue log as they appear. --max-minutes stops an
experiment that runs over the limit. The PC is kept from idle sleep meanwhile.
"""
import argparse
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from axo_common import keep_awake, kill_tree  # noqa: E402

KEEP = re.compile(r"^fold \d|^RESULT|^PROJECTED|Traceback|Error")


def pending(queue_file, started):
    """First line of the queue whose tag has not been started; 'STOP' if a STOP line comes first."""
    for raw in Path(queue_file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "STOP":
            return "STOP", None
        tag, *rest = shlex.split(line)
        if tag not in started:
            return tag, rest
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue")
    ap.add_argument("--public-dir", required=True)
    ap.add_argument("--harness", default="solution_dev.py")
    ap.add_argument("--log", default="out/queue.log")
    ap.add_argument("--common", default="", help="options given to every experiment")
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    log_path = Path(a.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runs = log_path.parent / "runs"
    runs.mkdir(exist_ok=True)
    keep_awake()

    def say(text):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text, flush=True)

    while True:
        started = set()
        if log_path.exists():
            started = set(re.findall(r"^### (\S+):", log_path.read_text(encoding="utf-8"), flags=re.M))
        tag, overrides = pending(a.queue, started)
        if tag is None or tag == "STOP":
            say("QUEUE STOPPED by a STOP line" if tag == "STOP" else "QUEUE DONE")
            return
        say(f"### {tag}: {' '.join(overrides)}")
        cmd = [a.python, "-u", a.harness, a.public_dir, f"tag={tag}"] + shlex.split(a.common) + overrides
        start = time.time()
        with open(runs / f"{tag}.log", "w", encoding="utf-8") as full:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")

            def pump():
                for line in proc.stdout:
                    full.write(line)
                    full.flush()
                    if KEEP.search(line):
                        say(line.rstrip())

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            try:
                proc.wait(timeout=a.max_minutes * 60 if a.max_minutes else None)
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                say(f"RESULT {tag}: STOPPED after {(time.time() - start) / 60:.1f} min (limit {a.max_minutes:g} min)")
            reader.join(timeout=10)
        say(f"--- {tag} finished in {(time.time() - start) / 60:.1f} min (exit {proc.returncode})")


if __name__ == "__main__":
    main()
