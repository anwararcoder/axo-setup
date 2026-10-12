#!/usr/bin/env python3
"""Run one experiment on this machine under hard resource limits, log it, and record it in a ledger.

    python3 guarded_run.py --tag NAME [options] -- <command> [args ...]

This PC has no swap: a process that outgrows RAM freezes the desktop. So every local run that loads data or trains
goes through this wrapper:

  before    refuses to start when free RAM minus the caps of this run and of every other guarded run still active
            would leave less than --reserve-gib for the desktop; with --gpu, when another guarded GPU run holds the
            GPU lock, another process uses the GPU (unless --allow-shared-gpu) or less than --min-free-vram is free;
            without --gpu, when --max-cpu-jobs guarded CPU runs are already active
  during    the command runs in a systemd user scope with MemoryMax (the whole process tree is killed at the cap,
            exit code 137, instead of freezing the machine), MemorySwapMax=0 and a CPU quota; output goes straight
            to a log file (no pipe that buffers); CPU-only runs get CUDA_VISIBLE_DEVICES="" so they cannot touch the
            GPU by accident; --timeout-min stops the whole scope
  after     one JSON line in the ledger: command, sha256 of every .py file named on the command line, start and end,
            wall time, exit status, the cap, peak memory of the scope, peak GPU memory in use, the log path, and
            the metrics the command printed as lines `METRIC {"name": value, ...}` (and the last `RESULT ...` line)

Options: --mem 2G  --cpus 4  --gpu  --allow-shared-gpu  --min-free-vram 3000  --timeout-min M  --max-cpu-jobs 3
         --reserve-gib 1.5  --log-dir out/runs  --ledger experiments.jsonl  --note TEXT  --dry-run
         --no-cap (only for a machine without systemd-run; never on this PC)
Exit code: the command's, 124 after a timeout, 75 when a check refused the start. Standard library only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOCKS = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "axo-elite" / "locks"
REFUSED, TIMEOUT = 75, 124


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_size(text):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?)i?B?", text.strip(), re.I)
    if not match:
        raise argparse.ArgumentTypeError(f"bad size {text!r}; use e.g. 800M or 2G")
    return int(float(match.group(1)) * 1024 ** "BKMGT".index((match.group(2) or "B").upper()))


def mem_available():
    text = Path("/proc/meminfo").read_text()
    return int(re.search(r"^MemAvailable:\s+(\d+) kB", text, re.M).group(1)) * 1024


def gpu_state():
    """(used MiB, free MiB, [other compute processes]) or None without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split("\n")[0]
        used, free = (int(x) for x in out.split(","))
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return used, free, [line.strip() for line in apps.splitlines() if line.strip()]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def active_holders():
    """Info dicts of the guarded runs that hold a lock right now (stale lock files are ignored)."""
    holders = []
    for path in sorted(LOCKS.glob("*.lock")):
        try:
            with open(path, "r+") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(handle, fcntl.LOCK_UN)       # nobody holds it: stale
                except BlockingIOError:
                    holders.append(json.loads(handle.read() or "{}"))
        except (OSError, ValueError):
            continue
    return holders


def take_lock(names, info):
    """Exclusive lock on the first free name; returns the open handle or None. Released when the process ends."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    for name in names:
        handle = open(LOCKS / name, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(dict(info, lock=name)))
        handle.flush()
        return handle
    return None


def refuse(reason):
    print(f"REFUSED: {reason}", flush=True)
    sys.exit(REFUSED)


def scope_peak(pid, unit):
    """memory.peak of the run's own scope (cgroup v2), in bytes, or None. Until systemd has moved the process into
    its scope, /proc/<pid>/cgroup still names the caller's cgroup, whose peak must not be reported."""
    try:
        line = next(l for l in Path(f"/proc/{pid}/cgroup").read_text().splitlines() if l.startswith("0::"))
        if not line.rstrip().endswith(f"/{unit}.scope"):
            return None
        return int((Path("/sys/fs/cgroup") / line[3:].strip().lstrip("/") / "memory.peak").read_text().split()[0])
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def on_term(signum, frame):
    raise KeyboardInterrupt


def sha_of_scripts(command):
    found = {}
    for arg in command:
        path = Path(arg)
        if path.suffix == ".py" and path.is_file():
            found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def read_metrics(log):
    metrics, result = {}, None
    try:
        lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return metrics, result
    for line in lines:
        if line.startswith("METRIC "):
            try:
                metrics.update(json.loads(line[7:]))
            except ValueError:
                pass
        elif line.startswith("RESULT"):
            result = line.strip()[:400]
    return metrics, result


def main():
    if "--" not in sys.argv:
        if "-h" in sys.argv or "--help" in sys.argv:
            print(__doc__)
            return
        sys.exit("usage: guarded_run.py --tag NAME [options] -- <command> [args ...]   (see --help)")
    split = sys.argv.index("--")
    command = sys.argv[split + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mem", type=parse_size, default=parse_size("2G"))
    ap.add_argument("--cpus", type=float, default=4.0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--min-free-vram", type=int, default=3000, help="MiB")
    ap.add_argument("--timeout-min", type=float)
    ap.add_argument("--max-cpu-jobs", type=int, default=3)
    ap.add_argument("--reserve-gib", type=float, default=1.5)
    ap.add_argument("--log-dir", default="out/runs")
    ap.add_argument("--ledger", default="experiments.jsonl")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-cap", action="store_true")
    a = ap.parse_args(sys.argv[1:split])
    if not command:
        refuse("no command after --")
    if not re.fullmatch(r"[\w.-]{1,60}", a.tag):
        refuse("--tag may hold letters, digits, '.', '_' and '-' only (at most 60)")

    # ---- checks before anything starts
    capped = not a.no_cap
    if capped and not shutil.which("systemd-run"):
        refuse("systemd-run not found, so memory cannot be capped. On this PC that is a hard stop; elsewhere pass "
               "--no-cap knowingly")
    others = active_holders()
    committed = sum(h.get("mem_cap", 0) for h in others)
    available = mem_available()
    left = available - a.mem - committed
    print(f"checks     RAM available {available / 2**30:.2f} GiB; this cap {a.mem / 2**30:.2f} GiB; caps of "
          f"{len(others)} active guarded run(s) {committed / 2**30:.2f} GiB; left for the desktop {left / 2**30:.2f} GiB "
          f"(reserve {a.reserve_gib} GiB)", flush=True)
    if left < a.reserve_gib * 2**30:
        refuse("not enough free RAM for this cap. Lower --mem, wait for a run to end, or close something heavy "
               "(the profile lists processes over 500 MiB)")
    info = dict(pid=os.getpid(), tag=a.tag, start=now_utc(), mem_cap=a.mem, gpu=a.gpu, cwd=os.getcwd())
    gpu_before = None
    if a.gpu:
        state = gpu_state()
        if state is None:
            refuse("--gpu given but nvidia-smi does not answer")
        gpu_before, free, apps = state
        print(f"checks     GPU {gpu_before} MiB in use, {free} MiB free; compute processes: {apps or 'none'}", flush=True)
        if apps and not a.allow_shared_gpu:
            refuse("another process is computing on the GPU; two jobs on 4 GB crash or crawl. Wait, or pass "
                   "--allow-shared-gpu after checking that both fit")
        if free < a.min_free_vram:
            refuse(f"only {free} MiB of VRAM free (minimum {a.min_free_vram})")
        lock = take_lock(["gpu.lock"], info)
        if lock is None:
            refuse("another guarded GPU run holds the GPU lock: " + json.dumps([h for h in others if h.get("gpu")]))
    else:
        lock = take_lock([f"cpu.{i}.lock" for i in range(a.max_cpu_jobs)], info)
        if lock is None:
            refuse(f"{a.max_cpu_jobs} guarded CPU runs are active already (--max-cpu-jobs)")

    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_dir / f"{a.tag}_{stamp}.log"
    unit = f"axoe-{a.tag}-{stamp}".replace("_", "-").replace(".", "-")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    threads = str(max(1, int(a.cpus)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, threads)                  # a launcher default; the command may set its own
    if not a.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""               # a CPU run cannot take GPU memory by accident
    full = command
    if capped:
        full = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}", "-p", f"MemoryMax={a.mem}",
                "-p", "MemorySwapMax=0", "-p", f"CPUQuota={int(a.cpus * 100)}%", "--"] + command
    print(f"start      {a.tag}: {shlex.join(command)}\n           cap {a.mem / 2**30:.2f} GiB, {a.cpus:g} CPUs, "
          f"{'GPU' if a.gpu else 'no GPU'}, timeout {a.timeout_min or 'none'} min, log {log}", flush=True)
    if a.dry_run:
        print("dry run    nothing started")
        return

    # ---- run
    record = dict(tag=a.tag, note=a.note, command=command, cwd=os.getcwd(), scripts_sha256=sha_of_scripts(command),
                  start_utc=now_utc(), mem_cap_bytes=a.mem, cpus=a.cpus, gpu=a.gpu, unit=unit if capped else None,
                  log=str(log))
    status, code, peak, gpu_peak = "ok", None, None, None
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as out:
        out.write(f"# {now_utc()} guarded_run {a.tag}: {shlex.join(command)}\n")
        out.flush()
        proc = subprocess.Popen(full, stdout=out, stderr=subprocess.STDOUT, env=env)
        signal.signal(signal.SIGTERM, on_term)
        last_gpu = 0.0
        try:
            while proc.poll() is None:
                time.sleep(0.5 if time.time() - t0 < 10 else 2)
                peak = (scope_peak(proc.pid, unit) if capped else None) or peak
                if a.gpu and time.time() - last_gpu >= 5:
                    state, last_gpu = gpu_state(), time.time()
                    if state:
                        gpu_peak = max(gpu_peak or 0, state[0])
                if a.timeout_min and time.time() - t0 > a.timeout_min * 60:
                    status = "timeout"
                    break
        except KeyboardInterrupt:
            status = "interrupted"
        if proc.poll() is None:                        # stop the whole process tree, not only the first process
            if capped:
                subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", f"{unit}.scope"], capture_output=True)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if capped:
                    subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", f"{unit}.scope"],
                                   capture_output=True)
                proc.kill()
                proc.wait()
        code = proc.returncode
    if status == "ok" and code != 0:
        status = "killed at the memory cap" if code in (137, -9) and capped else "failed"
    wall = time.time() - t0
    max_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    metrics, result = read_metrics(log)
    record.update(end_utc=now_utc(), wall_s=round(wall, 1), status=status, exit_code=code,
                  peak_scope_mem_mib=round(peak / 2**20) if peak else None, max_rss_mib=round(max_rss),
                  gpu_used_before_mib=gpu_before, gpu_used_peak_mib=gpu_peak, metrics=metrics, result_line=result)
    ledger = Path(a.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"end        {a.tag}: {status} (exit {code}) after {wall / 60:.1f} min; peak memory "
          f"{record['peak_scope_mem_mib'] or '?'} MiB of {a.mem / 2**20:.0f}; max RSS {record['max_rss_mib']} MiB"
          + (f"; GPU in use up to {gpu_peak} MiB" if a.gpu else "") + (f"; metrics {json.dumps(metrics)}" if metrics else "")
          + f"\n           ledger {ledger}", flush=True)
    lock.close()
    sys.exit(TIMEOUT if status == "timeout" else (code if code >= 0 else 128 - code))   # killed by signal N -> 128+N


if __name__ == "__main__":
    main()
