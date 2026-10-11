"""Shared helpers for the axo scripts. Never imported by a submitted solution."""
import hashlib
import os
import subprocess
import time
from pathlib import Path


def keep_awake(on=True):
    """Ask Windows not to go to sleep from idleness while this process is running.

    It does not stop sleep from closing the lid or pressing the power button, and it
    does nothing about running on battery (see preflight.py). No-op on other systems.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))
        return True
    except Exception:
        return False


def kill_tree(proc):
    """Stop a process and everything it started (a Python run can leave children on Windows)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def run_limited(cmd, log_path, limit_seconds=None, cwd=None, env=None):
    """Run cmd with its output in log_path. Returns (exit_code or None, seconds, timed_out).
    `env` adds or overrides environment variables for this run only."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    full_env = None
    if env:
        import os
        full_env = dict(os.environ)
        full_env.update(env)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=full_env)
        try:
            code = proc.wait(timeout=limit_seconds)
            return code, time.time() - start, False
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            return None, time.time() - start, True


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
