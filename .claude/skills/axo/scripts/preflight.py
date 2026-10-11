"""Is this machine ready for a training run? Run before the first run and before every long one.

    python preflight.py [--need torch,pandas,PIL] [--dir .] [--min-free-gb 2]

Checks, each printed as OK / WARN / FAIL:
  power    on mains power (a laptop on battery throttles the GPU; one run crawled for 71 minutes this way)
  gpu      present, and whether something else is already using it
  jobs     other Python processes running on this machine
  disk     free space in the working directory
  libs     the libraries named with --need import in this Python (and CUDA is visible to torch)
Last line: READY, READY WITH WARNINGS, or NOT READY with the reasons. Exit code 1 only for NOT READY.
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys


def power():
    if os.name != "nt":
        return "WARN", "power source not checked on this system"
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = Status()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        battery = "no battery" if s.BatteryFlag == 128 else f"battery {s.BatteryLifePercent}%"
        if s.ACLineStatus == 1:
            return "OK", f"on mains power ({battery})"
        if s.ACLineStatus == 0:
            return "FAIL", f"running on battery ({battery}) - plug in before training"
        return "WARN", f"power source unknown ({battery})"
    except Exception as e:
        return "WARN", f"power source not readable ({e})"


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        name, util, used, total = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        return "WARN", "no NVIDIA GPU visible (nvidia-smi failed) - CPU only"
    busy = float(util) > 20 or float(used) > 600
    text = f"{name}: {util}% used, {used} of {total} MiB"
    if busy:
        return "WARN", text + " - ALREADY IN USE by another job; runs will be slower, timings unreliable"
    return "OK", text + " - free"


def other_python_jobs():
    me = os.getpid()
    rows = []
    try:
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
                   "ForEach-Object { \"$($_.ProcessId)`t$($_.ParentProcessId)`t$($_.CommandLine)\" }"]
        else:
            cmd = ["ps", "-eo", "pid,ppid,args"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            parts = line.split("\t") if os.name == "nt" else line.split(None, 2)
            if len(parts) < 3 or not parts[0].strip().isdigit():
                continue
            pid, ppid, args = int(parts[0]), parts[1].strip(), parts[2]
            if pid == me or str(me) == ppid or "python" not in args.lower() or "preflight" in args:
                continue
            rows.append(f"pid {pid}: {args.strip()[:110]}")
    except Exception as e:
        return "WARN", f"could not list processes ({e})", []
    if rows:
        return "WARN", f"{len(rows)} other Python job(s) running", rows
    return "OK", "no other Python jobs", []


def disk(path, min_free_gb):
    free = shutil.disk_usage(path).free / 1e9
    return ("OK" if free >= min_free_gb else "FAIL"), f"{free:.1f} GB free in {os.path.abspath(path)}"


def libs(names):
    out = []
    for name in names:
        try:
            m = importlib.import_module(name)
            text = f"{name} {getattr(m, '__version__', '')}".strip()
            if name == "torch":
                text += " (CUDA available)" if m.cuda.is_available() else " (CUDA NOT available)"
                out.append(("OK" if m.cuda.is_available() else "WARN", text))
            else:
                out.append(("OK", text))
        except Exception:
            out.append(("FAIL", f"{name} does not import in {sys.executable}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", default="", help="comma-separated libraries the solution imports")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    a = ap.parse_args()

    results = [("power",) + power(), ("gpu",) + gpu()]
    state, text, rows = other_python_jobs()
    results.append(("jobs", state, text))
    results.append(("disk",) + disk(a.dir, a.min_free_gb))
    for state_l, text_l in libs([n for n in a.need.split(",") if n]):
        results.append(("libs", state_l, text_l))

    for name, st, tx in results:
        print(f"{st:4s} {name:5s} {tx}")
        if name == "jobs":
            for r in rows:
                print(f"           {r}")
    fails = [tx for _, st, tx in results if st == "FAIL"]
    warns = [tx for _, st, tx in results if st == "WARN"]
    if fails:
        print("NOT READY: " + "; ".join(fails))
        sys.exit(1)
    print("READY WITH WARNINGS: " + "; ".join(warns) if warns else "READY")


if __name__ == "__main__":
    main()
