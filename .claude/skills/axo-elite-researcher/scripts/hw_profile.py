#!/usr/bin/env python3
"""Read-only profile of this machine for ML challenge work. Standard library only.

    python3 hw_profile.py [--python EXE ...] [--torch] [--disk PATH ...] [--network] [--json OUT]

Prints a markdown profile: OS and kernel, CPU, RAM and swap, NVIDIA GPU (model, VRAM, driver, CUDA, what
already uses it), power source, disk space, other heavy processes, whether `systemd-run --user --scope`
can cap memory (the basis of guarded_run.py), and for each --python interpreter its version and the
versions of common ML packages, read from package metadata so nothing heavy is imported.

--torch     also imports torch in each --python interpreter (inside a 2 GiB memory scope when systemd-run
            works) to report its CUDA build and whether it sees the GPU. Uses a few hundred MB of VRAM briefly.
--network   one HTTPS HEAD request each to pypi.org, kaggle.com and arxiv.org. Nothing else leaves the machine.
--json OUT  also writes the profile as JSON. No other file is written; nothing is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "torch", "torchvision", "torchaudio", "lightgbm", "xgboost",
    "catboost", "transformers", "tokenizers", "sentence-transformers", "timm", "datasets", "accelerate", "peft",
    "polars", "pyarrow", "opencv-python", "opencv-python-headless", "pillow", "matplotlib", "threadpoolctl",
    "numba", "onnxruntime", "kaggle",
)
CAP = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=2G", "-p", "MemorySwapMax=0"]


def run(cmd, timeout=30):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, "", f"{type(error).__name__}: {error}"


def read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def os_info():
    release = read("/etc/os-release") or ""
    pretty = re.search(r'^PRETTY_NAME="?([^"\n]+)', release, re.M)
    return {"os": pretty.group(1) if pretty else platform.platform(), "kernel": platform.release(),
            "machine": platform.machine()}


def cpu_info():
    text = read("/proc/cpuinfo") or ""
    model = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    cores = {(p, c) for p, c in re.findall(r"physical id\s*:\s*(\d+)[\s\S]*?core id\s*:\s*(\d+)", text)}
    max_khz = read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    return {"model": model.group(1).strip() if model else platform.processor() or "unknown",
            "logical_cpus": os.cpu_count(), "physical_cores": len(cores) or None,
            "max_ghz": round(int(max_khz) / 1e6, 2) if max_khz and max_khz.isdigit() else None}


def memory_info():
    text = read("/proc/meminfo")
    if not text:
        return {}
    kb = {k: int(v) for k, v in re.findall(r"^(\w+):\s+(\d+) kB", text, re.M)}
    gib = lambda key: round(kb.get(key, 0) / 2 ** 20, 2)
    return {"total_gib": gib("MemTotal"), "available_gib": gib("MemAvailable"),
            "swap_total_gib": gib("SwapTotal"), "swap_free_gib": gib("SwapFree")}


def gpu_info():
    if not shutil.which("nvidia-smi"):
        return {"present": False, "note": "nvidia-smi not found"}
    code, out, err = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,"
                          "utilization.gpu,pstate", "--format=csv,noheader,nounits"])
    if code != 0 or not out:
        return {"present": False, "note": err or "nvidia-smi failed"}
    gpus = []
    for line in out.splitlines():
        name, total, used, driver, util, pstate = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "vram_total_mib": int(total), "vram_used_mib": int(used),
                     "driver": driver, "utilization_pct": util, "pstate": pstate})
    _, header, _ = run(["nvidia-smi"])
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", header)
    _, apps, _ = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                      "--format=csv,noheader,nounits"])
    compute = [dict(zip(("pid", "process", "used_mib"), [x.strip() for x in line.split(",")]))
               for line in apps.splitlines() if line.strip()]
    return {"present": True, "gpus": gpus, "driver_cuda": cuda.group(1) if cuda else None,
            "compute_processes": compute}


def power_info():
    root = Path("/sys/class/power_supply")
    if not root.is_dir():
        return {"state": "unknown", "detail": "no /sys/class/power_supply"}
    mains, batteries = [], []
    for supply in sorted(root.iterdir()):
        kind = read(supply / "type")
        if kind in ("Mains", "USB") and read(supply / "online") is not None:
            mains.append(read(supply / "online") == "1")
        elif kind == "Battery" and read(supply / "scope") != "Device":   # skip mouse and headset batteries
            batteries.append(f"{read(supply / 'status')} {read(supply / 'capacity')}%")
    if any(mains):
        state = "mains"
    elif batteries and all(b.startswith("Discharging") for b in batteries):
        state = "battery"
    elif not batteries and not mains:
        state = "unknown"
    else:
        state = "mains (inferred)" if not any(b.startswith("Discharging") for b in batteries) else "battery"
    return {"state": state, "batteries": batteries}


def disk_info(paths):
    rows = []
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
            rows.append({"path": str(Path(path).resolve()), "free_gib": round(usage.free / 2 ** 30, 1),
                         "total_gib": round(usage.total / 2 ** 30, 1)})
        except OSError as error:
            rows.append({"path": str(path), "error": str(error)})
    return rows


def heavy_processes(min_rss_mib=500):
    code, out, _ = run(["ps", "-eo", "pid,rss,etime,args", "--sort=-rss"])
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[1].isdigit() and int(parts[1]) // 1024 >= min_rss_mib:
            rows.append({"pid": parts[0], "rss_mib": int(parts[1]) // 1024, "elapsed": parts[2],
                         "command": parts[3][:110]})
    return rows[:12]


def memory_cap_works():
    if not shutil.which("systemd-run"):
        return {"available": False, "note": "systemd-run not found: guarded_run.py cannot cap memory"}
    code, _, err = run(CAP + ["true"], timeout=20)
    return {"available": code == 0, "note": "systemd-run --user --scope works" if code == 0 else err[:200]}


PROBE = r"""
import json, sys
from importlib import metadata
found = {}
for name in sys.argv[1].split(","):
    try:
        found[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        found[name] = None
print(json.dumps({"python": sys.version.split()[0], "executable": sys.executable, "packages": found}))
"""
TORCH = r"""
import json, torch
info = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
if info["cuda_available"]:
    p = torch.cuda.get_device_properties(0)
    info.update(device=p.name, vram_gib=round(p.total_memory / 2 ** 30, 2), capability=f"{p.major}.{p.minor}")
print(json.dumps(info))
"""


def python_info(exe, with_torch, cap_ok):
    result = {"interpreter": exe}
    prefix = CAP if cap_ok else []
    code, out, err = run(prefix + [exe, "-I", "-B", "-c", PROBE, ",".join(PACKAGES)], timeout=60)
    if code != 0:
        result["error"] = (err or out)[:300]
        return result
    result.update(json.loads(out.splitlines()[-1]))
    if with_torch and result["packages"].get("torch"):
        code, out, err = run(prefix + [exe, "-I", "-B", "-c", TORCH], timeout=180)
        result["torch_check"] = json.loads(out.splitlines()[-1]) if code == 0 and out else {"error": (err or out)[-300:]}
    return result


def network_info():
    rows = {}
    for url in ("https://pypi.org/simple/", "https://www.kaggle.com/", "https://arxiv.org/"):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10) as response:
                rows[url] = response.status
        except urllib.error.HTTPError as error:  # the server answered: reachable, it only refused HEAD or the path
            rows[url] = error.code
        except Exception as error:  # report the failure, never raise
            rows[url] = f"unreachable ({type(error).__name__})"
    return rows


def markdown(p):
    lines = ["# Machine profile", "", f"- OS: {p['os']['os']} (kernel {p['os']['kernel']}, {p['os']['machine']})"]
    c = p["cpu"]
    lines.append(f"- CPU: {c['model']}: {c['physical_cores']} cores / {c['logical_cpus']} threads, max {c['max_ghz']} GHz")
    m = p["memory"]
    if m:
        lines.append(f"- RAM: {m['total_gib']} GiB total, {m['available_gib']} GiB available now; "
                     f"swap {m['swap_total_gib']} GiB ({m['swap_free_gib']} GiB free)")
    g = p["gpu"]
    if g.get("present"):
        for gpu in g["gpus"]:
            lines.append(f"- GPU: {gpu['name']}, {gpu['vram_total_mib']} MiB VRAM ({gpu['vram_used_mib']} MiB in use, "
                         f"{gpu['utilization_pct']}% busy), driver {gpu['driver']}, driver CUDA {g['driver_cuda']}")
        for app in g["compute_processes"]:
            lines.append(f"  - compute process on the GPU: pid {app['pid']} {app['process']} {app['used_mib']} MiB")
    else:
        lines.append(f"- GPU: none usable ({g.get('note')})")
    pw = p["power"]
    lines.append(f"- Power: {pw['state']}" + (f" ({', '.join(pw['batteries'])})" if pw.get("batteries") else ""))
    for d in p["disks"]:
        lines.append(f"- Disk: {d['path']}: " + (f"{d['free_gib']} GiB free of {d['total_gib']}" if "free_gib" in d else d["error"]))
    lines.append(f"- Memory cap for local runs: {p['memory_cap']['note']}")
    if p["heavy_processes"]:
        lines.append("- Processes over 500 MiB RSS:")
        lines += [f"  - pid {r['pid']} {r['rss_mib']} MiB, up {r['elapsed']}: {r['command']}" for r in p["heavy_processes"]]
    for py in p["pythons"]:
        lines.append(f"- Python {py.get('python', '?')} at {py['interpreter']}")
        if "error" in py:
            lines.append(f"  - error: {py['error']}")
            continue
        have = [f"{k} {v}" for k, v in py["packages"].items() if v]
        lines.append(f"  - present: {', '.join(have) or 'none of the listed packages'}")
        missing = [k for k, v in py["packages"].items() if not v]
        lines.append(f"  - missing: {', '.join(missing)}")
        if "torch_check" in py:
            lines.append(f"  - torch check: {json.dumps(py['torch_check'])}")
    if p.get("network"):
        lines.append("- Network: " + ", ".join(f"{u} -> {s}" for u, s in p["network"].items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", action="append", default=[], help="interpreter to inspect (repeatable)")
    ap.add_argument("--torch", action="store_true", help="import torch in each interpreter to check CUDA")
    ap.add_argument("--disk", action="append", default=[], help="path whose free space to report (repeatable)")
    ap.add_argument("--network", action="store_true", help="three HTTPS HEAD requests")
    ap.add_argument("--json", help="also write the profile as JSON here")
    a = ap.parse_args()
    cap = memory_cap_works()
    profile = {
        "os": os_info(), "cpu": cpu_info(), "memory": memory_info(), "gpu": gpu_info(), "power": power_info(),
        "disks": disk_info(a.disk or ["."]), "heavy_processes": heavy_processes(), "memory_cap": cap,
        "pythons": [python_info(exe, a.torch, cap["available"]) for exe in (a.python or [sys.executable])],
        "network": network_info() if a.network else None,
    }
    print(markdown(profile))
    if a.json:
        Path(a.json).write_text(json.dumps(profile, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
