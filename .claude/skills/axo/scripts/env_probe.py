"""Print what the runtime offers, so that references/target-env.md is checked instead of trusted.

Run it in a Kaggle notebook cell (`!python env_probe.py`, or paste the file into a cell) or locally, and paste the whole
output back. It changes nothing and downloads nothing; the one network line only tries to open a connection.
Run it in a notebook, never inside a submitted solution.
"""
import importlib.util
import os
import platform
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path

# import name -> distribution name
LIBS = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn", "torch": "torch",
    "torchvision": "torchvision", "tensorflow": "tensorflow", "lightgbm": "lightgbm", "xgboost": "xgboost",
    "catboost": "catboost", "cv2": "opencv-python", "PIL": "pillow", "matplotlib": "matplotlib",
    "statsmodels": "statsmodels", "nltk": "nltk", "transformers": "transformers", "tokenizers": "tokenizers",
    "safetensors": "safetensors", "accelerate": "accelerate", "datasets": "datasets",
    "huggingface_hub": "huggingface-hub", "evaluate": "evaluate", "peft": "peft", "trl": "trl", "optimum": "optimum",
    "diffusers": "diffusers", "sentence_transformers": "sentence-transformers", "timm": "timm",
    "onnxruntime": "onnxruntime", "sacrebleu": "sacrebleu", "rouge_score": "rouge-score", "ultralytics": "ultralytics",
    "librosa": "librosa", "polars": "polars", "pyarrow": "pyarrow", "numba": "numba",
}
WEIGHTS = ("model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack")


def line(label, value):
    print(f"{label:14s} {value}")


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or "none"
    except Exception as error:
        return f"none ({type(error).__name__})"


line("python", f"{sys.version.split()[0]} on {platform.platform()}")
line("cpus", os.cpu_count())
try:
    meminfo = dict(row.split(":") for row in Path("/proc/meminfo").read_text().splitlines())
    line("memory", f"{int(meminfo['MemTotal'].split()[0]) // 1024} MB total, {int(meminfo['MemAvailable'].split()[0]) // 1024} MB available")
except Exception:
    line("memory", "unknown (no /proc/meminfo)")
line("gpu", run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]))

print("\nlibraries (version, or MISSING):")
for name, dist in LIBS.items():
    if importlib.util.find_spec(name) is None:
        status = "MISSING"
    else:
        try:
            status = metadata.version(dist)
        except metadata.PackageNotFoundError:
            status = "present (version unknown)"
    print(f"  {name:22s} {status}")

print("\ncheckpoint folders (a config.json next to weights) under the usual places:")
roots = [Path("/kaggle/input"), Path.home() / ".cache" / "huggingface", Path.home() / ".cache" / "torch"]
roots += [Path(os.environ[v]) for v in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME") if os.environ.get(v)]
found = []
for root in roots:
    for folder, subdirs, files in os.walk(root):
        if folder[len(str(root)):].count(os.sep) >= 7:
            subdirs[:] = []
        if "config.json" in files and any(w in files for w in WEIGHTS):
            found.append(folder)
for folder in found[:20]:
    print("  ", folder)
print("  none found" if not found else f"  {len(found)} found" + (" (first 20 shown)" if len(found) > 20 else ""))

print("\n/kaggle/input, two levels:")
root = Path("/kaggle/input")
if root.is_dir():
    for first in sorted(root.iterdir())[:20]:
        print("  ", first, "->", [p.name for p in sorted(first.iterdir())[:6]] if first.is_dir() else "")
else:
    print("   not present")

try:
    socket.create_connection(("huggingface.co", 443), timeout=3).close()
    line("\nnetwork", "huggingface.co:443 reachable")
except Exception as error:
    line("\nnetwork", f"huggingface.co:443 not reachable ({type(error).__name__})")
