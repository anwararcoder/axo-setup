"""
<Challenge name>

Task
  <one sentence: what goes in, what comes out, how it is scored>

Approach
  <what the solution does and why it fits this task, in a few lines>

Deterministic runtime plan (keep this section; the platform checks for it)
  * One fixed device (<cuda:0 or cpu>). No device selection, no fallback, no multi-GPU path, no data-loader
    workers, no mixed precision, no retry logic.
  * Fixed seeds for Python, NumPy and the framework, set again before every model is created; fixed numbers of
    passes and fixed schedules; no early stopping; the clock is read only to print progress.
  * Deterministic backends: use_deterministic_algorithms(True), cuDNN deterministic, benchmarking off, TF32 off,
    fixed cuBLAS workspace.
  * No result depends on the iteration order of a set or on Python's hash seed: everything is sorted.

Rules followed
  <the notable ones: weights/data sources, runtime limit; for a From Scratch challenge say: every learned
  parameter is created from RANDOM INITIALISATION in this file and trained here>
"""
import os
import random
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # assigned (never setdefault), before torch is imported

import numpy as np
import pandas as pd

# One fixed plan (the challenge-engine pack): the amount of work, the device and the paths never depend on the
# clock, a score, the GPU or CPU count, CUDA availability or files found on disk.
CONFIG = dict(
    seeds=[0],
    epochs=10,
)
DEVICE = "cuda:0"  # FILL IN, one fixed device per challenge: "cuda:0" for a GPU challenge, "cpu" for a CPU challenge


def make_deterministic(seed):
    """Fixed seeds and deterministic kernels. Call before each model is built.
    Seeds alone are not enough on a GPU, hence use_deterministic_algorithms; if it
    raises for an operation, replace the operation rather than silencing the error.
    Delete the torch lines if torch is not used."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def progress_logger():
    """Returns on_epoch(fraction_done): prints elapsed time and the estimated time left after the first pass and
    about every 10 percent, so a crawling run is visible in the first minute."""
    start, last = time.time(), [0.0]

    def on_epoch(fraction):
        elapsed = time.time() - start
        if last[0] == 0.0 or fraction - last[0] >= 0.1 or fraction >= 1.0:
            last[0] = fraction
            left = elapsed / fraction * (1.0 - fraction)
            print(f"{fraction:6.1%} done, {elapsed / 60:.1f} min elapsed, about {left / 60:.1f} min left", flush=True)
    return on_epoch


# ----------------------------------------------------------------------------
# Data. Both loaders return a dict whose values are all aligned on the rows
# (same length), so the development harness can hold rows out by index.
# ----------------------------------------------------------------------------
def load_train(public_dir):
    train = pd.read_csv(public_dir / "train.csv")
    return dict(table=train)                     # FILL IN: inputs and targets, row-aligned


def load_test(public_dir):
    test = pd.read_csv(public_dir / "test.csv")
    return dict(table=test)                      # FILL IN: the same inputs as load_train, without targets


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def fit(train, cfg, seed, on_epoch=None):
    """Train one model on the given rows. `on_epoch(fraction_done)` is called after
    every pass so a caller can print progress and the estimated time left."""
    make_deterministic(seed)
    model = None                                 # FILL IN: build on DEVICE and train; exactly cfg["epochs"] passes
    for epoch in range(cfg["epochs"]):
        if on_epoch:
            on_epoch((epoch + 1) / cfg["epochs"])
    return model


def predict(models, data, cfg):
    """Predictions for the rows of `data`, averaged over the list of models."""
    return None                                  # FILL IN: array / list aligned with the rows of `data`


def to_submission(test, predictions, sample):
    """The submission table in exactly the sample's columns and id order."""
    submission = sample.copy()                   # FILL IN: put the predictions into the prediction column(s)
    return submission


def main():
    # Required input/output pattern. Keep the names public_dir / submission_out everywhere in the file.
    # Arguments are tolerated, never required: the file must also run when pasted into one notebook cell.
    public_dir = (
        Path(sys.argv[1])
        if len(sys.argv) >= 3 and not sys.argv[1].startswith("-f")
        else Path("/kaggle/working/extracted_dataset")
    )
    submission_out = (
        Path(sys.argv[2])
        if len(sys.argv) >= 3 and not sys.argv[1].startswith("-f")
        else Path("/kaggle/working/submission.csv")
    )
    t_start = time.time()
    minutes = lambda: (time.time() - t_start) / 60
    cfg = CONFIG

    sample = pd.read_csv(public_dir / "sample_submission.csv")

    # Safety net: a valid submission exists from the first seconds, so a run that is cut off at the time limit
    # still leaves a scorable file. It never replaces the prediction method. Use the challenge's neutral
    # prediction here if the sample is not neutral.
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(submission_out, index=False, encoding="utf-8")

    train = load_train(public_dir)
    test = load_test(public_dir)
    print(f"data loaded at {minutes():.1f} min")

    models = []
    for seed in cfg["seeds"]:
        models.append(fit(train, cfg, seed, on_epoch=progress_logger()))
        print(f"seed {seed} trained at {minutes():.1f} min")

    # Final stage: inference, format checks, CSV.
    submission = to_submission(test, predict(models, test, cfg), sample)
    assert list(submission.columns) == list(sample.columns)
    assert submission.iloc[:, 0].tolist() == sample.iloc[:, 0].tolist()
    submission.to_csv(submission_out, index=False, encoding="utf-8")
    print(f"total {minutes():.1f} min -> {submission_out}")


if __name__ == "__main__":
    main()
