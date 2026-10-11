# Deterministic Finalization

Goal: pass deterministic-runtime checks while preserving the strongest solution as much as possible.

## Freeze first

Before editing the deterministic copy, freeze:
- model family
- preprocessing
- feature construction
- split logic
- hyperparameters
- checkpoint selection
- inference logic
- submission column/order logic

## Remove runtime branching

Bad final patterns:
- choose GPU count dynamically
- CPU fallback if CUDA is missing
- try one scaler/backend then another
- retry with fewer workers after failure
- change batch size after OOM
- load whichever checkpoint happens to exist
- use caches opportunistically
- use `os.cpu_count()` to determine the plan
- alter behavior based on discovered files in `/kaggle/input`

## Make the plan explicit

Prefer:
- one device strategy
- one model name/path
- fixed workers
- fixed order
- fixed seeds
- fixed backend settings
- no recovery path that changes execution semantics

## Important

Determinism can change speed and sometimes score. The deterministic copy is an evaluation-safe artifact, not an opportunity to redesign the model.
