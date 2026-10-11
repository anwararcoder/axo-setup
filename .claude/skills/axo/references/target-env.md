# The target runtime: what a solution can import, and what it cannot get

Axo runs the submitted `solution.py` itself, in "the configured Kaggle Docker runtime" (`platform-rules.md`). What that
runtime holds decides which recipes are open from the start. Four levels of certainty, kept apart on purpose.

## 1. Confirmed on Axo's runtime (a scored run or a message from the platform)

| Fact | Evidence |
|---|---|
| numpy, pandas and scikit-learn (`TfidfVectorizer` and the rest) import and run; scipy comes with scikit-learn | Why This Name v1, 2026-10-04: scored 36.30 with a script built on them |
| threadpoolctl is available to enforce the fixed scikit-learn CPU thread scope | Recipe Ingredients v4 (2026-10-08), S74 scored 0.5772, using the required scikit-learn dependency; the local preloaded-library audit verified active limits |
| A crash is reported back with the exception type | Instruction Stack v1: "Your solution failed with ValueError" |
| torch imports and trains on the CPU: GRUs, embeddings, AdamW, `use_deterministic_algorithms(True)`, four threads, about 40 minutes locally | The Enclosure Mention v5, 2026-10-06: scored 0.835 |
| An `assert` that fires in the script is reported as "Your solution failed with AssertionError", and the status block still showed 6/6 credits afterwards | Building Use v1, 2026-10-06 |
| Compute is per challenge: A10G (Instruction Stack), CPU (Why This Name) | the `Compute` field of each challenge header |
| The challenge texts forbid internet at run time ("no network access during the run"; "do not use ... internet access during solution execution") | both descriptions |

Not confirmed, although it looks likely: no pretrained weights are present. Instruction Stack v1 raised ValueError from a
bounded search for a required folder; the folder searched for was the model checkpoint (or, if the platform passed no
arguments, the dataset folder). Which search failed is unknown.

## 2. Stated by the user, not verified (2026-10-04)

The image carries the Hugging Face stack: `transformers`, `datasets`, `huggingface_hub`, `tokenizers`, `accelerate`,
`diffusers`, `safetensors`, `evaluate`, `peft`, `trl`, `optimum`. Versions unknown; the source is a general description of
the public `kaggle/python` image, not a look at Axo's runtime.

## 3. Normally in the public kaggle/python image (general knowledge, unverified here)

torch and torchvision (CUDA on the GPU challenges), tensorflow and keras, lightgbm, xgboost, catboost, opencv, pillow,
matplotlib, statsmodels, nltk (the code, not downloaded data). Not assumed: `ultralytics`, `sentence-transformers`, and
anything not named in this file.

## 4. Settle levels 2 and 3

The user does not run notebooks, so levels 2 and 3 are settled only by what Axo's own scored results and error messages show (a crash message names the exception; a scored run proves the imports it used). Record each such fact with its date and keep everything else labelled unverified. `scripts/env_probe.py` is meant for a Kaggle notebook and is used only if the user offers to run it.

## What this opens, and what it does not

Libraries are not weights, and nothing can be downloaded during the run.

- `from_pretrained("name")`, `load_dataset(...)`, `evaluate.load(...)` and `hf_hub_download(...)` all go to the network
  and fail. Never build the solution on them. Write metrics by hand, and check them against a one-pair reference
  (Why This Name: exact chrF2 in about 30 lines).
- Pretrained weights count as available only if the challenge text says they are provided or the user has confirmed the
  path on Axo's runtime. A challenge that says "pretrained weights available in the environment may be used" does not
  say which; ask the user or the owner which, before building on them.
- Open without weights: a transformer trained from random initialisation (`AutoModel.from_config`), a tokenizer trained
  on the challenge text (`tokenizers`), mixed precision (torch autocast or `accelerate`), saving and hashing a
  checkpoint (`safetensors`; some challenges ask for the checkpoint or its hash). With a few thousand examples, compare a
  from-scratch model with retrieval or a counted baseline, and test its complementary contribution when combination is legal and justified; a weaker standalone score alone does not decide that question (`recipes.md` 3 and 6; `climb.md`).
- Closed without weights: `peft`, `trl`, `diffusers`, `optimum`, and the fine-tuning of any named model.
- Open and cheap on CPU: scikit-learn, scipy sparse algebra, the gradient-boosting libraries (level 3).

## Rules for the script

1. The plan is fixed when the script is written (the challenge-engine pack, `references/pack/`; it replaces the earlier
   "search a few places and fall back" rule). Use only what the challenge provides or the user has confirmed, at its
   fixed path; leave an optional resource out of the solution altogether. No discovery of files under `/kaggle/input`,
   no fallback to another method, no change of model because an optional import is missing, no device chosen from CUDA
   availability. A missing required resource fails clearly instead of changing the plan. (Instruction Stack v1: a search
   for an unconfirmed checkpoint folder ended in a ValueError on the platform.)
2. Platform versions are unknown and the development machine's are newer (locally scikit-learn 1.9, numpy 2.4,
   pandas 2.3). Write for pandas 2.x, numpy 1.26 or 2.x and scikit-learn 1.x, avoid APIs newer than a year, and print
   the versions of the libraries used at the top of the log.
3. Import only what this file lists as present (levels 1 to 3); anything else needs the probe or the user's word first.
