# Select: inventory, model policy, gates and ranking

## Contents
1. Inventory fields and where they come from
2. Pretrained-model policy: classes A, B, C
3. Gates
4. Ranking criteria
5. Reading the status signals
6. Hardware fit on this PC
7. What select returns

## 1. Inventory fields and where they come from

| Field | Source | Notes |
|---|---|---|
| Objective, metric and direction, submission format, validation errors | `description.md` | Quote. An ambiguous metric sentence gets every reading implemented later (Anchored Layout Alignment lost its first hour to one). |
| Dataset size and format | `description.md` (counts the text names) | The archive size needs a fresh link (`curl -sIL`); only with the user's agreement, never before selection. |
| Baseline | `baseline.md`, status block | "Not announced" when absent. |
| State, seats (solvers above the baseline out of 10), top, 3rd, 10th, submissions, credits, countdown | `public-leaderboard.md` (age = file modification time) | The live bot wins. "Submitted", the marked row and the credits belong to the shared account that other group members use too. |
| Closing time | countdown plus the file time (an estimate) | Unknown when absent: ask. |
| Pretrained models, external data, internet, ensembles | quoted sentences of `description.md` | Section 2. |
| Compute and runtime limit | card `Compute` and the runtime sentences | Platform default about 1 hour; the stricter limit wins; record both. |
| Active work | `--also` folders (`WORKLOG.md`, `NOTES.md`, solutions, submission folders) | Another session may own the challenge: ask before overlapping. |

## 2. Pretrained-model policy: classes A, B, C

- **A, permitted.** The text allows pretrained weights explicitly, for example a sentence that allows
  general-purpose pretrained weights and names model families. Still check which
  weights are excluded (trained on the challenge's domain or data), how the grader's runtime obtains them (the
  Axo runtime has no network unless the text says otherwise: weights must be provided, attached or explicitly
  downloadable by the script), licence and provenance.
- **B, prohibited.** "No pretrained weights", "train from scratch", "random initialization". It covers
  pretrained language models, embeddings, word vectors, tokenizers trained elsewhere, and dictionaries or
  analyzers when the text lists them. Hand-written rules only where the text allows them (some texts allow
  simple rules such as matching word endings explicitly).
- **C, ambiguous.** Mixed or missing statements, or a permission that names no artifact (for example "the
  permitted checkpoint" with no checkpoint named anywhere in the text). Treat C as B until the dataset package
  or the organiser names the artifact, and ask the user. When the text also allows training from scratch, a
  from-scratch solution needs no answer to the question.
- Never: assume a model is allowed because it is installed, popular or allowed in another challenge; call a
  model "from scratch" when only its head is new; let a pretrained tokenizer or vocabulary into a B solution.
- Write into `RULES.md`: the class, the quoted sentences, and the answer to "would a reviewer reading only the
  code confirm it?". If the answer needs an argument, change the code.

## 3. Gates (all must pass)

1. **State:** OPEN, or the grace hour with a package that can be ready well before the cutoff; the account is
   not shut out of the continuing round.
2. **Time:** hours to close at least cover the critical path: rule lock and data (0.5 h), probe (0.5 h),
   strong version (2 h or more), two full gate runs, upload (0.5 h). Unknown closing time: ask, and assume the
   seats can fill at any time.
3. **Credits:** at least 2 on the shared account (one buys calibration, one the strong version).
4. **Compute:** a credible approach fits this PC within the guarded limits (section 6), or the Kaggle option is
   available and assessed separately.
5. **Rules:** nothing forbids the method family the task needs.

## 4. Ranking criteria (score 0 to 5, one line of evidence per score)

| Criterion | Weight | 5 means | 0 means |
|---|---|---|---|
| Expected gain over the bar | 0.25 | a credible method family should clear the baseline and the current prize or 10th-place cutoff with margin | no credible path above the baseline |
| Probability of a validated strong solution before closing | 0.20 | small clean data, clear metric, 12 h or more left, a known recipe | under 4 h left or major unknowns |
| Hardware fit | 0.20 | everything runs here: about 3 GB VRAM, about 2 GiB RAM per job, experiments of minutes | needs A10G-scale memory or hours per experiment |
| Research leverage | 0.15 | established methods and implementations exist for this exact task, and research is permitted | research forbidden ("do not use ... web search") or nothing transfers |
| Competition status | 0.10 | many seats free, few solvers above the baseline although the bar is beatable, 6/6 credits | 8/10 seats or more, at most 1 credit |
| Invalidation risk (inverted) | 0.10 | clear rules, simple output, an easy deterministic plan | ambiguous policy, review hazards, fragile format |

Total = sum of weight times score. Give the main uncertainty for each challenge and a range when unsure. Never
rank by a speculative maximum. Prize structure (bot `/prizes`): $120, $70 and $40 for the top 3; ranks 4 and
below that beat the baseline share $70. Beating the baseline is the first goal, a top-3 place the second.

## 5. Reading the status signals (Axo skill lessons, `references/prior-knowledge.md`)

- Many submissions and few solvers above the baseline: a plateau just under the baseline, which is hard for
  common methods. Seats are less of a constraint, but the bar is.
- The public board can mislead: on Membrane Confirmation Delay and Station Siting, small test sets with rare
  classes, local grouped CV matched the private result while public ranks were off by up to 19 places.
- A marked "Axo solver" row or "Submitted: Yes" may be another group member's work: ask before relying on it,
  and check the parallel workspace.
- A countdown is as old as its file; the live bot is authoritative.

## 6. Hardware fit on this PC (re-check with `hw_profile.py`)

- RTX 3050 Laptop GPU, 4 GB (3.68 GiB visible to torch); 15 GiB RAM, often only 2 to 6 GiB free, no swap;
  Ryzen 5 5600H with 6 cores and 12 threads (measured 2026-10-12).
- Fits here (estimate): tabular and text tasks with linear, boosted or small neural models; small images
  (28 to 64 px) with small CNNs; from-scratch language models of about 20 to 30 M parameters at sequence length
  256 or less (fp32 AdamW needs about 16 bytes per parameter, so 30 M is about 0.5 GB plus activations); data
  that fits in about 1.5 GiB of RAM per process.
- Kaggle private kernels, through the parallel workspace's runner: T4 16 GB or P100, sessions up to 12 h, 2 GPU
  kernels at once, a weekly GPU quota (`kaggle quota`). Not the grader's A10G (no bfloat16 on a T4). Use it when
  a run needs more than about 3 GB of VRAM or many GPU hours, under that runner's rules only.
- A stated A10G does not rule a challenge out: check whether the description's own method is small.

## 7. What select returns

A table (challenge, gates, the six scores, total, main uncertainty), then: the recommendation (why, expected
range with its basis, plan sketch, time per experiment here), the strongest alternative (why second), what
would change the decision, and the questions for the user.
