# Research before the main solution

## Contents
1. When, and how long
2. What may be researched
3. Questions to answer
4. Sources and how to record them
5. Class A: pretrained models permitted
6. Class B: training from scratch
7. Output

## 1. When, and how long

- After `select` and the rule lock, before the strong version. The probe can be built in parallel when the
  clock is tight: the probe is not the main solution.
- Time box (ours): the smaller of 2 hours and 15% of the hours left; 30 minutes or less when under 6 hours are
  left or 8/10 seats are taken. Stop earlier when sources start to repeat.
- Blocked when the description forbids it. A strict reading of a rule such as "do not use ... web search" rules
  out web research for that challenge: then work from knowledge already held, the description's own guidance
  and local experiments, and say so.

## 2. What may be researched

- Methods, architectures, training and evaluation techniques, library documentation.
- Never: the challenge's own text, data, ids or answers; its source corpus (the original texts, dumps or image
  archives the data was built from); other solvers' code or the leaderboard; domain resources that the rules
  forbid as inputs. When external linguistic resources are banned, a grammar or dictionary of the challenge's
  language is one: reading it to hand-write rules imports it into the solution.
- No challenge material leaves the machine. Queries describe the task in generic terms ("morphological
  segmentation with lemma copy from a context sentence"); never paste rows or description text into a search.

## 3. Questions to answer

1. The exact task family and metric, and the formulations that exist for it (classification, ranking,
   generation, structured prediction, matching).
2. The state of the art and the strong simple baselines for that family and metric; what wins on small data
   from scratch and what wins with pretraining.
3. Inductive biases and failure modes specific to this data: invariances, structure, label noise, and the shift
   between training groups and test groups.
4. Training recipes: optimiser, schedule, regularisation, augmentation (label-exact only), duration, and what
   the ablations show actually matters.
5. Inference-time gains: decoding under constraints, calibration, test-time augmentation and ensembles where
   the rules allow them.
6. Memory-efficient implementations for 4 GB of VRAM: smaller widths, gradient accumulation, activation
   checkpointing. Mixed precision is for development speed only: the submitted file stays in fp32 (Axo
   determinism skeleton), so re-measure any fp16 finding in fp32.
7. Validation and leakage: how this kind of split is validated and which leaks are typical.

## 4. Sources and how to record them

In order of trust:
1. Primary papers (arXiv, ACL Anthology, CVF, OpenReview, PMLR, JMLR): the method and the ablation tables,
   with their dataset and metric.
2. Official documentation of the libraries in the target image.
3. Reference implementations: the authors' code or maintained repositories (licence, last update).
4. Benchmark studies and write-ups of similar competitions, for their methods only.

For each source record: URL, publication or update date, the claim, its numbers labelled `published` with
their dataset, why it applies here (data size, metric, rules), the cost on this PC, and the compliance verdict.
A paper's gain on another dataset is a hypothesis for this one, not evidence.

## 5. Class A: pretrained models permitted

- Candidates: domain-appropriate first; compare two or three realistic sizes, not the largest by default.
- For each: licence and provenance (who trained it, on what; exclude anything trained on this challenge's data
  or a domain the text excludes), memory (4 GB here, A10G at the grader), and availability at grading time
  (provided in the dataset, attached, or downloadable by the script only if the text allows network access).
- Options to compare: frozen features with a light head, full fine-tuning, parameter-efficient tuning (only if
  the method is in the target image or written by hand), distillation into a small model.
- Data sufficiency: labelled examples per class, few-shot behaviour, the overfitting risk of fine-tuning.
- Decide by the official metric on the validation split, never by popularity. Ensembles only when the measured
  gain exceeds the noise and the runtime allows it.

## 6. Class B: training from scratch

- No downloaded weights, embeddings, tokenizers or vocabularies built elsewhere. Every learned parameter is
  initialised randomly in the submitted file and trained on the provided files only. A pretrained backbone
  with a random head is not from scratch.
- Compare structurally different families that suit the data instead of assuming one: counting and rule
  baselines (often strongest with tens of examples), gradient boosting on engineered features, small CNNs and
  residual networks, small transformers or recurrent networks, graph networks for graph structure, classical
  statistical models.
- Study initialisation, normalisation, regularisation, label-exact augmentation, schedules and training length;
  data efficiency matters more than size.
- State in the code and in the description how each learned part is initialised and trained.

## 7. Output

`$C/RESEARCH.md` from `assets/research_report.md`: findings with sources and labels; ranked candidates with
expected gain (range and basis), build hours, run cost here, risk and compliance; the probe definition; the
first wave (two to four structurally different families); the validation design; open questions for the user.
