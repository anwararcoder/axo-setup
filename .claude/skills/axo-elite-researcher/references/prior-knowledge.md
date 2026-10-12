# Prior knowledge reused from the Axo skill

The Axo skill (`.claude/skills/axo/`, read-only from here) holds the lessons of seven postmortems, 2026-10-04 to
2026-10-11. They are reused below as starting points. Check each one against the current challenge's rules and
data before acting on it, and cite the source file when you do.

## Contents
- Workflow lessons
- Platform facts
- This machine
- Inconsistencies found in the Axo skill

## Workflow lessons (CONFIRMED there unless marked)

1. **Probe first.** Tray Elytra Metrology's strong first version validated at about 54 locally and scored 9.43
   on the board; a probe in the first half hour would have shown the gap (`fast-v1.md`).
2. **Biggest step first after the probe.** Read raw examples (30 minutes), measure ceilings, rank ideas by
   expected gain per hour. Step-by-step refinement lost Anchored Layout Alignment; twenty minutes with one raw
   episode gave +18 points on Variable Coarsening Flow Recovery (`climb.md`).
3. **Waves, not sequences; no clones.** One parallel wave of different output heads found the best Membrane
   model; versions sharing 98.3% of test predictions scored identically (`climb.md`).
4. **Noise.** Identical configurations differ; compare paired differences on grouped folds. A local gain of 3.2
   standard errors scored lower on a 276-item board (Anchored Layout Alignment, `lessons.md`).
5. **The public board can be the outlier.** Membrane (public 9th, private 2nd) and Station Siting (public 20th,
   private 2nd): local grouped CV matched the private result (`lessons.md`).
6. **Determinism and platform checks.** The deterministic skeleton from the first file; "Prompt Compliance" and
   "Deterministic Execution" can stop compliant files several times; plan for retries (`determinism.md` §8,
   `platform-rules.md`).
7. **Human review after scoring.** The Enclosure Mention scored 0.835 and was rejected for a cross-row step its
   markdown had ruled out: strict reading of each challenge's own text (`compliance-firewall.md`).
8. **Deadlines.** Compute the whole critical path (two gate runs and the upload) before promising; freeze early
   (FlockSight, `lessons.md`).
9. **Macro metrics with rare classes.** False alarms on rare classes are the cheapest points (Membrane); one
   rare test row can be worth 0.01 (Station Siting). HYPOTHESIS as a general setting.
10. **Research right after the probe** (Membrane, the user's correction). This skill moves research earlier,
    before the main solution, by the user's request of 2026-10-12.
11. **Recipes** for 17 task types with what was measured: `recipes.md`. Use one as the probe's starting point
    when the task matches; its numbers come from another PC and another dataset.

## Platform facts (dated; the live bot wins)

- 10 solver seats. When they fill: a grace hour in which anyone may submit, a lockdown of up to one hour, then
  only the top 10 keep submitting until closing (bot `/instructions`, 2026-10-06 and later).
- Credits: 6 per challenge, one back every 4 hours; only a submit costs one; scoring takes 15 to 30 minutes.
- Prizes: $120, $70, $40 for the top 3; ranks 4 and below that beat the baseline share $70.
- A submission is a prediction file, the solution code, a description (at most 1500 characters when pasted, as
  recorded in the parallel workspace's `CLAUDE.md`) and the spent time.
- Automated checks before scoring: Prompt Compliance, Deterministic Execution, CSV Score Validation. A generic
  block returns the credit; a retry is possible after five minutes.
- The solution file must not contain the platform's name (bot, 2026-10-10).
- Script contract: `public_dir` / `submission_out` argv pattern, a Kaggle fallback path, about one hour of
  runtime, standard Kaggle image libraries, no download or extraction code (`platform-rules.md`).
- The board account and its credits are shared by a group: "Submitted: Yes" can be someone else's work.
- Confidentiality: challenge data, descriptions, links, submissions and code are private; publishing them can
  get the account blacklisted (bot `/rules`).

## This machine (hw_profile.py, 2026-10-12)

Zorin OS 18.1, kernel 7.0; AMD Ryzen 5 5600H, 6 cores and 12 threads; 14.97 GiB RAM and no swap; RTX 3050
Laptop GPU with 4096 MiB (3.68 GiB visible to torch), driver 580.178, CUDA 13.0. ML environment
`/mnt/work/shipd-ai-eris/axo-eris/.venv-ml`: torch 2.14.1+cu130, scikit-learn 1.9.1, lightgbm 4.7.0,
transformers 5.19.0, timm 1.0.30; no xgboost or catboost. A Docker Desktop VM held about 2 GiB of RAM. The
Kaggle image of October 2026 had python 3.13, torch 2.11, numpy 2.1 and pandas 2.3 (parallel workspace), so
write code that runs on both.

## Inconsistencies found in the Axo skill (2026-10-12; that skill is not edited from here)

- `fast-v1.md` describes an older PC (RTX 3060 with 6 GB, Windows): do not size plans from its timings.
- `recipes.md` (section 1) and `target-env.md` list mixed precision as a default or as open, while its
  `determinism.md` section 8 and its gate forbid it in submitted files: follow section 8.
- `lessons.md` says a missing optional resource gets a fallback; the fixed-plan rule says no fallbacks: follow
  the fixed plan.
- `lessons.md` caps a version 1 run at 8 minutes, `fast-v1.md` at 2 minutes: use the 2-minute target.
- Its helpers `keep_awake` and the power check work on Windows only, and `kill_tree` stops only the direct
  child on Linux (`audit-finalize.md`, section C).
- Its lesson "on the 6 GB card a run that exceeds the memory does not fail, it crawls" came from Windows. On
  this Linux PC an over-full GPU is expected to raise a CUDA out-of-memory error instead (not yet observed here).
