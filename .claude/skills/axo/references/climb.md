# After the first score: diagnose, calibrate, then climb

The aim of this phase is the largest margin for the time left, which means spending the time on the changes most likely to pay first. In the transcript challenge the opposite happened: analysis-driven ideas went first, the queue was restarted six times to make room for newer ones, and the plain checks (dropout, weight decay, network size, weight averaging) never produced a single result before the challenge closed.

## Contents
- Biggest step first (read this before anything else in this file)
- Continue improving the work smartly
- Diagnose before you change anything
- Noise: what counts as a gain
- Read the first score
- Calibrate when the first score arrives
- The ladder
- Rules of the climb
- Packaging while climbing
- Watching the board
- Stop-loss

## Biggest step first

The user's decision, 2026-10-08. It outranks the cost order of the ladder below for as long as the place in the top ten is not safe.

**Why.** Admission is by rank and the field moves fast. Measured on 2026-10-08:

- *Anchored Layout Alignment (lost).* Probe 0.239 on the board. Then came two versions built by adding to the probe's model step by step: beam search and more features (0.345), wall-contact features (0.338, lower on the board although +0.016 and 3.2 standard errors better locally). Sweep features, a path re-ranker, off-path training and a minimum-risk decoder each gave nothing or less. Three hours in, the tenth place was 0.391 and the top 0.52. Every experiment was a refinement of one step-by-step scorer whose ceiling was about 0.375 locally; nothing was built to test a different formulation.
- *Variable Coarsening Flow Recovery (same day).* Probe 50.7. One hour of small architecture work gave +0.7. Then twenty minutes of printing one whole raw episode and checking the loss by hand found three things at once: role tokens pair one-to-one across the two sides, the training loss punished confident correct matches, and the cross-channel term pointed the wrong way. The next run went from 48 to 66.5 on the same validation direction. The board's top went from 67 to 98 in the same hour, so even that was late.

**The order of work, starting the minute the probe is handed over:**

1. **Look before building (30 minutes, boxed, mandatory).** Print or plot whole raw examples with their answers, at least ten, and read them. Ask what generates the answer: what pairs with what, what is constant, what a person would use. Check the training objective by hand on one example: does a perfectly right, confident answer get the minimum loss?
2. **Measure the ceilings (minutes each, no training).** For each part of the pipeline, the score if that part were perfect (oracle decoder, oracle first step, oracle candidate filter), and the score of the simplest structural rule (for example "random inside the classes that the data defines"). The largest gap between ceiling and current is where the next version goes.
3. **Write the candidate list with an expected gain and a build time for each, and sort by gain per hour.** A new formulation of the problem, a corrected objective or a newly found structure goes above any addition to the current model. Say the list to the user in that order.
4. **Build the top item as version 2, alone.** Hand it over as soon as it clears the bar; do not wait to stack smaller items on it.
5. **Re-rank after every board update.** Record the gap to the tenth place and to the top. If the top is far above what the current formulation can reach (its ceiling from step 2), the formulation is wrong: go back to step 1, do not tune.

**What is deferred until the place is safe:** settings sweeps, more seeds, extra features of the kind already tried, decoder variants, calibration, and any idea whose expected gain is under about a tenth of the gap to the tenth place. They are the ladder's rungs 1 to 4 and they are for holding a place, not for getting one.

**Stop rule for a family of ideas.** Two additions to the same model inside the noise, or one that gains locally and not on the board, and that family is closed for this challenge. The next experiment must change the formulation, the objective or the inputs.

**Run a wave, not a sequence (user, 2026-10-09: "don't test models sequentially, use the PC's capabilities and try different ones in parallel").** The candidates of step 3 are tested together. Check free RAM, then launch 8 to 9 single-thread processes at once (about 300 MB each on this PC, `torch.set_num_threads(1)`, one configuration per process, results to `out/exp_<tag>.txt`), each a structurally different family (output head, architecture, loss, input information), not a tweak of one model; ensemble the winners from their saved out-of-fold outputs before building anything. Measured on Membrane Confirmation Delay: one wave of output-head families found that wide-and-deep + hierarchical + mixture-cure heads together scored 0.5216 against 0.504 for the plain softmax MLP, with no single head above 0.512.

**No clones (user, 2026-10-09: "why start it if you know it is a clone").** Before a candidate becomes a folder, compare its test predictions with every uploaded version. At 97% shared rows or more it is the same bet (versions 2 and 3 shared 98.3% and scored the identical 0.3852; a later candidate shared 97.6% with version 4): do not build it unless the small change is itself the point of the upload, and say so.

**How this fits the older rules.** The probe still goes first and every rule check stays. The 10-minute cap on analysis ("Working out how the labels were made did not convert into score") applies to open-ended study with no experiment queued; the 30-minute look in step 1 is separate, always done, and must end with the ranked list of step 3. The ladder's "cheapest first" orders the deferred items only.

## Continue improving the work smartly

The user's confirmed preference (2026-10-08): improve the chance of reaching and retaining admission before the submission window ends. The number of experiments or versions is not the objective. Keep the biggest-step-first order above, and update the decision as evidence arrives.

- **Re-rank against the moving cutoff.** After each scored version or material board update, record the latest official score, baseline margin, tenth-place margin, cutoff freshness and observed movement. Compare each candidate's supported gain and uncertainty with that gap. Local significance establishes an improvement, not admission sufficiency; no fixed offset or multiplier guarantees a future board gain. When the approach cannot plausibly close the gap in the available time, say so and reconsider the formulation.
- **Count delivery, not training alone.** Candidate cost includes diagnosis, construction, representative validation, two full exact-script runs, required checks, packaging and manual upload. Rank expected admission impact against that complete cost. Use conservative timings measured under the chosen deterministic backend/device/thread plan, and mark untimed estimates provisional. Set the last experiment start and freeze point backwards from the cutoff. If the critical path no longer fits, report that immediately and preserve the best checked package; do not weaken checks or promise an assured new version.
- **Judge a different model's complementary value.** A weaker standalone model can still help the current system. If combination is legal and there is a credible complementarity hypothesis, make a cheap comparison using saved out-of-fold outputs before rejecting it solely for its own score. Confirm any adoption on representative validation and within the runtime budget. Distinguish "tested and added nothing" from "complementary value not tested"; do not assume a blend would help or always require one.

Keep failures recorded and protect the winner. A feasible modest improvement may still be worth handing over when no stronger checked option fits, but call its admission prospects uncertain. Stop an unproductive family based on the decision-relevant test, not architecture novelty or an untested assumption about complementarity.

## Diagnose before you change anything (the pack's protocol)

Source: the challenge-engine pack (`references/pack/`, `improvement_protocol.md`). Input: the current solution, the logs, the score, any traceback or errors, optional failure examples.

**Order of diagnosis.** Do not jump to architecture changes while an earlier item is the real problem:

1. runtime correctness; 2. data and split correctness; 3. leakage and compliance; 4. validation quality; 5. optimisation and training stability; 6. architecture and features; 7. post-processing and thresholds.

The ladder below orders cost, this list orders what to suspect: settle 1-4 first, then run the cheaper rung among the remaining candidates. A post-processing change on saved predictions is cheap, and it needs the same evidence as any other change.

**Evidence labels.** Label every proposed change CONFIRMED (directly supported by the code, logs or error), STRONG HYPOTHESIS (highly plausible, not proven) or SPECULATIVE (weak evidence). Do not implement a speculative change while a confirmed or strong-hypothesis fix exists.

**Preserve the winners.** `NOTES.md` keeps one line per version (version, score, key changes, training time, validation behaviour, known failure modes) and a `BEST:` line. A change is a new candidate file until it is measured. Never replace a higher-scoring valid solution with a lower-scoring one because the new approach is more sophisticated; a worse result is discarded, analysed and recorded, and work returns to the best.

**Research** only through `research-gate.md`.

## Noise: what counts as a gain

The same configuration differs from run to run. In the spoken-corrections challenge identical two-seed ensembles scored 0.5058, 0.5038 and 0.4933 on one 35-speaker hold-out, and a last version that gained +0.004 locally scored 0.012 below its predecessor on the board. So:

- Measure the seed-to-seed spread of the reference once (the same configuration run twice) and write it in `NOTES.md`; a gain below about twice that spread is no evidence.
- Compare candidates with grouped K-fold over all rows (one model per fold, every row scored once, out-of-fold predictions), not one small hold-out; tune decision thresholds on the out-of-fold predictions, not on the hold-out that chose the model.
- A gain that stays inside the noise is handed over, if at all, as a lottery ticket and said to be one. The board keeps the best version, so it costs a credit and the user's attention.

## Read the first score (5 minutes, when it arrives)

The first score is the probe's (`fast-v1.md`); it may be below the baseline. Record three numbers in `NOTES.md` and decide from them:

| Number | How | What it decides |
|---|---|---|
| Board minus local | the probe's board score minus its hold-out score | how far local validation can be trusted; which hold-out design to keep |
| Board noise | per-row score spread / sqrt(test rows) | how large a gain must be to show on the board; the chance a version lands under the baseline or under the previous one |
| Distance to baseline and to the top | board score against both | under the baseline by less than two noise units → tune the same approach; far under, or the top far above → the approach itself has to change |

Measured: soundtrack challenge, board 0.3699 against local 0.39–0.40, noise 0.021, baseline 0.041 above chance.

**Check the gap before anything else.** If the board is far from local, by more than the fold-to-fold spread and the board's noise together, nothing below applies yet. Stop building, submit no stronger version on top of an unexplained gap, and find the cause in the diagnosis order above, starting with what local runs cannot show: whether the platform's run of the script gives the file we uploaded, whether our copy of the metric is right, and whether the test answers behave like the training answers. A cheap way to locate the failing part is to replay failure modes on held-out predictions (drop one component, break one output column) and see which reproduces the board score. Measured: Tray Elytra Metrology, local about 54 (folds 40 to 61), board 9.43; the replay showed that only the loss of a whole component gives a score that low, but the first submission came after 2.5 hours and the challenge closed before the cause was found.

Work on the next candidate does not wait for this step; it started when the probe was handed over. The score redirects it, or stops it.

## Calibrate when the first score arrives (20 minutes)

1. **Compare** the leaderboard score with the local hold-out score of the same configuration. Record both in `NOTES.md`.
2. **If local overstates**, rebuild the hold-out from fewer, larger groups (about 12 clusters of simple input statistics, whole clusters held out, 4 folds) and re-score the submitted configuration, until local is about equal to the leaderboard. Every later decision uses this split. Measured: 60 small groups gave 0.339 where the leaderboard gave 0.2915; 12 large groups gave 0.2907.
3. **Two one-minute checks** on the saved held-out predictions decide which rungs are worth climbing:
   - *Is the confidence accurate?* Bin predictions by confidence and compare with accuracy. If they match, the decision is already as good as the model allows: skip decision biases, priors and prediction averaging over shifted inputs (all three were measured as no gain on a calibrated model).
   - *How far apart are training-set and held-out scores?* Far apart → regularise. Close and both low → more capacity or better inputs. Training-set score low even with long training → part of the labels cannot be predicted from the input; expect small gains and consider the stop-loss early.
4. **Fix the reference:** the submitted configuration's first-fold score is the pruning threshold; its pooled score is the number to beat.

## The ladder

Write the whole queue at once, cheapest and most likely first, then start it and leave it running.

| Rung | What | Cost |
|---|---|---|
| 1 | Decisions on saved predictions | seconds, no training (skip if the confidence check says calibrated) |
| 2 | Standard settings, one change each: passes ×0.5 and ×2, dropout up, weight decay up, width ×0.5 and ×1.5, learning rate ×0.5, weight averaging | one pruned run each |
| 3 | Standard augmentation for the data type, one at a time | one pruned run each |
| 4 | Scale: more seeds or models, more exact test-time views, up to the runtime limit | measured once; usually small |
| 5 | Data-specific ideas, one at a time | 30 minutes each including the run, written down before starting |

A queue file for rung 2 looks like this (`scripts/run_queue.py`, options shared by every line in `--common`):

```
half      epochs=20
double    epochs=80
drop      dropout=0.4
wd        weight_decay=0.2
narrow    width=(24,48,96,128,160)
wide      width=(48,96,192,256,384)
lowlr     lr=1e-3
```

## Rules of the climb

- **Prune from the first run.** Every experiment gets `prune=<reference first-fold score>`; a losing idea then costs one fold.
- **Never kill a running experiment to reorder.** The queue is a file that is re-read before each run: put the new idea in the pending lines. Each restart in the transcript challenge threw away the fold in progress, about 30 minutes in total.
- **One change per experiment**, against the same reference, on the same folds.
- **Analysis cap:** at most 10 minutes of looking at data or errors without an experiment in the queue that came from it.
- **Two-strike rule:** two data-specific ideas in a row with no gain → return to the unfinished rungs above, or stop.
- **Log every result** in the `NOTES.md` table as it lands, including the losers, so nothing is tried twice.

## Packaging while climbing

- **Submit increments.** When a configuration beats the last submitted version on the calibrated split by more than the noise, package it and hand it over. Measure the noise once (see "Noise" above); a gain below about twice the seed spread is not worth a credit unless grouped K-fold confirms it. Credits refill every four hours and are worth nothing at closing.
- **The gate needs the GPU.** Add a line `STOP` to the queue file; when the current experiment ends, run `scripts/gate.py`; then remove `STOP` and start the queue again (tags already started are skipped).
- **Keep the candidate's full run at 20 minutes or less**, so the gate fits inside the one-hour window that remains if the slots fill.
- The submitted file is always a clean `solution.py` with only the adopted options; experimental switches stay in the development copy.

## Watching the board

Ask the user for the status block at every hand-over and whenever a result changes the plan. Solvers who beat the baseline take slots; at 10 the challenge closes to everyone except those with a scored submission, who get a limited window (one hour in the current workflow).

- **At 8/10:** stop exploring. Package the best configuration that beats the last submitted version; if none does, say so.
- **At 10/10:** only a candidate whose gate fits in the remaining window is worth starting.
- **After each score, read the paired difference.** Cross-tabulate the test rows that changed between the two nearest uploaded versions (our own predictions, no labels needed): which classes moved shows what the board rewarded. Membrane Confirmation Delay: version 3 -> 4 changed 38 of 540 rows and gained +0.037; 29 of them were rare-class labels moved to class 0 (17 class 3, 9 class 2, 3 class 1).

## Stop-loss

Stop and tell the user the realistic outcome when either is true:

- 90 minutes have passed since version 1's score with no gain on the calibrated split, or
- rungs 1–4 are finished and nothing beat the reference by more than the noise.

Before recommending to stop or switch, make one targeted research pass if the rules permit it and the bottleneck is specific (`research-gate.md`). Give the numbers (best local, leaderboard, top of the board), say what is left untried and what you expect from it, and recommend continuing or switching challenge. The decision is the user's.
