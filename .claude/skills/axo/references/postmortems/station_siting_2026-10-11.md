# Station Siting: Monitor Class from NO2 and Ozone — postmortem (closed 2026-10-11)

From Scratch, CPU, 90 minutes. 1,457 training stations, 412 test stations from countries absent from training; six
28-day windows of hourly NO2 and ozone per station, each divided by its own mean. Score: balanced accuracy as skill
over chance on station type and area type, averaged. Baseline 0.4984.

## A. Challenge facts

**Result: FINAL private 2nd, 0.5203 (winner 0.5219). Public 20th, 0.4761.** Private ranks moved by up to 19 places;
the public leader region (0.52 to 0.60) fell to 0.49 to 0.52 on the private set.

| Version | Local country folds (mean / pooled) | Public | What |
|---|---|---|---|
| v1 probe | 0.395 / 0.338 | 0.3543 | 156 window-averaged shape features, MLP 128-64 |
| v2 | 0.451 / 0.450 | 0.4628 | 396 features per window, one small network per window, six encodings averaged, 20 passes, class weight (1/freq)^1.5 |
| v3 (best, scored privately) | 0.470 / 0.461 | 0.4761 | v2 + month-wise standardisation + titration slope block, 10 seeds |
| v4 | 0.495 / 0.469 | 0.4417 | v3 + same-station contrastive CNN features + raw CNN; which model served which axis was chosen on the same five folds |
| v5 | 0.483 / 0.426 (folds 1-4: 0.504) | never scored | v3 with weights by class and country group; stopped by the compliance check until lockdown |

- The step that mattered: describing each window separately and averaging the encodings (window-averaged MLP 0.36 to
  0.38 against 0.45), then short training and stronger rare-class weights. Everything after v3 was inside the noise.
- About 150 experiments converged at 0.44 to 0.48 locally: boosted trees 0.37, linear 0.42, deep set over days 0.37 to
  0.39, 1-D and 2-D CNNs 0.38 to 0.45, random convolution kernels 0.39 to 0.42, contrastive embeddings 0.41 to 0.44
  alone, compact 44-feature set 0.43 to 0.46, V-REx, simulated shift, feature filters, joint and ordinal heads, GCE,
  label smoothing, window mixing, blends. In-country control (random folds) 0.546.
- Measured and never used: value granularity (no signal), pooled statistics of held-out stations (0.44, no gain).
- Removed on a strict reading: per-station clock alignment to a training template (+0.008 locally; the rules say do
  not align evaluation stations).
- Local against official: v2 and v3 landed within 0.012 of the local fold mean on the public board, v4 0.05 under it,
  and the private score of v3 was 0.05 above its local mean. With rare classes of a few dozen test stations, one
  station is worth about 0.01.
- Compliance: the automatic "Prompt Compliance" check stopped v2 twice, v3 three times, v4 once and v5 until lockdown,
  each time with the same generic text and the credit returned; identical files passed on a later attempt. The first
  upload of v1 was refused for the platform's name in a comment.
- Time: probe folder 35 minutes after the data; full runs 0.4 to 0.8 min (v4 9.9 min).

## B. Reusable principles

1. CONFIRMED (second time, after Membrane): on a small test set with rare classes the public board reorders by
   noise; the locally validated simple version won privately while public rank said 20th.
2. CONFIRMED: a version whose composition was chosen on the same folds that score it (v4) lost 0.05 on the board.
3. CONFIRMED: the compliance check fails on unchanged, compliant files several times in a row; a version handed over
   inside the grace hour (v5) never got scored.
4. HYPOTHESIS, untested: operating points should be chosen on the folds whose training set resembles the final
   training set (here the four folds that train with the rare-class-rich country), not on the pooled score.

## C. The winner, for comparison (script shared by the user on 2026-10-11)

Private 0.5219 against our 0.5203. Station-level vectors (per-window features aggregated as mean, spread, summer mean, winter mean), quantile-transformed; station type from two strongly regularised logistic regressions, area type from dropout-heavy MLPs; clock offset removed by anchoring each station on its own ozone peak. Details are in recipe 17. HYPOTHESIS, one case: for the axis with rare, noisy classes a heavily regularised linear model was the safer family, and a different family per axis is worth testing early as a planned design, not as a late choice on the scoring folds (our version 4).

The winner was 10th on the public board at 0.4973, itself under the 0.4984 baseline there, and first privately at 0.5219: the public shard read both of the top two private scores about 0.03 to 0.045 too low.
