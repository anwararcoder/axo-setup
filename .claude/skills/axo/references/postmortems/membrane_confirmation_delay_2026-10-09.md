# Postmortem - Membrane Confirmation Delay (2026-10-09; CLOSED: public 9th at 0.4259, FINAL private 2nd at 0.5274)

## A. Challenge facts

- Task: From Scratch, CPU, macro-F1 over 4 classes. For each of 848 training / 540 test compounds, 12 dose-response curves (2 cell systems x GLO/FLOR x 0/8/16 h x 15 concentrations; every numeric column rank-encoded with a reversible calibration file) plus SMILES; predict the first of 24/32/40 h at which FLOR reaches 25 at the anchor (class 0 = none). Training classes 640/89/62/57. Rules: neural model trained from scratch, predictions from the trained model, no pretrained weights, no test-to-test use.
- Split: scaffold families, 619 train / 241 test (own Murcko-style parser reproduced 617 / 241). One test family (a single benzene ring) holds 192 of 540 test compounds (36%); training has 24 such compounds and its largest family has 23.
- Board (baseline 0.4048, first feedback said 0.4006): v1 0.3906, v2 0.3852, v3 0.3852, v4 0.4224, v9 0.4254 (best). Version 10 (local 0.5225) scored 0.4259, the best by 0.0005 over v9 (the same within noise: 97.2% of its test rows are shared with v9). At the grace hour (24 min left): top 0.4975, second 0.4919, eighth (us) 0.4254, tenth 0.4210, twelfth 0.4191; 21 submissions; 1 credit left.
- Local: 5-fold StratifiedGroupKFold on own scaffold keys, three fold seeds, five model seeds, pooled OOF macro-F1; fold-seed spread +-0.015. Local 0.496 (v1), 0.514 (v2), 0.520 (v3, v4), 0.522 (v9): board about 0.10 lower for every version, and the local order of v1-v3 did not match the board.
- Versions: v1 46 features around the anchor, MLP 96-48, class weights ^0.75; v2 26 features with the other cell smoothed 1-2-1, 4 heads; v3 + four views; v4 no absolute concentrations, weights ^0.5, 5 net types (0.4224); v9 wide-and-deep + hierarchical + two mixture-cure heads, 20 nets (0.4254); v10 v9 + descriptor-only structure network at 30% (local 0.5225, 97.2% of test rows shared with v9).
- Paired board difference v3 -> v4: 38 of 540 test rows changed (93% shared), +0.037; 29 of them were rare-class labels moved to class 0 (17 class 3, 9 class 2, 3 class 1).
- Worked: fewer inputs (26 > 46 > 120); no absolute doses or top doses; smoothing only the cell that was not used to select the anchor (+0.01); the other cell as a block (-0.03 without); class-weight exponent 0.5-0.75; an ensemble of different output heads (0.5216 against 0.504 for the plain softmax MLP).
- Did not work (three fold seeds): neighbouring concentrations, smoothing the anchor's own reading, shift-extrapolation and crossing-time features, eligibility features, GLO shape features, concatenated SMILES descriptors (0.467) or fingerprints (0.445), structure-neighbour (Tanimoto kNN) features, label smoothing, input dropout, bagging, robust scaling with smooth clipping (0.475), TabM-mini, additive nets, periodic embeddings (0.43), CNN over the full curves (0.477), RBF (0.40), prototype (0.49), logit-adjusted loss with the cure/hierarchical/wide heads, per-class weights that nearly remove class 3 (0.485), per-class offset tuning on OOF (-0.01 to -0.02 on held-out halves), LightGBM as a ceiling check (0.47).
- Timing: probe folder 11 minutes after the data (gate 0.6 min, one run 13 s); ensembles 1 minute per run on one CPU thread; exact-ensemble CV 3 to 5 minutes with three processes in parallel; about 140 minutes of work in the session.
- Never found: what the solvers at 0.44 to 0.50 did differently.
- User statements: "v2 did worse than v1, didn't you validate on the same data"; "you didn't research what approaches work best ... from scratch and that really is hurting my standing"; "why start it if you know it is a clone"; "don't test models sequentially, use the PC's capabilities ... in parallel"; "why are you giving me a submission that will perform worse"; "0.45 is supposed to be easy, most of the people got it".

## B. Reusable lessons (promoted to lessons.md, research-gate.md, climb.md, submission-package.md)

1. When the test is dominated by one group absent from training, drop features that identify the series or tested range, check the group sizes of the public test inputs at intake, and expect the board far below local (CONFIRMED, v3 -> v4).
2. Short methods-only research right after the probe, a parallel wave of structurally different model families, and no clone folders (CONFIRMED by three user corrections).
3. With a macro metric and weak rare classes, rare-class false alarms are the cheapest points: start the class-weight exponent at 0.5 and read the paired board difference by class (CONFIRMED on this board, HYPOTHESIS as a general setting).

## C. Final result (reported by the user on 2026-10-09, folder deleted afterwards at their request)

- Private leaderboard: 1st 0.5414, 2nd 0.5274 (us, +7 places), 3rd 0.5149, 4th 0.5049 (+9), 10th 0.4767; private baseline between 0.4767 and 0.4840. Public: top 0.5398, us 9th at 0.4259 (version 10). 42 submissions.
- Local scaffold-grouped CV (0.52) matched the private score; the public board read 0.10 lower. The "+0.037" of version 3 -> 4 and the rare-class reading are public-only evidence; which version would have been best on the private part is not known, because the board kept the best public version.
- "Never found: what the solvers at 0.44 to 0.50 did differently" is answered for the winner only: its code is summarised in recipe 15. Differences from ours: three feature views with one small network each instead of one input set; very short full-batch training (40 steps) on the smoothed view; softmax + ordinal heads instead of wide/hierarchical/cure heads; 480 fold models as the test ensemble instead of 20 networks trained on all rows; bagged out-of-fold class weights instead of a fixed exponent; absolute log concentration kept. Its script chose its data path from whichever folder existed and was accepted, which says nothing about how strict the platform's check is on other files.

## D. Lessons added at closing (lessons.md)

4. The public board was the outlier and local grouped validation matched the private result (CONFIRMED once).
5. A rejected idea is rejected only in the form it was tested; the winner used several of ours in a view-per-network, bagged form (HYPOTHESIS: a 0.014 gap on 540 compounds).
