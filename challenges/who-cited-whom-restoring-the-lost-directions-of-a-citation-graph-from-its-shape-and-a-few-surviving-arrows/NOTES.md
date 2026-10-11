# NOTES — Who Cited Whom? (CPU, MCC, baseline 0.5478)

## Clock

- Picked: 2026-10-11 19:53 UTC (`/axo select`, user "go"). Data on disk: 19:55 UTC.
- Closing time: not known. Status at pick (bot comment, 19:52 UTC fetch): OPEN, 1/10 beat AI, top 0.5486, 39 submissions,
  credits 4/6 (the group's account), Axo row 5th at 0.5455.
- Spent time: tracked from 19:53 UTC.

## Machine and runs

- Linux, RTX 3050 Laptop 4 GB (unused: CPU challenge), 12 cores, 14 GiB RAM with about 2.5 GB free; another Claude
  session (axo-eris) is working on FragmentTrace through Kaggle kernels.
- Python: `/mnt/work/shipd-ai-eris/axo-eris/.venv-ml/bin/python` (numpy 2.5, pandas 3.0, scikit-learn 1.9).
  Platform (seen by the other session on Kaggle): python 3.13, numpy 2.1.3, pandas 2.3.3: write for pandas 2.x.
- Every run under `systemd-run --user --scope -q -p MemoryMax=2G` (this PC freezes when RAM runs out).

## Data facts (inspect_data.py)

- 729 training graphs, 30-80 papers (median 53), 31-452 links (median 93); 294 test graphs, median 52 papers.
- Labels balanced (0.4991). Test graphs keep 10.6 % of directions on average (0 to 21 %), one test graph keeps none.
- 702 of 729 training graphs are acyclic; 27 have a cycle. The true closure orders a median 14 % of node pairs.
- Median 23 papers per graph nobody in the graph cites (newest), 10 that cite nobody (oldest).
- corr(degree, out - in) median -0.52: high degree = cited a lot = old.

## Validation protocol (fixed for the whole challenge)

`dev_eval.py`: training graphs clustered on shape statistics (log n, log m, density, degree CV, max degree / n) into
12 KMeans groups, 4 folds of whole groups; held-out graphs get a simulated 10 % mask (seed 777); MCC pooled over their
answered links. Probe: fold 0 only (176 graphs, 16,802 links).

**Correction (20:10 UTC):** fold 0 is the hardest shape cluster (small sparse graphs). The probe pooled over all four
folds: cluster design 0.4999 (folds 0.4077 / 0.5333 / 0.4294 / 0.5626), random-graph design 0.5003 (0.5164 / 0.5124 /
0.4608 / 0.5091). Both designs agree pooled; **the protocol is the pooled 4-fold cluster design**. Fold score tracks
graph size (fold 3 = 27,763 links, 0.56). Mask draw alone moves a fold by about 0.01 (dev_eval vs canonical masks).
Canonical masks for all later experiments: held-out graph i keeps RandomState(777 * 1_000_003 + i) < 0.1.

## Ceilings (ceilings.py, true directions, no training)

Orienting each link by comparing one per-paper number: true cited share in/(in+out) 0.945; in - out 0.852; in-degree
0.784; never-cited flag 0.515; degree 0.360; peeling order 0.257. 55 % of links start at a never-cited paper.
=> the problem is estimating each paper's cited share.

## Research pass (methods only, 3 searches, 20:00 UTC)

Peeling / Jordan centrality arrival-order estimators (network archaeology); maximum acyclic subgraph / feedback arc
set decoding; iterative classification (relational features from predicted labels).

## Versions

| Version | Local (pooled 4-fold) | Board | Key changes | Full run | Notes |
|---|---|---|---|---|---|
| v1 probe (`solution_v1.py`, `submission_v1/`) | 0.4999 (fold 0 0.4077 quoted at hand-over); 0.4935 on canonical masks | **0.5075** (S153, ~20:45 UTC) | 41 node + pair + graph features, HGB 200 it., 1 mask draw, swap augmentation | 0.7 min, IDENTICAL | handed over 20:06 UTC; uploaded by ~20:25 UTC as Version 1, review pending, credits 3/6 |

| v2 (`solution_v2.py`, `submission_v2/`) | 0.5241 (3-seed average; seed 0 alone 0.5247) | **0.5413** (the user's own board score; board - local +0.017; 0.5520 on the board is another solver) | two-round message-passing network from scratch, antisymmetric link head, fresh 10 % draw per pass, 30 passes, seeds 0/1/2 averaged, each test graph alone | 2.0 / 2.1 min, IDENTICAL | built 20:55 UTC at the user's request, below the baseline locally (gate `--below-baseline-ok`) |

BEST: v2 (local 0.5241, board 0.5413 = the user's; not first). Target 0.5550 (+0.0137 board, about +0.016 local). Credits after v2: not verified (user checks /my_work).

## Experiments (fold 0 unless stated; canonical masks)

| Tag | Result | Reading |
|---|---|---|
| reg_leaf / reg_slow | 0.4149 / 0.4092 (dev_eval masks, v1 0.4077) | regularisation no gain: the gap was the fold, not memorising. CLOSED |
| ICA 3 stages, 4 folds | f0 0.4195->0.4270->0.4313; f1 0.5155->0.5381->0.5391; f2 0.4289->0.4416->0.4386; f3 0.5497->0.5718->0.5757 | stage 2 +0.016 mean, positive on all folds; stage 3 ~0 |
| GNN d64 L4 30 ep, seeds 0/1 | 0.4340 / 0.4306 (tree same masks 0.4195) | seed spread ~0.003 |
| GNN 80 ep / d96 L6 60 ep | 0.4241 / 0.4300 | more capacity no gain |
| GNN 2 rounds, 30 ep | 0.4375 | +0.004 over base, near noise |
| GNN 2 rounds, 4 folds | 0.4375 / 0.5548 / 0.4485 / 0.5884, **pooled 0.5247**; probe-tree on same links 0.4935 (paired +0.018/+0.039/+0.020/+0.039) | the step: +0.031 pooled, all folds |
| ICA3 trees pooled / blend with GNN / acyclic decoding | 0.5132 / 0.5256 (w_gnn 0.8) / 0.5250 | blend and decoding inside noise. CLOSED |
| GNN seeds 1, 2 pooled; 3-seed average | 0.5231, 0.5228; average 0.5241 | seed spread ~0.001 pooled; averaging no gain (kept for stability) |
| Error look (GNN, OOF) | citing paper never cited: 0.846 right; both papers mid-aged: 0.675; max degree <= 4: 0.611; > 16: 0.869; calibrated | the loss is in middle-aged pairs |
| comparison-message network (exp_mpnn) | fold 0 0.4286 (GNN 0.4375) | worse. CLOSED |
| GNN + distance features (ecc, closeness, harmonic, reach3, peel) | 0.4328 / 0.5560 / 0.4488 / 0.5908 | inside noise. CLOSED |
| solution_v2.py own functions, fold 0, seed 0 | 0.4390 (experiment 0.4375) | the submission file reproduces the experiment |

| average of GNN variants (r2 3 seeds + distance features) | 0.5254 (r2 3-seed 0.5241) | noise. CLOSED |
| per-paper age targets as extra loss (position depth/(depth+height) - exact on 702 acyclic graphs -, cited share, height); weight 0.5 / 1.0 / 0.5 + position term in link score | f0 0.4298 / 0.4310 / 0.4305 (ref 0.4375); f1 0.5563 / 0.5588 / 0.5542 (ref 0.5548) | noise or worse. CLOSED |
| GNN training-graph score, new mask | f0 model 0.5665 (held-out 0.4375); f1 model 0.5355 (held-out 0.5548) | no overfitting: capacity / regularisation / augmentation are not the bottleneck |
| spectral seriation (Fiedler vector) ceiling | oracle sign 0.132, sign from known 0.052, from degree -0.051 | graphs are not banded in time. CLOSED |

| (after v2, target 0.5550) GNN 3 rounds / 4 rounds x 2 layers | pooled 0.5244 / 0.5260 (r2 0.5247) | noise. CLOSED |
| graph transformer (full attention, link / known / distance / predicted-direction biases), alone | 0.4149 / 0.5330 / 0.4151 / 0.5646, pooled 0.5002 | weaker alone |
| v2 GNN (3 seeds) + transformer blend, w 0.2 / 0.3 | pooled 0.5258 / 0.5256 (v2 0.5241); per fold +0.004, +0.002, -0.003, +0.002 | +0.0017, not consistent: a lottery ticket, not built |

## Correction (user, 2026-10-12 ~21:15 UTC)

The 0.5520 row marked "<- @..." on the channel board is ANOTHER solver (the marker names a handle, now
"@Alaaeeldeen"). **The user's own public score is 0.5413** (given by the user; v1 = 0.5075 by S153, so 0.5413 is v2 by
elimination - user asked to confirm with v2's scored message). Board - local: v1 +0.014 (canonical 0.4935), v2 +0.017
(0.5241): offset about +0.015. Target 0.5550 = +0.0137 board = local about 0.540 (+0.016 over v2).
Credits: channel block says 2/6, but it showed 4/6 before the user's v1 while the user's bot said 3/6 after it - the
channel number may not be the user's; the user checks /my_work. Live board 21:14 UTC: 1st 0.5528, 2nd 0.5520, 3/10 above.

| Phase-2 diagnostic | Result | Reading |
|---|---|---|
| ensemble of every trained model (6 GNN variants + transformer + trees) | best 0.5278 (GNNs 0.8 + transformer 0.2); error-signal correlation GNN/transformer/trees 0.93-0.96 | all families learned the same thing; ensembling tops out at about +0.004 |
| learning curve: GNN on half the training graphs (273/fold, 60 epochs) | 0.4161 / 0.5361 / 0.4313 / 0.5584 vs 0.4375 / 0.5548 / 0.4485 / 0.5884 (-0.021/-0.019/-0.017/-0.030); fitted-graph score rises (0.5957 vs 0.5665) | **graph-count starved**: more distinct graphs is the measured bottleneck |

Board noise (graph bootstrap of GNN OOF): sd of pooled MCC 0.0127 for 294 graphs; 0.025 if the public part is 30 %
(88 graphs), 0.016 for a 70 % private part. The board's top 12 span 0.009, far inside it.

Board calibration (probe): board 0.5075 vs local 0.4999 (dev masks) / 0.4935 (canonical): board ~ local + 0.008 to 0.014.
v2 board estimate: 0.5241 + 0.008..0.014 = about 0.532 to 0.538 (an extrapolation from one scored version).


| Phase-2/3 experiment (canonical masks, same 4 folds) | Result | Reading |
|---|---|---|
| test masks vs validation | description: "Every test link kept its direction with a chance of one in ten, independently of all the others" | canonical Bernoulli 0.1 masks match the test; nothing to fix |
| pair-conditioned network (exp_nbf, labeling trick: every paper re-embedded as seen from each source paper), d32 L4 20 ep, 4 folds | 0.4400 / 0.5547 / 0.4451 / 0.5894, pooled 0.5251 (GNN r2 seed 0 0.4375 / 0.5548 / 0.4485 / 0.5884); 2-5 min per fold | a different family that lands on the same number; corr with GNN 0.987 |
| NESTED blend (weights on 3 folds, scored on the 4th): GNN r2 3 seeds, nbf, transformer, aug60 | per fold 0.4404 / 0.5604 / 0.4488 / 0.5934, **pooled 0.5286** (v2 0.5241: +0.0045; per fold +0.003 / +0.006 / +0.000 / +0.005); weights mostly aug60 0.4-0.7 + nbf 0.2-0.3 | the best honest local number so far; board estimate with v2's offset about 0.546, still 0.009 short of 0.5550 |

Runtime note (2026-10-13 ~00:40 UTC): with the other session holding 3.2 of 4 GB of GPU memory, torch's AdamW.step()
queries the CUDA stream even for CPU tensors and fails with "CUDA error: out of memory" (cuDevicePrimaryCtxRetain).
All local runs (experiments and any gate run) now use CUDA_VISIBLE_DEVICES= . The platform's CPU runtime is unaffected.

Status block from the user (2026-10-12 ~21:45 UTC): OPEN, submitted yes, 3/10 solvers beat AI, baseline 0.5478, top
0.5528, 51 submissions, **credits 2/6**. The user's v2 = 0.5413 is BELOW the baseline (-0.0065); target 0.5550 (+0.0137).
User's instruction: let the running wave (2-round pair network; extra seeds nbf s1 + aug60 s1) finish, compare against
v2 and the current blend on the same folds, no new redundant runs, no build, no credit, no edits to the best candidates.

### Final wave after the correction (exp_compare.py, same 4 folds, canonical masks; bootstrap = 500 draws of whole held-out graphs)

| Candidate | Per fold | Pooled | vs v2 (diff, sd, P<=0) | vs current blend | Runtime per fold (546 graphs, 4 threads) |
|---|---|---|---|---|---|
| v2 (GNN r2, 3 seeds) | 0.4375 / 0.5584 / 0.4472 / 0.5849 | 0.5241 | - | | 0.7 min per seed |
| augmented GNN (60 ep, crops 0.5), seeds 0+1 averaged ("aug2") | 0.4399 / 0.5608 / 0.4506 / 0.5946 | **0.5294** | +0.0055, 0.0017, 0.000 (agree 96.2 %) | +0.0007, 0.0013, 0.30 | 2.4-3.3 min per seed, peak 1.05 GB |
| current nested blend (gnn3, nbf, gt, aug) | 0.4404 / 0.5604 / 0.4488 / 0.5934 | 0.5286 | +0.0048, 0.0017, 0.002 | - | sum of members |
| + seeds (gnn3, nbf2, gt, aug2), nested | 0.4412 / 0.5594 / 0.4492 / 0.5913 | 0.5279 | +0.0039, 0.0015, 0.000 | -0.0008, 0.0012, 0.75 | |
| nbf -> nbfr2, nested | 0.4434 / 0.5611 / 0.4508 / 0.5917 | 0.5292 | +0.0053, 0.0016, 0.000 | +0.0006, 0.0010, 0.28 | |
| seeds + nbfr2, nested | 0.4402 / 0.5606 / 0.4497 / 0.5950 | 0.5294 | +0.0055, 0.0015, 0.000 | +0.0007, 0.0012, 0.30 | |
| aug2 + nbf2, nested | 0.4411 / 0.5585 / 0.4466 / 0.5947 | 0.5285 | +0.0046, 0.0015, 0.000 | -0.0002, 0.0012, 0.57 | |
| pair network 2 rounds alone (nbfr2) | 0.4361 / 0.5552 / 0.4449 / 0.5900 | 0.5246 | | | 3.7-5.9 min, peak 2.0 GB |
| pair network seeds 0+1 (nbf2) | 0.4403 / 0.5529 / 0.4457 / 0.5887 | 0.5246 | | | 1.8-3.0 min, peak 1.46 GB |

Reading: every candidate lands at 0.528-0.5295, +0.004 to +0.0055 over v2 (convincing locally, all folds up), and all
are indistinguishable from each other (differences <= 0.001, sd 0.0012). The 2-round pair network: no gain, CLOSED. A
second pair-network seed: no gain. The augmented GNN gains from its second seed (+0.003) and is the simplest best: one
family, no blend weights. Board estimate (paired, anchored on v2's 0.5413): about 0.547, sd of the difference 0.003 (all
294 test graphs) to 0.005 (a 30 % public part). P(>= baseline 0.5478) about 35-40 %; P(>= 0.5550) about 5 % or less.
