# Recipes: the default first version per task type

Version 1 is assembled from one of these, not designed. Pick the recipe by the task type; if the challenge description names its own method ("Expected Approach", "…provides a compact supervised formulation"), that method wins and the recipe supplies the settings.

Each recipe says whether it has actually been run in a challenge. **Untested recipes are reasonable defaults, not measured ones** — record the result here after the first use.

## Contents
1. Settings shared by all recipes
2. Image → per-position labels or a sequence, from scratch (run)
3. Text spans / sequence labelling from scratch, few examples of the target cases (run locally)
4. Object detection with pretrained weights allowed (written, never run)
5. Tabular (untested)
6. Text: choose one candidate / retrieval, from scratch (untested)
7. Image grouping into a known structure, from scratch (untested)
8. Grid or multi-part scores (untested)
9. Choose one of K candidates across two kinds of tensor, from scratch, about a thousand pairs (run)
10. Audio panels with shuffled windows plus per-row candidate cards, output = a program of operations (run)
11. Photograph → every object with measured lines and a scale, from scratch on CPU (run; board far below local, unexplained)
12. Two texts → the span of one that refers to the other, from scratch on CPU (run)
13. Small context raster → class probabilities per held-out district (probe built, never scored)
14. No recipe yet

## 1. Settings shared by all recipes

- **Version 1 (probe) size:** one model, one seed, the recipe's default setting, a full run of at most 2 minutes; it has to beat chance, not the baseline (`fast-v1.md`). No settings grid before the probe is uploaded. Time the first pass (the template prints it) and set the number of passes from that.
- **Version 2 and later:** the strongest credible plan that fits the runtime, sized so two full runs fit, at most about 20 minutes per run (our size). Measured examples below were sized smaller; they are evidence of what a small plan scored, not a ceiling.
- **Determinism:** the template's `make_deterministic`; fixed number of passes; see `determinism.md`.
- **Hold-out:** one hold-out that imitates the test split. If the data has the held-out factor as a column (station, operator, document), hold out whole values of it. If not, cluster simple input statistics into about 12 large groups and hold out whole groups; many small groups overstate the score (measured: 60 small groups said 0.339, the leaderboard said 0.2915; 12 large groups said 0.2907).
- **Neural defaults (PyTorch):** AdamW, learning rate 2e-3, weight decay 0.05, one-cycle schedule with 10% warm-up, batch 64, mixed precision, the whole dataset as tensors on the GPU when it fits.
- **Decision:** when the metric adds up per item, the most probable label per item is the best decision. Check that the model's confidence is accurate before trying any decision trick.
- **Runtime stack:** `target-env.md` lists what the platform image holds (scikit-learn, torch, the gradient-boosting libraries, the Hugging Face stack). It has no pretrained weights and no network at run time, so every recipe below is from scratch unless the challenge text says a named model is provided.

## 2. Image → per-position labels or a sequence, from scratch

**Status: run.** Baseline-Direction Transcript Reconstruction — leaderboard 0.2915 with three models (baseline 0.3000, top 0.3155).

- **Model:** residual CNN, widths 32-64-128-192-256, four stride-2 stages so the width shrinks to one column per output position; average over height; bidirectional GRU (128) across positions; one linear layer to the classes.
- **Training:** 40 epochs. Measured on 5,900 images of 96×256: 8 minutes per model on an RTX 3060; 20 epochs underfit (0.19 vs 0.21 on one fold), 80 memorise (0.18, training score 0.61).
- **Augmentation:** only transforms whose effect on the label is exact (here: left-right mirror reverses positions and swaps up/down; upside-down swaps up/down), brightness / contrast / gamma / noise, and a random vertical shift. The vertical shift mattered: without it the hold-out fell from 0.2907 to 0.2666 and the training score rose from 0.39 to 0.55.
- **Prediction:** average over the same exact symmetries.
- **Later versions:** more models gave about +0.003 for several times the run time. Slope-style output heads, extra consistency losses, input normalisation per image, absolute-position inputs, averaging over vertical shifts, a sequence prior and a decision bias were all measured and did not help.

## 3. Text spans / sequence labelling from scratch, few examples of the target cases

**Status: run locally, never submitted.** Unseen-Operator Negation — estimated 0.73 on 64 support sentences (baseline 0.5412; other solvers 0.60–0.71).

- **First, a counted rule baseline (minutes):** word lists and linking patterns counted from the labels; score them without any training. With tens of labelled examples of the target constructions, this beats a network (cue sets right in 53 of 64 sentences against 45).
- **Model for what has enough data:** character-aware BiLSTM (word embedding 100, character CNN 64, two layers of 200) for the parts with thousands of examples; rules decide the rare items.
- **Decode under the invariants the labels obey** (for example, a span is contiguous and contains its cue).
- **Adaptation:** train on the main set, then a few passes with the small target-like set heavily repeated. Rebuild every counted list inside each validation fold.
- **Run time:** 6.4 minutes with three seeds on an RTX 3060.

## 4. Object detection, pretrained weights allowed

**Status: written, never run.** (Wildlife camera-trap bursts.)

- Fine-tune a COCO-pretrained YOLO (small model, image size above the native width when objects are tiny), empty images kept as negatives, hold out whole stations.
- Keep low-confidence boxes for the ranking metric; where a second score term counts boxes above a fixed confidence, remap confidences per class so the best count threshold lands on that fixed value (ranking is unchanged).
- Check first that the detection library is in the Kaggle image; installing it at run time may conflict with the platform rule.

## 5. Tabular

**Status: untested.**

- Gradient boosting (LightGBM: `deterministic=True`, `force_row_wise=True`, fixed `seed` and `num_threads`), 500–1000 trees at learning rate 0.05, default depth.
- Hold out whole values of the entity the test set is new in. Any encoding computed from the target is fitted inside the fold.
- Later versions: a second boosted model with different seeds/depth averaged; feature engineering suggested by the errors.

## 6. Text: choose one candidate / retrieval, from scratch

**Status: untested.**

- **Version 1 without a network:** for each (query, candidate) pair, features from the text around the highlighted span — TF-IDF cosine on character and word n-grams of the left context, the right context and the whole sentence separately, overlap counts, relative position of the span — into logistic regression or a boosted ranker; choose the top candidate per query. Trains in seconds on CPU.
- Hold out whole source documents.
- Later versions: a small siamese encoder (the BiLSTM of recipe 3) trained on matching / non-matching pairs, averaged with the feature model.

## 7. Image grouping into a known structure, from scratch

**Status: untested.**

- Train a small CNN embedding with a pair loss on the training groups (same / different at each level of the hierarchy), on crops augmented with flips and small shifts.
- At test time score all pairs, then **search the valid structures exhaustively** when the description fixes the structure (for example 8 images as 2 families × 2 genera × 2 images is only 315 hierarchies) and take the best-scoring one; this guarantees a valid, nested answer.
- Hold out whole classes that never appear in the fitting part, since the test set uses unseen classes.

## 8. Grid or multi-part scores

**Status: untested.**

- One backbone, features pooled to the grid, one classifier per cell.
- Treat each term of the score separately: a class-balanced recall term wants the label that maximises probability divided by class frequency; a ranking term wants cells ordered by expected value computed from the same probabilities.

## 9. Choose one of K candidates across two kinds of tensor, from scratch, about a thousand pairs

**Status: run.** Movie-Disjoint Editorial Soundtrack Shortlists — leaderboard 0.3699 (baseline 0.3617, chance 0.3208, top 0.4772). Folder ready 10 minutes after the data was on disk; a full run takes 20 seconds on the CPU.

- **Inputs:** fixed descriptors of each tensor, nothing learned. Image side: colour, brightness histogram, saturation, black bars, gradient strength at several resolutions, change between frames, and noise and fine detail (Laplacian and gradient quantiles, spatial spectrum). Spectrogram side: level and variation per band, envelope statistics, rhythm spectrum, spectral centre / spread / entropy, coarse blocks.
- **Model:** standardise, keep 16–24 principal components per side (not whitened), bilinear score `v' W a` with no bias and no slot input. Loss: softmax over the K slots of each row, plus the mirrored term (for each candidate, which listing row is its own), plus L2 0.1. Convex; L-BFGS in float64; identical output on every run.
- **Hold-out:** k-means clusters of the query-side descriptors only, whole clusters held out, 4 folds, every row scored out of fold. Mask candidates whose own row is held out while fitting.
- **Measured:** whitening the components −0.03; 8 or 32 components slightly lower; an all-pairs term helps only on random folds; squared components, bagging, averaging settings and raw thumbnails do not help; a two-tower CNN memorises (0.34 held out).
- **Later versions:** drop one descriptor group at a time, then measure the group that carries the score more precisely (here picture grain: about +0.01 locally, never scored).

## 10. Audio panels with shuffled windows plus per-row candidate cards, output = a program of operations

**Status: run.** Spoken Corrections (A10G, baseline 0.5100): v1 0.4957 (10 min after the data), v2 0.5178, v3 0.5223 (best; final rank 15, top 0.5597, 12th 0.5285), v4 0.5101. About 1,500 training rows, four 48 x 384 panels per row.

- **Data:** each turn = 24 windows of 16 frames, shuffled per turn (independent between the turns of a row: chains from two turns of one row share adjacent window pairs only at chance level) and some reversed; 20 anonymous cards per row (lexical, subword and pronunciation unit bags, 42 phone-like units); one system-context card per turn.
- **v3 model (best), about 75 s per model on a 3060:** window CNN (kernel 3) with one self-attention layer over the 24 windows (single seeds 0.486 against 0.474-0.485 without), mean/max pooling, plus context embedding and turn index, a 2-layer transformer across the turns; card = unit-bag embeddings + slot class; card-to-window cross-attention; pair MLP whose input also holds each card's evidence averaged over the turns (+0.006); multi-label BCE heads (operations; SET per card); 40 passes, AdamW 2e-3, one-cycle; average of six models (+0.02 over one); thresholds 0.30 operations / 0.45 SET.
- **Free gains at decode time:** order the operations of a block by pairwise majority of training precedence counts (oracle 0.929 to 0.970; +0.004 on predictions); at most one value per slot except dates and a slot gate (+0.002 to +0.008 depending on the models).
- **Window order can be recovered:** greedy edge matching of the 48 window ends gave complete chains in every turn (nearest end 0.5x a normal frame step; joint cost about 0.8x). A timeline CNN alone gave nothing; a differentiable monotone phone-sequence alignment of each card against the timeline gave about +0.005 at ensemble level, within the noise.
- **Did not help:** window dropout, input noise, card-frequency features, phone MaxSim, slot-presence head with a within-slot listwise loss, per-turn input standardisation, a second attention layer, 60 passes.
- **Where the errors were:** SET operations (half of all operations) at F1 0.37, SET_DATE 0.26; ASK 0.68; ACCEPT/REJECT/DECLINE/AFFIRM 0.82-0.95.
- **Noise:** identical configurations differ by about 0.01 on a 35-speaker hold-out at two seeds. Use grouped K-fold.
- **The field moved late:** the 5th place went from 0.5256 to 0.5469 and the top from 0.5549 to 0.5597 in the last hours, while nothing tried here gained after v3.

## 11. Photograph → every object with measured lines and a scale, from scratch on CPU

**Status: run. Board 9.43 against a local estimate of about 54 (baseline 18.98, top 60.40). The gap was never explained, so nothing below is a validated recipe.** Tray Elytra Metrology, 2026-10-05; score = detection F1, end-point accuracy, sizes in mm and scale, the last two chance-corrected per tray.

- **The probe this task needed (not sent):** the scale reading plus one small detector whose own length and width vectors give the lines; no end-point network, no correction of any label oddity. It would have shown the gap in the first half hour.
- **What version 1 was:** a classical scale-bar reader (checkerboard X-corners, white-square size), a small U-shaped detector on the quarter-resolution image (centre heat map plus length vector), an end-point network on patches cut in each object's own frame (soft-argmax, applied twice), a correction for shrunken label frames, and a snap of the scale to the training median.
- **Local numbers (four held-out pseudo-species folds, never confirmed by the board):** 40.3, 61.3, 53.8, 60.6. End-point network at 96 px beat 64 px by about 7 points; a cross-entropy on the heat maps halved the passes needed; three refinement passes, a peak-windowed soft-argmax and five-view averaging gave nothing consistent.
- **Cost:** 21 minutes per full run on 4 CPU threads (reading 420 trays 2, detector 6.5, end-point network 11, test 1.5), so the gate took 49 minutes.
- **Data findings to re-check before trusting labels on a similar set:** on 72 of 420 training trays the curators' lines and scale were in a frame smaller than the photograph by exactly 30/n about the image origin (the lines did not lie on the objects); a tray-level correction that gained 5 points on one fold lost 2 to 3 on the next.
- **After the score:** a replay of failure modes on held-out predictions showed that only the loss of a whole component (all scale and size credit, or all objects) gives a score near 9; which one, and why, was not found before the challenge closed.

## 12. Two texts → the span of one that refers to the other, from scratch on CPU, about 1,600 pairs

**Status: run.** The Enclosure Mention (CPU, 90-minute limit, baseline 0.7331): v1 0.6682 (probe, ten minutes of work), v2 0.7941, v3 0.8072, v4 0.8065, v5 0.835 (final; the leader was at 0.8519 when last seen). Local 5-fold over all labelled rows read about 0.06 under the board from v2 on.

- **Formulation:** choose among candidates. Almost every pair had one short answer (a document noun, "Allegato II", a quoted title). Candidates: every word, multi-word expressions annotated at least twice, quoted strings; split words where a lower-case letter is glued to a capital.
- **Probe:** hashed context features (the word, three neighbours each side, an eight-word window, position) into liblinear logistic regression, top candidate per pair: 0.654 on the supplied validation file, 20 seconds.
- **What gained (local 5-fold):** quoted candidates and overlap between the candidate's surroundings and the other text, five-letter stems weighted by rarity (+0.058); the other text's most frequent short words crossed with the candidate, a language signal with no outside word list (+0.015); returning the one or two spans with the highest expected character F1 (+0.007); giving different spans to two test rows that share the same cover text (+0.016 locally, +0.013 on the board). **That last step got the submission rejected in the human review after scoring** (the user's account, 2026-10-07): it makes one test row depend on another, which that challenge's markdown had ruled out. See `compliance-firewall.md`. Without it the board score would have been about 0.82 (0.835 minus the step's +0.013; a subtraction, not a measured run).
- **Second model:** word plus hashed character n-gram vectors, BiGRU over the cover, BiGRU over the first 250 words of the other text, attention, a second BiGRU, start and end outputs; 25 passes, two seeds, batches of similar length; blended at weight 0.2 over the classifier's candidates: +0.019 locally, +0.028 on the board. 48 seconds per pass on four CPU threads; 40 minutes per full run.
- **Did not gain:** a separate span-extent model (the multi-word answers stayed near 0.5: the annotation of "copia della lettera" against "copia" is not consistent); document-type model, sibling contrast, per-pair softmax ranker, context-word ridge, LightGBM stack, punctuation gaps (each 0.005 or less); a convolutional reader (0.707 alone, blend +0.005); PPMI-SVD word vectors; more reader seeds.
- **Where the errors stayed:** covers that send several documents (rows sharing a cover 0.56 against 0.79), where an Italian description has to be matched to a German, Latin or French text without pretrained models.

## 13. Small context raster → class probabilities, scored per class and per held-out district (CPU)

**Status: probe built, never scored.** Building Use Classification from Context Rasters (baseline 0.4748): the platform's run of version 1 stopped with AssertionError and the challenge was paused there by the user.

- **Probe:** CNN 24-48-96-128 from scratch on 6,664 rasters of 64x64, the eight rotations and mirrors in training and prediction, 12 passes: 8.6 minutes on four CPU threads. Local, one pair of held-out districts: 0.557 against 0.387 for uniform probabilities.
- **The metric wanted two different outputs:** balanced log loss per class and a district mix close to the truth. Dividing the probabilities by the training class frequencies to the power 0.5 to 0.75 gave about +0.05 over raw probabilities (28 pairs of held-out districts).
- **Noise:** the same model scored 0.46 to 0.60 depending on which two districts were held out, and the test set was two districts.
- **The failure:** the script ended with three asserts on its own output (header and row count, finite and inside (0, 1), rows summing to 1); none fired locally in three runs and the bot does not say which fired. A failed run did not use a credit. Build the output so that it cannot fail (explicit columns, clip, renormalise) instead of asserting on it.

## 14. No recipe yet

Time-series or track association, generation tasks. For these, version 1 is the description's own suggested method at the smallest size that runs, and a recipe is written here afterwards.

## CPU ordinal sensor streams to canonical exposure events — FlockSight (scored)

- Status: officially scored above baseline, but failed top-10 admission; user ended the run at reported20th on2026-10-07. Latest supplied score54.0876, baseline53.5336; final private score not supplied.
- Data:1932 labelled16-bin rows,16 locally permuted candidates;983 public queries. Exactly8 informed roles; this is not8 active members per bin. Severe dropout absent from training and a majority in test; participant keys absent.
- Calibration:76 within-row numeric/interpolation/missingness/context/trajectory features, CPU HGB60rounds/15leaves, fixed smoother and threshold0.3; official51.0569. Full checks46.29/54.24s.
- First improvement: HGB120rounds/63leaves plus a from-scratch20-pass Linear48/BiGRU24 membership reader,70/30 blend and threshold0.35; official53.1065, platformVersion3/S50. Full checks160.08/153.27s.
- Best: same tree plus natural and natural+consistent-severe-mask readers,20passes each,50/25/25 blend; official54.0876, platformVersion4/S52. Full checks237.77/220.91s. Recompute repeated gaze and visible-candidate summaries after masking; labels stay uncorrupted. Fit normalization in-fold/on training only.
- Local validation was a row/profile proxy, not participant-held-out proof. Paired gains2.6254->2.0496 and1.0021->0.9811 carried over; repeated387-row confirmation and naive row SE did not account for correlated windows.
- ExtraTrees, larger/longer readers, extra geometry/motion, crowd attention and several decoder variants did not support a better late submission within its budget. Do not infer that all alternatives are ineffective or that tiny gains beat a cutoff.
- Platform incidents: initial unrelated bill-task grader/prompt, then generic determinism check errors without user execution logs; later fixed-plan source scored. Do not assert that source restyling caused the platform recovery. See [postmortem](postmortems/flocksight_2026-10-07.md).

## 15. Dose-response curves + compound structure → ordered time-to-event class, macro-F1, CPU, from scratch, about 850 compounds — Membrane Confirmation Delay (scored)

- Status: CLOSED. FINAL private 0.5274, 2nd (winner 0.5414; public 9th at 0.4259): local scaffold-grouped CV (0.52) matched the private score and the public board did not. Public board (baseline 0.4048): 0.3906, 0.3852, 0.3852, 0.4224 (version 4), 0.4254 (version 9), 0.4259 (version 10, best); 8th at the grace hour, tenth 0.4210, top 0.4975. Local scaffold-grouped CV 0.50 to 0.52 for all versions: the board ran about 0.10 lower.
- Data: every numeric column rank-encoded, with a reversible per-column calibration file (swap the knots, linear tails); 12 curves of 15 points per compound; the anchor sits exactly on the compound's own grid; the test is scaffold-held-out and one unseen family holds 36% of it.
- Versions 4 and 9: physical values by inverting the calibration; about 24 anchor-aligned inputs (the anchor cell's own readings raw, the other cell smoothed 1-2-1 across concentrations because it was not selected on, changes between times, counts of higher-concentration points past thresholds, position on the grid); no absolute concentrations; MLP ensembles with class weights (n_0 / n_k)^0.5, five seeds per type; output heads wide-and-deep, hierarchical gate, and a mixture cure model (gate x log-normal delay observed in the 24/32/40 h windows); 20 networks, one minute on one CPU thread.
- Measured (three fold seeds, spread +-0.015): fewer inputs beat more (26 > 46 > 120); smoothing only the unselected cell +0.01; the other cell was the largest block (-0.03 without); structure descriptors or fingerprints concatenated hurt, a separate descriptor network at 25 to 30% weight added about +0.005; LightGBM check 0.47 (the MLP was not the bottleneck). Nothing gained: ordinal or monotone-hazard heads, label smoothing, input dropout, bagging, robust scaling, TabM-mini, additive, periodic-embedding, CNN-over-curves, RBF and prototype nets, logit-adjusted loss, per-class offsets.
- Never found: what the solvers at 0.44 to 0.50 did differently. See [postmortem](postmortems/membrane_confirmation_delay_2026-10-09.md).
- The winner's method (code shared by the user after closing; private 0.5414, public 0.5398): physical values through the calibration file; three feature views kept apart, full (56: FLOR/GLO at 0/8/16 h at the anchor and two concentrations either side, rises, gap to 25 over the latest rise, the other cell at the nearest concentration, log anchor concentration and index, SMILES token counts), compact (34: 11 anchor values plus pool-adjacent-violators monotone fits of each 8 h and 16 h dose-response and the series position where the fit reaches 10/25 and -25/-50), kinetics (18: the compound's own well-to-well noise over its six lowest concentrations, signal-to-noise of the anchor reading, FLOR-on-GLO slope, the other cell's first eligible concentration); per view one hidden layer of 64 tanh units in NumPy, full-batch Adam, 600 steps for the full view and only 40 steps for the compact view, softmax and cumulative-logit ordinal heads; three 5-fold scaffold-grouped partitions x 4 seeds x 8 members = 480 fold models averaged on the test; per-class probability weights tuned on out-of-fold macro-F1 and averaged over ten bootstrap resamples. Several of these were recorded above as not gaining inside one concatenated network; the 0.014 private gap on 540 compounds does not show which part mattered.

## 16. One microscope image of a particle -> main type + attribute record, test = unseen sources (volcanoes), fine-tuning allowed, A10G — Volcanic Ash Componentry (scored, open when written)

- Status: board (baseline 0.6573) v1 0.6401, v2 0.6616, v3 0.6779 (1st on the public board at lockdown, top ten 0.6622 to 0.6779), v4 handed over, score not known; private result not known. 6,515 train / 2,497 test images of 256 px, 7 training volcanoes (43% of the rows one volcano on another microscope), 3 unseen test volcanoes. Score: class-balanced over 4 main types, 0 for a wrong main type or an impossible record, else 0.5 + 0.5 x attribute credit.
- Validation: two whole volcanoes held out (2,172 rows, all four types). Board minus hold-out +0.07 to +0.11; a hold-out gain reached the board at about 40% (v1 -> v2: +0.058 local, +0.0215 board). Seed spread 0.013 on the hold-out and 0.016 between two seeds on the board.
- v1 probe: ConvNeXt V2 nano (ImageNet-22k) fine-tuned 4 passes, six linear heads, loss masked where an attribute is not recorded, inverse-frequency weights on the main type, record assembled within the allowed combinations with expected-credit decoding of the ordinal fields: hold-out 0.534. Ceilings on it: perfect main type 0.853, perfect attributes 0.628, so the main type is the bottleneck.
- v2: `facebook/dinov2-base` at 224 px, class token + mean patch token, fully fine-tuned 3 passes, AdamW 3e-5 with layer decay 0.75, heads 1e-3, flips and quarter turns only: hold-out 0.5925. The position embedding is resampled once to 16 x 16 at build time, so no bicubic interpolation runs under `use_deterministic_algorithms`. 14.8 min per run on the 3060 in 32-bit.
- Less fine-tuning is better: 2 to 4 passes 0.56 to 0.59, 8 passes 0.54, lr 1e-4 0.54.
- Measured, no gain: frozen linear probes of DINOv2-S/B and CLIP-B/16 (main-type recall 0.58 to 0.61, below the fine-tuned nano's 0.63, so a frozen-feature wave does not rank backbones here); heads first then fine-tuning with weight mix-back to the pretrained model (0.55 to 0.56 at every mix); DINOv2 with registers (0.561, 0.542); free rotation, zoom and shift (0.583), plus brightness and gamma (0.579); volcano x class balanced sampling (0.571); averaging with the ConvNeXt (0.581). Three seeds averaged 0.5945 against a member mean of 0.590.
- Attributes with the main type right: sub-type 0.66, colour 0.66, crystallinity 0.84, alteration 0.83 credit, shape 0.35; a perfect sub-type or shape is worth about +0.03 each, but the errors follow the held-out volcanoes' own label mix.
- Not tried: EMA or trajectory averaging, SAM, label smoothing, light hue and saturation jitter, a larger backbone (a ViT-L cannot be fine-tuned in 32-bit on 6 GB). See [postmortem](postmortems/volcanic_ash_componentry_2026-10-09.md).

## 17. Six hourly two-pollutant windows per station -> two 3-class labels, balanced accuracy, test = unseen countries, CPU, from scratch — Station Siting (scored)

- Status: CLOSED. FINAL private 2nd at 0.5203 (winner 0.5219); public 20th at 0.4761 (baseline 0.4984, public top 0.5961). Public board: v1 0.3543, v2 0.4628, v3 0.4761 (best), v4 0.4417. Local country-fold mean 0.395, 0.451, 0.470, 0.495.
- Data: 1,457 training stations (1,181 background, 161 industrial, 115 traffic), five supplied folds of whole countries; one fold holds 100 of the industrial and 66 of the traffic stations. Each window is divided by its own mean; hour stamps sit at different clock offsets per country (peak hours differ by one to two hours).
- v3 (the version that won 2nd): per window about 405 shape measurements (mean weekday and weekend day, weekday means, quantiles, autocorrelation; variance shares at weekly and daily spectrum lines and bands, NO2-ozone phase; hour-to-hour variability, spikes, residual from the window's mean week; weekend effect in 4-hour blocks; titration slope = ozone change per NO2 change); each standardised against training windows of the same and neighbouring start months; one network 405-64 per window, the six encodings averaged, a linear layer per axis; 20 passes, AdamW 2e-3, dropout 0.5, windows kept with probability 0.7, class weight (1/freq)^1.5, 10 seeds; 0.6 min on 4 CPU threads.
- Measured: window-averaged features into an MLP 0.36 to 0.38, the same features per window 0.45; 60 passes worse than 20; class weight power 1 -> 1.5 +0.05 pooled; month-wise scaling +0.01; everything else (trees, linear, CNNs, random kernels, contrastive features, robust objectives, feature filters, blends) 0.37 to 0.48. In-country control 0.546.
- Not to repeat: choosing which model serves which axis on the scoring folds (v4: local 0.495, board 0.4417). See [postmortem](postmortems/station_siting_2026-10-11.md).
- The winner's method (script shared by the user after closing; private 0.5219 against our 0.5203): one vector per station, not per window. Two feature banks computed per window and then aggregated over the six windows as mean, spread, summer-window mean (April to September) and winter-window mean: bank 1 = quantiles, skew, kurtosis, hour-to-hour jumps, autocorrelation at 1 to 168 h, weekend ratios, daily harmonics, 24-hour weekday and weekend profiles, NO2-ozone correlations at several lags, and the profiles re-anchored at the station's own ozone maximum so the clock offset drops out; bank 2 = residual from the weekly cycle (spread, kurtosis, spikes), variance shares, spectrum power at 168/24/12/8/6 h, daily peak sharpness, 3-hour weekly blocks, weekday-to-weekday shape stability, and rush-hour ratios taken at hours relative to the ozone peak. Mean imputation, then a quantile transform to a normal distribution, fitted on training stations. A different model family per axis: station type = two strongly regularised scikit-learn logistic regressions (C 0.003 on bank 2, C 0.0003 on both banks, balanced class weights, log-probabilities averaged); area type = two torch MLPs (256 GELU units, input dropout 0.2, dropout 0.5 to 0.6, 30 and 60 passes, 5 seeds each, plain balanced class weights). About 3 minutes including its own fold estimate. Things recorded above as weaker in our form and used by the winner in another: window-averaged features (ours went into an unregularised MLP), a linear model (ours 0.42 before short training and stronger weights were found), summer/winter splits, and the clock offset (ours aligned to a training template and was removed on the rule about aligning evaluation stations; anchoring each station on its own ozone peak needs no template). The 0.0016 private gap on 412 stations does not show which part mattered.
- Form of the winner's file, as accepted by the platform: about 290 lines, a short header comment, scikit-learn LogisticRegression in a From Scratch challenge, plain seeds and a fixed thread count without the full deterministic-backend block, default paths `dataset/public` and `working/submission.csv` with the two positional arguments, and a complete logistic-regression submission written before the main models.
