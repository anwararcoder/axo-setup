# Lessons from past challenges

Each entry: what happened, what it cost, and the habit that prevents it.

Promotion rule (the challenge-engine pack, `postmortem.md`): add an entry only when the lesson is clearly reusable, supported by the results, and at most 3 per review; challenge-specific facts belong in the recipe entry, not here; merge duplicates instead of appending. Older entries below keep their challenge detail as evidence.

## Process

**The first version was built for score: 2.5 hours, local 54, board 9.43, and no time left to find out why.**
(Tray Elytra Metrology, 2026-10-05.) The user asked for more than 50 on the first submission, so the probe was skipped. A detector, an end-point network, a scale reader and a correction for an oddity found in the labels all went into version 1 together. Four held-out folds said 40, 61, 54 and 61; the board said 9.43 against a baseline of 18.98. "About 50, comfortably above the baseline" had been told to the user as if it were a board estimate. A replay of failure modes on held-out trays showed only that a whole component must have earned nothing; the challenge closed before the cause was found, with five credits unused. → Version 1 is always the fast probe that only has to beat chance (`fast-v1.md`), whatever first score is asked for; say why. Until its board score exists, call every local number "local, not yet checked against the board". When the board is far from local, stop and find the cause before building more (`climb.md`). A part that rests on a discovery about the data goes in alone, after calibration, in its own submission.

**Never asked for the closing time → deadline missed.**
Work started without knowing when the challenge closed. A working solution scoring above the baseline existed in validation, but no submission was made in time. → Step 2 of the pipeline: get the closing time, solver count and credits before any modelling, and fix a "ship v1" point.

**Dataset link pointed at a different challenge.**
A link pasted alongside one challenge description downloaded another challenge's data (text CSVs instead of images). → Compare file names and columns to the "Data" section of the markdown immediately after download.

**Dataset link expired.**
Links refresh hourly; a link from the day before returned an XML "Request has expired" error saved as `dataset.zip`. → Check the HTTP status and that the file is a real zip; ask for a fresh link straight away.

**Started a 2.5 GB PyTorch install while a ready environment existed.**
A conda environment with CUDA PyTorch, pandas and numpy was already on the machine. → List conda environments and probe each for the needed libraries before installing anything.

**A self-imposed budget was presented as if it were the rule.**
The document allowed 90 minutes; a 30-minute working target was used without saying it was ours. → Quote limits verbatim; label any tighter target as "our margin".

**A background run was killed by its own timeout before writing results.**
One-hour timeout on a run that needed longer; the final training never finished and the only output file was the empty fallback. → Estimate duration from a small timed run first; set the timeout well above it; keep development validation and the final fit as separate runs.

**Determinism was assumed, not tested, until the very end.**
The script had fixed seeds and `cudnn.deterministic = True` from the start, yet two full runs produced different submissions (2,412 vs 2,413 frames): GPU embedding gradients accumulate in a non-deterministic order unless `torch.use_deterministic_algorithms(True)` is set. The challenge required a deterministic plan, so every earlier version would have broken a rule, and all development scores were measured on a non-reproducible script. → `references/determinism.md`: put the full settings in before the first training run and run `scripts/check_determinism.py` on the first working version, then on every version submitted.

**Slow experiment loop.**
The first training loop was Python-bound (GPU at ~30%): 679 s for three base models. Pre-building arrays and batching by length cut it to 195 s. → Check GPU utilisation on the first run; fix the data path before running more experiments.

**The first version reached the user after almost three hours; a working model existed after 40 minutes.**
(Transcript Reconstruction.) Version 1 was sized at three models and 24 minutes per run, so the two-run reproducibility proof alone took 49 minutes, and the extra models were later measured at about +0.003. By then four other solvers were above the baseline. → Version 1 is one model, one seed, a full run of at most 8 minutes (`fast-v1.md`); more models are a later version.

**A full run crawled for 71 minutes and was lost.**
The laptop was apparently on battery and throttling the GPU; a run that should take 8 minutes per model had not finished one when the session ended. Nobody looked because nothing reported a pace. → `scripts/preflight.py` before training (mains power, GPU free, other jobs); the template prints the projected time after the first pass; `--expect-minutes` stops a run that passes twice its expected time.

**Eleven ideas in 2 h 46 min after the first score, and the plain checks never ran.**
Analysis-led ideas went first (how the labels were made, a slope-consistency loss, absolute-position inputs). The queue was stopped and restarted six times to put a newer idea in front, discarding about 30 minutes of GPU work, and dropout, weight decay, network size and weight averaging never produced one result before the challenge closed. → `climb.md`: write the whole queue once, cheapest and most likely first; reorder by editing pending lines (`scripts/run_queue.py`), never by killing a run; data-specific ideas come last, one at a time, each with a time box.

**Pruning was added only when the user asked, 2 h 40 min in.**
Until then every losing idea ran all its folds, 20–35 minutes each. → The harness prunes after the first fold from the first experiment (`prune=<reference first-fold score>`).

**No second submission; five credits unused at closing.**
The search was for one big step, so nothing was packaged while the number of solvers above the baseline went from 4 to 8 in about two hours and the challenge closed that evening. → Submit every gain that exceeds the seed-to-seed noise; ask for the status block at each hand-over; at 8/10 stop exploring and package.

**The challenge was picked from the descriptions alone.**
It looked like the easiest of three on paper. Its board, seen only afterwards, showed 11 submissions and nobody above the baseline — the sign of a plateau just under it, which is where version 1 landed (0.2915 against 0.3000). → Ask for every candidate's status block before recommending one.

**The development harness and the experiment queue were written by hand, mid-challenge, several times.**
Fold construction, saved predictions, training-set score, pruning and the queue loop were each added in a separate edit while the clock ran. → They are bundled: `assets/dev_harness.py`, `scripts/run_queue.py`, `scripts/gate.py`.

**Another session was using the same GPU.**
By the end of the challenge a training job from another session was running on the same machine. It did not visibly slow these runs, but nothing would have flagged it if it had. → `scripts/preflight.py` lists other Python jobs and GPU use; say so and size the plan from the measured pace, or ask which session has priority.

**The second version reached the user seven minutes after they asked for it at 9/10, and the challenge closed during their upload.**
(Soundtrack Shortlists.) The best measured candidate was already known when the user wrote "I want the second submission now". One more test was started first, and the notes and the rule file were updated before the hand-over message; about two to three of the seven minutes. Version 2 was never scored. → From 8/10, or whenever the user says "now": package the best candidate already measured, run the gate, send the hand-over message, and only then update notes or run anything else.

**The board itself was noise-limited and nobody had computed it.**
(Soundtrack Shortlists.) 300 test rows and a per-row score spread of 0.36 give one standard error of 0.021; the baseline was 0.041 above chance. Local validation said 0.39–0.40, the board said 0.3699 — inside the noise. → At intake compute (per-row score spread) / sqrt(test rows), and say with every hand-over how likely the version is to land under the baseline. (The board keeps a solver's best version, so a lower later score costs only the credit.)

**Inspecting the archive during planning made version 1 a ten-minute job.**
(Soundtrack Shortlists.) While the plan was being written, the archive's file list, the CSVs and two tensors were read over HTTP range requests into memory, with nothing saved. Row counts, tensor types and the path mismatch between the CSVs and the archive were known before the clock started; the folder was built 10 minutes after the data was on disk. → When a challenge arrives with planning still to do, read the archive's listing and CSV heads first.

**A required folder was searched for at run time, and the platform's run of the script failed.**
(Instruction Stack v1.) The script looked for an encoder checkpoint (and, with no arguments, for the dataset) in a few
places and raised ValueError when the search found nothing. Axo answered "Your solution failed with ValueError"; the credit
and the hand-over time were lost, and nothing local could have shown it because the weights existed on the development
machine only. → `references/target-env.md`: the platform image has libraries, not weights, and no network. A missing optional
resource is a fallback, never an error; run the script once with it absent (rename the folder, empty `HF_HOME`) before the
gate; when a challenge says pretrained weights "available in the environment" may be used, ask which, before building on them.

**The last version, +0.004 locally, scored 0.012 below the previous one.**
(Spoken Corrections.) v3: local 0.5136, board 0.5223 (+0.009). v4 (window re-assembly, phone alignment, slot-aware decoding, five models): local 0.5165, board 0.5101 (-0.006). The same two-seed alignment ensemble had scored 0.5058, 0.5038 and 0.4933 on the 35-speaker hold-out in three runs, so a +0.004 gain was inside the noise, and the decoder thresholds had been tuned on that same hold-out. The board keeps the best version, so it cost one credit, but it was handed over as "a better solution" under a 20-minute deadline. Final standing: 15th with 0.5223; places 4-14 lay between 0.5487 and 0.5285. → At the first multi-seed comparison run the identical configuration twice and write down the spread; a local gain below twice that spread is no evidence. Before a late "improvement" is packaged, check it with grouped K-fold over all speakers; if it stays inside the noise, say at hand-over that it is a lottery ticket, not a better solution. Never assume the board-minus-local offset of earlier versions carries over to a version whose decoder was tuned on the hold-out.

**The gate needs twice the run time, and the user had 20 minutes.**
(Spoken Corrections v4.) The script took 13.3 minutes, so the two mandatory runs took 27.5 minutes; the folder reached the user 16 minutes after their request only because the gate had been started before it. → When the user names a deadline, read the gate's ETA at once and cut models until two full runs fit with margin.

**Two-seed comparisons on one small hold-out ranked ideas worth 0.005.**
(Spoken Corrections.) Eleven ideas were compared on one hold-out of 35 speakers with one or two seeds; the seed-to-seed spread of identical configurations (0.481-0.499 for single models) was as large as the effects being ranked. → From the second experiment on use grouped K-fold with out-of-fold predictions (`cv5.py`-style: one model per fold, all speakers scored once), and tune decision thresholds on the out-of-fold predictions, not on one hold-out.

**A local gain of two standard errors scored nothing; one of three and a half scored +0.028.**
(The Enclosure Mention, 2026-10-06; 387 test rows.) Every version was compared with the best one group by group on the same five folds over all 1,601 labelled rows. v4: +0.0088 (standard error 0.0044) locally, 0.8065 on the board against 0.8072: a credit for nothing, handed over as "probably real but small". v5: +0.0187 (0.0055) locally, 0.835 on the board. The board-minus-local offset was not constant either (+0.020 for the probe, +0.063 for the later versions), so only differences between versions carried over. Same pattern as the Spoken Corrections v4 entry in this file. → Keep one validation protocol for the whole challenge (the user asked for this) and judge a candidate by its paired difference against the best version, not by two separate means. Hand over only at three standard errors or more and at least +0.02 when the test set has a few hundred rows (the board's own paired noise is about the local standard error times the square root of labelled rows over test rows). Below that, say it is a lottery ticket and do not recommend the credit.

Additional confirmed FlockSight evidence (2026-10-07): paired local gains2.6254 and1.0021 transferred as official gains2.0496 and0.9811; absolute natural-row proxies remained higher than the board and participant keys were unavailable. Compare paired changes under one fixed protocol, keep proxy/correlation caveats explicit, and do not assume a universal board offset or treat noise-level follow-ups as supported gains. Challenge-specific details are in the separate postmortem.

**A model cut for speed kept half its gain, and only cross-validating the submission code showed it.**
(The Enclosure Mention v5, CPU, 90-minute limit.) The neural reader was developed on the GPU for 1.5 hours before one pass was timed on the CPU: about 75 minutes per model. Under a 20-minute request from the user, a convolutional version (blend +0.005) and a slimmer recurrent one were tried; the slim one read +0.016 in the development copy but +0.010 when the submission file's own functions were run through the same folds, against +0.019 for the full-size reader, which fitted after all (batches of similar length, 40 minutes per run). The user then said to cut nothing; the full version scored 0.835. → On a CPU challenge, time one pass of any new model on the CPU with the fixed thread count before developing it. Cross-validate the submission file's own functions in their exact configuration before the gate; a cut configuration never inherits the uncut one's evidence. When the user names a deadline, check the standing first: at 3rd of 10 with the top 10 still able to submit after the lockdown, the right answer was "you are safe, the real version comes later", and it was.

## Modelling

**Hold-out groups were built from the inputs and the paired target together, and every setting scored below chance.**
(Soundtrack Shortlists.) Scenes and their paired soundtracks were clustered jointly and whole clusters held out: 0.30 for all 48 settings where chance is 0.32. Holding out such a cluster leaves in training exactly the similar scenes whose soundtrack is of another kind, so the split teaches the wrong pairing. Clusters of scene descriptors alone gave 0.40. → Build hold-out groups from the query-side inputs only, never from anything that involves the label or the paired item.

**Dropping one descriptor group at a time found what the model used in three minutes.**
(Soundtrack Shortlists.) Eight sharpness numbers out of eighty carried the whole scene side (alone 0.40; without them 0.35). Measuring that one property more precisely was the only change that gained (about +0.01); a CNN, squared components, bagging, averaging and six other ideas did not. → On a descriptor model, run the group ablation straight after version 1 and spend the next experiments on the group that carries the score.

**A network from scratch on about a thousand pairs memorised.**
(Soundtrack Shortlists.) Two-tower CNN on raw frames and spectra, 1,200 pairs: 0.34 and 0.31 held out, 0.6 on the rows it was fitted on, against 0.40 for fixed descriptors with a convex matcher that fits in seconds. → Under a few thousand examples, version 1 is fixed descriptors plus a small convex model; a network is a later experiment, pruned after one fold.

**A cheap lexical baseline was measured too late.**
On a few-shot transfer task (64 labelled examples of the target constructions), a neural tagger was built first and reached an estimated 0.58–0.60. A word list and linking rules counted from the same labels took seconds to evaluate, and replacing the network's cue detection with them raised the estimate to 0.70, then 0.73 with two decoding rules read off the labels (baseline 0.5412; other solvers 0.60–0.71). → When the target distribution has only tens of labelled examples, measure a counting/rule baseline before training anything, and let the network do only what it has enough data for.

**The split design is information.**
The organisers removed the target operator families from the training set. That meant the training set's statistics for those words were misleading (they showed only the non-target uses) while the small support set reflected the test distribution. → Work out what was held out and which set mirrors the test set; weight evidence accordingly.

**Label invariants make decoding exact.**
Every labelled scope was one contiguous span containing the cue; every focus a contiguous span inside the scope. Decoding as "best span" enforces this and removes a class of errors. → Check structural invariants over all labels, assert them in the script, and decode under them.

**Find where the metric multiplies errors.**
Items were keyed by the cue set, so one wrong cue token zeroed the entire frame. Cue-set accuracy was therefore the first thing to fix, ahead of scope or focus quality. → Read the metric for all-or-nothing keys and fix that stage first.

**Decouple what the metric decouples.**
(Detection task.) The counting term used a fixed confidence threshold of 0.5 while AP used only the per-class ranking, so a monotone per-class remap of confidences improves the count term at no cost to AP. → Look for parts of the score that depend on different properties of the output and optimise them separately.

**Compare like with like.**
A cross-validated estimate (model sees 75% of the adaptation set, scored on tens of sentences) is not a leaderboard score. → Report it as an estimate with its sample size, and calibrate it against the first real leaderboard score.

**Many small validation groups overstated the score by 0.047.**
With no page id in the data, lines were clustered into 60 small style groups; held-out groups still shared pages with training and the hold-out said 0.339 where the leaderboard said 0.2915. Twelve large groups gave 0.2907. → When the held-out factor is not in the data, use a few large groups from the start, and re-calibrate against the first leaderboard score.

**The training-set score showed the limiter in one line.**
With the standard recipe the model scored 0.39 on its own training lines and 0.29 held out; without the vertical-shift augmentation 0.55 and 0.27; with twice the training 0.61 and 0.18. Memorising was the problem, and the labels were only partly predictable. → The harness prints the training-set score next to every held-out score; read it before choosing what to try.

**An accurate confidence means decision tricks cannot help.**
The model's confidence matched its accuracy (93% right when it said 90%), and a decision bias, a sequence prior and averaging over shifted inputs each made no difference or hurt. → Check confidence against accuracy on saved predictions (one minute) before spending runs on decision rules.

**Averaging similar models is worth little.**
Two models averaged gave about +0.003. → A scale-up step for the end, never part of version 1.

**Working out how the labels were made did not convert into score.**
The labels turned out to follow a baseline through six evenly spaced points whose heights behave like whole pixels of a smaller original image. It explained the plateau and took over an hour; none of the three experiments built on it gained anything. → Analysis is capped at 10 minutes without an experiment queued from it; two such ideas without a gain and it stops.

**A text identical for every case scored as well as the whole case-specific system.**
(A text-generation task scored by character n-gram overlap, chrF.) A generic text built greedily from whole sentences of the training targets scored 36.5 and 38.0 on two hold-outs; retrieval of similar cases with expected-score selection scored 36.0 and 38.5. It was found about 50 minutes in, after four ideas for the case-specific part had each gained 0.1 or less. → For overlap metrics (chrF, BLEU, ROUGE), measure the generic-text floor and the top-1 retrieval score in the probe stage, then look for what the case adds above that floor. Using a generic text as the output is a rule question: ask before doing it.

**Under a recall-weighted overlap, whole texts added nothing and short sentences added three points.**
On one hold-out fold, joining the two best retrieved texts scored 35.99 against 35.98 for one, and three scored 33.06. Appending up to two sentences of at most 70 characters, each picked for its marginal expected gain against the similar cases' texts, gave +3.35 on average over four folds, positive on each. → Extend at sentence level by marginal expected gain, never by whole candidates, and measure each added piece before keeping it. Say plainly in the description that the output is built this way.

**Two hold-out designs bracketed the board score.**
A pessimistic design (whole clusters of source documents held out) and an optimistic one (whole documents held out) scored 34.70 and 38.15 for a version the board scored 36.30; their mean was 0.13 away, and 0.2 away for a later version (41.43 against 41.64). → When the held-out factor is only partly in the data, run both designs, report both, and calibrate on their mean.

**Six additions to one model were each inside the noise; a different kind of model, weaker alone, blended for +0.02.**
(The Enclosure Mention.) A linear classifier over hand-built candidate features reached 0.753 locally. A span-extent model, a document-type model, sibling-contrast features, a per-pair softmax ranker, an enclosure-to-context ridge model, a boosted model stacked on the scores and punctuation features each moved it by 0.005 or less, and its learning curve was flat from 1,000 to 1,280 rows. A bidirectional-GRU reader with attention, trained from scratch, scored 0.746 alone; blending its log-probabilities over the same candidates at weight 0.2 gave 0.772, and the board went from 0.8072 to 0.835. More seeds of the reader added nothing to the blend; a larger or longer-trained reader blended worse (more confident, not better). → After three consecutive additions inside the noise, stop adding to that model: check the learning curve and a stacked learner (minutes each), then build a structurally different model and blend log-probabilities at a small weight. Judge the second model by what the blend gains, not by its own score. Where combination is legal and a complementarity hypothesis exists, save out-of-fold outputs and test that contribution cheaply before closing the approach. If it was not tested, record that limitation rather than treating a weak standalone score as proof that the model cannot help. This narrows two older entries: a from-scratch network can be worth building at about 1,600 examples as a blend partner, and averaging helps when the models are different in kind.

**A version that scored was still rejected in a later human review: the challenge's own markdown had ruled the step out (The Enclosure Mention).**
(The user's account, 2026-10-07.) Version 5 gave different spans to two test rows that shared the same cover text (+0.016 locally, +0.013 on the board, 0.835). The challenge markdown said, in its own way, not to do this. The version was scored and shown on the board, and that was taken to mean the step was acceptable; nobody knew there is a third phase in which people review the solutions. In that phase it was rejected: "[R]ejected. Final inference modifies a pair's candidate probabilities using another test pair's probabilities when their covers match. Each prediction must use its own cover–enclosure pair and training-fitted state. Remove the cross-pair adjustmen[t]" → Three things. (1) A score from the bot is not approval. After scoring comes a human review of the code against the challenge's markdown, and it can remove a result at the very end. (2) The rules that count are the ones in that challenge's markdown. Each challenge has its own: something allowed in one (pretrained weights, using test rows together, a named library) is forbidden in another, so nothing carries over from an earlier challenge, in either direction. Read the markdown's rule sections line by line at intake, quote them in `RULES.md`, and before every hand-over check the inference code against each quoted line. (3) When the markdown restricts something "in its own way" (a sentence about what each prediction may use, about test rows, about sources), take the strict reading of that sentence and build to it. Do not reason that a step is fine because an earlier version containing it was scored.

## Confirmed deadline and admission lessons (2026-10-07)

**A short deadline was attempted without enough reserved gate/upload time.**
What happened: the final FlockSight request allowed ten minutes while the protected plan's two full checks alone took about7m39s; candidate checks and analysis consumed the remaining margin, and no supported new package was delivered before the deadline. Cost: the last improvement window was missed; the user reported20th and ended the run. Habit: compute the measured full critical path before committing, state infeasibility immediately, freeze early and communicate the outcome before the deadline. Never skip or fabricate the exact-script proof to make an ETA. Measure the candidate under its actual deterministic backend/thread plan; include validation and upload, and use a conservative full-run time for both proof runs. Re-rank against the moving admission cutoff: a supported improvement can still be too small or too late. See `climb.md`, "Continue improving the work smartly". **CONFIRMED process failure**, not evidence that an arbitrary new model would have won.

**A baseline win did not secure continued participation.**
What happened: FlockSight scored54.0876 against53.5336, but was14th during grace and finally20th by the user's report; the displayed tenth-place score was54.7702. Cost: being above baseline did not retain top-10 eligibility after the window. Habit: when admission is rank-limited, track the live top-10 margin and grace cutoff separately from baseline qualification. Preserve uncertainty about forwarded countdown freshness and future scores. **CONFIRMED**, with live platform rules authoritative.

## Anchored Layout Alignment and the same day's lessons (2026-10-08)

**Careful step-by-step improvement lost the place (Anchored Layout Alignment, lost at 13th).**
What happened: the probe scored 0.239; version 2 (beam search, 74 features) 0.345; version 3 (wall-contact features, +0.016 locally at 3.2 standard errors) 0.338. Five further additions to the same step-by-step scorer were inside the noise or negative. The tenth place rose from 0.33 to 0.39 in three hours and the top to 0.52; the grace hour ended with nothing that could reach it. The first plot of raw items was drawn two and a half hours in, and it showed in one look that the run is a cluster of attached outlines. Cost: the challenge. Habit: `climb.md`, "Biggest step first" - look at raw examples and measure ceilings in the first 30 minutes after the probe, rank ideas by expected gain per hour, build the largest one as version 2, close a family of ideas after two results inside the noise. **CONFIRMED** by the same day's second challenge, where twenty minutes of reading one raw episode gave +18 points.

**An ambiguous sentence in the metric cost the first hour.**
What happened: "share of consecutive ordered pairs that are both right and in the right order" was implemented as "positions i and i+1 both right". The probe scored 0.239 against 0.189 locally; the other reading (a predicted pair that is a true consecutive pair, wherever it sits) gave 0.2475, and the second submission confirmed it (0.359 local, 0.345 board). Habit: at intake, for every metric sentence that can be read two ways, implement both readings and carry both numbers until a board score separates them. The probe's board score is the test; compare it with each reading before any other diagnosis.

**A local gain of three standard errors did not show on a 276-item board.**
What happened: version 3 was +0.016 better than version 2 on paired grouped folds (3.2 standard errors) and scored 0.006 lower on the board. The board's own spread for 276 items in whole localities was about 0.0145. Habit: compute the board's spread from the held-out scores at the first calibration and quote the gain in units of it. A gain under one board spread is not worth a credit when the place is at stake, whatever its paired standard error says.

**The user uploads a folder as soon as it exists (three times a hand-over lesson).**
What happened: the version 3 folder was uploaded before the hand-over message; its description still had three tokens to be filled after the gate, and the gate's placeholder check did not know them. Habit: the description is complete, with real numbers, before the gate builds anything; build into `out/_building_vN` and rename to `submission_vN` only when final; say "not ready, do not upload yet" while a folder is being built under time pressure. Third time (Membrane Confirmation Delay, 2026-10-09): a finished folder for a candidate that was below the best version locally and shared 97.6% of its test predictions with it was left among the upload folders, and the user reacted angrily; only a recommended candidate becomes a `submission_vN` folder (`submission-package.md`).

**A process that was "killed" kept the GPU for twenty minutes during a grace hour.**
What happened: `wmic ... call terminate` returned without stopping a background experiment; the validation that followed shared 6 GB with it and did not finish. Habit: stop processes with PowerShell `Stop-Process -Id` chosen by command line, then confirm with `nvidia-smi` (0 MiB) and a process count before starting the next run. One torch process at a time.

## Variable Coarsening Flow Recovery (lost, 2026-10-08)

**A loss on per-item marginals punished confident correct answers; a loss on the matching itself gave +6 and removed a collapse.**
What happened: only block-level counts were known, so each card's probability mass per block was trained towards the block row in proportion. For a block whose two cards go to two different blocks, that target is 0.5 / 0.5 for each card, so a confident correct match has infinite loss. The score peaked at 48.7 at pass 6 and fell to 23 by pass 12. Training the expected block flow under a soft one-to-one matching (Sinkhorn marginals summed per block pair) gave 54.3 and a flat curve with the same model. Habit: when the label is an aggregate of hidden item-level answers, write down by hand what loss a perfectly right, confident prediction gets before the first training run; if it is not the minimum, change the objective.

**One printed episode showed the structure that an hour of architecture work had missed.**
What happened: printing all cards of one training episode showed that the six role tokens of side A pair one-to-one with the six of side B (exact in every training episode), which splits an 18-card matching into classes of three or four; and a channel diagnostic showed that a card's neighbour excerpt relates to the other side's name tokens, not name to name. A cross-attention term from excerpt (query) to name (document) with a count-style aggregator went 48 -> 66.5 on one validation direction; the same term in the opposite direction had given +0.7. Habit: `climb.md`, "Biggest step first", step 1; and when a signal is "a few tokens of X appear inside Y", make the short side the query.

**One validation direction overstated the board by 6 points, and the cut-down version missed the cutoff rank.**
What happened: version 2 was validated on one direction only (63.2 at 10 passes) because a leftover process held the GPU during the grace hour; the board gave 57.42 against a tenth place of 62.27. Version 1, validated on both directions, had been 2.9 under the board. Habit: under a grace hour, hand over what is already validated both ways at the largest safe pass count, and say the one-direction number is not a board estimate.

## Conversational Functional Span Decomposition (lost at 14th, 2026-10-08)

**Three platform checks ate the grace hour.**
What happened: the probe (scikit-learn, CPU) was refused for "explicit implementation requirements" in a From Scratch challenge; version 3 scored and was then held for "Deterministic Execution"; its fix was accepted; version 5, the same file with one constant changed, was held again. About 70 of 170 minutes went to rebuilding and re-proving instead of improving, and the last version never counted. Habit: `determinism.md`, section 8: the skeleton from the first file, the probe in the final framework, the gate's stricter stage 1 and two hash seeds in stage 3. **CONFIRMED** for the causes found in version 3; the second refusal is unexplained.

**Local validation ran further ahead of the board with every version.**
What happened: leave-one-conversation-out gave 0.481, 0.514 and 0.514 where the board gave 0.4507, 0.4726 and 0.4607 (gaps 0.030, 0.041, 0.053) on three test conversations; the spread between held-out conversations was 0.06 to 0.07. Two versions with the same model and a different feature order differed by 0.012 on the board. Habit: with a handful of test groups, quote every local gain against the between-group spread divided by the square root of the number of test groups (here about 0.036), say that the rank is partly luck, and choose the challenge with more test groups when there is a choice.

**No large step existed here, and the board moved faster than refinement.**
What happened: ceilings measured right after the probe (oracle functions on the model's spans 0.74, oracle spans with the model's functions 0.61) showed the loss was in function labels that depend on annotator style; the label mix differs strongly between conversations and one short word carries five or more functions. Gains were +0.02 (two-stage model), +0.02 (recurrent reader blended, better alone than the feature model) and +0.01 (boundary head); a wider-context reader was worse. The tenth place went from 0.45 to 0.50 in the same two hours. Habit: when the ceiling check shows noise-limited labels and no structure, say at once that the task is a tight race decided by a few hundredths and by the platform checks, and put the first effort into a clean accepted submission in the final framework.

**An experiment that scores many settings per fold must batch its predictions.**
What happened: a grid of 24 decoding settings re-ran the whole per-row prediction for each setting and would have needed 90 minutes; it was stopped during the grace hour with no result. Habit: cache each fold's model outputs once (logits per row) and score the settings from the cache.

## Membrane Confirmation Delay (2026-10-09; public 9th at 0.4259, FINAL private 2nd at 0.5274)

**Features that identify the series, not the biology, cost 0.04 when the test is one unseen group.**
What happened: 192 of the 540 test compounds were one scaffold family (a single benzene ring), against 24 in training and a largest training family of 23. Versions 1 to 3 kept the absolute dose and top-dose features and scored 0.3906, 0.3852 and 0.3852 on the board while scaffold-grouped local CV said 0.496 to 0.520 (board about 0.10 lower, and local differences of 0.01 to 0.02 between versions meant nothing: the fold-seed spread was +-0.015). Version 4 dropped them, smoothed the cell that had not been used to select the anchor and predicted fewer rare-class labels: 0.4224. Cost: about 100 minutes of local tuning on a signal that did not transfer. Habit: at intake compute the group sizes of the public test inputs (descriptive only, never used to fit or select); when one unseen group holds a large share, drop features that describe the series or the tested range (absolute doses, ranges, ids), use at least three fold seeds, validate also on the training analogs of the dominant group, and expect the board far below local. **CONFIRMED** (version 3 -> 4, +0.037).

**Research and structurally different models came only after three clones.**
What happened: versions 2, 3, 5, 6 and 8 were tweaks of one MLP family (2 and 3 shared 98.3% of the test predictions and scored the identical 0.3852). The user wrote "you didn't research what approaches work best on this task from scratch and that really is hurting my standing", "why start it if you know it is a clone" and "don't test models sequentially". One parallel wave of different output heads then showed that wide-and-deep + hierarchical-gate + mixture-cure heads together (0.5216) beat the plain softmax MLP (0.504); that version scored 0.4254, and the same heads plus a descriptor-only structure network 0.4259 (the best). Cost: the first hour and the standing during the grace hour. Habit: a short methods-only research pass right after the probe (`research-gate.md`), a parallel wave of structurally different families, and no clone folders (`climb.md`, "Run a wave" and "No clones"). **CONFIRMED** by three user corrections; the board gain of the family change itself (+0.003) is inside the noise.

**Rare-class false alarms are the cheapest macro-F1 on a noisy shard.**
What happened: with class 2 and 3 F1 near 0.2, a wrong rare-class label costs about 0.001 of macro-F1 (break-even precision about 0.12); the board paid +0.037 for removing 26 of them (class-weight exponent 0.75 -> 0.5) while the local score fell 0.01 to 0.02. Per-class offsets tuned on out-of-fold predictions lost 0.01 to 0.02 on held-out halves, and cutting class 3 almost to zero cost 0.03 locally. Habit: with a macro metric and weak rare classes start the class-weight exponent at 0.5, read the paired board difference by class (`climb.md`, "Watching the board"), and do not tune per-class offsets. **CONFIRMED** on this board, **HYPOTHESIS** as a general setting.

**The public board was the outlier; local grouped validation matched the private result.**
What happened: the best public score was 0.4259 (9th) while scaffold-grouped local CV said 0.52. The final private score was 0.5274 (2nd; the winner 0.5414), and ranks moved by up to 9 places between the public and the private board. About 100 minutes went into reshaping the model toward the public number, and the "+0.037" of version 3 -> 4 is public-only evidence. Habit: when local validation reproduces the stated split and is stable over fold seeds, select by it and treat a gap to the public board as a property of the public shard until an independent signal says otherwise; never read a public rank as the final rank. **CONFIRMED** once. It qualifies the first and third lessons of this section, whose board evidence is public-only.

**A rejected idea is rejected only in the form it was tested.**
What happened: the winner's script (shared by the user after closing) uses several things recorded here as "did not work": neighbouring concentrations, monotone smoothing along the series, crossing-position and gap-over-slope features, an ordinal head, the absolute log concentration, and class weights tuned on out-of-fold macro-F1. The form differed: three separate feature views (56 / 34 / 18 values), each with its own one-hidden-layer network of 64 units trained full-batch for only 40 or 600 steps, softmax and ordinal heads, the 480 fold models averaged as the test ensemble, and the class weights averaged over ten bootstrap resamples. Its private lead over our version was 0.014 on 540 compounds, so this does not show that those ideas caused it. Habit: record a failed idea with the form it was tested in (concatenated into one network, one weight search), and before closing a family test the view-per-network and the bagged form once. **HYPOTHESIS**.

## Volcanic Ash Componentry (2026-10-09, open: 1st on the public board at lockdown, private result not known)

**Seed noise was as large as the whole top ten, and it was measured late.**
What happened: two seeds of the same plan scored 0.6616 and 0.6779 on the board (hold-out spread 0.013 over three seeds), while the top ten spanned 0.016. Every candidate after version 2 (weight mix-back, a registers variant, stronger augmentation, a second architecture in the average) was judged against single runs inside that noise, and the top-10 place came from a second seed uploaded in the grace hour, not from any of them. Cost: about three hours of experiments that could not have been read either way. Habit: run three seeds of the first strong version as soon as it exists, state every later comparison in units of that spread, and plan the seed average as the final form when the run time allows (it beat the mean of its members in 5 of 5 combinations: +0.0045, paired SE 0.0036). **CONFIRMED** for the spread; the board value of the average is not known.

**A bare seed re-draw was recommended before it was read against the rule text.**
What happened: version 3 was version 2 with seed 1, recommended as a named lottery ticket. The description says "no tuning against private labels", which a strict reader can apply to "same file, new seed, keep the better leaderboard score". This was raised with the user only after the upload, when they asked whether a rule was broken. Cost: an open review risk on the version that holds the place. Habit: before offering any re-draw or leaderboard-driven resubmission, quote the sentences about private labels and leaderboard use in `RULES.md` and take the strict reading; the compliant form of the same idea is an average of the first N seeds, fixed before any score exists. **CONFIRMED** as a lapse; the review outcome is not known.

**On the 6 GB card a run that exceeds the memory does not fail, it crawls.**
What happened: a ViT-B in 32-bit at batch 32 needed 6.7 GB and ran at 5 images per second instead of 30; inference at batch 128 took 3 minutes instead of 0.4; a second model trained after the first in one process ran four times slower. Two of these runs were piped through `| tail`, so nothing was visible for 20 minutes each. Cost: about 45 minutes and a late version 2. Habit: before any long run time ten steps and print the peak memory (stay under about 4.5 GB), process a batch in fixed halves with accumulated gradients when it does not fit, use the training batch size for inference, release a model (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) before building the next, and log to a file with flush, never through a pipe that buffers. **CONFIRMED**.

## Station Siting (2026-10-11; public 20th at 0.4761, FINAL private 2nd at 0.5203)

**The public board was the outlier again; the plain locally validated version won.**
What happened: version 3 sat 20th on the public board, 0.022 under the baseline, while nine solvers passed it and the top reached 0.596. On the private set version 3 scored 0.5203 and finished 2nd; ranks moved by up to 19 places. This is the second time after Membrane. What it cost: the last three hours went into chasing the public gap (versions 4 and 5), and the user was told the baseline might be out of reach. Habit: when the test set is a few hundred rows with rare classes, work out what one rare row is worth (here about 0.01) and say at the start that public rank is noise of that size; keep the best locally validated version as the standing answer and do not describe the public gap as a method gap without evidence.

**A version assembled on the folds that score it lost 0.05 on the board.**
What happened: version 4 combined three models and chose, on the same five folds, which models served which axis; local 0.495, board 0.4417, below version 3 (0.4761). Habit: a choice made on the validation folds needs its own held-out check (nested folds) or it is reported at the unselected number; under time pressure prefer the single change with the plainest evidence.

**The automatic compliance check needs time in the plan.**
What happened: the "Prompt Compliance" check stopped compliant, unchanged files repeatedly (version 2 twice, version 3 three times, version 5 until lockdown) and returned the credit each time; version 5, handed over 30 minutes before the end, was never scored. Habit: write every per-test-row rule visibly from the first version (one station at a time, test read after training), tell the user that an identical generic message means "send the same files again", and treat one hour before closing as the last moment for a new version.
