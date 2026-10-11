# NOTES — StanzaOrder (From Scratch, A10G, adjacency accuracy, baseline 0.4846)

## Clock

- Picked: 2026-10-11 21:03 UTC (`/axo select`, user "go"). Data on disk: 21:05 UTC.
- Closing time: not shown by the bot (no countdown in the status block).
- Status at pick (fetch 21:04 UTC): OPEN, 2/10 beat AI, top 0.5220, 2nd 0.4879, 53 submissions, credits 5/6
  (the group's shared account; its existing 0.4028 entry has no working folder on this PC).
- Spent time: tracked from 21:03 UTC.

## Machine

- RTX 3050 Laptop 4 GB, ~14 GiB RAM (tight), 12 cores; who-cited session (exp_gt.py) and axo-eris session share it.
- Python: `/mnt/work/shipd-ai-eris/axo-eris/.venv-ml/bin/python`; runs under `systemd-run --user --scope -q -p MemoryMax=2G`.

## Validation protocol

`dev_eval.py`: 5 folds of whole poets (greedy by size, seed 0); probe numbers on fold 0 (22 poets, 2,075 stanzas).
Reported: hold-out mean, and the mean reweighted to the test size mix (4:1103 5:201 6:683 7:108 8:626 9:74 10:190;
train is 53 % 4-line, test 37 %). Chance on the test mix 0.1845. Per-row sd 0.37 -> hold-out SE 0.008, board SE ~0.007.

## Versions

| Version | Local (fold 0, test-mix weighted) | Board | Key changes | Full run | Notes |
|---|---|---|---|---|---|
| v1 probe (`solution_v1.py`, `submission_v1/`) | 0.3886 (raw mean 0.4205; train fit 0.5066) | pending | line + pair features (rhyme rules, end punctuation, first-word vocab 100, syllables), PyTorch pair net on cuda:0, Held-Karp exact decoding | 0.3 / 0.3 min, IDENTICAL | handed over 21:12 UTC |

BEST: none scored yet.

## Look at raw examples (21:13 UTC)

- Schemes seen: abab, aabb, abba, aabccb, ababccd, abcb, abaab; unrhymed Latin hexameter exists. End punctuation and
  syllable counts follow the scheme (8-7-8-7, last line '.'/'!').
- Rhyme rule bug: "qu" read as vowel (Qual/tausendmal, Quell/hell missed) -> fixed in v2 (qu -> kw).
- Rhyme pattern counts on train gold orders, n=4: abab 1684, aabb 744, abcd 458, abba 432, abcb 391, abac 212.

## Ceilings (ceilings.py, fold 0, n<=8, v1 pair scores, exhaustive permutations; test-mix weighted n<=8)

| decoder | score |
|---|---|
| v1 exact | 0.4001 |
| oracle gold rhyme pattern | 0.4866 (+0.087) |
| oracle gold gender pattern | 0.4934 |
| oracle gold punctuation sequence | 0.5863 |
| count prior rhyme / gender / punct / all (lambda 1) | 0.4074 / 0.3907 / 0.3993 / 0.4033 |

=> patterns carry information but marginal frequency priors cannot pick the stanza's pattern (+0.003..0.007).
v1 train fit only 0.507: the model cannot see enough. Next: let it read the text and the whole stanza.

## Research pass (methods only, 3 searches, 21:20 UTC)

- Pairwise ordering predictions add to a pointer/listwise decoder (Yin et al., AAAI 2020); B-TSort predicts
  before/after for all pairs (Prabhumoye et al., ACL 2020) -> auxiliary before/after head, used in the exact DP.
- Rhyme schemes as latent stanza structure learned without dictionaries (Reddy & Knight 2011) -> context encoder
  over the stanza's lines instead of a fixed prior.

## Ranked list for v2 (expected gain / build time)

1. Stanza context (set transformer) + character encoders (head, last word, whole line) + position and
   before/after heads used in exact decoding. Expected +0.03..0.08. 30 min build. -> wave of 4 variants (fold 0).
2. Seed / model ensembles of the winner (deferred until the place is safe).

## Wave 1 (fold 0, test-mix weighted; 21:15-21:25 UTC) — the capacity family is closed

| variant | hold-out | train fit |
|---|---|---|
| v1 probe | 0.3886 | 0.507 |
| plain (v2 heads, vocab 200, qu fix, dropout) | 0.3812 | 0.566 |
| context transformer, no chars | 0.3684 | 0.624 |
| chars, no context | 0.2691 | 0.958 |
| full (chars + context) | 0.3372 | 0.812 |

Characters memorise the training poets' lines. Position / before-after heads in decoding: no gain (all variants).

## v1 on all 5 folds (exp_mbr.py): honest probe estimate 0.3709 (test mix), folds 0.351-0.389; fold 0 easiest.
Expected-adjacency decoding (sum of successor probabilities) +0.005 (SE 0.003): small, deferred.

## Diagnosis (21:32 UTC): direction is the bottleneck
Pooled v1: directed adjacency 0.390, UNDIRECTED 0.578; 18.9 % of predicted pairs are gold pairs reversed;
first line right 40 %, last line right 75 %. Typical error: [3,2,1,4] (score 0). Rhyme / metre are symmetric; only
syntax and sense give direction. -> character LM from scratch (exp_lm.py) scored as P(b | a) in the exact DP.
LM 1 epoch: LM alone 0.224, any lambda hurts (undertrained). 8 epochs running.

## Character LM (exp_lm.py, fold 0, test-mix weighted; 21:36 UTC)
3 layers d192, 8 epochs (340 s on the 3050 for 80 % of train), train loss 1.41 and still falling.
lambda 0 0.3886 | 0.1 0.3981 | 0.2 0.4003 | **0.3 0.4012** | 0.5 0.3882 | 1.0 0.3801 | LM alone 0.2918.
=> +0.013 on fold 0 (raw mean +0.017). Fold-1 check OOM'd (GPU shared with the big LM); rerun after.
Running: 4 layers d256, 12 epochs on fold 0 (does the signal grow with LM strength?).
Candidate file: solution_v2.py = v1 network + LM (lm_weight 0.3). Capacity wave file moved to out/not_for_upload/.
