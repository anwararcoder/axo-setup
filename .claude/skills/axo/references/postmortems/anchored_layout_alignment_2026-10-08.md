# Postmortem - Anchored Layout Alignment (2026-10-08, lost)

## A. Challenge facts

- Task: order 14 of 21 candidate outlines after an anchor; score 0.5 P + 0.3 A + 0.2 J; CPU; 1,227 training and 276 test items; no pretrained weights.
- Result: best board 0.3446 (version 2), baseline 0.3128, 13th when the grace hour ended; tenth place 0.3912, top 0.5201.
- Versions: v1 greedy next-step scorer 0.2390 (local 0.189 strict / 0.248 pairs); v2 74 features + beam search 0.3446 (0.307 / 0.359); v3 + wall-contact graph features 0.3382 (0.324 / 0.375).
- The platform's adjacency term A counts a predicted consecutive pair that is a true consecutive pair (the "pairs" reading); board minus pairs estimate: -0.009, -0.014, -0.037.
- Validation: 48 localities rebuilt among training items from shared outlines (two shared edge-length signatures), 5 grouped folds. Board spread for 276 items about 0.0145.
- Worked: softened step scores in the beam (temperature 4) +0.07; richer step features +0.044; wall-contact graph features +0.016 locally only.
- Did not work: sweep-direction features (0), path re-ranker (-0.01), off-path training (-0.02), minimum-risk path choice (0), set network alone (-0.03; +0.02 as a blend at weight 0.25, never submitted).
- Data facts found late: the true run is a cluster of attached outlines (64% of true steps go to a touching outline); the six extras stand apart (median 34 m from the run); the order sweeps in one direction with 2 to 4 backward steps; stored vertex order carries no signal.
- Never found: what the solvers at 0.41 to 0.52 modelled.

## B. Reusable lessons (promoted to lessons.md and climb.md)

1. Biggest step first: after the probe, look at raw examples, measure ceilings, rank by expected gain per hour, build the largest step as version 2.
2. Implement every reading of an ambiguous metric sentence at intake; let the probe's board score choose.
3. Quote gains in units of the board's own spread; a 3-standard-error paired gain can still be under one board spread.
