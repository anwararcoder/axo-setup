# Postmortem - Variable Coarsening Flow Recovery (2026-10-08, lost)

## A. Challenge facts

- Task: per episode, integer flow matrix between A blocks and B blocks (hidden one-to-one card links, blocks of 2 to 4 cards); from scratch, GPU training required; count, hash, n-gram and co-occurrence features prohibited; 999 training and 192 test episodes; validation G0 <-> G1 prescribed.
- Result: v1 50.73 (local 47.8, both directions), v2 57.42 (local 63.2, one direction, 10 passes); baseline 47.65; tenth place 62.27 and top 98.73 during the grace hour.
- Method that worked: card pairs scored, Hungarian matching per episode, pairs counted per block pair (marginals correct by construction); cross-attention from a card's neighbour-excerpt tokens to the other side's name tokens, both directions, plus a small structural tower; Sinkhorn expected-flow loss. 16 passes validated at 66.5 on one direction and was still rising.
- Data facts: roles pair one-to-one across sides; 3 or 4 links per role class; random matching inside role classes scores 22.5; a valid evidence-blind transport scores 1.6; lex and alias_lex share a token space within a side.
- Not done when stopped: second validation direction, more passes, padding trim for speed, other cross terms, full-data model.
- Never found: how the top solvers reached 87 to 98.

## B. Reusable lessons (promoted to lessons.md)

1. Check by hand that a confident correct prediction minimises the loss when labels are aggregates.
2. Read whole raw examples right after the probe; put the short evidence on the query side of a matching term.
3. A one-direction validation number is not a board estimate; under a grace hour ship what is validated both ways.
