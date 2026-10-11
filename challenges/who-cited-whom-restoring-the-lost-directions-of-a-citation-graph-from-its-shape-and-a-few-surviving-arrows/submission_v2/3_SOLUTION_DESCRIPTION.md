# Solution description - Who Cited Whom?

## Approach
A citation runs from the newer paper to the older, so a link's direction compares the ages of its papers. Per paper, structure comes from its own graph (degree, triangles, neighbour degrees, core number, two-step reach, centrality, graph size). A network trained from random initialisation builds a vector per paper: each layer mixes it with the mean over linked papers and, separately, the papers it is known to cite and be cited by (the surviving directions). An antisymmetric head scores a link: logit(a cites b) = f(a,b) - f(b,a). A second round passes the first round's predicted directions as soft citations. Training imitates the test: each pass, every training direction survives with probability 0.1 (fixed seeds). Three networks (seeds 0, 1, 2) are averaged. Each test graph is answered alone.

## Validation
Training graphs clustered by shape into 12 groups, 4 folds of whole groups, simulated 10% surviving directions: pooled MCC 0.5241 (first version: 0.4999 locally, 0.5075 on the board). A local estimate only.

## Rule compliance
Only the five released files; no pretrained weights. All weights and settings come from training graphs. Each test graph uses only its own links and surviving directions, after training. Ids, paper numbers and listing order are not used.

## Runtime
2.1 min on 4 CPU threads. Fixed seeds, 30 passes, no early stopping; two full runs gave byte-identical files.
