# Solution description - Who Cited Whom?

## Approach
A citation runs from the newer paper to the older, so a link's direction compares the ages of its two papers. For each graph, per-paper structure is computed from that graph alone (degree, rank, triangles, neighbour degrees, core number, two-step reach, eigenvector centrality) plus the paper's known outgoing and incoming citations among the graph's surviving directions. A link gets both papers' values, their differences, shared neighbours and known citation chains between the two. Training imitates the test: each training direction survives with probability 0.1 (fixed seed), the others are learned, in both listing orders. One scikit-learn boosted tree model, 200 iterations. Each test graph is answered on its own; both orders of a link are scored and averaged.

## Validation
Training graphs were clustered by shape into 12 groups and whole groups held out (176 graphs, 16,802 links, simulated 10% surviving directions): MCC 0.4077. A local estimate only.

## Rule compliance
Only the five released files; no pretrained weights. Model and settings come from training graphs only. Each test graph uses only its own links and surviving directions, after training; nothing across test graphs. Ids, paper numbers and listing order are not used.

## Runtime
0.7 min on 4 CPU threads. Fixed seeds, fixed iterations, no early stopping; two full runs gave byte-identical files.
