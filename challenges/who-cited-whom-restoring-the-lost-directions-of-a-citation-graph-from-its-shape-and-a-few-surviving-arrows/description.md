# Who Cited Whom? Restoring the Lost Directions of a Citation Graph from Its Shape and a Few Surviving Arrows

- Domain: Other
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: -

---

Overview

A citation points from one paper to another, usually from the newer to the older. Take a few dozen research papers of one area that cite one another, strip everything but the wiring (no titles, no dates, no authors, random paper numbers) and wipe the direction off all but about one link in ten. **For every test graph, orient the links that have lost their direction: for each of them, say which of its two papers cites the other.**

What is left to go on is the shape of the graph and the few directions that survive. The rows are not independent examples: the links of a graph share their papers, and the answer for a test graph is one orientation of the whole graph, written out link by link. Training graphs come whole, with every direction. The test graphs come from research areas that supply no training graph, so no paper and no citation of a test graph is in the training set.

Relation to existing work

Link prediction asks whether an edge exists; here every edge is given and only directions are missing. Related lines of work are these.

* *Direction prediction on one graph.* MagNet (Zhang, He, Brugnone, Perlmutter and Hirn, NeurIPS 2021) predicts the direction of held-out links of a citation network from the rest of the same network, where most directions are known.
* *Direction recovery in social networks.* Zhang, Wang and Wang (WWW 2014) recover the hidden directions of the ties of a large social network from its topology, again one network at a time.
* *Network archaeology.* Sreedharan, Magner, Grama and Szpankowski (Scientific Reports, 2019) infer a partial order of node arrival from one snapshot of a grown network, analysed for the preferential-attachment model, with no training data.

Here the unit is a small graph and there are hundreds of them. A solution is fitted on graphs whose directions are all known and is scored on graphs it has never seen, from research areas it has never seen, where it gets about one direction in ten.

Evaluation

The Matthews correlation coefficient between the true and the submitted value of `a_cites_b`, over the links to answer of all test graphs taken together (the value `sklearn.metrics.matthews_corrcoef` returns). Higher is better: 1.0 for the true directions, 0.0 for the same answer on every link, -1.0 for every direction reversed. A file with a missing or repeated `link_id`, or with a value other than 0 or 1, is rejected with an error.

Dataset

```
public/
  train.csv              81,781 rows   link_id, graph_id, paper_a, paper_b
  train_labels.csv       81,781 rows   link_id, a_cites_b
  test.csv               25,257 rows   link_id, graph_id, paper_a, paper_b
  test_known.csv          2,966 rows   link_id, graph_id, paper_a, paper_b, a_cites_b
  sample_submission.csv  25,257 rows   link_id, a_cites_b
```

* A row of `train.csv`, `test.csv` or `test_known.csv` is one link: two papers of one graph, one of which cites the other. `graph_id` names the graph and `link_id` the link; both are random codes, and the rows are sorted by `link_id`.
* `paper_a` and `paper_b` are numbers from 1 to the number of papers of the graph; they were given out at random and mean something only inside their graph. Which paper of a link is listed first was decided by a fair coin.
* `a_cites_b` is 1 if `paper_a` cites `paper_b`, and 0 if `paper_b` cites `paper_a`. `train_labels.csv` has it for every training link.
* The links of a test graph are split over two files. Every test link kept its direction with a chance of one in ten, independently of all the others: the links that kept it are in `test_known.csv`, with `a_cites_b`, and the links that lost it are in `test.csv`. A test graph can have very few links in `test_known.csv`, or none. The 25,257 links of `test.csv` are the ones to answer.
* A graph holds 30 to 80 papers of one research area and every citation among them; two papers that cite each other are not linked. A graph is connected.
* There are 729 training graphs and 294 test graphs, and the research areas of the test graphs supply no training graph. The files do not say which area a graph belongs to.
* `sample_submission.csv` lists the links to answer, with 0 on each.

Rules

* Only the five released files may be used: no other citation data, no pretrained weights, and no attempt to find out which papers these are.
* Everything a solution learns or tunes (weights, settings) comes from the training graphs alone. The surviving directions of a test graph are inputs for that graph only: the answer for a test graph is computed from its own links and its own surviving directions, and may not depend on any other test graph.

Submission

A CSV file with a header row (`link_id`, `a_cites_b`) and one row for each of the 25,257 links of `test.csv`, in any order, where `a_cites_b` is 0 or 1. Rows with any other `link_id` are not scored.
