# Budgeted Taphonomic Probe Portfolio

- Domain: Other
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: feature-engineering, small-data

---

Budgeted Taphonomic Probe Portfolio

Objective

Build a decision policy for scientific verification under a hard resource budget. Each row is an independent episode containing a sparse occurrence-count matrix for anonymous horizons and anonymous taxa. Twenty-four currently unobserved horizon-taxon cells are offered as verification probes. For every probe, predict the probability that verification will confirm an occurrence, then select the portfolio of probes that should actually be funded.

This is not a chronology-reconstruction or row-ordering task. A submission never reports a horizon order. Latent temporal structure is only one nuisance signal that a solver may use while estimating uncertain outcomes and allocating a limited verification budget.

Data

`train.csv` has 1,080 labelled episodes. `test.csv` has 360 unlabelled episodes. Source families are disjoint across the two partitions.

Public columns:

* `id`: opaque episode identifier.
* `assemblage_matrix_json`: sparse JSON object with nine anonymous horizons `H1H9`) and observed counts for fourteen anonymous taxa `T1T14`). Missing taxon keys mean an observed zero, not a missing JSON value.
* `probe_bank_json`: JSON object for exactly 24 probes `P1P24`). Each probe gives `horizon`, `taxon`, integer `cost` in `[1,3]`, and integer scientific `value` in `[1,5]`. Every probe addresses a zero cell in the public matrix.
* `taxon_support_json`: an ordinal support level in `[1,5]` for every anonymous taxon. It summarizes how broadly the source profile is supported; it is not a target probability.
* `probe_plan_json`: present only in `train.csv`; this is the labelled submission object described below.

Each evaluation episode contains exactly eight confirmable probes and sixteen genuine absences. This fixed balance makes the uninformed occurrence probability `1/3` and keeps calibration comparable across episodes. Probe costs and values remain public, but outcomes and the optimal funded portfolio are private.

Submission Contract

Submit exactly two columns, in this order:

```
id,probe_plan_json
```

`probe_plan_json` must be valid JSON with exactly these keys:

```
{

  "probabilities": {

    "P1": 0.37,

    "P2": 0.61

  },

  "selected_probes": ["P2", "P7", "P13"]

}
```

Requirements:

* `probabilities` must contain every key from `P1` through `P24` exactly once.
* Every probability must be finite and in `[0,1]`.
* `selected_probes` must contain between one and six unique valid probe IDs.
* The sum of public costs for selected probes must not exceed 10.
* Extra JSON keys, duplicate IDs, missing test IDs, extra test IDs, or incorrect CSV columns invalidate the affected submission or raise a grading error as specified by `grade.py`.

Evaluation Metric

For probe `i`, let `p_i` be the submitted probability, `y_i` the private binary outcome, `v_i` the public scientific value, and `c_i` the public cost.

1. Calibrated occurrence skill (25%)

```
BS = mean_i((p_i - y_i)^2)

calibration_skill = max(0, 1 - BS / (2/9))
```

`2/9` is the Brier score of the known uninformed prevalence forecast `p_i = 1/3`. This term rewards probability resolution only when it improves on that explicit baseline.

2. Value-aware priority ranking (35%)

Probes are ranked by submitted expected scientific value `p_i v_i`*. Their realized gains are* `y_i v_i`. The component is standard NDCG across all 24 probes, with deterministic probe-ID tie breaking.

3. Budgeted realized utility (40%)

The selected portfolio earns `sum(y_i * v_i)`. It is divided by the exact maximum realized value achievable for that episode under both constraints: at most six probes and total cost at most 10. The private optimum is computed by dynamic programming.

The row score is:

```
0.25 * calibration_skill

+ 0.35 * value_aware_NDCG

+ 0.40 * budget_utility
```

The challenge score is the mean row score. The minimum is `0`, the maximum is `1`, and a perfect occurrence forecast paired with an optimal feasible portfolio scores exactly `1`.

The weights follow the operational objective: the funded decision receives the largest share, the quality of the complete expected-value queue receives the next largest share, and global probability calibration receives the remaining share. The terms are complementary rather than duplicate accuracies: one evaluates beliefs, one evaluates prioritization, and one evaluates the final constrained action.

Leakage Controls

* Source families are assigned wholly to train or test by a deterministic family hash; no family contributes profiles to both partitions.
* Episode RNG seeds include private profile nonces that are absent from all prepared public files. Public row position, split name, or episode ID is therefore insufficient to reconstruct a shuffle or target.
* Horizon, taxon, and probe labels are independently permuted within every episode and carry no global meaning.
* Prepared files contain no source taxon names, family names, order names, absolute ages, source digests, or private nonces.
* Test targets are generated only in `private/answers.csv`; no target-derived column is written to `test.csv` or `sample_submission.csv`.
* Costs and values are public decision inputs. They do not reveal outcomes; value is deliberately useful only when combined with a probability estimate.

Expected Approaches

Useful systems may combine sparse matrix completion, co-occurrence similarity, latent path inference, calibrated binary classification, and knapsack optimization. The two outputs must be coherent: high-value probes should be selected only when their predicted confirmation probability justifies their cost.

Dataset Limitations

* Episodes are controlled simulations anchored to empirical temporal ranges; they are not records of actual follow-up field campaigns.
* The preservation, facies, abundance, and detection processes are simplified and omit geography, lithology, collector behavior, and taxonomic revision history.
* Counts are capped and support is discretized, so neither should be interpreted as a real abundance estimate.
* Fixed 8-of-24 class balance is an evaluation design choice and does not estimate prevalence in nature.
* Public-source coverage is uneven across taxa, time, geography, and research effort. The challenge is unsuitable for claims about biological diversity, extinction timing, or funding priorities outside the benchmark.
* A high score demonstrates performance on this constrained verification policy, not scientific validity for unsampled formations.
