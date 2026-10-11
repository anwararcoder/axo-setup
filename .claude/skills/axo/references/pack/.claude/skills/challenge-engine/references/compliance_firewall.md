# Challenge Compliance Firewall

The challenge description is the source of truth.

## High-risk leakage patterns

Reject unless explicitly permitted:

- using test/evaluation examples as training inputs
- deriving labels or pseudo-labels from evaluation populations
- pooling information across test cases to predict another test case
- retrieving sibling or same-lemma test cases when the benchmark forbids cross-test information
- building synthetic examples when synthetic data is forbidden
- fitting preprocessing, vocabulary, thresholds, normalization, embeddings, retrieval indexes, or encoders on forbidden evaluation data
- using external annotations that recreate or reveal evaluation labels
- using the public/private split as a training signal when the rules prohibit depending on it

## Safe reasoning

Ask:

1. What exact rows/examples are legally available for training?
2. What information may be used at inference time?
3. Can any feature depend on another evaluation example?
4. Can the model or preprocessing see the target population before prediction?
5. Is any generated data forbidden?
6. Is any external resource effectively an annotation source?

When uncertain, choose the conservative interpretation and state the uncertainty to the user rather than silently taking the risky path.
