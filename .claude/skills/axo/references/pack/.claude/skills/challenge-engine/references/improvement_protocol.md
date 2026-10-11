# Score-Driven Improvement Protocol

Input: current solution + training logs + score + traceback/errors + optional examples.

## Order

1. Runtime correctness
2. Data/split correctness
3. Leakage/compliance
4. Validation quality
5. Optimization/training stability
6. Architecture/features
7. Post-processing/thresholding

Do not jump to architecture changes if the real problem is a split, target construction, runtime, or evaluation mismatch.

## Evidence table

For every proposed change, internally classify it as:
- CONFIRMED: directly supported by the code/logs/error.
- STRONG HYPOTHESIS: highly plausible but not proven.
- SPECULATIVE: weak evidence.

Avoid implementing speculative changes when a confirmed or strong-hypothesis fix exists.

## Preserve winners

Keep a small record of:
- solution version
- score
- key changes
- training time
- validation behavior
- known failure modes

Never replace a higher-scoring valid solution with a lower-scoring one solely because the new approach is more sophisticated.
