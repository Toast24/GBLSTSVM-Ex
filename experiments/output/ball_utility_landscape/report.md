# Ball utility landscape

Read-only leave-one-ball-out (LOBO) diagnostic: which granular balls help or hurt Gaussian purity_compact GBLSTSVM?

Ball records: **21108** | Valid LOBO: **21042** | Configurations: 306

## 1. What properties do useful balls have?

Top 10% **positive influence** (removing hurts macro F1) vs top 10% **negative influence** (removing improves macro F1).

| metric | positive influence | negative influence | Δ (pos − neg) |
|---|---:|---:|---:|
| purity | 0.9519 | 0.9575 | -0.0056 |
| radius | 303.4336 | 248.4620 | +54.9715 |
| normalized_radius | 9.4984 | 8.4342 | +1.0642 |
| density | 32763651116.4602 | 23752969130.5602 | +9010681985.9000 |
| entropy | 0.1409 | 0.1208 | +0.0201 |
| opposite_class_fraction | 0.5619 | 0.6111 | -0.0492 |
| boundary_score | 1.0544 | 0.6215 | +0.4328 |
| support_score | 32773125268.9408 | 23752969139.8320 | +9020156129.1088 |
| margin_score | 1.5895 | 1.5575 | +0.0320 |
| is_mixed | 0.209 | 0.173 | +0.036 |

**Summary:** Useful balls tend toward higher boundary score and heterogeneity; harmful balls tend toward higher purity and larger radius when those deltas are positive.

## 2. Are mixed balls actually useful?

| metric | top-quartile useful | all balls |
|---|---:|---:|
| entropy | 0.0854 | 0.0948 |
| opposite_class_fraction | 0.5727 | 0.5790 |
| boundary_score | 0.3012 | 0.3316 |
| mixed rate | 0.124 | 0.137 |

- Positive-influence balls have higher entropy: **False**
- Higher opposite-class fraction: **False**
- Higher boundary score: **False**
- Mixed-ball mean usefulness: 0.0020
- Pure-ball mean usefulness: -0.0007

## 3. Is purity misleading?

| correlation | Pearson | Spearman | n |
|---|---:|---:|---:|
| purity vs ball usefulness | -0.010 | -0.009 | 21042 |

**Purity misleading:** True

Purity is weakly or negatively associated with ball usefulness; maximizing purity alone is not supported as an optimization target.

## 4. Missing granulation objective (evidence only)

Evidence-based landscape (no algorithm): maximize boundary_score (entropy × density) + margin_score (opposite/same center separation) + entropy / class heterogeneity + boundary-aware mixed support; minimize blind purity maximization + purity-only granulation objective.

Sketch: `utility_proxy ≈ +boundary_score (entropy × density) + margin_score (opposite/same center separation) - blind purity maximization`

**Maximize:** boundary_score (entropy × density), margin_score (opposite/same center separation), entropy / class heterogeneity, boundary-aware mixed support

**Minimize:** blind purity maximization, purity-only granulation objective

## Outputs

- `ball_records.csv`
- `summary.json`

