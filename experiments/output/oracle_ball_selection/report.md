# Oracle ball selection diagnostic

Read-only: can subsampling balls by ranking criteria recover performance?

Rows: **8568** | Weighting: **Gaussian purity_compact**

## 1. How much performance is recoverable with oracle selection?

| representation | all balls | LOBO oracle 50% | LOBO oracle 75% | gain @50% | gain @75% |
|---|---:|---:|---:|---:|---:|
| B0 | 0.569 | 0.594 | 0.637 | +0.0243 | +0.0674 |
| rho0.80 | 0.576 | 0.635 | 0.632 | +0.0593 | +0.0562 |
| rho0.85 | 0.582 | 0.598 | 0.620 | +0.0154 | +0.0372 |
| rho0.90 | 0.597 | 0.603 | 0.649 | +0.0061 | +0.0523 |
| rho0.95 | 0.569 | 0.594 | 0.637 | +0.0243 | +0.0674 |
| SU3 | 0.556 | 0.595 | 0.598 | +0.0389 | +0.0427 |

## 2. Which ranking criterion approaches oracle? (50% selection)

| criterion | mean macro F1 | gap to oracle |
|---|---:|---:|
| boundary_compact | 0.583 | 0.0191 |
| boundary_score | 0.582 | 0.0204 |
| entropy | 0.579 | 0.0236 |
| random | 0.567 | 0.0356 |
| density | 0.561 | 0.0413 |
| purity | 0.559 | 0.0431 |

## 3. Does selecting boundary balls improve over all-ball training?

- Mean Δ macro F1 (boundary 50% − all balls): **+0.0066**
- Mean Δ macro F1 (boundary 75% − all balls): **+0.0113**
- Fraction improved @50%: 47.1%
- Fraction improved @75%: 39.5%

## 4. Bottleneck diagnosis

**Verdict: both**

Oracle subsampling recovers macro F1 over all-ball training, and best-generation representation (rho0.90) still outperforms SU3 at 100% — both selection and generation limit performance.

- Best generation @100%: **rho0.90** (macro F1 0.597)
- Mean oracle recoverable gain @50%: +0.0281
- Mean boundary selection gain @50%: +0.0066

## Outputs

- `results.csv`
- `summary.json`

