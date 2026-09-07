# Granular complexity budget diagnostic

Read-only analysis: is performance controlled by ball count (complexity) or geometry quality at a given complexity level?

Reference: **rho=0.90** split tree | Budgets: [25, 50, 75, 100, 125, 150]% of final ball count
Runs: **306** (6 budgets × dataset × seed)

## Research questions

### 1. Is there a clear optimal ball-count range?

**Answer A (ball count) dominates.** Macro F1 vs n_balls spearman=-0.494, vs split_count spearman=-0.455. Mean balls range from 19.4 (25%) to 90.8 (150%). Peak at 25% (F1=0.571) includes under-granulation artefacts (≤3 balls on 8 runs); among 50–100% budgets, peak is 100%.

### 2. Does performance peak before maximum refinement?

Performance is relatively flat between 75–125% budgets: best 50–100% budget is 100% vs rho0.90 reference 100% (F1=0.546). 75%: F1=0.542, 100%: 0.546, 125%: 0.564. Over-refinement beyond rho0.90 (125–150%) does not consistently improve F1.

### 3. Does boundary score explain performance better than ball count?

**Ball count explains performance better than boundary score** (spearman -0.494 vs 0.091 for log-boundary). Geometry quality at a fixed budget is secondary to complexity level: fewer balls consistently yield higher macro F1. Boundary score still carries weak positive signal but is much weaker than fragmentation.

### 4. What should a global stopping criterion optimize?

A global stopping criterion should optimize **cumulative ball-count budget** near 75–100% of the rho0.90 reference (~78 balls mean), not maximum purity or per-split utility. Split count (spearman=-0.455) and ball count (-0.494) are the primary negative controls; log-boundary (0.091) is a secondary positive signal. Stop when the budget is exhausted, not when local purity stops improving.

## Budget sweep (aggregate means)

| budget% | balls | splits | min surv | log-boundary | mixed | macro F1 | min F1 | gmean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 19.4 | 23.8 | 0.993 | 1.490 | 0.831 | 0.571 | 0.480 | 0.517 |
| 50 | 38.8 | 56.3 | 0.970 | 1.661 | 0.726 | 0.524 | 0.421 | 0.410 |
| 75 | 58.5 | 100.1 | 0.919 | 1.408 | 0.598 | 0.542 | 0.421 | 0.414 |
| 100 | 77.8 | 166.1 | 0.828 | 1.251 | 0.458 | 0.546 | 0.402 | 0.388 |
| 125 | 90.1 | 231.4 | 0.693 | 0.610 | 0.272 | 0.564 | 0.424 | 0.422 |
| 150 | 90.8 | 237.0 | 0.680 | 0.584 | 0.261 | 0.554 | 0.409 | 0.405 |

## Predictor correlations (macro F1)

| predictor | pearson | spearman | n |
|---|---:|---:|---:|
| macro_f1_vs_n_balls | -0.183 | -0.494 | 283 |
| macro_f1_vs_mean_log_boundary_score | 0.040 | 0.091 | 283 |
| macro_f1_vs_split_count | -0.160 | -0.455 | 283 |
| n_balls_vs_log_boundary_at_same_budget | -0.142 | -0.420 | 283 |

Peak macro F1 budget (50–100% range): **100%** | rho0.90 100%: F1=0.546


## Outputs

- `results.csv`
- `summary.json`

