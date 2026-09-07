# Adaptive complexity budget validation

Read-only validation: can explicit complexity control reproduce rho=0.90 without using rho as the stopping rule?

Reference: **rho=0.9** purity stopping | Fixed budgets: [50, 75, 100, 125]% | Adaptive: LOOCV log-linear on (n, d, imbalance)
Classifier: Gaussian-weighted GBLSTSVM (purity_compact) | Runs: **306**

## Policy comparison (aggregate means)

| policy | budget% | balls | splits | min surv | purity | boundary | macro F1 | min F1 | gmean | Δ macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rho90-reference | 100.0 | 77.8 | 166.1 | 0.828 | 0.885 | 1265022140.968 | 0.546 | 0.402 | 0.388 | +0.0000 |
| fixed-50pct | 50.0 | 41.2 | 59.8 | 0.968 | 0.805 | 2604166672.087 | 0.524 | 0.421 | 0.410 | -0.0035 |
| fixed-75pct | 75.0 | 58.5 | 100.1 | 0.919 | 0.848 | 1705029842.288 | 0.542 | 0.421 | 0.414 | -0.0041 |
| fixed-100pct | 100.0 | 77.8 | 166.1 | 0.828 | 0.885 | 1265022140.968 | 0.546 | 0.402 | 0.388 | +0.0000 |
| fixed-125pct | 125.0 | 90.1 | 231.4 | 0.693 | 0.938 | 2.279 | 0.564 | 0.424 | 0.422 | +0.0177 |
| adaptive-loocv | 222.0 | 60.9 | 125.1 | 0.906 | 0.852 | 3565062391.759 | 0.541 | 0.404 | 0.414 | -0.0048 |

## 1. Can explicit complexity control replace rho?

rho=0.90 reference macro F1 0.546 (77.8 balls mean). Adaptive LOOCV: F1 0.541, Δ vs rho90 -0.0048, mean |Δballs|=21.2. **Partially** — complexity control captures the rho=0.90 ball-count regime but dataset-only prediction does not fully match rho stopping on classification metrics.

## 2. Is a universal budget enough?

Best universal fixed budget: **fixed-125pct** (macro F1 0.564, Δ vs rho90 +0.0177). Fixed 75%: F1 0.542. A single universal budget is close but not identical to rho=0.90; per-dataset calibration may be needed.

## 3. Does dataset-adaptive complexity improve over fixed budgets?

Adaptive LOOCV macro F1 0.541 (Δ -0.0048) vs best fixed fixed-125pct 0.564 (Δ +0.0177). **Marginal** — adaptive targeting tracks rho=0.90 ball counts but does not consistently beat the best universal fixed fraction (typically 75–100%).

## 4. What should the final stopping criterion optimize?

The final stopping criterion should optimize **cumulative ball-count budget** (target ~75–100% of rho=0.90 depth), not purity or per-split utility. rho=0.90 acts as a hidden regularizer: fixed-100% matches reference (F1 0.546), fixed-50% under-splits (41.2 balls, F1 0.524). Stop when the complexity budget is exhausted; use dataset (n, d, imbalance) only to set that budget, not local purity gates.

## Adaptive calibration (LOOCV model)

- Features: log(n_samples), log(n_features), imbalance_ratio, minority_frac
- In-sample R² (log balls): 0.449
- Mean adaptive budget as % of rho90: 222.0%
- Mean |Δballs| adaptive vs rho90: 21.2

## Outputs

- `results.csv`
- `summary.json`

