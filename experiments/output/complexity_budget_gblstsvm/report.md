# Complexity budget GBLSTSVM experiment

New granulation wrapper: stop at target ball budget **B** instead of rho purity.
Baseline: rho=0.9 | Budget fracs: [0.5, 0.75, 1.0, 1.25] × rho90 balls | Sqrt budgets: k×sqrt(n), k∈[2, 4, 6, 8]
Classifier: Gaussian-weighted GBLSTSVM (purity_compact) | Runs: **459**

## Configuration means

| config | family | balls | purity | boundary | min surv | macro F1 | min F1 | gmean | Δ macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rho90-baseline | rho | 77.8 | 0.992 | 2.100 | 0.620 | 0.597 | 0.468 | 0.475 | +0.0000 |
| budget-50pct-rho90 | budget_frac | 39.4 | 0.816 | 6.065 | 1.000 | 0.514 | 0.381 | 0.374 | -0.0735 |
| budget-75pct-rho90 | budget_frac | 59.2 | 0.825 | 4.053 | 1.000 | 0.514 | 0.356 | 0.323 | -0.0739 |
| budget-100pct-rho90 | budget_frac | 77.3 | 0.838 | 3.717 | 1.000 | 0.492 | 0.311 | 0.268 | -0.1005 |
| budget-125pct-rho90 | budget_frac | 95.7 | 0.846 | 1111111114.351 | 1.000 | 0.478 | 0.296 | 0.262 | -0.1143 |
| sqrt-k2 | sqrt | 44.0 | 0.829 | 1.915 | 1.000 | 0.507 | 0.327 | 0.321 | -0.0852 |
| sqrt-k4 | sqrt | 85.6 | 0.847 | 1644518273.576 | 1.000 | 0.440 | 0.228 | 0.182 | -0.1518 |
| sqrt-k6 | sqrt | 113.7 | 0.854 | 1644518273.451 | 1.000 | 0.432 | 0.211 | 0.172 | -0.1602 |
| sqrt-k8 | sqrt | 131.8 | 0.857 | 1644518273.399 | 1.000 | 0.432 | 0.211 | 0.171 | -0.1605 |

## 1. Can explicit complexity control replace rho?

rho=0.90 baseline: 77.8 balls, macro F1 0.597. Budget-100%rho90 wrapper: 77.3 balls, F1 0.492 (Δ -0.1005). **Partially** — budget stopping reaches similar ball counts but split selection without purity differs from rho stopping on some datasets.

## 2. Which budget scaling works best?

Best budget-frac: **budget-50pct-rho90** (F1 0.514, Δ -0.0735). Best sqrt-scaled: **sqrt-k2** (F1 0.507, Δ -0.0852). Overall best non-baseline: **budget-50pct-rho90**.

## 3. Does performance improve over rho90?

Best config **budget-50pct-rho90** macro F1 0.514 vs rho90 0.597 (Δ -0.0735). **No** — rho=0.90 outperforms all budget-only wrappers on macro F1; matching ball count without purity-based split selection is insufficient.

## 4. Does ball count explain gains?

Paired Δ macro F1 (vs rho90) vs n_balls: spearman=-0.191 (n=398). **Weakly** — gains are not fully explained by ball count alone; which balls are split (purity vs budget order) also matters.

## Ball-count correlations

- delta_macro_f1_vs_n_balls: pearson=-0.141, spearman=-0.191 (n=398)
- macro_f1_vs_n_balls_all_configs: pearson=-0.316, spearman=-0.479 (n=398)

## Outputs

- `results.csv`
- `summary.json`

