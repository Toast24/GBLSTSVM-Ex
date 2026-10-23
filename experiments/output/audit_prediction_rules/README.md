# A3 — Prediction rule audit

## Rules
- **RULE_REPO**: sign(|f2(x)| - |f1(x)|); +1 if |f2|>|f1|, -1 if |f2|<|f1|, 0 if equal
- **RULE_REFERENCE (paper)**: argmin_i |f_i(x)|/||w_i||; predict +1 if d1<=d2 else -1 (ties -> +1)

## Mathematical definitions
- f_i(x) = w_i^T x + b_i (bias included in the affine score, not in ||w_i||)
- w_i are the weight vectors from solve_cw_gblstsvm_systems (bias b_i separate)
- ||w_i|| = Euclidean L2 norm of w_i only; bias excluded from normalization
- w1,b1: first twin subproblem (positive ball centers A, opposite class B + R-)
- w2,b2: second twin subproblem (negative ball centers B, opposite class A + R+)

## Class / sign convention
- Binary labels mapped to +1 (minority or CSV 1) and -1 (majority or CSV 0)
- predictions 0 mapped to -1 at metric time (phase8_common.evaluate)
- Ambiguity: None for this implementation; rules map directly to src predict_twin_hyperplanes.

## Protocol
- 17 discoverable datasets × seeds 0–9 (170 pairs).
- Split: 70% train / 15% val / 15% test (stratified, `phase8_common.stratified_three_way_split`).
- Granulation: rho=0.90 (`validation_selected_k_common.REPRESENTATION`).
- Fit: single twin-hyperplane solve per pair (CW + Gaussian weights, fixed c1/c2) — same as `fit_gaussian_with_predictions` / FULL_RHO90 path; **not refitted** between rules.
- Predictions: both rules applied to identical (w1,b1,w2,b2).

## Runs
- 170 dataset×seed fits × 2 rules = 340 run-level metric rows.

## Failure / collapse
- Collapse: n_balls<=3, classifier failure, non-finite macro F1, or single predicted class (`validation_selected_k` policy).
- Status `finite` / `nan` / `exception` preserved; metrics not silently zeroed in primary tables.
- Secondary `dataset_comparison_views.csv`: finite-only vs failure-aware (zero on collapse).

## Primary statistics
- Unit of inference: **dataset** (mean delta macro F1 across 10 seeds).
- Wilcoxon signed-rank on dataset means; bootstrap 95% CI over datasets.

## Deviations from prior scripts
- Does not run compression / K-selection; uses full rho90 ball set only.
- Does not retune c1/c2 or scale features (matches principled-complexity FULL_RHO90 loader path).

## Key synthetic diagnostic
- Reproduced disagree_rate: 0.2900 (target ~0.29)

## Pooled benchmark test disagreement
- 0.14086516336861482

## Dataset-level delta macro F1 (reference − repo)
- mean=0.06455333960634103, median=0.029970674184856372, W/T/L=10/0/7, Wilcoxon p=0.079681396484375

## Failure rates
- RULE_REPO: collapse_rate=0.200, finite=170/170
- RULE_REFERENCE: collapse_rate=0.106, finite=170/170

## Seed-level (descriptive only)
{
  "mean_disagree_rate": 0.13613318752196912,
  "median_disagree_rate": 0.10085708590791305,
  "std_disagree_rate": 0.12571610614532064
}