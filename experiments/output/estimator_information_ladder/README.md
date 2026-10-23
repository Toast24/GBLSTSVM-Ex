# A4 — Estimator-information ladder

## Prediction rule

**Primary:** RULE_REPO = `sign(|f2(x)| - |f1(x)|)` via `predict_twin_hyperplanes(..., rule='reference')`.

RULE_REPO is locked for continuity with Phase 7/8 and audit A3. The published normalized-distance rule is **not** primary here (A3 showed ~14% pooled test disagreement and different collapse).

## Research question

Does granular-ball compression look weak because the estimator underuses per-ball information (population, scatter)?

## Arms (radius target shift **absent**)

| Arm | Design | RHS |
|-----|--------|-----|
| RAW_LSTSVM | per-sample L3 moments | n_j u (unit sample) |
| L0_CENTER_UNIT | sum u u^T | u |
| L1_POPULATION | sum n_j u u^T | n_j u |
| L2 / L3 | sum n_j u u^T + blockdiag(S_j,0) | n_j u |
| MATCHED_K_SAMPLES | RAW on stratified k subsample, k = baseline ball count | (sensitivity) |

Twin solve follows `src/ladder_gblstsvm.py` structure with `ridge_scale(reg_base=1e-4)`.

## Granulation

- **Baseline path (performance):** rho=0.90, delete balls with n<2 (baseline-compatible).
- **Partition path (identity):** rho=0.90, min_ball_count=0, full sample coverage.

Do not claim L3=RAW on the baseline-deletion path.

## Protocol

- 17 datasets x seeds 0–9 (170 runs per arm).
- 70/15/15 stratified split (`phase8_common`); test never used for c1/c2 tuning.
- c1/c2: `C_GRID` from `rebaseline_common`, tuned per arm on internal 20% train validation (`make_train_val_partition`), refit on full train.
- No scaling (matches principled-complexity loader).

## Identity checks

- H_class, g_class from samples vs partition L3 balls (tol 2e-5 on max abs error).
- L3 vs RAW parameters on partition path (max abs param err reported in `identity_audit.csv`).
- Scatter split identity on root ball (logged in `scatter_identity_audit.csv`).

## Statistics

Dataset-level mean over 10 seeds; Wilcoxon + bootstrap CI across 17 datasets.

## Verdict (auto-generated)

See `summary.json`. Latest run: **B — PARTIAL ESTIMATOR-MISMATCH EVIDENCE**.

## Storage note

`storage_analysis.csv` uses augmented moment storage ~ k*(d+1)^2 vs n*d. All 17 datasets have k*(d+1)^2 > n*d at mean baseline k — exact scatter storage is not universally cheaper than raw samples.

## Outputs

`all_runs.csv`, `summary_by_method.csv`, `dataset_summary.csv`, `paired_comparisons.csv`, `identity_audit.csv`, `scatter_identity_audit.csv`, `storage_analysis.csv`, `failure_analysis.csv`, `hyperparameter_summary.csv`, `plots/`.
