# A6 — Scatter mechanism experiment

## Guardrails

1. L3=RAW is partition-invariant (A5); mathematical control only.
2. A5 showed L2−L1 is **prediction-rule sensitive** (RULE_REPO vs RULE_REF).
3. A6 tests whether scatter gains are **directional** vs **isotropic magnitude / ridge**.
4. FULL beating ISOTROPIC alone is **not** sufficient; compare to **RIDGE_TRACE_MATCH**.
5. Consistent FULL > DIAG and FULL > ISOTROPIC supports directional second-order information.
6. If ISOTROPIC ≈ trace-matched ridge, do **not** claim covariance structure is necessary.

## Ridge placement

Per class c, L1 moments accumulate as in A4. Ridge/scatter add to `H_c[:d,:d]`:
- FULL/DIAG/ISOTROPIC: sum_j S_j^(variant) per ball.
- RIDGE_TRACE_MATCH: add `lambda_c I_d` with `lambda_c = (sum_j tr(S_j))/d`.

## Tuning

- **MODE_A_SHARED_C**: c1/c2 from L1 validation grid (A4-comparable); RIDGE_VAL_TUNED tunes lambda multiplier only.
- **MODE_B_PER_ARM_C**: independent c1/c2 per arm; RIDGE_VAL_TUNED also tunes lambda on validation.

## Classification

`NO_CLEAR_SCATTER_ADVANTAGE`

Primary inferential unit: **dataset** (mean over seeds). Exploratory sequence A2–A6; report full comparison matrix.
