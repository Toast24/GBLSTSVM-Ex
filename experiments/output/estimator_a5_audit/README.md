# A5 — Estimator audit (confirmatory / robustness)

## Purpose

1. **k-invariance (Part A):** Exact moment-preserving L3 on a **complete partition** should match RAW regardless of ball count k. This is a property of linear LSTSVM sufficient statistics, **not** evidence that granular balls help.
2. **RULE_REF rescoring (Part B):** Sensitivity check on A4 fitted hyperparameters (c1/c2 from `estimator_information_ladder/all_runs.csv`); weights recomputed, **no C-grid retuning**.
3. **Collapse (Part C):** A4 collapse definition: `n_balls<=3` OR classifier failure OR non-finite macro-F1.

## Framing

- The k-sweep establishes **partition invariance** for exact L3.
- The empirical phenomenon of interest remains the gap **L0 → L1 → L2** (lossy representations).
- L3=RAW is a **mathematical control**, not the proposed contribution.
- Do not interpret L2 as proving granular balls are necessary; directional scatter claims await later anisotropy experiments.

## Gates

- GATE A: `PASS`
- GATE B: `PARTIAL PASS`
- GATE C: `NEUTRAL`
- OVERALL: `PARTIAL`

## Outputs

See CSV files and `plots/` in this directory.
