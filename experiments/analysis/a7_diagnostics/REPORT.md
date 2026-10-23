# A7 diagnostics REPORT

## Entry point

```text
python experiments/analysis/a7_diagnostics/run_a7_diagnostics.py --from-csv
python experiments/analysis/a7_diagnostics/run_a7_diagnostics.py --from-csv --angles
python experiments/analysis/a7_diagnostics/run_a7_diagnostics.py --dump-params  # calls A6 --dump-params
```

Bootstrap seed **42**, BCa **10,000** resamples (`contrast_distributions.csv`).

## Task 0

See `inventory.md`. `runs.csv` has `c1`, `c2` per row; no `w1`/`w2` in CSV. Params in
`experiments/output/estimator_a6_scatter_mechanism/params/` (2,380 `.npz` + `manifest.csv`).

## Task 1 (`iso_railing_*`)

- MODE A: all arms share L1-tuned `c1`/`c2` → identical railing rates per arm (c1@min ≈ 0.265).
- MODE B ISOTROPIC: c1@min ≈ 0.165, c2@min ≈ 0.135 (vs FULL c1@min ≈ 0.171).
- ISOTROPIC−L1 mean delta: −0.158 (A+REPO), −0.028 (B+REPO).
- Dose-response Spearman (log ridge/c1 vs ISOTROPIC−L1): all |rho| < 0.25, p > 0.34.

## Task 2 (`contrast_distributions_*`)

- Collapse rate **0.0** all arms/rules under derived definition.
- Holm family: **32** tests (8 contrasts × 2 modes × 2 rules); **5** survive Holm α=0.05,
  all MODE A or B with **RULE_REPO**: ISOTROPIC−L1, FULL−ISOTROPIC, FULL−RIDGE (×2 modes).
- MODE B FULL−ISOTROPIC: mean +0.056, median +0.039, Wilcoxon p=0.0011 (REPO); p=0.109 (REF).
- Per-dataset vectors: `contrast_distributions.md` § FULL−ISOTROPIC MODE B.

## Task 3 (`angle_to_raw_*`)

- ISOTROPIC / RIDGE_TRACE_MATCH: mean θ1 ≈ **52.5°** vs RAW; FULL ≈ **23.1°**; L1 ≈ **37.8°**.
- ISOTROPIC `||w||/||w_RAW||` mean ≈ **2.68** (w1); FULL ≈ **6.79**.
- Pairwise θ1 contrasts in `angle_to_raw.md`.

## Task 4 (A6 script only)

Auto-label keyed to **MODE_B_PER_ARM_C** + **RULE_REPO**; printed as
`A6_AUTO_LABEL_MODE_B_RULE_REPO: MODE_B_RULE_REPO_*`.

## Schema corrections vs brief

- Arm names use `_POPULATION` / `_SCATTER` suffixes.
- No `estimator_a4_*` output tree; A4 ladder is `estimator_information_ladder/`.
