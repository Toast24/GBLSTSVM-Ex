# Session 9 — Paper evidence: held-out ladder, closed-form test, and a split-seed bug

Date: 19 Nov 2026 · Code: `experiments/analysis/a9_paper_evidence/` · Outputs: `experiments/output/a9_paper_evidence/`

## 0. What this session did

Session 8 left a list of coding steps the paper depends on. This session ran them:

| # | Step | Script |
|---|------|--------|
| 1 | Re-run the A4/A6 ladder with held-out tuning, both coordinate systems, all scatter arms | `run_heldout_ladder.py [--standardize] [--draw k]` |
| 2 | Test the closed form: does ‖M⁻¹ΔM‖ predict θ before solving? | `run_closed_form.py [--standardize]` |
| 3 | Per-dataset cost of `del_ball` | `analyze_a9.py` (from step 1) |
| 4 | θ–η and θ–q figures | `analyze_a9.py` → `figures/` |

Protocol as in Session 8: 17 datasets, seeds 0–9, 70/15/15 split, ρ = 0.9 (sweep ρ ∈ {0.6, 0.7, 0.8, 0.9, 0.95, 1.0} for step 2). The unit is the dataset mean; tests are Wilcoxon with a percentile bootstrap (seed 42, 10k) and Holm correction over the 9 contrasts in each table. The code rule is primary.

Run commands (GBenv; put `Library/bin` on PATH; `OMP_NUM_THREADS=1` avoids the Windows KMeans/MKL issue):

```
python experiments/analysis/a9_paper_evidence/run_heldout_ladder.py [--standardize] [--draw 1|2]
python experiments/analysis/a9_paper_evidence/run_closed_form.py [--standardize]
python experiments/analysis/a9_paper_evidence/analyze_a9.py
```

## 1. Bug: the "deterministic" train/validation split was not deterministic

`experiments/granular_compression_k_common.deterministic_rng_seed` derived seeds with Python's built-in `hash()` on a string. String hashing is salted per process (PYTHONHASHSEED), so **every run drew a different 80/20 selection/validation split**. That function also seeds `train_val_split_seed` and `compression_seed`.

Found because RAW, which has no randomness of its own, picked a different c in 114 of 170 runs between A8-E2 and the first A9 run.

- **Fixed:** the function now uses sha256. Seeds are stable across processes (checked).
- **Affected:** every held-out tuning result since Session 6 (validation-selected K, principled K rule, A8-E2). Each is one random draw and cannot be reproduced exactly. The numbers are not invalid, but any single run's significance can be split luck.
- **Not fixed (old one-off scripts):** `oracle_ball_selection_analysis.py` and `validate_boundary_selection_rho90.py` also use salted `hash()`.
- The two pre-fix A9 runs are kept in `prefix_hash_salted/` and used below as extra split draws.

## 2. Held-out ladder (step 1)

Test macro-F1, code rule, dataset means, standard split draw:

| Arm | Unscaled | Standardized |
|---|---|---|
| RAW = P_FULL | 0.786 | 0.829 |
| D_FULL | 0.775 | 0.813 |
| P_DIAG | 0.765 | 0.808 |
| P_ISO | 0.740 | 0.808 |
| P_L1 | 0.755 | 0.777 |
| D_L1 | 0.746 | 0.746 |
| P_L0 | 0.732 | 0.753 |
| PUB (published) | 0.721 | 0.780 |

Contrasts (Δ F1, Holm p). **Stable** = sign identical in every split draw (5 unscaled, 4 standardized); Holm-significant draws are counted in brackets.

| Contrast | Unscaled | Standardized | Stable? |
|---|---|---|---|
| P_L1 − P_L0 (population) | +0.024 (0.19) | +0.024 (0.28) | sign yes [1/5, 1/4] |
| P_DIAG − P_L1 (diag. scatter) | +0.009 (0.96) | +0.031 (0.29) | sign yes [0, 0] |
| P_FULL − P_DIAG (off-diag.) | +0.021 (0.54) | +0.021 (0.29) | sign yes [0, 0] |
| P_DIAG − P_ISO (where mass goes) | +0.025 (0.24) | 0.000 (0.89) | unscaled only |
| D_FULL − D_L1 (scatter, deletion path) | +0.029 (0.53) | **+0.066 (0.005)** | sign yes [1/5, **4/4**] |
| RAW − D_FULL (exact cost of deletion) | +0.011 (0.54) | **+0.016 (0.026)** | sign yes [0/5, 2/4] |
| RAW − PUB | +0.065 (0.08) | **+0.049 (0.003)** | sign yes [1/5, **4/4**] |

Paper rule: no contrast survives Holm in either scaling (as in Session 8).

**Inferences**
1. **Every rung points the same way in every split draw.** The direction of the ladder (L0 < L1 < DIAG < FULL = RAW) is robust. Individual steps are underpowered with 17 datasets.
2. **Session 8's headline "D_FULL − D_L1 = +0.048, Holm 0.018" (unscaled) was split luck.** It is significant in 1 of 5 unscaled draws. In **standardized** coordinates it is significant in 4 of 4 draws (+0.055 to +0.066).
3. **Standardized coordinates should be primary** for classification as well as angles. Effects there are larger and more stable, and scaling is standard practice for LSTSVM.
4. **RAW beats the published GBLSTSVM by about 5 F1 points** (standardized, 4/4 draws, 15/17 datasets). It is also roughly 700× cheaper (Session 8).
5. "ISO is worse than DIAG" for classification holds only unscaled. As with angles, it is coordinate-dependent and should not be a headline.

## 3. Closed-form test (step 2)

On the partition path, z_arm = (I + M_arm⁻¹ΔM) z_RAW exactly, so ‖z_arm − z_RAW‖/‖z_RAW‖ ≤ q := ‖M_arm⁻¹ΔM‖₂. Here q uses only ball moments and c; no solve is needed.

- **Identity verified:** max residual 2.2e-8 over 6,120 (dataset, seed, ρ, arm, scaling) rows.
- **Bound holds in 100% of rows.** It is loose: median relative error / q = 0.002 (L1, std) to 0.34 (ISO, unscaled).
- **q predicts θ** (Spearman over dataset × ρ cells):

| Arm | std: q | std: η | unscaled: q | unscaled: η |
|---|---|---|---|---|
| L1 | **0.78** | 0.65 | **0.82** | 0.76 |
| ISO | **0.84** | 0.44 | **0.69** | 0.35 |
| DIAG | **0.80** | 0.39 | 0.39 | **0.67** |

Within a dataset, θ rises with q in 15–17 of 17 datasets for every arm (median within-dataset Spearman 0.94–1.0).

**Inferences**
1. The closed form is not only an identity: **its operator norm is a better pre-fit predictor of the angle than η** in 5 of 6 arm × scaling cells. Unlike η, it also covers the ISO and DIAG arms.
2. q ranks *settings*. It does not rank *arms within a setting*: the arm order by q matched the order by θ in only 24% (unscaled) and 49% (std) of cells. The bound is too loose for that.
3. Paper framing: an exact identity, a provable bound, and an empirically strong ranking predictor. Do not claim q predicts the angle value.

## 4. Cost of deletion (step 3)

`table_deletion_cost.csv`: `del_ball` drops **26.4%** of training samples on average (minority 30.2%, majority 28.1%). The range runs from 0% (breast_cancer_wisconsin_original) to 49% (heart).

- Exact cost of deletion (RAW − D_FULL): +0.011 F1 unscaled, +0.016 standardized (Holm 0.026). Largest on heart (+0.08) and credit_approval (+0.05).
- Spearman(dropped fraction, cost) across datasets = 0.46.
- **Correction to Session 8:** "deletion costs about 3 F1 points" came from P_L1 − D_L1 in one split draw. The robust statement is **1–2 F1 points exactly, larger (up to 3) for center-only balls, in standardized coordinates.**

## 5. Recommendations

1. **Make standardized coordinates primary** for angles and classification; report unscaled as robustness.
2. **Report split-draw stability** (sign agreement and significant draws) next to every classification contrast. With 17 datasets, single-draw Holm p-values are not reliable.
3. **Lead the classification evidence with the three robust results:** scatter restoration on the deletion path, the exact cost of deletion, and RAW vs published.
4. **Promote the closed form** to a proposition: the identity, the bound, and q as a pre-fit ranking predictor (Spearman 0.78–0.84 standardized).
5. **Re-run Session 6–7 K-selection results** with the fixed seed before citing any of their p-values.
6. Still open, and not code: the literature review on whether any GB paper states the moment identity, positioning against Nyström and reduced-kernel LSTSVM, and the exactness proof for the regularized, weighted variant.
7. Optional: a large kernel dataset (n ≈ 20k) to settle the compute question; not run here.
