# Session 8 — Angle-pipeline gate, clean compression geometry, and paper strategy

Date: 29 Sep 2026 · Code: `experiments/analysis/a8_angle_gate/` · Outputs: `experiments/output/a8_angle_gate/`

## 0. Plan (ideation)

The thesis after A7 rests on the angle result, so the order of work was gate first, then the claims that depend on it:

| # | Step | Question | Decides |
|---|------|----------|---------|
| E0 | **Gate** | Partition-preserving FULL at c matched to RAW, through the A7 angle definition: is θ ≈ 0? What produced A7's 23°? | Whether any angle claim stands |
| E1 | Clean geometry + dose–response | On the partition path at matched c, what are θ(L0/L1/DIAG/ISO)? Does θ_L1 track discarded within-ball variance across ρ? | The "directional, quantifiable error" claim |
| E2 | Held-out classification ladder | A4/A6 tuned c on rows that were inside the fitted balls. Do the classification conclusions survive leak-free tuning? Cost vs RAW? | The supporting classification evidence; the linear cost argument |
| E3 | A3 norm ratio | Does ‖w₁‖/‖w₂‖ explain prediction-rule disagreement? | The A3 methodological finding |
| E4/E5 | Kernel pilot + scaling | Does the ladder replicate in feature space? Do granular balls earn their keep as the compression device, and where does compression beat RAW on compute? | Critique paper vs. method paper |

Run commands (GBenv interpreter; put `$CONDA_PREFIX/Library/bin` on PATH on Windows):

```
python experiments/analysis/a8_angle_gate/run_angle_gate.py              # E0 unscaled (+A6 repro)
python experiments/analysis/a8_angle_gate/run_angle_gate.py --standardize
python experiments/analysis/a8_angle_gate/run_angle_gate.py --sweep      # E1 rho sweep
python experiments/analysis/a8_angle_gate/run_linear_heldout.py          # E2
python experiments/analysis/a8_angle_gate/run_kernel_pilot.py --seeds 5  # E4
python experiments/analysis/a8_angle_gate/run_kernel_scaling.py          # E5
python experiments/analysis/a8_angle_gate/analyze_a8.py                  # all tables
```

Protocol: 17 datasets (all now load; see §6), seeds 0–9 (kernel: 0–4), 70/15/15 split, unit of inference = dataset mean, Wilcoxon + BCa-free percentile bootstrap (seed 42, 10k), Holm within each table.

## 1. E0 — Gate: **PASSED**

The pipeline reproduces A6's saved RAW parameters to ≤1.3e-10 (170/170) and FULL parameters to machine precision on 152/170 runs. The 18 exceptions are congressional_voting_records and tic_tac_toe: all-binary one-hot data where KMeans tie-breaking under a different sklearn version yields different balls. That is a granulation difference, not a pipeline bug.

| Arm (A7 angle def.) | c source | θ vs RAW | θ vs RAW-on-retained |
|---|---|---|---|
| FULL, **partition** path | A6 shared c | **0.000°** (rel. param err 0) | 16.4° |
| FULL, **partition** path | RAW held-out c | **0.000°** | 15.0° |
| FULL, deletion path | A6 shared c | 16.8° | **0.000°** |
| FULL, deletion path | RAW held-out c | 15.5° | **0.000°** |

**Decomposition of A7's θ_FULL = 23°:** it contains no granular-ball geometry at all.
- About 16° comes from **sample deletion.** The baseline path drops balls with n<2, which removes a mean of **26% of training samples** (up to 49% on heart, 47% messidor, 46% german). FULL on the deletion path equals RAW-on-retained-samples exactly.
- The rest (MODE B, 30°) comes from **c mismatch**: each arm tunes its own c, RAW uses L1's c.

Consequence: every A6/A7 arm comparison "vs RAW" conflated compression with deletion. The deletion step is itself a finding, because the published pipeline silently discards a quarter of the training set.

## 2. E1 — Clean compression geometry (partition path, c matched to RAW)

Dataset-mean θ vs RAW, (θ₁+θ₂)/2:

| | FULL | DIAG | L1 | L0 | ISO |
|---|---|---|---|---|---|
| unscaled, mean / median | 0 / 0 | 18.3 / 7.3 | 23.6 / 10.3 | 28.8 / 17.2 | 40.6 / 42.2 |
| standardized, mean / median | 0 / 0 | 28.8 / 21.2 | 38.5 / 43.2 | 46.2 / 49.5 | 36.8 / 29.3 |

Paired contrasts (Holm over 6):
- **L0 > L1** (population helps): unscaled +5.2° (Holm p=0.019); standardized +7.7° (0.047). Robust.
- **DIAG < ISO**: −22.3° (0.001); −8.0° (0.0006). Robust: *where* the mass goes matters, not just how much.
- **L1 vs DIAG**: +5.3° (0.24); +9.7° (0.13). Direction consistent, not significant after Holm.
- **L1 vs ISO**: −17.0° (raw p=0.02, Holm 0.08) unscaled, but **+1.7° (p=0.58) standardized.** "Isotropic mass is worse than restoring nothing" is **coordinate-dependent and should be dropped** as a headline. The robust statement is DIAG/FULL ≫ ISO.

**Dose–response (the "quantifiable" claim).** Define η = tr(Σⱼ Sⱼ) / tr(S_class), the fraction of class variance that center-only representation discards.
- Across ρ ∈ {0.6,…,1.0}, mean η falls from 0.81 to 0.12 and θ_L1 falls from 62° to 17°.
- Spearman(θ_L1, η) over (dataset, ρ) cells = **0.75 (p=2e-19)**. The within-dataset correlation is positive in **17/17** datasets (median 0.94).
- At ρ=0.9: dataset-level ρ=0.58 (unscaled) and **0.84** (standardized, p<1e-4).

**Why ISO hurts.** θ_ISO tracks the anisotropy of the pooled within-ball scatter, ‖S_W − (tr S_W/d)I‖_F/‖S_W‖_F: Spearman 0.75, p=0.0005, standardized. This replaces the refuted over-regularization story. Isotropic mass penalizes low-variance directions, which are exactly the directions a twin hyperplane normal wants to occupy.

**Exact form of the error.** With Mz = r the (ridged) normal equations, M_FULL = M_L1 + ΔM, where ΔM is the within-ball scatter blocks (plus the ridge difference). Because M_L1 z_L1 = r = M_FULL z_FULL, it follows exactly that z_L1 = (I + M_L1⁻¹ ΔM) z_FULL. The center-only solution is the RAW solution pushed through a known linear map, and the size of that map is governed by η.

Note also: **ISOTROPIC_SCATTER ≡ RIDGE_TRACE_MATCH by construction**, since Σⱼ(tr Sⱼ/d)I = λ_c I. A6/A7 reported them as two arms, but they are one piece of evidence.

## 3. E2 — Held-out-tuned linear ladder (17 datasets × 10 seeds)

**Protocol flaw fixed.** A4 `tune_c1_c2` and A6 `_val_metrics` fit on balls built from all of X_train, then score `X_train[val_idx]`. That is in-sample validation. Here each arm is re-granulated and fit on the selection rows only, scored on the disjoint validation rows, then refit on the full train split.

Test macro-F1, RULE_REPO (Holm over 8):

| Arm | F1 | vs RAW | Holm p |
|---|---|---|---|
| RAW | 0.785 | — | — |
| D_FULL (= RAW on retained) | 0.775 | −0.011 | 0.26 |
| P_L1 (partition, population) | 0.757 | −0.028 | 0.18 |
| D_L1 | 0.727 | −0.059 | **0.003** |
| P_L0 | 0.725 | −0.060 | **0.03** |
| PUB (published GBLSTSVM) | 0.714 | −0.071 | **0.03** |

- **D_FULL − D_L1 = +0.048 (Holm 0.018, 14/17 wins).** Under leak-free tuning, restoring scatter *does* improve classification over population-only. A6's "no clear scatter advantage" was plausibly a tuning artifact. This is RULE_REPO only.
- P_L1 − D_L1 = +0.031 (Holm 0.019): deletion itself costs accuracy.
- P_L1 − P_L0 = +0.032 (Holm 0.028): population weights help.
- **Under the paper rule, nothing separates** (all arms 0.67–0.70, every Holm p ≥ 0.44). The rule effect on RAW (0.785 vs 0.685) is larger than any representation effect.

**Linear cost:** granulation ≈ 180 ms per fit vs 0.26 ms for the entire RAW moment + solve, about 700×. In the linear case GBLSTSVM is strictly dominated by RAW.

## 4. E3 — A3 norm ratio

Rule disagreement (paper vs repo) against |log ‖w₁‖/‖w₂‖|, run level:
- Pooled across arms: Spearman **0.58** (unscaled, n=2040) and **0.76** (standardized).
- RAW alone: 0.39 / 0.67.

Larger imbalance also shifts F1 toward the repo rule (ρ ≈ −0.2 to −0.3, p<0.02). The earlier dataset-level test (ρ=0.27, n.s., `audit_prediction_rules/norm_geometry.csv`) was underpowered by averaging to 17 points. The residual is the density of test points in the wedge |f₁|/|f₂| ∈ (1, r), which is tautological given r.

## 5. E4/E5 — Kernel pilot (17 datasets × 5 seeds, RBF, standardized, held-out tuned)

| Arm | test F1 | agree w/ K_RAW (at RAW hp) | fit s (n≥500) |
|---|---|---|---|
| K_RAW | 0.843 | 1.000 | 0.20 |
| K_GB_EXACT (GB basis, exact moments) | 0.839 | 0.964 | 0.44 |
| K_KM_EXACT (k-means basis, same k) | 0.839 | 0.972 | 0.49 |
| K_RND_EXACT | 0.831 | 0.945 | 0.43 |
| K_GB_MEAN (mean embedding, no feature scatter) | 0.821 | 0.932 | 0.47 |
| K_GB_L1 (standard GB kernel, population) | 0.822 | 0.908 | 0.42 |
| K_GB_L0 | 0.797 | 0.879 | 0.42 |

- **The ladder replicates in feature space.** L0 < L1 ≈ MEAN < EXACT ≈ RAW.
  - EXACT − L1 = +0.017 F1 (Holm 0.03).
  - EXACT − MEAN = +0.018 (Holm 0.03), so feature-space *second-order* structure matters, not only the mean embedding.
  - Agreement with RAW orders the same way; every step is Holm-significant.
- **Granular balls add nothing as the compression device.**
  - k-means landmarks at the same k: F1 +0.0005 (n.s.), agreement 0.008 *higher* than GB (Holm 0.046).
  - Random landmarks are within 0.009 F1 (n.s.).
- **Compute.** At benchmark sizes (n ≤ 3.2k), granulation alone costs more than the full O(n³) K_RAW fit. On overlapping synthetic data (d=20), ρ=0.9 granulation leaves k/n ≈ 0.65–0.73, and GB is slower than K_RAW until n≈8000, where it saves about 10%. GB's k is data-determined and largest precisely on hard problems.

## 6. Housekeeping

- `experiments/dataset_registry.py`: `np.issubdtype` → `pd.api.types.is_numeric_dtype`. Under pandas 3 the old check raised on `StringDtype`, silently dropping 5 of 17 datasets (chronic_kidney_disease, congressional_voting_records, credit_approval, german_credit, tic_tac_toe).

## 7. Where the thesis stands

**Survives / strengthened**
1. Sufficient-statistic identity (L3 ≡ RAW, any partition), now verified at the angle level (θ = 0.000°).
2. Center-only compression is a *directional*, *quantifiable* approximation error. θ_L1 is monotone in discarded within-ball variance η (17/17 datasets), with an exact closed form z_L1 = (I + M_L1⁻¹ΔM) z_RAW.
3. Where restored mass goes matters: DIAG/FULL ≫ ISO (robust to scaling), explained by scatter anisotropy.
4. Under leak-free tuning, scatter restoration improves classification (RULE_REPO).
5. The prediction-rule discrepancy is real and is driven substantially by ‖w₁‖/‖w₂‖.

**Must be revised or dropped**
- A7's specific numbers (FULL 23°, L1 38°, ISO 52°): they mix in 26% sample deletion and c mismatch. Replace with the §2 partition-path table.
- "ISO is worse than restoring nothing": coordinate-dependent.
- "No classification advantage for scatter" (A6): an artifact of in-sample tuning.
- Any A4/A6 classification number: in-sample tuning. Cite §3 instead.

**New findings to add**
- The published pipeline's n<2 deletion discards about 26% of training data and costs about 3 F1 points on its own.
- The paper-vs-repo rule gap (≈0.10 F1 on RAW) exceeds every representation effect.

## 8. Strategic recommendation

Write the **analytical / critique paper**, with the kernel pilot as a section rather than the basis of a method paper.

- **Linear case (the core).** GBLSTSVM is a lossy, deletion-contaminated approximation of an estimator that is about 700× cheaper to compute exactly. The loss is directional, has a closed form, and scales with discarded within-ball variance. The published rule and the reference code disagree, driven by the norm ratio.
- **Kernel case (a supporting section).** The same ladder holds in feature space: exact moments on a reduced basis recover RAW, center-only does not. But the reduced basis need not be granular balls; k-means or random landmarks do as well, and granulation is the dominant cost. This pre-empts the "it matters in the kernel case" objection instead of opening a method paper we could not defend.
- A method paper would need something GB-specific that beats Nyström / reduced-kernel LSTSVM with k-means landmarks. Nothing in E4/E5 suggests that exists.

**Next steps**
1. Re-run A4/A6 with the held-out protocol for the paper tables.
2. Report angles in standardized coordinates as primary (unscaled as a robustness check).
3. Plot θ_L1 vs η across ρ.
4. Test the closed-form prediction: predict θ from ‖M_L1⁻¹ΔM‖ before fitting.
5. Optional: one larger kernel dataset (n ≈ 20k) to close the compute question empirically.
