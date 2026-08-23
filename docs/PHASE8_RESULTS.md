# Phase 8: Systematic Ablation and Validation Results

Script suite: `experiments/phase8_*.py`  
Outputs: `experiments/output/phase8/`  
Pre-run tests: `python -m pytest tests/ -v` → **59 passed**  
Post-run tests: **59 passed** (frozen implementations unchanged)

---

## 1. Experimental protocol

- Datasets: `diabetes.csv`, `ecoli2.csv`, `vehicle0.csv` (last column = label).
- Split: stratified 70/15/15, `random_state=42` (Experiment 5 uses additional seeds with new splits).
- Validation used for all selection; test untouched until final evaluation.
- Prediction rule: **reference** (`sign(|f2| − |f1|)`).
- Frozen baseline / CW / R / CR implementations not modified.

**Experiments:**

| # | Script | Purpose |
|---|--------|---------|
| 1 | `phase8_granulation_sensitivity.py` | purity ∈ {0.80, 0.85, 0.90, 0.95, 0.98} |
| 2 | `phase8_scaling_ablation.py` | none / StandardScaler / MinMaxScaler |
| 3 | `phase8_hyperparameter_grid.py` | c1 × c2 grid; top-5 val configs tested |
| 4 | `phase8_final_protocol.py` | Combined best protocol; train+val → test once |
| 5 | `phase8_multiseed.py` | Seeds {0,1,2,3,4,42} on selected protocol |
| 6 | `phase8_mechanism_diagnostics.py` | CR vs CW hyperplane / prediction analysis |

---

## 2. Dataset statistics

| Dataset | Samples | Features | Class counts | Imbalance |
|---------|---------|----------|--------------|-----------|
| diabetes | 768 | 8 | 0:500, 1:268 | 1.87 |
| ecoli2 | 336 | 7 | 0:284, 1:52 | 5.46 |
| vehicle0 | 846 | 18 | 0:647, 1:199 | 3.25 |

Label mapping: 0 → −1, 1 → +1. No missing values; no duplicates dropped.

---

## 3. Granulation results (Experiment 1)

**Key finding:** Lower `purity_threshold` reduces purity/reliability saturation and increases ball heterogeneity, but does not consistently improve test Macro F1.

| Dataset | Best val Macro F1 (Clean) | Best purity | Notes |
|---------|---------------------------|-------------|-------|
| diabetes | ~0.42 (default 0.95) | 0.95 | At 0.80 some runs fail (empty ball class) |
| ecoli2 | ~0.81 | 0.95 | All thresholds similar |
| vehicle0 | ~0.68 | 0.95 | Modest variation |

At **purity=0.80**, vehicle0 CR model shows mean reliability ≈ 0.92 vs ≈ 0.99 at 0.95 (reliability more active). At purity=0.80 on diabetes, granulation can fail (no balls for one class).

**Lowering purity activates reliability numerically** but **does not translate into CR outperforming CW** on test Macro F1 in this grid.

---

## 4. Scaling results (Experiment 2)

| Dataset | Best scaling (val Clean Macro F1) | Effect |
|---------|-----------------------------------|--------|
| diabetes | **minmax** | Fixes majority-class collapse under defaults |
| ecoli2 | none ≈ standard ≈ minmax | Minimal difference |
| vehicle0 | none | Slight edge over scalers |

**MinMaxScaler on diabetes** is the largest single protocol change: validation Macro F1 jumps from ~0.39 (no scaling, Phase 7) to ~0.75+ when combined with tuned c1/c2 in Experiment 4.

Scaling materially changes granulation on diabetes (ball radii/purities on normalized [0,1] features).

---

## 5. Hyperparameter results (Experiment 3)

Grid: c1 ∈ {0.01, 0.1, 1, 10, 100, 1000}, c2 ∈ {1e-5 … 10} (49 combinations × 4 models × 3 datasets on validation).

**Best validation Macro F1 per model (representative):**

| Dataset | Clean best (c1, c2) | CW best | CR best |
|---------|---------------------|---------|---------|
| diabetes | (0.1, 1.0) ~0.76 | similar | similar |
| ecoli2 | (100, 10) ~0.84 | similar | similar |
| vehicle0 | (0.1, 0.1) ~0.97 | similar | similar |

Defaults (c1=100, c2=0.01) are **not** optimal on these datasets. Tuning changes conclusions for **diabetes** dramatically; **ecoli2** and **vehicle0** conclusions about CW vs CR remain similar.

Same/similar (c1, c2) often selected across Clean/CW/R/CR when ranked by validation Macro F1.

---

## 6. Final selected configurations (Experiment 4)

Selection: grid over scaling × purity × top-5 (c1,c2) from Exp 3; **validation Clean Macro F1** as criterion; train on **train+validation**; single test evaluation.

| Dataset | Scaling | Purity | c1 | c2 | Val Macro F1 (Clean) |
|---------|---------|--------|----|----|------------------------|
| diabetes | minmax | 0.85 | 0.1 | 1.0 | 0.756 |
| ecoli2 | none | 0.95 | 100 | 10 | 0.838 |
| vehicle0 | none | 0.80 | 0.1 | 0.1 | 0.968 |

---

## 7. Final protocol — test results

| Dataset | Model | Acc | Bal Acc | Macro F1 | Min Recall | G-mean | AUROC |
|---------|-------|-----|---------|----------|------------|--------|-------|
| diabetes | Clean | 0.629 | 0.614 | 0.608 | 0.561 | 0.612 | 0.690 |
| diabetes | CW | 0.621 | 0.629 | 0.611 | **0.659** | 0.629 | 0.717 |
| diabetes | R | 0.621 | 0.607 | 0.601 | 0.561 | 0.605 | 0.690 |
| diabetes | CR | 0.612 | 0.623 | 0.604 | **0.659** | 0.622 | 0.720 |
| ecoli2 | Clean | 0.902 | 0.891 | **0.838** | 0.875 | 0.891 | 0.884 |
| ecoli2 | CW | 0.882 | 0.879 | 0.813 | 0.875 | 0.879 | 0.901 |
| ecoli2 | R | 0.902 | 0.891 | **0.838** | 0.875 | 0.891 | 0.884 |
| ecoli2 | CR | 0.882 | 0.879 | 0.813 | 0.875 | 0.879 | 0.901 |
| vehicle0 | Clean | 0.921 | 0.914 | 0.896 | 0.900 | 0.914 | 0.975 |
| vehicle0 | CW | 0.929 | **0.931** | **0.907** | **0.933** | **0.931** | 0.976 |
| vehicle0 | R | 0.913 | 0.897 | 0.884 | 0.867 | 0.897 | 0.973 |
| vehicle0 | CR | 0.929 | **0.931** | **0.907** | **0.933** | **0.931** | 0.975 |

### Test deltas vs Clean (Macro F1)

| Dataset | CW − Clean | R − Clean | CR − Clean | CR − CW |
|---------|------------|-----------|------------|---------|
| diabetes | +0.003 | −0.008 | −0.005 | −0.007 |
| ecoli2 | −0.025 | 0.000 | −0.025 | 0.000 |
| vehicle0 | +0.011 | −0.012 | +0.011 | 0.000 |

**CR does not improve Macro F1 over CW on any dataset.** On vehicle0 and ecoli2, CR ≡ CW (identical test metrics and predictions). On diabetes, CR is slightly worse than CW on Macro F1.

---

## 8. Multi-seed stability (Experiment 5)

Protocol fixed per dataset; seeds {0, 1, 2, 3, 4, 42}; new stratified splits per seed.

| Dataset | mean(CW−Clean) Macro F1 | std | mean(CR−CW) Macro F1 | std |
|---------|---------------------------|-----|----------------------|-----|
| diabetes | **−0.029** | ~0.04 | **−0.007** | ~0.02 |
| ecoli2 | **−0.008** | ~0.01 | **0.000** | 0.000 |
| vehicle0 | **−0.017** | ~0.02 | **+0.010** | ~0.01 |

CW does **not** stably beat Clean across seeds on Macro F1 (mean deltas negative on all datasets). CR−CW mean is near zero or negative; **no stable CR advantage**.

On **vehicle0**, seed-level CR and CW test Macro F1 are often identical; when they differ, CR does not consistently win.

---

## 9. Mechanism diagnostics (Experiment 6)

| Dataset | Mean reliability | frac ≥0.99 | frac CR≠CW weights | test pred CR≠CW | ‖w1_CR−w1_CW‖ |
|---------|------------------|------------|--------------------|-----------------|----------------|
| diabetes | 0.964 | 0.879 | 0.121 | 0.009 | 0.128 |
| ecoli2 | 0.995 | 0.882 | 0.118 | 0.000 | 0.004 |
| vehicle0 | 0.923 | 0.741 | 0.259 | 0.000 | 0.458 |

- **vehicle0** has the most active reliability (26% balls &lt;0.95, mean 0.92), yet **CR ≡ CW** on test predictions and Macro F1.
- Hyperplane differences exist (especially vehicle0) but are **too small or aligned** to change discrete predictions.
- On ecoli2, R ≡ Clean (R reduces to baseline due to saturation).

**Reliability influences weights but rarely changes the learned classifier enough to alter predictions or beat CW.**

---

## 10. Scientific interpretation (Experiment 7)

1. **Does CW consistently improve over Clean?** No. Under tuned protocol, CW helps minority recall on diabetes and metrics on vehicle0 test, but **multi-seed mean Macro F1 is lower for CW than Clean** on all three datasets.

2. **Does R ever improve over Clean?** Rarely. On vehicle0 final test, R is worse; on ecoli2 R ≡ Clean (saturated reliability).

3. **Does CR improve over CW?** **No.** CR ≤ CW on Macro F1 everywhere; often identical.

4. **Does lowering purity_threshold activate reliability?** Yes numerically (lower mean reliability, more balls &lt;0.95), especially vehicle0 at 0.80. Does not yield CR &gt; CW.

5. **Does feature scaling materially change granulation?** **Yes on diabetes** (essential to avoid collapse and enable learning). Minor on ecoli2/vehicle0.

6. **Does hyperparameter tuning change the conclusion?** **Yes for diabetes** (null collapse was largely a regime problem). **No for the CR vs CW question** — tuning does not make CR beat CW.

7. **Does CR remain equal/worse across seeds?** Equal or worse; mean(CR−CW) ≤ +0.01 on vehicle0 only, negative on diabetes.

8. **Is reliability influencing hyperplanes?** Sometimes (non-zero ‖w_CR−w_CW‖), but **not enough to change predictions** in final protocol.

9. **Stable enough for further research?** Gains are small, seed-dependent, and **not attributable to reliability**. Class weighting shows occasional minority-recall benefit but unstable Macro F1.

10. **Should the reliability formulation be modified?** Evidence supports **revising or replacing the current reliability mechanism** rather than more tuning of c1/c2/purity alone. The null result is **not solely** a poor experimental regime (scaling + tuning fixed diabetes), but reliability still **fails to add value beyond CW**.

---

## 11. Limitations

- Validation-based selection on small validation folds (especially ecoli2).
- KMeans in reference granulation lacks full determinism beyond NumPy seed.
- Hyperparameter grid excluded joint tuning of purity/scaling in Exp 3 (addressed in Exp 4).
- Some granulation settings fail (empty ball class) — recorded, not silently dropped.

---

## 12. Recommendation for Phase 9

1. **Reliability redesign** — current formula saturates or changes weights without predictive impact; consider impurity-aware terms that bind to decision boundaries, or reliability defined on **held-out ball samples** rather than geometry alone.

2. **Focus on minority-stable objectives** — if CW helps minority recall, optimize and report **minority recall / G-mean** with cross-validation rather than Macro F1 alone.

3. **Scaling as standard protocol for raw-feature datasets** — document MinMax/Standard as part of protocol for diabetes-like data, separate from frozen baseline comparison.

4. **Stronger baselines** — compare against SMOTE + SVM or cost-sensitive SVM to contextualize small CW gains.

5. **Do not pursue CR tuning** until reliability produces **prediction-level** separation from CW on validation.

---

## Output files

| File | Description |
|------|-------------|
| `granulation_sensitivity.csv/json` | Exp 1 |
| `scaling_ablation.csv/json` | Exp 2 |
| `hyperparameter_grid.csv/json` | Exp 3 |
| `final_protocol.csv/json` | Exp 4 |
| `multiseed.csv/json` | Exp 5 |
| `mechanism_diagnostics.csv/json` | Exp 6 |
| `reliability_vs_purity.png` | Plot |
| `scaling_comparison.png` | Plot |
| `macro_f1_comparison.png` | Plot |
| `cr_vs_cw_delta.png` | Plot |
| `seed_stability.png` | Plot |
| `reliability_distribution.png` | Plot |
