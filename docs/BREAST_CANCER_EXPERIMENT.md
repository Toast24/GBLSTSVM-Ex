# Phase 6: Breast Cancer Train / Validation / Test Experiment

This document records the first real-dataset comparison of frozen GBLSTSVM variants. Hyperparameters were **not** tuned on validation or test. No frozen model file was modified.

Script: `experiments/validate_breast_cancer.py`  
Outputs:

- `experiments/output/breast_cancer_train_val_test.json`
- `experiments/output/breast_cancer_train_val_test.csv`
- `experiments/output/breast_cancer_ball_diagnostics.csv`

Existing tests: `python -m pytest tests/ -v` → **59 passed** (unchanged), then the experiment was executed.

---

## 1. Dataset

File: `Data/Data/breast_cancer.csv` (no header).

| Property | Value |
|----------|--------|
| Shape | 286 rows × 10 columns |
| Features | columns 0–8 (A:I), all numeric `float64` |
| Label | column 9 (J), values `{0, 1}` |
| Class counts | `0`: 201 (majority), `1`: 85 (minority) |
| Missing values | 0 |
| Duplicate rows | 14 (retained; not dropped) |
| Non-numeric features | none |

Feature columns already have mean ≈ 0 and standard deviation ≈ 1 (pre-standardized in the shipped CSV). Labels are mapped at fit time as **0 → −1**, **1 → +1**, matching the frozen GBLSTSVM API.

Minority class from **training** labels: **+1**.

---

## 2. Split strategy

Seed **42**. Stratified three-way split via two `sklearn.model_selection.train_test_split` calls (`test_size=0.30`, then `0.50` on the hold-out):

| Split | n | +1 | −1 |
|-------|---|----|----|
| Train | 200 | 59 | 141 |
| Validation | 43 | 13 | 30 |
| Test | 43 | 13 | 30 |

Raw rows are split **before** any model statistics. Validation is used only to confirm that models train and predict; it is **not** used to pick a winner or to construct the test set.

---

## 3. Preprocessing

**None.**

`MinMaxScaler` is imported in `gen_ball.py` and `GBLSTSVM.py` but is unused. `main.py` loads CSV values with `np.loadtxt` and does not scale. Frozen Phase 1–5 estimators do not scale internally. The CSV is already approximately standardized. Applying MinMaxScaler here would be a **new** pipeline step, so it was not introduced.

Leakage controls:

- split raw data first;
- no scaler fitted on val/test;
- granular balls from training rows only;
- class weights from training labels only;
- reliability radius scale from training balls only;
- test unused until final evaluation.

---

## 4. Models

| Name | Implementation | Notes |
|------|----------------|-------|
| Original/reference GBLSTSVM | unmodified `gen_ball.py` via `src/reference_pipeline.py`; predictions match `LSTWGBSVM` `sign(\|f2\|−\|f1\|)` | Executed cleanly |
| Clean GBLSTSVM | frozen `src/gblstsvm.py` | Primary research baseline |
| CW-GBLSTSVM | frozen `src/cw_gblstsvm.py` | kappa class weights |
| R-GBLSTSVM | frozen `src/r_gblstsvm.py` | reliability only |
| CR-GBLSTSVM | frozen `src/cr_gblstsvm.py` | `kappa * reliability` |

`GBLSTSVM.py` and `gen_ball.py` were not edited. Original KMeans has no `random_state`; NumPy was seeded with 42 immediately before reference granulation.

---

## 5. Hyperparameters

Phase 2–5 defaults (not tuned):

| Setting | Value |
|---------|--------|
| c1 | 100 |
| c2 | 1e-2 |
| purity_threshold | 0.95 |
| min_ball_count | 2 |
| prediction_rule | `"reference"` |
| random_state | 42 (clean family) |
| CW | standard kappa, `weight_placement="opposite_only"` |
| R / CR | ε=0.05, γ=1.0, λ_r=1.0, `reliability_variant="full"`, `opposite_only` |
| CR final weight | κ × a_i |

---

## 6. Leakage prevention

See §2–3. Validation metrics are reported for sanity only. Test metrics were computed once after all models were fit on train.

---

## 7. Validation results

All five models produced **identical** hard predictions: every validation point classified as majority (−1).

Confusion matrix (labels [−1, +1]): `[[30, 0], [13, 0]]`.

| Model | Val Acc | Val Bal Acc | Val Macro F1 | Val Min Rec | Val AUROC\* |
|-------|---------|-------------|--------------|-------------|-------------|
| Original/reference GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.5949 |
| Clean GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.5949 |
| CW-GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.5923 |
| R-GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.5949 |
| CR-GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.5923 |

\*AUROC uses the reference twin score `|f2|−|f1|` computed from fitted hyperplanes. Models do not expose `decision_function`. G-mean is 0 because minority recall is 0.

Accuracy 0.6977 is exactly 30/43 majority-class accuracy.

---

## 8. Test results

Same majority-only hard predictions. Confusion matrix: `[[30, 0], [13, 0]]`.

| Model | Test Acc | Test Bal Acc | Test Macro F1 | Test Min Rec | Test Min Prec | Test G-mean | Test AUROC |
|-------|----------|--------------|---------------|--------------|---------------|-------------|------------|
| Original/reference GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.0000 | 0.0000 | 0.5513 |
| Clean GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.0000 | 0.0000 | 0.5513 |
| CW-GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.0000 | 0.0000 | 0.5615 |
| R-GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.0000 | 0.0000 | 0.5513 |
| CR-GBLSTSVM | 0.6977 | 0.5000 | 0.4110 | 0.0000 | 0.0000 | 0.0000 | 0.5615 |

Original/reference and clean GBLSTSVM match on hard metrics and AUROC (reproduction check passed for this split).

CW and CR share a slightly different ranking AUROC than baseline/R, but **identical hard labels**. That AUROC gap (~0.01) is a ranking-only difference and does not change Macro F1, balanced accuracy, or minority recall.

---

## 9. Diagnostics

Training labels: **n_pos = 59**, **n_neg = 141**.

Granular balls (identical across original, clean, CW, R, CR on this seed):

| Quantity | Value |
|----------|--------|
| Positive balls | 9 |
| Negative balls | 34 |
| Total balls | 43 |
| Purity min / mean / max | 1.0 / 1.0 / 1.0 |
| Radius min / mean / max | 0.0 / 0.8041 / 2.1014 |
| Radius scale (R/CR) | 0.7785 |
| Samples retained in balls | 123 / 200 |
| Positive samples retained | 20 / 59 |
| Negative samples retained | 103 / 141 |

`min_ball_count=2` (`del_ball`) **drops balls with fewer than 2 samples**. On this fold that removes a large fraction of training points from the optimizer, especially minority points (39 of 59 +1 samples never enter the twin systems). This is frozen reference behavior, not a CR-specific change.

Class weights (training-label kappa):

- κ_pos = 200 / (2 × 59) = **1.6949**
- κ_neg = 200 / (2 × 141) = **0.7092**

Reliability (R and CR):

- min = mean = max = **1.0**
- fraction ≥ 0.99: **1.0**
- fraction < 0.95: **0.0**
- **does not vary**

This follows the frozen formula: with purity p_i = 1 for every ball,

```text
a_i = ε + (1−ε) · 1^γ · exp(−λ_r · r̃_i · 0) = 1
```

Radius variation is irrelevant once impurity is zero.

CR final weights: min 0.7092, max 1.6949 (exactly the two kappa values).  
Fraction of CR weights differing from CW: **0**.  
Max |CR − CW| weight: **0**.

---

## 10. Improvement deltas (test)

Baseline = frozen clean GBLSTSVM.

| Contrast | Bal Acc | Macro F1 | Minority Recall |
|----------|---------|----------|-----------------|
| CW − baseline | 0 | 0 | 0 |
| R − baseline | 0 | 0 | 0 |
| CR − baseline | 0 | 0 | 0 |
| CR − CW | 0 | 0 | 0 |
| CR − R | 0 | 0 | 0 |

One test sample is 1/43 ≈ 0.023 in accuracy. Observed hard-metric deltas are exactly zero, not noise around a small gain.

---

## 11. Scientific interpretation

Checklist:

| Question | Answer |
|----------|--------|
| A. Does CR outperform frozen GBLSTSVM? | **No.** Hard metrics are identical. |
| B. Does CR outperform CW? | **No.** Weights, hard predictions, and AUROC match CW. |
| C. Does R outperform baseline? | **No.** R reduces to baseline because a_i = 1 for all balls. |
| D. Does reliability vary? | **No.** Saturated at 1.0. |
| E. Does CR produce different predictions from CW? | **No.** Test agreement = 1.0 (0 / 43 differences). |
| F. Are gains meaningful? | **No gains** on Macro F1, balanced accuracy, or minority recall. |

**CR == CW on this dataset because reliability saturates.** All balls are purity-1 after granulation at ρ = 0.95, so a_i ≡ 1 and `final_weight = kappa`. CR cannot add information beyond class weighting here.

**CW also failed to change hard predictions vs baseline.** Kappa is applied and κ_pos ≠ κ_neg, and val/test AUROC differs slightly from baseline, but every point still has `|f2| < |f1|` under the reference rule (predicted −1). Class weighting moved scores, not labels.

**All models collapsed to a majority classifier** on both val and test (minority recall 0). This is a first-experiment failure of the *entire* GBLSTSVM stack on this split/hyperparameter point, not evidence that CR is worse than CW.

Contributing factors (not tuned away in this study):

1. Untuned `c1=100`, `c2=1e-2`, `purity_threshold=0.95` on a small, imbalanced table.
2. `del_ball(num_data=2)` discarded 77/200 training samples, including most minority samples (20 retained).
3. Reliability formula is degenerate at purity 1, which is the typical outcome of the frozen splitter at ρ = 0.95.
4. Test n = 43; this is a single seed, not a resampling study.

Original vs clean: metrics and ball summaries match; the clean reproduction did not itself change results on this experiment.

This experiment **does not** support a claim that CR-GBLSTSVM improves real-data performance over GBLSTSVM or CW-GBLSTSVM.
