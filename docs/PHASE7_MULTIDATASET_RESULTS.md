# Phase 7: Multi-Dataset Train / Validation / Test Results

This document records the imbalanced-benchmark comparison of frozen GBLSTSVM variants on `diabetes.csv`, `ecoli2.csv`, and `vehicle0.csv`. Hyperparameters were **not** tuned on validation or test. No frozen model file was modified.

Script: `experiments/validate_multidataset.py`

Existing tests: `python -m pytest tests/ -v` → **59 passed** (unchanged), then experiment executed.

---

## 1. Dataset descriptions

| Dataset | Samples | Features | Class distribution | Imbalance ratio |
|---------|---------|----------|-------------------|-----------------|
| diabetes | 768 | 8 | 0: 500, 1: 268 | 1.87 |
| ecoli2 | 336 | 7 | 0: 284, 1: 52 | 5.46 |
| vehicle0 | 846 | 18 | 0: 647, 1: 199 | 3.25 |

Label mapping (all datasets binary 0/1 in CSV): **0 → −1**, **1 → +1**.

Inspection: no missing values; no duplicate rows; all feature columns numeric; no rows or columns silently discarded.

---

## 2. Experimental protocol

- Seed **42**; stratified 70/15/15 via two `train_test_split` calls.
- Validation used only for diagnostic evaluation; **no tuning performed**.
- Test set used only for final evaluation.
- Prediction rule: **reference** (`sign(|f2| − |f1|)`).

---

## 3. Preprocessing

**None** (consistent with frozen baseline protocol and Phase 6 breast_cancer experiment).

Frozen implementations and reference code do not scale features internally. `diabetes.csv` and `vehicle0.csv` are not pre-standardized; `ecoli2.csv` features are in [0, 1] range.

---

## 4. Model configurations

| Parameter | Baseline/Clean | CW | R | CR |
|-----------|----------------|----|----|-----|
| c1 | 100.0 | 100.0 | 100.0 | 100.0 |
| c2 | 0.01 | 0.01 | 0.01 | 0.01 |
| purity_threshold | 0.95 | 0.95 | 0.95 | 0.95 |
| min_ball_count | 2 | 2 | 2 | 2 |
| epsilon | N/A | N/A | 0.05 | 0.05 |
| gamma | N/A | N/A | 1.0 | 1.0 |
| lambda_r | N/A | N/A | 1.0 | 1.0 |
| weight_placement | N/A | opposite_only | opposite_only | opposite_only |
| prediction_rule | reference | reference | reference | reference |
| random_state | 42 | 42 | 42 | 42 |

---

## 5. Validation results

| Dataset | Model | Accuracy | Bal Acc | Macro F1 | Minority Recall | G-mean |
|---------|-------|----------|---------|----------|-----------------|--------|
| diabetes | Original/reference GBLSTSVM | 0.6609 | 0.5125 | 0.4212 | 0.0250 | 0.1581 |
| diabetes | Clean GBLSTSVM | 0.6609 | 0.5125 | 0.4212 | 0.0250 | 0.1581 |
| diabetes | CW-GBLSTSVM | 0.6609 | 0.5125 | 0.4212 | 0.0250 | 0.1581 |
| diabetes | R-GBLSTSVM | 0.6609 | 0.5125 | 0.4212 | 0.0250 | 0.1581 |
| diabetes | CR-GBLSTSVM | 0.6609 | 0.5125 | 0.4212 | 0.0250 | 0.1581 |
| ecoli2 | Original/reference GBLSTSVM | 0.8200 | 0.6399 | 0.6471 | 0.3750 | 0.5825 |
| ecoli2 | Clean GBLSTSVM | 0.8200 | 0.6399 | 0.6471 | 0.3750 | 0.5825 |
| ecoli2 | CW-GBLSTSVM | 0.8200 | 0.6399 | 0.6471 | 0.3750 | 0.5825 |
| ecoli2 | R-GBLSTSVM | 0.8200 | 0.6399 | 0.6471 | 0.3750 | 0.5825 |
| ecoli2 | CR-GBLSTSVM | 0.8200 | 0.6399 | 0.6471 | 0.3750 | 0.5825 |
| vehicle0 | Original/reference GBLSTSVM | 0.8110 | 0.6460 | 0.6701 | 0.3333 | 0.5653 |
| vehicle0 | Clean GBLSTSVM | 0.8110 | 0.6460 | 0.6701 | 0.3333 | 0.5653 |
| vehicle0 | CW-GBLSTSVM | 0.8110 | 0.6576 | 0.6814 | 0.3667 | 0.5897 |
| vehicle0 | R-GBLSTSVM | 0.8110 | 0.6460 | 0.6701 | 0.3333 | 0.5653 |
| vehicle0 | CR-GBLSTSVM | 0.8110 | 0.6576 | 0.6814 | 0.3667 | 0.5897 |

---

## 6. Test results

| Dataset | Model | Accuracy | Bal Acc | Macro F1 | Minority Recall | G-mean | AUROC | Collapse |
|---------|-------|----------|---------|----------|-----------------|--------|-------|----------|
| diabetes | Original/reference GBLSTSVM | 0.6466 | 0.5000 | 0.3927 | 0.0000 | 0.0000 | 0.4982 | YES |
| diabetes | Clean GBLSTSVM | 0.6466 | 0.5000 | 0.3927 | 0.0000 | 0.0000 | 0.4982 | YES |
| diabetes | CW-GBLSTSVM | 0.6466 | 0.5000 | 0.3927 | 0.0000 | 0.0000 | 0.5054 | YES |
| diabetes | R-GBLSTSVM | 0.6466 | 0.5000 | 0.3927 | 0.0000 | 0.0000 | 0.4930 | YES |
| diabetes | CR-GBLSTSVM | 0.6466 | 0.5000 | 0.3927 | 0.0000 | 0.0000 | 0.4985 | YES |
| ecoli2 | Original/reference GBLSTSVM | 0.9020 | 0.7892 | 0.8046 | 0.6250 | 0.7720 | 0.9331 | NO |
| ecoli2 | Clean GBLSTSVM | 0.9020 | 0.7892 | 0.8046 | 0.6250 | 0.7720 | 0.9331 | NO |
| ecoli2 | CW-GBLSTSVM | 0.9020 | 0.7892 | 0.8046 | 0.6250 | 0.7720 | 0.9302 | NO |
| ecoli2 | R-GBLSTSVM | 0.9020 | 0.7892 | 0.8046 | 0.6250 | 0.7720 | 0.9331 | NO |
| ecoli2 | CR-GBLSTSVM | 0.9020 | 0.7892 | 0.8046 | 0.6250 | 0.7720 | 0.9302 | NO |
| vehicle0 | Original/reference GBLSTSVM | 0.8189 | 0.6512 | 0.6781 | 0.3333 | 0.5684 | 0.9103 | NO |
| vehicle0 | Clean GBLSTSVM | 0.8189 | 0.6512 | 0.6781 | 0.3333 | 0.5684 | 0.9103 | NO |
| vehicle0 | CW-GBLSTSVM | 0.8110 | 0.6691 | 0.6917 | 0.4000 | 0.6126 | 0.9244 | NO |
| vehicle0 | R-GBLSTSVM | 0.8189 | 0.6512 | 0.6781 | 0.3333 | 0.5684 | 0.9120 | NO |
| vehicle0 | CR-GBLSTSVM | 0.8110 | 0.6691 | 0.6917 | 0.4000 | 0.6126 | 0.9241 | NO |

---

## 7. Reference vs clean verification

### diabetes
- Verified: **True**
- Balls: ref=85, clean=85, match=True
- Val prediction agreement: 1.0; test: 1.0
- Score max abs diff: val=2.877698079828406e-13, test=4.2454928461665986e-13

### ecoli2
- Verified: **True**
- Balls: ref=13, clean=13, match=True
- Val prediction agreement: 1.0; test: 1.0
- Score max abs diff: val=0.0, test=0.0

### vehicle0
- Verified: **True**
- Balls: ref=59, clean=59, match=True
- Val prediction agreement: 1.0; test: 1.0
- Score max abs diff: val=1.9610979506978765e-12, test=2.0463630789890885e-12

---

## 8. Ball diagnostics (summary)

### diabetes
- **Original/reference GBLSTSVM**: balls=85 (pos=27, neg=58), samples_in_balls=None, purity mean=0.9992, radius mean=10.9703
- **Clean GBLSTSVM**: balls=85 (pos=27, neg=58), samples_in_balls=326, purity mean=0.9992, radius mean=10.9703
- **CW-GBLSTSVM**: balls=85 (pos=27, neg=58), samples_in_balls=326, purity mean=0.9992, radius mean=10.9703
- **R-GBLSTSVM**: balls=85 (pos=27, neg=58), samples_in_balls=326, purity mean=0.9992, radius mean=10.9703
- **CR-GBLSTSVM**: balls=85 (pos=27, neg=58), samples_in_balls=326, purity mean=0.9992, radius mean=10.9703

### ecoli2
- **Original/reference GBLSTSVM**: balls=13 (pos=6, neg=7), samples_in_balls=None, purity mean=0.9982, radius mean=0.1394
- **Clean GBLSTSVM**: balls=13 (pos=6, neg=7), samples_in_balls=217, purity mean=0.9982, radius mean=0.1394
- **CW-GBLSTSVM**: balls=13 (pos=6, neg=7), samples_in_balls=217, purity mean=0.9982, radius mean=0.1394
- **R-GBLSTSVM**: balls=13 (pos=6, neg=7), samples_in_balls=217, purity mean=0.9982, radius mean=0.1394
- **CR-GBLSTSVM**: balls=13 (pos=6, neg=7), samples_in_balls=217, purity mean=0.9982, radius mean=0.1394

### vehicle0
- **Original/reference GBLSTSVM**: balls=59 (pos=27, neg=32), samples_in_balls=None, purity mean=0.9997, radius mean=14.9227
- **Clean GBLSTSVM**: balls=59 (pos=27, neg=32), samples_in_balls=525, purity mean=0.9997, radius mean=14.9227
- **CW-GBLSTSVM**: balls=59 (pos=27, neg=32), samples_in_balls=525, purity mean=0.9997, radius mean=14.9227
- **R-GBLSTSVM**: balls=59 (pos=27, neg=32), samples_in_balls=525, purity mean=0.9997, radius mean=14.9227
- **CR-GBLSTSVM**: balls=59 (pos=27, neg=32), samples_in_balls=525, purity mean=0.9997, radius mean=14.9227

---

## 9. Reliability diagnostics

### diabetes: **RELIABILITY ACTIVE**
- Mean purity: 0.9992; mean reliability: 0.9976; frac rel ≥ 0.99: 0.9765
- R effectively baseline: False; CR effectively CW: False; CR weights differ from CW: True

### ecoli2: **RELIABILITY ACTIVE**
- Mean purity: 0.9982; mean reliability: 0.9950; frac rel ≥ 0.99: 0.8462
- R effectively baseline: False; CR effectively CW: False; CR weights differ from CW: True

### vehicle0: **RELIABILITY ACTIVE**
- Mean purity: 0.9997; mean reliability: 0.9989; frac rel ≥ 0.99: 0.9831
- R effectively baseline: False; CR effectively CW: False; CR weights differ from CW: True

---

## 10. CW / R / CR comparisons (test deltas vs clean)

### diabetes
- CW_minus_clean: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- R_minus_clean: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- CR_minus_clean: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- CR_minus_CW: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- CR_minus_R: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000

### ecoli2
- CW_minus_clean: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- R_minus_clean: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- CR_minus_clean: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- CR_minus_CW: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- CR_minus_R: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000

### vehicle0
- CW_minus_clean: Δmacro_f1=+0.0137, Δbal_acc=+0.0179, Δminority_recall=+0.0667
- R_minus_clean: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- CR_minus_clean: Δmacro_f1=+0.0137, Δbal_acc=+0.0179, Δminority_recall=+0.0667
- CR_minus_CW: Δmacro_f1=+0.0000, Δbal_acc=+0.0000, Δminority_recall=+0.0000
- CR_minus_R: Δmacro_f1=+0.0137, Δbal_acc=+0.0179, Δminority_recall=+0.0667

---

## 11. Scientific interpretation

- CR improves over clean on test (any of macro F1 / bal acc / minority recall): ['vehicle0']
- CR improves over CW on test: none
- CW improves macro F1 over clean: ['vehicle0']
- Reliability SATURATED datasets: none
- Reliability ACTIVE datasets: ['diabetes', 'ecoli2', 'vehicle0']

Single-metric improvements should not be treated as evidence of superiority without consistency across datasets and non-collapsed minority recall.

---

## 12. Limitations

- No hyperparameter tuning; defaults may not be optimal per dataset.
- No feature scaling (frozen protocol); distance-based granulation on raw features for diabetes/vehicle0.
- Small test sets (15% stratified) limit statistical power.
- Original KMeans in reference granulation is not fully deterministic beyond numpy seed.
- AUROC uses twin hyperplane score |f2|−|f1|, not a calibrated probability.

---

## 13. Recommendation for next experiment

1. **Granulation sensitivity**: sweep `purity_threshold` and `min_ball_count` on datasets where reliability is saturated to find regimes where R/CR diverge from baseline/CW.
2. **Scaling ablation**: separate controlled experiment with StandardScaler fit on train only (document as new protocol, not frozen baseline).
3. **Hyperparameter grid**: c1/c2 per dataset with validation-only selection, then single test evaluation.
4. **Resampling / cross-validation** for stable estimates on small test folds.
5. If CW helps but CR does not, prioritize diagnosing reliability saturation before further CR formulation changes.
