# contrast_distributions.md

Bootstrap: BCa, n=10000, seed=42.

## Collapse definition used
- `collapsed` := (not classifier_ok) OR (finite_macro_f1==0) OR (macro_f1 non-finite).
- `runs.csv` has no `collapsed` column; derived at analysis time.

## Collapse rate by arm and rule
              arm prediction_rule  collapse_rate
     DIAG_SCATTER  RULE_REFERENCE            0.0
     DIAG_SCATTER       RULE_REPO            0.0
     FULL_SCATTER  RULE_REFERENCE            0.0
     FULL_SCATTER       RULE_REPO            0.0
ISOTROPIC_SCATTER  RULE_REFERENCE            0.0
ISOTROPIC_SCATTER       RULE_REPO            0.0
    L1_POPULATION  RULE_REFERENCE            0.0
    L1_POPULATION       RULE_REPO            0.0
RIDGE_TRACE_MATCH  RULE_REFERENCE            0.0
RIDGE_TRACE_MATCH       RULE_REPO            0.0
  RIDGE_VAL_TUNED  RULE_REFERENCE            0.0
  RIDGE_VAL_TUNED       RULE_REPO            0.0

## Multiplicity (Holm-Bonferroni)
- Family: 32 tests = 8 contrasts × 2 tuning modes × 2 rules, policy=exclude_collapsed, two-sided Wilcoxon on dataset-level deltas.

                                 contrast      tuning_mode prediction_rule  wilcoxon_p   holm_p  survives_0.05
    ISOTROPIC_SCATTER_minus_L1_POPULATION  MODE_A_SHARED_C       RULE_REPO    0.000290 0.009277           True
     FULL_SCATTER_minus_ISOTROPIC_SCATTER  MODE_A_SHARED_C       RULE_REPO    0.000381 0.011826           True
     FULL_SCATTER_minus_RIDGE_TRACE_MATCH  MODE_A_SHARED_C       RULE_REPO    0.000381 0.011444           True
     FULL_SCATTER_minus_ISOTROPIC_SCATTER MODE_B_PER_ARM_C       RULE_REPO    0.001068 0.030975           True
     FULL_SCATTER_minus_RIDGE_TRACE_MATCH MODE_B_PER_ARM_C       RULE_REPO    0.001068 0.029907           True
         DIAG_SCATTER_minus_L1_POPULATION  MODE_A_SHARED_C       RULE_REPO    0.002090 0.056442          False
         FULL_SCATTER_minus_L1_POPULATION  MODE_A_SHARED_C       RULE_REPO    0.002579 0.067047          False
     DIAG_SCATTER_minus_ISOTROPIC_SCATTER  MODE_A_SHARED_C       RULE_REPO    0.003845 0.096130          False
         FULL_SCATTER_minus_L1_POPULATION MODE_B_PER_ARM_C       RULE_REPO    0.007904 0.189697          False
     DIAG_SCATTER_minus_ISOTROPIC_SCATTER MODE_B_PER_ARM_C       RULE_REPO    0.017426 0.400787          False
          FULL_SCATTER_minus_DIAG_SCATTER  MODE_A_SHARED_C  RULE_REFERENCE    0.039536 0.869781          False
          FULL_SCATTER_minus_DIAG_SCATTER MODE_B_PER_ARM_C       RULE_REPO    0.039536 0.830246          False
         DIAG_SCATTER_minus_L1_POPULATION  MODE_A_SHARED_C  RULE_REFERENCE    0.050537 1.000000          False
    ISOTROPIC_SCATTER_minus_L1_POPULATION  MODE_A_SHARED_C  RULE_REFERENCE    0.063828 1.000000          False
          FULL_SCATTER_minus_DIAG_SCATTER MODE_B_PER_ARM_C  RULE_REFERENCE    0.079681 1.000000          False
     FULL_SCATTER_minus_ISOTROPIC_SCATTER  MODE_A_SHARED_C  RULE_REFERENCE    0.088654 1.000000          False
     FULL_SCATTER_minus_RIDGE_TRACE_MATCH  MODE_A_SHARED_C  RULE_REFERENCE    0.088654 1.000000          False
     FULL_SCATTER_minus_ISOTROPIC_SCATTER MODE_B_PER_ARM_C  RULE_REFERENCE    0.108887 1.000000          False
     FULL_SCATTER_minus_RIDGE_TRACE_MATCH MODE_B_PER_ARM_C  RULE_REFERENCE    0.108887 1.000000          False
    ISOTROPIC_SCATTER_minus_L1_POPULATION MODE_B_PER_ARM_C  RULE_REFERENCE    0.132370 1.000000          False
          FULL_SCATTER_minus_DIAG_SCATTER  MODE_A_SHARED_C       RULE_REPO    0.206894 1.000000          False
    ISOTROPIC_SCATTER_minus_L1_POPULATION MODE_B_PER_ARM_C       RULE_REPO    0.263321 1.000000          False
         DIAG_SCATTER_minus_L1_POPULATION MODE_B_PER_ARM_C  RULE_REFERENCE    0.284164 1.000000          False
         FULL_SCATTER_minus_L1_POPULATION  MODE_A_SHARED_C  RULE_REFERENCE    0.306046 1.000000          False
     DIAG_SCATTER_minus_ISOTROPIC_SCATTER  MODE_A_SHARED_C  RULE_REFERENCE    0.377823 1.000000          False
     DIAG_SCATTER_minus_ISOTROPIC_SCATTER MODE_B_PER_ARM_C  RULE_REFERENCE    0.579056 1.000000          False
         DIAG_SCATTER_minus_L1_POPULATION MODE_B_PER_ARM_C       RULE_REPO    0.746658 1.000000          False
         FULL_SCATTER_minus_L1_POPULATION MODE_B_PER_ARM_C  RULE_REFERENCE    0.817581 1.000000          False
ISOTROPIC_SCATTER_minus_RIDGE_TRACE_MATCH  MODE_A_SHARED_C       RULE_REPO         NaN 1.000000          False
ISOTROPIC_SCATTER_minus_RIDGE_TRACE_MATCH MODE_B_PER_ARM_C       RULE_REPO         NaN 1.000000          False
ISOTROPIC_SCATTER_minus_RIDGE_TRACE_MATCH  MODE_A_SHARED_C  RULE_REFERENCE         NaN 1.000000          False
ISOTROPIC_SCATTER_minus_RIDGE_TRACE_MATCH MODE_B_PER_ARM_C  RULE_REFERENCE         NaN 1.000000          False

## FULL − ISOTROPIC under MODE B (per-dataset vectors)

### RULE_REPO
```
dataset
australian                            0.038582
breast_cancer                         0.031897
breast_cancer_wisconsin_diagnostic    0.027624
breast_cancer_wisconsin_original     -0.000968
chronic_kidney_disease                0.057448
congressional_voting_records         -0.001812
credit_approval                       0.093470
diabetes                              0.008101
ecoli2                               -0.000951
german_credit                        -0.014120
heart                                 0.117377
ionosphere                            0.051033
messidor_features                     0.046310
spambase                              0.143428
tic_tac_toe                           0.287678
vehicle0                              0.067990
wholesale_customers                   0.006384
```

### RULE_REFERENCE
```
dataset
australian                           -0.161627
breast_cancer                         0.006570
breast_cancer_wisconsin_diagnostic    0.097222
breast_cancer_wisconsin_original      0.004409
chronic_kidney_disease                0.432881
congressional_voting_records          0.012877
credit_approval                      -0.198313
diabetes                              0.051517
ecoli2                               -0.073935
german_credit                        -0.043797
heart                                 0.200434
ionosphere                           -0.085918
messidor_features                     0.192934
spambase                              0.389608
tic_tac_toe                           0.317686
vehicle0                              0.334417
wholesale_customers                   0.001565
```
