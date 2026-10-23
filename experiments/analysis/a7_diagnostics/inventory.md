# Task 0 — Artifact inventory

## runs.csv
- rows: 4080
- columns and dtypes:

```
dataset                      object
seed                          int64
arm                          object
tuning_mode                  object
prediction_rule              object
c1                          float64
c2                          float64
lambda                      float64
ridge_lambda_pos            float64
ridge_lambda_neg            float64
validation_macro_f1         float64
n_balls                       int64
macro_f1                    float64
accuracy                    float64
balanced_accuracy           float64
classifier_ok                  bool
finite_macro_f1               int64
classifier_error            float64
train_time_seconds          float64
predict_time_seconds        float64
trace_S_total               float64
trace_contribution          float64
condition_number_comb1      float64
trace_full                  float64
trace_diagonal              float64
trace_isotropic             float64
fro_full_minus_diag         float64
fro_full_minus_isotropic    float64
fro_diag_minus_isotropic    float64
n_balls_geom                  int64
```

## ridge_matching.csv
```
dataset                    object
seed                        int64
lambda_pos_trace_match    float64
lambda_neg_trace_match    float64
trace_pos                 float64
trace_neg                 float64
formula                    object
```

## scatter_geometry.csv
```
dataset                      object
seed                          int64
trace_full                  float64
trace_diagonal              float64
trace_isotropic             float64
fro_full_minus_diag         float64
fro_full_minus_isotropic    float64
fro_diag_minus_isotropic    float64
n_balls_geom                  int64
```

## dataset_pairwise_deltas.csv
```
dataset             object
prediction_rule     object
tuning_mode         object
comparison          object
category            object
delta_macro_f1     float64
```

## Distinct factor levels
- `arm`: ['DIAG_SCATTER', 'FULL_SCATTER', 'ISOTROPIC_SCATTER', 'L1_POPULATION', 'RIDGE_TRACE_MATCH', 'RIDGE_VAL_TUNED']
- `tuning_mode`: ['MODE_A_SHARED_C', 'MODE_B_PER_ARM_C']
- `prediction_rule`: ['RULE_REFERENCE', 'RULE_REPO']

## Fitted parameters on disk
- `runs.csv` columns w1/w2/b1/b2: **none** (checked columns: ['dataset', 'seed', 'arm', 'tuning_mode', 'prediction_rule', 'c1', 'c2', 'lambda', 'ridge_lambda_pos', 'ridge_lambda_neg', 'validation_macro_f1', 'n_balls', 'macro_f1', 'accuracy', 'balanced_accuracy', 'classifier_ok', 'finite_macro_f1', 'classifier_error', 'train_time_seconds', 'predict_time_seconds', 'trace_S_total', 'trace_contribution', 'condition_number_comb1', 'trace_full', 'trace_diagonal', 'trace_isotropic', 'fro_full_minus_diag', 'fro_full_minus_isotropic', 'fro_diag_minus_isotropic', 'n_balls_geom'])
- `.npz`/`.pkl`/`.joblib` under `experiments/output/`: **none found** (glob at inventory time)
- A6 script `--dump-params` writes to `C:\Users\kedar\Desktop\BITS\Acads\BharatSOP\GBLSTSVM\experiments\output\estimator_a6_scatter_mechanism\params/` when invoked

## A4 (`estimator_information_ladder`) RAW parameters
- `all_runs.csv` has no w1/w2/b1/b2 columns; no serialized models in that output tree.

## c-grid (MODE B)
- Source: `experiments/rebaseline_common.py` → `C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`
- Per-run selected c1/c2 are in `runs.csv` (`c1`, `c2` columns).

## Hyperparameter columns present
- **Yes**: `c1`, `c2`, `lambda`, `ridge_lambda_pos`, `ridge_lambda_neg`, `validation_macro_f1`.
