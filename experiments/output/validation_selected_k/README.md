# Validation-based K-selection
Deployability experiment: can internal validation turn rho90 compression into a model-selection procedure?
## Protocol
```
benchmark train (70%) → stratified 80/20 → train_selection | validation_internal
train_selection → rho90 pool → random-K compression (5 reps) → val macro F1
select K (max-F1 or 1-SE) on validation only
full benchmark train → rebuild rho90 → compress at K → train → evaluate on untouched test (15%)
```
## Leakage prevention
- Test data never enters granulation, compression, or K selection.
- Benchmark val (15%) is unused; internal validation is carved from benchmark train.
- `validation` phase rows use internal val features only; `final_test` uses benchmark test only.
## Selection rules
- **K_selected_max**: argmax mean validation macro F1 across 5 random compressions; tie-break: lower collapse rate, then smaller K.
- **K_selected_1se**: threshold = best_mean − SE(best), SE = std(reps)/√n_successful; smallest K with mean ≥ threshold. Reps are dependent compressions of the same pool — SE is approximate.
## Controls
- `full_rho90`: uncompressed rho90 on full train.
- `fixed_K6/8/10`: random ball compression at fixed K.
- `full_sample`: one ball per training sample (no granulation).
- `sample_stratified_matched`: stratified sample subset matched to granular sample count.
## Paired comparisons (test macro F1)
```csv
focal,baseline,n_pairs,mean_delta_macro_f1,median_delta_macro_f1,wins,ties,losses,pct_wins,wilcoxon_stat,wilcoxon_p
validation_selected_max,full_rho90,151,0.038978981889736,0.040544563222765284,87,8,56,57.615894039735096,3621.0,0.002089608568073366
validation_selected_1se,full_rho90,151,0.038978981889736,0.040544563222765284,87,8,56,57.615894039735096,3621.0,0.002089608568073366
fixed_K6,full_rho90,139,0.0241241815866112,0.032830317200716386,79,5,55,56.83453237410072,3581.0,0.036538096683919066
fixed_K8,full_rho90,144,0.01876291787521494,0.026101644304055815,79,12,53,54.861111111111114,3473.0,0.0374807265670095
fixed_K10,full_rho90,141,0.03670922388268823,0.050762356996580826,83,4,54,58.86524822695036,3315.0,0.002424268828853684
validation_selected_max,sample_stratified_matched,151,0.19232513772164853,0.19681349578256807,129,4,18,85.43046357615894,681.0,3.550119316259197e-20
validation_selected_1se,sample_stratified_matched,151,0.19232513772164853,0.19681349578256807,129,4,18,85.43046357615894,681.0,3.550119316259197e-20
```

## Selection stability

```csv
rule,K,count,fraction,median_K,mean_K,iqr_K,dataset
K_selected_max,4.0,19,0.11176470588235295,,,,
K_selected_max,6.0,14,0.08235294117647059,,,,
K_selected_max,8.0,23,0.13529411764705881,,,,
K_selected_max,10.0,21,0.12352941176470589,,,,
K_selected_max,12.0,24,0.1411764705882353,,,,
K_selected_max,16.0,15,0.08823529411764706,,,,
K_selected_max,20.0,10,0.058823529411764705,,,,
K_selected_max,24.0,14,0.08235294117647059,,,,
K_selected_max,32.0,20,0.11764705882352941,,,,
K_selected_max,,170,,12.0,14.1125,12.0,
K_selected_1se,4.0,19,0.11176470588235295,,,,
K_selected_1se,6.0,14,0.08235294117647059,,,,
K_selected_1se,8.0,23,0.13529411764705881,,,,
K_selected_1se,10.0,21,0.12352941176470589,,,,
K_selected_1se,12.0,24,0.1411764705882353,,,,
K_selected_1se,16.0,15,0.08823529411764706,,,,
K_selected_1se,20.0,10,0.058823529411764705,,,,
K_selected_1se,24.0,14,0.08235294117647059,,,,
K_selected_1se,32.0,20,0.11764705882352941,,,,
K_selected_1se,,170,,12.0,14.1125,12.0,
dataset_mode_max,4.0,2,0.2,,,,australian
dataset_mode_1se,4.0,2,0.2,,,,australian
dataset_mode_max,8.0,5,0.5,,,,breast_cancer
dataset_mode_1se,8.0,5,0.5,,,,breast_cancer
dataset_mode_max,12.0,4,0.4,,,,breast_cancer_wisconsin_diagnostic
dataset_mode_1se,12.0,4,0.4,,,,breast_cancer_wisconsin_diagnostic
dataset_mode_max,10.0,5,0.5,,,,chronic_kidney_disease
dataset_mode_1se,10.0,5,0.5,,,,chronic_kidney_disease
dataset_mode_max,4.0,3,0.3,,,,congressional_voting_records
dataset_mode_1se,4.0,3,0.3,,,,congressional_voting_records
```

## Interpretation

Automated interpretation: A

**Case A** — Validation-selected granular compression significantly improves over full rho90 and over matched sample compression. Evidence supports a granular-specific contribution.

validation_selected_max vs full_rho90: Δ=+0.0390 (sig=True)
validation_selected_max vs sample_matched: Δ=+0.1923 (sig=True)
validation_selected_1se vs full_rho90: Δ=+0.0390

## Outputs

- raw_results.csv
- validation_results.csv
- test_results.csv
- selected_k_by_run.csv
- selection_stability.csv
- paired_comparisons.csv
- dataset_level_results.csv
- failure_analysis.csv
- summary.json
- plots/*.png

Classifier: Gaussian GBLSTSVM (purity_compact); C1=100.0; C2=0.01
