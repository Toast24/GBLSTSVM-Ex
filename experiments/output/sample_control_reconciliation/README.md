# Sample control reconciliation

Diagnostic reconciling K-sensitivity vs validation sample-control discrepancy.

## Why previous results disagreed

| Aspect | sample_stratified_k (K-sensitivity) | sample_stratified_matched (validation) |
|--------|--------------------------------------|----------------------------------------|
| K meaning | # rho90 balls (entropy_k) | n_target only (from granular sample count) |
| Sample selection RNG | `deterministic_rng_seed(..., method='sample_stratified_k')` | `sample_control_seed(..., method='sample_stratified_matched')` |
| Post-selection | **Rebuild rho90** on subsample | **Singleton balls** (radius=1) |
| Class weights y_train | **Full train** | **Subset only** |

## raw_sample_subset limitation

No non-ball raw-sample Gaussian GBLSTSVM path exists without modifying `src/`. 
`raw_sample_subset` uses identical indices with singleton balls but **full-train y weights** as a proxy.

## Conclusion

**CASE C — PREVIOUS RESULTS WERE NOT COMPARABLE**

The K-sensitivity `sample_stratified_k` and validation `sample_stratified_matched` controls differ in sample selection RNG, post-selection representation (rho90-rebuild vs singleton balls), and classifier weighting (full y_train vs y_sub). Identical sample sets: 1.1% of runs; semantically matched: 0.0%.


============================================================
SAMPLE CONTROL RECONCILIATION
============================================================

Previous K-sensitivity sample control (sample_stratified_k_original):
    Mean Macro F1: 0.6173
    Mean retained samples: 143.3
    Collapse rate: 0.191

Previous validation matched control (sample_stratified_matched_original):
    Mean Macro F1: 0.4743
    Mean retained samples: 143.3
    Collapse rate: 0.000

Are they semantically equivalent?
    NO (identical indices: 1.1%, full semantic match: 0.0%)

Primary discrepancy:
    Different RNG seeds for sample selection; different post-selection representation
    (rho90-rebuild vs singleton balls); different y_train for class weights (full vs subset).

Raw sample subset (proxy — full-y singleton balls):
    Mean Macro F1: 0.4752
    Collapse rate: 0.000

Singleton-ball subset (same indices as k_original):
    Mean Macro F1: 0.4751
    Collapse rate: 0.000

Granular compressed (entropy_k):
    Mean Macro F1: 0.6022
    Collapse rate: 0.275

Granular - raw proxy:
    Mean Delta: +0.1239
    Wilcoxon p: 6.811e-46
    Win rate: 73.7%

Granular - singleton (same indices):
    Mean Delta: +0.1241
    Wilcoxon p: 4.65e-46
    Win rate: 73.7%

FINAL SCIENTIFIC CONCLUSION:
    CASE C
**CASE C — PREVIOUS RESULTS WERE NOT COMPARABLE**

The K-sensitivity `sample_stratified_k` and validation `sample_stratified_matched` controls differ in sample selection RNG, post-selection representation (rho90-rebuild vs singleton balls), and classifier weighting (full y_train vs y_sub). Identical sample sets: 1.1% of runs; semantically matched: 0.0%.

RECOMMENDATION:
    Do not claim granular-specific advantage from validation experiment alone; reconcile controls before new variants.
============================================================
