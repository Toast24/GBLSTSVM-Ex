# Re-granulation ablation

Controlled experiment: fixed stratified sample subset; compare singleton vs re-granulation at multiple rho.

## Design

- **Matched subset**: one `sample_stratified_k` selection per dataset×seed×K.
- **Singleton**: existing reconciliation singleton-ball control (`y_sub` weights).
- **Re-granulate**: rho ∈ {0.80, 0.85, 0.90, 0.95, 0.98} on identical indices (`full y_train` weights).
- **Secondary baseline**: `sample_stratified_k_original` (rho90 rebuild on subset).
- **Reference**: `full_rho90` on full train (`full_data_reference=True`).

## Conclusion (CASE B)

> Re-granulation materially changes the compressed representation, but there is no evidence that granular structure provides an advantage beyond ordinary sample-level compression.

### Post-hoc descriptive optimum

Best mean F1 among re-granulate methods (NOT a deployable selected hyperparameter):

```json
{
  "method": "regranulate_rho80",
  "rho": 0.8,
  "K": 2,
  "n_runs": 170,
  "mean_macro_f1": 0.6655620646988406,
  "median_macro_f1": 0.6651785714285714,
  "std_macro_f1": 0.16144045050402703,
  "frac_collapsed": 0.7176470588235294,
  "mean_n_balls": 4.141176470588236,
  "median_n_balls": 2.0,
  "mean_actual_retained_samples": 96.4235294117647
}
```


============================================================
RE-GRANULATION ABLATION
============================================================

Runs:
    17 datasets × 10 seeds × 6 K values × methods

Selected samples identical across branches:
    YES

Singleton:
    Mean F1: 0.4775
    Collapse: 0.000
    Mean balls: 143.3

Regranulate rho=0.80:
    Mean F1: 0.6211
    Collapse: 0.372
    Mean balls: 10.6
    Delta vs singleton: +0.1339

Regranulate rho=0.85:
    Mean F1: 0.6193
    Collapse: 0.218
    Mean balls: 13.7
    Delta vs singleton: +0.1401

Regranulate rho=0.90:
    Mean F1: 0.6179
    Collapse: 0.202
    Mean balls: 15.6
    Delta vs singleton: +0.1397

Regranulate rho=0.95:
    Mean F1: 0.6116
    Collapse: 0.197
    Mean balls: 17.8
    Delta vs singleton: +0.1337

Regranulate rho=0.98:
    Mean F1: 0.6038
    Collapse: 0.189
    Mean balls: 19.5
    Delta vs singleton: +0.1259


Sample-level compression baseline:
    Mean F1: 0.6179

Best descriptive rho (post-hoc descriptive optimum):
    regranulate_rho80 (mean F1=0.6656)

Does re-granulation beat singleton?
    YES

Does re-granulation beat sample compression?
    NO

Does performance primarily track ball count?
    UNCLEAR

FINAL CASE:
    B

SCIENTIFIC CONCLUSION:
    > Re-granulation materially changes the compressed representation, but there is no evidence that granular structure provides an advantage beyond ordinary sample-level compression.

RECOMMENDED NEXT STEP:
    Treat gains as sample-compression effect; stop granular-specific branch.
============================================================
