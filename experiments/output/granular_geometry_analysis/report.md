# Granular geometry vs GBLSTSVM performance

Read-only diagnostic: which granular-ball properties predict Gaussian-weighted GBLSTSVM success across frozen granulation variants.

Runs: **306** | Weighting: **Gaussian purity_compact** | Seeds: [42, 0, 1]

## 1. Geometry vs performance correlation

Pearson and Spearman correlations between representation-level geometry metrics and classifier performance (all dataset×seed×representation rows).

| geometry | performance | Pearson | Spearman | n |
|---|---|---:|---:|---:|
| mean_purity | macro_f1 | 0.038 | -0.129 | 291 |
| mean_purity | minority_f1 | 0.022 | -0.123 | 291 |
| mean_purity | gmean | 0.000 | -0.142 | 291 |
| mean_radius | macro_f1 | -0.129 | -0.028 | 291 |
| mean_radius | minority_f1 | -0.144 | -0.098 | 291 |
| mean_radius | gmean | -0.014 | 0.017 | 291 |
| radius_variance | macro_f1 | -0.139 | -0.067 | 291 |
| radius_variance | minority_f1 | -0.106 | -0.127 | 291 |
| radius_variance | gmean | -0.018 | -0.028 | 291 |
| mean_density | macro_f1 | 0.078 | 0.100 | 291 |
| mean_density | minority_f1 | -0.011 | 0.187 | 291 |
| mean_density | gmean | -0.005 | 0.074 | 291 |
| mean_entropy | macro_f1 | 0.065 | 0.173 | 291 |
| mean_entropy | minority_f1 | 0.063 | 0.157 | 291 |
| mean_entropy | gmean | 0.089 | 0.184 | 291 |
| fraction_mixed_balls | macro_f1 | 0.353 | 0.308 | 291 |
| fraction_mixed_balls | minority_f1 | 0.290 | 0.261 | 291 |
| fraction_mixed_balls | gmean | 0.342 | 0.317 | 291 |
| mean_margin_proxy | macro_f1 | -0.058 | -0.035 | 268 |
| mean_margin_proxy | minority_f1 | -0.219 | -0.205 | 268 |
| mean_margin_proxy | gmean | -0.110 | -0.072 | 268 |
| minority_survival | macro_f1 | 0.571 | 0.574 | 291 |
| minority_survival | minority_f1 | 0.454 | 0.468 | 291 |
| minority_survival | gmean | 0.548 | 0.575 | 291 |

## 2. Compare representations

| Representation | Balls | Purity | Radius | Density | Heterogeneous balls | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 88.8 | 0.997 | 187.9909 | 35430059497.71 | 1.0 | 0.569 |
| SU3 | 41.9 | 0.762 | 389.5903 | 1250966141.77 | 26.6 | 0.556 |
| rho0.80 | 50.4 | 0.931 | 908.0187 | 19865415245.33 | 15.3 | 0.576 |
| rho0.85 | 66.1 | 0.977 | 219.3029 | 28694967217.42 | 9.6 | 0.582 |
| rho0.90 | 77.8 | 0.992 | 206.0065 | 23307632061.38 | 4.2 | 0.597 |
| rho0.95 | 88.8 | 0.997 | 187.9909 | 35430059497.71 | 1.0 | 0.569 |

### Why does B0 outperform SU3 despite worse minority survival?

B0 Gaussian macro F1 (0.569) exceeds SU3 (0.556) despite SU3 improving minority survival (Δ=+0.406). SU3 tends to retain larger balls (Δ radius=+201.5994). SU3 reduces the margin proxy (Δ=-0.0473), weakening class separation geometry. SU3 increases mixed-ball fraction (Δ=+0.6868). Interpretation: minority sample retention alone does not yield better balls for Gaussian-weighted twin hyperplanes; geometry (radius, margin, heterogeneity) appears to dominate macro classification.

## 3. Optimal granular properties

**Verdict: C** — Top performers carry more heterogeneous (mixed-label) balls.

| profile | mean purity | mean radius | margin proxy | fraction mixed | mean density |
|---|---:|---:|---:|---:|---:|
| top macro-F1 half | 0.959 | 103.1457 | 1.488 | 0.352 | 27894590173.07 |
| bottom macro-F1 half | 0.949 | 331.8212 | 1.627 | 0.199 | 22516870307.75 |

Options: **A** high purity + small radius | **B** moderate purity + strong margin | **C** large heterogeneous support | **D** no clear relationship

## 4. Failure mechanism analysis (SU3)

SU3 failure co-occurs with: radius_increases, heterogeneity_increases, margin_decreases, density_decreases (macro F1 Δ=-0.0201 vs B0).

| metric | SU3 − B0 (paired mean) |
|---|---:|
| mean_radius | +201.5994 |
| fraction_mixed_balls | +0.6868 |
| mean_margin_proxy | -0.0473 |
| mean_density | -34179093355.9360 |
| n_heterogeneous_balls | +25.5686 |
| mean_purity | -0.2348 |
| minority_survival | +0.4058 |
| macro_f1 | -0.0201 |

### Mechanism checks

- radius_increases: **yes**
- heterogeneity_increases: **yes**
- margin_decreases: **yes**
- density_decreases: **yes**

## Outputs

- `results.csv`
- `summary.json`

