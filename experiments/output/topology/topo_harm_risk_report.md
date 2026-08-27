# Topo harm_risk diagnostic report

Read-only extension of Gaussian-membership and Topo diagnostics.
Fixed rules only — no tuning, no CV, no model changes.

## harm_risk definition

Per run:

```
μ_G = purity * exp(-(r / median_positive_radius)^2)
harm_risk = frac balls with compA>0.5 AND μ_G<run_median AND (n_opp<=2 OR R_opp==0)
```

Runs analyzed: **51**

## Correlations with Topo outcomes (Pearson r)

| Signal | Δ Macro F1 | Δ G-mean | Δ minority F1 | mean |Topo−CA| |
|---|---:|---:|---:|---:|
| n_opp alone (compA>0.5, n_opp<=2) | 0.063 | 0.187 | 0.225 | 0.393 |
| R_opp=0 alone (compA>0.5) | -0.185 | 0.010 | -0.034 | 0.637 |
| μ_G alone (compA>0.5, μ_G<median) | 0.088 | 0.021 | -0.041 | 0.469 |
| harm_risk (joint rule) | -0.098 | 0.008 | 0.058 | 0.397 |
| mean n_opp (run) | 0.249 | 0.080 | -0.133 | 0.235 |
| frac R_opp=0 (all balls) | -0.111 | 0.019 | 0.048 | 0.193 |

## Group means by R_opp category

| Category | n | harm_risk | frac_n_opp | frac_R0 hc | frac_low_μG hc | Δ Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| adversarial_r_opp | 19 | 0.109 | 0.153 | 0.074 | 0.198 | -0.092 |
| informative_r_opp | 29 | 0.113 | 0.214 | 0.047 | 0.194 | 0.059 |
| redundant_r_opp | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Adversarial vs informative separation (excludes redundant)

| Signal | mean adversarial | mean informative | Cohen's d | AUROC |
|---|---:|---:|---:|---:|
| n_opp alone (compA>0.5, n_opp<=2) | 0.153 | 0.214 | -0.564 | 0.356 |
| R_opp=0 alone (compA>0.5) | 0.074 | 0.047 | 0.448 | 0.566 |
| μ_G alone (compA>0.5, μ_G<median) | 0.198 | 0.194 | 0.038 | 0.532 |
| harm_risk (joint rule) | 0.109 | 0.113 | -0.057 | 0.525 |

## Predictive separation ranking

Best AUROC (adversarial vs informative): **R_opp=0 alone (compA>0.5)** (AUROC=0.566)

AUROC order (higher = better separation):
- R_opp=0 alone (compA>0.5): 0.566
- μ_G alone (compA>0.5, μ_G<median): 0.532
- harm_risk (joint rule): 0.525
- n_opp alone (compA>0.5, n_opp<=2): 0.356

## Interpretation

R_opp=0 on high-compA balls alone matches or exceeds harm_risk for adversarial/informative separation; μ_G adds little beyond zero-reciprocity.

## Outputs

- `topo_harm_risk_runs.csv`
- `topo_harm_risk_per_ball.csv`
