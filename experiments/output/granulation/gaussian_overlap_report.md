# Gaussian overlap split diagnostic

Read-only diagnostic on frozen granular balls (51 Topo screening runs).
No granulation or model code modified.

## Proposed statistic

Per heterogeneous ball (both classes present), fit diagonal Gaussians
N(mu_c, diag(sigma_c^2)) to in-ball samples per class (variance floor 1e-6).

**BC_overlap** = Bhattacharyya coefficient

```
BC = exp(-B),  B = 1/4 sum_j log((s1j^2+s2j^2)/(2*s1j*s2j))
              + 1/4 sum_j (mu1j-mu2j)^2/(s1j^2+s2j^2)
```

Structural interpretation (no outcome-tuned thresholds):
- High BC_overlap: class Gaussians overlap - genuine mixed structure.
- Low BC_overlap + high minority_hd1_norm: compact majority + isolated minority.

Companion signals on the same balls:
- minority_hd1_norm = median minority-point hd1 / ball radius
- kmeans_purity_gain = replay of current KMeans(k=2) split
- purity, radius, n_opp, R_opp from frozen Topo membership

Runs: **51** | heterogeneous balls: **51**

## Informative vs adversarial means (R_opp category)

| Category | n | BC_overlap | hd1_norm | purity(het) | kmeans_gain | frac isolated | Delta Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial_r_opp | 19 | 0.000 | 2.082 | 0.971 | 0.276 | 0.000 | -0.092 |
| informative_r_opp | 29 | 0.018 | 1.131 | 0.971 | 0.085 | 0.000 | 0.059 |
| redundant_r_opp | 3 | 0.238 | 1.162 | 0.963 | 0.027 | 0.000 | 0.000 |

## Correlation with Topo performance (Pearson r)

| Signal | Delta Macro F1 | Delta G-mean | Delta minority F1 |
|---|---:|---:|---:|
| BC_overlap (het mean) | 0.087 | 0.245 | 0.221 |
| minority_hd1_norm (het mean) | -0.340 | -0.394 | -0.343 |
| purity (het mean) | -0.088 | 0.192 | 0.209 |
| radius (het mean) | 0.385 | 0.421 | 0.597 |
| kmeans_purity_gain (het mean) | -0.303 | -0.347 | -0.450 |
| frac kmeans minority isolated | nan | nan | nan |
| mean n_opp | 0.249 | 0.080 | -0.133 |
| frac R_opp=0 | -0.111 | 0.019 | 0.048 |
| mean R_opp | 0.061 | -0.042 | -0.059 |

## Incremental information beyond purity + hd1

Partial corr(BC_overlap, Delta Macro F1 | purity, hd1_norm): **0.020**
Partial corr(hd1_norm, Delta Macro F1 | purity, BC_overlap): **-0.337**

| Predictor set | |r| with Delta Macro F1 |
|---|---:|
| purity alone | 0.088 |
| hd1_norm alone | 0.340 |
| BC_overlap alone | 0.087 |
| purity + hd1_norm (sum) | 0.341 |
| purity + hd1 + BC (sum) | 0.338 |

## Adversarial vs informative AUROC (higher = better separation)

| Signal | AUROC |
|---|---:|
| BC_overlap | 0.330 |
| minority_hd1_norm | 0.747 |
| purity | 0.505 |
| frac R_opp=0 | 0.515 |
| mean n_opp | 0.543 |

## Verdict

BC_overlap does not materially outperform purity + minority_hd1_norm for predicting Topo success/failure or separating adversarial from informative runs. A Gaussian-aware split experiment is not strongly justified by these diagnostics.

## Outputs

- `gaussian_overlap_runs.csv`
- `gaussian_overlap_per_ball.csv`
