# Split-utility × weighting ablation

Read-only diagnostic: SU3 (utility p75) granulation vs baseline B0, crossed with frozen CA / Topo / Gaussian / product weightings. Solver, prediction rule, splits, and c1/c2 unchanged (c1=100.0, c2=0.01, rule=reference).

Runs: **612** (51 dataset×seed × 2 granulation × 6 weighting).

## Research questions

### 1. Does SU3 improve classification with CA?

SU3 vs B0 with CA: macro F1 -0.1046, minority F1 -0.1117. No — SU3 does not reliably improve CA-weighted classification.

### 2. Does SU3 improve Topo?

SU3 vs B0 with Topo: macro F1 -0.0945, minority F1 -0.1127. No — Topo does not gain consistently from SU3 granulation.

### 3. Does SU3 make Gaussian membership useful?

SU3 vs B0 with Gaussian: macro F1 -0.0201, minority F1 +0.0078. Marginal — a small minority-F1 gain (+0.008 mean) but macro F1 still falls; Gaussian alone does not become clearly useful under SU3.

### 4. Which component is the bottleneck: granulation or weighting?

Bottleneck: **granulation** (larger effect size than weighting choice). |SU3−B0| mean macro F1 (0.0874) exceeds B0 weighting spread (0.0351). Ball construction moves metrics more than weighting choice. Interpretation: ball construction dominates downstream variance, but SU3 p75 does not fix weighted GBLSTSVM—CA/Topo/combined modes all lose macro F1 vs their B0 counterparts.

## Configuration means

| config | balls | min surv | del sing | het | mean w | w var | macro F1 | min F1 | gmean | acc | d macro vs B0 | d min F1 vs B0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-CA | 88.8 | 0.583 | 179.9 | 1.0 | 0.556 | 0.2188 | 0.537 | 0.386 | 0.371 | 0.671 | — | — |
| B0-CAGaussian | 88.8 | 0.583 | 179.9 | 1.0 | 0.219 | 0.0845 | 0.553 | 0.387 | 0.379 | 0.691 | — | — |
| B0-CATopo | 88.8 | 0.583 | 179.9 | 1.0 | 0.233 | 0.0874 | 0.534 | 0.376 | 0.367 | 0.672 | — | — |
| B0-CATopoGaussian | 88.8 | 0.583 | 179.9 | 1.0 | 0.102 | 0.0305 | 0.546 | 0.372 | 0.357 | 0.692 | — | — |
| B0-Gaussian | 88.8 | 0.583 | 179.9 | 1.0 | 0.414 | 0.1542 | 0.569 | 0.427 | 0.434 | 0.684 | — | — |
| B0-Topo | 88.8 | 0.583 | 179.9 | 1.0 | 0.275 | 0.1041 | 0.536 | 0.378 | 0.367 | 0.673 | — | — |
| SU3-CA | 41.9 | 0.989 | 8.0 | 26.6 | 0.607 | 0.1769 | 0.470 | 0.305 | 0.277 | 0.622 | -0.1046 | -0.1117 |
| SU3-CAGaussian | 41.9 | 0.989 | 8.0 | 26.6 | 0.184 | 0.0444 | 0.481 | 0.289 | 0.282 | 0.638 | -0.1076 | -0.1265 |
| SU3-CATopo | 41.9 | 0.989 | 8.0 | 26.6 | 0.316 | 0.1002 | 0.478 | 0.292 | 0.282 | 0.631 | -0.0907 | -0.1111 |
| SU3-CATopoGaussian | 41.9 | 0.989 | 8.0 | 26.6 | 0.103 | 0.0199 | 0.473 | 0.280 | 0.263 | 0.635 | -0.1068 | -0.1172 |
| SU3-Gaussian | 41.9 | 0.989 | 8.0 | 26.6 | 0.309 | 0.0619 | 0.556 | 0.415 | 0.443 | 0.667 | -0.0201 | +0.0078 |
| SU3-Topo | 41.9 | 0.989 | 8.0 | 26.6 | 0.362 | 0.1127 | 0.476 | 0.293 | 0.281 | 0.629 | -0.0945 | -0.1127 |

## Bottleneck summary

- B0 weighting macro F1 spread (best − worst): 0.0351
- Mean SU3 − B0 macro F1 (same weighting, paired): -0.0874

## Per-weight SU3 − B0 (paired means)

| weighting | d macro F1 | d minority F1 |
|---|---:|---:|
| CA | -0.1046 | -0.1117 |
| Topo | -0.0945 | -0.1127 |
| Gaussian | -0.0201 | +0.0078 |
| CATopo | -0.0907 | -0.1111 |
| CAGaussian | -0.1076 | -0.1265 |
| CATopoGaussian | -0.1068 | -0.1172 |

## Outputs

- `results.csv`
- `summary.json`

