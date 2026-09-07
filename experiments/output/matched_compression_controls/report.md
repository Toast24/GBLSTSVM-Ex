# Matched compression controls

Read-only diagnostic: subsample frozen rho90 to the online-entropy ball count and compare classifier outcomes under identical Gaussian evaluation.

Runs: **1190** rows (170 dataset×seed × 7 methods)

## Method summary

| method | macro F1 | minority F1 | G-mean | balls | F1/ball | collapse% | purity | entropy | boundary | mixed% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.597 | 0.529 | 0.580 | 4.7 | 0.159 | 51.2% | 0.992 | 0.044 | 2.96 | 0.15 |
| purity | 0.584 | 0.494 | 0.546 | 4.7 | 0.152 | 49.4% | 0.997 | 0.015 | 1.11 | 0.06 |
| entropy | 0.641 | 0.530 | 0.601 | 4.7 | 0.179 | 54.7% | 0.943 | 0.283 | 5.08 | 0.73 |
| boundary | 0.647 | 0.535 | 0.606 | 4.7 | 0.177 | 57.1% | 0.948 | 0.267 | 6.17 | 0.73 |
| stratified_random | 0.573 | 0.482 | 0.545 | 4.7 | 0.169 | 44.7% | 0.991 | 0.046 | 1.77 | 0.14 |
| full_rho90 | 0.584 | 0.463 | 0.456 | 77.3 | 0.045 | 5.9% | 0.991 | 0.049 | 2.51 | 0.16 |
| online_entropy | 0.610 | 0.496 | 0.569 | 4.7 | 0.179 | 51.8% | 0.786 | 0.590 | 15.57 | 0.78 |

## Paired online − control (macro F1)

- **random**: mean Δ -0.003, online wins 43.3%, losses 46.2%
- **purity**: mean Δ +0.010, online wins 47.5%, losses 42.6%
- **entropy**: mean Δ -0.029, online wins 34.7%, losses 54.1%
- **boundary**: mean Δ -0.029, online wins 35.5%, losses 52.7%
- **stratified_random**: mean Δ +0.010, online wins 40.5%, losses 51.9%
- **full_rho90**: mean Δ +0.022, online wins 57.3%, losses 35.1%

Online − full rho90: mean ΔF1 +0.022 (online 0.610 vs rho 0.584); collapse online 51.8% vs rho 5.9%.

## 1. Does entropy beat random matched compression?

Paired online − random: mean ΔF1 **-0.003**, online wins **43.3%** of pairs. **Mixed** — online entropy does not consistently dominate random at matched size.

## 2. Does entropy beat purity-based compression?

vs purity: ΔF1 **+0.010**, wins **47.5%**. **No** — purity subsampling is competitive with online entropy.

## 3. Does entropy beat boundary-based compression?

vs boundary: ΔF1 **-0.029**, wins **35.5%**. **No** — boundary selection matches online entropy at equal ball counts.

## 4. Does entropy beat random stratified compression?

vs stratified random: ΔF1 **+0.010**, wins **40.5%**. **Yes** — online entropy beats class-stratified random subsampling.

## 5. Compression vs selection

**Not solely compression** — controls do not fully match online entropy; some selection signal remains beyond ball-count reduction.

## 6. Structural region selection

**No consistent structural advantage** — online entropy does not reliably beat matched rho90 subsets; compactness drives the effect.

## 7. Dataset consistency

**0** datasets show consistent online advantage vs random (≥8/10 seeds); **17** are unstable. Largest mean ΔF1 vs random: breast_cancer_wisconsin_diagnostic (+0.153), chronic_kidney_disease (+nan), ionosphere (+0.106).

## Outputs

- `results.csv`
- `summary.json`

