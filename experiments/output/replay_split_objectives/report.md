# Split-objective replay diagnostic

Read-only counterfactual: can boundary-aware split acceptance reproduce rho=0.90 granulation without an explicit purity cutoff?

Runs: **408** (8 configs × dataset × seed)

## Research questions

### 1. Can a utility objective reproduce rho=0.90 without using rho?

**No.** Per-split utility gates cannot reproduce rho0.90 without an explicit rho cutoff. The best utility rule (rule1_purity_gain) accepts only 1.3 splits/run vs 239.7 for rho0.90 (2.3 vs 77.8 balls). Elevated macro F1 (0.688) on utility rules is an artefact of near-trivial granulation (1–3 balls), not rho0.90-equivalent structure. Geometry distance to rho0.90 remains large (mean |Δ|=313.3).

### 2. Which terms are necessary?

Necessary terms for rho0.90-equivalent structure: all per-split utility rules collapse to 1–3 balls (fail to granulate); purity-only accepts even fewer splits than boundary-only; minority survival term changes acceptance rate; fragmentation penalty further reduces splits (0.4 vs 1.7); rho0.90 actual (rule5) is the only config with realistic split depth (~240 splits). Classifier ranking among rho baselines: rho0.90 (0.597) > rho0.85 (0.582) > rho0.95 (0.569).

### 3. Which existing assumption causes failure?

The failing assumption is **per-split greediness without global depth control**: utility rules accept 1.3 splits/run (near root-only), while rho0.95 over-splits at 268 (macro F1 0.569 vs rho0.90 0.597). Purity-gain gating rejects almost all KMeans proposals (weighted child purity rarely exceeds parent at a single step). Boundary-gain gating is similarly too conservative. What works is **stopping recursion near ρ≈0.90** — a global depth/purity ceiling — not local accept/reject on individual split quality. Future algorithms need cumulative fragmentation budgets, not myopic per-split utility.

## Configuration comparison (means)

| config | balls | splits | min surv | boundary | het | radius | macro F1 | min F1 | gmean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rule1_purity_gain | 2.3 | 1.3 | 1.000 | 47.423 | 0.968 | 1492.26 | 0.688 | 0.646 | 0.674 |
| rule2_boundary_gain | 2.6 | 1.8 | 1.000 | 59.953 | 0.921 | 1400.66 | 0.550 | 0.527 | 0.533 |
| rule3_boundary_minority | 2.6 | 1.7 | 1.000 | 59.938 | 0.921 | 1405.89 | 0.559 | 0.530 | 0.545 |
| rule4_boundary_minority_cost | 1.4 | 0.4 | 1.000 | 60.125 | 0.977 | 1615.08 | 0.618 | 0.580 | 0.612 |
| rule5_rho090_actual | 77.8 | 239.7 | 0.620 | 2.100 | 0.149 | 206.01 | 0.597 | 0.468 | 0.475 |
| rho0.80 | 50.4 | 141.3 | 0.807 | 24.957 | 0.506 | 908.02 | 0.576 | 0.478 | 0.436 |
| rho0.85 | 66.1 | 201.1 | 0.689 | 3.243 | 0.266 | 219.30 | 0.582 | 0.465 | 0.466 |
| rho0.95 | 88.8 | 267.7 | 0.583 | 1.496 | 0.087 | 187.99 | 0.569 | 0.427 | 0.434 |

## Geometry distance to rho0.90 (mean |Δ| per metric)

| config | n_balls | split_count | boundary | heterogeneity | radius |
|---|---:|---:|---:|---:|---:|
| rule1_purity_gain | 75.51 | 238.37 | 45.32 | 0.8185 | 1286.25 |
| rule2_boundary_gain | 75.16 | 237.94 | 57.85 | 0.7723 | 1194.65 |
| rule3_boundary_minority | 75.16 | 238.00 | 57.84 | 0.7718 | 1199.89 |
| rule4_boundary_minority_cost | 76.43 | 239.33 | 58.03 | 0.8283 | 1409.07 |
| rule5_rho090_actual | 0.00 | 0.00 | 0.00 | 0.0000 | 0.00 |
| rho0.80 | 27.43 | 98.37 | 22.86 | 0.3572 | 702.70 |
| rho0.85 | 11.92 | 38.59 | 1.14 | 0.1182 | 26.01 |
| rho0.95 | 11.71 | 28.96 | 0.60 | 0.0619 | 19.75 |

## Paired Δ vs rho0.90 actual (macro F1)

| config | Δ macro F1 | Δ split_count | Δ n_balls |
|---|---:|---:|---:|
| rule1_purity_gain | +0.0111 | -238.4 | -75.5 |
| rule2_boundary_gain | -0.0581 | -237.9 | -75.2 |
| rule3_boundary_minority | -0.0492 | -238.0 | -75.2 |
| rule4_boundary_minority_cost | -0.0165 | -239.3 | -76.4 |
| rho0.80 | -0.0293 | -98.4 | -27.4 |
| rho0.85 | -0.0144 | -38.6 | -11.7 |
| rho0.95 | -0.0274 | +28.0 | +11.0 |

## Outputs

- `results.csv`
- `summary.json`

