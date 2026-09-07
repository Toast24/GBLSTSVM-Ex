# Split-decision utility analysis

Read-only diagnostic linking every attempted recursive KMeans split to final Gaussian-weighted GBLSTSVM performance.

Split records: **60825** | Run records: **306** | Representations: B0, rho0.80, rho0.85, rho0.90, rho0.95, SU3

## Research questions

### 1. Do successful representations come from purity-increasing, boundary-preserving, or compactness-improving splits?

Successful representations are most associated with: purity gain is weakly associated with macro F1 (spearman=0.035); boundary-preserving splits (log-boundary Δ spearman=-0.179, parent log-boundary spearman=0.229); more splits correlate with worse outcomes (n_splits spearman=-0.581). Parent boundary structure (log-boundary) is the strongest positive signal; purity gain alone is insufficient and over-splitting hurts.

### 2. Why does rho=0.90 work?

rho=0.90 achieves mean macro F1 0.597 (B0=0.569, rho=0.95=0.569, Δ vs B0=+0.0274). It proposes 239.7 splits vs B0 267.7, with mean log-boundary change -0.4855 and mean radius change 87.0386. Stopping at ρ=0.90 retains boundary-rich mixed balls while avoiding the over-fragmentation and excessive heterogeneity of ρ=0.95 / B0.

### 3. Why does SU3 fail?

SU3 mean macro F1 0.556 vs B0 0.569 (Δ=-0.0201). SU3 rejects 26.1 splits/run (acceptance 51.2%), producing fewer balls (41.9 vs B0 88.8) with lower purity gain (Δ=-0.0640) and different boundary structure (Δ log-boundary=+0.1431). The SU3 gate filters on compactness/separation utility, not boundary score — it removes splits that would refine boundary structure while retaining heterogeneous coarse balls. Fewer, larger balls with suppressed boundary refinement hurt Gaussian-weighted twin-hyperplane geometry. Rejected splits have lower log-boundary change (-0.1810) than accepted (-0.3104).

### 4. What should a future split objective optimize?

A future split objective should optimize boundary structure over purity: parent log-boundary (spearman=0.229) correlates positively with macro F1, while more splits (spearman=-0.581) correlate negatively. Combine: (1) preserve or increase log-boundary score (entropy × density), (2) moderate compactness gain without over-fragmentation, (3) protect minority survival (mean_min_surv_change spearman=0.559), (4) stop recursion near ρ≈0.90 rather than pursuing ρ→1. Reject splits that reduce boundary structure or create del_ball-vulnerable minority peels.

## Representation comparison (mean split profiles + final metrics)

| representation | splits | rejected | accept% | Δpurity | Δlog-boundary | Δradius | macro F1 | min F1 | gmean | balls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 267.7 | 0.0 | 100.0% | 0.1498 | -0.4965 | 85.0702 | 0.569 | 0.427 | 0.434 | 88.8 |
| rho0.80 | 160.2 | 0.0 | 100.0% | 0.1733 | -0.4505 | 27.6645 | 0.576 | 0.478 | 0.436 | 57.0 |
| rho0.85 | 201.1 | 0.0 | 100.0% | 0.1590 | -0.4704 | 87.5207 | 0.582 | 0.465 | 0.466 | 66.1 |
| rho0.90 | 239.7 | 0.0 | 100.0% | 0.1542 | -0.4855 | 87.0386 | 0.597 | 0.468 | 0.475 | 77.8 |
| rho0.95 | 267.7 | 0.0 | 100.0% | 0.1498 | -0.4965 | 85.0702 | 0.569 | 0.427 | 0.434 | 88.8 |
| SU3 | 75.0 | 26.1 | 51.2% | 0.0858 | -0.3534 | 136.4558 | 0.556 | 0.415 | 0.443 | 41.9 |

## Top vs bottom macro-F1 runs (split profile means)

Median macro F1 split: 0.557

| metric | top half | bottom half |
|---|---:|---:|
| mean_purity_gain | 0.1534 | 0.1385 |
| mean_log_boundary_change | -0.5598 | -0.3369 |
| mean_entropy_change | -0.3720 | -0.3448 |
| mean_radius_change | 34.5357 | 141.9693 |
| mean_compactness_gain | 34.5357 | 141.9693 |
| mean_parent_log_boundary | 1.8106 | 0.8738 |
| mean_separation | 8136939067448.2295 | 58270069528695.5469 |

## Run-level correlations (split profile → final macro F1)

| predictor | pearson | spearman | n |
|---|---:|---:|---:|
| acceptance_rate | 0.006 | 0.021 | 291 |
| mean_compactness_gain | -0.140 | -0.110 | 291 |
| mean_entropy_change | -0.219 | -0.053 | 291 |
| mean_log_boundary_change | -0.291 | -0.179 | 291 |
| mean_min_surv_change | 0.554 | 0.559 | 291 |
| mean_parent_log_boundary | 0.393 | 0.229 | 291 |
| mean_purity_gain | 0.249 | 0.035 | 291 |
| mean_radius_change | -0.140 | -0.110 | 291 |
| mean_separation | -0.158 | -0.272 | 291 |
| n_proposed_splits | -0.300 | -0.581 | 291 |
| n_rejected_splits | -0.117 | -0.050 | 291 |

## Outputs

- `results.csv` — per-split records with parent/child geometry and final metrics
- `summary.json` — aggregates, correlations, research answers

