# Branch-priority refinement replay diagnostic

Read-only counterfactual: can adaptive split prioritization + global split budget reproduce rho=0.90 granulation without BFS ordering?

Policies: ['rho90_actual', 'purity_priority', 'entropy_priority', 'boundary_priority', 'combined_priority']
Budgets: [50, 75, 100, 125]% of rho=0.90 split count
Runs: **1020** (5 policies × 4 budgets × dataset × seed)

## Research questions

### 1. Can branch prioritization reproduce rho=0.90?

**Yes at 100% budget; partially at lower budgets.** Applying the full rho=0.90 split set (240 splits mean) yields identical final balls regardless of ordering — all policies converge to rho90_actual (macro F1 0.597). At 50% budget, rho90_actual F1=0.514 vs best adaptive 0.544 — ordering matters when the budget is partial.

### 2. Which signals best allocate refinement?

Signal ranking at 100% budget (macro F1, then geometry distance): combined_priority (F1=0.597, |Δ|geom=0.00), boundary_priority (F1=0.597, |Δ|geom=0.00), purity_priority (F1=0.597, |Δ|geom=0.00), entropy_priority (F1=0.597, |Δ|geom=0.00). **combined_priority** best matches rho90 geometry; **purity_priority** best among adaptive policies on classification.

### 3. Is budget + priority sufficient?

At 100% split budget, all policies apply the complete rho=0.90 split tree and converge to macro F1=0.597. At 50% budget, rho90_actual F1=0.514 vs best adaptive 0.544 (purity/entropy priority). **Only at partial budgets** — budget + priority changes which branches refine first; once the full rho=0.90 split set is applied, ordering is irrelevant.

### 4. Does rho remain better than adaptive replay?

At 100% budget all policies tie (macro F1=0.597) because the full rho=0.90 split tree is applied. At partial budgets, adaptive policies win in 12/16 policy×budget aggregates — primarily 50% (purity/entropy priority F1=0.544 vs rho 0.514). **Adaptive replay can beat BFS rho at sub-100% budgets**, but rho BFS is not uniquely optimal — impurity-first ordering matches or exceeds it at 50%.

## Policy × budget (aggregate means)

| policy | budget% | balls | splits | min surv | purity | boundary | het | depth | macro F1 | min F1 | gmean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rho90_actual | 50 | 64.3 | 119.8 | 0.876 | 0.860 | 1307189546.110 | 0.545 | 3.3 | 0.514 | 0.368 | 0.358 |
| rho90_actual | 75 | 79.1 | 179.9 | 0.769 | 0.906 | 2.758 | 0.383 | 3.9 | 0.520 | 0.375 | 0.352 |
| rho90_actual | 100 | 77.8 | 239.7 | 0.620 | 0.992 | 2.100 | 0.149 | 4.2 | 0.597 | 0.468 | 0.475 |
| rho90_actual | 125 | 87.0 | 263.4 | 0.593 | 0.994 | 1.807 | 0.111 | 4.3 | 0.571 | 0.429 | 0.438 |
| purity_priority | 50 | 47.8 | 119.8 | 0.831 | 0.922 | 2450980395.993 | 0.490 | 3.8 | 0.544 | 0.435 | 0.425 |
| purity_priority | 75 | 62.0 | 179.9 | 0.723 | 0.959 | 2.899 | 0.319 | 4.1 | 0.566 | 0.443 | 0.427 |
| purity_priority | 100 | 77.8 | 239.7 | 0.620 | 0.992 | 2.100 | 0.149 | 4.2 | 0.597 | 0.468 | 0.475 |
| purity_priority | 125 | 86.6 | 263.4 | 0.592 | 0.993 | 1.812 | 0.113 | 4.3 | 0.576 | 0.435 | 0.443 |
| entropy_priority | 50 | 47.8 | 119.8 | 0.831 | 0.922 | 2450980395.993 | 0.490 | 3.8 | 0.544 | 0.435 | 0.425 |
| entropy_priority | 75 | 62.1 | 179.9 | 0.723 | 0.959 | 2.899 | 0.319 | 4.1 | 0.566 | 0.443 | 0.427 |
| entropy_priority | 100 | 77.8 | 239.7 | 0.620 | 0.992 | 2.100 | 0.149 | 4.2 | 0.597 | 0.468 | 0.475 |
| entropy_priority | 125 | 86.6 | 263.4 | 0.592 | 0.993 | 1.812 | 0.113 | 4.3 | 0.576 | 0.435 | 0.443 |
| boundary_priority | 50 | 72.9 | 119.8 | 0.893 | 0.851 | 3.493 | 0.590 | 4.0 | 0.524 | 0.382 | 0.358 |
| boundary_priority | 75 | 79.5 | 179.9 | 0.773 | 0.907 | 4357298477.564 | 0.395 | 4.2 | 0.542 | 0.401 | 0.378 |
| boundary_priority | 100 | 77.8 | 239.7 | 0.620 | 0.992 | 2.100 | 0.149 | 4.2 | 0.597 | 0.468 | 0.475 |
| boundary_priority | 125 | 87.0 | 263.4 | 0.594 | 0.990 | 1.703 | 0.115 | 4.3 | 0.582 | 0.448 | 0.459 |
| combined_priority | 50 | 70.2 | 119.8 | 0.872 | 0.859 | 3.517 | 0.600 | 3.9 | 0.535 | 0.408 | 0.382 |
| combined_priority | 75 | 74.3 | 179.9 | 0.745 | 0.925 | 2.692 | 0.401 | 4.1 | 0.558 | 0.416 | 0.398 |
| combined_priority | 100 | 77.8 | 239.7 | 0.620 | 0.992 | 2.100 | 0.149 | 4.2 | 0.597 | 0.468 | 0.475 |
| combined_priority | 125 | 87.1 | 263.4 | 0.593 | 0.991 | 1.699 | 0.114 | 4.3 | 0.582 | 0.448 | 0.460 |

## Geometry distance to rho90_actual at 100% budget (mean |Δ|)

| policy | n_balls | splits | boundary | het | depth | purity | min surv | mean |Δ| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rho90_actual | 0.00 | 0.00 | 0.000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.00 |
| purity_priority | 0.00 | 0.00 | 0.000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.00 |
| entropy_priority | 0.00 | 0.00 | 0.000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.00 |
| boundary_priority | 0.00 | 0.00 | 0.000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.00 |
| combined_priority | 0.00 | 0.00 | 0.000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.00 |

## Paired Δ vs rho90_actual (macro F1) at 100% budget

| policy | Δ macro F1 | Δ n_balls | Δ minority_survival | Δ mean_depth |
|---|---:|---:|---:|---:|
| purity_priority | +0.0000 | +0.0 | +0.0000 | +0.00 |
| entropy_priority | +0.0000 | +0.0 | +0.0000 | +0.00 |
| boundary_priority | +0.0000 | +0.0 | +0.0000 | +0.00 |
| combined_priority | +0.0000 | +0.0 | +0.0000 | +0.00 |

## Outputs

- `results.csv`
- `summary.json`

