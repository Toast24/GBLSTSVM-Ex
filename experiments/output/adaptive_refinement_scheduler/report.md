# Adaptive refinement scheduler experiment

New granulation wrapper: priority-queue split scheduling under a global split budget (no changes to solver, weighting, or membership).

Policies: ['baseline_rho90', 'entropy_priority', 'boundary_priority', 'combined_priority']
Budgets: [50, 75, 100]% of rho=0.90 split count
Runs: **612**

## Research questions

### 1. Can adaptive scheduling beat rho90 under equal split budget?

At 100% split budget, baseline_rho90 macro F1=0.597 (77.8 balls). Best adaptive (entropy_priority): 0.597 (Δ=+0.0000, balls 77.8). **Marginal at 100%** — equal split budgets yield near-identical F1; scheduling differences emerge at partial budgets. At 50% budget: baseline F1=0.514 vs entropy_priority 0.544 (Δ=+0.0297).

### 2. Which priority signal is best?

Best policy×budget cells by macro F1: entropy_priority@100%=0.597, boundary_priority@100%=0.597, combined_priority@100%=0.597, entropy_priority@75%=0.566, combined_priority@75%=0.561, entropy_priority@50%=0.544. **entropy_priority** is the strongest priority signal overall.

### 3. Does adaptive scheduling reduce required splits?

At 75% budget, best adaptive (entropy_priority) macro F1=0.566 vs baseline 100% F1=0.597 (25% fewer splits). **Partially** — 25% split reduction costs 0.031 macro F1; priority allocation improves partial-budget efficiency but does not fully replace the full rho=0.90 split tree.

### 4. Is refinement allocation the real missing component?

Compared to BFS rho at equal split budgets, priority scheduling improves macro F1 by +0.0297 at 50% and +0.0000 at 100%. Branch replay diagnostics showed ordering matters under partial budgets; this experiment confirms that **refinement allocation (scheduling) is a real missing component** beyond purity stopping and ball-count budgets. Impurity/boundary-weighted scheduling concentrates early splits on high-entropy regions, improving limited-budget granulation.

## Policy × budget (aggregate means)

| policy | budget% | splits | balls | min surv | purity | boundary | depth | macro F1 | min F1 | gmean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_rho90 | 50 | 119.8 | 64.3 | 0.876 | 0.860 | 1307189546.110 | 3.3 | 0.514 | 0.368 | 0.358 |
| baseline_rho90 | 75 | 179.9 | 79.1 | 0.769 | 0.906 | 2.758 | 3.9 | 0.520 | 0.375 | 0.352 |
| baseline_rho90 | 100 | 239.7 | 77.8 | 0.620 | 0.992 | 2.100 | 4.2 | 0.597 | 0.468 | 0.475 |
| entropy_priority | 50 | 119.8 | 47.8 | 0.831 | 0.922 | 2450980395.993 | 3.8 | 0.544 | 0.435 | 0.425 |
| entropy_priority | 75 | 179.9 | 62.1 | 0.723 | 0.959 | 2.899 | 4.1 | 0.566 | 0.443 | 0.427 |
| entropy_priority | 100 | 239.7 | 77.8 | 0.620 | 0.992 | 2.100 | 4.2 | 0.597 | 0.468 | 0.475 |
| boundary_priority | 50 | 119.8 | 72.9 | 0.893 | 0.851 | 3.493 | 4.0 | 0.524 | 0.382 | 0.358 |
| boundary_priority | 75 | 179.9 | 79.5 | 0.773 | 0.907 | 4357298477.564 | 4.2 | 0.542 | 0.401 | 0.378 |
| boundary_priority | 100 | 239.7 | 77.8 | 0.620 | 0.992 | 2.100 | 4.2 | 0.597 | 0.468 | 0.475 |
| combined_priority | 50 | 119.8 | 67.4 | 0.866 | 0.865 | 3.514 | 3.9 | 0.538 | 0.415 | 0.387 |
| combined_priority | 75 | 179.9 | 72.4 | 0.742 | 0.930 | 2.688 | 4.1 | 0.561 | 0.423 | 0.408 |
| combined_priority | 100 | 239.7 | 77.8 | 0.620 | 0.992 | 2.100 | 4.2 | 0.597 | 0.468 | 0.475 |

## Paired Δ vs baseline_rho90 (macro F1)

| policy | 50% | 75% | 100% |
|---|---:|---:|---:|
| entropy_priority | +0.0297 | +0.0458 | +0.0000 |
| boundary_priority | +0.0095 | +0.0219 | +0.0000 |
| combined_priority | +0.0239 | +0.0404 | +0.0000 |

## Outputs

- `results.csv`
- `summary.json`

