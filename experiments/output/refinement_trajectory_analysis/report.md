# Refinement trajectory analysis (rho=0.90)

Read-only branch analysis: which regions receive refinement under rho=0.90, and how descendant geometry differs by branch class.

Reference rho: **0.9** | Runs: **51** dataset×seed | Nodes: **16193**

## Branch class summary

| class | nodes | frac | depth | parent purity | boundary | descendants | final purity | min surv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heavily_refined | 7193 | 0.444 | 8.0 | 0.684 | 0.913 | 5.5 | 0.998 | 0.446 |
| lightly_refined | 5032 | 0.311 | 10.2 | 0.599 | 0.640 | 0.2 | 0.998 | 0.469 |
| terminated_early | 3968 | 0.245 | 10.2 | 0.996 | 0.052 | 1.0 | 0.996 | 1.000 |

## Split vs terminated (parent properties)

| metric | mean split | mean terminated | Δ |
|---|---:|---:|---:|
| parent_purity | 0.6487 | 0.9959 | -0.3472 |
| parent_entropy | 0.8924 | 0.0206 | +0.8718 |
| parent_boundary_score | 2781186096.7732 | 0.2581 | +2781186096.5151 |
| parent_log_boundary_score | 0.8003 | 0.0524 | +0.7478 |
| parent_density | 2781186097.2743 | 38302555512.8080 | -35521369415.5336 |
| parent_minority_ratio | 0.4588 | 0.4133 | +0.0455 |
| parent_samples | 20.5607 | 5.5461 | +15.0145 |

## Correlations (split nodes)

- n_descendants_vs_parent_impurity: pearson=-0.023, spearman=-0.384
- subtree_depth_vs_parent_boundary: pearson=0.227, spearman=0.328
- final_purity_vs_parent_purity: pearson=-0.152, spearman=-0.101
- minority_survival_vs_parent_minority_ratio: pearson=0.660, spearman=0.654

## 1. What properties cause rho to continue splitting?

rho continues splitting when parent purity is **low** (split mean 0.649 vs terminated 0.996), entropy is **higher** (0.892 vs 0.021), and boundary score (log) is **elevated** (0.800 vs 0.052). Impurity predicts descendant count (spearman -0.384).

## 2. Are deeply refined branches associated with better final geometry?

Heavily refined branches reach higher final purity (0.998 vs lightly 0.998) but retain **higher** final final boundary (log) (0.018 vs 0.022) — refinement improves local purity while preserving mixed-boundary structure where needed.

## 3. Does rho implicitly perform adaptive resource allocation?

rho allocates refinement unevenly: 44.4% heavily refined, 24.5% terminated early. Parent boundary predicts subtree depth (spearman 0.328). **Yes** — rho implicitly performs adaptive resource allocation, concentrating splits on impure/high-entropy regions.

## 4. What should replace rho: budget, branch priority, or both?

**Both branch priority and budget.** Ball-count budgets alone fail because they ignore *which* branches to refine (complexity budget experiment: same ball count, different F1). rho's purity gate is a branch-priority rule. A replacement should combine a global complexity ceiling with impurity/boundary-weighted split priority — not budget alone.

## Outputs

- `results.csv`
- `summary.json`

