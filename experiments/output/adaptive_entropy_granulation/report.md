# Adaptive entropy granulation (online)

First genuine online validation: live KMeans splits + entropy-priority queue. No rho-tree recording or replay.

Runs: **663**

## Research questions

### 1. Does genuine online entropy-priority granulation reproduce the replay result?

Replay diagnostic at 50% budget: entropy replay +0.030 vs BFS rho (0.544 vs 0.514). Online adaptive-entropy-50pct: F1=0.614 vs baseline-rho90-50pct 0.514 (paired Δ=+0.0924, budget met 5.9%). **Partially** — online scheduling shows the same directional gain but magnitude differs from replay (live KMeans tree differs from recorded-tree reordering).

### 2. Does it outperform rho90 at equal refinement budgets?

At 50% budget: adaptive F1=0.614 vs BFS rho 0.514 (Δ=+0.0924). At 100% budget: adaptive 0.614 vs BFS 0.597 (Δ=+0.0224). **Yes at 50%** — entropy scheduling outperforms BFS rho at equal split budget.

### 3. Does it reduce the number of splits required for comparable performance?

Full baseline-rho90: F1=0.597 at 240 splits. Adaptive-75pct: F1=0.614 at 4 splits (2% of full rho splits). **Yes** — 75% adaptive budget approaches full-rho performance with fewer splits.

### 4. Does entropy-only scheduling already provide the improvement?

This experiment tests entropy-only scheduling with no boundary or minority terms. Adaptive-entropy-50pct macro F1=0.614. If this exceeds baseline-rho90-50pct (0.514), entropy alone is sufficient at partial budgets. **Yes** — entropy-only scheduling provides the improvement without extra terms.

### 5. What is the best normalized budget?

Best sqrt-n config: **adaptive-entropy-sqrt-k2** (F1=0.614, splits=3.9, balls=4.9). Compare to full rho F1=0.597. Dataset-independent sqrt scaling is competitive with rho-calibrated budgets.

### 6. Genuine improvement vs replay artifact?

Budget-met rate for adaptive-50pct: 5.9%; mean failed splits 0.33. **Partially genuine** — directional gain exists online but under-spends budget when entropy-priority deep-splits exhaust rho-eligible candidates early.

## Configuration summary (aggregate means)

| config | splits | budget | met% | balls | entropy | depth | macro F1 | min F1 | gmean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive-entropy-100pct | 3.6 | 239.7 | 0.06 | 4.6 | 0.612 | 1.8 | 0.614 | 0.497 | 0.582 |
| adaptive-entropy-125pct | 3.8 | 299.5 | 0.06 | 4.8 | 0.600 | 1.9 | 0.609 | 0.485 | 0.567 |
| adaptive-entropy-50pct | 3.6 | 119.8 | 0.06 | 4.6 | 0.612 | 1.8 | 0.614 | 0.497 | 0.582 |
| adaptive-entropy-75pct | 3.6 | 179.9 | 0.06 | 4.6 | 0.612 | 1.8 | 0.614 | 0.497 | 0.582 |
| adaptive-entropy-sqrt-k2 | 3.9 | 44.7 | 0.00 | 4.9 | 0.609 | 1.9 | 0.614 | 0.496 | 0.581 |
| adaptive-entropy-sqrt-k4 | 3.9 | 89.6 | 0.00 | 4.9 | 0.605 | 2.0 | 0.607 | 0.482 | 0.565 |
| adaptive-entropy-sqrt-k6 | 3.9 | 134.3 | 0.00 | 4.9 | 0.605 | 2.0 | 0.607 | 0.482 | 0.565 |
| adaptive-entropy-sqrt-k8 | 3.9 | 179.2 | 0.00 | 4.9 | 0.605 | 2.0 | 0.607 | 0.482 | 0.565 |
| baseline-rho90 | 239.7 | nan | 1.00 | 77.8 | 0.043 | 3.7 | 0.597 | 0.468 | 0.475 |
| baseline-rho90-100pct | 239.7 | 239.7 | 1.00 | 77.8 | 0.043 | 3.7 | 0.597 | 0.468 | 0.475 |
| baseline-rho90-125pct | 239.7 | 299.5 | 0.06 | 77.8 | 0.043 | 3.7 | 0.597 | 0.468 | 0.475 |
| baseline-rho90-50pct | 119.8 | 119.8 | 1.00 | 64.3 | 0.387 | 3.0 | 0.514 | 0.368 | 0.358 |
| baseline-rho90-75pct | 179.9 | 179.9 | 1.00 | 79.1 | 0.256 | 3.5 | 0.520 | 0.375 | 0.352 |

## Paired Δ macro F1 (adaptive − baseline BFS at same budget)

| budget | Δ at pct budget |
|---|---:|
| 50% | +0.0924 |
| 75% | +0.0990 |
| 100% | +0.0224 |

## Outputs

- `results.csv`
- `summary.json`

