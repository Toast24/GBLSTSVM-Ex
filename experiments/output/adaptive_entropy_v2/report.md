# Adaptive entropy granulation v2

Online live KMeans + entropy priority. v2 preserves failed-split parents (re-queue priority = entropy/(1+attempts), max_attempts=3).

Runs: **918**

## Research questions

### 1. Does candidate preservation recover replay performance?

Replay reference at 50% budget: entropy replay F1=0.544 vs BFS rho 0.514 (+0.030). v1 online: F1=0.614 (budget met 6%, splits 3.6). v2 online: F1=0.613 (budget met 6%, splits 3.8, paired Δ vs v1=-0.0003). **No** — candidate preservation alone does not recover replay gains.

### 2. Does entropy priority remain beneficial?

At 50% budget: v2 F1=0.613 vs rho 0.514 (Δ=+0.0977); v1 Δ=+0.0924. At 100%: v2=0.613 vs rho 0.597 (Δ=+0.0195). **Yes** — entropy priority remains beneficial once candidate paths are preserved.

### 3. How many splits are required to match rho90?

Full rho90: F1=0.597 at 240 splits. v2 closest match at 50% budget: F1=0.613 at 4 splits (2% of rho90). v1 closest: 4 splits at 50% budget. **Yes** — v2 matches rho90 F1 with fewer splits than full rho.

### 4. Is the problem solved by scheduling + state management?

v2 at 100% budget: splits=3.8/240 (met 6%), requeued=0.8, abandoned=0.3. v1 met 6% at 3.6 splits. **No** — scheduling + state management alone does not solve exhaustion.

## Configuration summary (aggregate means)

| config | splits | budget | met% | requeued | balls | depth | macro F1 | min F1 | gmean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-rho90 | 239.7 | nan | 1.00 | 0.0 | 77.8 | 3.7 | 0.597 | 0.468 | 0.475 |
| baseline-rho90-50pct | 119.8 | 119.8 | 1.00 | 0.0 | 64.3 | 3.0 | 0.514 | 0.368 | 0.358 |
| baseline-rho90-100pct | 239.7 | 239.7 | 1.00 | 0.0 | 77.8 | 3.7 | 0.597 | 0.468 | 0.475 |
| adaptive-entropy-v1-50pct | 3.6 | 119.8 | 0.06 | 0.0 | 4.6 | 1.8 | 0.614 | 0.497 | 0.582 |
| adaptive-entropy-v2-50pct | 3.8 | 119.8 | 0.06 | 0.8 | 4.8 | 1.9 | 0.613 | 0.491 | 0.577 |
| adaptive-entropy-v1-100pct | 3.6 | 239.7 | 0.06 | 0.0 | 4.6 | 1.8 | 0.614 | 0.497 | 0.582 |
| adaptive-entropy-v2-100pct | 3.8 | 239.7 | 0.06 | 0.8 | 4.8 | 1.9 | 0.613 | 0.491 | 0.577 |

## Paired Δ macro F1

| comparison | Δ |
|---|---:|
| v2 − rho @ 50% | +0.0977 |
| v1 − rho @ 50% | +0.0924 |
| v2 − v1 @ 50% | -0.0003 |
| v2 − rho @ 100% | +0.0195 |

## Splits to match full rho90 F1

- v2: {'closest_budget_pct': 50, 'mean_macro_f1': 0.6126213739924444, 'mean_split_count': 3.803921568627451, 'rho90_split_count': 239.7058823529412, 'split_fraction_of_rho90': 0.015869120654396725}
- v1: {'closest_budget_pct': 50, 'mean_macro_f1': 0.6143008181928805, 'mean_split_count': 3.607843137254902, 'rho90_split_count': 239.7058823529412, 'split_fraction_of_rho90': 0.015051124744376278}

## Outputs

- `results.csv`
- `summary.json`

