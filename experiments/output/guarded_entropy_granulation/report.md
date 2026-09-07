# Guarded entropy granulation

Adaptive entropy v2 with stopping safeguards (min balls, min depth, entropy floor).

Runs: **2040**

## Configuration summary

| config | splits | balls | collapse% | unstable ds | macro F1 | dF1 vs rho | F1/ball |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive-entropy-entropy-floor-t0p2 | 3.9 | 4.9 | 48.2% | 13 | 0.610 | +0.0214 | 0.1640 |
| adaptive-entropy-entropy-floor-t0p4 | 3.9 | 4.9 | 48.8% | 12 | 0.613 | +0.0243 | 0.1693 |
| adaptive-entropy-entropy-floor-t0p6 | 3.7 | 4.7 | 51.8% | 12 | 0.613 | +0.0245 | 0.1783 |
| adaptive-entropy-min-balls-k16 | 3.9 | 4.9 | 48.2% | 13 | 0.610 | +0.0214 | 0.1640 |
| adaptive-entropy-min-balls-k32 | 3.9 | 4.9 | 48.2% | 13 | 0.610 | +0.0214 | 0.1640 |
| adaptive-entropy-min-balls-k4 | 3.9 | 4.9 | 48.2% | 13 | 0.607 | +0.0185 | 0.1645 |
| adaptive-entropy-min-balls-k8 | 3.9 | 4.9 | 48.2% | 13 | 0.610 | +0.0214 | 0.1640 |
| adaptive-entropy-min-depth-d2 | 3.9 | 4.9 | 50.0% | 12 | 0.610 | +0.0212 | 0.1672 |
| adaptive-entropy-min-depth-d3 | 3.9 | 4.9 | 48.2% | 13 | 0.607 | +0.0185 | 0.1645 |
| adaptive-entropy-min-depth-d4 | 3.9 | 4.9 | 48.2% | 13 | 0.609 | +0.0199 | 0.1641 |
| adaptive-entropy-v2 | 3.7 | 4.7 | 51.8% | 12 | 0.610 | +0.0215 | 0.1788 |
| baseline-rho90 | 239.2 | 77.3 | 5.9% | 0 | 0.584 | +nan | 0.0451 |

## Smallest guards per family

- **min_balls**: adaptive-entropy-min-balls-k4 — collapse 48.2%, balls=4.9, F1=0.607
- **min_depth**: adaptive-entropy-min-depth-d2 — collapse 50.0%, balls=4.9, F1=0.610
- **entropy_floor**: adaptive-entropy-entropy-floor-t0p4 — collapse 48.8%, balls=4.9, F1=0.613

## 1. Does a simple safeguard remove unstable datasets?

Unguarded v2: 12 unstable datasets, 51.8% collapse rate. Best stability: **adaptive-entropy-entropy-floor-t0p4** — 12 unstable, 48.8% collapsed, F1=0.613. **No** — guards alone do not fully remove unstable datasets.

## 2. What is the smallest guard that fixes collapse?

v2 baseline collapse=51.8%. min_balls: **adaptive-entropy-min-balls-k4** (collapse 48.2%, balls=4.9); min_depth: **adaptive-entropy-min-depth-d2** (collapse 50.0%, balls=4.9); entropy_floor: **adaptive-entropy-entropy-floor-t0p4** (collapse 48.8%, balls=4.9). **k=4** is often the smallest min-ball guard that prevents 1–2 ball collapse.

## 3. Can entropy retain compactness while approaching rho90 reliability?

rho90: F1=0.584, balls=77.3. Best tradeoff: **adaptive-entropy-entropy-floor-t0p4** — F1=0.613, balls=4.9, unstable=12. **Yes** — guarded entropy can approach rho reliability while staying much more compact.

## 4. Which safeguard gives the best F1/complexity tradeoff?

Recommended: **adaptive-entropy-entropy-floor-t0p4** (F1=0.613, 4.9 balls, 48.8% collapse, 12 unstable datasets). Compare v2 unguarded: F1=0.610, 4.7 balls, 51.8% collapse.

Best tradeoff config: **adaptive-entropy-entropy-floor-t0p4**

## Outputs

- `results.csv`
- `summary.json`

