# Minority repair replay (B0 vs hypothetical B1)

Read-only virtual replay. No src/, granulation, Topo, CW, or GBLSTSVM changes.

Runs: **51** | repairs proposed: **4663**

## B0 vs B1 (mean across runs)

| Metric | B0 | B1 | Delta (B1-B0) |
|---|---:|---:|---:|
| 1. minority sample survival | 0.5829 | 0.7733 | +0.1904 |
| 2. majority sample survival | 0.7498 | 0.8710 | +0.1211 |
| 3a. minority-label ball count | 34.5098 | 49.3922 | +14.8824 |
| 3b. majority-label ball count | 54.3333 | 70.7059 | +16.3725 |
| 4a. minority singleton-ball rate | 0.0000 | 0.0000 | +0.0000 |
| 4b. minority small-ball rate | 0.6824 | 0.7276 | +0.0453 |
| 5a. mean purity | 0.9971 | 0.9970 | -0.0000 |
| 5b. purity std | 0.0027 | 0.0031 | +0.0004 |
| 6a. mean radius | 187.9909 | 197.6022 | +9.6113 |
| 6b. radius std | 411.8877 | 356.7472 | -55.1405 |
| 7. minority-to-opposite support | 47.4338 | 38.2067 | -2.6169 |
| 8a. mean nearest opposite-ball dist | 804.6422 | 483.4791 | -321.1630 |
| 8b. mean nearest same-ball dist | 481.3011 | 471.7653 | -9.5358 |
| 8c. opp/same distance ratio | 1.6995 | 1.1834 | -0.5161 |
| 9. heterogeneous-ball separability | 1.7283 | 1.5634 | -0.0394 |
| 10. samples with changed representation | 0.0000 | 426.1961 | +426.1961 |

## Repair classification

- A (likely beneficial): **2839** (60.9%)
- B (ambiguous): **1822** (39.1%)
- C (likely harmful): **2** (0.0%)

## Geometric defensibility

MIXED / not clearly defensible: B1 improves minority survival, but minority-to-opposite support and cross-ball geometry weaken on average. This is closer to survival inflation than a geometrically principled granulation fix.

## Outputs

- `minority_repair_records.csv`
- `minority_repair_runs.csv`
