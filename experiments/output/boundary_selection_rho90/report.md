# Boundary selection validation (rho=0.90)

Controlled read-only validation: does practical boundary-based ball selection improve the best existing representation under Gaussian purity_compact?

Runs: **357** (51 dataset×seed × 7 configs)

## Configuration means

| config | balls | purity | mixed | boundary | macro F1 | min F1 | gmean | Δ macro vs all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rho90-all | 77.8 | 0.992 | 0.149 | 2.100 | 0.597 | 0.468 | 0.475 | +0.0000 |
| rho90-boundary_score-top50% | 39.3 | 0.987 | 0.233 | 3.096 | 0.601 | 0.506 | 0.510 | +0.0044 |
| rho90-boundary_score-top75% | 58.8 | 0.991 | 0.175 | 2.399 | 0.598 | 0.478 | 0.479 | +0.0008 |
| rho90-boundary_compact-top50% | 39.3 | 0.987 | 0.233 | 3.096 | 0.601 | 0.506 | 0.510 | +0.0044 |
| rho90-boundary_compact-top75% | 58.8 | 0.991 | 0.175 | 2.399 | 0.598 | 0.478 | 0.479 | +0.0008 |
| random-top50% | 39.3 | 0.991 | 0.172 | 2.754 | 0.571 | 0.462 | 0.490 | -0.0253 |
| random-top75% | 58.8 | 0.992 | 0.145 | 2.019 | 0.596 | 0.479 | 0.489 | -0.0006 |

## 1. Does practical boundary selection improve rho=0.90?

Boundary selection on rho=0.90: best config **rho90-boundary_score-top50%** macro F1 0.601 vs all-balls 0.597 (Δ=+0.0044). No — gains are negligible or negative on average.

## 2. Which criterion is better: boundary_score or boundary_compact?

At 50/75%: boundary_score mean macro F1 0.599, boundary_compact 0.599. **boundary_score** is slightly better overall; gaps are small.

## 3. Is ball selection a meaningful extension direction?

Selection beats random (Δ macro F1 vs all: boundary best +0.0044, random50 -0.0253, random75 -0.0006). Marginal — selection helps modestly; not a large standalone gain.

## 4. Remaining bottleneck: generation, selection, or both?

**generation** — Best boundary selection nears oracle; further gains require better ball generation.

## Outputs

- `results.csv`
- `summary.json`

