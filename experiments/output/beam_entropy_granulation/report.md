# Beam entropy granulation

Read-only experiment: beam-search entropy refinement with global top-k candidate pool. Budget = rho90 split count.

Runs: **1020** (170 dataset×seed × 6 configs)

## Config summary

| config | macro F1 | minority F1 | G-mean | balls | splits | collapse% | budget met% | no-cand% | mean pool | max pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-rho90 | 0.584 | 0.463 | 0.456 | 77.3 | 239.2 | 5.9% | 100.0% | 0.0% | 0.00 | 0.0 |
| beam-entropy-k1 | 0.631 | 0.552 | 0.615 | 9.6 | 8.6 | 10.6% | 6.5% | 93.5% | 1.00 | 1.0 |
| beam-entropy-k2 | 0.593 | 0.496 | 0.550 | 14.7 | 13.7 | 8.2% | 7.6% | 92.4% | 1.64 | 1.9 |
| beam-entropy-k4 | 0.543 | 0.453 | 0.449 | 27.6 | 26.6 | 8.2% | 8.2% | 91.8% | 2.70 | 3.6 |
| beam-entropy-k8 | 0.515 | 0.392 | 0.354 | 54.8 | 53.8 | 8.2% | 8.2% | 91.8% | 4.38 | 6.6 |
| beam-entropy-k16 | 0.508 | 0.371 | 0.331 | 80.2 | 79.2 | 8.2% | 8.2% | 91.8% | 6.54 | 10.8 |

## Paired deltas vs rho90 (best beam)

- Best beam: **beam-entropy-k1** (k=1)
- Δ macro F1 vs rho90: +0.047
- Collapse reduction k1→best: 0.0pp

## 1. Does preserving a refinement frontier remove collapse?

rho90 collapse **5.9%**; beam-k1 **10.6%** (mean pool 1.00); beam-k1 **10.6%** (mean pool 1.00). **No** — preserving a wider frontier does not materially reduce collapse.

## 2. What beam width gives the best compactness/performance tradeoff?

Best tradeoff: **beam-entropy-k1** — macro F1 0.631, 9.6 balls, collapse 10.6%, mean pool 1.00. Modest beam widths suffice; very wide beams add pool overhead without proportional gain.

## 3. Does beam search approach rho90 reliability?

rho90: F1 0.584, collapse 5.9%, 77.3 balls. Best beam: F1 0.631 (Δ +0.047), collapse 10.6% (Δ +4.7%). Partial convergence — beam improves frontier maintenance but gap to rho90 remains.

## 4. Is the improvement from better search rather than better split scoring?

k=1 mean pool 1.00 vs k=1 pool 1.00 (+0.00). F1 gain k1→best: +0.000. **Unclear** — pool size barely increases; collapse may need structural changes beyond beam width.

## Outputs

- `results.csv`
- `summary.json`

