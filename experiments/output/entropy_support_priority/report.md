# Entropy support-priority diagnostic

Read-only comparison of four online entropy refinement priority signals. Budget = rho90 split count; traces every selected split with parent/child geometry and links each run to classifier outcomes.

Runs: **170** dataset×seed pairs × **4** policies | Split records: **3228**

## Policy summary

| policy | macro F1 | minority F1 | G-mean | balls | collapse% | parent n | parent frac | split min surv | heap disagree% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| entropy | 0.614 | 0.504 | 0.575 | 4.5 | 55.3% | 317.8 | 0.613 | 1.000 | 0.0% |
| entropy_x_frac | 0.614 | 0.504 | 0.575 | 4.5 | 55.3% | 317.8 | 0.613 | 1.000 | 0.0% |
| entropy_x_sqrt_frac | 0.614 | 0.504 | 0.575 | 4.5 | 55.3% | 317.8 | 0.613 | 1.000 | 0.0% |
| entropy_x_log_n | 0.614 | 0.504 | 0.575 | 4.5 | 55.3% | 317.8 | 0.613 | 1.000 | 0.0% |

## Compactness vs reliability

- **entropy**: 4.5 balls, F1 0.614 (Δ vs entropy +0.000), collapse 55.3%
- **entropy_x_frac**: 4.5 balls, F1 0.614 (Δ vs entropy +0.000), collapse 55.3%
- **entropy_x_sqrt_frac**: 4.5 balls, F1 0.614 (Δ vs entropy +0.000), collapse 55.3%
- **entropy_x_log_n**: 4.5 balls, F1 0.614 (Δ vs entropy +0.000), collapse 55.3%

## 1. Does support-aware entropy avoid collapse?

Pure entropy collapses **55.3%** of runs (4.5 balls). **entropy_x_frac**: collapse 55.3% (entropy 55.3%), macro F1 0.614; **entropy_x_sqrt_frac**: collapse 55.3% (entropy 55.3%), macro F1 0.614; **entropy_x_log_n**: collapse 55.3% (entropy 55.3%), macro F1 0.614. Support-aware priorities do not materially reduce collapse on this sweep.

## 2. Which priority best balances ambiguity and coverage?

Best ambiguity–coverage tradeoff: **entropy** (macro F1 0.614, mean parent fraction 0.613, collapse 55.3%). Highest raw F1: **entropy** (0.614).

## 3. Are failures caused by selecting small unstable balls?

Entropy failures: parent n=49.5 vs success n=251.2; sample fraction failed=0.126 vs success=0.431. **Yes** — smaller parents are more likely to fail (negative size–failure correlation).

## 4. Can entropy remain compact while improving reliability?

Entropy baseline: 4.5 balls, F1 0.614, collapse 55.3%. **entropy**: 4.5 balls (Δ +0.0), F1 0.614 (Δ +0.000), collapse 55.3%. Compactness–reliability tradeoff remains; priority alone is insufficient.

## Outputs

- `results.csv`
- `summary.json`

