# Entropy compact representation validation

Multi-seed (0–9) read-only comparison: full rho90 BFS vs adaptive entropy v2.

Runs: **340** (170 dataset×seed pairs)

## Scheduler summary

| config | splits | balls | F1/ball | macro F1 | min F1 | gmean | depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline-rho90 | 239.2 | 77.3 | 0.0451 | 0.584 ± 0.197 | 0.463 | 0.456 | 3.6 |
| adaptive-entropy-v2 | 3.7 | 4.7 | 0.1788 | 0.610 ± 0.168 | 0.496 | 0.569 | 1.8 |

Adaptive split stability: mean=3.7, median=4, ≤10 splits in 99% of runs.

## Per-dataset classification (adaptive − rho90 macro F1)

| dataset | class | wins | losses | mean ΔF1 | adapt splits | rho splits |
|---|---|---:|---:|---:|---:|---:|
| tic_tac_toe | consistent_win | 8 | 0 | +0.2336 | 7.6 | 386.5 |
| messidor_features | consistent_win | 10 | 0 | +0.1237 | 6.5 | 508.0 |
| breast_cancer | unstable | 4 | 1 | +0.0929 | 5.4 | 106.8 |
| breast_cancer_wisconsin_diagnostic | consistent_win | 9 | 1 | +0.0782 | 1.6 | 49.1 |
| australian | unstable | 7 | 2 | +0.0748 | 1.4 | 288.1 |
| ecoli2 | unstable | 3 | 2 | +0.0437 | 2.7 | 19.2 |
| diabetes | unstable | 7 | 3 | +0.0426 | 5.0 | 255.1 |
| congressional_voting_records | unstable | 7 | 3 | +0.0298 | 3.5 | 26.8 |
| german_credit | unstable | 7 | 1 | +0.0268 | 6.3 | 433.9 |
| breast_cancer_wisconsin_original | unstable | 0 | 0 | +0.0000 | 1.0 | 1.0 |
| wholesale_customers | unstable | 0 | 1 | -0.0002 | 3.4 | 139.0 |
| ionosphere | unstable | 5 | 5 | -0.0072 | 3.8 | 42.8 |
| heart | unstable | 5 | 5 | -0.0550 | 4.6 | 120.3 |
| spambase | consistent_loss | 1 | 9 | -0.0675 | 3.8 | 1146.9 |
| vehicle0 | unstable | 0 | 2 | -0.0918 | 2.4 | 115.0 |
| credit_approval | consistent_loss | 2 | 8 | -0.1217 | 1.5 | 293.7 |
| chronic_kidney_disease | unstable | 0 | 3 | -0.2221 | 2.2 | 134.0 |

## 1. Is the ~4-split result stable?

Across 170 runs (17 datasets × 10 seeds): adaptive splits mean=3.7 (std=2.5, median=4, P10–P90=[1, 7]). Paired Δ macro F1 = +0.0215 ± 0.1336. Dataset outcomes: 3 consistent wins, 2 consistent losses, 12 unstable. **Mostly stable** — adaptive stays ≤10 splits in 99% of runs with positive mean ΔF1, but per-dataset variance exists.

## 2. Does entropy produce a compact but useful representation?

Adaptive: 4.7 balls, F1/ball=0.1788. rho90: 77.3 balls, F1/ball=0.0451 (ratio 4.0×). Mean purity adaptive=0.786 vs rho=0.991; entropy 0.590 vs 0.049. **Unclear** — compactness is real but per-ball efficiency alone does not explain gains.

## 3. Which datasets benefit?

Consistent wins (3): tic_tac_toe, messidor_features, breast_cancer_wisconsin_diagnostic. Consistent losses (2): spambase, credit_approval. Largest mean ΔF1: tic_tac_toe (+0.234). **Mixed** — benefits are dataset-specific, not a single benchmark artifact.

## 4. Is the improvement caused by regularization from fewer balls?

Mean ball reduction: 72.7; ΔF1 vs ball-reduction spearman=0.095. Mean Δ(f1_per_ball)=+0.1258. **Partially** — fewer balls co-occur with gains but correlation is weak; may reflect implicit structural regularization rather than ball count alone.

## 5. Is this a viable direction for a final method?

Aggregate: adaptive F1=0.610 vs rho=0.584 (Δ=+0.0215) at 3.7/239 splits. Unstable on 12/17 datasets. **Promising direction** — compact entropy scheduling is a viable research path, but requires per-dataset safeguards and fixing candidate exhaustion before production.

## Outputs

- `results.csv`
- `summary.json`

