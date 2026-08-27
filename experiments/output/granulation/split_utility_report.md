# Split utility diagnostic

Read-only replay of frozen recursive KMeans(k=2) splits (purity=0.95, min_samples_per_ball=1, del_ball num_data=2).

Split geometry and del_ball fate only — **no Topo / CW / GBLSTSVM**.

## Headline

- Splits: 13654 across 51 runs
- Class counts: {'A': 6013, 'C': 4480, 'B': 2545, 'D': 616}
- AUROC gate: 0.8

## Main question

Does a non-tautological split-quality statistic separate harmful minority peels (A) from useful retained splits (C) with AUROC >= 0.8?

**Answer:** YES (best non-tautological: `center_radius_separation`, AUROC(C vs A)=0.864).

## Verdict

**1. principled split criterion supported**

Principled geometry metric `center_radius_separation` reaches AUROC=0.864 for C vs A (gate 0.8). parent_n AUROC=0.930, split_depth AUROC=0.708 (structural covariates, not required for separation).

## Depth / parent size

- A vs C pairs: 10493
- parent_n AUROC(C vs A): 0.930
- split_depth AUROC(C vs A): 0.708
- Depth/parent_n alone explains C vs A (>= gate): True

## Composite scores (pre-specified, no threshold tuning)

| composite | AUROC A | AUROC C vs A | tautological |
|---|---:|---:|---|
| composite_center_separation | 0.759 | 0.864 | False |
| composite_radius_plus_separation | 0.758 | 0.862 | False |
| composite_separation_over_compactness | 0.744 | 0.824 | False |
| composite_minority_margin | 0.669 | 0.660 | False |
| composite_margin_plus_separation | 0.537 | 0.633 | False |
| composite_radius_reduction | 0.529 | 0.532 | False |

## Metrics ranked by AUROC(C vs A)

| metric | AUROC A | AUROC C vs A | tautological | dir C vs A |
|---|---:|---:|---|---|
| either_child_deleted | 0.793 | 1.000 | True | lower |
| parent_n | 0.811 | 0.930 | False | higher |
| child_n2 | 0.761 | 0.920 | True | higher |
| child2_later_deleted | 0.758 | 0.913 | True | lower |
| child_n1 | 0.778 | 0.900 | True | higher |
| child1_later_deleted | 0.744 | 0.868 | True | lower |
| center_radius_separation | 0.759 | 0.864 | False | lower |
| composite_center_separation | 0.759 | 0.864 | False | lower |
| composite_radius_plus_separation | 0.758 | 0.862 | False | lower |
| either_child_singleton | 0.694 | 0.861 | True | lower |
| r2 | 0.709 | 0.833 | False | higher |
| child_compactness_gain | 0.750 | 0.829 | False | lower |
| composite_separation_over_compactness | 0.744 | 0.824 | False | lower |
| r1 | 0.719 | 0.800 | False | higher |
| weighted_child_radius | 0.736 | 0.779 | False | higher |
| center_distance_over_parent_r | 0.669 | 0.768 | False | lower |
| n_min_retained_child2 | 0.671 | 0.757 | True | higher |
| n_min_retained_child1 | 0.681 | 0.749 | True | higher |
| parent_purity | 0.703 | 0.716 | False | higher |
| purity_gain | 0.705 | 0.716 | False | lower |
| split_depth | 0.625 | 0.708 | False | lower |
| minority_concentrated_in_small | 0.776 | 0.695 | True | lower |
| purity2 | 0.597 | 0.661 | False | lower |
| minority_margin_mean | 0.669 | 0.660 | False | higher |
| composite_minority_margin | 0.669 | 0.660 | False | higher |

## Distributions by class (p10 / p50 / p90)

### Class A (n=6013)
- size_ratio: p10=0.250 p50=0.667 p90=1.000
- radius_reduction: p10=0.445 p50=3.115 p90=19.055
- center_radius_separation: p10=0.901 p50=2.874 p90=17644490879113.617
- minority_margin_mean: p10=-13.521 p50=-0.606 p90=5.701
- parent_n: p10=2.000 p50=3.000 p90=9.000
- split_depth: p10=6.000 p50=10.000 p90=14.000
- child_compactness_gain: p10=0.197 p50=0.537 p90=1.000

### Class B (n=2545)
- size_ratio: p10=0.143 p50=0.333 p90=0.500
- radius_reduction: p10=0.449 p50=2.615 p90=23.998
- center_radius_separation: p10=1.684 p50=2.517 p90=5.107
- minority_margin_mean: p10=-3.003 p50=2.119 p90=19.886
- parent_n: p10=3.000 p50=4.000 p90=8.000
- split_depth: p10=7.000 p50=10.000 p90=15.000
- child_compactness_gain: p10=0.221 p50=0.459 p90=0.689

### Class C (n=4480)
- size_ratio: p10=0.250 p50=0.615 p90=0.941
- radius_reduction: p10=0.208 p50=2.696 p90=48.445
- center_radius_separation: p10=0.565 p50=0.923 p90=1.786
- minority_margin_mean: p10=-5.884 p50=0.972 p90=16.660
- parent_n: p10=6.000 p50=16.000 p90=90.000
- split_depth: p10=3.000 p50=7.000 p90=12.000
- child_compactness_gain: p10=0.106 p50=0.218 p90=0.434

### Class D (n=616)
- size_ratio: p10=0.477 p50=0.667 p90=1.000
- radius_reduction: p10=0.321 p50=2.858 p90=22.766
- center_radius_separation: p10=0.697 p50=1.082 p90=1.926
- minority_margin_mean: p10=-6.535 p50=0.166 p90=9.633
- parent_n: p10=4.000 p50=6.000 p90=11.000
- split_depth: p10=6.000 p50=8.000 p90=12.000
- child_compactness_gain: p10=0.161 p50=0.289 p90=0.490

## Per-run counts

| dataset | seed | splits | A | B | C | D |
|---|---:|---:|---:|---:|---:|---:|
| australian | 0 | 318 | 159 | 54 | 87 | 18 |
| australian | 1 | 299 | 143 | 53 | 96 | 7 |
| australian | 42 | 303 | 162 | 50 | 82 | 9 |
| breast_cancer | 0 | 127 | 59 | 27 | 34 | 7 |
| breast_cancer | 1 | 129 | 55 | 32 | 36 | 6 |
| breast_cancer | 42 | 119 | 56 | 15 | 42 | 6 |
| breast_cancer_wisconsin_diagnostic | 0 | 59 | 29 | 8 | 19 | 3 |
| breast_cancer_wisconsin_diagnostic | 1 | 49 | 23 | 5 | 19 | 2 |
| breast_cancer_wisconsin_diagnostic | 42 | 52 | 25 | 3 | 22 | 2 |
| breast_cancer_wisconsin_original | 0 | 1 | 0 | 0 | 1 | 0 |
| breast_cancer_wisconsin_original | 1 | 1 | 0 | 0 | 1 | 0 |
| breast_cancer_wisconsin_original | 42 | 1 | 0 | 0 | 1 | 0 |
| chronic_kidney_disease | 0 | 147 | 49 | 42 | 51 | 5 |
| chronic_kidney_disease | 1 | 157 | 57 | 43 | 50 | 7 |
| chronic_kidney_disease | 42 | 146 | 47 | 41 | 51 | 7 |
| congressional_voting_records | 0 | 13 | 4 | 2 | 7 | 0 |
| congressional_voting_records | 1 | 56 | 14 | 11 | 29 | 2 |
| congressional_voting_records | 42 | 52 | 18 | 9 | 23 | 2 |
| credit_approval | 0 | 299 | 141 | 48 | 98 | 12 |
| credit_approval | 1 | 294 | 145 | 45 | 96 | 8 |
| credit_approval | 42 | 298 | 165 | 42 | 81 | 10 |
| diabetes | 0 | 289 | 141 | 45 | 83 | 20 |
| diabetes | 1 | 299 | 154 | 48 | 85 | 12 |
| diabetes | 42 | 295 | 140 | 52 | 84 | 19 |
| ecoli2 | 0 | 25 | 8 | 2 | 14 | 1 |
| ecoli2 | 1 | 27 | 7 | 3 | 16 | 1 |
| ecoli2 | 42 | 30 | 13 | 3 | 12 | 2 |
| german_credit | 0 | 476 | 233 | 96 | 118 | 29 |
| german_credit | 1 | 439 | 194 | 88 | 129 | 28 |
| german_credit | 42 | 473 | 239 | 87 | 117 | 30 |
| heart | 0 | 128 | 63 | 25 | 30 | 10 |
| heart | 1 | 132 | 67 | 27 | 33 | 5 |
| heart | 42 | 116 | 49 | 24 | 37 | 6 |
| ionosphere | 0 | 57 | 28 | 2 | 26 | 1 |
| ionosphere | 1 | 76 | 40 | 3 | 31 | 2 |
| ionosphere | 42 | 60 | 32 | 5 | 23 | 0 |
| messidor_features | 0 | 538 | 263 | 94 | 146 | 35 |
| messidor_features | 1 | 525 | 279 | 83 | 138 | 25 |
| messidor_features | 42 | 512 | 247 | 84 | 146 | 35 |
| spambase | 0 | 1406 | 524 | 301 | 531 | 50 |
| spambase | 1 | 1346 | 484 | 289 | 524 | 49 |
| spambase | 42 | 1346 | 478 | 265 | 550 | 53 |
| tic_tac_toe | 0 | 417 | 213 | 62 | 123 | 19 |
| tic_tac_toe | 1 | 438 | 234 | 62 | 125 | 17 |
| tic_tac_toe | 42 | 410 | 188 | 72 | 131 | 19 |
| vehicle0 | 0 | 141 | 56 | 23 | 58 | 4 |
| vehicle0 | 1 | 125 | 45 | 23 | 50 | 7 |
| vehicle0 | 42 | 125 | 42 | 21 | 58 | 4 |
| wholesale_customers | 0 | 155 | 65 | 35 | 50 | 5 |
| wholesale_customers | 1 | 164 | 68 | 43 | 44 | 9 |
| wholesale_customers | 42 | 164 | 68 | 48 | 42 | 6 |

## Outputs

- `split_utility.csv`
- `split_utility_summary.json`

