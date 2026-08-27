# Split quality diagnostic

Read-only replay of frozen recursive KMeans(k=2) splits (purity=0.95, min_samples_per_ball=1, del_ball num_data=2).

A/B/C/D labels use split geometry and del_ball fate only. **Topo / classifier performance is not used.**

## Label definitions

- **A** — small child contains global minority and is fully deleted by del_ball
- **B** — singleton split whose small child is not a minority peel
- **C** — min(n1,n2)>=2 and small child is not fully deleted (retained split)
- **D** — other

## Headline

- Splits: 13654 across 51 runs
- Class counts: {'A': 6013, 'C': 5096, 'B': 2545}
- Class fractions: {A: 0.440, C: 0.373, B: 0.186}
- AUROC gate: 0.8
- simple_rule_auroc_ge_0.80: True

## Verdict

YES on class-structure / parent-size features, NO on the five highlighted geometry scores. Best overall: minority_fraction_small_child (AUROC=0.878). Best among size_ratio / S_geom / purity_gain / d_same_global/hd1 / minority_hd1_norm: minority_hd1_norm (AUROC=0.783). A Youden threshold of 0.5 on minority_fraction_small_child marks most harmful peels (small child at least half minority). S_geom and size_ratio do not clear 0.80 on their own.

Best single feature: `minority_fraction_small_child` AUROC=0.878 (higher => A).
Youden threshold on the oriented score: t=0.500, J=0.658, TPR=0.939, FPR=0.281, acc=0.816.

## Highlighted features by class

| Feature | A mean/p50 | B mean/p50 | C mean/p50 | D mean/p50 |
|---|---|---|---|---|
| size_ratio | 0.683/0.667 | 0.350/0.333 | 0.615/0.649 | nan/nan |
| S_geom | 2.203/1.828 | 34.133/2.493 | 1.106/0.944 | nan/nan |
| purity_gain | 0.241/0.250 | 0.097/0.000 | 0.052/0.000 | nan/nan |
| d_same_global_over_hd1 | 1.373/1.131 | 1.570/1.263 | 0.942/0.837 | nan/nan |
| minority_hd1_norm | 1.718/1.977 | 1.367/1.335 | 1.054/0.968 | nan/nan |

### Class A (n=6013, frac=0.440)
- minority_loss mean=1.099, majority_loss mean=0.402, singleton=0.721, small_deleted=1.000
- size_ratio: n=6013 mean=0.683 p10=0.250 p50=0.667 p90=1.000
- S_geom: n=3705 mean=2.203 p10=0.808 p50=1.828 p90=3.598
- purity_gain: n=6013 mean=0.241 p10=0.000 p50=0.250 p90=0.500
- d_same_global_over_hd1: n=5983 mean=1.373 p10=0.707 p50=1.131 p90=2.152
- minority_hd1_norm: n=6013 mean=1.718 p10=0.946 p50=1.977 p90=2.035
- parent_n: n=6013 mean=5.106 p10=2.000 p50=3.000 p90=9.000
- parent_purity: n=6013 mean=0.613 p10=0.500 p50=0.600 p90=0.800
- size_ratio: n=6013 mean=0.683 p10=0.250 p50=0.667 p90=1.000
- minority_fraction_small_child: n=6013 mean=0.858 p10=0.500 p50=1.000 p90=1.000
- small_is_global_minority: n=6013 mean=0.851 p10=0.000 p50=1.000 p90=1.000

### Class B (n=2545, frac=0.186)
- minority_loss mean=0.000, majority_loss mean=1.000, singleton=1.000, small_deleted=1.000
- size_ratio: n=2545 mean=0.350 p10=0.143 p50=0.333 p90=0.500
- S_geom: n=2496 mean=34.133 p10=1.678 p50=2.493 p90=4.751
- purity_gain: n=2545 mean=0.097 p10=0.000 p50=0.000 p90=0.333
- d_same_global_over_hd1: n=2525 mean=1.570 p10=0.798 p50=1.263 p90=2.532
- minority_hd1_norm: n=2545 mean=1.367 p10=0.753 p50=1.335 p90=2.049
- parent_n: n=2545 mean=5.106 p10=3.000 p50=4.000 p90=8.000
- parent_purity: n=2545 mean=0.684 p10=0.556 p50=0.667 p90=0.800
- size_ratio: n=2545 mean=0.350 p10=0.143 p50=0.333 p90=0.500
- minority_fraction_small_child: n=2545 mean=0.000 p10=0.000 p50=0.000 p90=0.000
- small_is_global_minority: n=2545 mean=0.000 p10=0.000 p50=0.000 p90=0.000

### Class C (n=5096, frac=0.373)
- minority_loss mean=2.211, majority_loss mean=2.453, singleton=0.000, small_deleted=0.000
- size_ratio: n=5096 mean=0.615 p10=0.273 p50=0.649 p90=1.000
- S_geom: n=5095 mean=1.106 p10=0.575 p50=0.944 p90=1.802
- purity_gain: n=5096 mean=0.052 p10=0.000 p50=0.000 p90=0.200
- d_same_global_over_hd1: n=5076 mean=0.942 p10=0.270 p50=0.837 p90=1.538
- minority_hd1_norm: n=5096 mean=1.054 p10=0.386 p50=0.968 p90=1.717
- parent_n: n=5096 mean=45.088 p10=6.000 p50=14.000 p90=80.000
- parent_purity: n=5096 mean=0.707 p10=0.541 p50=0.714 p90=0.873
- size_ratio: n=5096 mean=0.615 p10=0.273 p50=0.649 p90=1.000
- minority_fraction_small_child: n=5096 mean=0.437 p10=0.000 p50=0.359 p90=1.000
- small_is_global_minority: n=5096 mean=0.405 p10=0.000 p50=0.000 p90=1.000

### Class D: none

## Correlations with minority loss and AUROC for A

| Feature | Pearson loss | Spearman loss | Pearson A | Spearman A | AUROC A | dir |
|---|---:|---:|---:|---:|---:|---|
| minority_fraction_small_child | 0.031 | 0.428 | 0.668 | 0.677 | 0.878 | higher |
| parent_n | 0.723 | 0.164 | -0.134 | -0.538 | 0.811 | lower |
| small_is_pure_minority | -0.095 | 0.119 | 0.608 | 0.608 | 0.798 | higher |
| small_is_global_minority | 0.008 | 0.315 | 0.578 | 0.578 | 0.791 | higher |
| minority_hd1_norm | -0.144 | -0.105 | 0.261 | 0.488 | 0.783 | higher |
| parent_n_maj | 0.732 | 0.176 | -0.130 | -0.493 | 0.782 | lower |
| n1 | 0.562 | 0.190 | -0.100 | -0.486 | 0.778 | lower |
| n2 | 0.804 | 0.237 | -0.156 | -0.464 | 0.761 | lower |
| parent_n_min | 0.669 | 0.225 | -0.133 | -0.409 | 0.731 | lower |
| r1 | 0.028 | 0.063 | -0.060 | -0.381 | 0.719 | lower |
| r2 | 0.028 | 0.131 | -0.058 | -0.370 | 0.709 | lower |
| purity_gain | -0.081 | 0.072 | 0.444 | 0.389 | 0.705 | higher |
| parent_purity | -0.033 | -0.190 | -0.348 | -0.352 | 0.703 | lower |
| size_ratio | 0.080 | 0.327 | 0.263 | 0.254 | 0.646 | higher |
| S_geom | -0.008 | -0.269 | -0.016 | 0.194 | 0.619 | higher |
| purity1 | -0.119 | -0.121 | 0.190 | 0.207 | 0.614 | higher |
| d_same_global_over_hd1 | -0.037 | 0.056 | 0.093 | 0.185 | 0.607 | higher |
| purity2 | -0.174 | -0.240 | 0.148 | 0.181 | 0.597 | higher |
| minority_fraction_parent | -0.045 | 0.022 | 0.139 | 0.162 | 0.593 | higher |
| parent_radius | 0.022 | 0.084 | -0.051 | -0.160 | 0.593 | lower |
| center_distance | 0.010 | 0.060 | -0.042 | -0.103 | 0.560 | lower |
| d_same_global | -0.013 | 0.058 | -0.008 | 0.096 | 0.556 | higher |
| d_opp | -0.017 | -0.070 | -0.032 | -0.076 | 0.544 | lower |
| minority_hd1 | -0.011 | 0.026 | -0.004 | 0.006 | 0.504 | higher |

## 2-feature signed z-sum (highlighted pair)

| Pair | AUROC A |
|---|---:|
| +d_same_global_over_hd1 +minority_hd1_norm | 0.793 |
| +purity_gain +minority_hd1_norm | 0.786 |
| +size_ratio +minority_hd1_norm | 0.730 |
| +purity_gain +d_same_global_over_hd1 | 0.718 |
| -S_geom +minority_hd1_norm | 0.703 |
| +size_ratio +purity_gain | 0.684 |
| +size_ratio +d_same_global_over_hd1 | 0.650 |
| +S_geom +purity_gain | 0.568 |
| +S_geom +d_same_global_over_hd1 | 0.568 |
| -size_ratio +S_geom | 0.551 |

## Top 2-feature signed z-sums (all predictive features)

| Pair | AUROC A |
|---|---:|
| -parent_n_min +minority_fraction_small_child | 0.932 |
| -n2 +minority_fraction_small_child | 0.930 |
| -parent_n +minority_fraction_small_child | 0.928 |
| -n1 +minority_fraction_small_child | 0.921 |
| +minority_fraction_small_child +d_same_global_over_hd1 | 0.914 |
| -r2 +minority_fraction_small_child | 0.911 |
| -parent_n_maj +minority_fraction_small_child | 0.907 |
| -r1 +minority_fraction_small_child | 0.903 |
| +minority_fraction_small_child +minority_hd1_norm | 0.896 |
| -parent_n_min +small_is_global_minority | 0.894 |
| -parent_n +small_is_global_minority | 0.891 |
| -n2 +small_is_global_minority | 0.888 |
| -n1 +small_is_global_minority | 0.887 |
| +small_is_global_minority +d_same_global_over_hd1 | 0.878 |
| +small_is_pure_minority +minority_fraction_small_child | 0.878 |
| +small_is_global_minority +minority_fraction_small_child | 0.878 |
| -parent_radius +minority_fraction_small_child | 0.875 |
| -center_distance +minority_fraction_small_child | 0.875 |
| +minority_fraction_small_child -d_opp | 0.873 |
| +minority_fraction_small_child +d_same_global | 0.870 |

## Dataset-wise

| dataset | splits | A | B | C | D | frac A | AUROC size_ratio | AUROC S_geom | AUROC purity_gain | AUROC d_same_global_over_hd1 | AUROC minority_hd1_norm |
|---|---:|---:|---:|---:|---:|---:||---:||---:||---:||---:||---:|
| ionosphere | 193 | 100 | 10 | 83 | 0 | 0.518 | 0.731 | 0.958 | 0.652 | 0.606 | 0.774 |
| credit_approval | 891 | 451 | 135 | 305 | 0 | 0.506 | 0.666 | 0.600 | 0.703 | 0.602 | 0.781 |
| australian | 920 | 464 | 157 | 299 | 0 | 0.504 | 0.682 | 0.598 | 0.712 | 0.591 | 0.785 |
| tic_tac_toe | 1265 | 635 | 196 | 434 | 0 | 0.502 | 0.590 | 0.735 | 0.756 | 0.603 | 0.803 |
| messidor_features | 1575 | 789 | 261 | 525 | 0 | 0.501 | 0.650 | 0.628 | 0.684 | 0.587 | 0.784 |
| diabetes | 883 | 435 | 145 | 303 | 0 | 0.493 | 0.657 | 0.641 | 0.715 | 0.572 | 0.827 |
| breast_cancer_wisconsin_diagnostic | 160 | 77 | 16 | 67 | 0 | 0.481 | 0.567 | 0.739 | 0.740 | 0.734 | 0.772 |
| german_credit | 1388 | 666 | 271 | 451 | 0 | 0.480 | 0.712 | 0.505 | 0.759 | 0.621 | 0.820 |
| heart | 376 | 179 | 76 | 121 | 0 | 0.476 | 0.674 | 0.631 | 0.708 | 0.590 | 0.800 |
| breast_cancer | 375 | 170 | 74 | 131 | 0 | 0.453 | 0.699 | 0.649 | 0.763 | 0.545 | 0.738 |
| wholesale_customers | 483 | 201 | 126 | 156 | 0 | 0.416 | 0.720 | 0.523 | 0.762 | 0.513 | 0.854 |
| vehicle0 | 391 | 143 | 67 | 181 | 0 | 0.366 | 0.590 | 0.650 | 0.646 | 0.609 | 0.796 |
| spambase | 4098 | 1486 | 855 | 1757 | 0 | 0.363 | 0.614 | 0.636 | 0.681 | 0.637 | 0.778 |
| ecoli2 | 82 | 28 | 8 | 46 | 0 | 0.341 | 0.723 | 0.725 | 0.675 | 0.528 | 0.750 |
| chronic_kidney_disease | 450 | 153 | 126 | 171 | 0 | 0.340 | 0.715 | 0.514 | 0.636 | 0.555 | 0.754 |
| congressional_voting_records | 121 | 36 | 22 | 63 | 0 | 0.298 | 0.592 | 0.693 | 0.627 | 0.582 | 0.714 |
| breast_cancer_wisconsin_original | 3 | 0 | 0 | 3 | 0 | 0.000 | nan | nan | nan | nan | nan |

## Per-run split counts

| dataset | seed | n_train | splits | A | B | C | D | min lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| australian | 0 | 483 | 318 | 159 | 54 | 105 | 0 | 410 |
| australian | 1 | 483 | 299 | 143 | 53 | 103 | 0 | 342 |
| australian | 42 | 483 | 303 | 162 | 50 | 91 | 0 | 420 |
| breast_cancer | 0 | 200 | 127 | 59 | 27 | 41 | 0 | 137 |
| breast_cancer | 1 | 200 | 129 | 55 | 32 | 42 | 0 | 129 |
| breast_cancer | 42 | 200 | 119 | 56 | 15 | 48 | 0 | 132 |
| breast_cancer_wisconsin_diagnostic | 0 | 398 | 59 | 29 | 8 | 22 | 0 | 78 |
| breast_cancer_wisconsin_diagnostic | 1 | 398 | 49 | 23 | 5 | 21 | 0 | 76 |
| breast_cancer_wisconsin_diagnostic | 42 | 398 | 52 | 25 | 3 | 24 | 0 | 89 |
| breast_cancer_wisconsin_original | 0 | 489 | 1 | 0 | 0 | 1 | 0 | 0 |
| breast_cancer_wisconsin_original | 1 | 489 | 1 | 0 | 0 | 1 | 0 | 0 |
| breast_cancer_wisconsin_original | 42 | 489 | 1 | 0 | 0 | 1 | 0 | 0 |
| chronic_kidney_disease | 0 | 280 | 147 | 49 | 42 | 56 | 0 | 103 |
| chronic_kidney_disease | 1 | 280 | 157 | 57 | 43 | 57 | 0 | 124 |
| chronic_kidney_disease | 42 | 280 | 146 | 47 | 41 | 58 | 0 | 118 |
| congressional_voting_records | 0 | 304 | 13 | 4 | 2 | 7 | 0 | 15 |
| congressional_voting_records | 1 | 304 | 56 | 14 | 11 | 31 | 0 | 46 |
| congressional_voting_records | 42 | 304 | 52 | 18 | 9 | 25 | 0 | 57 |
| credit_approval | 0 | 483 | 299 | 141 | 48 | 110 | 0 | 358 |
| credit_approval | 1 | 483 | 294 | 145 | 45 | 104 | 0 | 353 |
| credit_approval | 42 | 483 | 298 | 165 | 42 | 91 | 0 | 406 |
| diabetes | 0 | 537 | 289 | 141 | 45 | 103 | 0 | 374 |
| diabetes | 1 | 537 | 299 | 154 | 48 | 97 | 0 | 393 |
| diabetes | 42 | 537 | 295 | 140 | 52 | 103 | 0 | 387 |
| ecoli2 | 0 | 235 | 25 | 8 | 2 | 15 | 0 | 19 |
| ecoli2 | 1 | 235 | 27 | 7 | 3 | 17 | 0 | 18 |
| ecoli2 | 42 | 235 | 30 | 13 | 3 | 14 | 0 | 32 |
| german_credit | 0 | 700 | 476 | 233 | 96 | 147 | 0 | 649 |
| german_credit | 1 | 700 | 439 | 194 | 88 | 157 | 0 | 542 |
| german_credit | 42 | 700 | 473 | 239 | 87 | 147 | 0 | 672 |
| heart | 0 | 189 | 128 | 63 | 25 | 40 | 0 | 168 |
| heart | 1 | 189 | 132 | 67 | 27 | 38 | 0 | 155 |
| heart | 42 | 189 | 116 | 49 | 24 | 43 | 0 | 122 |
| ionosphere | 0 | 245 | 57 | 28 | 2 | 27 | 0 | 64 |
| ionosphere | 1 | 245 | 76 | 40 | 3 | 33 | 0 | 93 |
| ionosphere | 42 | 245 | 60 | 32 | 5 | 23 | 0 | 83 |
| messidor_features | 0 | 805 | 538 | 263 | 94 | 181 | 0 | 800 |
| messidor_features | 1 | 805 | 525 | 279 | 83 | 163 | 0 | 812 |
| messidor_features | 42 | 805 | 512 | 247 | 84 | 181 | 0 | 741 |
| spambase | 0 | 3220 | 1406 | 524 | 301 | 581 | 0 | 1917 |
| spambase | 1 | 3220 | 1346 | 484 | 289 | 573 | 0 | 1753 |
| spambase | 42 | 3220 | 1346 | 478 | 265 | 603 | 0 | 1783 |
| tic_tac_toe | 0 | 670 | 417 | 213 | 62 | 142 | 0 | 677 |
| tic_tac_toe | 1 | 670 | 438 | 234 | 62 | 142 | 0 | 752 |
| tic_tac_toe | 42 | 670 | 410 | 188 | 72 | 150 | 0 | 627 |
| vehicle0 | 0 | 592 | 141 | 56 | 23 | 62 | 0 | 145 |
| vehicle0 | 1 | 592 | 125 | 45 | 23 | 57 | 0 | 128 |
| vehicle0 | 42 | 592 | 125 | 42 | 21 | 62 | 0 | 111 |
| wholesale_customers | 0 | 275 | 155 | 65 | 35 | 55 | 0 | 154 |
| wholesale_customers | 1 | 275 | 164 | 68 | 43 | 53 | 0 | 168 |
| wholesale_customers | 42 | 275 | 164 | 68 | 48 | 48 | 0 | 143 |

## Outputs

- `split_quality_diagnostic.csv`
- `split_quality_summary.json`

