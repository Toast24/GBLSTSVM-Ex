# Singleton isolation diagnostic

Read-only replay of frozen recursive KMeans(k=2) splits (purity=0.95, min_samples_per_ball=1, del_ball num_data=2).

A/B/C labels use parent-local and train-set geometry only. **Topo / CW / GBLSTSVM performance is not used.**

## Label definitions

- **A** — parent-local or global minority peeled from a same-class neighbour assigned to the sibling, with d_same_parent <= 2*hd1
- **C** — no same-class neighbour anywhere in the train set within 2*hd1
- **B** — neither A nor C, or A/C conflict

## Headline

- Singleton events: 9174 across 51 runs
- A/B/C counts: {'B': 6552, 'A': 1615, 'C': 1007}
- A/B/C fractions: {'B': 0.714, 'A': 0.176, 'C': 0.11}
- Label (+1/−1) counts: {'-1': 4873, '+1': 4301}
- Fraction eventually deleted by del_ball: 1.000
- Fraction with only 1 opposite neighbour (at hd1): 0.974
- Fraction with ≥2 same-class neighbours (global, 2·hd1): 0.794
- Fraction with strong minority isolation: 0.388
- Fraction global minority: 0.473
- Fraction near class boundary (n_opp 2·hd1 ≥ 2): 0.284
- Fraction lone-in-parent: 0.638
- Fraction nearest same-class in sibling: 0.362
- Fraction hd2 defined: 0.267
- Fraction parent size == 2: 0.499

## Distributions

- hd1_norm (all): n=9174 mean=1.973 p10=1.623 p50=2.000 p90=2.209 [0.000, 63.765]
- d_same_parent / hd1 (all): n=3322 mean=0.990 p10=0.707 p50=0.971 p90=1.326 [0.327, 2.749]
- d_same_global / hd1 (all): n=9132 mean=1.274 p10=0.665 p50=1.026 p90=1.982 [0.152, 37.723]

### Class A (n=1615, frac=0.176)
- hd1_norm: n=1615 mean=2.021 p10=1.477 p50=1.906 p90=2.482 [0.000, 63.765]
- d_same_parent/hd1: n=1614 mean=1.025 p10=0.729 p50=0.998 p90=1.383 [0.327, 1.933]
- d_same_global/hd1: n=1614 mean=0.896 p10=0.595 p50=0.886 p90=1.210 [0.152, 1.866]
- KMeans purity gain: n=1615 mean=0.349 p10=0.200 p50=0.333 p90=0.500 [0.064, 0.500]
- parent size: n=1615 mean=6.840 p10=3.000 p50=4.000 p90=10.000 [3.000, 483.000]
- deleted=1.000, global_min=0.820, local_min=0.369, only_1_opp=0.965, n_opp_2hd1==1=0.462, ≥2 same=0.939, strong_iso=0.000, boundary=0.538, nearest_in_sibling=1.000, lone=0.000

### Class B (n=6552, frac=0.714)
- hd1_norm: n=6552 mean=1.974 p10=1.677 p50=2.000 p90=2.155 [0.000, 8.678]
- d_same_parent/hd1: n=1703 mean=0.955 p10=0.690 p50=0.944 p90=1.268 [0.340, 2.749]
- d_same_global/hd1: n=6551 mean=1.072 p10=0.667 p50=1.016 p90=1.559 [0.152, 2.000]
- KMeans purity gain: n=6552 mean=0.414 p10=0.250 p50=0.500 p90=0.500 [0.053, 0.500]
- parent size: n=6552 mean=3.164 p10=2.000 p50=2.000 p90=5.000 [2.000, 97.000]
- deleted=1.000, global_min=0.378, local_min=0.457, only_1_opp=0.974, n_opp_2hd1==1=0.761, ≥2 same=0.880, strong_iso=0.455, boundary=0.239, nearest_in_sibling=0.260, lone=0.740

### Class C (n=1007, frac=0.110)
- hd1_norm: n=1007 mean=1.888 p10=1.698 p50=2.000 p90=2.000 [0.000, 2.667]
- d_same_parent/hd1: n=5 mean=1.493 p10=0.822 p50=1.191 p90=2.318 [0.794, 2.358]
- d_same_global/hd1: n=967 mean=3.270 p10=2.019 p50=2.593 p90=4.790 [0.496, 37.723]
- KMeans purity gain: n=1007 mean=0.468 p10=0.333 p50=0.500 p90=0.500 [0.111, 0.500]
- parent size: n=1007 mean=2.240 p10=2.000 p50=2.000 p90=3.000 [2.000, 10.000]
- deleted=1.000, global_min=0.532, local_min=0.577, only_1_opp=0.988, n_opp_2hd1==1=0.831, ≥2 same=0.000, strong_iso=0.575, boundary=0.169, nearest_in_sibling=0.005, lone=0.995

## Can a simple local criterion separate A from C?

A vs C AUROC for `lone_in_parent` and `d_same_global/hd1` is near-tautological (those quantities enter the label definitions). The non-circular test is: among **lone-in-parent** events, can **parent-local** features predict global isolation (`n_same_global within 2·hd1 == 0`)?

- hd1_norm p90(A) < p10(C): False
- d_same_global/hd1 p90(A) < p10(C): True
- Max parent-local AUROC on A vs C (excluding definitional features): 0.711

| Feature (higher → more C-like) | AUROC C vs A |
|---|---:|
| n_opp_2hd1_eq1 | 0.685 |
| only_one_opp_neighbour | 0.512 |
| hd1_norm | 0.537 |
| ratio_d_same_global_over_hd1 | 0.992 |
| lone_in_parent | 0.998 |
| s_size_hd2 | 0.711 |
| near_class_boundary | 0.685 |

### Parameter-free rules (evaluated only on true A/C)

| Rule | n | acc | C prec | C rec | TP FP FN TN |
|---|---:|---:|---:|---:|---|
| n_opp_within_2hd1 == 1 → C else A | 2622 | 0.651 | 0.529 | 0.831 | 837 746 170 869 |
| hd2 |S|==1 → C, |S|>1 → A (defined hd2 only) | 1008 | 0.736 | 0.375 | 0.994 | 159 265 1 583 |
| nearest same-class not in sibling → C else A | 2622 | 0.998 | 1.000 | 0.995 | 1002 0 5 1615 |
| n_opp_within_hd1 == 1 → C else A | 2622 | 0.401 | 0.390 | 0.988 | 995 1558 12 57 |

### Non-circular test: lone-in-parent → globally isolated?

- Lone-in-parent events: 5851; peels: 3323
- Fraction of lones with no same-class neighbour in 2·hd1 globally: 0.171
- n_opp 2·hd1 == 1: isolated 0.833 vs still-supported 0.773
- hd1_norm isolated: n=1002 mean=1.889 p10=1.705 p50=2.000 p90=2.000 [0.000, 2.667]
- hd1_norm supported: n=4849 mean=1.964 p10=1.747 p50=2.000 p90=2.000 [0.000, 5.062]
- Max parent-local AUROC: 0.577

| Parent-local feature | AUROC (lone → globally isolated) |
|---|---:|
| hd1_norm | 0.492 |
| hd1 | 0.577 |
| n_opp_2hd1 | 0.532 |
| n_opp_3hd1 | 0.532 |
| only_one_opp_neighbour | 0.507 |
| near_class_boundary | 0.530 |
| parent_purity | 0.468 |
| kmeans_purity_gain | 0.532 |
| parent_num | 0.532 |
| sibling_purity | 0.500 |
| dist_to_sibling_center | 0.577 |
| is_global_minority | 0.513 |
| is_parent_local_minority | 0.482 |
| strong_minority_isolation | 0.482 |
| hd2_defined | 0.525 |
| n_opp_2hd1_eq1 | 0.530 |

## Verdict

NO parent-local criterion. Singleton children are always deleted by del_ball. Half of events are 2-sample parents, so n_opp at hd1 equals 1 and hd2 is usually undefined. hd1_norm, n_opp within {1,2,3}·hd1, sibling purity, and KMeans purity gain all sit near AUROC 0.5 for telling globally class-isolated outliers from lone-in-parent points that still have same-class neighbours outside the parent. The only clean separator is d_same_global / hd1 — looking outside the parent — which is the definition of C, not a new local rule. A repair that uses only the parent ball cannot distinguish legitimate isolated minority/boundary points from true outliers.

simple_parent_local_criterion_exists = False

## Per-run event counts

| dataset | seed | n_train | singletons | deleted | after del_ball |
|---|---:|---:|---:|---:|---:|
| australian | 0 | 483 | 231 | 231 | 88 |
| australian | 1 | 483 | 203 | 203 | 97 |
| australian | 42 | 483 | 221 | 221 | 83 |
| breast_cancer | 0 | 200 | 93 | 93 | 35 |
| breast_cancer | 1 | 200 | 93 | 93 | 37 |
| breast_cancer | 42 | 200 | 77 | 77 | 43 |
| breast_cancer_wisconsin_diagnostic | 0 | 398 | 40 | 40 | 20 |
| breast_cancer_wisconsin_diagnostic | 1 | 398 | 30 | 30 | 20 |
| breast_cancer_wisconsin_diagnostic | 42 | 398 | 30 | 30 | 23 |
| breast_cancer_wisconsin_original | 0 | 489 | 0 | 0 | 2 |
| breast_cancer_wisconsin_original | 1 | 489 | 0 | 0 | 2 |
| breast_cancer_wisconsin_original | 42 | 489 | 0 | 0 | 2 |
| chronic_kidney_disease | 0 | 280 | 96 | 96 | 52 |
| chronic_kidney_disease | 1 | 280 | 107 | 107 | 51 |
| chronic_kidney_disease | 42 | 280 | 95 | 95 | 52 |
| congressional_voting_records | 0 | 304 | 6 | 6 | 8 |
| congressional_voting_records | 1 | 304 | 27 | 27 | 30 |
| congressional_voting_records | 42 | 304 | 29 | 29 | 24 |
| credit_approval | 0 | 483 | 201 | 201 | 99 |
| credit_approval | 1 | 483 | 198 | 198 | 97 |
| credit_approval | 42 | 483 | 217 | 217 | 82 |
| diabetes | 0 | 537 | 206 | 206 | 84 |
| diabetes | 1 | 537 | 214 | 214 | 86 |
| diabetes | 42 | 537 | 211 | 211 | 85 |
| ecoli2 | 0 | 235 | 11 | 11 | 15 |
| ecoli2 | 1 | 235 | 11 | 11 | 17 |
| ecoli2 | 42 | 235 | 18 | 18 | 13 |
| german_credit | 0 | 700 | 358 | 358 | 119 |
| german_credit | 1 | 700 | 310 | 310 | 130 |
| german_credit | 42 | 700 | 356 | 356 | 118 |
| heart | 0 | 189 | 98 | 98 | 31 |
| heart | 1 | 189 | 99 | 99 | 34 |
| heart | 42 | 189 | 79 | 79 | 38 |
| ionosphere | 0 | 245 | 31 | 31 | 27 |
| ionosphere | 1 | 245 | 45 | 45 | 32 |
| ionosphere | 42 | 245 | 37 | 37 | 24 |
| messidor_features | 0 | 805 | 392 | 392 | 147 |
| messidor_features | 1 | 805 | 387 | 387 | 139 |
| messidor_features | 42 | 805 | 366 | 366 | 147 |
| spambase | 0 | 3220 | 875 | 875 | 532 |
| spambase | 1 | 3220 | 822 | 822 | 525 |
| spambase | 42 | 3220 | 796 | 796 | 551 |
| tic_tac_toe | 0 | 670 | 294 | 294 | 124 |
| tic_tac_toe | 1 | 670 | 313 | 313 | 126 |
| tic_tac_toe | 42 | 670 | 279 | 279 | 132 |
| vehicle0 | 0 | 592 | 83 | 83 | 59 |
| vehicle0 | 1 | 592 | 75 | 75 | 51 |
| vehicle0 | 42 | 592 | 67 | 67 | 59 |
| wholesale_customers | 0 | 275 | 105 | 105 | 51 |
| wholesale_customers | 1 | 275 | 120 | 120 | 45 |
| wholesale_customers | 42 | 275 | 122 | 122 | 43 |

## Outputs

- `singleton_isolation_events.csv`
- `singleton_isolation_runs.csv`
- `singleton_isolation_summary.json`

