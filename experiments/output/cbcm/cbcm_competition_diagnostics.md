# CBCM Competition Diagnostics

**Decision: B** — Promising but unstable: vehicle0 shows the only clean gain (d_gmean=+0.0723, d_minority_f1=+0.0882); diabetes baseline degenerate; ecoli2 has outlier/mapping artefacts. One controlled refinement (best mapping or SP) on vehicle0 only.

## Configuration
- Frozen granulation: purity=0.95, min_ball_count=2, seed=42
- Split: stratified 70/15/15 (seed=42)

## diabetes

N balls=85, k=5

### muA correlations
- cw_abs_decision_score: Pearson=-0.1220, Spearman=-0.1333
- CW: Pearson=0.3072, Spearman=0.3083
- R: Pearson=0.1915, Spearman=0.1751
- BMRU_mu: Pearson=0.0220, Spearman=0.0062
- OBRU_mu: Pearson=-0.1199, Spearman=-0.1682
- nearest_opp_dist: Pearson=-0.0095, Spearman=-0.3508
- nearest_same_dist: Pearson=0.2454, Spearman=0.2403
- opp_same_ratio: Pearson=-0.5090, Spearman=-0.5307
- opp_neighbor_fraction: Pearson=0.5001, Spearman=0.6713
- compA: Pearson=0.5001, Spearman=0.6713

### muA class distributions
- minority: {'n': 27, 'mean': 0.6803571595634508, 'median': 0.6611470026874644, 'std': 0.32421368746298057, 'min': 0.0, 'max': 0.9989131467004765}
- majority: {'n': 58, 'mean': 0.4153682382575145, 'median': 0.5654376930421618, 'std': 0.4064292305646087, 'min': 0.0, 'max': 0.9808123136531669}

### muA threshold fractions
- frac_gt_0.25: 0.6353
- frac_gt_0.5: 0.6353
- frac_gt_0.75: 0.3176

### CW vs CA changed-ball profile
- test_pred_diff_frac: 0.008620689655172414
- frac_high_muA_minority: 0.4090909090909091
- frac_high_muA_boundary_proxy: 0.3181818181818182
- frac_boundary_minority: 0.10588235294117647
- n_balls_high_muA: 22

### Synthetic noise (extended)
- noise_ball_muA=0.8859718825912294, percentile=63.33333333333333
- noise_ball_muC percentile=85.71428571428571 (gate threshold 70, unchanged)
- nearest_same=70.59655970312342, compA=0.33115975197781583

### Mapping test screen (vs CW)
- A1_comp: gmean=0.0000 (vs CW +0.0000), minority_f1=0.0000
- A2_ambiguity: gmean=0.1562 (vs CW +0.1562), minority_f1=0.0476
- A3_parabolic: gmean=0.1562 (vs CW +0.1562), minority_f1=0.0476
- A4_sqrt: gmean=0.1562 (vs CW +0.1562), minority_f1=0.0476
- bounded_odds: gmean=0.0000 (vs CW +0.0000), minority_f1=0.0000
- balance: gmean=0.1562 (vs CW +0.1562), minority_f1=0.0476

### Cheap screening
- CW: gmean=0.0000, minority_f1=0.0000, macro_f1=0.3927, auroc=0.5053658536585366
- CA: gmean=0.1562, minority_f1=0.0476, macro_f1=0.4185, auroc=0.4845528455284553
- CC: gmean=0.1562, minority_f1=0.0476, macro_f1=0.4185, auroc=0.4845528455284553
- best_map_A2_ambiguity: gmean=0.1562, minority_f1=0.0476, macro_f1=0.4185, auroc=0.49365853658536585
- SP_supported_pressure: gmean=0.1562, minority_f1=0.0476, macro_f1=0.4185, auroc=0.526178861788618

## ecoli2

N balls=13, k=5

### muA correlations
- cw_abs_decision_score: Pearson=0.5526, Spearman=0.6575
- CW: Pearson=-0.6673, Spearman=-0.5805
- R: Pearson=-0.2909, Spearman=-0.2635
- BMRU_mu: Pearson=-0.3434, Spearman=-0.2597
- OBRU_mu: Pearson=nan, Spearman=nan
- nearest_opp_dist: Pearson=-0.0566, Spearman=-0.0249
- nearest_same_dist: Pearson=0.4373, Spearman=0.5194
- opp_same_ratio: Pearson=-0.8002, Spearman=-0.4475
- opp_neighbor_fraction: Pearson=0.6913, Spearman=0.4500
- compA: Pearson=0.6913, Spearman=0.4500

### muA class distributions
- minority: {'n': 6, 'mean': 0.3038687348634391, 'median': 0.26245016314003544, 'std': 0.30706442953563035, 'min': 0.0, 'max': 0.6758126227161689}
- majority: {'n': 7, 'mean': 0.7304328125017542, 'median': 0.6663351962993979, 'std': 0.1542565220009802, 'min': 0.5899957510377443, 'max': 0.9675164487802607}

### muA threshold fractions
- frac_gt_0.25: 0.7692
- frac_gt_0.5: 0.7692
- frac_gt_0.75: 0.1538

### CW vs CA changed-ball profile
- test_pred_diff_frac: 0.0784313725490196
- frac_high_muA_minority: 0.25
- frac_high_muA_boundary_proxy: 0.5
- frac_boundary_minority: 0.0
- n_balls_high_muA: 4

### Synthetic noise (extended)
- noise_ball_muA=0.6573803484798899, percentile=84.61538461538461
- noise_ball_muC percentile=61.111111111111114 (gate threshold 70, unchanged)
- nearest_same=0.17131841699011807, compA=0.20733139409901255

### Mapping test screen (vs CW)
- A1_comp: gmean=0.8248 (vs CW +0.0528), minority_f1=0.6667
- A2_ambiguity: gmean=0.8248 (vs CW +0.0528), minority_f1=0.6667
- A3_parabolic: gmean=0.8141 (vs CW +0.0422), minority_f1=0.6316
- A4_sqrt: gmean=0.8141 (vs CW +0.0422), minority_f1=0.6316
- bounded_odds: gmean=0.8248 (vs CW +0.0528), minority_f1=0.6667
- balance: gmean=0.8248 (vs CW +0.0528), minority_f1=0.6667

### Cheap screening
- CW: gmean=0.7720, minority_f1=0.6667, macro_f1=0.8046, auroc=0.9302325581395349
- CA: gmean=0.8141, minority_f1=0.6316, macro_f1=0.7736, auroc=0.9069767441860466
- CC: gmean=0.8141, minority_f1=0.6316, macro_f1=0.7736, auroc=0.9069767441860466
- best_map_A1_comp: gmean=0.8248, minority_f1=0.6667, macro_f1=0.7976, auroc=0.9244186046511628
- SP_supported_pressure: gmean=0.8248, minority_f1=0.6667, macro_f1=0.7976, auroc=0.9186046511627907

## vehicle0

N balls=59, k=5

### muA correlations
- cw_abs_decision_score: Pearson=-0.1793, Spearman=-0.1601
- CW: Pearson=0.1554, Spearman=0.1718
- R: Pearson=0.1025, Spearman=0.1157
- BMRU_mu: Pearson=-0.2123, Spearman=-0.2739
- OBRU_mu: Pearson=0.1273, Spearman=0.2537
- nearest_opp_dist: Pearson=-0.2118, Spearman=-0.4045
- nearest_same_dist: Pearson=-0.2712, Spearman=-0.0241
- opp_same_ratio: Pearson=-0.0036, Spearman=-0.2785
- opp_neighbor_fraction: Pearson=0.3772, Spearman=0.7672
- compA: Pearson=0.3772, Spearman=0.7672

### muA class distributions
- minority: {'n': 27, 'mean': 0.8538421472979748, 'median': 0.9326185863247923, 'std': 0.15335593587346755, 'min': 0.5497122272359805, 'max': 0.9998665927132183}
- majority: {'n': 32, 'mean': 0.788043810000442, 'median': 0.9161089583692756, 'std': 0.24532790388682316, 'min': 0.0, 'max': 0.9923360358782707}

### muA threshold fractions
- frac_gt_0.25: 0.9661
- frac_gt_0.5: 0.9661
- frac_gt_0.75: 0.6610

### CW vs CA changed-ball profile
- test_pred_diff_frac: 0.03937007874015748
- frac_high_muA_minority: 0.6666666666666666
- frac_high_muA_boundary_proxy: 0.06666666666666667
- frac_boundary_minority: 0.0
- n_balls_high_muA: 15

### Synthetic noise (extended)
- noise_ball_muA=0.9546724712530666, percentile=81.9672131147541
- noise_ball_muC percentile=44.11764705882353 (gate threshold 70, unchanged)
- nearest_same=31.344962381005782, compA=0.6064513136919096

### Mapping test screen (vs CW)
- A1_comp: gmean=0.6159 (vs CW +0.0034), minority_f1=0.5106
- A2_ambiguity: gmean=0.6849 (vs CW +0.0723), minority_f1=0.5882
- A3_parabolic: gmean=0.6849 (vs CW +0.0723), minority_f1=0.5882
- A4_sqrt: gmean=0.5929 (vs CW -0.0197), minority_f1=0.4889
- bounded_odds: gmean=0.6159 (vs CW +0.0034), minority_f1=0.5106
- balance: gmean=0.6849 (vs CW +0.0723), minority_f1=0.5882

### Cheap screening
- CW: gmean=0.6126, minority_f1=0.5000, macro_f1=0.6917, auroc=0.9243986254295532
- CA: gmean=0.6849, minority_f1=0.5882, macro_f1=0.7424, auroc=0.9264604810996563
- CC: gmean=0.6617, minority_f1=0.5600, macro_f1=0.7261, auroc=0.9268041237113401
- best_map_A2_ambiguity: gmean=0.6849, minority_f1=0.5882, macro_f1=0.7424, auroc=0.9223367697594502
- SP_supported_pressure: gmean=0.5653, minority_f1=0.4545, macro_f1=0.6701, auroc=0.9192439862542956

## Evidence table

| Candidate | New vs CW | New vs R | New vs BMRU | New vs OBRU | Outlier risk | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| CA — diabetes | gmean +0.1562, mf1 +0.0476 | r=0.191 | r=0.022 | r=-0.120 | noise muA pct=63.3 | promising_unstable; best_map=A2_ambiguity (d_gmean=+0.1562); |r(muA,|score|)=-0.122 |
| SP — ecoli2 | gmean +0.0528, mf1 -0.0351 | r=-0.291 | r=-0.343 | r=nan | noise muA pct=84.6 | promising_unstable; best_map=A1_comp (d_gmean=+0.0528); |r(muA,|score|)=0.553 |
| CA — vehicle0 | gmean +0.0723, mf1 +0.0882 | r=0.102 | r=-0.212 | r=0.127 | noise muA pct=82.0 | promising_unstable; best_map=A2_ambiguity (d_gmean=+0.0723); |r(muA,|score|)=-0.179 |