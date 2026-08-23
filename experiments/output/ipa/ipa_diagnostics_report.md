# IPA Diagnostic Report

## Configuration
- Granulation: purity=0.95, min_ball_count=2, seed=42
- lambda=1.0 (fixed)
- Substantial change threshold: |muIPA - ambiguity| > 0.1

## diabetes

N balls=85, k=5

### Correlations
- ambiguity_vs_isolation: Pearson=0.0159, Spearman=0.0304
- muIPA_vs_CW: Pearson=0.2866, Spearman=0.2893
- ambiguity_vs_raw_isolation: Pearson=0.1212, Spearman=0.0667

### muIPA distributions
- minority: {'n': 27, 'mean': 0.4551573537184701, 'median': 0.3847022707774186, 'std': 0.3042353970081076, 'min': 0.0, 'max': 0.907195708474924}
- majority: {'n': 58, 'mean': 0.26672626654269527, 'median': 0.26262354260139875, 'std': 0.2880411499657696, 'min': 0.0, 'max': 0.8283898406556253}
- frac(muIPA > 0.5): 0.2588

### Penalization summary
- n substantially penalized: 17
- frac substantially penalized: 0.2000
- frac substantial among minority: 0.0824

### Synthetic noise (ambiguity vs IPA)
- Method: farthest_minority_balls
- noise_ball ambiguity: 0.6880843552940201, percentile=73.6
- noise_ball muIPA: 0.2283259740260215, percentile=31.9
- ambiguity reduction on noise ball: 0.4598
- raw_isolation: 3.0136052555089576

## ecoli2

N balls=13, k=5

### Correlations
- ambiguity_vs_isolation: Pearson=0.3982, Spearman=0.3964
- muIPA_vs_CW: Pearson=-0.5088, Spearman=-0.4147
- ambiguity_vs_raw_isolation: Pearson=0.0613, Spearman=0.2107

### muIPA distributions
- minority: {'n': 6, 'mean': 0.18643009446583156, 'median': 0.15536262763597117, 'std': 0.18929729623596192, 'min': 0.0, 'max': 0.4222660371908246}
- majority: {'n': 7, 'mean': 0.44951530125800815, 'median': 0.35968425838321316, 'std': 0.24641057265377309, 'min': 0.2176233133642082, 'max': 0.8197680627087993}
- frac(muIPA > 0.5): 0.1538

### Penalization summary
- n substantially penalized: 3
- frac substantially penalized: 0.2308
- frac substantial among minority: 0.0000

### Synthetic noise (ambiguity vs IPA)
- Method: farthest_minority_balls
- noise_ball ambiguity: 0.37868702323339815, percentile=55.6
- noise_ball muIPA: 0.3640434756064985, percentile=55.6
- ambiguity reduction on noise ball: 0.0146
- raw_isolation: 1.0402247220678886

## vehicle0

N balls=59, k=5

### Correlations
- ambiguity_vs_isolation: Pearson=-0.0770, Spearman=0.0306
- muIPA_vs_CW: Pearson=0.1736, Spearman=0.1019
- ambiguity_vs_raw_isolation: Pearson=-0.2775, Spearman=-0.0174

### muIPA distributions
- minority: {'n': 27, 'mean': 0.6171672429000857, 'median': 0.639240638868113, 'std': 0.1901731852884749, 'min': 0.2430551847887256, 'max': 0.9884497927818718}
- majority: {'n': 32, 'mean': 0.5348946227590003, 'median': 0.6613124629866067, 'std': 0.2630619739592167, 'min': 0.0, 'max': 0.8897570288467165}
- frac(muIPA > 0.5): 0.6271

### Penalization summary
- n substantially penalized: 17
- frac substantially penalized: 0.2881
- frac substantial among minority: 0.1525

### Synthetic noise (ambiguity vs IPA)
- Method: farthest_minority_balls
- noise_ball ambiguity: 0.7602769877237507, percentile=51.5
- noise_ball muIPA: 0.7602769877237507, percentile=63.2
- ambiguity reduction on noise ball: 0.0000
- raw_isolation: 0.908738921218352
