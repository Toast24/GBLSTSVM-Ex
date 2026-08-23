# Topology Membership Diagnostic Report

Formula frozen: muTopo = A2 * R_opp. No post-hoc retuning.

## Configuration
- Granulation: purity=0.95, min_ball_count=2, seed=42
- k = max(5, ceil(0.05 * N))

## Key question

Does reciprocal topology reduce high membership on unsupported noisy balls without destroying high membership of useful ambiguous balls?

## diabetes

N balls=85, k=5
degenerate=False

### Distributions
- comp: {'n': 85, 'mean': 0.3086380544147603, 'median': 0.20716185826780575, 'std': 0.2974816101022726, 'min': 0.0, 'max': 1.0}
- A: {'n': 85, 'mean': 0.37402648840086644, 'median': 0.366717187362361, 'std': 0.3295693751626272, 'min': 0.0, 'max': 0.9670325418097863}
- R_opp: {'n': 85, 'mean': 0.44392156583339865, 'median': 0.4999999975, 'std': 0.43540114963821147, 'min': 0.0, 'max': 0.9999999980000001}
- muTopo: {'n': 85, 'mean': 0.22941304429711773, 'median': 0.10349117123522228, 'std': 0.25976631685841106, 'min': 0.0, 'max': 0.8614803712820722}

### Correlations
- A_vs_CW: Pearson=0.3006, Spearman=0.3083
- R_opp_vs_A: Pearson=0.4417, Spearman=0.4856
- muTopo_vs_CW: Pearson=-0.0433, Spearman=0.0066

### Minority vs majority means
- A: minority=0.5193, majority=0.3064
- R_opp: minority=0.4963, majority=0.4195
- muTopo: minority=0.2129, majority=0.2371

### Fractions
- A > 0.5: 0.3176
- R_opp > 0: 0.5647
- muTopo > 0.5: 0.1176

### Synthetic noise
- Method: farthest_minority_balls
- A: 0.6880843552940201, percentile=73.6
- R_opp: 0.0, percentile=37.4
- muTopo: 0.0, percentile=42.9
- n_opp=2, n_recip=0

## ecoli2

N balls=13, k=5
degenerate=False

### Distributions
- comp: {'n': 13, 'mean': 0.44236963081394737, 'median': 0.5901159686456003, 'std': 0.33658501074354297, 'min': 0.0, 'max': 0.8201578708083934}
- A: {'n': 13, 'mean': 0.36555919689486, 'median': 0.3804788429712418, 'std': 0.25283944113668577, 'min': 0.0, 'max': 0.8197680627087993}
- R_opp: {'n': 13, 'mean': 0.28846153600961544, 'median': 0.0, 'std': 0.4368390996118642, 'min': 0.0, 'max': 0.9999999900000002}
- muTopo: {'n': 13, 'mean': 0.10743870478780829, 'median': 0.0, 'std': 0.16487285185772332, 'min': 0.0, 'max': 0.43062544648748624}

### Correlations
- A_vs_CW: Pearson=-0.6508, Spearman=-0.5805
- R_opp_vs_A: Pearson=0.0180, Spearman=-0.0204
- muTopo_vs_CW: Pearson=0.4514, Spearman=0.4533

### Minority vs majority means
- A: minority=0.1878, majority=0.5179
- R_opp: minority=0.5000, majority=0.1071
- muTopo: minority=0.1878, majority=0.0385

### Fractions
- A > 0.5: 0.1538
- R_opp > 0: 0.3077
- muTopo > 0.5: 0.0000

### Synthetic noise
- Method: farthest_minority_balls
- A: 0.37868702323339815, percentile=55.6
- R_opp: 0.9999999900000002, percentile=100.0
- muTopo: 0.378687019446528, percentile=94.4
- n_opp=1, n_recip=1

## vehicle0

N balls=59, k=5
degenerate=False

### Distributions
- comp: {'n': 59, 'mean': 0.386526401880237, 'median': 0.3681649353666364, 'std': 0.18051319661560455, 'min': 0.0, 'max': 1.0}
- A: {'n': 59, 'mean': 0.6358943178126655, 'median': 0.7175920859697937, 'std': 0.22197328372895092, 'min': 0.0, 'max': 0.9884497927818718}
- R_opp: {'n': 59, 'mean': 0.6720338943755649, 'median': 0.6666666644444444, 'std': 0.37188683240003995, 'min': 0.0, 'max': 0.9999999966666667}
- muTopo: {'n': 59, 'mean': 0.4498203672461316, 'median': 0.4255818779037318, 'std': 0.27436859672239783, 'min': 0.0, 'max': 0.9884497894870392}

### Correlations
- A_vs_CW: Pearson=0.1610, Spearman=0.1718
- R_opp_vs_A: Pearson=0.2723, Spearman=0.3018
- muTopo_vs_CW: Pearson=0.0978, Spearman=0.0381

### Minority vs majority means
- A: minority=0.6748, majority=0.6031
- R_opp: minority=0.6883, majority=0.6583
- muTopo: minority=0.4790, majority=0.4252

### Fractions
- A > 0.5: 0.6610
- R_opp > 0: 0.8305
- muTopo > 0.5: 0.4237

### Synthetic noise
- Method: farthest_minority_balls
- A: 0.7602769877237507, percentile=51.5
- R_opp: 0.4999999975, percentile=41.2
- muTopo: 0.38013849196118293, percentile=41.2
- n_opp=2, n_recip=1
