# Gaussian Membership Screening Report

Read-only experiment; frozen granulation / CA / Topo / solver unchanged.

- Primary Gaussian for G1/G2: **purity_compact** (purity * exp(-r_tilde^2); fuzzy granular-ball quality)
- Datasets: diabetes, ecoli2, vehicle0
- Seeds: [42, 0, 1]

## Model definitions

| Model | Weights |
|---|---|
| B0 | CW × μA (frozen CA-GBLSTSVM) |
| G1 | CW × μ_G (purity_compact) |
| G2 | CW × μ_G × μTopo |

## Classifier metrics (test set)

See CSV for full table. Mean test metrics by dataset:

### diabetes
- **B0_CA**: macro_f1=0.398, gmean=0.052, minority_f1=0.016, acc=0.641
- **G1_Gaussian_CW**: macro_f1=0.393, gmean=0.000, minority_f1=0.000, acc=0.647
- **G2_Gaussian_Topo**: macro_f1=0.401, gmean=0.052, minority_f1=0.016, acc=0.649

### ecoli2
- **B0_CA**: macro_f1=0.561, gmean=0.271, minority_f1=0.211, acc=0.843
- **G1_Gaussian_CW**: macro_f1=0.683, gmean=0.592, minority_f1=0.445, acc=0.863
- **G2_Gaussian_Topo**: macro_f1=0.546, gmean=0.365, minority_f1=0.225, acc=0.791

### vehicle0
- **B0_CA**: macro_f1=0.758, gmean=0.677, minority_f1=0.605, acc=0.856
- **G1_Gaussian_CW**: macro_f1=0.655, gmean=0.564, minority_f1=0.436, acc=0.798
- **G2_Gaussian_Topo**: macro_f1=0.713, gmean=0.617, minority_f1=0.526, acc=0.835

## Membership vs CA / Topo (primary Gaussian)

- **diabetes s42** (n_balls=85): corr(μ_G, μ_A)=0.173, corr(μ_G, μ_Topo)=0.325, corr(μ_A, μ_Topo)=0.728; pred change G1 vs B0=0.9%, G2 vs B0=1.7%
- **diabetes s0** (n_balls=84): corr(μ_G, μ_A)=0.346, corr(μ_G, μ_Topo)=0.381, corr(μ_A, μ_Topo)=0.663; pred change G1 vs B0=0.0%, G2 vs B0=0.0%
- **diabetes s1** (n_balls=86): corr(μ_G, μ_A)=-0.129, corr(μ_G, μ_Topo)=-0.134, corr(μ_A, μ_Topo)=0.773; pred change G1 vs B0=2.6%, G2 vs B0=2.6%
- **ecoli2 s42** (n_balls=13): corr(μ_G, μ_A)=-0.164, corr(μ_G, μ_Topo)=0.808, corr(μ_A, μ_Topo)=0.158; pred change G1 vs B0=3.9%, G2 vs B0=31.4%
- **ecoli2 s0** (n_balls=15): corr(μ_G, μ_A)=0.392, corr(μ_G, μ_Topo)=0.534, corr(μ_A, μ_Topo)=0.108; pred change G1 vs B0=9.8%, G2 vs B0=3.9%
- **ecoli2 s1** (n_balls=17): corr(μ_G, μ_A)=-0.081, corr(μ_G, μ_Topo)=0.587, corr(μ_A, μ_Topo)=0.209; pred change G1 vs B0=3.9%, G2 vs B0=0.0%
- **vehicle0 s42** (n_balls=59): corr(μ_G, μ_A)=0.134, corr(μ_G, μ_Topo)=0.195, corr(μ_A, μ_Topo)=0.623; pred change G1 vs B0=19.7%, G2 vs B0=18.1%
- **vehicle0 s0** (n_balls=59): corr(μ_G, μ_A)=-0.115, corr(μ_G, μ_Topo)=0.085, corr(μ_A, μ_Topo)=0.697; pred change G1 vs B0=11.0%, G2 vs B0=11.0%
- **vehicle0 s1** (n_balls=51): corr(μ_G, μ_A)=-0.093, corr(μ_G, μ_Topo)=0.242, corr(μ_A, μ_Topo)=0.687; pred change G1 vs B0=16.5%, G2 vs B0=11.8%

## Formulation correlations with μA

- **compact**: mean corr(μ, μ_A)=0.051, mean corr(μ, μ_Topo)=0.336
- **purity_compact**: mean corr(μ, μ_A)=0.051, mean corr(μ, μ_Topo)=0.336
- **density**: mean corr(μ, μ_A)=0.038, mean corr(μ, μ_Topo)=0.254
- **inverse_spread**: mean corr(μ, μ_A)=0.045, mean corr(μ, μ_Topo)=0.344

## Failures vs B0 (delta < -0.01)

- diabetes s42 G1_Gaussian_CW gmean: 0.000 vs B0 0.156 (delta=-0.156)
- diabetes s42 G1_Gaussian_CW minority_f1: 0.000 vs B0 0.048 (delta=-0.048)
- diabetes s42 G1_Gaussian_CW macro_f1: 0.393 vs B0 0.419 (delta=-0.026)
- ecoli2 s42 G2_Gaussian_Topo gmean: 0.741 vs B0 0.814 (delta=-0.073)
- ecoli2 s42 G2_Gaussian_Topo minority_f1: 0.452 vs B0 0.632 (delta=-0.180)
- ecoli2 s42 G2_Gaussian_Topo macro_f1: 0.606 vs B0 0.774 (delta=-0.168)
- vehicle0 s42 G1_Gaussian_CW gmean: 0.674 vs B0 0.685 (delta=-0.011)
- vehicle0 s42 G1_Gaussian_CW minority_f1: 0.556 vs B0 0.588 (delta=-0.033)
- vehicle0 s42 G1_Gaussian_CW macro_f1: 0.718 vs B0 0.742 (delta=-0.025)
- vehicle0 s0 G1_Gaussian_CW gmean: 0.315 vs B0 0.571 (delta=-0.257)
- vehicle0 s0 G1_Gaussian_CW minority_f1: 0.176 vs B0 0.476 (delta=-0.300)
- vehicle0 s0 G1_Gaussian_CW macro_f1: 0.525 vs B0 0.686 (delta=-0.162)
- vehicle0 s1 G1_Gaussian_CW gmean: 0.705 vs B0 0.775 (delta=-0.070)
- vehicle0 s1 G1_Gaussian_CW minority_f1: 0.576 vs B0 0.750 (delta=-0.174)
- vehicle0 s1 G1_Gaussian_CW macro_f1: 0.724 vs B0 0.846 (delta=-0.122)
- vehicle0 s42 G2_Gaussian_Topo gmean: 0.662 vs B0 0.685 (delta=-0.023)
- vehicle0 s42 G2_Gaussian_Topo minority_f1: 0.560 vs B0 0.588 (delta=-0.028)
- vehicle0 s42 G2_Gaussian_Topo macro_f1: 0.726 vs B0 0.742 (delta=-0.016)
- vehicle0 s0 G2_Gaussian_Topo gmean: 0.516 vs B0 0.571 (delta=-0.055)
- vehicle0 s0 G2_Gaussian_Topo minority_f1: 0.421 vs B0 0.476 (delta=-0.055)

## Interpretation guide

- **Independent signal**: low |corr(μ_G, μ_A)| and non-trivial pred-change vs B0.
- **Reproduces CA**: high |corr(μ_G, μ_A)| and near-zero pred-change with similar metrics.
- G2 adds Topo beyond Gaussian only if corr(μ_G, μ_Topo) is moderate and G2 ≠ G1.

Outputs: `gaussian_membership_screen_results.csv`, `gaussian_membership_formulations.csv`
