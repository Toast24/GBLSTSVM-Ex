# Topology Cheap-Screen Evaluation Report

**Global verdict: LOCALIZED**

Topo improves or preserves vehicle0 but remains unstable or unproven elsewhere. Report as localized evidence. Do not immediately run full CV.

## Configuration
- Split: stratified 70/15/15, seed=42
- Models: CW, CA (frozen parabolic muA), Topo (A2 * R_opp)
- No hyperparameter tuning after seeing results.

## diabetes

| Model | G-mean | Minority F1 | AUC | Balanced Acc | Accuracy |
| --- | --- | --- | --- | --- | --- |
| CW | 0.0000 | 0.0000 | 0.5054 | 0.5000 | 0.6466 |
| CA | 0.1562 | 0.0476 | 0.4846 | 0.5122 | 0.6552 |
| Topo | 0.1562 | 0.0476 | 0.4976 | 0.5122 | 0.6552 |

- Topo vs CA: d_gmean=0.0000, d_minority_f1=0.0000
- Topo vs CW: d_gmean=0.1562, d_minority_f1=0.0476
- Noise A pct=73.6, muTopo pct=42.9, not_worse=True, improved=True
- retains_ca_gain=True, r_opp_informative=True

## ecoli2

| Model | G-mean | Minority F1 | AUC | Balanced Acc | Accuracy |
| --- | --- | --- | --- | --- | --- |
| CW | 0.7720 | 0.6667 | 0.9302 | 0.7892 | 0.9020 |
| CA | 0.8141 | 0.6316 | 0.9070 | 0.8169 | 0.8627 |
| Topo | 0.6988 | 0.4118 | 0.8895 | 0.7166 | 0.6078 |

- Topo vs CA: d_gmean=-0.1153, d_minority_f1=-0.2198
- Topo vs CW: d_gmean=-0.0731, d_minority_f1=-0.2549
- Noise A pct=55.6, muTopo pct=94.4, not_worse=False, improved=False
- retains_ca_gain=False, r_opp_informative=True

## vehicle0

| Model | G-mean | Minority F1 | AUC | Balanced Acc | Accuracy |
| --- | --- | --- | --- | --- | --- |
| CW | 0.6126 | 0.5000 | 0.9244 | 0.6691 | 0.8110 |
| CA | 0.6849 | 0.5882 | 0.9265 | 0.7191 | 0.8346 |
| Topo | 0.7151 | 0.6400 | 0.8948 | 0.7460 | 0.8583 |

- Topo vs CA: d_gmean=0.0302, d_minority_f1=0.0518
- Topo vs CW: d_gmean=0.1025, d_minority_f1=0.1400
- Noise A pct=51.5, muTopo pct=41.2, not_worse=True, improved=True
- retains_ca_gain=True, r_opp_informative=True

## Research verdict

- **NOVELTY:** moderate (A2 × opposite-class reciprocal kNN is not a documented FSVM/GB-SVM membership; ingredients are known)
- **MECHANISM:** supported on vehicle0/diabetes diagnostics; unsupported as a general filter (ecoli2 noise ball received R_opp ≈ 1)
- **VEHICLE0:** improved (G-mean 0.715 vs CA 0.685 vs CW 0.613; minority F1 0.640 vs 0.588 vs 0.500)
- **NOISE BEHAVIOR:** improved on vehicle0 (A 51.5 → muTopo 41.2) and diabetes (73.6 → 42.9); **worse** on ecoli2 (55.6 → 94.4)
- **GENERALIZATION:** localized (ecoli2: G-mean −0.115 vs CA, −0.073 vs CW; N=13 balls with k=5 makes reciprocity weakly selective)
- **RECOMMENDATION:** investigate once more — do **not** promote to multi-seed/full CV; do **not** abandon yet (unlike IPA/SP, vehicle0 CA gain was preserved and increased). Next probe, if any, should explain ecoli2’s complete-reciprocity failure without adding density/isolation terms or retuning k after seeing these numbers.
