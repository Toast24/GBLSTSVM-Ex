# IPA Validation Report

**Global verdict: FAIL**

IPA does not improve on CA/CW consistently. Abandon competition-membership family.

## Configuration
- Split: stratified 70/15/15, seed=42
- Models: CW, CA (parabolic muA), IPA (lambda=1)

## diabetes

**Verdict: SUCCESS**

| Model | G-mean | Minority F1 | Macro F1 | Balanced Acc | Minority Rec | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CW | 0.0000 | 0.0000 | 0.3927 | 0.5000 | 0.0000 | 0.5053658536585366 |
| CA | 0.1562 | 0.0476 | 0.4185 | 0.5122 | 0.0244 | 0.4845528455284553 |
| IPA | 0.1562 | 0.0476 | 0.4185 | 0.5122 | 0.0244 | 0.5069918699186993 |

### IPA vs CA / CW
- IPA vs CA: d_gmean=+0.0000, d_minority_f1=+0.0000
- IPA vs CW: d_gmean=+0.1562, d_minority_f1=+0.0476
- Noise ambiguity pct=73.6, muIPA pct=31.9, decreased=True

## ecoli2

**Verdict: SUCCESS**

| Model | G-mean | Minority F1 | Macro F1 | Balanced Acc | Minority Rec | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CW | 0.7720 | 0.6667 | 0.8046 | 0.7892 | 0.6250 | 0.9302325581395349 |
| CA | 0.8141 | 0.6316 | 0.7736 | 0.8169 | 0.7500 | 0.9069767441860466 |
| IPA | 0.8248 | 0.6667 | 0.7976 | 0.8285 | 0.7500 | 0.9127906976744187 |

### IPA vs CA / CW
- IPA vs CA: d_gmean=+0.0106, d_minority_f1=+0.0351
- IPA vs CW: d_gmean=+0.0528, d_minority_f1=+0.0000
- Noise ambiguity pct=55.6, muIPA pct=55.6, decreased=False

## vehicle0

**Verdict: FAIL**

| Model | G-mean | Minority F1 | Macro F1 | Balanced Acc | Minority Rec | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CW | 0.6126 | 0.5000 | 0.6917 | 0.6691 | 0.4000 | 0.9243986254295532 |
| CA | 0.6849 | 0.5882 | 0.7424 | 0.7191 | 0.5000 | 0.9264604810996563 |
| IPA | 0.6653 | 0.5714 | 0.7345 | 0.7076 | 0.4667 | 0.9271477663230241 |

### IPA vs CA / CW
- IPA vs CA: d_gmean=-0.0196, d_minority_f1=-0.0168
- IPA vs CW: d_gmean=+0.0527, d_minority_f1=+0.0714
- Noise ambiguity pct=51.5, muIPA pct=63.2, decreased=False
