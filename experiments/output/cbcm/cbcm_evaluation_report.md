# CBCM Evaluation Report

**Diagnostic gate:** FAIL

**Excluded datasets (gate fail):** ecoli2

## Protocol
- 5-fold stratified CV, seed=42
- c1=100.0, c2=0.01, purity=0.95
- No per-dataset tuning of k, alpha, beta, or normalization

## diabetes

### 5-fold CV (mean ± std)

| Model | Accuracy | G-mean | Minority F1 | Majority F1 | AUC |
|-------|----------|--------|-------------|-------------|-----|
| GBLSTSVM | 0.6550 ± 0.0054 | 0.0660 ± 0.0826 | 0.0217 ± 0.0287 | 0.7905 ± 0.0028 | 0.5332 ± 0.0450 |
| CW-GBLSTSVM | 0.6550 ± 0.0054 | 0.0660 ± 0.0826 | 0.0217 ± 0.0287 | 0.7905 ± 0.0028 | 0.5446 ± 0.0462 |
| CA-GBLSTSVM | 0.6563 ± 0.0088 | 0.0748 ± 0.0971 | 0.0287 ± 0.0417 | 0.7912 ± 0.0046 | 0.4582 ± 0.0901 |
| CB-GBLSTSVM | 0.6550 ± 0.0054 | 0.0660 ± 0.0826 | 0.0217 ± 0.0287 | 0.7905 ± 0.0028 | 0.5431 ± 0.0465 |
| CC-GBLSTSVM | 0.6550 ± 0.0064 | 0.0661 ± 0.0830 | 0.0218 ± 0.0291 | 0.7905 ± 0.0034 | 0.4618 ± 0.0921 |

### Significance (CC vs CW, Wilcoxon across folds)
- **gmean**: stat=1.0, p=1.0
- **minority_f1**: stat=1.0, p=1.0

### Frozen Phase 7 reference (70/15/15 test, seed=42)
- GBLSTSVM: acc=0.6466, gmean=0.0000, minF1=0.3927, AUC=0.4982
- CW-GBLSTSVM: acc=0.6466, gmean=0.0000, minF1=0.3927, AUC=0.5054

## vehicle0

### 5-fold CV (mean ± std)

| Model | Accuracy | G-mean | Minority F1 | Majority F1 | AUC |
|-------|----------|--------|-------------|-------------|-----|
| GBLSTSVM | 0.8534 ± 0.0380 | 0.6206 ± 0.1330 | 0.5468 ± 0.1666 | 0.9122 ± 0.0210 | 0.9095 ± 0.0663 |
| CW-GBLSTSVM | 0.8758 ± 0.0384 | 0.7099 ± 0.1177 | 0.6500 ± 0.1416 | 0.9243 ± 0.0218 | 0.9340 ± 0.0428 |
| CA-GBLSTSVM | 0.8794 ± 0.0347 | 0.7197 ± 0.0847 | 0.6680 ± 0.1106 | 0.9262 ± 0.0207 | 0.9292 ± 0.0420 |
| CB-GBLSTSVM | 0.8770 ± 0.0392 | 0.6996 ± 0.1270 | 0.6422 ± 0.1520 | 0.9255 ± 0.0220 | 0.9349 ± 0.0405 |
| CC-GBLSTSVM | 0.8794 ± 0.0322 | 0.7175 ± 0.0797 | 0.6665 ± 0.1037 | 0.9263 ± 0.0191 | 0.9300 ± 0.0419 |

### Significance (CC vs CW, Wilcoxon across folds)
- **gmean**: stat=7.0, p=1.0
- **minority_f1**: stat=6.0, p=0.8125

### Frozen Phase 7 reference (70/15/15 test, seed=42)
- GBLSTSVM: acc=0.8189, gmean=0.5684, minF1=0.6781, AUC=0.9103
- CW-GBLSTSVM: acc=0.8110, gmean=0.6126, minF1=0.6917, AUC=0.9244
