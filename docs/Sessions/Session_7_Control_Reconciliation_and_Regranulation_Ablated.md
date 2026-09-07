# Session 7 — Control Reconciliation and Re-Granulation Ablation

**Date:** 7 Sep 2026  
**Prior context:** Session 6 showed compression gains but left sample-control semantics inconsistent across experiments. See [Session 6](Session_6_Compression_K_Sensitivity_and_Validation.md).

**Scope:** Read-only experiments under `experiments/` and `tests/` only. No `src/` modifications. **Terminal session for the granular-structure branch.**

---

## 1. Research objective

Answer three questions with **fair, matched controls**:

| Question | Test |
|---|---|
| **A** | Does re-granulation beat singleton-ball representation on the **same samples**? |
| **B** | Does re-granulation beat **sample-level compression** (rho90 rebuild on subset)? |
| **C** | Does the effect depend on **rho** (granulation threshold), or only on ball count? |

---

## 2. Experiment 1 — Sample control reconciliation

**Script:** `test_sample_control_reconciliation.py`  
**Helper:** `sample_control_reconciliation_common.py`  
**Scale:** 17 datasets × 10 seeds × K ∈ {2, 4, 6, 8, 10, 12}

### 2.1 Why previous results disagreed

| Aspect | `sample_stratified_k` (K-sensitivity) | `sample_stratified_matched` (validation) |
|---|---|---|
| K meaning | # rho90 balls (entropy_k) | n_target only |
| Sample RNG | `method='sample_stratified_k'` | `method='sample_stratified_matched'` |
| Post-selection | **Rebuild rho90** on subsample | **Singleton balls** |
| Class weights | **Full y_train** | **Subset y_sub only** |

**Identical sample indices:** 1.1% of runs. **Semantically matched:** 0.0%.

### 2.2 Reconciled numbers (properly aligned where possible)

| Method | Mean macro F1 | Collapse | Mean balls/samples |
|---|---:|---:|---:|
| sample_stratified_k_original | **0.617** | 19.1% | 143 samples → ~16 balls |
| sample_stratified_matched_original | 0.474 | 0.0% | 143 singleton balls |
| singleton_ball_subset (same indices as k_original) | **0.475** | 0.0% | 143 singleton balls |
| granular_compressed_entropy_k | **0.602** | 27.5% | K balls |
| raw_sample_subset (proxy) | 0.475 | 0.0% | 143 singleton balls |

### 2.3 Reconciliation conclusion — **CASE C (not comparable)**

The validation vs K-sensitivity discrepancy is **not a performance paradox** — the controls differ in three independent dimensions (RNG, representation, weighting). The large gap (0.617 vs 0.474) is explained by:

1. **rho90 rebuild + full-train class weights** (k_original), vs
2. **singleton balls + subset class weights** (matched/singleton).

Granular entropy_k (0.602) beats singleton proxies (+0.12, p < 1e-45) but this compares **different representations at different weight semantics**, not isolated granular structure.

---

## 3. Experiment 2 — Re-granulation ablation

**Script:** `test_regranulation_ablation.py`  
**Helper:** `regranulation_ablation_common.py`  
**Tests:** `tests/test_regranulation_ablation.py`  
**Scale:** 17 datasets × 10 seeds × 6 K × 7 matched methods + full_rho90 reference = **6,820 rows**

### 3.1 Design (fairness enforced)

For each dataset×seed×K:

1. One deterministic `sample_stratified_k` subset (shared across all branches).
2. **Hash verification:** `selection_hash(singleton) == selection_hash(rho80) == … == selection_hash(rho98)` — **PASSED on all runs**.
3. Methods evaluated with **identical** Gaussian GBLSTSVM, hyperparameters, and test split.

| Method | Representation | y_train for weights |
|---|---|---|
| singleton | 1 ball per sample | y_sub |
| regranulate_rho{80,85,90,95,98} | rho granulation on subset | full y_train |
| sample_stratified_k_original | rho90 rebuild on subset | full y_train |
| full_rho90 (reference) | full-train rho90 | full y_train |

### 3.2 Results

| Method | Mean macro F1 | Collapse | Mean balls | Δ vs singleton |
|---|---:|---:|---:|---:|
| singleton | 0.4775 | 0.0% | 143.3 | — |
| regranulate_rho80 | 0.6211 | 37.2% | 10.6 | **+0.134** |
| regranulate_rho85 | 0.6193 | 21.8% | 13.7 | **+0.140** |
| regranulate_rho90 | 0.6179 | 20.2% | 15.6 | **+0.140** |
| regranulate_rho95 | 0.6116 | 19.7% | 17.8 | **+0.134** |
| regranulate_rho98 | 0.6038 | 18.9% | 19.5 | **+0.126** |
| sample_stratified_k_original | **0.6179** | 20.2% | 15.6 | +0.140 |
| full_rho90 (reference) | 0.584 | 5.9% | 77.3 | — |

**Statistical support (regranulate vs singleton, pooled):** Wilcoxon p < 1e-57, Cohen's d ≈ 0.67, win rate ≈ 76%.

### 3.3 Answering the three questions

| Question | Answer | Evidence |
|---|---|---|
| **A — Re-granulation vs singleton?** | **YES** | +0.13–0.14 F1, highly significant |
| **B — Re-granulation vs sample compression?** | **NO** | regranulate_rho90 ≡ sample_stratified_k_original (0.6179); mean Δ ≈ −0.003 |
| **C — Rho sensitivity?** | **Modest** | F1 decreases 0.621→0.604 as rho rises; ball count increases 10.6→19.5; weak n_balls–F1 correlation (r ≈ 0.13) |

### 3.4 Post-hoc descriptive optimum (NOT deployable)

Best mean F1: **regranulate_rho80 @ K=2** (0.666) — but **71.8% collapse**. High-F1 regimes trade reliability for complexity reduction.

---

## 4. Final scientific conclusion — **CASE B**

> Re-granulation materially changes the compressed representation, but there is **no evidence that granular structure provides an advantage beyond ordinary sample-level compression**.

### What this means

1. The reconciliation gap (0.617 vs 0.475) is **not** evidence for granular balls — it is primarily **representation + class-weight semantics** (rho90 rebuild with full y_train vs singleton with y_sub).
2. When sample selection is fixed, re-granulation **does** beat singleton (+0.14 F1) — so granulation changes the representation materially.
3. But re-granulation at rho=0.90 is **indistinguishable** from simply subsampling and rebuilding rho90 (`sample_stratified_k_original`).
4. The apparent granular advantage in earlier experiments was **compression + weighting**, not a second-stage granular mechanism.

### What we abandon

- Online entropy schedulers (Session 5)
- New rho tuning / entropy ranking / boundary ranking heuristics
- Claims of granular-specific contribution without complexity-controlled baselines

### Recommended next step

**Stop the granular-structure branch.** If pursuing compression, treat it as **sample-level or ball-count regularization** with explicit collapse monitoring — not as a novel granular-ball mechanism.

---

## 5. Test coverage

`tests/test_regranulation_ablation.py` — 14 tests including:

- Sample-selection reproducibility
- Identical `selection_hash` across rho branches
- Rho passed to granulation mechanism
- Singleton semantics unchanged
- No test samples in granulation
- Collapse detection
- Full-data reference isolation
- Output schema validation

Full suite: **184 passed, 1 skipped**.

---

## 6. Outputs

| Directory | Description |
|---|---|
| `experiments/output/sample_control_reconciliation/` | Control semantics + discrepancy analysis |
| `experiments/output/regranulation_ablation/` | Ablation raw results, geometry, paired comparisons, plots |

---

## 7. Research arc summary (Sessions 5–7)

```
Session 4: split-utility signal (AUROC 0.864)
    ↓
Session 5: entropy granulation → rejected (collapse + exhaustion)
    ↓
Session 6: compression helps; ranking weak; controls inconsistent
    ↓
Session 7: fair ablation → CASE B → stop granular branch
```

---

*Session 7 compiled from experiment outputs and ablation final report. No `src/` changes. This closes the granular-structure research line.*
