# Session 2 — Phases 6–8, BMRU, and OBRU

**Scope:** Everything from Phase 6 (inclusive) through the completed OBRU experiment.  
**Prior context:** Phases 1–5 established a frozen clean GBLSTSVM reproduction and three weighting extensions — CW (class), R (reliability), CR (product). See [Session 1](Session_1_GBLSTSVM_Research_Findings.md).

**Research arc (stages):**

```text
Stage 1  GBLSTSVM              (frozen baseline)
Stage 2  CW-GBLSTSVM           (class weights κ)
Stage 3  R-GBLSTSVM            (reliability a_i)
Stage 3b CR-GBLSTSVM           (κ × a_i)
Stage 4  BMRU-GBLSTSVM         (margin uncertainty μ)     → FAILED, FROZEN
Stage 5  CW+BMRU-GBLSTSVM
Stage 6  CW+R+BMRU-GBLSTSVM
Stage 7  OBRU-GBLSTSVM         (overlap proxy μ)          → FAILED
Stage 8  CW+OBRU-GBLSTSVM
Stage 9  CW+R+OBRU-GBLSTSVM
```

---

## 0. What we were trying to learn

After synthetic validation (Phases 1–5), the open questions were:

1. Do CW / R / CR help on **real imbalanced data** under a fixed, honest protocol?
2. If not, is the failure due to **bad hyperparameters / preprocessing**, or **bad mechanisms**?
3. Can a **new granular-ball-specific signal** (boundary uncertainty, then overlap) add information beyond CW+R?

Phases 6–8 answer (1) and (2). BMRU and OBRU answer (3). All three new mechanisms failed.

---

## 1. Phase 6 — Breast Cancer (first real dataset)

**Goal:** First train/val/test experiment on a real table without tuning.

| Item | Detail |
|------|--------|
| Script | `experiments/validate_breast_cancer.py` |
| Data | `Data/Data/breast_cancer.csv` — 286 samples, 9 features, binary 0/1 |
| Split | Seed 42, stratified 70/15/15 |
| Preprocessing | **None** (frozen protocol) |
| Models | Original/reference, Clean GBLSTSVM, CW, R, CR |
| Hyperparameters | c1=100, c2=1e-2, purity=0.95, min_ball_count=2 |

**Doc:** `docs/BREAST_CANCER_EXPERIMENT.md`  
**Outputs:** `experiments/output/breast_cancer_train_val_test.{json,csv}`

### Results (test)

| Model | Macro F1 | Bal Acc | Minority Recall | Notes |
|-------|----------|---------|-----------------|-------|
| All five | 0.411 | 0.500 | **0.000** | Identical hard predictions |

Every model predicted **majority (−1) only** — full prediction collapse.

### Mechanism diagnostics

- All balls had **purity = 1.0** → reliability **a_i ≡ 1** → **CR ≡ CW** (weights differ only by κ, but labels unchanged).
- `min_ball_count=2` dropped 77/200 training samples; only 20/59 minority points entered balls.
- CW changed AUROC slightly but **not** hard labels.

### Verdict

**No evidence** that CW, R, or CR improve over frozen GBLSTSVM on this dataset. The failure is stack-wide (collapse + saturated reliability), not CR-specific.

---

## 2. Phase 7 — Multi-dataset benchmark (frozen protocol)

**Goal:** Repeat Phase 6 protocol on three standard imbalanced UCI-style benchmarks.

| Item | Detail |
|------|--------|
| Script | `experiments/validate_multidataset.py` |
| Datasets | `diabetes.csv`, `ecoli2.csv`, `vehicle0.csv` |
| Split | Seed 42, stratified 70/15/15 |
| Preprocessing | **None** |
| Models | Original/reference + Clean + CW + R + CR |

**Doc:** `docs/PHASE7_MULTIDATASET_RESULTS.md`  
**Outputs:** `experiments/output/multidataset_train_val_test.{json,csv}`, ball/reliability diagnostics

### Dataset summary

| Dataset | Samples | Imbalance | Balls (train) |
|---------|---------|-----------|---------------|
| diabetes | 768 | 1.87:1 | 85 |
| ecoli2 | 336 | 5.46:1 | 13 |
| vehicle0 | 846 | 3.25:1 | 59 |

### Test results (high level)

| Dataset | Best variant | Macro F1 | Min Recall | Collapse? |
|---------|--------------|----------|------------|-----------|
| diabetes | All tied | 0.393 | **0.000** | **YES** |
| ecoli2 | All tied | 0.805 | 0.625 | No |
| vehicle0 | **CW = CR** | 0.692 | **0.400** | No |

**Only vehicle0** showed any test improvement: CW (and CR, identically) beat Clean on Macro F1 (+0.014), Bal Acc (+0.018), Minority Recall (+0.067).

On diabetes and ecoli2, CW/R/CR did not beat Clean on primary metrics. Reliability was numerically active (not saturated at 1.0) on all three datasets, yet **CR did not beat CW** anywhere.

### Reference vs clean

Reproduction verified on all three datasets: identical balls, 100% prediction agreement, score diffs &lt; 1e-12.

### Verdict

Under the **frozen untuned protocol**, the CW+R hypothesis is **weakly supported at best** (one dataset, small test fold). CR adds **nothing beyond CW**. Diabetes collapse remains a serious problem.

---

## 3. Phase 8 — Systematic ablation (is the null result a protocol artifact?)

**Goal:** Determine whether poor results were due to defaults (no scaling, suboptimal c1/c2, purity saturation) rather than the mechanisms themselves.

| Item | Detail |
|------|--------|
| Scripts | `experiments/phase8_*.py` (6 experiments) |
| Shared utils | `experiments/phase8_common.py` |
| Outputs | `experiments/output/phase8/` |

**Doc:** `docs/PHASE8_RESULTS.md`

### Experiments run

| # | Experiment | What varied |
|---|------------|-------------|
| 1 | Granulation sensitivity | purity ∈ {0.80, 0.85, 0.90, 0.95, 0.98} |
| 2 | Scaling ablation | none / StandardScaler / MinMaxScaler |
| 3 | Hyperparameter grid | c1 × c2 (49 combos), val selection |
| 4 | Final protocol | Best scaling + purity + (c1,c2); train+val → test |
| 5 | Multi-seed | Seeds {0,1,2,3,4,42}, new splits |
| 6 | Mechanism diagnostics | CR vs CW hyperplane / weight / prediction diffs |

### Key findings

**Scaling (diabetes):** MinMaxScaler + tuned c1/c2 **fixed majority collapse** — validation Macro F1 jumped ~0.39 → ~0.76. Scaling materially changes granulation on raw-feature datasets.

**Hyperparameters:** Defaults (c1=100, c2=0.01) are **not optimal**. Tuning changes diabetes conclusions dramatically.

**Final tuned test (Experiment 4):**

| Dataset | CW − Clean (Macro F1) | CR − CW |
|---------|----------------------|---------|
| diabetes | +0.003 | **−0.007** |
| ecoli2 | −0.025 | 0.000 |
| vehicle0 | +0.011 | **0.000** |

**Multi-seed (Experiment 5):** Mean(CW − Clean) Macro F1 is **negative** on all three datasets. No stable CR advantage; mean(CR − CW) ≤ +0.01.

**Mechanism (Experiment 6):** Reliability changes weights and sometimes hyperplanes (‖w_CR − w_CW‖ &gt; 0 on vehicle0), but **test predictions rarely change**. CR ≡ CW on vehicle0 and ecoli2 test sets.

### Verdict

Tuning and scaling fix **regime problems** (especially diabetes collapse) but **do not rescue CR**. Reliability influences optimization without predictive benefit. Phase 8 recommended **reliability redesign**, not more CR tuning.

---

## 4. BMRU — Ball Margin Reliability Uncertainty (Stages 4–6)

**Motivation (Phase 9 precursor):** After R failed to add value beyond CW, we tested a **hyperplane-dependent** granular-ball signal: down-weight balls whose mean radius is large relative to the current twin-hyperplane margin at the ball center.

| Item | Detail |
|------|--------|
| Math | `u_i = 2r_i / (|s_i| + ε)`, `μ_i = exp(−α·u_i)` with α=1, ε=1e-6 |
| Implementation | `src/bmru.py`, `src/bmru_gblstsvm.py` |
| Models | BMRU, CW+BMRU, CW+R+BMRU (iterative reweighting) |
| Tests | `tests/test_bmru_gblstsvm.py` |
| Doc | `docs/BMRU_GBLSTSVM.md` |

**Status: FROZEN. Documented failed hypothesis.**

### Experiments

| Script | Purpose |
|--------|---------|
| `experiments/validate_bmru.py` | Phase 7 protocol; BMRU stages only |
| `experiments/analyze_bmru_diagnostics.py` | Weight/margin ablation |
| `experiments/analyze_bmru_boundary_diagnostics.py` | Falsification: does high u mean boundary straddling? |

**Outputs:** `experiments/output/validate_bmru.{json,csv}`, `bmru_boundary_*.csv/json`, etc.

### Test deltas vs Phase 7 baselines

| Dataset | Comparison | Δ Macro F1 | Δ Min Recall | Verdict |
|---------|------------|------------|--------------|---------|
| diabetes | All BMRU vs baseline | 0 | 0 | No change |
| ecoli2 | All BMRU vs baseline | 0 | 0 | No change |
| vehicle0 | BMRU vs R | **−0.245** | **−0.333** | **Harmful** |
| vehicle0 | CW+BMRU vs CW | **−0.261** | **−0.400** | **Harmful** |
| vehicle0 | CW+R+BMRU vs CR | **−0.228** | **−0.367** | **Harmful** |

On vehicle0, BMRU **destroyed** minority recall (0.40 → 0.00 for CW+BMRU). On diabetes/ecoli2, metrics were identical to baselines (collapse / tied predictions).

### Boundary falsification (critical)

High `u_i` does **not** identify balls whose training samples straddle the decision boundary:

| Dataset | frac u≥1 | frac empirical straddling | corr(u, min\|s(x)\|) |
|---------|----------|---------------------------|------------------------|
| diabetes | 98.7% | **0%** | −0.13 |
| vehicle0 | 98.2% | 5.4% | −0.07 |
| ecoli2 | 0% | 40% | +0.04 |

`u ≥ 1` has ~0% precision on diabetes. High uncertainty is driven by **small center margin / large mean radius**, not sample-level boundary crossing.

### Verdict

**FAILED HYPOTHESIS — abandon BMRU.** Frozen in codebase for reproducibility. Does not provide information beyond CW+R; actively harmful on vehicle0.

---

## 5. OBRU — Overlap-aware reweighting (Stages 7–9)

**Motivation:** After BMRU failed, we tested a **fixed geometric** signal: inter-class ball overlap proxy using centers and mean-distance radii (no hyperplane iteration).

| Item | Detail |
|------|--------|
| Math | `O_ij = max(0, (r_i+r_j−d_ij)/(r_i+r_j+ε))`; top-3 mean → `O_i`; minority `μ=1+O`, majority `μ=1−O` |
| Implementation | `src/overlap.py`, `src/overlap_gblstsvm.py` |
| Models | OBRU, CW+OBRU, CW+R+OBRU (single-pass, no iteration) |
| Tests | `tests/test_overlap_gblstsvm.py` (95 total tests pass with frozen baselines) |
| Doc | `docs/OVERLAP_GBLSTSVM.md` |

### Experiments

| Script | Purpose |
|--------|---------|
| `experiments/validate_overlap.py` | Phase 7 protocol; OBRU stages only |
| `experiments/analyze_overlap.py` | Overlap/mu diagnostics and correlations |

**Outputs:** `experiments/output/validate_overlap.{json,csv}`, `overlap_per_ball.csv`, `overlap_summary.csv`, `overlap_correlations.csv`, `overlap_eval_raw.csv`

### Mechanism saturation

| Dataset | frac O=0 | frac O≥0.5 | max O | mean μ |
|---------|----------|------------|-------|--------|
| diabetes | **92.9%** | 0% | 0.091 | ≈1.000 |
| ecoli2 | **100%** | 0% | 0.000 | 1.000 (identity) |
| vehicle0 | **61.0%** | 0% | 0.126 | ≈1.000 |

The overlap proxy **barely activates**. No ball reaches substantial overlap (O ≥ 0.5). ecoli2 is a literal no-op (tiny radii, sparse balls).

`corr(CW×μ, CW) ≈ 0.998` — OBRU is effectively a tiny perturbation of class weights, not a new signal.

### Test deltas vs Phase 7 baselines (primary comparisons)

| Dataset | CW+OBRU − CW | CW+R+OBRU − CR |
|---------|--------------|----------------|
| diabetes | 0 (all metrics) | 0 |
| ecoli2 | 0 | 0 |
| vehicle0 | 0 (AUROC +0.002 only) | 0 (AUROC +0.002 only) |

**0/3 datasets improved** on Macro F1, G-mean, Balanced Accuracy, or Minority Recall.

### Val-only noise (do not over-interpret)

| Dataset | CW+OBRU val Δ Min Recall | Test Δ |
|---------|--------------------------|--------|
| vehicle0 | **+0.033** | 0 |
| diabetes | CW+R+OBRU val Δ Macro F1 +0.025 | 0 |

Single-split validation fluctuations did not generalize.

### Verdict

**FAILED HYPOTHESIS — abandon overlap weighting (current formulation).** OBRU does not provide information beyond CW+R. The proxy does not measure meaningful geometric competition under this granulation regime.

---

## 6. Cross-phase comparison table (test set, Phase 7 frozen protocol)

All extensions evaluated under **same split, seed 42, no tuning** unless noted.

| Extension | vs baseline test gain | vs parent test gain | Mechanism active? | Overall |
|-----------|----------------------|---------------------|-------------------|---------|
| CW (Phase 7) | vehicle0 only | — | Yes | Weak |
| R (Phase 7) | None | — | Yes (weights) | No predictive gain |
| CR (Phase 7) | = CW | 0 vs CW | Yes | Redundant with CW |
| BMRU | None / harmful | Harmful on vehicle0 | Yes (μ ≪ 1) | **Failed** |
| OBRU | None | None | Barely (μ ≈ 1) | **Failed** |

Under **Phase 8 tuned protocol**, CW occasionally helps minority recall on diabetes; CR still does not beat CW.

---

## 7. What we implemented (file index)

### Frozen baselines (Phases 1–5, untouched in Phases 6–9)

| Component | Path |
|-----------|------|
| GBLSTSVM | `src/gblstsvm.py` |
| CW | `src/cw_gblstsvm.py` |
| R | `src/r_gblstsvm.py` |
| CR | `src/cr_gblstsvm.py` |

### Phase 6–8 experiments

| Phase | Scripts | Key outputs |
|-------|---------|-------------|
| 6 | `validate_breast_cancer.py` | `breast_cancer_train_val_test.*` |
| 7 | `validate_multidataset.py` | `multidataset_train_val_test.*` |
| 8 | `phase8_*.py` | `output/phase8/*` |

### BMRU (frozen failed hypothesis)

| Role | Path |
|------|------|
| Math | `src/bmru.py` |
| Models | `src/bmru_gblstsvm.py` |
| Tests | `tests/test_bmru_gblstsvm.py` |
| Experiments | `validate_bmru.py`, `analyze_bmru_*.py` |

### OBRU (failed hypothesis)

| Role | Path |
|------|------|
| Math | `src/overlap.py` |
| Models | `src/overlap_gblstsvm.py` |
| Tests | `tests/test_overlap_gblstsvm.py` |
| Experiments | `validate_overlap.py`, `analyze_overlap.py` |

---

## 8. Statistical notes

- **Phase 7 / BMRU / OBRU:** 3 datasets × 1 seed × 1 test fold = **3 paired observations** per comparison. Insufficient for significance testing; deltas prepared for future Wilcoxon if multi-seed/multi-fold runs are added.
- **Phase 8 Experiment 5:** 6 seeds × 3 datasets provides more stable **mean** estimates but still does not show CR &gt; CW.
- Do **not** treat val and test folds from the same split as independent datasets.

---

## 9. Lessons learned

1. **Purity-saturated granulation kills R.** At ρ=0.95, most balls have purity ≈ 1 → a_i ≈ 1 → CR ≡ CW. Lowering purity activates reliability numerically but not predictively.

2. **Raw features + distance granulation need scaling on some datasets.** Diabetes collapse was largely a protocol/regime issue, not proof that weighting cannot work.

3. **Weight changes ≠ prediction changes.** CR can alter hyperplanes without changing discrete labels. Mechanism diagnostics are essential.

4. **New signals must be falsifiable.** BMRU was falsified by boundary-straddling diagnostics. OBRU was falsified by overlap saturation (O ≈ 0 everywhere).

5. **Class weighting is the only mechanism with any positive signal** — and it is small, seed-dependent, and limited to minority recall on some datasets.

6. **Do not tune extensions to win.** Phase 8 showed tuning helps the base classifier but does not make CR (or BMRU/OBRU) beat CW.

---

## 10. Current research state (end of Session 2)

| Question | Answer |
|----------|--------|
| Does CR beat CW? | **No** (consistent across Phase 7, 8, BMRU, OBRU) |
| Does R add value? | **Rarely / never** at prediction level |
| Does BMRU add value? | **No** — harmful on vehicle0 |
| Does OBRU add value? | **No** — mechanism inert |
| Is the stack ready for paper-level claims? | **No** — one weak CW signal, two failed extensions |
| What is frozen? | GBLSTSVM, CW, R, CR, BMRU (math + code) |
| What failed and should not be revived? | BMRU, OBRU (current formulations) |

**Recommended next direction (not implemented):** Any new mechanism needs (a) a signal that **activates** on benchmark data, (b) **orthogonality** to κ and a_i, and (c) **prediction-level** validation before combination with CW+R. Reliability redesign or an entirely different ball-level signal would require a new research phase with a new falsification test.

---

## 11. Commands reference

```bash
# Phase 6
python experiments/validate_breast_cancer.py

# Phase 7
python experiments/validate_multidataset.py

# Phase 8
python experiments/phase8_granulation_sensitivity.py
python experiments/phase8_scaling_ablation.py
python experiments/phase8_hyperparameter_grid.py
python experiments/phase8_final_protocol.py
python experiments/phase8_multiseed.py
python experiments/phase8_mechanism_diagnostics.py

# BMRU
python experiments/validate_bmru.py
python experiments/analyze_bmru_diagnostics.py
python experiments/analyze_bmru_boundary_diagnostics.py

# OBRU
python experiments/validate_overlap.py
python experiments/analyze_overlap.py

# Tests (frozen + extensions)
python -m pytest tests/test_gblistsvm.py tests/test_cw_gblstsvm.py \
  tests/test_r_gblstsvm.py tests/test_cr_gblstsvm.py \
  tests/test_bmru_gblstsvm.py tests/test_overlap_gblstsvm.py -q
```

---

*Session 2 compiled from repository docs and experiment outputs. No models were modified in producing this document.*
