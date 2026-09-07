# Session 6 — Compression K-Sensitivity and Validation-Based Selection

**Date:** Sep 2026  
**Prior context:** Session 5 rejected online entropy granulation (>50% collapse) but showed compressed representations can beat full rho90. See [Session 5](Session_5_Split_Utility_Implementation_and_Entropy_Granulation.md).

**Scope:** Read-only experiments. No `src/` modifications.

---

## 1. Research objective

Determine whether **representation compression** (fewer balls or fewer samples) improves GBLSTSVM generalization, and whether:

1. **Ball ranking** (entropy, boundary, purity) beats random at fixed K,
2. **Granular-ball compression** differs from **sample-level compression**,
3. **Validation-based K selection** can be turned into a deployable procedure.

---

## 2. Experiments performed

| Experiment | Script | Scale | Main question |
|---|---|---|---|
| Granular compression K-sensitivity | `test_granular_compression_k_sensitivity.py` | 17 ds × 10 seeds × 13 K | Does compression improve F1? Which ranking wins? |
| Matched compression controls | `test_matched_compression_controls.py` | 170 × 7 methods | At matched ball count, does online entropy beat random/purity/boundary? |
| Validation selected K | `test_validation_selected_k.py` | 17 ds × 10 seeds | Can internal validation choose K without test leakage? |
| Complexity budget GBLSTSVM | `test_complexity_budget_gblstsvm.py` | — | Ball-count budget as explicit regularizer |
| Adaptive complexity budget | `validate_adaptive_complexity_budget.py` | — | Adaptive budget scheduling |
| Boundary selection rho90 | `validate_boundary_selection_rho90.py` | — | Boundary-ranked ball retention |
| Entropy compact validation | `validate_entropy_compact_representation.py` | — | Compact entropy representation check |
| Oracle ball selection | `oracle_ball_selection_analysis.py` | — | Upper bound on ball selection |

Shared helpers: `granular_compression_k_common.py`, `validation_selected_k_common.py`.

---

## 3. K-sensitivity results

`test_granular_compression_k_sensitivity.py` — frozen rho90 pool per dataset×seed; subsample K balls or K-matched samples.

### 3.1 Protocol

- **K meaning:** number of rho90 balls selected by ranking (not sample count).
- **Sample controls:** retain `n_samples` represented by entropy_k selection, rebuild rho90 on subset.
- **K values:** [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48].
- **Methods:** random_k, entropy_k, boundary_k, entropy_boundary_k, purity_k, stratified_random_k, sample_random_k, sample_stratified_k, full_rho90.

### 3.2 Key numbers (mean macro F1, pooled K≤10)

| Method | Mean F1 (K≤10) | vs full rho90 |
|---|---:|---|
| full_rho90 | 0.584 | — |
| entropy_k @ K=2 | **0.647** | +0.063 |
| sample_stratified_k @ K=2 | ~0.647 | +0.063 |
| random_k @ K=2 | 0.647 | +0.063 |

At K=2, **entropy ranking ≈ random ranking ≈ sample stratified** — ball selection adds little; **compression itself** drives the gain.

### 3.3 Mechanism decomposition (Session 6 answer)

| Component | Finding |
|---|---|
| **Compression** | Fewer balls/samples consistently improve F1 vs full rho90 at small K |
| **Ball ranking** | Entropy/boundary vs random: weak advantage at fixed K (Δ < 0.01 mean) |
| **Granular structure** | Ball subset vs subsample-then-granulate: **competitive**; no clear granular win |

---

## 4. Matched compression controls

`test_matched_compression_controls.py` — subsample frozen rho90 to **online-entropy ball count (~4.7)**:

| Method | Macro F1 | Collapse | Balls |
|---|---:|---:|---:|
| boundary | **0.647** | 57.1% | 4.7 |
| entropy | 0.641 | 54.7% | 4.7 |
| online_entropy | 0.610 | 51.8% | 4.7 |
| full_rho90 | 0.584 | 5.9% | 77.3 |

- Online entropy beats full rho90 (+0.022) but **loses to boundary/entropy subsampling** at matched size.
- **Compression vs selection:** some selection signal remains, but **static ball ranking matches or beats online scheduler**.

---

## 5. Validation-based K selection

`test_validation_selected_k.py` — leakage-safe protocol:

```
benchmark train → 80/20 → train_selection | validation_internal
train_selection → rho90 → random-K compression (5 reps) → val macro F1
select K on validation only → full train → compress → test
```

### 5.1 Results (test macro F1, paired)

| Focal | Baseline | Mean Δ | Wilcoxon p | Win% |
|---|---|---:|---:|---:|
| validation_selected_max | full_rho90 | **+0.039** | 0.002 | 57.6% |
| validation_selected_max | sample_stratified_matched | **+0.192** | 3.6e-20 | 85.4% |
| fixed_K10 | full_rho90 | +0.037 | 0.002 | 58.9% |

- Validation-selected K **beats full rho90** modestly (+0.039).
- Validation-selected K **beats sample_stratified_matched** strongly (+0.192) — but see Session 7: these controls were **not semantically matched**.

### 5.2 Selection stability

Median selected K ≈ **12** (IQR 12); spread across K ∈ {4, 6, 8, 10, 12, 16, 20, 24, 32}.

---

## 6. Decision (end of session)

| Claim | Verdict |
|---|---|
| Compression improves F1 vs full rho90 | **Supported** at small K |
| Entropy ranking beats random at fixed K | **Weak / not robust** |
| Online entropy scheduler is best compressor | **Rejected** — static ranking competitive |
| Validation K-selection is deployable | **Plausible** (+0.039 vs full rho90) but needs control reconciliation |
| Granular-specific advantage over sample compression | **Unresolved** — `sample_stratified_k` ≈ `entropy_k` at same K |

**Critical unresolved issue:** The validation experiment used `sample_stratified_matched` while K-sensitivity used `sample_stratified_k` — different RNG, representation, and class-weight semantics. Session 7 must reconcile before any granular-structure claim.

---

## 7. Outputs

| Directory | Description |
|---|---|
| `experiments/output/granular_compression_k_sensitivity/` | K-sweep raw + paired comparisons |
| `experiments/output/matched_compression_controls/` | Matched-ball-count ranking comparison |
| `experiments/output/validation_selected_k/` | Leakage-safe K selection |
| `experiments/output/complexity_budget_gblstsvm/` | Explicit complexity budget |
| `experiments/output/adaptive_complexity_budget/` | Adaptive budget variant |
| `experiments/output/boundary_selection_rho90/` | Boundary-ranked retention |
| `experiments/output/entropy_compact_validation/` | Compact representation check |
| `experiments/output/oracle_ball_selection/` | Oracle upper bound |

---

## 8. Handoff to Session 7

Before claiming granular structure contributes beyond compression:

1. **Reconcile** `sample_stratified_k` (K-sensitivity) vs `sample_stratified_matched` (validation).
2. **Isolate** re-granulation vs singleton representation with **identical sample indices**.
3. If re-granulation ≈ sample-compression baseline, **stop the granular-specific branch**.

---

*Session 6 compiled from experiment READMEs and summary outputs. No `src/` changes.*
