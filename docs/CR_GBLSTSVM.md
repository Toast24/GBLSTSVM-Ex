# Class- and Reliability-Weighted GBLSTSVM (CR-GBLSTSVM)

Phase 4 combines the Phase 2 class-weighting mechanism with the Phase 3
ball-level reliability mechanism as a **compositional product**:

```text
final_weight_i = kappa_{y_i} * a_i
```

**CR-GBLSTSVM is an empirical combination of class-level imbalance weighting
and ball-level reliability weighting.** The product is not claimed to be
theoretically optimal; it is an empirical compositional weighting mechanism
for testing whether reliability adds information beyond class weighting alone.

Frozen components (unchanged):

- Granular-ball generation (`src/granular_balls.py`)
- Baseline solver (`src/gblstsvm.py`)
- CW solver and class weights (`src/cw_gblstsvm.py`, `src/class_weights.py`)
- R reliability formula (`src/reliability.py`)

---

## 1. Class weight (Phase 2)

For binary training labels, each class `c ∈ {+1, −1}` receives:

```text
kappa_c = n_train / (2 · n_c)
```

where `n_c` counts **original training samples**, not balls.

Each ball with label `y_i` receives class factor `kappa_{y_i}`. Ball size
(`num_samples`) is **not** multiplied.

See `docs/CLASS_WEIGHTING.md` for full Phase 2 details.

---

## 2. Reliability weight (Phase 3)

For each granular ball `i`:

```text
a_i = epsilon
      + (1 - epsilon)
        * p_i^gamma
        * exp( -lambda_r * r_tilde_i * (1 - p_i) )
```

| Parameter | Default |
|-----------|---------|
| `epsilon` | `0.05` |
| `gamma` | `1.0` |
| `lambda_r` | `1.0` |

Purity, radius, and radius normalization follow the frozen Phase 3
implementation. See `docs/RELIABILITY_WEIGHTING.md`.

---

## 3. Combined product weighting

```text
final_weight_i = kappa_{y_i} * a_i
```

Implementation (`src/cr_gblstsvm.py`):

```text
final_negative_weights = kappa_neg * reliability_negative
final_positive_weights = kappa_pos * reliability_positive
```

Class and reliability factors are computed **independently** and multiplied
elementwise. There is no hidden normalization, no ball-size multiplier, and
no prediction-time weighting.

---

## 4. Weight placement

**Primary default: `weight_placement="opposite_only"`**

| Subproblem | Weighted term | Per-row weights |
|------------|---------------|-----------------|
| Hyperplane 1 | opposite-class separation | `final_weight` on each negative ball |
| Hyperplane 2 | opposite-class separation | `final_weight` on each positive ball |

Own-class terms remain unweighted.

**Secondary: `weight_placement="both"`** — retained for ablation only; not
the primary model.

The solver delegates to `solve_cw_gblstsvm_systems` (shared with CW and R).

---

## 5. Reduction properties

| Condition | Reduction |
|-----------|-----------|
| `a_i = 1` for all balls (`gamma=0`, `lambda_r=0`) | CR-GBLSTSVM ≡ CW-GBLSTSVM |
| `kappa_pos = kappa_neg = 1` (balanced training data) | CR-GBLSTSVM ≡ R-GBLSTSVM |
| Both conditions | CR-GBLSTSVM ≡ frozen GBLSTSVM |

These are tested at hyperplane-parameter and prediction levels in
`tests/test_cr_gblstsvm.py`.

**Balanced-data property:** when `n_pos = n_neg`, `kappa_pos = kappa_neg = 1`,
so CR equals R regardless of reliability values. The CR contribution on
balanced data is therefore entirely through the R mechanism, not an independent
third class-reliability interaction.

---

## 6. Implementation decisions

| Decision | Rationale |
|----------|-----------|
| Reuse CW weighted solver | CR differs only in weight construction |
| Separate `ball_cr_weights()` helper | Explicit product; testable intermediate factors |
| Store `kappa_pos_balls_`, `kappa_neg_balls_`, `reliabilities_` | Diagnostics and reduction tests |
| No changes to frozen Phase 1–3 files | Phase isolation per research protocol |
| Phase 2/3 defaults unchanged | No tuning in Phase 4 |

---

## 7. Relationship to CW and R variants

| Variant | Weight per ball |
|---------|-----------------|
| GBLSTSVM (baseline) | `1` |
| CW-GBLSTSVM | `kappa_{y_i}` |
| R-GBLSTSVM | `a_i` |
| CR-GBLSTSVM | `kappa_{y_i} * a_i` |

**Scientific question (Phase 4):** does adding reliability on top of class
weighting (`CR − CW`) improve metrics beyond what class weighting alone provides?

Secondary comparison: `CR − R` measures the class-weight contribution when
reliability is already active.

---

## 8. Limitations

- The product `kappa * a_i` is not derived from a unified loss function.
- High purity saturation (Phase 3 finding) may make `a_i ≈ 1`, reducing CR
  to CW in practice on some datasets.
- Singleton balls with `r=0` and `p=1` receive `a_i=1` (documented Phase 3
  limitation); no singleton correction is applied.
- No density, boundary weighting, or granulation redesign in this phase.
- Defaults are not tuned; results are preliminary validation only.

---

## 9. Files

| File | Purpose |
|------|---------|
| `src/cr_gblstsvm.py` | `CRGBLSTSVM`, `ball_cr_weights`, `solve_cr_gblstsvm_systems` |
| `tests/test_cr_gblstsvm.py` | Reduction, weight construction, placement tests |
| `experiments/validate_cr_gblstsvm.py` | Synthetic sanity experiment |
| `experiments/output/cr_weight_diagnostics.csv` | Per-ball diagnostic export |
| `docs/CR_GBLSTSVM.md` | This document |
