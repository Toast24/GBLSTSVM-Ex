# Ball-Level Reliability Weighting (R-GBLSTSVM)

Phase 3 introduces **R-GBLSTSVM**: granular-ball LSTSVM with **ball-level
reliability weights only**. There is **no class weighting** (`kappa = 1` for
every ball). The purpose is to isolate whether ball-level information carries
signal independent of class imbalance correction.

CR-GBLSTSVM (`w_i = kappa_{y_i} · a_i`) is **not** implemented in this phase.

---

## 1. Reliability score (empirical / heuristic)

For each granular ball `i`:

```text
a_i = epsilon
      + (1 - epsilon)
        * p_i^gamma
        * exp( -lambda_r * r_tilde_i * (1 - p_i) )
```

| Symbol | Meaning |
|--------|---------|
| `p_i` | Ball purity (majority count / ball size) — frozen definition |
| `r_i` | Ball radius — mean distance from center to member points |
| `r_tilde_i` | Normalized radius (training-fold only) |
| `gamma` | Purity exponent |
| `lambda_r` | Radius × impurity penalty strength |
| `epsilon` | Positive floor in `(0, 1)` |

This is an **empirically motivated** reliability weight. It is **not** claimed
to be a theoretically derived fuzzy membership.

### Ablation variants

| Variant | Core factor |
|---------|-------------|
| `purity_only` | `p_i^gamma` |
| `radius_impurity_only` | `exp(-lambda_r * r_tilde_i * (1 - p_i))` |
| `full` | product of both |

Primary default: `full`. No variant is assumed superior without experiment.

### Documented defaults (not tuned)

| Parameter | Default |
|-----------|---------|
| `epsilon` | `0.05` |
| `gamma` | `1.0` |
| `lambda_r` | `1.0` |

---

## 2. Purity definition

Unchanged from frozen baseline (`src/granular_balls.py`):

```text
purity = (count of majority label in ball) / (ball size)
```

---

## 3. Radius definition and normalization

### Frozen radius (baseline)

```text
r_i = (1 / n_i) * sum_{x in ball_i} || x - center_i ||
```

Singleton (`n_i = 1`): all member distances are zero, so **`r_i = 0`**.

### Normalization (training-fold only)

Computed from **training balls** after granulation:

```text
scale = median( { r_i : r_i > 0 } )
```

If no positive radius exists, `scale = 1.0`.

```text
r_tilde_i = r_i / scale
```

**Rationale:** the median of positive radii is robust to outliers and excludes
zero-radius singletons from collapsing the scale. Test-set radii never enter
`scale`.

Implementation: `compute_radius_scale` in `src/reliability.py`.

---

## 4. Singleton and small-ball behavior

| Condition | Radius | Radius penalty factor |
|-----------|--------|------------------------|
| Singleton (`n_i = 1`) | `r_i = 0` | `exp(0) = 1` — **no radius discrimination** |
| Small impure ball | small `r_tilde`, `(1-p) > 0` | moderate penalty |
| Large impure ball | large `r_tilde`, `(1-p) > 0` | stronger penalty |

**Known limitation:** a pure singleton always has `p_i = 1`, so

```text
a_i = epsilon + (1 - epsilon) * 1 = 1
```

regardless of isolation. Radius alone cannot flag singleton outliers under the
frozen radius definition. **No singleton penalty** was added in Phase 3; this is
documented rather than patched with an extra heuristic.

Run `experiments/analyze_ball_reliability.py` for empirical counts of
singletons and small balls.

---

## 5. Weight placement

Mirrors CW-GBLSTSVM (`docs/CLASS_WEIGHTING.md`), but weights are `a_i` not `kappa`.

**Default (`weight_placement="opposite_only"`):**

| Subproblem | Weighted term | Weights |
|------------|---------------|---------|
| Hyperplane 1 | opposite-class `‖W^{1/2}(G θ₁ + e₂ + R⁻)‖²` | `a_i` on each **negative** ball row |
| Hyperplane 2 | opposite-class `‖W^{1/2}(H θ₂ − (e₁ + R⁺))‖²` | `a_i` on each **positive** ball row |

Own-class terms remain unweighted.

**Ablation:** `weight_placement="both"` also weights own-class rows with the
same ball reliabilities.

Solver: reuses `solve_cw_gblstsvm_systems` from `src/cw_gblstsvm.py` (generic
diagonal-weighted normal equations). R-GBLSTSVM does **not** import
`src/class_weights.py`.

### Reduction to frozen baseline

When `a_i = 1` for all balls (e.g. `gamma=0` and `lambda_r=0`, which makes the
core factor identically 1), R-GBLSTSVM reproduces frozen GBLSTSVM hyperplanes
and predictions.

---

## 6. Monotonicity (sanity properties)

With fixed `r_tilde`, lower purity (higher impurity) does not increase `a_i`.

With fixed purity, larger `r_tilde` does not increase `a_i`.

Always: `epsilon <= a_i <= 1`.

These are implementation sanity checks, not optimality claims.

---

## 7. Files

| Role | Path |
|------|------|
| Reliability functions | `src/reliability.py` |
| R-GBLSTSVM model | `src/r_gblstsvm.py` |
| Ball diagnostics | `experiments/analyze_ball_reliability.py` |
| Sanity experiment | `experiments/validate_reliability.py` |
| Tests | `tests/test_r_gblstsvm.py` |
| Frozen baseline | `src/gblstsvm.py` (unchanged) |
| CW (Phase 2 frozen) | `src/cw_gblstsvm.py` (unchanged) |

---

## 8. Known limitations

1. Singleton pure balls receive maximum reliability (`a_i = 1`).
2. Reliability parameters are not yet tuned via nested CV.
3. Extreme imbalance can leave very few minority balls after granulation;
   reliability statistics become unstable (see Phase 2 sanity check).
4. The score is heuristic; formal derivation is not claimed.
