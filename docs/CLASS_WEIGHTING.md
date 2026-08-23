# Class Weighting in CW-GBLSTSVM

Phase 2 introduces **CW-GBLSTSVM**, a class-weighted control built on the
frozen GBLSTSVM baseline. Granular-ball generation, radius targets,
`c1`/`c2` semantics, preprocessing, and prediction rules are unchanged.

---

## 1. Class cost definition

For binary training labels, each class `c ∈ {+1, −1}` receives:

```text
kappa_c = n_train / (2 · n_c)
```

where:

- `n_train` = number of **original training samples**;
- `n_c` = number of **original training samples** in class `c`.

### Why original sample counts?

Class imbalance is a property of the training distribution over individual
samples. Granular balls are a compressed representation; ball counts and
purities depend on splitting thresholds and geometry. Using ball counts would
conflate granulation artifacts with class prevalence and would break the
intended comparison against reliability weighting (where `w_i = kappa_{y_i} · a_i`
uses the same `kappa` from sample counts).

---

## 2. Mapping class costs to balls

Each granular ball `i` with assigned label `y_i` receives:

```text
class_weight_i = kappa_{y_i}
```

**Ball size (`num_samples`) is not multiplied.** The frozen GBLSTSVM
optimization already treats each ball as one row in the center matrices `C`
and `D`; there is no per-ball multiplicity factor in the normal equations.
Introducing `num_samples` would be an unsupported reinterpretation of the
baseline least-squares structure.

Positive balls (+1) and negative balls (−1) are split as in
`balls_to_class_arrays`; weights align with those row orders.

---

## 3. Where weights enter the optimization

The frozen twin problems (repository-faithful targets) are:

**Hyperplane 1** (own class: positive balls `C`, opposite: negative balls `D`):

```text
min  ½ ‖C w₁ + e₁ b₁‖²  +  (c₁/2) ‖D w₁ + e₂ b₁ + e₂ + R⁻‖²
```

**Hyperplane 2** (own class: negative balls `D`, opposite: positive balls `C`):

```text
min  ½ ‖D w₂ + e₂ b₂‖²  +  (c₂/2) ‖C w₂ + e₁ b₂ − e₁ + R⁺‖²
```

(`R⁺` enters the implemented RHS as `e₁ + R⁺`; see
`docs/BASELINE_RECONCILIATION.md`.)

### Primary placement: opposite-class term only

**Default (`weight_placement="opposite_only"`):**

| Subproblem | Weighted term | Per-row weights |
|------------|---------------|-----------------|
| Hyperplane 1 | `‖W_opp^{1/2}(G θ₁ + e₂ + R⁻)‖²` | `κ_{−1}` on each negative ball row |
| Hyperplane 2 | `‖W_opp^{1/2}(H θ₂ − (e₁ + R⁺))‖²` | `κ_{+1}` on each positive ball row |

Own-class terms (`‖H θ₁‖²`, `‖G θ₂‖²`) remain unweighted.

Motivation: class cost up-weights how strongly **opposing-class** balls act as
separation constraints — the natural analogue of weighted misclassification
penalties in twin SVM formulations.

### Weighted normal equations

Let `G = [D | e₂]`, `H = [C | e₁]`, `W_opp` diagonal on opposite-class rows.

**Hyperplane 1:**

```text
(Gᵀ W_opp G + (1/c₁) Hᵀ H) θ₁ = −Gᵀ W_opp (e₂ + R⁻)
```

**Hyperplane 2:**

```text
(Hᵀ W_opp H + (1/c₂) Gᵀ G) θ₂ = Hᵀ W_opp (e₁ + R⁺)
```

Implementation: `src/cw_gblstsvm.py` → `solve_cw_gblstsvm_systems`.

### Ablation (not default): both terms

`weight_placement="both"` additionally weights own-class terms with
`κ_{+1}` on positive rows (hp1) and `κ_{−1}` on negative rows (hp2). Reserved
for explicit ablation; not used in the primary CW control.

---

## 4. `c1` and `c2` unchanged

`c1` and `c2` still trade off own-class vs opposite-class squared loss within
each subproblem. They are **not** replaced by `kappa` and are not reinterpreted
as class weights.

---

## 5. Balanced-data reduction

If `n₊ = n₋`, then `kappa₊ = kappa₋ = 1`. With `weight_placement="opposite_only"`,
all `W_opp = I` and the CW normal equations equal the frozen baseline. CW-GBLSTSVM
must reproduce GBLSTSVM hyperplanes and predictions (up to floating-point tolerance).

---

## 6. Prediction

CW-GBLSTSVM inherits `prediction_rule` from GBLSTSVM:

- `"reference"` — repository rule (primary for experiments);
- `"paper"` — normalized distance rule.

Experiments must record `prediction_rule` explicitly.

---

## 7. Files

| Role | Path |
|------|------|
| Class costs | `src/class_weights.py` |
| CW model + weighted solver | `src/cw_gblstsvm.py` |
| Frozen baseline (unchanged) | `src/gblstsvm.py` |
| Tests | `tests/test_cw_gblstsvm.py` |
| Sanity experiment | `experiments/validate_class_weighting.py` |
