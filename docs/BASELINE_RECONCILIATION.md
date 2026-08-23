# GBLSTSVM Baseline Reconciliation

This document records Phase 1 reconciliation between the **published GBLSTSVM
formulation**, the **reference repository implementation**, and the **clean
reproduction** in `src/`. The reference files `GBLSTSVM.py` and `gen_ball.py`
were not modified.

---

## A. Paper identity

| Field | Value |
|-------|--------|
| **Title** | Enhancing robustness and efficiency of least square twin SVM via granular computing |
| **Authors** | M. Tanveer, R. K. Sharma, A. Quadir, M. Sajid |
| **Venue** | Pattern Recognition, article **112021** (journal listing: Pattern Recognition **170**, 2026) |
| **DOI** | [10.1016/j.patcog.2025.112021](https://doi.org/10.1016/j.patcog.2025.112021) |
| **Reported code** | `https://github.com/mtanveer1/GBLSTSVM` |
| **Local citation** | `GBLSTSVM/README.md` lines 3–4 |
| **Local paper corpus** | `Papers/03_GBLSTSVM_LSGBLSTSVM_LLM.md` (derived from the PDF) |

The repository README explicitly states the shipped code was cleaned for
readability and is **not exactly the same** as used in the paper, and directs
readers to the paper for the detailed experimental setup.

---

## B. Equation mapping

Notation:

- `C` / `A`: centers of **positive** (+1) granular balls (`m1 × d`)
- `D` / `B`: centers of **negative** (−1) granular balls (`m2 × d`)
- `R+`: radii of positive balls (`m1 × 1`)
- `R−`: radii of negative balls (`m2 × 1`)
- `e1`, `e2`: column vectors of ones (length `m1`, `m2`)
- `H = [C\ e1]`, `G = [D\ e2]`
- `θ1 = [w1; b1]`, `θ2 = [w2; b2]`

### B.1 Granular-ball radius

| | |
|--|--|
| **Paper** | Section 3 (`03_GBLSTSVM…` §3): `d_j = (1/n_j) Σ ||x_i − o_j||` |
| **Reference** | `gen_ball.py` `GranularBall.__get_label_and_purity_and_r`, lines 40–43 |
| **Clean** | `src/granular_balls.py` `GranularBall._label_purity_radius` |

**Status:** aligned.

### B.2 Granular-ball purity

| | |
|--|--|
| **Paper** | Section 4: purity = majority count / ball size; split until purity ≥ threshold `ρ` |
| **Reference** | `gen_ball.py` lines 37–39, `GBList.init_granular_balls` lines 68–86 |
| **Clean** | `src/granular_balls.py` |

**Status:** aligned. Reference `main.py` also calls `del_ball(num_data=2)` to drop very small balls; exposed as `min_ball_count` in clean code.

### B.3 First twin hyperplane optimization

**Paper** (§8):

```
min_{w1,b1}  ½‖C w1 + e1 b1‖² + (c1/2) ‖D w1 + e2 b1 + e2 + R−‖²
```

Equivalent least-squares target for the second term: `G θ1 + e2 + R− → 0`.

**Normal equations** (weighting matches reference):

```
(GᵀG + (1/c1) HᵀH) θ1 = −Gᵀ(e2 + R−)
```

| | |
|--|--|
| **Reference** | `GBLSTSVM.py` `LSTWGBSVM`, lines 61–71: `inv(GᵀG + (1/c1)HᵀH + reg·I)`, RHS `−[Dᵀ(e2+R−); e2ᵀe2 + e2ᵀR−]` |
| **Clean** | `src/gblstsvm.py` `solve_gblstsvm_systems`, lines 76–84 |

**Status:** aligned (including global minus on `θ1`).

### B.4 Second twin hyperplane optimization

**Paper GBLSTSVM primal** (§8):

```
min_{w2,b2}  ½‖D w2 + e2 b2‖² + (c2/2) ‖C w2 + e1 b2 − e1 + R+‖²
```

Vector inside the second squared term:

```
r₂ = H θ2 − e1 + R+
```

Minimizing `‖r₂‖²` yields `HᵀH θ2 = Hᵀ(e1 − R+)`.

**Paper LS-GBLSTSVM constraint** (§13, same document):

```
C w2 + e1 b2 + q2 = e1 + R+
```

When `q2 = 0`, this requires `H θ2 = e1 + R+`, i.e. target `e1 + R+` — **opposite sign on `R+`** relative to §8.

| | |
|--|--|
| **Reference** | `GBLSTSVM.py` lines 75–84: `(HᵀH + (1/c2)GᵀG) θ2 = [Aᵀ(e1+R+); e1ᵀe1 + e1ᵀR+]` |
| **Clean** | `src/gblstsvm.py` lines 86–94 |

**Status:** reference and clean match **each other** and match **§13 LS-GBLSTSVM**
(`e1 + R+`). They differ from **§8 GBLSTSVM primal** on the **`R+` sign** in the
least-squares target (`+R+` vs `−R+`). The `e1` term is `+e1` in all sources for
the second hyperplane RHS; the apparent “−e1” in §8 appears **inside** the residual
`Hθ2 − e1 + R+`, not as a separate `+e1` target vector.

### B.5 c1 and c2

| | |
|--|--|
| **Paper** | §6–8: trade-off weights between the two squared-loss terms in each twin subproblem (LSTSVM heritage) |
| **Reference** | `LSTWGBSVM(Data, TestX, c1, c2)`; README demo `c1=1000`, `c2=1e−5` |
| **Clean** | `GBLSTSVM(c1=..., c2=...)` |

**Status:** **not class weights**; they scale the opposite-class separation term
relative to the own-class term in each subproblem.

### B.6 Extra diagonal regularization

| | |
|--|--|
| **Paper** | Not stated in the linear GBLSTSVM closed form (§10); LS-GBLSTSVM adds explicit SRM terms `c3, c4` |
| **Reference** | `reg_term = 0.0001` added to both normal-equation matrices (`GBLSTSVM.py` lines 31, 61–62, 75–76) |
| **Clean** | `reg_term=1e−4` default in `solve_gblstsvm_systems` |

**Status:** **implementation stabilization** in reference; documented deviation
from published linear GBLSTSVM equations.

### B.7 Prediction rule

| | |
|--|--|
| **Paper** | §11, §12 step 7: `Class(x) = argmin_{i∈{1,2}} |wᵢᵀx + bᵢ| / ‖wᵢ‖` |
| **Reference** | `GBLSTSVM.py` lines 92–95: `sign(|y2| − |y1|)` with `yᵢ = wᵢᵀx + bᵢ` (no normalization) |
| **Clean** | `predict_twin_hyperplanes`: `rule="paper"` or `rule="reference"` |

**Status:** **genuine discrepancy** between published algorithm text and reference code.

---

## C. Prediction discrepancy

### C.1 Rules in code

**Reference** (`GBLSTSVM.py`):

```
f₁(x) = w₁ᵀx + b₁,  f₂(x) = w₂ᵀx + b₂
ŷ_ref = sign(|f₂(x)| − |f₁(x)|)
```

**Paper**:

```
d₁(x) = |f₁(x)| / ‖w₁‖,  d₂(x) = |f₂(x)| / ‖w₂‖
ŷ_paper = +1  if d₁ ≤ d₂,  else −1
```

### C.2 When are they equivalent?

If `‖w₁‖ = ‖w₂‖`, then `d₁ ≤ d₂ ⟺ |f₁| ≤ |f₂|`, which matches the reference
rule except at ties (`|f₁| = |f₂|` → reference gives `sign(0) = 0`).

If `‖w₁‖ ≠ ‖w₂‖`, the two rules **need not agree** even when both hyperplanes are
identical.

### C.3 Synthetic diagnostic (seed=0, n=100, d=4, purity=0.985, c1=1000, c2=1e−5)

Run: `python experiments/diagnose_prediction_discrepancy.py`

| Metric | Value |
|--------|--------|
| Disagreement rate | **29%** (29/100 samples) |
| `‖w₁‖` | ≈ 0.523 |
| `‖w₂‖` | ≈ 0.00105 |
| `‖w₁‖/‖w₂‖` | ≈ **497** |

**Cause:** the disagreement is **not** from a bug in either rule implementation.
It is caused by:

1. **Normalization:** paper divides by `‖wᵢ‖`; reference compares raw `|fᵢ|`.
   With `‖w₁‖ ≫ ‖w₂‖`, normalized distances favor hyperplane 1 differently than
   unnormalized margins.
2. **Ties:** reference assigns label `0` when `|f₁| = |f₂|`; paper assigns `+1`
   when `d₁ ≤ d₂`.

No third hidden term appears in the diagnostic CSV: every disagreement row has
`|f₁| ≠ |f₂|` and opposite class preference under the two orderings.

Output file: `experiments/output/prediction_discrepancy_diagnostics.csv`

---

## D. Second-target discrepancy

### D.1 What the paper §8 equation means

Primal term: `‖C w₂ + e₁ b₂ − e₁ + R+‖² = ‖Hθ₂ − e₁ + R+‖²`.

Least-squares target for `Hθ₂`:

```
Hθ₂ → e₁ − R+     (not e₁ + R+)
```

Normal-equation RHS: `Hᵀ(e₁ − R+)`.

### D.2 What the reference code implements

Reference RHS uses `e₁ + R+`:

```
(HᵀH + (1/c2) GᵀG) θ₂ = Hᵀ(e₁ + R+)
```

This is the normal-equation solution for minimizing `‖Hθ₂ − (e₁ + R+)‖²`.

### D.3 Relation to baseline LSTSVM

Baseline LSTSVM second problem (§6):

```
min ½‖B w₂ + e₂ b₂‖² + (c2/2) ‖A w₂ + e₁ b₂ − e₁‖²
```

Target: `Hθ₂ → e₁` (RHS `Hᵀe₁`). GBLSTSVM reference replaces `e₁` with
`e₁ + R+` in the RHS — a natural radius extension of the **LSTSVM +e₁ target**,
not of the **§8 −e₁ + R+ residual form**.

### D.4 Classification of the discrepancy

| Hypothesis | Verdict |
|------------|---------|
| Genuine mathematical discrepancy | **Yes** between §8 GBLSTSVM primal (`R+` enters as `−R+` in target) and reference (`+R+`) |
| Notation-only | Partially: §8 writes `−e₁ + R+` **inside** one residual vector; that is equivalent to target `e₁ − R+`, not `e₁ + R+` |
| Paper typo | **Possible** — §13 LS-GBLSTSVM constraint `Hθ₂ = e₁ + R+` matches reference |
| Implementation deviation | Reference **deviates from §8 primal** but **matches §13** and LSTSVM extension pattern |

The clean implementation follows the **reference normal equations** (`e₁ + R+`).
A paper-faithful §8 variant would use `e₁ − R+` in the second RHS.

---

## E. Decision — primary baselines

| Layer | Paper-faithful | Repository-faithful | Recommendation |
|-------|----------------|---------------------|----------------|
| Granular balls | §3–4 splitting, mean-distance radius | `gen_ball.py` | **Same** — aligned |
| Optimization | §8: `θ₂` target `e₁ − R+` | `e₁ + R+` in RHS | **Repository-faithful** for this repo |
| `reg_term` | not in linear GBLSTSVM | `1e−4` diagonal | **Repository-faithful** (documented) |
| Prediction | normalized distance §11 | `sign(|f₂|−|f₁|)` | **Both preserved** |

### Primary research baseline (this project)

1. **Optimization / hyperplanes:** use **repository-faithful** target vectors
   (`e₂+R−`, `e₁+R+`) because:
   - this is what `GBLSTSVM.py` implements;
   - README cites this repository as the code artifact;
   - it matches §13 LS-GBLSTSVM constraints in the same paper;
   - §8 primal is internally inconsistent with §13 on the `R+` sign.

2. **Prediction:** expose both rules; do **not** remove either.
   - **`prediction_rule="reference"`** — reproduces shipped `LSTWGBSVM` behavior.
   - **`prediction_rule="paper"`** — reproduces published algorithm step 7 text.

3. **Default for strict reproduction experiments:** `prediction_rule="reference"`.

4. **Default for paper-aligned reporting:** `prediction_rule="paper"`.

Evidence is **sufficient** to freeze optimization as repository-faithful. The
`R+` sign in §8 remains ambiguous against §13; resolving it would require the
original PDF equation numbers or author confirmation.

### Phase 1 freeze criteria

| Criterion | Status |
|-----------|--------|
| Paper identified from repo sources | ✅ |
| Equations mapped to code | ✅ |
| Prediction discrepancy explained | ✅ |
| Second-target discrepancy explained | ✅ |
| Both prediction rules preserved | ✅ |
| Diagnostic CSV + tests | ✅ |
| Reference files untouched | ✅ |

**Phase 1 can be frozen** for scientific work **provided**:

- experiments record `prediction_rule` explicitly;
- optimization is documented as repository-faithful (`e₁+R+`);
- §8 vs §13 `R+` sign is cited as a known paper-internal inconsistency.

---

## File index

| Role | Path |
|------|------|
| Reference optimization | `GBLSTSVM/GBLSTSVM.py` |
| Reference granulation | `GBLSTSVM/gen_ball.py` |
| Clean implementation | `GBLSTSVM/src/gblstsvm.py` |
| Prediction diagnostics | `GBLSTSVM/src/prediction_diagnostics.py` |
| Diagnostic experiment | `GBLSTSVM/experiments/diagnose_prediction_discrepancy.py` |
| Tests | `GBLSTSVM/tests/test_prediction_discrepancy.py` |
