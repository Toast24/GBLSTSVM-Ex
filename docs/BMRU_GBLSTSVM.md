# Ball Margin Reliability Uncertainty (BMRU-GBLSTSVM)

**Status: FROZEN (Stages 4–6).** The BMRU formulation and model implementations
in `src/bmru.py` and `src/bmru_gblstsvm.py` are frozen for reproducibility.
Tests and experiment outputs are preserved. **Do not modify the formula or
tune thresholds** without opening a new research phase.

**Hypothesis outcome: FAILED.** The motivating claim — that high `u_i` identifies
granular balls whose training samples approach, touch, or cross the learned
TWSVM classification boundary — was tested empirically and **not supported**
under the repository default (`margin_mode="reference"`). See **§15–16** for the
full diagnostic record. No replacement uncertainty formula is implemented.

Stages 4–6 extend the frozen GBLSTSVM research pipeline with a
**granular-ball-specific geometric uncertainty** mechanism. BMRU down-weights
balls whose spatial extent is large relative to their distance from the
twin-hyperplane **classification decision boundary** (center-margin heuristic).

Frozen baselines (GBLSTSVM, CW, R, CR) are unchanged.

Implementation: `src/bmru.py`, `src/bmru_gblstsvm.py`.

---

## 1. Motivation

Class weighting (CW) addresses **class prevalence**. Reliability weighting (R)
uses **purity and normalized radius** as structural heuristics. Neither
directly measures proximity to the **learned TWSVM decision boundary**.

BMRU was designed to ask: *given the current twin hyperplanes, is this ball too
spatially extended to be a confident constraint near the classification
boundary?* The operational test was whether balls with high `u_i` empirically
**straddle or approach** the decision boundary at the training-sample level.

That test **failed** for the frozen formulation (§16). High `u_i` on diabetes
and vehicle0 is driven primarily by small **center** margins relative to mean
radius, not by member samples crossing `s(x) = 0`.

---

## 2. Why distance to a single hyperplane is insufficient

Each twin hyperplane is proximal to one class. Distance to only `w1` or only
`w2` does not identify the **class assignment boundary**. TWSVM assigns a
point to the class whose proximal hyperplane is closer (under the chosen
distance rule). Uncertainty must therefore reference the **relative** twin
distances, not either hyperplane alone.

---

## 3. Decision boundary in this repository

The clean implementation exposes two prediction rules (`src/gblstsvm.py`):

| Rule | Score | Boundary |
|------|-------|----------|
| **reference** (experiment default) | `s(x) = \|f2(x)\| − \|f1(x)\|` | `s = 0` |
| **paper** | `s(x) = d1(x) − d2(x)` with `dk = \|fk(x)\| / \|wk\|` | `s = 0` |

where `fk(x) = wkᵀ x + bk`.

These coincide when `‖w1‖ = ‖w2‖` but differ otherwise
(`docs/BASELINE_RECONCILIATION.md`, `tests/test_prediction_discrepancy.py`).

**BMRU default:** `margin_mode="reference"` so the margin matches the
repository prediction rule used in Phase 7/8 experiments. The paper-normalized
margin is available as `margin_mode="paper"`.

---

## 4. Ball uncertainty bound (heuristic — failed hypothesis)

> **Failed hypothesis.** The formula below was intended to quantify boundary
> uncertainty. Empirical boundary diagnostics show it does **not** reliably
> identify balls whose training samples straddle or hug the decision boundary
> (§16). The formula remains frozen for reproducibility; it is **not**
> endorsed as a valid boundary-uncertainty measure.

For ball `i` with center `ci` and radius `ri` (frozen mean-distance definition):

```text
s_i = s(c_i)                     # decision margin at center (see §3)
u_i = 2 r_i / (|s_i| + epsilon)  epsilon = 1e-6
mu_i = exp(-alpha * u_i)         alpha = 1.0
```

### Why the factor `2 r_i`?

Consider the paper-normalized margin `s(x) = d1(x) − d2(x)`. If each
normalized distance varies by at most `r_i` over the ball (first-order
Lipschitz heuristic using unit-normalized gradients), then
`|s(x) − s(c_i)| ≤ 2 r_i` over the ball. Dividing by `|s(c_i)|` yields a
dimensionless **relative uncertainty** `u_i`.

Under the frozen radius definition, `r_i` is the **mean** member distance, not
the maximum envelope radius. The factor `2 r_i` is therefore an approximate
scale — not a guaranteed bound — analogous to the empirical motivation of
Phase 3 reliability.

For `margin_mode="reference"`, the same formula is applied to
`s = |f2| − |f1|` using raw absolute hyperplane values.

---

## 5. BMRU weight definition

| Symbol | Meaning |
|--------|---------|
| `u_i` | Relative boundary uncertainty (larger ⇒ more uncertain) |
| `mu_i` | BMRU training weight in `(0, 1]` |
| Higher `mu_i` | More reliable / confident ball |
| Lower `mu_i` | Ball near boundary relative to its radius |

**No silent clipping or normalization** of `u_i` or `mu_i`.

Edge cases:

| Case | Result |
|------|--------|
| `r_i = 0` | `u_i = 0`, `mu_i = 1` |
| `s_i = 0` | finite `u_i` via `epsilon` |
| `‖w‖ ≈ 0` | uses `norm if norm > 0 else 1.0` (paper mode) |

---

## 6. Weight integration

All variants use the existing weighted solver
(`solve_cw_gblstsvm_systems`, default `weight_placement="opposite_only"`).

| Stage | Model | Final ball weight |
|-------|-------|-------------------|
| 4 | BMRU-GBLSTSVM | `mu_i` |
| 5 | CW+BMRU-GBLSTSVM | `kappa_{y_i} * mu_i` |
| 6 | CW+R+BMRU-GBLSTSVM | `kappa_{y_i} * a_i * mu_i` |

CW (`kappa`) and R (`a_i`) reuse frozen implementations exactly; BMRU is the
only new multiplicative factor in Stages 5–6.

---

## 7. Iterative optimization

BMRU weights depend on the current hyperplanes `(w1, b1, w2, b2)`.

```text
1. Fit ordinary GBLSTSVM  → initial hyperplanes
2. Repeat (max 5 iterations):
     a. Compute mu_i from current hyperplanes
     b. Solve weighted twin systems
     c. Monitor convergence (max |Δmu|)
3. Stop on convergence or max iterations
```

Recorded per iteration:

- iteration number
- max absolute weight change
- mean / min / max weight
- mean / min / max `u`
- count with `u < 1`, `u >= 1`

Default `weight_tolerance = 1e-8`.

---

## 8. Numerical stability

- `epsilon = 1e-6` prevents division by zero at `s_i = 0`
- Tikhonov `reg_term = 1e-4` unchanged from baseline
- `exp(-alpha * u)` checked for finite output
- Same `np.linalg.inv` convention as frozen implementations

---

## 9. Computational complexity

Per iteration: `O(n_balls · d)` for margin/weight computation plus two
`(d+1)×(d+1)` linear solves identical to CW/R/CR. With at most 5 iterations,
cost is a small constant multiple of a single GBLSTSVM fit.

---

## 10. Relationship to CW

CW up-weights opposing-class constraints for minority classes via `kappa`.
CW+BMRU multiplies that factor by `mu_i` without altering the CW formula.
The experiment asks whether BMRU adds information **beyond class importance**.

---

## 11. Relationship to R

R uses purity and normalized radius (`docs/RELIABILITY_WEIGHTING.md`). BMRU
uses **hyperplane geometry** and does not reference purity directly. Correlation
between R and BMRU weights should be measured empirically
(`experiments/analyze_bmru_diagnostics.py`).

---

## 12. Why BMRU is granular-ball-specific

BMRU requires ball center, ball radius, and **current twin hyperplanes** from
the granular-ball LSTSVM optimization. It is not defined at individual sample
level in this implementation and does not apply to non-granular LSTSVM without
an equivalent ball representation.

---

## 13. Experimental protocol

Matches Phase 7 fixed protocol (`experiments/validate_multidataset.py`):

| Parameter | Value |
|-----------|-------|
| Datasets | diabetes, ecoli2, vehicle0 |
| Split | stratified 70/15/15, seed 42 |
| `c1`, `c2` | 100, 1e-2 |
| `purity_threshold` | 0.95 |
| `min_ball_count` | 2 |
| `prediction_rule` | reference |
| `weight_placement` | opposite_only |
| BMRU `alpha` | 1.0 |
| BMRU `epsilon` | 1e-6 |
| `max_iterations` | 5 |
| `margin_mode` | reference |

Run:

```bash
cd GBLSTSVM
python -m pytest tests/test_bmru_gblstsvm.py -q
python experiments/validate_bmru.py
python experiments/analyze_bmru_diagnostics.py
python experiments/analyze_bmru_boundary_diagnostics.py
```

Outputs:

- `experiments/output/validate_bmru.json`
- `experiments/output/validate_bmru.csv`
- `experiments/output/validate_bmru_diagnostics.csv`
- `experiments/output/analyze_bmru_diagnostics.json`
- `experiments/output/analyze_bmru_per_ball.csv`
- `experiments/output/bmru_boundary_per_ball.csv`
- `experiments/output/bmru_boundary_summary.csv`
- `experiments/output/bmru_boundary_confusion.csv`
- `experiments/output/bmru_boundary_diagnostics.json`

Primary comparisons (vs existing Phase 7 results):

- BMRU vs R
- CW+BMRU vs CW
- CW+R+BMRU vs CR

---

## 14. Known limitations

1. **Failed boundary-uncertainty hypothesis:** high `u_i` does not correspond to
   empirical boundary straddling or near-boundary samples under reference margins
   (§16). Aggressive down-weighting on diabetes/vehicle0 is largely a center-margin
   artifact, not sample-level boundary crossing.
2. **Margin mode vs prediction rule:** default BMRU uses the reference margin
   while the `2 r_i` bound is motivated most directly for normalized distances.
3. **Mean radius:** `r_i` is not the maximum ball extent; uncertainty may be
   under- or over-estimated for irregular balls (max/mean extent ratio ≈ 1.2–1.8
   on benchmark datasets).
4. **Iterative dependence:** weights and hyperplanes co-evolve; no convergence
   guarantee is claimed.
5. **Possible saturation:** if most balls have `mu ≈ 1`, BMRU reduces toward
   the baseline — on diabetes/vehicle0 the opposite occurs (`mu` often ≪ 1).
6. To the best of our literature search, we did not identify an identical
   granular-ball boundary-uncertainty term in prior GBLSTSVM extensions; this
   does not imply uniqueness.

---

## 15. Freeze policy

| Item | Policy |
|------|--------|
| `u_i = 2 r_i / (\|s_i\| + \epsilon)` | **Frozen** — do not alter |
| `mu_i = exp(-\alpha u_i)` | **Frozen** — do not alter |
| `r_i` (mean member distance) | **Frozen** — do not replace with max radius |
| `alpha`, `epsilon`, `max_iterations` | **Frozen** at protocol defaults unless a new phase explicitly retunes |
| `src/bmru.py`, `src/bmru_gblstsvm.py` | **Frozen** implementations |
| `tests/test_bmru_gblstsvm.py` | **Preserved** — regression guard for frozen math |
| Experiment CSV/JSON outputs | **Preserved** — do not delete or overwrite silently |
| Replacement formula | **Not implemented** — any new uncertainty term requires a new research phase |

Rationale: Stages 4–6 completed implementation, ablation diagnostics, and
boundary falsification. The codebase must remain reproducible for audit even
though the scientific hypothesis failed.

---

## 16. Boundary-straddling falsification diagnostic

**Objective.** Test whether high `u_i` identifies balls whose **training
samples** approach, touch, or cross the learned TWSVM boundary under the
repository prediction rule (`s(x) = |f2(x)| - |f1(x)|`, boundary at `s = 0`).

**Script:** `experiments/analyze_bmru_boundary_diagnostics.py`

**Protocol:** diabetes, ecoli2, vehicle0; seed 42; Phase 7 hyperparameters;
`BMRU-GBLSTSVM`; `margin_mode="reference"` (primary) and `"paper"` (secondary).

### Empirical definitions

| Quantity | Definition |
|----------|------------|
| `boundary_straddles` | `min_x s(x) ≤ 0` and `max_x s(x) ≥ 0` over ball members |
| `boundary_near` | `min_x \|s(x)\| ≤ 10^{-6}` (repository `epsilon`) |
| `u ≥ 1` detector | BMRU flags ball as uncertain (mechanism diagnostic, not ML classifier metric) |

Per-ball extent diagnostics (diagnostic only, radius unchanged):
`max_center_distance / (r_i + \epsilon)` — actual max member distance vs mean radius.

### Results (`margin_mode="reference"`, final fit)

| Dataset | mean `u` | frac `u≥1` | frac straddling | frac near | corr(`u`, min\|s\|) | mean max-r / r |
|---------|----------|------------|-----------------|-----------|---------------------|----------------|
| diabetes | 13.52 | 0.987 | 0.000 | 0.000 | −0.133 | 1.23 |
| vehicle0 | 27.33 | 0.982 | 0.054 | 0.000 | −0.070 | 1.30 |
| ecoli2 | 0.33 | 0.000 | 0.400 | 0.000 | 0.044 | 1.76 |

**Diabetes iteration check:** mean `u` rises 2.03 (iter 1) → 3.19 (iter 5
snapshot) → 13.52 (final) while empirical straddling remains **0%** at every step.

### `u ≥ 1` mechanism confusion (`boundary_straddles`, reference mode)

| Dataset | TN | FN | FP | TP | Precision | Recall | FPR |
|---------|----|----|----|-----|-----------|--------|-----|
| diabetes | 1 | 0 | 77 | 0 | 0.00 | — | 0.99 |
| vehicle0 | 1 | 0 | 52 | 3 | 0.05 | 1.00 | 0.98 |
| ecoli2 | 6 | 4 | 0 | 0 | — | 0.00 | 0.00 |

(`boundary_near` confusion: all TP = 0 in reference mode — no ball has
`min_x |s(x)| ≤ 10^{-6}`.)

### Verdict

**NOT SUPPORTED:** high `u_i` does **not** correspond to empirical boundary
uncertainty under the frozen reference-margin formulation.

- On diabetes and vehicle0, ~98% of balls have `u ≥ 1` while almost none
  empirically straddle the boundary (0% and 5.4% respectively).
- On ecoli2, 40% straddle but none have `u ≥ 1`.
- Correlations between `u` and per-ball `min_x |s(x)|` are weak and often negative.
- The `u ≥ 1` flag has near-zero precision on diabetes and ~5% on vehicle0.

**Interpretation.** High `u_i` is driven mainly by `2 r_i / (|s(c_i)| + \epsilon)`
when the **center** margin is small relative to mean radius, not because member
samples cross the decision boundary. The factor `2 r_i` is not validated as a
geometric bound on `|s(x)|` variation over the ball in this repository.

**Paper margin mode** (`margin_mode="paper"`) shows higher straddling rates
(e.g. 28% on diabetes) but still poor alignment with `boundary_near` at
`10^{-6}` and does not rescue the reference-mode failure that motivates the
frozen experiment protocol.

### Preserved outputs

- `experiments/output/bmru_boundary_per_ball.csv`
- `experiments/output/bmru_boundary_summary.csv`
- `experiments/output/bmru_boundary_confusion.csv`
- `experiments/output/bmru_boundary_diagnostics.json`

Tests: `tests/test_bmru_gblstsvm.py` (`TestBMRUBoundaryDiagnostics`, 6 cases).

---

## File index

| Role | Path |
|------|------|
| BMRU math (frozen) | `src/bmru.py` |
| Model classes (frozen) | `src/bmru_gblstsvm.py` |
| Tests | `tests/test_bmru_gblstsvm.py` |
| Experiment | `experiments/validate_bmru.py` |
| Margin / weight diagnostics | `experiments/analyze_bmru_diagnostics.py` |
| Boundary falsification diagnostic | `experiments/analyze_bmru_boundary_diagnostics.py` |
