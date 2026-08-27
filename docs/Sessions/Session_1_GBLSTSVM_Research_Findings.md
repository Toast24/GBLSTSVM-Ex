# Session 1 — GBLSTSVM Research Findings

**Project status:** Phases 1–5 complete; real-dataset validation is the next step.

## 1. Research Goal

Investigate whether GBLSTSVM can be extended to jointly address:

- class imbalance;
- granular-ball purity;
- geometric spread/radius;
- local inter-class overlap.

The proposed progression is:

```text
GBLSTSVM
   |
   +-- CW-GBLSTSVM  (class imbalance)
   |
   +-- R-GBLSTSVM   (ball reliability)
   |
   +-- CR-GBLSTSVM  (class weight × reliability)
```

The central hypothesis is:

> Class weighting identifies which class needs greater emphasis, while local ball reliability identifies which regions should be trusted more or less.

The hypothesis is **not yet confirmed**.

---

## 2. Phase 1 — Frozen Baseline Reproduction

Reference paper:

**Tanveer, M., Sharma, R. K., Quadir, A., & Sajid, M. (2025).**  
*Enhancing robustness and efficiency of least square twin SVM via granular computing.*  
Pattern Recognition, article 112021.

Reference artifacts:

- `GBLSTSVM.py`
- `gen_ball.py`

Clean implementation:

- `src/gblstsvm.py`
- `src/granular_balls.py`
- `src/reference_pipeline.py`

The reference artifacts were left unchanged.

### Reproduction result

On the deterministic synthetic case:

- seed = 0
- n = 100
- d = 4
- purity = 0.985
- min ball count = 2
- c1 = 1000
- c2 = 1e-5

Results:

- clean balls = 18
- reference balls = 18
- max center difference = 0
- max radius difference = 0
- reference-rule prediction agreement = 100%

Hyperplanes, matrices, targets, balls, and reference predictions were numerically reproduced.

---

## 3. Important Baseline Discrepancy

The paper and repository use different prediction rules.

### Paper

```text
argmin_i |f_i(x)| / ||w_i||
```

### Repository

```text
sign(|f2(x)| - |f1(x)|)
```

These are not equivalent.

On the deterministic diagnostic:

- agreement = 71%
- disagreement = 29%
- `||w1|| = 0.643`
- `||w2|| = 0.00057`
- norm ratio ≈ 1118

Therefore both rules are preserved:

```text
prediction_rule="reference"
prediction_rule="paper"
```

The repository-faithful rule is the primary baseline for reproduction. Every experiment should explicitly record the rule used.

The optimization also follows the repository implementation, including:

- `e2 + R-` in the first RHS;
- `e1 + R+` in the second RHS;
- `reg_term = 1e-4`;
- radius entering through the target vectors.

The paper's Section 8 and implementation/Section 13 contain an `R+` sign inconsistency; the project follows the repository/Section 13 implementation.

---

## 4. Phase 2 — CW-GBLSTSVM

Created:

- `src/class_weights.py`
- `src/cw_gblstsvm.py`
- tests, validation script, and documentation.

Class weight:

```text
kappa_c = n_train / (2 * n_c)
```

Each ball receives:

```text
class_weight_i = kappa_{y_i}
```

There is **no ball-size multiplier**.

Primary placement:

```text
weight_placement = "opposite_only"
```

Weights affect optimization, not prediction.

When the dataset is balanced:

```text
kappa+ = kappa- = 1
```

so CW exactly reduces to the frozen baseline.

---

## 5. Phase 3 — R-GBLSTSVM

Reliability:

```text
a_i =
epsilon
+ (1-epsilon)
  * p_i^gamma
  * exp(-lambda_r * r_tilde_i * (1-p_i))
```

Training-fold radius normalization:

```text
scale = median({r_i : r_i > 0})
r_tilde_i = r_i / scale
```

Defaults:

```text
epsilon  = 0.05
gamma    = 1.0
lambda_r = 1.0
```

Verified properties:

- lower purity → lower reliability;
- larger radius at fixed impurity → lower reliability;
- `a_i >= epsilon`;
- identical balls → identical reliability;
- unit reliability → exact baseline.

### Problems discovered

**Purity saturation:** at high thresholds, especially `rho=0.95`, most balls are pure and `a_i ≈ 1`.

**Singleton issue:** singleton balls have `r=0`, so pure singletons receive maximum reliability. Standard `min_ball_count=2` removes them.

**Purity proxy behavior:** when impurity is rare, reliability is dominated by `p^gamma`.

---

## 6. Phase 4 — CR-GBLSTSVM

Combined weight:

```text
final_weight_i = kappa_{y_i} * a_i
```

Created:

- `src/cr_gblstsvm.py`
- tests, validation, diagnostics, and documentation.

Primary placement:

```text
weight_placement = "opposite_only"
```

The solver reuses the CW solver.

### Verified reductions

```text
CR -> CW   when a_i = 1
CR -> R    when kappa+ = kappa- = 1
CR -> baseline when both are disabled
balanced CR == R
```

All reductions were tested.

### Initial synthetic result

CR showed:

```text
CR - CW = 0
```

across the tested scenarios and metrics.

Diagnostics showed that under default `rho=0.95`, imbalanced datasets often had:

```text
a_i = 1
```

for all balls.

Therefore:

```text
CR = kappa * 1 = CW
```

The initial null result was therefore largely explained by reliability saturation.

---

## 7. Phase 5 — Granulation Sensitivity

Purpose:

> Determine whether granulation settings can expose meaningful reliability variation before moving to real datasets.

Granulation:

```text
initial ball
   |
   +-- if purity < threshold
   |      and n > minimum
   |
   +-- KMeans(k=2)
   |
   +-- recursively split
   |
   +-- filter balls below min_ball_count
```

Lower threshold produces more splitting and more impurity.

### Main result

Meaningful reliability variation appeared around:

```text
rho = 0.80–0.85
```

with reliability standard deviation reaching approximately 0.07–0.12 and substantial fractions of balls below 0.95 reliability.

At:

```text
rho >= 0.95
```

reliability frequently saturated on imbalanced data.

### Prediction-level result

Even when CR weights differed numerically from CW, predictions usually did not change.

For imbalanced 1:3 data, multi-seed experiments showed essentially:

```text
Delta(CR - CW) = 0
```

across tested thresholds.

One isolated positive result occurred:

```text
1:5 imbalance
rho = 0.85
seed = 42
Delta Macro-F1 = +0.014
```

This is not evidence of superiority.

One negative example:

```text
balanced
rho = 0.90
seed = 42
Delta Macro-F1 = -0.031
```

Thus the effect is currently sparse and inconsistent.

---

## 8. Current Scientific Interpretation

The evidence currently supports:

1. The clean implementation reproduces the repository baseline.
2. CW is a valid controlled extension.
3. Reliability has the intended mathematical monotonicity and reductions.
4. Reliability often saturates under aggressive purity thresholds.
5. Lower thresholds restore reliability dynamic range.
6. Dynamic range alone has **not** produced consistent predictive gains over CW on synthetic data.

Therefore:

> The hypothesis that reliability provides useful information beyond class weighting remains unresolved and currently has weak empirical support.

Do **not** claim that CR is superior.

Do **not** claim that reliability is useless.

---

## 9. Frozen Components

Treat these as frozen unless a later experiment provides a deliberate scientific reason to revise them:

```text
src/gblstsvm.py
src/granular_balls.py
src/cw_gblstsvm.py
src/r_gblstsvm.py
src/cr_gblstsvm.py
src/class_weights.py
src/reliability.py
```

Current test status:

```text
59 passed
0 failed
```

---

## 10. Phase 6 — Real-Data Validation

This is the next step.

Do **not** tune the model first.

First select a small, scientifically motivated suite of real binary datasets spanning:

- class imbalance;
- class overlap;
- sample size;
- dimensionality.

Compare at minimum:

```text
GBLSTSVM
CW-GBLSTSVM
R-GBLSTSVM
CR-GBLSTSVM
```

Potential conventional baselines can be added later.

### Training-only requirement

Only the training split may determine:

- scaling;
- granular balls;
- class weights;
- radius normalization;
- reliability.

No test information may enter these calculations.

### Metrics

Prioritize:

- Macro-F1;
- Balanced Accuracy;
- Minority Recall;
- Minority Precision;
- G-mean where appropriate;
- AUROC where appropriate.

Use multiple stratified seeds and report mean ± standard deviation.

### Diagnostics

Record:

```text
number of balls
ball counts by class
purity statistics
radius statistics
reliability statistics
fraction reliability < 0.95
fraction reliability >= 0.99
fraction of CR weights differing from CW
kappa values
```

This distinguishes two important outcomes:

```text
reliability has no variation
        -> CR ~= CW
```

from:

```text
reliability varies
        -> CR weights differ
        -> performance still does not improve
```

The second result would be much stronger evidence against the current reliability formulation.

---

## 11. Hyperparameter Policy for First Real-Data Run

Use the existing defaults:

```text
epsilon  = 0.05
gamma    = 1.0
lambda_r = 1.0
```

Use a predetermined granulation set such as:

```text
rho in {0.80, 0.85, 0.90, 0.95}
```

Do not optimize these specifically to make CR win.

Hyperparameter tuning should only happen after the untuned real-data experiment establishes whether there is a real signal.

---

## 12. Decision Framework

### If CR consistently improves

Proceed to:

```text
controlled tuning
-> ablations
-> statistical tests
-> paper preparation
```

### If CR differs from CW but does not improve

Investigate whether:

- reliability is poorly aligned with boundary uncertainty;
- radius is the wrong geometric quantity;
- ball weighting is too weak;
- opposite-only placement limits the effect.

### If CR always collapses to CW

The current reliability mechanism is likely redundant under the tested granulation regime.

---

# Final Session 1 Conclusion

The project has progressed from **repository reproduction** to a fully tested four-model research framework.

What has been demonstrated:

```text
correct reproduction
+
mathematical reduction tests
+
class weighting
+
reliability weighting
+
combined weighting
+
granulation sensitivity analysis
```

What remains unknown:

> **Does reliability contain useful information beyond class imbalance weighting on real datasets with heterogeneous local overlap?**

That is the central question for Phase 6.

**Next action: select the real dataset suite. Do not modify the model before that experiment.**
