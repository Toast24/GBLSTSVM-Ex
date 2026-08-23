# GBLSTSVM Research Specification

## 1. Project Objective

This project investigates whether **ball-level reliability information provides predictive value beyond class-level weighting in Granular Ball Least Squares Twin Support Vector Machines (GBLSTSVM)**.

The project is deliberately structured as a controlled progression:

```text
GBLSTSVM
    ↓
CW-GBLSTSVM
    ↓
CR-GBLSTSVM
    ↓
CRB-GBLSTSVM (secondary/optional)
```

Where:

- **GBLSTSVM** = paper-faithful baseline
- **CW-GBLSTSVM** = class-weighted baseline
- **CR-GBLSTSVM** = class-weighted + ball-level reliability
- **CRB-GBLSTSVM** = CR + boundary emphasis

The primary scientific comparison is:

```text
CR-GBLSTSVM vs CW-GBLSTSVM
```

The key research question is:

> Does ball-level reliability contain predictive information beyond class-level weighting?

The project must remain falsifiable. We do not assume that CR-GBLSTSVM will outperform CW-GBLSTSVM.

---

# 2. Current Repository

The repository contains an existing implementation of GBLSTSVM.

Important files should be discovered from the actual repository rather than assumed from this document.

Potentially relevant components include:

- GBLSTSVM implementation
- granular-ball generation
- dataset/preprocessing code
- tests
- experiment scripts
- documentation

The original implementation must be treated as a **reference artifact**.

Do not overwrite it while developing the clean reproduction.

---

# 3. Phase 1 — GBLSTSVM Reimplementation

Before implementing any proposed extension:

1. Inspect the existing repository.
2. Identify the actual GBLSTSVM implementation.
3. Identify the granular-ball generation procedure.
4. Identify preprocessing.
5. Identify the actual optimization procedure.
6. Reconstruct the equations from the code.
7. Compare the implementation with the published formulation.
8. Implement a clean reproduction.
9. Validate the reproduction numerically against the reference implementation.
10. Record discrepancies.

No research extension should be implemented until this baseline is sufficiently validated.

---

# 4. Source of Truth Hierarchy

When information conflicts, use the following priority:

1. Published mathematical formulation
2. Actual repository source code
3. Existing repository documentation
4. Existing comments
5. Previous model-generated interpretations

Model-generated summaries are NOT authoritative.

Do not convert an interpretation into a mathematical specification without verifying it against the source.

---

# 5. Baseline Reimplementation Requirements

The clean implementation should:

- support binary classification;
- expose a simple sklearn-like API;
- support configurable `c1` and `c2`;
- support deterministic experiments;
- preserve the actual granular-ball representation;
- reproduce the paper-faithful optimization;
- use numerical stabilization only where justified;
- document any deviation from the paper.

Desired interface:

```python
model = GBLSTSVM(
    c1=...,
    c2=...
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

The exact mathematical implementation must be derived from the paper and verified against the reference code.

---

# 6. Reference vs Reimplementation

The repository's original implementation and the clean reproduction must remain distinguishable.

Use the following conceptual structure:

```text
Reference implementation
        │
        │ validation
        ▼
Clean GBLSTSVM implementation
        │
        │ validated baseline
        ▼
Research extensions
```

Do not silently modify the reference implementation to make it agree with the proposed method.

---

# 7. Baseline Validation

Validation should compare the reference implementation and clean reproduction on deterministic datasets.

Where applicable compare:

- granular-ball count;
- ball centers;
- ball radii;
- ball labels;
- purity;
- constructed matrices;
- target vectors;
- hyperplane parameters;
- objective values;
- predictions;
- evaluation metrics.

Exact equality is not required when the implementations use different numerical solvers or floating-point operations.

Use explicit numerical tolerances.

Any discrepancy must be explained rather than ignored.

---

# 8. Phase 2 — Class-Weighted GBLSTSVM

After the baseline is validated, implement class weighting.

For binary classes, the initial class-cost convention is:

```text
kappa_c = n_train / (2 n_c)
```

where:

- `n_train` = number of training samples;
- `n_c` = number of original training samples belonging to class `c`.

Class costs must be based on original samples, NOT granular-ball counts.

If normalization is used, document it explicitly.

The exact location where class costs enter the optimization must be justified mathematically.

The class-weighted model is:

```text
CW-GBLSTSVM
```

Its purpose is to establish the strongest simple class-level control against which ball-level reliability can be tested.

---

# 9. Phase 3 — Ball-Level Reliability

The primary reliability experiment will investigate:

```text
a_i =
    epsilon
    + (1 - epsilon)
      p_i^gamma
      exp[-lambda_r * rtilde_i * (1 - p_i)]
```

where:

- `p_i` = purity of granular ball `i`;
- `rtilde_i` = normalized ball radius;
- `epsilon > 0`;
- `gamma >= 0`;
- `lambda_r >= 0`.

This is an **empirically motivated reliability score**.

It must NOT be described as theoretically derived unless a formal derivation is subsequently established.

Use the terminology:

> ball-level reliability weight

rather than claiming that it is automatically a theoretically grounded fuzzy membership.

---

# 10. Combined Class + Reliability Weight

The primary combined weight is:

```text
w_i = kappa_{y_i} * a_i
```

where:

- `kappa` represents class-level cost;
- `a_i` represents ball-level reliability.

The main comparison is therefore:

```text
CW-GBLSTSVM
        vs
CR-GBLSTSVM
```

This isolates whether ball-level information contributes beyond class imbalance correction.

---

# 11. Radius Handling

Radius must first be investigated in the paper-faithful baseline.

Determine whether the original GBLSTSVM formulation already uses ball radius in:

- ball construction;
- target vectors;
- optimization matrices;
- regularization;
- prediction.

If radius already affects the baseline geometry, distinguish that usage from its later use as a reliability signal.

Do not silently assume that radius is an independent contribution.

Radius normalization must use training-fold information only.

Potential robust normalization should be investigated rather than assuming a particular normalization in advance.

---

# 12. Purity Handling

Before using purity as a continuous reliability signal, measure its empirical distribution.

For each dataset/granulation configuration inspect:

- minimum;
- maximum;
- mean;
- standard deviation;
- 5th percentile;
- median;
- 95th percentile.

If almost all balls have purity close to 1, purity may provide little continuous information.

This must be determined empirically.

---

# 13. Weight Placement

The primary CR formulation should initially investigate weighting of the **opposite-class separation term**.

The motivation is:

> unreliable balls should contribute less as constraints imposed by the opposing class.

Both-term weighting should be treated as an explicit ablation rather than assumed to be superior.

Compare:

1. opposite-class term only;
2. both terms.

---

# 14. Singleton and Small-Ball Handling

Singleton balls have:

```text
r_i = 0
```

under common radius definitions.

Therefore radius alone cannot identify singleton outliers.

The implementation must explicitly investigate the behavior of:

- singleton balls;
- very small balls;
- small impure balls;
- isolated minority points.

Do not introduce an additional singleton penalty without experimental justification.

---

# 15. Boundary Variant

A secondary experiment may investigate:

```text
b_i = 4 p_i (1 - p_i)
```

as a boundary-related factor.

A possible extension is:

```text
a_i_plus = a_i * exp(-lambda_o * b_i)
```

This is NOT the primary research contribution.

CR must be validated before CRB is implemented.

The interpretation:

```text
impurity ≈ boundary usefulness
```

must not be assumed to be universally correct.

---

# 16. Required Model Variants

At minimum:

### Baselines

1. GBLSTSVM
2. CW-GBLSTSVM

### Reliability variants

3. R-GBLSTSVM
4. CR-GBLSTSVM

### Optional boundary variant

5. CRB-GBLSTSVM

---

# 17. Required Reliability Ablations

For CR, investigate:

1. purity only;
2. radius only;
3. purity + radius.

For weighting placement:

1. opposite-class term only;
2. both terms.

These experiments determine whether the full reliability score actually provides additive information.

---

# 18. Experimental Conditions

## Balanced

Approximately:

```text
1:1
```

Purpose:

Determine whether reliability introduces unnecessary degradation when imbalance is absent.

## Imbalanced

Use feasible controlled ratios such as:

```text
1:1
1:3
1:5
1:10
```

Do not force ratios that leave insufficient minority samples for reliable ball generation.

## Label Noise

Investigate:

- symmetric label noise;
- majority-class noise;
- minority-class noise;
- boundary-targeted noise;
- isolated/outlier corruption where feasible.

## Granulation Sensitivity

Evaluate multiple granularization configurations.

The method must not depend on one arbitrary granulation setting.

---

# 19. Required Evaluation Metrics

Do not rely only on accuracy.

At minimum consider:

- accuracy;
- balanced accuracy;
- macro F1;
- minority recall;
- minority precision;
- G-mean;
- AUPRC where appropriate;
- runtime.

For noisy settings also report degradation relative to the corresponding clean condition.

Use repeated stratified experiments where computationally feasible.

Report uncertainty/confidence intervals.

---

# 20. Required Baselines / Competitors

Investigate and justify the inclusion of:

- paper-faithful GBLSTSVM;
- original repository GBLSTSVM;
- CW-GBLSTSVM;
- R-GBLSTSVM;
- CR-GBLSTSVM;
- GBFTSVM;
- relevant fuzzy/density-weighted LSTSVM;
- standard LSTSVM;
- appropriate classical baselines.

The exact competitor set must be determined from the literature and implementation feasibility.

Do not claim novelty merely because a model name is different.

---

# 21. Leakage Prevention

All learned preprocessing and statistics must be computed using training data only.

This includes:

- feature scaling;
- radius normalization statistics;
- reliability normalization;
- hyperparameter selection.

Test data must never influence training-time reliability calculations.

---

# 22. Experiment Reproducibility

Every experiment must record:

- dataset;
- random seed;
- train/test split;
- preprocessing;
- granulation parameters;
- `c1`;
- `c2`;
- reliability parameters;
- model variant;
- metrics;
- runtime.

Results should be stored in machine-readable formats such as CSV/JSON.

---

# 23. Scientific Decision Rule

The central quantity is:

```text
Delta_reliability =
    Performance(CR-GBLSTSVM)
    -
    Performance(CW-GBLSTSVM)
```

The project does NOT assume:

```text
Delta_reliability > 0
```

A valid conclusion may be:

1. reliability consistently improves performance;
2. reliability helps only under particular conditions;
3. reliability provides little additional value beyond class weighting;
4. reliability harms performance.

All outcomes are scientifically acceptable.

The implementation and experiments must not be tuned solely to produce a positive result.

---

# 24. Development Rules for Coding Agents

Coding agents must:

1. Inspect existing code before changing it.
2. Preserve the original implementation.
3. Avoid unnecessary refactoring.
4. Avoid inventing mathematical behavior.
5. Distinguish source-code facts from interpretations.
6. Write tests for mathematical behavior.
7. Run tests after substantive changes.
8. Keep experimental results reproducible.
9. Record important implementation decisions.
10. Never silently alter the baseline to improve performance.

During Phase 1, do NOT implement CW, CR, or CRB.

The immediate goal is:

```text
EXISTING GBLSTSVM
        ↓
CLEAN PAPER-FAITHFUL GBLSTSVM
        ↓
NUMERICAL VALIDATION
```
