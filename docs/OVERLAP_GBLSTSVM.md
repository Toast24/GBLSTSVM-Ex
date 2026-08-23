# Overlap-aware GBLSTSVM (OBRU, Stages 7–9)

## 1. Motivation

Granular balls from opposite classes may compete in feature space near decision
boundaries. A ball-level signal based on geometric proximity to opposite-class
balls may complement class-level (CW) and purity/radius reliability (R) weights.
This module explores **overlap-aware reweighting (OBRU)** as a fixed geometric
factor computed after granulation.

## 2. Inter-class overlap proxy

For opposite-class balls *i* and *j*, with centers **c** and mean-distance radii
*r* (frozen repository definition):

```
d_ij = ||c_i - c_j||
O_ij = max(0, (r_i + r_j - d_ij) / (r_i + r_j + epsilon))
```

Because *r* is a mean-distance radius rather than a strict geometric envelope,
**O_ij is an overlap proxy**, not a guarantee of set intersection.

Defaults: `epsilon = 1e-6`.

## 3. Top-3 overlap definition

For each ball *i*:

```
O_i = mean of top-3 O_ij over opposite-class balls j
```

If fewer than three opposite-class balls exist, average all available values.
If no opposite-class balls exist, `O_i = 0`.

## 4. Class-conditional weighting

Minority and majority are defined from **training-label counts** (same convention
as Phase 7/8 experiments).

```
Minority ball:  mu_i = 1 + alpha * O_i
Majority ball:  mu_i = 1 - alpha * O_i
```

Default: `alpha = 1.0`. Weights are **not clipped or normalized**.

Interpretation (cautious): overlapping minority balls may mark difficult minority
regions; overlapping majority balls may mark redundant majority-side competition.

## 5. CW integration

Stage 8 (`CW+OBRU-GBLSTSVM`):

```
W_i = kappa_{y_i} * mu_i
```

Class weights reuse the frozen CW implementation exactly (`kappa_c = n / (2 n_c)`).

## 6. R integration

Stage 9 (`CW+R+OBRU-GBLSTSVM`):

```
W_i = kappa_{y_i} * a_i * mu_i
```

Reliability `a_i` reuses the frozen R implementation exactly. OBRU is the only
new multiplicative factor.

## 7. Computational complexity

Let *m* be the number of balls. Pairwise opposite-class distances require
O(m²) work in the worst case. Overlap is computed once from fixed ball geometry;
no hyperplane-dependent iteration is required (unlike margin-based BMRU).

## 8. Numerical limitations

- Majority weights can reach zero when `O_i = 1` and `alpha = 1`.
- Negative majority weights occur if `O_i > 1` (possible for the proxy when
  centers coincide and radii are positive).
- The implementation raises on non-finite or negative weights; it does **not**
  apply ad hoc clipping.
- Zero-weight balls effectively drop out of the weighted normal equations, consistent
  with existing diagonal-weight usage in CW/R solvers.

## 9. Experimental protocol

Benchmark datasets: `diabetes.csv`, `ecoli2.csv`, `vehicle0.csv`.

Fixed settings (Phase 7/8 protocol):

- Seed 42, stratified 70/15/15 train/val/test
- `c1=100`, `c2=1e-2`, purity 0.95, min_ball_count 2
- Reference prediction rule, opposite-only weight placement
- No hyperparameter tuning on validation or test

Primary comparisons:

- `CW+OBRU` vs `CW`
- `CW+R+OBRU` vs `CR`

## 10. Known limitations

- Proxy overlap may overstate or understate true ball interaction depending on
  purity splits and mean-distance radii.
- Top-3 aggregation is a heuristic; sensitivity to *k* was not explored here.
- Class-conditional up/down weighting assumes minority balls near opposite-class
  mass are informative; this may not hold on all datasets.
- OBRU does not modify granulation, prediction, or frozen baseline/CW/R/CR math.
