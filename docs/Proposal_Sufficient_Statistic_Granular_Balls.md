# Research Proposal — Sufficient-Statistic Granular Balls for Least-Squares Twin SVMs

*Working draft for the literature review. Citations marked **[verify]** are
recalled from memory and must be checked before use.*

---

## 1. One-paragraph summary

Granular-ball (GB) classifiers replace training samples with balls described
by a centre and a radius. The published GB least-squares twin SVM (GBLSTSVM)
uses only these two quantities. We observe that the least-squares twin SVM
(LSTSVM) objective depends on the data only through second-moment sums. So a
ball that stores its **count, mean and scatter matrix** reproduces the
full-data LSTSVM solution **exactly**, whatever the granulation. This proposal:

- (i) states and proves that exactness result for the GB setting;
- (ii) measures how much accuracy the centre-plus-radius summary loses, using an
  ordered "information ladder" of ball summaries;
- (iii) identifies what granulation still contributes once compression is
  lossless: computational cost, per-ball robustness weighting and
  streaming/federated aggregation;
- (iv) delimits where exactness fails: hinge-loss SVMs and kernels without a
  finite feature map.

---

## 2. Problem statement

GB learning is motivated by efficiency and robustness: fewer training units,
less sensitivity to label noise. Its cost is information loss, because
replacing many samples with a centre and radius discards within-ball
structure. The literature treats that loss as an unavoidable trade-off, which
generator design (purity, density, boundary awareness, adaptivity) tries to
manage.

For the least-squares family this trade-off is not fundamental. It is caused
by the choice of ball summary, not by granulation itself. No GB work we know
of states this. Whether that is true is the main question for the
literature review (see §8).

---

## 3. Research questions

- **RQ1 (Exactness).** For LSTSVM-type objectives, is a ball summary of
  (count, mean, scatter) sufficient to recover the sample-level solution
  exactly? Under which regularisation and weighting schemes does this still hold?
- **RQ2 (Cost of lossy summaries).** How much classification performance does
  each coarser ball summary lose relative to the exact one, across datasets
  with different geometry? The summaries, from coarsest to exact, are:
  - centre only;
  - centre and radius (published GBLSTSVM);
  - centre and diagonal scatter;
  - centre and full scatter.
- **RQ3 (Role of granulation after exactness).** Once compression is lossless,
  what does the choice of granulation still affect?
  - (a) training cost, which grows with ball count × d²;
  - (b) robustness through per-ball reweighting of noise or imbalance;
  - (c) incremental, streaming or federated training through mergeable
    statistics.
- **RQ4 (Boundary of the result).** How far from exact are GB versions of
  hinge-loss learners (SVM, TSVM, GBSVM, GBTSVM)? Does the size of that gap
  predict where boundary-aware generators (the earlier Proposals 1–2) actually
  help?
- **RQ5 (Kernels).** With explicit finite feature maps (Nyström, random
  Fourier features), do the exactness results carry over to kernel LSTSVM?
  What does the approximation itself cost?

---

## 4. Technical core

### 4.1 Exactness

The LSTSVM (Kumar & Gopal, 2009 **[verify]**) first plane is

$$
\min_{w_1,b_1}\ \tfrac12\|Aw_1+e_1b_1\|^2+\tfrac{c_1}{2}\|Bw_1+e_2b_1+e_2\|^2 .
$$

With $H=[A\ e_1]$ and $G=[B\ e_2]$, the solution is

$$
[w_1;b_1]=-\big(G^\top G+\tfrac1{c_1}H^\top H\big)^{-1}G^\top e_2 .
$$

Now partition the rows of $A$ into balls $k$, each with count $n_k$, mean
$\mu_k$ and scatter $S_k=\sum_{i\in k}(x_i-\mu_k)(x_i-\mu_k)^\top$. Then

$$
H^\top H=\sum_k\begin{bmatrix}S_k+n_k\mu_k\mu_k^\top & n_k\mu_k\\ n_k\mu_k^\top & n_k\end{bmatrix},
\qquad
G^\top e_2=\sum_k\begin{bmatrix}n_k\mu_k\\ n_k\end{bmatrix}.
$$

Hence $(n_k,\mu_k,S_k)$ are sufficient statistics, and the solution does not
depend on how the balls were generated. The second plane follows by symmetry.

**Repo evidence:** `tests/test_ladder.py` asserts that rung L3 reproduces raw
LSTSVM $[w;b]$ to $10^{-8}$. Re-run it before citing it.

### 4.2 What GBLSTSVM does instead

GBLSTSVM (Quadir & Tanveer, 2024 **[verify]**) treats each centre as one
sample, with weight 1, and adds the radius to the right-hand side of the
second loss term. Compared with §4.1 it:

- drops $S_k$ entirely;
- drops the count weighting $n_k$;
- injects $r_k$ as a margin shift.

Each of these changes is a separately testable step in the ladder (§5.2).

### 4.3 Where it breaks

- **Hinge loss:** max(0, ·) does not break down into per-ball sums, so balls
  can only bound the hinge loss (for example GBSVM's $\|w\|r$ term), not
  recover it exactly.
- **Weighting:** sample-level weights within a ball are fine if the ball
  stores weighted statistics. Per-ball weights are exact for a reweighted
  objective.
- **Kernels:** exact only when an explicit feature map exists. Kernel matrices
  over samples have no ball-level summary.

---

## 5. Planned contributions and experiments

1. **Proposition and proof** (§4.1), plus the weighted and finite-feature-map
   extensions.
2. **Information ladder.**
   - **Rungs:** L0 centre only; L1 centre and count; L2 centre and radius
     (GBLSTSVM); L2d diagonal scatter; L3 full scatter (exact).
   - **Data:** the existing benchmark CSVs plus synthetic geometries
     (elongated, variable-density, overlap).
   - **Metrics:** balanced accuracy, macro-F1, distance of $[w;b]$ from the
     exact solution, and training time.
3. **Cost model.** Training cost is O(K·d²) for L3 versus O(K·d) for L2,
   where K is the number of balls. Find the crossover point in d and K at
   which the exact model is still cheaper than training on the raw samples
   (O(N·d²)).
4. **Robust reweighting.** Use exact statistics plus per-ball weights based on
   purity or enemy ratio. Test under 0/10/20% label noise and class
   imbalance, against sample-level weighted LSTSVM.
5. **Streaming/federated demonstration.** Merge statistics across data
   shards; show the result equals a single-machine fit.
6. **Hinge-loss contrast.** Measure the gap to the sample-level model for
   GBSVM/GBTSVM. Relate it to the gains of boundary-aware generators, reusing
   the Proposal 1–2 code.

---

## 6. Expected claims (and claims we will *not* make)

**Will claim:**
- For squared-loss twin SVMs, the accuracy loss attributed to granulation is
  caused by the choice of ball summary.
- Storing scatter removes that loss at a known extra cost.
- The information ladder quantifies how much the published GBLSTSVM summary
  loses.

**Will not claim:**
- That sufficient statistics for least squares are new. That is textbook, and
  also in PSVM, BIRCH and federated ridge regression (§7).
- That the approach helps hinge-loss learners.
- New granulation algorithms.

---

## 7. Literature review map

### 7.1 Areas to cover

| # | Area | Why it matters | Key seeds **[verify all]** |
|---|------|----------------|----------------------------|
| A | GB computing foundations | Defines the ball representation we modify | Xia et al. 2019 (GB classifiers, *Information Sciences*); Xia et al. survey(s) on GB computing |
| B | GB + SVM family | Direct baselines; check whether any already store scatter | GBSVM (Xia et al. 2022/23, *IEEE TNNLS*); GBTSVM (Quadir, Tanveer et al. 2024); GBLSTSVM / GB-LSTSVM (Quadir & Tanveer 2024); GB-based kernel/fuzzy/universum TSVM variants |
| C | GB generators | To position "generator design is the wrong lever" | GBG / GBG++; adaptive, density-based, boundary-aware, ellipsoidal GB (ellipsoid = second-moment shape, closest prior art) |
| D | Twin SVM family | Objective structure | Jayadeva et al. 2007 (TWSVM); Kumar & Gopal 2009 (LSTSVM); Tanveer et al. TSVM review (2022) |
| E | Proximal / least-squares SVM | **Strongest prior art for exactness** | Fung & Mangasarian 2001 (PSVM); Fung & Mangasarian 2002 (incremental PSVM via $E^\top E$); Suykens & Vandewalle 1999 (LS-SVM) |
| F | Data summarisation | Same statistics, other fields | BIRCH clustering feature (N, LS, SS), Zhang et al. 1996; data squashing (DuMouchel et al. 1999); coresets for regression/SVM (Feldman, Bachem et al.) |
| G | Federated / distributed least squares | Mergeable statistics | One-shot federated ridge/linear regression via aggregated $X^\top X$; secure aggregation |
| H | Random features / Nyström | RQ5 kernel extension | Rahimi & Recht 2007; Williams & Seeger 2001 |
| I | Robust and imbalanced TSVM | RQ3(b) baselines | Fuzzy TSVM, intuitionistic fuzzy TSVM, weighted LSTSVM |

### 7.2 Search strings

```
"granular ball" AND ("support vector" OR "twin support vector" OR "least squares")
"granular ball" AND (covariance OR scatter OR ellipsoid* OR "second-order" OR moment)
"granular ball" AND ("information loss" OR "lossless" OR "sufficient statistic*")
"proximal support vector" AND (incremental OR "sufficient statistic*" OR distributed)
"least squares twin support vector" AND (incremental OR online OR distributed OR federated)
"clustering feature" AND classification AND (SVM OR "least squares")
"data squashing" OR "coreset" AND ("least squares" OR "support vector")
"twin support vector" AND ("random features" OR Nystrom)
```

Databases: IEEE Xplore, ScienceDirect (Information Sciences, Pattern
Recognition, Neural Networks, KBS), SpringerLink, ACM DL, arXiv, Google
Scholar forward citations from the GBLSTSVM and GBSVM papers.

### 7.3 Inclusion / exclusion

- **Include:** GB methods with any SVM-family learner; any method that
  summarises data blocks by moments for a squared-loss classifier; GB
  generators published from 2019 onward.
- **Exclude:** GB for clustering or rough sets only, unless the
  representation is reusable; deep-learning GB variants, unless they use
  moment summaries.

### 7.4 Extraction table (fill per paper)

| Field | Notes |
|-------|-------|
| Learner and loss | squared / hinge / other |
| Ball summary stored | centre, radius, count, scatter, label distribution |
| Exact or approximate w.r.t. sample-level objective | stated? proved? |
| Generator | purity / density / boundary / adaptive |
| Kernel handling | none / kernel on centres / feature map |
| Datasets, metrics, ball counts reported | |
| Robustness claims (noise, imbalance) | |

---

## 8. Novelty risks — what would weaken the proposal

Check each explicitly during the review.

1. **A GB paper already stores the covariance/scatter matrix and notes
   exactness.** Highest risk; look hardest at ellipsoidal GB and GB-LSTSVM
   follow-ups. *If found:* the contribution shrinks to the information ladder
   and the hinge-versus-squared-loss boundary study.
2. **The PSVM incremental papers already cover block/cluster summaries.**
   Likely partially true. Position this work as transferring that to GB and
   showing that the GB field's summary is lossy, not as a new mathematical
   result.
3. **BIRCH-style CF-based classifiers already exist with LSTSVM.** Check area F
   combined with D.
4. **The empirical gap L2→L3 is negligible on benchmarks.** Then the story
   becomes "the published summary is already near-sufficient, here is why",
   which is still a reportable result but weaker.

---

## 9. Positioning statement (draft)

> Prior granular-ball work treats information loss as the price of
> compression and improves generators to manage it. We show that for
> least-squares twin SVMs this loss comes from the ball summary, not from
> granulation: count-mean-scatter balls recover the sample-level solution
> exactly, as known for proximal SVMs, while centre-radius balls do not.
> We quantify the gap, identify what granulation still contributes
> (cost, robust reweighting, mergeability), and delimit the result to
> squared-loss learners.

---

## 10. Links to existing repo work

- `src/ladder_gblstsvm.py`, `tests/test_ladder.py`, `tests/test_ladder_k2.py`:
  exactness gate and ladder solver.
- `experiments/estimator_information_ladder_common.py`,
  `experiments/output/estimator_information_ladder/`: existing ladder results;
  review them before rerunning.
- `src/svm_aware_granulation.py`, `src/margin_multigranularity.py`: Proposal
  1–2 code, reusable for RQ4.
