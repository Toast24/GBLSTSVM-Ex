# Novelty assessment: topology-supported competition membership

**Date:** 2026-08-22
**Proposed mechanism:** \(\mu^{\mathrm{Topo}}_i = A_i \cdot R_{\mathrm{opp}}(i)\), where \(A_i = 2\min(\mathrm{comp}_i, 1-\mathrm{comp}_i)\) is local class ambiguity on a ball-center kNN graph, and \(R_{\mathrm{opp}}(i)\) is the fraction of opposite-class kNN neighbors that also list \(i\) in their kNN set.
**Verdict:** **sufficiently distinct to implement** (novelty: **moderate**, not redundant). This is not a stop.

## Questions

### 1. Is “ambiguity × reciprocal neighborhood support” already a known fuzzy membership?

**No close match found.** Standard FSVM memberships are:

- Distance to class centroid (Lin & Wang FSVM and many LSTSVM/FTSVM variants).
- Local class ratio / fuzzy kNN membership (Keller FKNN; Heo & Gader FSVM; fuzzy k-NN SVM).
- Entropy of neighborhood labels (IEFSVM).
- Relative / inverse kNN density (intuitionistic-fuzzy SVM for imbalance).
- “Environmental” membership from similar-vs-heterogeneous counts in a radius (motor-fault FSVM).

These use **one-way** neighbor class counts or density. None multiply a peak-at-0.5 ambiguity remap by **mutual / reciprocal opposite-class kNN fraction**.

Closest conceptual cousins:

| Method | What it uses | Difference |
|--------|----------------|------------|
| Fuzzy kNN / FSVM-kNN | Distance-weighted class votes among k neighbors | No reciprocity; membership is class confidence, not “supported boundary” |
| IEFSVM | Neighborhood label entropy | Entropy ≈ ambiguity; no graph reciprocity |
| Environmental FSVM | Same/hetero counts and mean distances in a radius | Geometric counts, not mutual kNN |
| This repo’s CA | \(\mathrm{comp}_i\) (Gaussian-weighted opposite fraction) | One-way competition only |
| This repo’s μB / SP | Same-class connectivity / support | Own-class density, not cross-class reciprocity |
| This repo’s IPA | Ambiguity / (1 + isolation) | Same-class nearest-neighbor **distance**, not reciprocal opposite edges |

### 2. Has reciprocal kNN already been used to define SVM sample membership / weights?

**Not in the literature reviewed here.** Reciprocal / mutual kNN appears as:

- A **classifier** (M-kNN): drop non-mutual neighbors, then majority vote.
- A **graph** for clustering (mutual-kNN density peaks) and manifold methods (UMAP mutual kNN).
- Noise filtering of a training set before a separate learner.

No FSVM / twin-SVM / LSTSVM paper was found that sets slack weights to a reciprocal-kNN statistic. Graph-regularized SVM (e.g. FSVM-CIP nearest within-class neighbor graph) uses Laplacian geometry, not reciprocal opposite-class membership.

### 3. Does a granular-ball SVM using ball-center graph topology already exist?

**Partially, and not this topology.**

- GBSVM / GBTWSVM / GBFTSVM replace points by balls; GBFTSVM membership is typically **Pythagorean fuzzy membership from ball-center to class-center distance**, not a kNN graph over balls.
- GBkNN uses balls as adaptive neighborhoods for **classification**, not as SVM weights.
- **This repository already introduced ball-center kNN topology** in CBCM (competition + same-class connectivity). That is prior art **within this project**, not in published GB-SVM membership.

The proposed \(R_{\mathrm{opp}}\) is a **different graph statistic** on that same ball-center kNN object: cross-class **edge reciprocity**, not same-class degree/density and not one-way competition.

### 4. What is genuinely different?

1. **Information source:** directed kNN edge symmetry between opposite-class ball centers.
2. **Hypothesis:** one-way “I see the other class” can be noise or a hanging outlier; a **mutual** opposite-class link is evidence of a two-sided frontier.
3. **Not** density, radius, margin, volume overlap, purity, class-frequency, or same-class isolation (IPA) / connectivity (μB).
4. **Product with frozen A2** tests whether reciprocity filters CA’s isolated-noise failure without IPA’s same-class distance penalty.

What is **not** claimed as new: kNN competition / local class mixture (known FSVM idea; CA already instantiated it on balls); reciprocal kNN as a general graph notion (known in kNN classification and clustering).

### 5. Closest methods and exact differences

| Closest method | Shared idea | Exact difference |
|----------------|-------------|------------------|
| CA / A2 (this repo) | Local opposite-class kNN fraction, peak at 0.5 | Topo multiplies by reciprocity; CA does not |
| CBCM μB / μC | Ball-center kNN graph as membership | μB is **same-class** support; \(R_{\mathrm{opp}}\) is **opposite-class mutual** support |
| IPA | Keep A2, downweight unsupported balls | IPA uses **distance to nearest same-class** center; Topo uses **reciprocal opposite-class kNN** |
| M-kNN | Reciprocal neighbors are more reliable | M-kNN is a classifier / filter, not an SVM/LSTSVM membership |
| GBFTSVM | Fuzzy membership on granular balls | Class-center geometry, not ball-center graph reciprocity |
| FSVM-kNN / IEFSVM | Neighborhood labels → membership | One-way mixture or entropy; no mutual opposite-class edges |

## Decision

The combination **A2 ambiguity × opposite-class reciprocal kNN** on **granular-ball centers** as **GBLSTSVM weights** is not documented as an existing membership in FSVM / fuzzy twin SVM / GB-SVM literature, and it is not a retuning of frozen or failed mechanisms in this repo.

Novelty is **moderate**: recombination of known ingredients (kNN ambiguity + reciprocal kNN) in a GB-LSTSVM weight, with a distinct operationalization from μB and IPA.

**Proceed to implementation.** Do not claim “first use of kNN membership” or “first GB fuzzy SVM.” Claim, at most, a specific reciprocal-boundary membership on ball-center graphs, pending empirical support.
