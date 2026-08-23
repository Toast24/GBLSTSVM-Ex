# GBLSTSVM Fuzzy Membership — Research Design Notes

## 1. What we've learned from failures

- **CW works** because class imbalance is a *global label-frequency* problem — reweighting by class prevalence is the right shape of fix for that specific problem.
- **R fails to add value** because purity/radius are *intra-ball* compactness measures, and CW already captures most of the exploitable signal at the ball level (imbalance dominates; local compactness is nearly orthogonal-but-redundant once class counts are handled).
- **BMRU fails** because `2r/|s|` is a *self-referential* geometric ratio — it says nothing about where other classes' mass actually sits. A ball can have huge radius relative to its slack/margin purely from GB construction density, with zero actual opposing-class presence nearby. Radius is a poor proxy for "am I near a decision boundary."
- **OBRU fails** because literal ball-to-ball intersection is a near-empty event under our granulation (balls are constructed to be pure/separated), so *overlap* as geometry basically never fires. The right notion of "boundary-ness" must be defined via **relative position among ball centers/populations across the whole local neighborhood**, not via literal geometric collision of two ball volumes.

Common thread: signals derived from a ball's own geometry (its own radius, its own purity, its own volume vs one nearest opponent) are too local/self-referential to detect what actually matters — the ball's position within the cross-class competitive landscape. The untried direction is a signal that is inherently *relational and graph/neighborhood-based across many balls*, not defined from a single ball's intrinsic geometry or a single pairwise overlap test.

## 2. What information is still missing from GBLSTSVM

Currently exploited: label counts (CW), intra-ball purity/radius (R), single-ball margin/radius self-ratio (BMRU), pairwise ball-volume intersection (OBRU).

Never exploited:
- The rank-ordered structure of a ball's neighbors by class (k-NN among ball centers, not points) — soft local class mixture rather than a hard-intersection test.
- Directionality — is the opposing class on one side (a real frontier) or scattered/absent (interior)?
- Consistency across scales — does a ball's neighborhood classification agree at k=3 vs 10 vs 20? (stability signal, unused)
- Anything from the twin-hyperplane structure itself post/during optimization (BMRU/OBRU are pre-hoc geometric heuristics, not model-informed).
- Graph connectivity between balls of the same class — is a minority ball part of a connected chain of same-class balls (a "region") or an isolated singleton in majority territory?

That last distinction — region vs. singleton — is exactly what separates "genuine minority boundary structure" from "isolated minority noise," which BMRU/OBRU/R all failed to distinguish.

## 3. Candidate fuzzy mechanisms (shortlist)

- **C1 — k-NN Class-Competition Membership (soft local dominance).** k nearest ball centers overall, distance-weighted fraction opposite-class. Replaces "does my volume intersect yours" (OBRU, almost always false) with "how much opposing mass surrounds me in rank-order" (always defined, continuous).
- **C2 — Same-Class Connectivity / Isolation Index.** Same-class k-NN graph over centers; membership from local degree/component support within own class. Targets "genuine minority boundary structure vs isolated minority noise" — a criterion no existing mechanism encodes.
- **C3 — Directional Frontier Signal (asymmetric competition).** Whether the direction to nearest opposite-class mass is consistent (real frontier) vs scattered (noise), via circular variance of nearest-opponent direction vectors.
- **C4 — Multi-Scale Neighborhood Consistency.** Competition fraction at multiple k (3,7,15); membership = agreement (low variance across k) × competition level.
- **C5 — Twin-Hyperplane Disagreement / Distance-to-Both-Planes.** Model-in-the-loop signal: distance of each ball to both twin planes, not just pre-hoc geometry.
- **C6 — Perturbation Stability of Local Class Assignment.** Bootstrap/jitter ball centers or resampling; check if local-neighborhood majority-vote classification flips. Expensive but principled.
- **C7 — Prototype Agreement.** Agreement between a ball's local direction and its class's global centroid direction, combined with C1 to separate "typical minority boundary" from "atypical minority noise."
- **C8 — Hybrid: Competition × Connectivity.** μ_i = competition_i × connectivity_i — only balls that are both contested by the opposite class AND structurally supported by their own class get high importance. Explicitly blocks "isolated minority outlier surrounded by majority" from being rewarded (the failure mode naive inverse-density weighting falls into).

## 4. Ranked comparison

| Candidate | Novelty | Complement to CW | GB-specificity | Likely active on our data | Outlier resistance | Math defensibility | Feasibility | Pub. potential |
|---|---|---|---|---|---|---|---|---|
| C8 (Competition×Connectivity) | High | High | High | High | High | High | Med | High |
| C2 (Connectivity/Isolation) | High | Med-High | High | High | High | High | High | Med-High |
| C1 (k-NN competition) | Med | Med | High | High | Med | High | High | Med |
| C4 (Multi-scale consistency) | Med-High | Med | Med | Med-High | High | Med | Med | Med |
| C3 (Directional frontier) | High | Med | Med | Med | Med | Med | Med | Med |
| C5 (Twin-plane disagreement) | Med (known in fuzzy-SVM lit) | Low-Med (may duplicate margin info) | Low-Med | High | Low (planes chase outliers) | Med | Low-Med (iterative) | Med |
| C7 (Prototype agreement) | Med | Med | Med | Med | Med | Med | High | Low-Med |
| C6 (Perturbation stability) | High | Med | Med | Unknown (expensive to test) | High | Med | Low | Med |

**Reasoning:** OBRU's core weakness was literal *intersection* being sparse — the underlying "local class competition" idea wasn't wrong, only the operationalization was. Reformulating as rank-based k-NN over centers (always defined, continuous, no threshold-emptiness problem) directly fixes OBRU's mechanistic failure while keeping the intuition. C2 supplies the missing "is this real structure or an outlier" filter that inverse-density weighting cannot supply safely. Their product (C8) is the most defensible new fuzzy membership concept — it uses ball-center graph topology across classes, which none of the four failed/frozen mechanisms use at all.

## 5. Top candidates — precise mathematics

Balls $\{B_i\}_{i=1}^N$, centers $c_i \in \mathbb{R}^d$, radii $r_i$, labels $y_i \in \{+1,-1\}$, size $n_i$.

### Candidate A: C1 — Soft k-NN Class-Competition Membership

Let $\mathcal{N}_k(i)$ = k nearest ball centers to $c_i$ (excluding $i$), any class, by $d_{ij}=\|c_i-c_j\|$.

$$
\text{comp}_i = \frac{\sum_{j \in \mathcal{N}_k(i)} \mathbb{1}[y_j \neq y_i]\, w_{ij}}{\sum_{j \in \mathcal{N}_k(i)} w_{ij}}, \qquad w_{ij} = \exp\!\left(-\frac{d_{ij}^2}{2\sigma_i^2}\right)
$$

$\sigma_i$ = median distance to the $k$ neighbors of $i$ (per-ball adaptive bandwidth, avoids a single global scale hyperparameter).

$$
\mu^{A}_i = 4\,\text{comp}_i\,(1-\text{comp}_i)
$$

Parabola peaking at comp=0.5 (true frontier), vanishing at both extremes (pure interior OR ball deep in enemy mass, likely mislabel/noise).

- Uses rank-neighborhood class labels + distances among ball centers (never used before).
- vs CW: CW is a scalar per class from global counts; this is per-ball, purely local/geometric.
- vs R: R uses a ball's own point purity/radius; this uses inter-ball neighbor labels.
- vs BMRU: BMRU is self-referential (radius vs own slack); this never touches radius or margin.
- vs OBRU: OBRU needs literal ball-volume intersection (binary, mostly 0); this is continuous, always well-defined.
- Active on diabetes/vehicle0/ecoli2 by construction (doesn't require overlap).
- Failure mode: if classes are globally well-separated in center-space, comp_i ≈ 0 everywhere → inert (check empirically first).
- Outliers: a mislabeled ball deep in the opposite class has comp_i→1, mapped back toward 0 by the parabola — not rewarded.
- Complexity: O(N log N) via k-d tree.
- Under severe imbalance: minority balls will trivially have more opposite-class neighbors by base rate alone — risk of just re-deriving class frequency (duplicating CW); must correct via combination with connectivity (Candidate C).
- Novelty risk: conceptually related to established "local class ratio" fuzzy membership in fuzzy-SVM literature (e.g., FSVM-CIL) — a known idea adapted to granular balls, not fully novel alone. Novelty here is the ball-adaptive bandwidth, parabolic remap, and combination with Candidate B.
- Ablation: compare against a label-permutation null (same class proportions) — if comp distribution ≈ null, discard.
- Falsifies if: comp_i distribution under real labels is statistically indistinguishable (KS test) from the permutation null, or CW+μ^A shows no significant gain over CW alone in CV.

### Candidate B: C2 — Same-Class Connectivity / Structural Support Index

Same-class k-NN graph $G_y$ per class over ball centers.

$$
\text{supp}_i = \frac{1}{Z_i}\sum_{j \in \mathcal{N}_k^{(y_i)}(i)} \exp\!\left(-\frac{d_{ij}^2}{2\sigma_i^2}\right), \qquad Z_i = \max_{l:y_l=y_i}\Big[\sum_{j\in\mathcal{N}_k^{(y_i)}(l)} \exp(-d_{lj}^2/2\sigma_l^2)\Big]
$$

Normalize by the maximum such sum *within the same class* so supp_i ∈ (0,1] regardless of overall class density (avoids re-encoding raw class frequency, which would duplicate CW). This is a relative, within-class notion of support.

$$
\mu^{B}_i = \text{supp}_i
$$

- Uses same-class neighbor graph density, normalized per-class — never used by CW/R/BMRU/OBRU.
- vs CW: CW is a single scalar per class; this varies ball-to-ball within a class.
- vs R: R is intra-ball point-level compactness; this is inter-ball graph structure across many same-class balls.
- vs BMRU/OBRU: no radius/margin ratio, no opposite-class overlap anywhere.
- Failure mode: normalizing by max could let one abnormally dense region dominate the scale — consider a robust normalizer (e.g. median-of-top-quartile) instead of max.
- Outliers: an isolated point (no nearby same-class neighbors) gets low supp_i — explicitly down-weighted, directly solving the "density weighting upweights minority noise" problem instead of reproducing it.
- Complexity: O(N log N) per class.
- Under severe imbalance: minority class has fewer balls total, so k should be capped by available same-class population, e.g. $k=\min(k_0, n_{\text{class}}-1)$.
- Novelty risk: graph/manifold-based sample weighting exists generally (e.g. graph-based SSL confidence), but a same-class connectivity index over granular-ball centers, normalized within-class, as an SVM ball weight is not present in the frozen baselines or failed mechanisms — moderate-to-genuine novelty in the GBLSTSVM context.
- Ablation: CW vs CW+μ^B alone, isolating whether structural support by itself adds anything beyond CW.
- Falsifies if: μ^B correlates ≥0.9 with CW (same failure mode as R vs CW), or down-weighting low-support minority balls hurts minority recall more than it helps (real boundary minority balls wrongly penalized as "isolated").

### Candidate C (primary/recommended): C8 — Competition-Connectivity Product Membership

$$
\mu^{C}_i = (\mu^{A}_i)^{\alpha}\cdot(\mu^{B}_i)^{\beta}, \qquad \alpha=\beta=1 \text{ fixed a priori}
$$

$k$ and per-ball bandwidth $\sigma_i$ fixed once before results are seen, e.g. $k=\max(5,\lceil 0.05N\rceil)$, chosen structurally rather than fit per dataset.

Final weight, drop-in replacement for the R factor in CR:
$$
w_i = \text{CW}(y_i)\cdot \mu^{C}_i
$$

- A ball deserves high importance only if it is BOTH genuinely contested (real frontier, Candidate A) AND structurally supported by its own class (real region, not noise, Candidate B). This conjunctive structure blocks BMRU's failure (looks contested but ignores own-class support) and naive density-weighting's failure (support alone would reward isolated same-class islands or deep-interior balls that aren't informative for the boundary).
- Failure mode: if μ^A and μ^B are strongly correlated in practice, the product collapses to redundant information — check correlation empirically before any classifier run.
- Outliers: an isolated outlier has low μ^B regardless of μ^A, suppressing the product — explicit fix for minority-noise-upweighting risk.
- Complexity: O(N log N), negligible relative to QPP solves.
- Under severe imbalance: μ^B's within-class normalization means minority balls are compared only against other minority balls, so imbalance doesn't mechanically crush μ^B for them.
- Novelty risk: individual ingredients have general ML precedent, but the specific formulation over granular-ball centers with within-class normalization, combined multiplicatively as a GBLSTSVM ball-weight complementary to CW, is not present in the frozen baselines or general fuzzy-SVM literature to my knowledge — should be checked against FSVM/fuzzy-rough-SVM literature before claiming novelty in the paper (cite and differentiate, don't just assert).
- Ablation: four-arm comparison — CW alone; CW×μ^A only; CW×μ^B only; CW×μ^C (full product) — on all three datasets, to attribute gain to competition, connectivity, or their interaction.
- Falsifies if: CW×μ^C does not beat CW alone (G-mean/minority-F1/AUC, proper CV) by a margin larger than run-to-run variance; or μ^A/μ^B correlate >~0.8 (redundant); or synthetically injected isolated-noise balls get non-trivially high μ^C in a controlled sanity check.

## 6. Recommended primary candidate

Candidate C (C8: Competition × Connectivity product membership), with Candidates A and B run as mandatory ablation arms rather than optional extras — the paper's core claim should be about the *interaction*, and the ablation proves it's not just A or B alone doing the work (which would reduce novelty to "another local-ratio fuzzy SVM").

## 7. Minimal implementation/ablation plan

1. Diagnostic pass (no classifier yet): on diabetes/vehicle0/ecoli2, compute μ^A, μ^B for every ball; report marginal distributions, correlation with each other, and correlation with CW's per-ball weight and with R's purity/radius. Reject/redesign before any classifier training if: (a) μ^A ≈ label-permutation null, (b) μ^B correlates ≥0.9 with CW, or (c) μ^A/μ^B correlate ≥0.8 with each other.
2. Fix k, bandwidth rule, α=β=1 before step 3 — no tuning to the three datasets.
3. Train/evaluate 5 arms with identical CV protocol: GBLSTSVM, CW, CW×μ^A, CW×μ^B, CW×μ^C. Metrics: G-mean, minority-class F1, AUC — not just accuracy.
4. Only after this fixed-protocol run, analyze (not re-tune) why it helped or failed — e.g. which balls changed weight most and whether those are minority boundary balls or noise, via the synthetic-outlier-injection sanity check.

## 8. Explicit falsification criteria (summary)

- μ^A or μ^B statistically indistinguishable from label-permutation baseline → signal is not real, discard.
- μ^B correlates ≥0.9 with CW, or μ^A/μ^B correlate ≥0.8 with each other → redundant, not a genuine bivariate mechanism.
- CW×μ^C shows no CV-significant improvement over CW alone on minority-F1/G-mean across the three target datasets → mechanism rejected, no post-hoc reformulation permitted without a new independent justification.
- Synthetic isolated-noise injection test shows μ^C assigns high weight to injected noise balls → mechanism is unsafe, reject regardless of accuracy gains.

## Research discipline notes

- Do not optimize the formula to the three datasets after seeing results.
- Distinguish clearly between genuinely new signal, recombination of existing signals, a known fuzzy-SVM idea adapted to granular balls, and a genuinely novel granular-ball-specific mechanism. Component A leans toward "known idea adapted"; Component B leans toward "genuine novelty in this context"; Component C (the product) is the actual novel contribution claim.
- See the companion implementation prompt: [CBCM_implementation_prompt.md](CBCM_implementation_prompt.md) for exact, unambiguous coding instructions derived from this analysis.
