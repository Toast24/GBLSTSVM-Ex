# Session 4 — Granulation Diagnostics and Split Utility

**Date:** 27 Aug 2026  
**Scope:** Read-only granulation replays across 17 datasets × seeds `[42, 0, 1]` (51 runs). No changes to `src/`, granulation, `del_ball`, CW, CA/reliability, or Topo unless explicitly noted.

---

## 1. Research objective

We are trying to extend GBLSTSVM into a **research-grade contribution** — not another ad hoc ball-weight function.

The working hypothesis for the last several weeks has been: GBLSTSVM underperforms on minority-sensitive metrics because **something structural in granular-ball generation or ball weighting** degrades minority representation before the LSTSVM ever trains. Today's work systematically **screened candidate mechanisms** (singleton geometry, nested conflicts, Gaussian membership/overlap, minority isolation/repair, Topo harm signals, split quality) **before touching production code**.

A contribution counts as research-grade here only if it:

1. Identifies a **reproducible failure mode** with measurable structure,
2. Proposes an intervention tied to that structure (not post-hoc weight tuning),
3. Can be falsified with pre-specified metrics independent of the intervention.

---

## 2. Starting point / baseline

### Pipeline under study

| Stage | Frozen behavior (today's replays) |
|---|---|
| **Granulation** | Recursive split while `purity < 0.95` and `num > 1`; each split = `GranularBall.split_2balls()` → sklearn `KMeans(n_clusters=2)` |
| **Ball stats** | Label = majority vote; purity = majority fraction; radius = mean distance to center |
| **del_ball** | Keeps balls with `num >= 2`; purity threshold inactive at ρ=0.95 in practice |
| **Training matrix** | Surviving balls → CW × CA (reliability) × optional Topo membership → GBLSTSVM |

### What is frozen vs. allowed to change

| Frozen today | Allowed to propose (not yet implemented) |
|---|---|
| KMeans(k=2), ρ=0.95, min_samples=1, del_ball n<2 | Virtual/counterfactual granulation gates only |
| CW, CA, Topo formulas, GBLSTSVM solver | New **split-acceptance** logic upstream of del_ball |
| `src/` implementations | Nothing in classifier until granulation hypothesis survives Stage 4+ |

### Baseline failure question

> Where is GBLSTSVM actually losing performance, and what structural property of the data/granulation causes those losses?

Today's answer is **partial and layered**:

- **Established:** granulation produces massive singleton/small-ball deletion, asymmetric minority vs majority survival, and late-stage small-parent fragmentation.
- **Not established:** that any single downstream fix (Topo weighting, Gaussian membership, parent-local singleton repair) recovers classifier performance.
- **Newest signal:** split-time geometry (`center_radius_separation`) separates harmful retained-vs-deleted splits with AUROC ≥ 0.80 — pointing **upstream** to granulation, not membership reweighting.

---

## 3. Hypotheses investigated TODAY

| Hypothesis / Mechanism | Diagnostic performed | Main result | Verdict |
|---|---|---|---|
| **hd1/hd2 singleton outlier rule** | `analyze_hd2_singleton_diagnostic.py` (9 runs, 3 datasets; **no del_ball**) | 896 singleton children; 676/896 (75.4%) **UNDEFINED_HD2**; 484/896 (54.0%) from **2-sample parents**; outlier vs legitimate split weak | **Rejected** as operational gate |
| **Nested-ball heterogeneous conflict** | `analyze_nested_conflict_diagnostic.py` (9 runs) | **9** heterogeneous nested pairs total; **100%** involve singleton (n=1, r=0) inside pure ball; **0** conflicts in all 4 known Topo-failure runs (ecoli2×3, vehicle0_s1) | **Rejected** as Topo failure driver |
| **Gaussian membership (B0/G1/G2)** | `gaussian_membership_screen.py` (9 classifier runs) | mean \|corr(μ_G, μ_A)\| ≈ **0.051** (independent of CA); G1 beats B0 on ecoli2 but **hurts** diabetes & vehicle0 on gmean/minority_f1 | **Rejected** as general CA replacement |
| **Gaussian overlap / BC_overlap split signal** | `analyze_gaussian_overlap_diagnostic.py` (51 runs) | BC_overlap AUROC adversarial vs informative = **0.330**; hd1_norm = **0.747**; kmeans purity gain negatively correlates with Δ Macro F1 (r = **−0.303**) | **Rejected** as primary split mechanism |
| **Minority isolation chain** | `analyze_minority_isolation_diagnostic.py` (51 runs, 13,654 splits) | Isolation → del_ball loss chain **reproducible**; corr(isolating-split rate, Δ Macro F1) = **0.050**; adversarial vs informative isolation rates **0.262 vs 0.272** | **Partially confirmed** mechanistically; **rejected** as classifier predictor |
| **Minority repair replay (B1)** | `analyze_minority_repair_replay.py` (51 runs) | Minority survival **0.583 → 0.773** but minority-to-opposite support **47.4 → 38.2**, opp/same dist ratio **1.70 → 1.18** | **Rejected** as principled fix |
| **Singleton isolation (parent-local A/B/C)** | `analyze_singleton_isolation_diagnostic.py` (51 runs, **9,174** singleton-child events) | **100%** deleted by del_ball; max parent-local AUROC (non-definitional) ≈ **0.71**; lone-in-parent globally-isolated AUROC ≈ **0.58** | **Rejected** — no parameter-free parent-local rule |
| **Split-quality diagnostic** | `analyze_split_quality_diagnostic.py` (51 runs, 13,654 splits) | Best AUROC(**A**) = minority_fraction_small_child **0.878**; highlighted geometry (S_geom, size_ratio) **< 0.80** | **Partial** — class-structure works; simple geometry alone did not |
| **Split-utility diagnostic** | `analyze_split_utility.py` (51 runs, 13,654 splits) | Best principled AUROC(**C vs A**) = center_radius_separation **0.864**; parent_n **0.930**; split_depth **0.708** | **Supported** — first non-tautological geometry clears 0.80 gate |
| **Minority safeguard counterfactual (M0–M3)** | `analyze_minority_safeguard_counterfactual.py` (51 runs × 4 methods) | M3 best counterfactual: minority survival **0.583 → 0.940** at parent_n≤4 gate; M1/M2 halt large-parent recursion | **Exploratory only** — not validated at classifier level |
| **Topo harm_risk / compactness interaction** | `analyze_topo_harm_risk.py` (51 runs) | harm_risk AUROC adversarial vs informative = **0.525**; R_opp=0 alone = **0.566**; harm_risk Δ Macro F1 corr = **−0.098** | **Rejected** as reliability score |
| **R_opp mechanism / forensics** | `analyze_r_opp_mechanism.py`, `analyze_r_opp_forensics.py` (51 runs) | Adversarial runs: higher frac R_opp=0 on high-compA balls (**0.316 vs 0.022** successful); mean \|Topo−CA\| higher (**0.552 vs 0.338**) | **Supported** as diagnostic signal; **not** a validated fix |

---

## 4. Singleton / minority fragmentation findings

### Scale

- **9,174 singleton-child events** across 17 datasets × seeds 42/0/1 (`singleton_isolation_events.csv`).
- **100%** eventually deleted by `del_ball` (`frac_deleted = 1.0`).
- **49.9%** arise from **2-sample mixed parents** (`frac_parent_num_eq2`).
- **97.4%** have only **one opposite-class neighbour at hd1** (`frac_only_one_opp_neighbour`).
- **26.7%** have **hd2 defined** (`frac_hd2_defined`); in hd2-only diagnostic (3 datasets), **676/896 = 75.4%** are UNDEFINED_HD2.
- **63.8%** are **lone-in-parent** (`frac_lone_in_parent`).

### Singleton A/B/C (different labeling scheme from split-utility A/B/C)

From `singleton_isolation_summary.json`:

| Class | n | frac | Definition |
|---|---:|---:|---|
| **B** | 6,552 | 71.4% | Neither parent-local peel nor globally isolated |
| **A** | 1,615 | 17.6% | Parent-local minority peel (d_same_parent ≤ 2·hd1) |
| **C** | 1,007 | 10.9% | No same-class neighbour anywhere within 2·hd1 |

### Parent-local AUROC — the negative result

Among **lone-in-parent** events, parent-local features predict global isolation with max AUROC ≈ **0.577** (`dist_to_sibling_center`). On full A vs C (excluding definitional features): max ≈ **0.711** (`s_size_hd2`, but only where hd2 exists).

**Near-tautological separators:**

- `lone_in_parent`: AUROC C vs A = **0.998**
- `d_same_global/hd1`: AUROC C vs A = **0.992**
- Rule "nearest same-class not in sibling → C": accuracy on A∪C = **99.8%**

These encode the **label definition**, not a discovered local rule.

### Scientific implication

Singleton creation is not a rare edge case — it is the **dominant deletion pathway** of granulation. The pathology is structural:

1. Late recursion on **tiny impure parents** (often n=2),
2. KMeans always produces a **singleton child**,
3. `del_ball` **always removes it**,
4. Parent-local geometry **cannot** distinguish "legitimate boundary minority" from "outlier" without looking **outside the parent** — which is equivalent to redefining global isolation, not granulation logic.

**Interpretation (not proven as classifier fix):** singleton repair based on hd1/hd2/n_opp inside the parent ball is a **dead end** for a novel local criterion.

---

## 5. Minority isolation and repair findings

### Granulation chain (13,654 splits, 51 runs)

| Quantity | Value | Status |
|---|---|---|
| Mean minority survival | **0.583** | Established |
| Mean majority survival | **0.750** | Established |
| Survival gap (min − maj) | **−0.167** | Established; 92.2% runs min < maj |
| Isolating splits | **3,750** (25.2% of splits) | Established |
| Preserving splits | **6,087** (49.3%) | Established |
| Isolating splits later del_ball-affected | **80.4%** | Established |
| corr(isolating rate, minority survival) | **−0.521** | Established |
| corr(isolating rate, Δ Macro F1 Topo vs CA) | **+0.050** | **Negative result** — no classifier link |
| Adversarial vs informative isolating rate | **0.262 vs 0.272** | **Negative result** — indistinguishable |

**Verdict (`minority_isolation_summary.json`):** *"PARTIAL… Do not modify the splitter for classifier gain yet."*

### Proposed B1 repair (NOT implemented)

Condition: reassign small/singleton child with parent-local minority **only if** `n_opp_within_2·hd1 ≥ 2`.

### B0 vs B1 counterfactual (51 runs, 4,663 virtual repairs)

| Metric | B0 | B1 | Δ |
|---|---:|---:|---:|
| Minority survival | 0.583 | 0.773 | **+0.190** |
| Minority-to-opposite support | 47.4 | 38.2 | **−9.2** |
| Nearest opposite-ball distance | 804.6 | 483.5 | **−321.2** |
| Opp/same distance ratio | 1.70 | 1.18 | **−0.52** |
| Heterogeneous-ball separability | 1.73 | 1.56 | **−0.04** |
| Samples with changed representation | 0 | **426.2** (mean/run) | — |

**Verdict:** *"MIXED / not clearly defensible… closer to survival inflation than a geometrically principled granulation fix."*

### Why B1 misses the main failure path

B1 requires `n_opp_within_2·hd1 ≥ 2`, but **97.4%** of singleton events have only **one** opposite neighbour at hd1, and **~50%** come from 2-sample parents where hd2 is undefined. The repair gate **does not fire** on the dominant singleton mechanism.

### Minority safeguard counterfactual (M0–M3) — exploratory only

| Method | Rule | Mean minority survival | Notes |
|---|---|---:|---|
| M0 | accept all splits | 0.583 | baseline |
| M1 | reject singleton minority peel | 1.000 | 100% rejections are peel-singletons; **16.5%** at parent_n>4 → recursion halt |
| M2 | reject n_small≤3 & min_frac≥0.5 | 1.000 | **57.9%** rejections at parent_n>4; 3 collapse runs |
| M3 | reject if parent_n≤4 & min_frac≥0.5 | **0.940** | best tradeoff in counterfactual; **0%** large-parent rejections |

M3 is the best **granulation-only counterfactual**, but it was **not** validated downstream and uses a **hard parent-size threshold** — not yet derived from split-utility geometry.

---

## 6. Gaussian / membership investigations

### Gaussian membership screen (B0 / G1 / G2)

- **Datasets:** diabetes, ecoli2, vehicle0 × 3 seeds (9 classifier runs).
- **Formulations tested:** compact, purity_compact, density, inverse_spread.
- **Primary:** `purity_compact = purity × exp(−r̃²)`.

| Model | Weights |
|---|---|
| B0 | CW × μ_A (frozen CA) |
| G1 | CW × μ_G |
| G2 | CW × μ_G × μ_Topo |

**Correlations (mean across runs):**

- corr(μ_G, μ_A) ≈ **0.051** → μ_G is **statistically independent** of CA.
- corr(μ_A, μ_Topo) ≈ **0.63–0.73** per run → Topo largely tracks CA, not μ_G.

**Classifier performance (mixed, dataset-dependent):**

- **ecoli2:** G1 improves macro_f1 (e.g. s0: 0.561→0.683) — **isolated positive**.
- **diabetes:** G1 collapses gmean to **0.000** on s42.
- **vehicle0:** G1/G2 consistently **below** B0 on gmean and minority_f1 across seeds.

**Why rejected as general direction:** independence from CA is necessary but not sufficient; G1 is **not uniformly better**, often **worse on minority metrics**, and prediction changes vs B0 are small except pathological cases. Replacing CA with another Gaussian envelope is **not a credible general contribution**.

### Why purity is almost inert at ρ=0.95

Post-granulation mean ball purity ≈ **0.997** (M0 safeguard baseline). The active filter is **`del_ball` on n<2**, not purity thresholding. At ρ=0.95, purity governs **when to keep splitting**, not **which balls survive**.

### Gaussian overlap diagnostic (51 Topo-screening runs)

| Signal | corr(Δ Macro F1) | AUROC adv vs inf |
|---|---:|---:|
| BC_overlap | +0.087 | **0.330** |
| minority_hd1_norm | **−0.340** | **0.747** |
| purity | −0.088 | 0.505 |
| kmeans_purity_gain | **−0.303** | — |
| frac R_opp=0 | −0.111 | 0.515 |

Adversarial vs informative means: BC_overlap **0.000 vs 0.018** (both near zero); hd1_norm **2.08 vs 1.13**.

**Verdict:** *"BC_overlap does not materially outperform purity + minority_hd1_norm… Gaussian-aware split experiment is not strongly justified."*

**Conclusion:** "Another Gaussian membership/overlap function" is a **weak research direction** unless tied to a different failure mode than today's evidence supports.

---

## 7. Nested-ball investigation

**Scope:** 9 runs (3 datasets × 3 seeds), pre-`del_ball`.

| Finding | Value |
|---|---|
| Total heterogeneous nested conflict pairs | **9** across all 9 runs |
| Pattern | Pure ball (n≥2) nesting **singleton** (n=1, r=0) of opposite label |
| Either ball would del_ball | **100%** |
| Either ball impure | **0%** |
| Known Topo-failure runs (ecoli2×3, vehicle0_s1) | **0** heterogeneous conflicts each |

**Interpretation:** nested heterogeneous conflicts exist but are **rare**, always **del_ball-bound**, and **absent** in the canonical Topo-failure regimes. They do **not** explain Topo failures.

---

## 8. Topo failure interaction

### R_opp / n_opp landscape (51 broad-screening runs)

Categories: **19 adversarial**, **29 informative**, **3 redundant** R_opp runs.

| Category | mean Δ Macro F1 (Topo−CA) | frac Topo beats CA |
|---|---:|---:|
| adversarial_r_opp | **−0.092** | **0%** |
| informative_r_opp | **+0.059** | **55.2%** |
| redundant_r_opp | 0.000 | 0% |

### harm_risk joint rule

```
harm_risk = frac balls with compA>0.5 AND μ_G<run_median AND (n_opp≤2 OR R_opp==0)
```

- Adversarial mean harm_risk: **0.109** vs informative **0.113** — **indistinguishable**.
- harm_risk AUROC (adv vs inf): **0.525**.
- Best single signal: **R_opp=0 alone** on high-compA balls, AUROC **0.566** — still weak.

### R_opp=0 forensics (established pattern, not fix)

On high-compA corrected balls: frac R_opp=0 = **0.316** (adversarial) vs **0.022** (successful runs in forensics sample).

**Interpretation:**

- **R_opp=0 on high-compA balls** is a stronger Topo-diagnostic signal than Gaussian compactness.
- **μ_G / harm_risk** adds little beyond zero-reciprocity structure.
- Compactness is better viewed as a **secondary ball-quality descriptor**, not a Topo reliability mechanism.

**Not established:** that downweighting these balls fixes minority performance (no intervention tested today).

---

## 9. The major breakthrough: split utility

### Experimental setup

- **17 datasets × seeds [42,0,1] = 51 runs**
- **13,654** proposed KMeans splits recorded
- Frozen: ρ=0.95, min_samples=1, del_ball n<2
- **Read-only** — no `src/` changes

### Split-utility A/B/C/D (distinct from singleton-isolation A/B/C)

| Class | n | frac | Definition |
|---|---:|---:|---|
| **A** | 6,013 | 44.0% | Small child contains global minority **and** is fully deleted by del_ball |
| **C** | 4,480 | 32.8% | **Both** children survive del_ball |
| **B** | 2,545 | 18.6% | Singleton split, small child **not** a minority peel |
| **D** | 616 | 4.5% | Other |

*(Split-quality diagnostic used a slightly different C definition — n_small≥2 and small child not deleted — yielding C=5,096, no D class. AUROC rankings are consistent; C vs A counts differ.)*

### Key AUROCs — task matters

#### AUROC for identifying **A** (all 13,654 splits) — `split_quality_diagnostic`

| Metric | AUROC(A) |
|---|---:|
| minority_fraction_small_child | **0.878** |
| parent_n | 0.811 |
| minority_hd1_norm | **0.783** |
| purity_gain | **0.705** |
| size_ratio | 0.646 |
| S_geom (= center/radius separation) | **0.619** |
| d_same_global/hd1 | **0.607** |

#### AUROC for **C vs A** (10,493 pairs) — `split_utility_diagnostic`

| Metric | AUROC(C vs A) | Notes |
|---|---:|---|
| parent_n | **0.930** | structural, not geometric |
| center_radius_separation | **0.864** | **best principled geometry** |
| composite_radius_plus_separation | **0.862** | pre-specified composite |
| composite_separation/compactness | **0.824** | pre-specified composite |
| r1 | **0.800** | clears gate |
| split_depth | 0.708 | **below 0.80** |
| purity_gain | 0.716 | |
| minority_margin_mean | 0.660 | |
| radius_reduction | **0.532** | fails |
| size_ratio | 0.593 | tautological-adjacent |

**Gate:** AUROC ≥ **0.80** for non-tautological C vs A separation.

**Today's split-utility verdict:** *"principled split criterion supported"* via `center_radius_separation` (||c₁−c₂|| / (r₁+r₂+ε)), direction: **lower → C (useful retained split)**.

### Distribution contrast (established)

| Metric | Class A p50 | Class C p50 |
|---|---:|---:|
| parent_n | **3** | **16** |
| center_radius_separation | **2.87** | **0.92** |
| child_compactness_gain | **0.54** | **0.22** |
| split_depth | **10** | **7** |

Harmful peels (A) occur on **small parents**, produce **high center/radius separation** (geometrically "successful" KMeans split), then lose the small child to `del_ball`.

### Critical distinction (do not over-claim)

| Statement | Status |
|---|---|
| "This split will produce a deleted singleton minority peel" | **Predictable** (AUROC up to 0.93 with parent_n; 0.86 with geometry) |
| "This split is geometrically/class-structurally unjustified independent of del_ball" | **NOT proven** — high separation may be valid KMeans refinement that becomes harmful only **after** n<2 filtering |
| "Split-depth alone explains everything" | **Rejected** — split_depth AUROC(C vs A) = **0.708** < 0.80 |
| "Parent size alone explains everything" | **Partially supported** — parent_n = 0.930, but geometry clears 0.80 **without** using parent_n |

### Why this matters

1. First **non-tautological geometric** statistic clearing the 0.80 gate for harmful peel vs useful split.
2. Points intervention **upstream** (split acceptance), not CW/CA/Topo reweighting.
3. Separates **late-stage fragmentation** (small parent_n) from **recursion depth** alone.
4. Pre-specified composites (separation + compactness) work; **minority margin** and **radius reduction alone do not**.

---

## 10. Current failure model of GBLSTSVM

Proposed causal chain with evidence tags:

```
[ESTABLISHED] Impure parent (purity < 0.95, often small n)
    ↓
[ESTABLISHED] Recursive KMeans(k=2) split
    ↓
[ESTABLISHED] Late-stage small minority peel (A class: 44% of splits;
               parent_n median 3; small child often singleton)
    ↓
[ESTABLISHED] High center_radius_separation + compact small child
    ↓
[ESTABLISHED] del_ball removes child (n < 2) — 100% of class-A small children
    ↓
[ESTABLISHED] Minority survival 0.583 vs majority 0.750; mean gap −0.167
    ↓
[ESTABLISHED] Sparse / heterogeneous ball graph (low n_opp on minority balls;
               adversarial runs: higher frac R_opp=0 on high-compA balls)
    ↓
[HYPOTHESIS] Topo / GBLSTSVM boundary instability on adversarial R_opp runs
               (Δ Macro F1 −0.092 adversarial vs +0.059 informative)
    ↓
[NOT ESTABLISHED] Improved granulation → improved minority classifier metrics
```

**Strongest established link today:** granulation split quality → minority deletion.  
**Weakest link:** granulation statistics → Topo ΔF1 (corr ≈ 0.05 for isolating-split rate).

---

## 11. What we have ruled out

| Mechanism | Why rejected |
|---|---|
| **Gaussian membership as CA replacement** | Independent of CA (r≈0.05) but **hurts** minority metrics on 2/3 screening datasets |
| **Gaussian overlap / BC_overlap split criterion** | AUROC 0.330 adv vs inf; adds nothing beyond hd1_norm |
| **Nested-ball conflict repair** | 9 pairs total; 0 in Topo-failure runs; all del_ball-bound |
| **Parent-local singleton repair (hd1/hd2/n_opp)** | Max parent-local AUROC ≈ 0.58–0.71; hd2 undefined 75%+; clean separator = global d_same/hd1 (tautological) |
| **B1 minority reassignment repair** | Survival ↑ but cross-ball geometry ↓; misses 97% single-opp singletons |
| **Simple delete/keep singleton heuristics** | 100% singleton children deleted regardless of label geometry |
| **Compactness / harm_risk as Topo reliability** | harm_risk AUROC 0.525; adversarial/informative means indistinguishable |
| **M1/M2 hard safeguard thresholds (implemented)** | Counterfactual only; M1/M2 halt large-parent recursion |
| **Arbitrary new weighting functions** | No diagnostic showed consistent classifier gain independent of granulation pathology |
| **Splitting for Topo F1 based on isolation rate** | corr(isolating rate, Δ Macro F1) ≈ **0.05** |

---

## 12. Current research proposal (NOT YET IMPLEMENTED)

### Working title

**Geometry-Aware / Split-Utility-Based Granular-Ball Generation for GBLSTSVM**

### Core idea

Do **not** change GBLSTSVM optimization, CW, CA, or Topo formulas.  
Modify **only split acceptance** in granulation: reject or defer splits that look like **late-stage minority fragmentation** rather than **structural refinement**.

### Candidate ingredients (not a final formula)

| Signal | Evidence |
|---|---|
| `center_radius_separation` | AUROC(C vs A) = **0.864** |
| Child compactness / weighted child radius | composite separation/compactness = **0.824** |
| Minority concentration in small child | AUROC(A) = 0.878 but **tautological-adjacent** |
| Parent size / granularity state | parent_n AUROC = **0.930** — likely necessary but not sufficient alone |

### Intended principle

> **Not every purity-improving KMeans split is classification-useful.**

A split may increase child purity and geometric separation while **destroying** minority representation via `del_ball`. The intervention targets **that joint failure mode**, not ball weights.

**Status:** proposal only. No production granulation code changed today.

---

## 13. Research hypotheses (to be tested, NOT results)

| ID | Hypothesis | Falsified if… |
|---|---|---|
| **H1** | A non-tautological split-utility score (e.g. center_radius_separation + parent-size guard) distinguishes harmful fragmentation from useful splits with AUROC ≥ 0.80 **on held-out datasets/seeds** | AUROC collapses under proper holdout or after implementing gate |
| **H2** | Geometry-aware split rejection reduces minority fragmentation **without** merely inflating ball counts or freezing recursion on large mixed parents | Survival ↑ but ball count ↓ or large-parent purity stalls |
| **H3** | Improved minority representation after granulation translates to better minority recall / gmean in GBLSTSVM | Classifier metrics unchanged despite survival gains (as with B1) |
| **H4** | Gains concentrate on datasets with sparse minority support and high class-A split rate | Uniform null effect across datasets |
| **H5** | Gains persist beyond CW + CA + Topo baselines | Effect disappears when weights are frozen to B0 |

---

## 14. Proposed experimental roadmap

| Stage | Objective | Inputs | Metrics | Success | Failure → decision |
|---|---|---|---|---|---|
| **1 — Split-utility derivation** | Finalize gate formula from today's diagnostics | `split_utility.csv`, held-out seed split | AUROC(C vs A), calibration curves, tautology audit | ≥1 non-tautological feature ≥0.80 on holdout | Return to diagnostic design |
| **2 — Simplest rule** | Derive minimal reject/accept rule (no tuning) | Top features + parent_n guard | Confusion on A/C; rejection rate by parent_n | Rule fires primarily on parent_n≤4–6 **and** high separation | If only parent_n works → size gate, not geometry |
| **3 — Implement granulation mechanism** | Virtual then real split gate in granulation | Stage 2 rule | Same as safeguard M3/M? but geometry-motivated | Minority survival ↑, large-parent recursion preserved | Abort if collapse like M1/M2 |
| **4 — Representation validation** | Does graph structure improve? | Post-granulation balls | n_opp, frac R_opp=0, hd1_norm, separability | Adversarial-like graph stats shift toward informative | Stop before classifier — B1 pattern |
| **5 — Classifier validation** | GBLSTSVM performance | 17 datasets × seeds | macro_f1, gmean, minority_recall, auroc | Statistically meaningful gain vs B0 | Abandon granulation gate |
| **6 — Ablations** | Isolate mechanism | Remove one ingredient at a time | Same | Geometry term necessary | parent_n-only suffices → weaker novelty |
| **7 — Failure-regime analysis** | Where does gate fail? | Per-dataset A/C rates | Residual A splits, D class growth | Failures interpretable | Uninterpretable failures → redesign |
| **8 — Statistical validation** | Multi-seed significance | Bootstrap over runs | CI on Δ metrics | CI excludes 0 on primary metric | Report negative honestly |

---

## 15. What should NOT be coded yet

1. **Production changes to `src/granular_balls.py`** until Stage 2 rule is frozen on holdout.
2. **CW / CA / Topo / GBLSTSVM weight modifications** — no evidence they address root cause.
3. **B1-style minority reassignment** — rejected as survival inflation.
4. **M1/M2 safeguards** — counterfactual only; cause recursion halt.
5. **Gaussian membership or overlap modules** in the training pipeline.
6. **Nested-ball or hd2 singleton repair hooks**.
7. **Threshold tuning loops** optimizing AUROC on the same 51 runs used for discovery.

---

## 16. Final research status

### What we KNOW (experimentally established today)

- Granulation produces **13,654 splits / 51 runs** with **44% class-A harmful minority peels**.
- **9,174 singleton-child events**, **100%** deleted by `del_ball`.
- Minority survival **0.583** vs majority **0.750**; isolating splits anti-correlate with minority survival (r = **−0.521**).
- Isolating-split rate does **not** predict Topo Δ Macro F1 (r ≈ **0.05**).
- **`center_radius_separation`** achieves AUROC **0.864** (C vs A) without using deletion flags.
- Nested conflicts, Gaussian overlap, harm_risk, parent-local singleton rules **failed** as primary mechanisms.
- B1 repair **inflates survival** while **weakening** cross-ball geometry.

### What we THINK (plausible, not fully validated)

- The core bug is **upstream granulation**: purity-driven KMeans keeps splitting small mixed parents into **geometrically separated but del_ball-doomed** minority peels.
- A **split-utility gate** is the most promising intervention axis discovered today.
- Topo failures on adversarial R_opp runs are better explained by **post-granulation graph sparsity / R_opp=0** than by nested conflicts or Gaussian compactness.

### What we DON'T KNOW

- Whether a split gate **improves GBLSTSVM classifier metrics** (never tested end-to-end).
- Whether high center_radius_separation marks **bad biology** or **bad interaction with del_ball** — causally unresolved.
- Optimal gate formula and whether parent_n threshold alone is enough (strong covariate, weak novelty if alone).
- Holdout robustness of AUROC **0.864**.

### What we SHOULD TEST NEXT

1. Holdout validation of split-utility features (Stage 1).
2. Derive minimal non-tautological reject rule (Stage 2).
3. Virtual replay of geometry-aware gate vs M3 counterfactual on **same 51 runs** before any `src/` edit.

### What would make us abandon the current proposal

- Holdout AUROC(C vs A) **< 0.80** for all non-tautological geometry features.
- Virtual gate raises minority survival but **replicates B1 pattern** (support/separability degradation).
- Classifier-level validation shows **null or negative** minority metrics despite representation gains.

### Strongest potential novelty claim (current, cautious)

> **Split-utility-aware granular-ball generation**: rejecting or deferring late-stage KMeans splits that are geometrically separated but structurally equivalent to minority fragmentation, without modifying the GBLSTSVM optimizer or inventing a new membership weight.

This is **diagnostically supported** but **not yet intervention-validated**.

---

## Handoff paragraph

We are extending GBLSTSVM toward a research-grade contribution by diagnosing structural failures before modifying code. Today's work (27 Aug 2026) ran read-only granulation replays on **17 datasets × seeds [42,0,1] = 51 runs** with frozen KMeans(k=2), ρ=0.95, min_samples=1, del_ball n<2. We ruled out: hd2/parent-local singleton repair (9,174 singleton events, 100% del_ball deleted, max parent-local AUROC ~0.58), nested-ball conflicts (9 heterogeneous pairs, 0 in Topo-failure runs), Gaussian membership as CA replacement (mean corr(μ_G,μ_A)≈0.05 but hurts minority metrics on 2/3 screening datasets), Gaussian overlap (BC_overlap AUROC 0.33), B1 minority reassignment (survival 0.58→0.77 but cross-ball geometry worsens; gate misses 97% single-opp singletons), and Topo harm_risk (AUROC 0.525). Minority isolation chain is reproducible (minority survival 0.583 vs majority 0.750; isolating-split rate corr with Δ Macro F1 ≈ 0.05) but does not predict Topo failure. **Major positive finding:** split-utility diagnostic on **13,654 splits** — harmful minority peels (class A, 44%) vs useful retained splits (class C, 33%) are separable by **`center_radius_separation` AUROC 0.864** (C vs A); parent_n AUROC 0.930; split_depth 0.708. Split-quality AUROC(A) also shows minority_fraction_small_child 0.878, minority_hd1_norm 0.783, S_geom 0.619. **Proposed but NOT implemented:** geometry-aware split-acceptance in granulation only — principle: not every purity-improving KMeans split is classification-useful. **Next steps:** holdout validation, derive minimal gate, virtual replay, then representation and classifier validation. Do not code src/ changes or new weight functions until Stage 2 completes.

---

## Key output files

| Diagnostic | Primary outputs |
|---|---|
| Split utility | `GBLSTSVM/experiments/output/granulation/split_utility.{csv,summary.json,report.md}` |
| Split quality | `GBLSTSVM/experiments/output/granulation/split_quality_*` |
| Singleton isolation | `GBLSTSVM/experiments/output/granulation/singleton_isolation_*` |
| Minority isolation | `GBLSTSVM/experiments/output/granulation/minority_isolation_*` |
| Minority repair | `GBLSTSVM/experiments/output/granulation/minority_repair_*` |
| Safeguard counterfactual | `GBLSTSVM/experiments/output/granulation/minority_safeguard_counterfactual_*` |
| Gaussian overlap | `GBLSTSVM/experiments/output/granulation/gaussian_overlap_*` |
| Gaussian membership | `GBLSTSVM/experiments/output/gaussian_membership/*` |
| Nested conflict | `GBLSTSVM/experiments/output/granulation/nested_conflict_*` |
| hd2 singleton | `GBLSTSVM/experiments/output/granulation/hd2_singleton_*` |
| Topo harm risk | `GBLSTSVM/experiments/output/topology/topo_harm_risk_*` |
