# Session 3 — Topo and SupportAware-Topo

**Date:** 2026-08-22  
**Scope:** Implementation, screening, mechanism forensics, and one principled fix for topology-supported competition membership (Topo-GBLSTSVM).  
**Prior context:** CA (competition ambiguity) was the best fuzzy membership signal so far; IPA, SP, CBCM μB/μC failed. See [Session 2](Session_2_Phases_6_to_OBRU.md).

**Research arc (today):**

```text
CA-GBLSTSVM  (frozen baseline)
    |
    +-- Topo-GBLSTSVM           μ = A × R_opp
    |       |
    |       +-- mechanism analysis (R_opp forensics)
    |       |
    |       +-- SupportAware-Topo  R_opp^S = c(n_opp)·R_opp + (1−c)·1
    |
    └── (no CV; no further modifications after SA test)
```

---

## 0. What we were trying to learn

After CA, the open question was whether **cross-class reciprocal kNN support** on the ball-center graph could filter one-way noisy competition balls without repeating IPA's same-class isolation failure.

Today's questions:

1. Is **A₂ × R_opp** novel enough to pursue?
2. Does Topo help beyond CA on a broad screen (not just vehicle0)?
3. **Why** does Topo catastrophically fail on some datasets (ecoli2, ionosphere, vehicle0)?
4. Can a **single, non-tuned** support-confidence function fix sparse-graph failures **without** destroying real gains?

---

## 1. Mechanism (frozen definitions)

### Original Topo (unchanged baseline)

\[
A_i = 2\min(\mathrm{comp}_i,\; 1-\mathrm{comp}_i), \qquad
R_{\mathrm{opp}}(i) = \frac{n_{\mathrm{reciprocal\_opp}}}{n_{\mathrm{opp}}+\varepsilon}, \qquad
\mu^{\mathrm{Topo}}_i = A_i \cdot R_{\mathrm{opp}}(i), \qquad
w_i = \mathrm{CW}(y_i)\,\mu^{\mathrm{Topo}}_i
\]

- Ball-center kNN graph, \(k = \max(5,\lceil 0.05N\rceil)\).
- Uses only centers and labels (no radius, purity, density, IPA, SP, CBCM μB).
- Code: `src/topology_membership.py`, `src/topology_gblstsvm.py`.

### SupportAware-Topo (Topo-S) — tested once, not iterated

Unit-information Laplace / credibility shrinkage of \(R_{\mathrm{opp}}\) toward **1** (\(\lambda=1\), not dataset-tuned):

\[
c(n_{\mathrm{opp}}) = \frac{n_{\mathrm{opp}}}{n_{\mathrm{opp}}+1}, \qquad
R_{\mathrm{opp}}^{S} = c\cdot R_{\mathrm{opp}} + (1-c)\cdot 1, \qquad
\mu^{\mathrm{Topo\text{-}S}} = A \cdot R_{\mathrm{opp}}^{S}
\]

- Low \(n_{\mathrm{opp}}\) → \(R_{\mathrm{opp}}^{S}\to 1\) → CA not fully suppressed even when \(R_{\mathrm{opp}}=0\).
- High \(n_{\mathrm{opp}}\) → \(R_{\mathrm{opp}}^{S}\approx R_{\mathrm{opp}}\) (recovers original Topo).
- Code: `adjust_r_opp`, `SupportAwareTopoGBLSTSVM`.

---

## 2. Experiment 1 — Novelty assessment

| Item | Detail |
|------|--------|
| Doc | `experiments/output/topology/novelty_assessment.md` |
| Verdict | **Moderate novelty — proceed** |

**Finding:** A₂ × reciprocal opposite-class kNN fraction is not a documented FSVM/GB-SVM membership. Closest cousins are one-way fuzzy kNN (CA), same-class connectivity (μB/SP), and IPA isolation. Reciprocal **opposite-class** edge symmetry on a ball graph is a distinct statistic.

**Inference:** Worth implementing and screening; not a rehash of CA or failed mechanisms.

---

## 3. Experiment 2 — Pre-training diagnostics (3 datasets)

| Item | Detail |
|------|--------|
| Script | `experiments/analyze_topology.py` |
| Datasets | diabetes, ecoli2, vehicle0 (seed 42) |
| Outputs | `topology_diagnostics_report.md`, per-ball CSVs |

**Finding:** R_opp and μTopo distributions differ across datasets. vehicle0 shows sensible down-ranking of ambiguous balls; ecoli2 has very few balls (N≈13) with unstable reciprocity.

**Inference:** Mechanism behavior is **dataset-conditional** before any classifier is trained. Sparse granulation is an early warning sign.

---

## 4. Experiment 3 — Cheap screen (CW / CA / Topo)

| Item | Detail |
|------|--------|
| Script | `experiments/validate_topology.py` |
| Protocol | Stratified 70/15/15, seed 42, frozen C1/C2/purity |
| Doc | `experiments/output/topology/topology_validation_report.md` |

### Results (test G-mean / minority F1 vs CA)

| Dataset | Topo vs CA Δ G-mean | Topo vs CA Δ min-F1 | Verdict |
|---------|---------------------|----------------------|---------|
| diabetes | 0.000 | 0.000 | Neutral (retains CA gain) |
| vehicle0 | **+0.030** | **+0.052** | **Improved** |
| ecoli2 | **−0.115** | **−0.220** | **Failed** |

**Global verdict in report:** **LOCALIZED** — do not promote to full CV yet; do not abandon (unlike IPA/SP, vehicle0 gain was real).

**Inference:** Topo is not a universal CA upgrade. ecoli2 failure with N=13 balls and k=5 flags **graph sparsity** as the likely failure mode, not ambiguity itself.

---

## 5. Experiment 4 — Broad Topo screening

| Item | Detail |
|------|--------|
| Script | `experiments/topo_broad_screening.py` |
| Models | GBLSTSVM, CW, CA, **Topo** |
| Data | 17 available datasets × seeds {42, 0, 1} |
| Protocol | Same frozen split/hyperparameters as prior screens; **no tuning** |
| Outputs | `topo_broad_screening_results.csv`, `topo_broad_screening_summary.json` |

### Overall vs CA (51 runs = 17×3)

| Metric | Wins / Ties / Losses | Mean Δ | Median Δ |
|--------|----------------------|--------|----------|
| Macro F1 | 6 / 2 / 9 (dataset-level aggregate) | −0.0008 | ≈0 |
| G-mean | 6 / 3 / 8 | −0.0049 | 0 |

Run-level (from support-aware re-screen, identical Topo numbers): macro F1 **16/15/20**, mean Δ **−0.0008**.

### Mechanism aggregates (Topo rows)

| Diagnostic | Value |
|------------|-------|
| Mean prediction change rate vs CW | 25.3% |
| Datasets where Topo beats CA (G-mean, dataset mean) | 6 / 17 |
| Degenerate R_opp runs (std = 0) | 3 |
| Mean corr(Topo weights, CA weights) | 0.74 |

### Notable dataset means (macro F1, Topo − CA)

| Dataset | Δ macro F1 | Notes |
|---------|------------|-------|
| chronic_kidney_disease | **+0.189** | Large win; sparse n_opp (~1.6 mean) |
| spambase | **+0.112** | Large win; dense graph (551 balls, n_opp ~11) |
| credit_approval | +0.023 | Small win |
| ecoli2 | −0.037 | Consistent loss |
| ionosphere | −0.085 | Consistent loss |
| vehicle0 | −0.028 | Mixed across seeds |

**Inference:** Topo is **bimodal** — strong wins on a few imbalanced sets (chronic kidney, spambase) but net-neutral to slightly negative on average. It is **not ready for CV** as a general replacement for CA.

---

## 6. Experiment 5 — R_opp mechanism analysis (all 51 Topo runs)

| Item | Detail |
|------|--------|
| Script | `experiments/analyze_r_opp_mechanism.py` |
| Method | Recompute ball weights; categorize each run; correlate with performance |
| Output | `r_opp_mechanism_summary.json` |

### Run categories (fixed thresholds, not tuned on outcomes)

| Category | n | Mean Δ macro F1 vs CA | Frac Topo > CA |
|----------|---|-------------------------|----------------|
| informative_r_opp | 29 | **+0.059** | 55% |
| adversarial_r_opp | 19 | **−0.092** | **0%** |
| redundant_r_opp | 3 | 0.000 | 0% |

**Largest adversarial cases:** ecoli2_s42 (Δ=−0.215), ionosphere_s0 (Δ=−0.193), vehicle0_s1 (Δ=−0.167).

**Inference:** When R_opp meaningfully reshapes weights, outcomes split cleanly: **informative → help**, **adversarial → always hurt**. The problem is not "R_opp is always bad" but "R_opp is sometimes catastrophically wrong." r_opp_std alone does not separate wins from losses; **sparse opposite-class support** is the sharper story (see forensics).

---

## 7. Experiment 6 — R_opp ball-level forensics (6 curated runs)

| Item | Detail |
|------|--------|
| Script | `experiments/analyze_r_opp_forensics.py` |
| Successful | chronic_kidney_disease_s42, credit_approval_s42, spambase_s42 |
| Adversarial | ecoli2_s42, ionosphere_s0, vehicle0_s1 |
| Output | `r_opp_forensics_summary.json`, per-ball CSV |

### Compact comparison (successful vs adversarial, aggregated)

| Metric | Successful | Adversarial |
|--------|------------|-------------|
| Total balls (3 runs) | 685 | 91 |
| Downweighted balls mean \|Topo−CA\| | 0.338 | **0.552** |
| Downweighted mean n_opp | **10.1** | **2.3** |
| Downweighted mean R_opp | **0.758** | 0.568 |
| Frac high-comp minority downweighted with R_opp=0 | **2.2%** | **31.6%** |
| High-comp minority mean \|Topo−CA\| | 0.372 | **0.649** |

### Per-run highlights

| Run | n_balls | Key pathology |
|-----|---------|---------------|
| spambase_s42 | 551 | Stable R_opp (0.77), rich n_opp (11.4) → large gain |
| chronic_kidney_disease_s42 | 52 | Moderate n_opp (2.2), R_opp=0 on 50% of balls but corrections still net positive |
| ecoli2_s42 | **13** | kNN graph too sparse; 69% R_opp=0 |
| ionosphere_s0 | 27 | R_opp mean 0.08; high-comp minority \|Topo−CA\|≈1.16 (full CA wipeout) |
| vehicle0_s1 | 51 | Sparse n_opp (2.1); strong minority suppression despite moderate R_opp |

**Three strongest patterns:**

1. **Sparse granulation destabilizes R_opp** — adversarial runs have far fewer balls and mean n_opp≈2 vs 10 in successful runs.
2. **R_opp=0 is catastrophic, not selective** — on high-competition minority balls, zero reciprocity removes the entire CA weight (~|Topo−CA|≈1), not a gentle noise filter.
3. **Purity/radius do not explain failures** — harmful corrections are larger in magnitude on the same high-compA minority balls, not on impure or oversized balls.

**Inference (central hypothesis confirmed):**  
\(R_{\mathrm{opp}}\) is informative only when the ball graph has **sufficient opposite-class support** (high n_opp, stable reciprocity). When support is weak, \(R_{\mathrm{opp}}=0\) **fully suppresses** high-competition minority balls → minority recall collapse.

---

## 8. Experiment 7 — SupportAware-Topo screening

| Item | Detail |
|------|--------|
| Script | `experiments/topo_support_aware_screening.py` |
| Models | GBLSTSVM, CW, CA, Topo, **SupportAware-Topo** |
| Data | Same 17 datasets × seeds {42, 0, 1} |
| Diagnostics | Per-ball CSV for 6 focus runs |
| Output | `topo_support_aware_report.md`, summary JSON |

### Overall vs CA (51 runs)

| Method | Macro F1 W/T/L | Mean Δ | G-mean W/T/L | Mean Δ | Minority F1 W/T/L | Mean Δ |
|--------|----------------|--------|--------------|--------|-------------------|--------|
| Topo | 16/15/20 | −0.0008 | 15/17/19 | −0.0049 | 13/17/21 | −0.0083 |
| **SupportAware-Topo** | 15/20/16 | **+0.0031** | 12/23/16 | −0.0066 | 12/23/16 | −0.0108 |

SA: fewer losses, more ties, slightly positive mean macro F1 vs CA. G-mean and minority F1 remain slightly negative on average.

### Focus runs — did attenuation fix catastrophes while keeping gains?

| Run | Topo Δ macro F1 vs CA | SA Δ macro F1 vs CA | Outcome |
|-----|------------------------|---------------------|---------|
| ecoli2_s42 | −0.215 | **+0.024** | Catastrophe removed |
| ionosphere_s0 | −0.193 | **0.000** | Ties CA (R_opp 0.08→0.60 adjusted) |
| vehicle0_s1 | −0.167 | **+0.005** | Catastrophe removed |
| chronic_kidney_disease_s42 | **+0.336** | **0.000** | **Gain erased** (n_opp mean 1.6) |
| credit_approval_s42 | −0.061 | −0.010 | Still below CA at this seed |
| spambase_s42 | +0.128 | **+0.177** | **Gain preserved and improved** |

### Weight diagnostics (focus runs)

| Run | frac R_opp=0 | R_opp → R_opp^S (mean) | mean \|Topo−CA\| → \|SA−CA\| |
|-----|--------------|------------------------|-------------------------------|
| ionosphere_s0 | 0.85 | 0.08 → 0.60 | 0.50 → 0.38 |
| ecoli2_s42 | 0.69 | 0.29 → 0.62 | 0.39 → 0.36 |
| spambase_s42 | 0.02 | 0.77 → 0.79 | 0.34 → 0.33 (minimal change) |

**Inference:** The Laplace shrinkage **works as designed** — it stops R_opp=0 from zeroing CA on sparse graphs. But the **same guard** collapses SA onto CA when n_opp is low everywhere (chronic_kidney_disease), erasing Topo’s largest win. The fix is **partial**: safer average behavior, not a strict Pareto improvement.

---

## 9. Synthesis — what we learned today

### What worked

1. **Topo mechanism is real on dense ball graphs.** spambase (551 balls, n_opp≈11) and vehicle0 (seed 42 cheap screen) show CA gains preserved or increased when reciprocal opposite support is stable.
2. **Forensics identified a precise failure mode:** not ambiguity, not purity, but **untrusted R_opp on sparse opposite-class neighborhoods** causing full downweight of high-competition minority balls.
3. **SupportAware-Topo is a principled one-parameter fix** that removes the three worst adversarial runs without dataset-specific tuning.

### What did not work

1. **Topo is not a general CA upgrade.** Broad screen: mean macro F1 vs CA ≈ 0, with 9 dataset-level losses.
2. **SupportAware-Topo does not preserve all meaningful gains.** chronic_kidney_disease (+0.34 macro F1) collapses to CA because its balls also have weak n_opp.
3. **No free lunch:** attenuating untrusted R_opp and preserving trusted R_opp cannot be separated by n_opp alone when both success and failure share low per-ball n_opp but differ in graph scale (52 vs 13 balls).

### Mechanism diagram

```text
                    ┌─────────────────────────────────────┐
                    │  Ball graph density (N balls, k)   │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
        N >> k, rich n_opp      N ≈ k, low n_opp        R_opp ≈ 0
        (spambase)              (ecoli2, ionosphere)    on high-compA
              │                       │                  minority balls
              ▼                       ▼                       │
        R_opp stable              R_opp noisy                 ▼
        informative               adversarial            CA weight → 0
              │                       │                       │
              ▼                       ▼                       ▼
        Topo > CA                 Topo << CA              minority recall
                                                        collapse
```

### Decision (end of session)

| Question | Answer |
|----------|--------|
| Promote original Topo to CV? | **No** — localized, bimodal, net-neutral |
| Promote SupportAware-Topo to CV? | **No** — partial fix; erases largest win |
| Further modifications today? | **No** — per protocol, one test only |
| Is R_opp mechanism understood? | **Yes** — sparse opposite-class graph support |
| Best candidate for future work? | Topo only where n_balls >> k **and** n_opp is reliably high; any guard must not shrink toward CA on genuinely informative sparse wins |

---

## 10. Artifact index

| Artifact | Path |
|----------|------|
| Topo docs | `GBLSTSVM/docs/TOPOLOGY_GBLSTSVM.md` |
| Novelty | `GBLSTSVM/experiments/output/topology/novelty_assessment.md` |
| Diagnostics | `GBLSTSVM/experiments/output/topology/topology_diagnostics_report.md` |
| Cheap validation | `GBLSTSVM/experiments/output/topology/topology_validation_report.md` |
| Broad Topo screen | `GBLSTSVM/experiments/output/topology/topo_broad_screening_results.csv` |
| R_opp mechanism | `GBLSTSVM/experiments/output/topology/r_opp_mechanism_summary.json` |
| R_opp forensics | `GBLSTSVM/experiments/output/topology/r_opp_forensics_summary.json` |
| SA screen results | `GBLSTSVM/experiments/output/topology/topo_support_aware_screening_results.csv` |
| SA report | `GBLSTSVM/experiments/output/topology/topo_support_aware_report.md` |
| SA ball diagnostics | `GBLSTSVM/experiments/output/topology/topo_support_aware_ball_diagnostics.csv` |
| Unit tests | `GBLSTSVM/tests/test_topology.py`, `GBLSTSVM/tests/test_support_aware_topo.py` |
| SA screening script | `GBLSTSVM/experiments/topo_support_aware_screening.py` |

---

## 11. One-line takeaway

**Topo’s reciprocal-opposite signal helps when the ball graph is dense enough to trust it; when n_opp is weak, R_opp=0 silences high-competition minority balls — and shrinking R_opp toward 1 fixes the catastrophes but also washes out the wins that depended on aggressive downweighting.**
