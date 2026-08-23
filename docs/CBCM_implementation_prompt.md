# Implementation Prompt: Competition–Connectivity Fuzzy Membership (CBCM) for GBLSTSVM

## Context (read fully before writing code)

We are extending GBLSTSVM (Granular-Ball Least Squares Twin SVM) with a new **fuzzy ball-membership mechanism**. Several mechanisms have already been tried and their outcomes are FROZEN FACTS — do not re-derive, re-tune, or "fix" them. Use them only as baselines to beat/compare against.

**Frozen baseline implementations (do not modify):**
- `GBLSTSVM` — vanilla.
- `CW-GBLSTSVM` — class-weighted (weights balls by inverse class frequency or similar global class weight `CW(y_i)`).
- `R-GBLSTSVM` — reliability weight from intra-ball purity/radius.
- `CR-GBLSTSVM` — CW × R combined.

**Known experimental results (ground truth, do not contradict or re-litigate):**
- CW gives a real, meaningful improvement over vanilla GBLSTSVM under class imbalance. Treat `CW(y_i)` as a fixed, given per-ball scalar weight that must remain in the final weight formula.
- R adds negligible value once CW is present (CR ≈ CW). Purity/radius signal is not to be reused.
- BMRU (a margin/radius uncertainty measure, `u = 2r/(|s|+eps)`) failed: it saturates/collapses and does not correlate with actual boundary-straddling. Do not implement anything based on a ball's own radius-to-margin ratio.
- OBRU (literal inter-ball volumetric overlap) failed: true geometric overlap between granular balls is almost always zero (93%+ zero on diabetes, identically zero on ecoli2, ~61% zero on vehicle0). Do not implement anything requiring literal ball-volume intersection as its core signal.
- Class-aware granulation is rejected — do not change how granular balls are constructed.
- Naive inverse-density minority weighting is rejected as a direction — it risks upweighting minority noise/outliers and is not being pursued.

**New direction being implemented now:** a fuzzy membership built from **inter-ball rank-based relationships among granular-ball centers** — specifically (1) how contested a ball's neighborhood is by the opposite class, and (2) how structurally supported a ball is by its own class — combined multiplicatively. This uses a completely different information source (ball-center graph/neighborhood topology) than CW (class counts), R (intra-ball purity/radius), BMRU (self-referential margin/radius ratio), or OBRU (literal volume intersection).

---

## What to implement

Implement three new per-ball fuzzy membership signals, computed once per training set as a preprocessing step (not inside the SVM solver), operating only on granular-ball **centers**, **labels**, and **class sizes** — the same ball objects already produced by the frozen granulation step. Do not alter granulation.

### Notation

- Balls $B_1, \dots, B_N$ with centers $c_i \in \mathbb{R}^d$, labels $y_i \in \{+1, -1\}$.
- $\text{CW}(y_i)$: the existing frozen class-weight function/lookup already used by CW-GBLSTSVM. Reuse it as-is (import/call the existing implementation; do not reimplement or rescale it).
- $k$: neighbor count, fixed once as $k = \max(5, \lceil 0.05 \, N \rceil)$, computed automatically from $N$ at runtime. Never hand-tune $k$ per dataset. Log the resolved $k$ for every dataset run.
- All distances are Euclidean on ball centers.

### Component A — Soft k-NN Class-Competition Membership, `compA(i)`

**Step A1.** For each ball $i$, find its $k$ nearest other ball centers overall (any class), call this set $\mathcal{N}_k(i)$, via a k-d tree / ball tree over all $N$ centers (exclude $i$ itself).

**Step A2.** Per-ball adaptive bandwidth:
$$\sigma_i = \text{median}_{j \in \mathcal{N}_k(i)} \, \|c_i - c_j\|$$
If $\sigma_i = 0$ (degenerate, duplicate centers), set $\sigma_i = \epsilon = 10^{-8}$.

**Step A3.** Distance weights and competition ratio:
$$w_{ij} = \exp\left(-\frac{\|c_i-c_j\|^2}{2\sigma_i^2}\right), \quad j \in \mathcal{N}_k(i)$$
$$\text{comp}_i = \frac{\sum_{j \in \mathcal{N}_k(i)} \mathbb{1}[y_j \ne y_i]\, w_{ij}}{\sum_{j \in \mathcal{N}_k(i)} w_{ij}}$$

**Step A4.** Parabolic remap (peaks at comp=0.5, vanishes at 0 and 1):
$$\mu^A_i = 4 \cdot \text{comp}_i \cdot (1 - \text{comp}_i)$$

Output: array `compA` (raw comp_i, for diagnostics) and `muA` (remapped), both length $N$, values in $[0,1]$.

### Component B — Within-Class Structural Support Membership, `muB(i)`

**Step B1.** For each class $y \in \{+1,-1\}$ separately, build a same-class k-NN graph: for each ball $i$ with label $y_i = y$, find its $k_y$ nearest same-class ball centers, where
$$k_y = \min(k, \,n_y - 1), \qquad n_y = |\{j : y_j = y\}|$$
(cap neighbor count by available same-class population; log when capping occurs).

**Step B2.** Per-ball same-class bandwidth $\sigma_i$ computed the same way as Step A2, but restricted to same-class neighbors.

**Step B3.** Raw same-class density:
$$d_i = \sum_{j \in \mathcal{N}_{k_y}^{(y)}(i)} \exp\left(-\frac{\|c_i-c_j\|^2}{2\sigma_i^2}\right)$$

**Step B4.** Within-class normalization (normalize only against other balls of the SAME class — do not mix classes here, this is what prevents re-deriving CW):
$$Z_y = \text{median-of-top-quartile}\big(\{d_l : y_l = y\}\big)$$
i.e. sort all $d_l$ for balls of class $y$ descending, take the top 25% of values, and use their median as $Z_y$ (robust normalizer — avoids one abnormally dense cluster dominating the scale; do NOT use raw max).
$$\mu^B_i = \min\!\left(1, \frac{d_i}{Z_y}\right)$$

Output: array `muB`, length $N$, values in $(0,1]$.

### Component C — Combined Competition–Connectivity Membership (primary candidate)

$$\mu^C_i = \mu^A_i \cdot \mu^B_i \qquad (\alpha=\beta=1 \text{ fixed}, \text{ not tuned})$$

Final per-ball training weight, drop-in replacement for the CR combination slot:
$$w_i = \text{CW}(y_i) \cdot \mu^C_i$$

Produce a new model class **`CC-GBLSTSVM`** (Competition-Connectivity GBLSTSVM) that is structurally identical to `CR-GBLSTSVM` (same code path, same place the per-ball weight is injected into the QPP/least-squares objective) except the weight `R_i` is replaced by `μ^C_i` as computed above. Do not touch the underlying QPP/solver code beyond swapping in this new weight vector.

Also produce two standalone diagnostic-only variants for ablation (not necessarily separate model classes if the weight vector can just be swapped): 
- `CA-GBLSTSVM`: $w_i = \text{CW}(y_i)\cdot \mu^A_i$
- `CB-GBLSTSVM`: $w_i = \text{CW}(y_i)\cdot \mu^B_i$

---

## Mandatory diagnostic pass — run BEFORE any classifier training

This is a gate, not an optional step. Implement it as a standalone script/notebook cell that runs first and prints/saves a report. **Do not proceed to classifier training/evaluation until this report is generated and reviewed.**

For each of the three target datasets (diabetes, vehicle0, ecoli2), and using the existing frozen granulation to get balls:

1. Compute `muA`, `compA`, `muB`, `muC` for all balls.
2. **Null-model check for A:** randomly permute ball labels $y_i$ (preserving class proportions) 200 times, recompute `compA` under each permutation, and report whether the real `compA` distribution differs from the permuted-null distribution (e.g., two-sample KS test, report statistic + p-value). Also report this per class.
3. **Redundancy check for B vs CW:** report Pearson and Spearman correlation between `muB` and the per-ball `CW(y_i)` value (which for a given ball is just a constant per class, so also report `muB`'s within-class variance to confirm it isn't degenerate/constant).
4. **Redundancy check between A and B:** report Pearson and Spearman correlation between `muA` and `muB` across all balls, and separately within minority-class balls only.
5. **Redundancy check for muA vs R and BMRU (if those per-ball values are available/loggable from earlier frozen experiments):** report correlation, purely for reporting/context — do not modify anything based on this.
6. **Synthetic-noise sanity check:** inject $m = \max(3, \lceil 0.02N \rceil)$ synthetic "isolated minority noise" balls by taking $m$ existing minority balls farthest from all other minority balls (by nearest same-class distance) — or if none are naturally isolated, synthetically place $m$ new minority-labeled points at random majority-region locations (e.g., mean ± small jitter of $m$ randomly chosen majority balls) and re-run granulation/membership just for this check. Report the `muC` value assigned to these known-noise balls vs the dataset's overall `muC` distribution (report percentile rank). Also do this for `CW×density-only` (a naive inverse local density baseline computed the same way as `d_i` in Step B3 but without B4's per-class normalization and without the competition factor) for contrast, to explicitly demonstrate whether `muC` avoids the "reward minority outliers" failure mode that naive density weighting has.
7. Save all of the above (numbers + at least histogram plots of muA, muB, muC per dataset) to a report file `cbcm_diagnostics_report.md` (or `.json` + plots) before any training.

**Explicit stop conditions — halt and report back to the user instead of proceeding to training if any of these hold** (do not silently adjust the formula to route around them):
- `compA` distribution is statistically indistinguishable from the label-permutation null (p > 0.05 after multiple-comparison correction) on a dataset.
- `muB` correlates with `CW` (or is near-constant within a class) with |Pearson r| ≥ 0.9.
- `muA` and `muB` correlate with |Pearson r| ≥ 0.8.
- The synthetic-noise balls land above the 70th percentile of `muC` on any dataset (i.e., mechanism is rewarding known noise).

---

## Classifier evaluation protocol (only after diagnostics pass)

On diabetes, vehicle0, ecoli2, with proper stratified k-fold cross-validation (use whatever fold protocol the existing frozen experiments used, for comparability — do not introduce a different CV scheme):

Run and report, per dataset, for each of these 5 arms:
1. `GBLSTSVM` (vanilla) — reuse existing frozen results, don't rerun if already available.
2. `CW-GBLSTSVM` — reuse existing frozen results.
3. `CA-GBLSTSVM` (competition only)
4. `CB-GBLSTSVM` (connectivity only)
5. `CC-GBLSTSVM` (combined, primary candidate)

Metrics (report all, not just accuracy): G-mean, minority-class F1, AUC, majority-class F1, overall accuracy. Report mean ± std across folds. Include a significance test (e.g., paired Wilcoxon across folds) comparing `CC-GBLSTSVM` vs `CW-GBLSTSVM` on G-mean and minority-F1.

**Hyperparameters that must be fixed identically across all three datasets before seeing any result** (no per-dataset tuning): $k$ formula (given above), the top-quartile normalization in Step B4, $\alpha=\beta=1$. If any of these need to vary per dataset to get the mechanism to "work," that is a failure of the mechanism, not a tuning opportunity — report it as such rather than adjusting and rerunning silently.

---

## Deliverables

1. Code implementing `compA`/`muA` (Component A), `muB` (Component B), `muC` (Component C), as pure functions operating on `(centers, labels)` arrays — unit-testable independent of the SVM solver.
2. `CA-GBLSTSVM`, `CB-GBLSTSVM`, `CC-GBLSTSVM` model classes/variants, sharing all code with `CR-GBLSTSVM` except the weight vector.
3. The diagnostic gate script producing `cbcm_diagnostics_report.md` + plots, run first, with the stop conditions checked and explicitly reported pass/fail per dataset.
4. The 5-arm CV evaluation script and results table, run only if the diagnostic gate passes (or with explicit notes on which datasets failed the gate and were excluded/flagged).
5. A short README section explaining, in the same terms as this prompt, why `muC` is mechanistically different from CW/R/BMRU/OBRU (for paper-writing reuse) — do not invent new justifications not stated here.

## Explicit non-goals / guardrails

- Do not modify the frozen `GBLSTSVM`, `CW-GBLSTSVM`, `R-GBLSTSVM`, `CR-GBLSTSVM` implementations.
- Do not reintroduce radius, margin, or literal ball-volume-intersection terms anywhere in Component A/B/C.
- Do not tune $k$, $\alpha$, $\beta$, or the normalization method per dataset after seeing results.
- Do not skip or soft-pedal the diagnostic gate in section "Mandatory diagnostic pass" — if it fails, report the failure, do not silently patch the formula and rerun until it passes.
