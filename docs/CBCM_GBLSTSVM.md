# Competition–Connectivity Fuzzy Membership (CBCM) for GBLSTSVM

## Mechanistic distinction from prior weighting schemes

CBCM assigns per-ball training weights from **inter-ball center neighborhood topology** — how contested a ball's local region is by the opposite class (Component A) and how structurally supported it is within its own class (Component B). The final fuzzy membership is their product:

\[
\mu^C_i = \mu^A_i \cdot \mu^B_i, \qquad w_i = \mathrm{CW}(y_i) \cdot \mu^C_i
\]

This uses a **completely different information source** than the frozen baselines:

| Mechanism | Signal source | Why CBCM differs |
|-----------|---------------|------------------|
| **CW** | Global class counts \(n_c\) | CBCM does not re-derive class frequency; CW remains as a fixed multiplicative factor. |
| **R** | Intra-ball purity and mean-distance radius | CBCM uses no radius or purity; R was shown to add negligible value once CW is present (CR ≈ CW). |
| **BMRU** | Self-referential margin/radius ratio \(u = 2r/(|s|+\epsilon)\) | CBCM uses no margin or radius; BMRU saturated and did not correlate with boundary straddling. |
| **OBRU** | Literal inter-ball volumetric overlap proxy | CBCM uses no ball-volume intersection; true geometric overlap was ~zero on benchmark datasets. |

Component A (`muA`) measures **class competition** in a soft k-NN graph over all ball centers: balls in contested neighborhoods (mixed-class neighbors) receive higher competition scores, remapped to peak at `comp = 0.5`.

Component B (`muB`) measures **within-class structural support** via same-class k-NN density, normalized per class using a robust top-quartile median (not global max). This is intentionally **not** inverse global density weighting and does not mix classes during normalization.

## Variants

| Model | Weight formula |
|-------|----------------|
| CA-GBLSTSVM | \(w_i = \mathrm{CW}(y_i) \cdot \mu^A_i\) |
| CB-GBLSTSVM | \(w_i = \mathrm{CW}(y_i) \cdot \mu^B_i\) |
| CC-GBLSTSVM | \(w_i = \mathrm{CW}(y_i) \cdot \mu^C_i\) |

## Fixed hyperparameters (not tuned per dataset)

- \(k = \max(5, \lceil 0.05 N \rceil)\) from ball count \(N\)
- \(\alpha = \beta = 1\) (multiplicative combination, no exponents)
- Top-quartile median normalization in Component B
- \(\sigma\) fallback \(\epsilon = 10^{-8}\) for degenerate bandwidths

## Code layout

- `src/cbcm.py` — pure membership functions (`compute_competition_membership`, `compute_connectivity_membership`, `compute_cbcm_membership`)
- `src/cbcm_gblstsvm.py` — `CAGBLSTSVM`, `CBGBLSTSVM`, `CCGBLSTSVM` model classes
- `experiments/cbcm_diagnostics.py` — **mandatory diagnostic gate** (run first)
- `experiments/validate_cbcm.py` — 5-arm CV evaluation (run only after gate passes)

## Running

```bash
cd GBLSTSVM
python -m experiments.cbcm_diagnostics
python -m experiments.validate_cbcm   # only if diagnostics pass
python -m unittest tests.test_cbcm
```

Diagnostic output: `experiments/output/cbcm/cbcm_diagnostics_report.md`

## Diagnostic stop conditions

Training must not proceed silently if any dataset triggers:

1. `compA` statistically indistinguishable from label-permutation null (Bonferroni-corrected p > 0.05)
2. `muB` redundant with CW (|Pearson r| ≥ 0.9) or near-constant within class
3. `muA` and `muB` highly correlated (|Pearson r| ≥ 0.8)
4. Synthetic noise balls land above 70th percentile of `muC`

Frozen baselines (`GBLSTSVM`, `CW-GBLSTSVM`, `R-GBLSTSVM`, `CR-GBLSTSVM`) are not modified.
