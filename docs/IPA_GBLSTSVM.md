# Isolation-Penalized Ambiguity (IPA) for GBLSTSVM

## Mechanism

IPA is a controlled refinement of CBCM Component A (competition). It retains the
successful **local class ambiguity** signal while explicitly penalizing
**isolated minority/noise balls** that also receive high ambiguity.

\[
w_i = \mathrm{CW}(y_i) \cdot \mu^{\mathrm{IPA}}_i
\]

### Step A — Ambiguity (A2 remap)

From the Gaussian-weighted opposite-class k-NN fraction \(\mathrm{comp}_i\):

\[
\mathrm{ambiguity}_i = 2 \cdot \min(\mathrm{comp}_i,\; 1 - \mathrm{comp}_i)
\]

### Step B — Within-class isolation

Let \(q_i\) be the Euclidean distance from ball \(i\) to its nearest
**same-class** center. Per class \(y\):

\[
Q_y = \mathrm{median}\{q_j : y_j = y\}, \qquad
r_i = \frac{q_i}{Q_y + \epsilon}
\]

Reported isolation: \(\min(1, r_i)\). Balls with \(r_i > 1\) are unusually
isolated relative to their class.

### Step C — Penalized ambiguity

Fixed \(\lambda = 1\) (not tuned per dataset):

\[
\mu^{\mathrm{IPA}}_i = \frac{\mathrm{ambiguity}_i}{1 + \lambda \cdot \max(0,\; r_i - 1)}
\]

- Supported ambiguous balls (\(r_i \le 1\)): \(\mu^{\mathrm{IPA}} = \mathrm{ambiguity}\)
- Isolated ambiguous balls (\(r_i > 1\)): membership reduced

## Technical distinctness (not literature novelty)

| Mechanism | Signal source | IPA differs because |
|-----------|---------------|---------------------|
| **CW** | Global class frequency \(\kappa_c\) | IPA is a per-ball fuzzy factor; CW remains multiplicative. |
| **R** | Intra-ball purity and mean-distance radius | IPA uses inter-ball center topology only; no radius. |
| **BMRU** | Self margin/radius ratio | IPA uses no margin or radius. |
| **OBRU** | Literal ball-volume overlap proxy | IPA uses no volumetric intersection. |
| **Naive density** | Inverse local same-class density | IPA penalizes isolation only when ambiguity is high; no inverse-density weighting. |
| **CA (muA)** | Parabolic remap \(4c(1-c)\) of competition | IPA uses A2 ambiguity and subtracts isolated-ball reward via isolation penalty. |
| **CC (muC)** | muA × muB connectivity | IPA abandons muB; connectivity showed no incremental benefit. |

## Fixed hyperparameters

- \(k = \max(5, \lceil 0.05 N \rceil)\) from ball count \(N\) (same as CBCM)
- \(\lambda = 1\) (isolation penalty strength)
- \(\epsilon = 10^{-8}\) bandwidth/isolation fallback
- Granulation: purity=0.95, min_ball_count=2, seed=42 (frozen)

## Code layout

- `src/ipa.py` — membership functions (`compute_ipa_membership`, `ball_ipa_weights`)
- `src/ipa_gblstsvm.py` — `IPAGBLSTSVM` model class
- `experiments/analyze_ipa.py` — **mandatory diagnostic report** (run first)
- `experiments/validate_ipa.py` — CW / CA / IPA screening (run after diagnostics)
- `tests/test_ipa.py` — unit tests

## Running

```bash
cd GBLSTSVM
python -m unittest tests.test_ipa
python experiments/analyze_ipa.py
python experiments/validate_ipa.py
```

Diagnostic output: `experiments/output/ipa/ipa_diagnostics_report.md`
Validation output: `experiments/output/ipa/ipa_validation_report.md`

## Success / failure criteria

**PASS (promising controlled refinement):** vehicle0 retains most CA gain;
known synthetic noise balls receive lower muIPA percentile than ambiguity;
IPA > CW and IPA ≈ CA.

**PARTIAL:** noise percentile decreases but CA performance largely lost —
ambiguity useful, isolation penalty too destructive.

**FAIL:** no improvement over CA/CW or unstable across datasets — abandon
competition-membership family.

Frozen baselines (`GBLSTSVM`, `CW-GBLSTSVM`, `R-GBLSTSVM`, `CR-GBLSTSVM`,
`BMRU`, `OBRU`, `CBCM` variants) are not modified.
