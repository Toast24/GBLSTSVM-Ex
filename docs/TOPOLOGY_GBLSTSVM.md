# Topology-supported competition membership (Topo-GBLSTSVM)

## Mechanism

Topo keeps the useful local **class-ambiguity** signal from CA/A2 and multiplies it by **reciprocal opposite-class kNN support** on the ball-center graph.

\[
w_i = \mathrm{CW}(y_i)\cdot \mu^{\mathrm{Topo}}_i, \qquad
\mu^{\mathrm{Topo}}_i = A_i \cdot R_{\mathrm{opp}}(i)
\]

### Step A — A2 ambiguity (unchanged)

\(\mathrm{comp}_i\) is the Gaussian-weighted fraction of opposite-class neighbors among the \(k\) nearest ball centers (same definition as CA/CBCM).

\[
A_i = 2 \cdot \min(\mathrm{comp}_i,\; 1-\mathrm{comp}_i)
\]

Peaks at a mixed neighborhood (\(\mathrm{comp}=0.5\)); vanishes for pure interiors and for balls buried in the opposite class.

### Step B — reciprocal opposite-class support

On the directed kNN graph of ball centers, for each opposite-class neighbor \(j\) of \(i\), test whether \(i \in \mathcal{N}_k(j)\).

\[
R_{\mathrm{opp}}(i) =
\frac{\#\{\text{reciprocal opposite-class neighbors}\}}
{\#\{\text{opposite-class neighbors}\} + \varepsilon}
\]

\(R_{\mathrm{opp}}=0\) if \(i\) has no opposite-class neighbors, or if all such links are one-way.

### Fixed \(k\)

\[
k = \max\bigl(5,\; \lceil 0.05 N \rceil\bigr)
\]

logged per dataset. No \(\alpha/\beta\) and no learned parameters.

## Relationship to CA / CBCM / IPA / SP

| Mechanism | What it measures | Why Topo is different |
|-----------|------------------|------------------------|
| **CA** | One-way local competition \(\mathrm{comp}_i\) (parabolic \(\mu^A=4c(1-c)\) in the frozen model) | Topo uses A2 and **requires mutual** opposite-class kNN links |
| **CBCM \(\mu^B\)** | Same-class kNN density / connectivity | Failed as an add-on; Topo does **not** use own-class support |
| **CBCM \(\mu^C\)** | \(\mu^A \times \mu^B\) | Combined connectivity added nothing beyond competition |
| **SP** | Competition × same-class support | Hurt the useful competition signal |
| **IPA** | \(A_i / (1+\lambda\cdot\text{same-class isolation})\) | Isolation is **distance to nearest same-class** center; it damaged CA on vehicle0. Topo filters by **cross-class reciprocity**, not same-class isolation |
| **CW** | Global class frequency | Remains a multiplicative factor |

Hypothesis: local ambiguity is useful, but ambiguity alone cannot tell a **structurally two-sided boundary** from an **unsupported / one-way** noisy ball.

## Code layout

- `src/topology_membership.py` — kNN graph, \(\mathrm{comp}\), \(A\), \(R_{\mathrm{opp}}\), \(\mu^{\mathrm{Topo}}\); SupportAware transform
- `src/topology_gblstsvm.py` — `TopoGBLSTSVM`, `SupportAwareTopoGBLSTSVM`
- `experiments/analyze_topology.py` — diagnostics before training
- `experiments/validate_topology.py` — cheap CW / CA / Topo screen
- `tests/test_topology.py`
- Novelty write-up: `experiments/output/topology/novelty_assessment.md`

## Running

```bash
cd GBLSTSVM
python -m unittest tests.test_topology
python experiments/analyze_topology.py
python experiments/validate_topology.py
```

Frozen baselines (`GBLSTSVM`, `CW-GBLSTSVM`, `R-GBLSTSVM`, `CR-GBLSTSVM`) and failed mechanisms (BMRU, OBRU, CBCM μB/μC, SP, IPA) are not modified or retuned.

## SupportAware-Topo (Topo-S)

Original Topo is unchanged. SupportAware-Topo attenuates \(R_{\mathrm{opp}}\) when opposite-class graph support is weak, using a single unit-information Laplace / credibility shrinkage toward 1 (no dataset-specific thresholds):

\[
c(n_{\mathrm{opp}}) = \frac{n_{\mathrm{opp}}}{n_{\mathrm{opp}}+\lambda},\qquad \lambda=1
\]

\[
R_{\mathrm{opp}}^{S}(i) = c(n_{\mathrm{opp}}(i))\, R_{\mathrm{opp}}(i) + \bigl(1-c(n_{\mathrm{opp}}(i))\bigr)\cdot 1
\]

\[
\mu^{\mathrm{Topo\text{-}S}}_i = A_i \cdot R_{\mathrm{opp}}^{S}(i),\qquad
w_i = \mathrm{CW}(y_i)\cdot \mu^{\mathrm{Topo\text{-}S}}_i
\]

- Low \(n_{\mathrm{opp}}\): \(c\to 0\), so \(R_{\mathrm{opp}}^{S}\to 1\) and CA is not fully suppressed even if \(R_{\mathrm{opp}}=0\).
- High \(n_{\mathrm{opp}}\): \(c\to 1\), so \(R_{\mathrm{opp}}^{S}\approx R_{\mathrm{opp}}\) (original Topo).

Code: `adjust_r_opp` / `SupportAwareTopoGBLSTSVM`. Screening: `experiments/topo_support_aware_screening.py`.
