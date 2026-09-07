# GBLSTSVM Research Sessions

Chronological log of research sessions. Each session documents hypotheses, experiments, verdicts, and handoff decisions.

| Session | Title | Date | Status |
|---|---|---|---|
| [1](Session_1_GBLSTSVM_Research_Findings.md) | GBLSTSVM Research Findings (Phases 1–5) | — | Complete |
| [2](Session_2_Phases_6_to_OBRU.md) | Phases 6–8, BMRU, and OBRU | — | Complete |
| [3](Session_3_Topo_and_SupportAware_Topo.md) | Topo and SupportAware-Topo | — | Complete |
| [4](Session_4_Granulation_Diagnostics_and_Split_Utility.md) | Granulation Diagnostics and Split Utility | 27 Aug 2026 | Complete |
| [5](Session_5_Split_Utility_Implementation_and_Entropy_Granulation.md) | Split-Utility Implementation and Entropy Granulation | Aug–Sep 2026 | Complete |
| [6](Session_6_Compression_K_Sensitivity_and_Validation.md) | Compression K-Sensitivity and Validation | Sep 2026 | Complete |
| [7](Session_7_Control_Reconciliation_and_Regranulation_Ablated.md) | Control Reconciliation and Re-Granulation Ablation | 7 Sep 2026 | **Terminal** |

## Research arc (Sessions 4–7)

1. **Session 4** — Diagnosed upstream granulation failures; identified `center_radius_separation` as split-quality signal.
2. **Session 5** — Implemented split-utility gates and entropy schedulers; rejected online entropy (>50% collapse).
3. **Session 6** — Showed compression at small K improves F1; ball ranking weak; validation K-selection promising but controls inconsistent.
4. **Session 7** — Reconciled sample controls; ablated re-granulation vs singleton at fixed samples → **CASE B: stop granular-structure branch**.

## Current recommendation

Do not pursue new granular heuristics (entropy ranking, boundary ranking, adaptive schedulers, rho tuning). Gains are explained by **sample-level compression + class-weight semantics**, not granular-ball structure.
