# Session 5 — Split-Utility Implementation and Entropy Granulation

**Date:** Aug–Sep 2026  
**Prior context:** Session 4 identified `center_radius_separation` as the first non-tautological split-quality signal (AUROC 0.864) and proposed a geometry-aware split gate upstream of `del_ball`. See [Session 4](Session_4_Granulation_Diagnostics_and_Split_Utility.md).

**Scope:** Read-only experiments under `experiments/` and `tests/` only. No changes to `src/`, baseline granulation, CW, R, CR, Topo, Gaussian membership, or prediction.

---

## 1. Research objective

Translate the Session 4 split-utility signal into **actionable granulation interventions** and test whether **entropy-priority scheduling** can produce compact, high-F1 representations without modifying the GBLSTSVM classifier.

Two parallel threads:

1. **Split-utility gates (SU1–SU3)** — virtual replay of geometry-aware split rejection at ρ=0.90–0.95.
2. **Adaptive entropy granulation** — online KMeans splits with entropy-priority queue instead of BFS rho90.

---

## 2. Experiments performed

| Experiment | Script | Runs | Main question |
|---|---|---:|---|
| Split-utility granulation replay | `test_split_utility_granulation.py` | 51×configs | Do SU gates improve representation vs B0 baseline? |
| Granular geometry vs performance | `analyze_granular_geometry_vs_performance.py` | 306 | Which geometry metrics predict Gaussian GBLSTSVM F1? |
| Ball utility landscape | `analyze_ball_utility_landscape.py` | 51×reps | Leave-one-ball-out influence of individual balls |
| SU weighting ablation | `test_su_weighting_ablation.py` | — | Is SU signal in granulation or weighting? |
| Adaptive entropy v1 | `test_adaptive_entropy_granulation.py` | 918 | Online entropy queue vs BFS rho90 |
| Adaptive entropy v2 | `test_adaptive_entropy_granulation_v2.py` | 918 | Candidate preservation on failed splits |
| Entropy scheduler failure | `analyze_entropy_scheduler_failure.py` | 51 | Why does online entropy stop after ~4 splits? |
| Entropy support priority | `analyze_entropy_support_priority.py` | — | Heap vs FIFO candidate ordering |
| Guarded entropy | `test_guarded_entropy_granulation.py` | 2040 | min-balls / min-depth / entropy-floor safeguards |
| Beam entropy | `test_beam_entropy_granulation.py` | — | Beam search over split candidates |
| Replay branch priority | `replay_branch_priority.py` | — | Breadth vs depth scheduling variants |
| Replay split objectives | `replay_split_objectives.py` | — | Alternative split scoring objectives |

---

## 3. Key findings

### 3.1 Geometry predicts performance (but purity does not)

`analyze_granular_geometry_vs_performance.py` on 17 datasets × 3 seeds × 6 representations (B0, SU3, rho0.80–0.95):

| Geometry metric | Pearson r vs macro F1 | Verdict |
|---|---:|---|
| `fraction_mixed_balls` | **+0.35** | Strongest geometry predictor |
| `minority_survival` | **+0.57** | Dominant representation-level signal |
| `mean_purity` | +0.04 | Weak / misleading |
| `mean_radius` | −0.13 | Weak |

**Conclusion:** Classifier success tracks **minority sample survival and mixed-ball structure**, not high purity.

### 3.2 Split-utility gates (SU) — partial signal, no classifier win

Split-utility replay (`test_split_utility_granulation.py`) implemented virtual reject rules using `center_radius_separation`, compactness gain, and combined utility scores at ρ=0.90–0.95.

- SU variants change ball counts and minority survival modestly.
- No consistent classifier-level improvement over B0 at matched representation budgets.
- **Verdict:** Geometry gate is diagnostically valid but **not sufficient** as a standalone intervention without broader scheduling changes.

### 3.3 Adaptive entropy — high F1, catastrophic collapse

Adaptive entropy v1/v2 (`test_adaptive_entropy_granulation_v2.py`):

| Config | Mean macro F1 | Mean balls | Collapse rate | Splits |
|---|---:|---:|---:|---:|
| baseline-rho90 | 0.597 | 77.8 | 5.9% | 239.7 |
| adaptive-entropy-v2 @ 50% budget | **0.613** | **4.8** | **51.8%** | **3.8** |

Entropy priority achieves **higher F1 with ~2% of rho90 splits** but **>50% collapse** and only **6% budget met**.

### 3.4 Entropy scheduler failure — candidate exhaustion

`analyze_entropy_scheduler_failure.py` traced 12,474 split records:

- Online entropy stops after **3.6 splits** vs rho90 **239.7**.
- **94%** of runs hit `no_remaining_candidates` before budget met.
- Entropy selects **deeper, smaller parents** (depth 3.94 failed vs 2.02 success).
- rho90 accepts singleton-producing splits in **50.8%** of cases; adaptive rejects them pre-commit.
- **Root cause:** depth-first entropy exhausts the candidate pool; not split-quality rejection.

### 3.5 Guarded entropy — guards do not fix collapse

`test_guarded_entropy_granulation.py` (2040 runs): min-balls, min-depth, and entropy-floor guards reduce collapse from 51.8% to ~48–50% but **do not remove unstable datasets** (12/17 still flagged). Best guard: entropy-floor t=0.4, F1=0.613, collapse=48.8%.

### 3.6 Matched compression preview

Early `test_matched_compression_controls.py` showed online entropy (F1=0.610) beats full rho90 (0.584) at **matched ball count (~4.7)** but **loses to boundary/entropy ball subsampling** at the same size — suggesting gains may be **compression + selection**, not the online scheduler itself.

---

## 4. Decision (end of session)

| Proposal | Verdict |
|---|---|
| SU geometry gate alone | **Rejected** — no robust classifier gain |
| Online entropy scheduling | **Rejected** — candidate exhaustion, >50% collapse |
| Guarded entropy (min-balls, floors) | **Rejected** — marginal collapse reduction only |
| Entropy as split priority within BFS | **Deferred** — needs complexity-controlled evaluation |

**Pivot:** Stop inventing new granulation heuristics. Evaluate whether **ball-count compression** at fixed K explains observed F1 gains before claiming a granular mechanism.

---

## 5. Outputs

| Directory | Description |
|---|---|
| `experiments/output/split_utility/` | SU1–SU3 replay results |
| `experiments/output/granular_geometry_analysis/` | Geometry–performance correlations |
| `experiments/output/ball_utility_landscape/` | Leave-one-ball-out influence |
| `experiments/output/adaptive_entropy_granulation/` | Entropy v1 results |
| `experiments/output/adaptive_entropy_v2/` | Entropy v2 + candidate preservation |
| `experiments/output/entropy_scheduler_failure/` | Split-trace failure analysis |
| `experiments/output/entropy_support_priority/` | Heap policy comparison |
| `experiments/output/guarded_entropy_granulation/` | Safeguard ablation |
| `experiments/output/beam_entropy_granulation/` | Beam search variant |
| `experiments/output/su_weighting_ablation/` | SU in granulation vs weighting |

---

## 6. Handoff to Session 6

Session 5 established that **entropy-driven online granulation is not viable** (collapse + exhaustion) but that **compressed representations can outperform full rho90**. Session 6 must:

1. Systematically vary **K** (number of retained rho90 balls) with frozen ranking rules.
2. Compare **granular-ball compression** vs **sample-level compression** at matched sample counts.
3. Test whether validation-based **K selection** is deployable without test leakage.

---

*Session 5 compiled from experiment READMEs and `summary.json` outputs. No `src/` changes.*
