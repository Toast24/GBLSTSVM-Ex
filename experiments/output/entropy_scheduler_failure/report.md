# Entropy scheduler failure diagnostic

Read-only comparison of rho=0.90 BFS vs online entropy-priority scheduling. Traces every attempted split, child del_ball survival, and candidate exhaustion.

Runs: **51** dataset×seed pairs | Split records: **12474**

## Scheduler summary

| scheduler | success splits | failed | singleton fail | no-cand stop | budget met | final balls | mean depth | max depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rho90 | 239.7 | 0.00 | 0.00 | 0.00 | 1.00 | 77.8 | 3.69 | 7.4 |
| adaptive_entropy | 3.6 | 0.33 | 0.33 | 0.94 | 0.06 | 4.6 | 1.80 | 3.6 |

## Adaptive failure breakdown (split records)

- success: 73.9%
- singleton deletion: 6.8%
- invalid geometry: 0.0%
- no remaining candidates: 19.3%

## Candidate trajectory

- **rho90**: after last split — candidates=0.0, splittable=0.0
- **adaptive_entropy**: after last split — candidates=0.3, splittable=0.3

## 1. Why does online entropy stop after few splits?

Online entropy stops after **3.6** splits on average vs rho90 **239.7** (budget = rho90 count). **94%** of runs hit `no_remaining_candidates` before budget is met; only **6%** meet budget. Singleton failures contribute **0.33** failed splits/run vs **0.33** total failures. **Primary cause: candidate exhaustion** — entropy-priority deep-splits high-entropy branches until no ball remains below ρ=0.90, not split rejection.

## 2. Is entropy selecting unstable branches?

Failed adaptive splits have parent depth 3.94 vs success 2.02, parent entropy 0.914 vs 0.853. **Yes — entropy selects deeper, smaller parents** that are more likely to produce singleton children and terminate refinement paths on failure.

## 3. Is del_ball removing useful refinement paths?

rho90 accepts splits with singleton children in **50.8%** of splits; entropy rejects them pre-commit (only **0.0%** of its successful splits produce singleton children). rho90 both-survive rate: **49.2%**; entropy: **100.0%**. **Yes — del_ball would remove refinement paths** that rho90 preserves by accepting singleton-producing splits; adaptive pre-rejection plus `split_done` on failure permanently removes branches.

## 4. What minimal change is required?

### a) Candidate preservation

After last successful split, entropy leaves **0.3** splittable balls (heap candidates **0.3**). **Candidate preservation** — do not mark failed-split parents `split_done`; re-queue siblings and relax purity only after genuine exhaustion.

### b) Softer deletion

Singleton rejection blocks **6.8%** of entropy split attempts. **Softer deletion** — accept singleton splits like rho90 and defer to del_ball, or retry with alternate split geometry before abandoning the parent.

### c) Different split selection

Entropy mean depth **1.80** vs rho90 **3.69**; max depth 3.6 vs 7.4. **Different split selection** alone is insufficient — BFS breadth-first maintains more parallel refinement paths; pure entropy depth-first exhausts candidates early. Combine entropy priority with breadth caps or minimum parent-size guards.

## Outputs

- `results.csv`
- `summary.json`

