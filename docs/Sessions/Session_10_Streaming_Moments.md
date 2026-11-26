# Session 10 — Streaming with mergeable moments

Date: 26 Nov 2026 · Code: `experiments/analysis/a10_streaming/` · Outputs: `experiments/output/a10_streaming/`

## 0. Question

Session 9 left the paper with a critical message and no constructive one. The candidate constructive claim was that once balls store exact moments, granulation's value shifts to systems properties: balls can be merged, updated incrementally and unlearned. This session tests that claim, and in particular whether **balls** add anything beyond **class-level moments**.

## 1. Design

- 17 datasets × seeds 0–4 × label noise {0, 10%, 20%} = 255 streams. Standardized features; noise is applied to training labels only.
- The shuffled training split arrives in 10 batches. Each arm updates after every batch and is scored on the clean test split (code rule).
- c1, c2 are fixed before streaming, tuned held-out on the (noisy) training split: RAW's c for the moment arms, PUB's c for the published arms.

| Arm | What it stores | Update |
|---|---|---|
| RAW_REFIT | every sample seen | refit LSTSVM on all of them (exact reference) |
| CLASS_MOM | per-class Σuuᵀ and Σu | add the batch's sums, solve |
| BALL_MOM | same totals | granulate the batch, add each ball's exact moments |
| BALL_MOM_W | same totals | as BALL_MOM, each class-part of a ball weighted by local label agreement (count / parent-ball size). Pre-specified; the only ball-specific ingredient. |
| PUB_STREAM | all ball centers and radii | granulate the batch (deletion path), append, solve |
| PUB_REGRAN | every sample seen | re-granulate everything, solve |

Unlearning check: after the last batch, subtract batch 1's moments and compare with a refit on batches 2–10.

```
python experiments/analysis/a10_streaming/run_streaming.py
python experiments/analysis/a10_streaming/analyze_streaming.py
```

## 2. Results

**Exactness (all 255 streams, all 10 steps).**
- CLASS_MOM and BALL_MOM equal RAW_REFIT to a relative error ≤ 3.4e-11.
- Unlearning by subtraction equals a refit without the removed batch to ≤ 2.9e-11.

**Final test macro-F1 (dataset means, after 10 batches):**

| Noise | RAW_REFIT = CLASS_MOM = BALL_MOM | BALL_MOM_W | PUB_REGRAN | PUB_STREAM |
|---|---|---|---|---|
| 0% | 0.825 | 0.826 | 0.784 | 0.727 |
| 10% | 0.807 | 0.810 | 0.731 | 0.722 |
| 20% | 0.779 | 0.781 | 0.679 | 0.686 |

Contrasts (Holm over 12 tests):

| Contrast | 0% | 10% | 20% |
|---|---|---|---|
| CLASS_MOM − PUB_STREAM | +0.098 (0.007), 15/17 | +0.086 (0.003), 16/17 | +0.093 (0.001), 16/17 |
| CLASS_MOM − PUB_REGRAN | +0.041 (0.022), 14/17 | +0.076 (0.0002), 17/17 | +0.101 (0.003), 15/17 |
| BALL_MOM_W − CLASS_MOM | +0.000 (1.0) | +0.002 (0.79) | +0.002 (0.79) |

**Cost after the last batch** (median over streams; spambase in brackets, n = 3,220, d = 57):

| Arm | Update time | Numbers stored |
|---|---|---|
| CLASS_MOM | 0.10 ms (0.28 ms) | 840 (6,844), constant |
| RAW_REFIT | 0.14 ms (1.2 ms) | 9,800 (186,760), grows with n |
| BALL_MOM(_W) | 10 ms (94 ms) | 840 (6,844), constant |
| PUB_STREAM | 10 ms (91 ms) | 1,393 (27,318), grows with balls |
| PUB_REGRAN | 83 ms (655 ms) | 9,800 (186,760), grows with n |

## 3. Inferences

1. **Exact streaming and exact unlearning work.** That is a real, useful property of the least-squares twin SVM. Storage is constant: 27× less than keeping the samples on spambase, independent of n. Every update and every deletion is exact.
2. **But it comes from moments, not balls.** Per-class running totals do all of it. Granulating the batch first gives the identical model, at 100–300× the update time.
3. **The one ball-specific idea did not help.** Local label-agreement weights gave +0.000 to +0.002 F1, not significant even at 20% noise.
4. **The published GBLSTSVM cannot stream well.** Appending batch-wise balls loses about 9–10 F1 points to exact streaming at every noise level. Re-granulating everything costs 800–2,300× the update time of CLASS_MOM and still loses 4–10 points.
5. Speed is not the argument in the linear case. Refitting from samples is already sub-millisecond at these sizes; the advantages are memory, privacy (sharing moments instead of samples) and exact unlearning.

## 4. What this means for the paper

- The constructive section exists, but its message is **"store moments"**, not **"store better balls"**. Exact, constant-memory streaming and unlearning, plus 9–10 F1 points over the published method in the same setting, is a clear positive result to recommend.
- It sharpens the critique rather than rescuing granular balls. For linear least-squares twin SVMs, every benefit tested so far (accuracy, streaming, unlearning, memory) is delivered by class-level moments without granulation.
- The only granular-specific direction still untested is a q-minimising partition under a memory budget too small for full scatter matrices. Under a budget that allows class moments, which for d ≤ ~100 is always, it has nothing to improve. It matters only for very high d or the kernel case.
- Recommended framing: *"Least-squares twin SVMs need no granulation: exact moments give accuracy, streaming and unlearning; granular-ball summaries lose all three."*
