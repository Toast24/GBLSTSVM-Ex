"""
Adaptive entropy granulation v2 — candidate-preservation experiment.

Online live KMeans splits with entropy-priority queue. v2 changes only
candidate management: failed splits do not permanently remove the parent;
parents are re-queued with decaying priority up to max_attempts=3.

Does NOT modify src/, solver, prediction, weighting, or baseline granulation.

Policies:
  baseline-rho90              — BFS rho=0.90 (full)
  baseline-rho90-{pct}pct     — BFS rho capped at pct% of rho90 split count
  adaptive-entropy-v1-{pct}pct — v1 online entropy (permanent removal on failure)
  adaptive-entropy-v2-{pct}pct — v2 candidate preservation on failure
  adaptive-entropy-v1-sqrt-k{k} / v2-sqrt-k{k} — budget = k * sqrt(n)

Run from GBLSTSVM root:
  python experiments/test_adaptive_entropy_granulation_v2.py --smoke
  python experiments/test_adaptive_entropy_granulation_v2.py
  python experiments/test_adaptive_entropy_granulation_v2.py --from-csv
"""

from __future__ import annotations

import heapq
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    RHO_VALUES,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _labels,
    _to_ball_info,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "adaptive_entropy_v2"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
MAX_ATTEMPTS = 3
EPS = 1e-12
RHO_REF = RHO_VALUES["rho0.90"]
SAVE_EVERY = 10

BUDGET_PCTS = (50, 75, 100)
SQRT_K_VALUES = (2, 4, 6, 8)
BASELINE_FULL = "baseline-rho90"
REPLAY_REF_F1_50 = 0.544
REPLAY_REF_F1_RHO_50 = 0.514


@dataclass(frozen=True)
class GranulationConfig:
    name: str
    family: str
    split_budget: int | None = None
    budget_pct: int | float | None = None
    sqrt_k: float | None = None
    version: str | None = None  # v1 | v2 for adaptive


@dataclass
class GranulationResult:
    kept: list[GranularBall]
    depths: list[int]
    n_successful_splits: int
    n_failed_splits: int
    n_budget_iterations: int
    split_budget: int | None
    budget_met: bool
    n_requeued: int = 0
    n_abandoned_max_attempts: int = 0


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _global_min_label(indexed: np.ndarray) -> float:
    y_all = indexed[:, -2]
    return 1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0


def _ball_entropy(ball: GranularBall, global_min: float) -> float:
    labels = _labels(ball)
    n = int(ball.num)
    n_min = int(np.sum(labels == global_min))
    n_maj = n - n_min
    if n_min <= 0 or n_maj <= 0:
        return 0.0
    return _binary_entropy(n_min / n)


def _splittable(ball: GranularBall, purity_thr: float) -> bool:
    return float(ball.purity) < purity_thr and int(ball.num) > MIN_SAMPLES_PER_BALL


def _children_valid(c1: GranularBall, c2: GranularBall) -> bool:
    return int(c1.num) >= MIN_BALL_COUNT and int(c2.num) >= MIN_BALL_COUNT


def measure_rho90_split_count(indexed: np.ndarray, *, seed: int) -> int:
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    n_splits = 0
    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if _splittable(ball, RHO_REF):
            c1, c2 = ball.split_2balls()
            balls[i] = c1
            balls.append(c2)
            ll += 1
            n_splits += 1
        else:
            i += 1
    return n_splits


def build_baseline_rho90(
    indexed: np.ndarray,
    *,
    seed: int,
    split_budget: int | None = None,
) -> GranulationResult:
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    n_success = 0
    n_iterations = 0
    ll = len(balls)
    i = 0

    while i < ll:
        if split_budget is not None and n_success >= split_budget:
            break
        n_iterations += 1
        ball = balls[i]
        if _splittable(ball, RHO_REF):
            c1, c2 = ball.split_2balls()
            balls[i] = c1
            balls.append(c2)
            depths.append(depths[i] + 1)
            ll += 1
            n_success += 1
        else:
            i += 1

    kept, kept_depths = _finalize_with_depths(balls, depths)
    return GranulationResult(
        kept=kept,
        depths=kept_depths,
        n_successful_splits=n_success,
        n_failed_splits=0,
        n_budget_iterations=n_iterations,
        split_budget=split_budget,
        budget_met=split_budget is None or n_success >= split_budget,
    )


def _adaptive_entropy_core(
    indexed: np.ndarray,
    *,
    seed: int,
    split_budget: int,
    global_min: float,
    rho90_split_count: int,
    preserve_on_failure: bool,
) -> GranulationResult:
    """
    Online entropy-priority granulation.

    v1 (preserve_on_failure=False): permanently removes parent on failed split.
    v2 (preserve_on_failure=True): re-queues parent with
        priority = entropy / (1 + failed_attempt_count), max_attempts=3.
    """
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    split_done = [False]
    failed_attempts = [0]
    counter = 0
    heap: list[tuple[float, int, int]] = []
    relax_level = 0
    n_success = 0
    n_failed = 0
    n_requeued = 0
    n_abandoned = 0
    n_iterations = 0

    def purity_thr() -> float:
        if relax_level <= 0:
            return RHO_REF
        if relax_level == 1:
            return PURITY
        return 1.0

    def effective_priority(ball: GranularBall, idx: int) -> float:
        ent = _ball_entropy(ball, global_min)
        if preserve_on_failure:
            return ent / (1.0 + failed_attempts[idx])
        return ent

    def push_candidate(idx: int) -> None:
        nonlocal counter
        if split_done[idx]:
            return
        ball = balls[idx]
        if not _splittable(ball, purity_thr()):
            return
        pri = effective_priority(ball, idx)
        heapq.heappush(heap, (-pri, counter, idx))
        counter += 1

    def refill_heap() -> bool:
        nonlocal relax_level
        for idx in range(len(balls)):
            push_candidate(idx)
        if heap:
            return True
        max_relax = 2 if split_budget > rho90_split_count else 1
        while relax_level < max_relax:
            relax_level += 1
            for idx in range(len(balls)):
                push_candidate(idx)
            if heap:
                return True
        return False

    refill_heap()

    while n_success < split_budget:
        n_iterations += 1
        if not heap and not refill_heap():
            break

        _, _, idx = heapq.heappop(heap)
        if idx >= len(balls) or split_done[idx]:
            continue

        ball = balls[idx]
        if not _splittable(ball, purity_thr()):
            continue

        c1, c2 = ball.split_2balls()
        if not _children_valid(c1, c2):
            n_failed += 1
            if preserve_on_failure:
                failed_attempts[idx] += 1
                if failed_attempts[idx] < MAX_ATTEMPTS:
                    n_requeued += 1
                    push_candidate(idx)
                else:
                    split_done[idx] = True
                    n_abandoned += 1
            else:
                split_done[idx] = True
            continue

        balls[idx] = c1
        balls.append(c2)
        depths.append(depths[idx] + 1)
        split_done[idx] = True
        split_done.append(False)
        failed_attempts.append(0)
        n_success += 1

        push_candidate(idx)
        push_candidate(len(balls) - 1)

    kept, kept_depths = _finalize_with_depths(balls, depths)
    return GranulationResult(
        kept=kept,
        depths=kept_depths,
        n_successful_splits=n_success,
        n_failed_splits=n_failed,
        n_budget_iterations=n_iterations,
        split_budget=split_budget,
        budget_met=n_success >= split_budget,
        n_requeued=n_requeued,
        n_abandoned_max_attempts=n_abandoned,
    )


def build_adaptive_entropy_v1(
    indexed: np.ndarray,
    *,
    seed: int,
    split_budget: int,
    global_min: float,
    rho90_split_count: int,
) -> GranulationResult:
    return _adaptive_entropy_core(
        indexed,
        seed=seed,
        split_budget=split_budget,
        global_min=global_min,
        rho90_split_count=rho90_split_count,
        preserve_on_failure=False,
    )


def build_adaptive_entropy_v2(
    indexed: np.ndarray,
    *,
    seed: int,
    split_budget: int,
    global_min: float,
    rho90_split_count: int,
) -> GranulationResult:
    return _adaptive_entropy_core(
        indexed,
        seed=seed,
        split_budget=split_budget,
        global_min=global_min,
        rho90_split_count=rho90_split_count,
        preserve_on_failure=True,
    )


def _finalize_with_depths(
    balls: list[GranularBall],
    depths: list[int],
) -> tuple[list[GranularBall], list[int]]:
    kept: list[GranularBall] = []
    kept_depths: list[int] = []
    for ball, depth in zip(balls, depths):
        if int(ball.num) >= MIN_BALL_COUNT:
            kept.append(ball)
            kept_depths.append(depth)
    return kept, kept_depths


def compute_extended_stats(
    kept: list[GranularBall],
    depths: list[int],
    indexed: np.ndarray,
    global_min: float,
) -> dict[str, Any]:
    n_train = int(indexed.shape[0])
    n_min_train = int(np.sum(indexed[:, -2] == global_min))
    n_maj_train = n_train - n_min_train

    survived_min = 0
    survived_maj = 0
    purities: list[float] = []
    radii: list[float] = []
    entropies: list[float] = []
    boundaries: list[float] = []
    mixed = 0

    for ball in kept:
        labs = _labels(ball)
        n = int(ball.num)
        n_min = int(np.sum(labs == global_min))
        n_maj = n - n_min
        survived_min += n_min
        survived_maj += n_maj
        r = float(ball.r)
        radii.append(r)
        purities.append(float(ball.purity))
        ent = _ball_entropy(ball, global_min)
        entropies.append(ent)
        density = n / (r + EPS)
        boundaries.append(ent * density)
        if n_min > 0 and n_maj > 0:
            mixed += 1

    n_balls = len(kept)
    depth_arr = np.asarray(depths, dtype=float) if depths else np.array([0.0])
    return {
        "n_balls": n_balls,
        "minority_survival": survived_min / n_min_train if n_min_train else float("nan"),
        "majority_survival": survived_maj / n_maj_train if n_maj_train else float("nan"),
        "mean_purity": _nanmean(purities),
        "mean_radius": _nanmean(radii),
        "mean_entropy": _nanmean(entropies),
        "mean_boundary_score": _nanmean(boundaries),
        "fraction_mixed_balls": mixed / n_balls if n_balls else float("nan"),
        "mean_depth": float(np.mean(depth_arr)),
        "max_depth": int(np.max(depth_arr)) if depth_arr.size else 0,
        "depth_std": float(np.std(depth_arr)),
    }


def _make_configs(rho90_split_count: int, n_samples: int) -> list[GranulationConfig]:
    configs: list[GranulationConfig] = [
        GranulationConfig(name=BASELINE_FULL, family="baseline_full"),
    ]
    for pct in BUDGET_PCTS:
        budget = max(1, int(round(pct / 100.0 * rho90_split_count)))
        configs.append(
            GranulationConfig(
                name=f"baseline-rho90-{pct}pct",
                family="baseline_budget",
                split_budget=budget,
                budget_pct=pct,
            )
        )
        configs.append(
            GranulationConfig(
                name=f"adaptive-entropy-v1-{pct}pct",
                family="adaptive_pct",
                split_budget=budget,
                budget_pct=pct,
                version="v1",
            )
        )
        configs.append(
            GranulationConfig(
                name=f"adaptive-entropy-v2-{pct}pct",
                family="adaptive_pct",
                split_budget=budget,
                budget_pct=pct,
                version="v2",
            )
        )
    for k in SQRT_K_VALUES:
        budget = max(1, int(round(k * np.sqrt(n_samples))))
        configs.append(
            GranulationConfig(
                name=f"adaptive-entropy-v1-sqrt-k{k}",
                family="adaptive_sqrt",
                split_budget=budget,
                sqrt_k=float(k),
                version="v1",
            )
        )
        configs.append(
            GranulationConfig(
                name=f"adaptive-entropy-v2-sqrt-k{k}",
                family="adaptive_sqrt",
                split_budget=budget,
                sqrt_k=float(k),
                version="v2",
            )
        )
    return configs


def run_granulation(
    indexed: np.ndarray,
    *,
    seed: int,
    config: GranulationConfig,
    global_min: float,
    rho90_split_count: int,
) -> GranulationResult:
    if config.family == "baseline_full":
        return build_baseline_rho90(indexed, seed=seed, split_budget=None)
    if config.family == "baseline_budget":
        return build_baseline_rho90(indexed, seed=seed, split_budget=config.split_budget)
    builder = build_adaptive_entropy_v2 if config.version == "v2" else build_adaptive_entropy_v1
    return builder(
        indexed,
        seed=seed,
        split_budget=int(config.split_budget),
        global_min=global_min,
        rho90_split_count=rho90_split_count,
    )


def run_single(
    split,
    seed: int,
    config: GranulationConfig,
    *,
    indexed: np.ndarray,
    global_min: float,
    rho90_split_count: int,
) -> dict[str, Any]:
    result = run_granulation(
        indexed,
        seed=seed,
        config=config,
        global_min=global_min,
        rho90_split_count=rho90_split_count,
    )
    rep = compute_extended_stats(result.kept, result.depths, indexed, global_min)
    balls = [_to_ball_info(b) for b in result.kept]
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    return {
        "dataset": split.name,
        "seed": seed,
        "config": config.name,
        "family": config.family,
        "version": config.version,
        "budget_pct": config.budget_pct,
        "sqrt_k": config.sqrt_k,
        "rho90_split_count": rho90_split_count,
        "split_budget": result.split_budget,
        "split_count": result.n_successful_splits,
        "n_failed_splits": result.n_failed_splits,
        "n_requeued": result.n_requeued,
        "n_abandoned_max_attempts": result.n_abandoned_max_attempts,
        "n_budget_iterations": result.n_budget_iterations,
        "budget_met": int(result.budget_met),
        "n_balls": rep["n_balls"],
        "minority_survival": rep["minority_survival"],
        "majority_survival": rep["majority_survival"],
        "mean_purity": rep["mean_purity"],
        "mean_radius": rep["mean_radius"],
        "mean_entropy": rep["mean_entropy"],
        "mean_boundary_score": rep["mean_boundary_score"],
        "fraction_mixed_balls": rep["fraction_mixed_balls"],
        "mean_depth": rep["mean_depth"],
        "max_depth": rep["max_depth"],
        "depth_std": rep["depth_std"],
        "accuracy": perf["accuracy"],
        "macro_f1": perf["macro_f1"],
        "minority_f1": perf["minority_f1"],
        "gmean": perf["gmean"],
        "classifier_ok": bool(perf.get("classifier_ok", False)),
    }


def _paired_delta(df: pd.DataFrame, cfg_a: str, cfg_b: str, metric: str) -> float:
    a = df[df["config"] == cfg_a].set_index(["dataset", "seed"])
    b = df[df["config"] == cfg_b].set_index(["dataset", "seed"])
    deltas: list[float] = []
    for key in a.index.intersection(b.index):
        av = float(a.loc[key, metric])
        bv = float(b.loc[key, metric])
        if np.isfinite(av) and np.isfinite(bv):
            deltas.append(av - bv)
    return _nanmean(deltas)


def aggregate_by_config(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in sorted(df["config"].unique()):
        sub = df[df["config"] == cfg]
        if sub.empty:
            continue
        rows.append(
            {
                "config": cfg,
                "family": sub["family"].iloc[0],
                "version": sub["version"].iloc[0] if "version" in sub else None,
                "n_runs": int(len(sub)),
                "mean_split_count": _nanmean(sub["split_count"].tolist()),
                "mean_split_budget": _nanmean(
                    [x for x in sub["split_budget"].tolist() if np.isfinite(x)]
                ),
                "frac_budget_met": _nanmean(sub["budget_met"].tolist()),
                "mean_failed_splits": _nanmean(sub["n_failed_splits"].tolist()),
                "mean_requeued": _nanmean(sub["n_requeued"].tolist()),
                "mean_abandoned": _nanmean(sub["n_abandoned_max_attempts"].tolist()),
                "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
                "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
                "mean_majority_survival": _nanmean(sub["majority_survival"].tolist()),
                "mean_purity": _nanmean(sub["mean_purity"].tolist()),
                "mean_radius": _nanmean(sub["mean_radius"].tolist()),
                "mean_entropy": _nanmean(sub["mean_entropy"].tolist()),
                "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
                "fraction_mixed_balls": _nanmean(sub["fraction_mixed_balls"].tolist()),
                "mean_depth": _nanmean(sub["mean_depth"].tolist()),
                "max_depth_mean": _nanmean(sub["max_depth"].tolist()),
                "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["gmean"].tolist()),
            }
        )
    return rows


def _splits_to_match_rho90(df: pd.DataFrame, aggs: list[dict[str, Any]]) -> dict[str, Any]:
    """Find budget pct where v2 mean F1 is closest to full rho90."""
    by_name = {a["config"]: a for a in aggs}
    base_full = by_name.get(BASELINE_FULL, {})
    target_f1 = base_full.get("mean_macro_f1", float("nan"))
    out: dict[str, Any] = {"target_rho90_f1": target_f1}
    for ver in ("v1", "v2"):
        best_pct = None
        best_gap = float("inf")
        for pct in BUDGET_PCTS:
            cfg = f"adaptive-entropy-{ver}-{pct}pct"
            f1 = by_name.get(cfg, {}).get("mean_macro_f1", float("nan"))
            if np.isfinite(f1) and np.isfinite(target_f1):
                gap = abs(f1 - target_f1)
                if gap < best_gap:
                    best_gap = gap
                    best_pct = pct
        if best_pct is not None:
            cfg = f"adaptive-entropy-{ver}-{best_pct}pct"
            out[ver] = {
                "closest_budget_pct": best_pct,
                "mean_macro_f1": by_name[cfg].get("mean_macro_f1"),
                "mean_split_count": by_name[cfg].get("mean_split_count"),
                "rho90_split_count": by_name[BASELINE_FULL].get("mean_split_count"),
                "split_fraction_of_rho90": (
                    by_name[cfg].get("mean_split_count", 0)
                    / max(by_name[BASELINE_FULL].get("mean_split_count", 1), 1)
                ),
            }
    return out


def answer_questions(df: pd.DataFrame, aggs: list[dict[str, Any]]) -> dict[str, str]:
    by_name = {a["config"]: a for a in aggs}
    base_full = by_name.get(BASELINE_FULL, {})
    v1_50 = by_name.get("adaptive-entropy-v1-50pct", {})
    v2_50 = by_name.get("adaptive-entropy-v2-50pct", {})
    v1_100 = by_name.get("adaptive-entropy-v1-100pct", {})
    v2_100 = by_name.get("adaptive-entropy-v2-100pct", {})
    base_50 = by_name.get("baseline-rho90-50pct", {})
    base_100 = by_name.get("baseline-rho90-100pct", {})

    d_v2_50 = _paired_delta(df, "adaptive-entropy-v2-50pct", "baseline-rho90-50pct", "macro_f1")
    d_v1_50 = _paired_delta(df, "adaptive-entropy-v1-50pct", "baseline-rho90-50pct", "macro_f1")
    d_v2_v1_50 = _paired_delta(df, "adaptive-entropy-v2-50pct", "adaptive-entropy-v1-50pct", "macro_f1")
    d_v2_100 = _paired_delta(df, "adaptive-entropy-v2-100pct", "baseline-rho90-100pct", "macro_f1")

    q1 = (
        f"Replay reference at 50% budget: entropy replay F1={REPLAY_REF_F1_50:.3f} "
        f"vs BFS rho {REPLAY_REF_F1_RHO_50:.3f} (+0.030). "
        f"v1 online: F1={v1_50.get('mean_macro_f1', float('nan')):.3f} "
        f"(budget met {v1_50.get('frac_budget_met', 0):.0%}, "
        f"splits {v1_50.get('mean_split_count', float('nan')):.1f}). "
        f"v2 online: F1={v2_50.get('mean_macro_f1', float('nan')):.3f} "
        f"(budget met {v2_50.get('frac_budget_met', 0):.0%}, "
        f"splits {v2_50.get('mean_split_count', float('nan')):.1f}, "
        f"paired Δ vs v1={d_v2_v1_50:+.4f}). "
    )
    if d_v2_50 > 0.02 and v2_50.get("frac_budget_met", 0) > 0.5:
        q1 += "**Yes** — candidate preservation recovers replay-level performance online."
    elif d_v2_v1_50 > 0.01:
        q1 += "**Partially** — v2 improves over v1 but has not fully closed the replay gap."
    else:
        q1 += "**No** — candidate preservation alone does not recover replay gains."

    q2 = (
        f"At 50% budget: v2 F1={v2_50.get('mean_macro_f1', float('nan')):.3f} "
        f"vs rho {base_50.get('mean_macro_f1', float('nan')):.3f} (Δ={d_v2_50:+.4f}); "
        f"v1 Δ={d_v1_50:+.4f}. "
        f"At 100%: v2={v2_100.get('mean_macro_f1', float('nan')):.3f} "
        f"vs rho {base_100.get('mean_macro_f1', float('nan')):.3f} (Δ={d_v2_100:+.4f}). "
    )
    if d_v2_50 > d_v1_50 and d_v2_50 > 0.005:
        q2 += "**Yes** — entropy priority remains beneficial once candidate paths are preserved."
    elif d_v2_50 > 0:
        q2 += "**Marginal** — entropy helps but advantage is small vs rho at equal budget."
    else:
        q2 += "**No clear benefit** — v2 does not beat rho at matched budgets."

    match = _splits_to_match_rho90(df, aggs)
    v2_match = match.get("v2", {})
    v1_match = match.get("v1", {})
    q3 = (
        f"Full rho90: F1={base_full.get('mean_macro_f1', float('nan')):.3f} "
        f"at {base_full.get('mean_split_count', float('nan')):.0f} splits. "
        f"v2 closest match at {v2_match.get('closest_budget_pct', 'n/a')}% budget: "
        f"F1={v2_match.get('mean_macro_f1', float('nan')):.3f} "
        f"at {v2_match.get('mean_split_count', float('nan')):.0f} splits "
        f"({v2_match.get('split_fraction_of_rho90', 0):.0%} of rho90). "
        f"v1 closest: {v1_match.get('mean_split_count', float('nan')):.0f} splits "
        f"at {v1_match.get('closest_budget_pct', 'n/a')}% budget. "
    )
    if v2_match.get("split_fraction_of_rho90", 1) < 0.8:
        q3 += "**Yes** — v2 matches rho90 F1 with fewer splits than full rho."
    else:
        q3 += "**Not yet** — v2 still requires near-full split budgets to match rho90."

    budget_ok = v2_100.get("frac_budget_met", 0) > 0.8
    q4 = (
        f"v2 at 100% budget: splits={v2_100.get('mean_split_count', float('nan')):.1f}/"
        f"{v2_100.get('mean_split_budget', float('nan')):.0f} "
        f"(met {v2_100.get('frac_budget_met', 0):.0%}), "
        f"requeued={v2_100.get('mean_requeued', 0):.1f}, "
        f"abandoned={v2_100.get('mean_abandoned', 0):.1f}. "
        f"v1 met {v1_100.get('frac_budget_met', 0):.0%} at "
        f"{v1_100.get('mean_split_count', 0):.1f} splits. "
    )
    if budget_ok and d_v2_100 > -0.01:
        q4 += (
            "**Yes** — scheduling + candidate state management resolves premature "
            "branch termination; budget utilization and F1 approach rho90."
        )
    elif d_v2_v1_50 > 0.01 or v2_100.get("mean_split_count", 0) > v1_100.get("mean_split_count", 0) * 1.5:
        q4 += (
            "**Partially** — state management improves split utilization vs v1, "
            "but candidate exhaustion at ρ=0.90 may still limit full-budget runs."
        )
    else:
        q4 += "**No** — scheduling + state management alone does not solve exhaustion."

    return {
        "q1_recover_replay": q1,
        "q2_entropy_beneficial": q2,
        "q3_splits_to_match_rho90": q3,
        "q4_scheduling_solved": q4,
    }


def write_report(path: Path, *, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    aggs = summary["by_config"]
    answers = summary["answers"]
    match = summary.get("splits_to_match_rho90", {})

    lines = [
        "# Adaptive entropy granulation v2",
        "",
        "Online live KMeans + entropy priority. v2 preserves failed-split parents "
        f"(re-queue priority = entropy/(1+attempts), max_attempts={MAX_ATTEMPTS}).",
        "",
        f"Runs: **{summary['n_rows']}**",
        "",
        "## Research questions",
        "",
        "### 1. Does candidate preservation recover replay performance?",
        "",
        answers["q1_recover_replay"],
        "",
        "### 2. Does entropy priority remain beneficial?",
        "",
        answers["q2_entropy_beneficial"],
        "",
        "### 3. How many splits are required to match rho90?",
        "",
        answers["q3_splits_to_match_rho90"],
        "",
        "### 4. Is the problem solved by scheduling + state management?",
        "",
        answers["q4_scheduling_solved"],
        "",
        "## Configuration summary (aggregate means)",
        "",
        "| config | splits | budget | met% | requeued | balls | depth | macro F1 | min F1 | gmean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    key_configs = [
        BASELINE_FULL,
        "baseline-rho90-50pct",
        "baseline-rho90-100pct",
        "adaptive-entropy-v1-50pct",
        "adaptive-entropy-v2-50pct",
        "adaptive-entropy-v1-100pct",
        "adaptive-entropy-v2-100pct",
    ]
    by_name = {a["config"]: a for a in aggs}
    for cfg in key_configs:
        a = by_name.get(cfg)
        if not a:
            continue
        lines.append(
            f"| {a['config']} | {a['mean_split_count']:.1f} | "
            f"{a.get('mean_split_budget', float('nan')):.1f} | "
            f"{a.get('frac_budget_met', float('nan')):.2f} | "
            f"{a.get('mean_requeued', 0):.1f} | {a['mean_n_balls']:.1f} | "
            f"{a['mean_depth']:.1f} | {a['mean_macro_f1']:.3f} | "
            f"{a['mean_minority_f1']:.3f} | {a['mean_gmean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Δ macro F1",
            "",
            "| comparison | Δ |",
            "|---|---:|",
            f"| v2 − rho @ 50% | {_paired_delta(df, 'adaptive-entropy-v2-50pct', 'baseline-rho90-50pct', 'macro_f1'):+.4f} |",
            f"| v1 − rho @ 50% | {_paired_delta(df, 'adaptive-entropy-v1-50pct', 'baseline-rho90-50pct', 'macro_f1'):+.4f} |",
            f"| v2 − v1 @ 50% | {_paired_delta(df, 'adaptive-entropy-v2-50pct', 'adaptive-entropy-v1-50pct', 'macro_f1'):+.4f} |",
            f"| v2 − rho @ 100% | {_paired_delta(df, 'adaptive-entropy-v2-100pct', 'baseline-rho90-100pct', 'macro_f1'):+.4f} |",
            "",
            "## Splits to match full rho90 F1",
            "",
            f"- v2: {match.get('v2', {})}",
            f"- v1: {match.get('v1', {})}",
            "",
            "## Outputs",
            "",
            f"- `{OUT_CSV.name}`",
            f"- `{OUT_SUMMARY.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(df: pd.DataFrame, failed: list[dict[str, str]], smoke: bool) -> dict[str, Any]:
    aggs = aggregate_by_config(df)
    answers = answer_questions(df, aggs)
    return {
        "protocol": {
            "online_granulation": True,
            "v2_candidate_preservation": True,
            "max_attempts": MAX_ATTEMPTS,
            "priority_decay": "entropy / (1 + failed_attempt_count)",
            "smoke_mode": smoke,
            "baseline_full": BASELINE_FULL,
            "budget_pcts": list(BUDGET_PCTS),
            "sqrt_k_values": list(SQRT_K_VALUES),
            "rho_reference": RHO_REF,
            "seeds": SEEDS,
            "classifier": f"Gaussian-weighted GBLSTSVM ({PRIMARY_GAUSSIAN})",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "min_ball_count": MIN_BALL_COUNT,
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "by_config": aggs,
        "splits_to_match_rho90": _splits_to_match_rho90(df, aggs),
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print("SMOKE: 1 dataset × seed 42")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append(
                    {"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue

            try:
                y_train = normalize_labels(split.y_train)
                indexed = _build_indexed_matrix(split.X_train, y_train)
                global_min = _global_min_label(indexed)
                rho90_split_count = measure_rho90_split_count(indexed, seed=seed)
                configs = _make_configs(rho90_split_count, int(indexed.shape[0]))

                for config in configs:
                    row = run_single(
                        split,
                        seed,
                        config,
                        indexed=indexed,
                        global_min=global_min,
                        rho90_split_count=rho90_split_count,
                    )
                    rows.append(row)
                    done += 1

                recent = rows[-len(configs):]
                ref = next(r for r in recent if r["config"] == BASELINE_FULL)
                v1 = next(r for r in recent if r["config"] == "adaptive-entropy-v1-50pct")
                v2 = next(r for r in recent if r["config"] == "adaptive-entropy-v2-50pct")
                print(
                    f"{split.name} s{seed}: rho90 F1={ref['macro_f1']:.3f} "
                    f"splits={ref['split_count']} | "
                    f"v1@50 F1={v1['macro_f1']:.3f} splits={v1['split_count']}/{v1['split_budget']} | "
                    f"v2@50 F1={v2['macro_f1']:.3f} splits={v2['split_count']}/{v2['split_budget']} "
                    f"req={v2['n_requeued']}"
                )

                if done % (SAVE_EVERY * len(configs)) == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    print(f"  [checkpoint] {done} rows")

            except Exception as exc:
                failed.append({"dataset": split.name, "seed": str(seed), "error": str(exc)})
                print(f"FAIL {desc['name']} seed={seed}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
