"""
Entropy scheduler failure diagnostic (read-only).

Traces every attempted split under rho=0.90 BFS (baseline) vs online
entropy-priority scheduling. Records parent/child geometry, del_ball survival,
candidate-ball exhaustion, and classifies adaptive failures.

Does NOT modify src/, solver, prediction, weighting, or baseline granulation.

Run from GBLSTSVM root:
  python experiments/analyze_entropy_scheduler_failure.py --smoke
  python experiments/analyze_entropy_scheduler_failure.py
  python experiments/analyze_entropy_scheduler_failure.py --from-csv
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
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_geometry_vs_performance import RHO_VALUES  # noqa: E402
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _labels,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "entropy_scheduler_failure"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
RHO_REF = RHO_VALUES["rho0.90"]
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
SAVE_EVERY = 5

SCHED_RHO90 = "rho90"
SCHED_ENTROPY = "adaptive_entropy"

OUTCOME_SUCCESS = "success"
FAIL_SINGLETON = "child_deletion_singleton"
FAIL_INVALID_GEOMETRY = "child_deletion_invalid_geometry"
FAIL_NO_CANDIDATES = "no_remaining_candidates"


@dataclass
class ChildRecord:
    samples: int
    purity: float
    entropy: float
    radius: float
    survives_del_ball: bool


@dataclass
class ParentRecord:
    depth: int
    samples: int
    purity: float
    entropy: float
    radius: float
    minority_ratio: float
    boundary_score: float


@dataclass
class RunTrace:
    scheduler: str
    split_budget: int | None
    records: list[dict[str, Any]]
    run_summary: dict[str, Any]


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


def _parent_props(ball: GranularBall, depth: int, global_min: float) -> ParentRecord:
    labels = _labels(ball)
    n = int(ball.num)
    r = float(ball.r)
    n_min = int(np.sum(labels == global_min))
    minority_ratio = n_min / n if n else 0.0
    entropy = _ball_entropy(ball, global_min)
    density = n / (r + EPS)
    boundary = entropy * density
    return ParentRecord(
        depth=depth,
        samples=n,
        purity=float(ball.purity),
        entropy=entropy,
        radius=r,
        minority_ratio=minority_ratio,
        boundary_score=boundary,
    )


def _child_props(child: GranularBall, global_min: float) -> ChildRecord:
    n = int(child.num)
    r = float(child.r)
    survives = n >= MIN_BALL_COUNT and np.isfinite(r)
    return ChildRecord(
        samples=n,
        purity=float(child.purity),
        entropy=_ball_entropy(child, global_min),
        radius=r,
        survives_del_ball=survives,
    )


def _splittable(ball: GranularBall, purity_thr: float) -> bool:
    return float(ball.purity) < purity_thr and int(ball.num) > MIN_SAMPLES_PER_BALL


def _invalid_geometry(c1: GranularBall, c2: GranularBall) -> bool:
    for child in (c1, c2):
        n = int(child.num)
        r = float(child.r)
        if n == 0:
            return True
        if not np.isfinite(r) or not np.all(np.isfinite(child.center)):
            return True
    return False


def _classify_failed_split(c1: GranularBall, c2: GranularBall) -> str:
    if _invalid_geometry(c1, c2):
        return FAIL_INVALID_GEOMETRY
    if int(c1.num) < MIN_BALL_COUNT or int(c2.num) < MIN_BALL_COUNT:
        return FAIL_SINGLETON
    return OUTCOME_SUCCESS


def _count_splittable(
    balls: list[GranularBall],
    depths: list[int],
    split_done: list[bool] | None,
    purity_thr: float,
) -> int:
    n = 0
    for idx, ball in enumerate(balls):
        if split_done is not None and split_done[idx]:
            continue
        if _splittable(ball, purity_thr):
            n += 1
    return n


def _depth_histogram(depths: list[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for d in depths:
        key = str(int(d))
        hist[key] = hist.get(key, 0) + 1
    return hist


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


def _base_row(
    *,
    dataset: str,
    seed: int,
    scheduler: str,
    split_index: int,
    parent: ParentRecord,
    c1: ChildRecord,
    c2: ChildRecord,
    outcome: str,
    n_candidates_after: int,
    n_splittable_after: int,
    heap_size_after: int,
    relax_level: int,
    split_budget: int | None,
    rho90_split_count: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": seed,
        "scheduler": scheduler,
        "split_index": split_index,
        "split_budget": split_budget,
        "rho90_split_count": rho90_split_count,
        "parent_depth": parent.depth,
        "parent_samples": parent.samples,
        "parent_purity": parent.purity,
        "parent_entropy": parent.entropy,
        "parent_radius": parent.radius,
        "parent_minority_ratio": parent.minority_ratio,
        "parent_boundary_score": parent.boundary_score,
        "child1_samples": c1.samples,
        "child1_purity": c1.purity,
        "child1_entropy": c1.entropy,
        "child1_radius": c1.radius,
        "child1_survives_del_ball": int(c1.survives_del_ball),
        "child2_samples": c2.samples,
        "child2_purity": c2.purity,
        "child2_entropy": c2.entropy,
        "child2_radius": c2.radius,
        "child2_survives_del_ball": int(c2.survives_del_ball),
        "either_child_singleton": int(c1.samples == 1 or c2.samples == 1),
        "both_children_survive_del_ball": int(c1.survives_del_ball and c2.survives_del_ball),
        "outcome": outcome,
        "n_candidates_after": n_candidates_after,
        "n_splittable_after": n_splittable_after,
        "heap_size_after": heap_size_after,
        "relax_level": relax_level,
    }


def trace_rho90(
    indexed: np.ndarray,
    *,
    seed: int,
    dataset: str,
    global_min: float,
    rho90_split_count: int,
    split_budget: int | None = None,
) -> RunTrace:
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    records: list[dict[str, Any]] = []
    split_index = 0
    n_success = 0

    ll = len(balls)
    i = 0
    while i < ll:
        if split_budget is not None and n_success >= split_budget:
            break
        ball = balls[i]
        depth = depths[i]
        if not _splittable(ball, RHO_REF):
            i += 1
            continue

        parent = _parent_props(ball, depth, global_min)
        c1_raw, c2_raw = ball.split_2balls()
        c1 = _child_props(c1_raw, global_min)
        c2 = _child_props(c2_raw, global_min)

        balls[i] = c1_raw
        balls.append(c2_raw)
        depths.append(depth + 1)
        ll += 1
        n_success += 1
        split_index += 1

        n_splittable = _count_splittable(balls, depths, None, RHO_REF)
        records.append(
            _base_row(
                dataset=dataset,
                seed=seed,
                scheduler=SCHED_RHO90,
                split_index=split_index,
                parent=parent,
                c1=c1,
                c2=c2,
                outcome=OUTCOME_SUCCESS,
                n_candidates_after=n_splittable,
                n_splittable_after=n_splittable,
                heap_size_after=0,
                relax_level=0,
                split_budget=split_budget,
                rho90_split_count=rho90_split_count,
            )
        )

    kept_depths = [d for ball, d in zip(balls, depths) if int(ball.num) >= MIN_BALL_COUNT]
    run_summary = {
        "scheduler": SCHED_RHO90,
        "n_successful_splits": n_success,
        "n_failed_splits": 0,
        "n_singleton_failures": 0,
        "n_invalid_geometry_failures": 0,
        "terminated_no_candidates": 0,
        "split_budget": split_budget,
        "budget_met": split_budget is None or n_success >= split_budget,
        "final_n_balls": len(kept_depths),
        "final_n_splittable": _count_splittable(balls, depths, None, RHO_REF),
        "mean_depth": _nanmean(kept_depths),
        "max_depth": int(max(kept_depths)) if kept_depths else 0,
        "depth_histogram": _depth_histogram(kept_depths),
        "frac_splits_with_singleton_child": _nanmean(
            [r["either_child_singleton"] for r in records]
        ),
        "frac_splits_both_survive_del_ball": _nanmean(
            [r["both_children_survive_del_ball"] for r in records]
        ),
    }
    return RunTrace(
        scheduler=SCHED_RHO90,
        split_budget=split_budget,
        records=records,
        run_summary=run_summary,
    )


def trace_adaptive_entropy(
    indexed: np.ndarray,
    *,
    seed: int,
    dataset: str,
    global_min: float,
    rho90_split_count: int,
    split_budget: int,
) -> RunTrace:
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    split_done = [False]
    counter = 0
    heap: list[tuple[float, int, int]] = []
    relax_level = 0
    records: list[dict[str, Any]] = []
    split_index = 0
    n_success = 0
    n_failed = 0
    n_singleton_fail = 0
    n_invalid_fail = 0

    def purity_thr() -> float:
        if relax_level <= 0:
            return RHO_REF
        if relax_level == 1:
            return PURITY
        return 1.0

    def push_candidate(idx: int) -> None:
        nonlocal counter
        if split_done[idx]:
            return
        ball = balls[idx]
        if not _splittable(ball, purity_thr()):
            return
        ent = _ball_entropy(ball, global_min)
        heapq.heappush(heap, (-ent, counter, idx))
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

    def post_split_counts() -> tuple[int, int, int]:
        n_splittable = _count_splittable(balls, depths, split_done, purity_thr())
        return len(heap), n_splittable, n_splittable

    refill_heap()

    while n_success < split_budget:
        if not heap and not refill_heap():
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "scheduler": SCHED_ENTROPY,
                    "split_index": split_index + 1,
                    "split_budget": split_budget,
                    "rho90_split_count": rho90_split_count,
                    "parent_depth": float("nan"),
                    "parent_samples": float("nan"),
                    "parent_purity": float("nan"),
                    "parent_entropy": float("nan"),
                    "parent_radius": float("nan"),
                    "parent_minority_ratio": float("nan"),
                    "parent_boundary_score": float("nan"),
                    "child1_samples": float("nan"),
                    "child1_purity": float("nan"),
                    "child1_entropy": float("nan"),
                    "child1_radius": float("nan"),
                    "child1_survives_del_ball": float("nan"),
                    "child2_samples": float("nan"),
                    "child2_purity": float("nan"),
                    "child2_entropy": float("nan"),
                    "child2_radius": float("nan"),
                    "child2_survives_del_ball": float("nan"),
                    "either_child_singleton": float("nan"),
                    "both_children_survive_del_ball": float("nan"),
                    "outcome": FAIL_NO_CANDIDATES,
                    "n_candidates_after": 0,
                    "n_splittable_after": _count_splittable(
                        balls, depths, split_done, purity_thr()
                    ),
                    "heap_size_after": 0,
                    "relax_level": relax_level,
                }
            )
            break

        _, _, idx = heapq.heappop(heap)
        if idx >= len(balls) or split_done[idx]:
            continue

        ball = balls[idx]
        if not _splittable(ball, purity_thr()):
            continue

        parent = _parent_props(ball, depths[idx], global_min)
        c1_raw, c2_raw = ball.split_2balls()
        c1 = _child_props(c1_raw, global_min)
        c2 = _child_props(c2_raw, global_min)
        outcome = _classify_failed_split(c1_raw, c2_raw)
        split_index += 1

        if outcome != OUTCOME_SUCCESS:
            n_failed += 1
            if outcome == FAIL_SINGLETON:
                n_singleton_fail += 1
            else:
                n_invalid_fail += 1
            split_done[idx] = True
            heap_sz, n_splittable, n_candidates = post_split_counts()
            records.append(
                _base_row(
                    dataset=dataset,
                    seed=seed,
                    scheduler=SCHED_ENTROPY,
                    split_index=split_index,
                    parent=parent,
                    c1=c1,
                    c2=c2,
                    outcome=outcome,
                    n_candidates_after=n_candidates,
                    n_splittable_after=n_splittable,
                    heap_size_after=heap_sz,
                    relax_level=relax_level,
                    split_budget=split_budget,
                    rho90_split_count=rho90_split_count,
                )
            )
            continue

        balls[idx] = c1_raw
        balls.append(c2_raw)
        depths.append(depths[idx] + 1)
        split_done[idx] = True
        split_done.append(False)
        n_success += 1

        push_candidate(idx)
        push_candidate(len(balls) - 1)
        heap_sz, n_splittable, n_candidates = post_split_counts()
        records.append(
            _base_row(
                dataset=dataset,
                seed=seed,
                scheduler=SCHED_ENTROPY,
                split_index=split_index,
                parent=parent,
                c1=c1,
                c2=c2,
                outcome=OUTCOME_SUCCESS,
                n_candidates_after=n_candidates,
                n_splittable_after=n_splittable,
                heap_size_after=heap_sz,
                relax_level=relax_level,
                split_budget=split_budget,
                rho90_split_count=rho90_split_count,
            )
        )

    kept_depths = [
        d for ball, d in zip(balls, depths) if int(ball.num) >= MIN_BALL_COUNT
    ]
    success_records = [r for r in records if r["outcome"] == OUTCOME_SUCCESS]
    run_summary = {
        "scheduler": SCHED_ENTROPY,
        "n_successful_splits": n_success,
        "n_failed_splits": n_failed,
        "n_singleton_failures": n_singleton_fail,
        "n_invalid_geometry_failures": n_invalid_fail,
        "terminated_no_candidates": int(
            any(r["outcome"] == FAIL_NO_CANDIDATES for r in records)
        ),
        "split_budget": split_budget,
        "budget_met": n_success >= split_budget,
        "final_n_balls": len(kept_depths),
        "final_n_splittable": _count_splittable(
            balls, depths, split_done, purity_thr()
        ),
        "mean_depth": _nanmean(kept_depths),
        "max_depth": int(max(kept_depths)) if kept_depths else 0,
        "depth_histogram": _depth_histogram(kept_depths),
        "frac_splits_with_singleton_child": _nanmean(
            [r["either_child_singleton"] for r in success_records]
        ),
        "frac_splits_both_survive_del_ball": _nanmean(
            [r["both_children_survive_del_ball"] for r in success_records]
        ),
        "final_relax_level": relax_level,
    }
    return RunTrace(
        scheduler=SCHED_ENTROPY,
        split_budget=split_budget,
        records=records,
        run_summary=run_summary,
    )


def analyze_run(
    split,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    rho90_count = measure_rho90_split_count(indexed, seed=seed)

    rho_trace = trace_rho90(
        indexed,
        seed=seed,
        dataset=split.name,
        global_min=global_min,
        rho90_split_count=rho90_count,
        split_budget=None,
    )
    entropy_trace = trace_adaptive_entropy(
        indexed,
        seed=seed,
        dataset=split.name,
        global_min=global_min,
        rho90_split_count=rho90_count,
        split_budget=rho90_count,
    )

    run_rows: list[dict[str, Any]] = []
    for trace in (rho_trace, entropy_trace):
        summary = trace.run_summary.copy()
        summary.update(
            {
                "dataset": split.name,
                "seed": seed,
                "rho90_split_count": rho90_count,
            }
        )
        run_rows.append(summary)

    split_rows: list[dict[str, Any]] = []
    split_rows.extend(rho_trace.records)
    split_rows.extend(entropy_trace.records)
    return split_rows, run_rows


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(np.sum(mask))}


def _aggregate_by_scheduler(run_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sched in (SCHED_RHO90, SCHED_ENTROPY):
        sub = run_df[run_df["scheduler"] == sched]
        if sub.empty:
            continue
        rows.append(
            {
                "scheduler": sched,
                "n_runs": int(len(sub)),
                "mean_successful_splits": _nanmean(sub["n_successful_splits"].tolist()),
                "mean_failed_splits": _nanmean(sub["n_failed_splits"].tolist()),
                "mean_singleton_failures": _nanmean(sub["n_singleton_failures"].tolist()),
                "mean_invalid_geometry_failures": _nanmean(
                    sub["n_invalid_geometry_failures"].tolist()
                ),
                "frac_terminated_no_candidates": _nanmean(
                    sub["terminated_no_candidates"].tolist()
                ),
                "frac_budget_met": _nanmean(sub["budget_met"].astype(float).tolist()),
                "mean_final_n_balls": _nanmean(sub["final_n_balls"].tolist()),
                "mean_final_n_splittable": _nanmean(sub["final_n_splittable"].tolist()),
                "mean_depth": _nanmean(sub["mean_depth"].tolist()),
                "mean_max_depth": _nanmean(sub["max_depth"].tolist()),
                "mean_frac_singleton_child": _nanmean(
                    sub["frac_splits_with_singleton_child"].tolist()
                ),
                "mean_frac_both_survive": _nanmean(
                    sub["frac_splits_both_survive_del_ball"].tolist()
                ),
            }
        )
    return rows


def _failure_breakdown(split_df: pd.DataFrame) -> dict[str, Any]:
    ent = split_df[split_df["scheduler"] == SCHED_ENTROPY]
    if ent.empty:
        return {}
    counts = ent["outcome"].value_counts().to_dict()
    total = len(ent)
    return {
        "n_records": total,
        "outcome_counts": counts,
        "frac_success": counts.get(OUTCOME_SUCCESS, 0) / total,
        "frac_singleton": counts.get(FAIL_SINGLETON, 0) / total,
        "frac_invalid_geometry": counts.get(FAIL_INVALID_GEOMETRY, 0) / total,
        "frac_no_candidates": counts.get(FAIL_NO_CANDIDATES, 0) / total,
    }


def _compare_parent_geometry(split_df: pd.DataFrame) -> dict[str, Any]:
    ent = split_df[
        (split_df["scheduler"] == SCHED_ENTROPY)
        & split_df["outcome"].isin([OUTCOME_SUCCESS, FAIL_SINGLETON, FAIL_INVALID_GEOMETRY])
    ].copy()
    if ent.empty:
        return {}

    success = ent[ent["outcome"] == OUTCOME_SUCCESS]
    failed = ent[ent["outcome"] != OUTCOME_SUCCESS]
    metrics = [
        "parent_depth",
        "parent_samples",
        "parent_purity",
        "parent_entropy",
        "parent_radius",
        "parent_minority_ratio",
        "parent_boundary_score",
    ]
    out: dict[str, Any] = {}
    for m in metrics:
        out[m] = {
            "mean_success": _nanmean(success[m].tolist()) if not success.empty else float("nan"),
            "mean_failed": _nanmean(failed[m].tolist()) if not failed.empty else float("nan"),
        }
    out["failed_parent_depth_vs_success_corr"] = _safe_corr(
        ent["parent_depth"].to_numpy(dtype=float),
        (ent["outcome"] != OUTCOME_SUCCESS).astype(float),
    )
    out["failed_parent_entropy_vs_success_corr"] = _safe_corr(
        ent["parent_entropy"].to_numpy(dtype=float),
        (ent["outcome"] != OUTCOME_SUCCESS).astype(float),
    )
    return out


def _candidate_trajectory(split_df: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (dataset, seed, scheduler), grp in split_df.groupby(
        ["dataset", "seed", "scheduler"]
    ):
        succ = grp[grp["outcome"] == OUTCOME_SUCCESS].sort_values("split_index")
        if succ.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "scheduler": scheduler,
                "n_successful_splits": int(len(succ)),
                "candidates_after_first_split": float(succ.iloc[0]["n_candidates_after"]),
                "candidates_after_last_split": float(succ.iloc[-1]["n_candidates_after"]),
                "splittable_after_last_split": float(succ.iloc[-1]["n_splittable_after"]),
            }
        )
    if not rows:
        return {}
    traj = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    for sched in (SCHED_RHO90, SCHED_ENTROPY):
        sub = traj[traj["scheduler"] == sched]
        if sub.empty:
            continue
        out[sched] = {
            "mean_successful_splits": _nanmean(sub["n_successful_splits"].tolist()),
            "mean_candidates_after_first": _nanmean(
                sub["candidates_after_first_split"].tolist()
            ),
            "mean_candidates_after_last": _nanmean(
                sub["candidates_after_last_split"].tolist()
            ),
            "mean_splittable_after_last": _nanmean(
                sub["splittable_after_last_split"].tolist()
            ),
        }
    return out


def _depth_comparison(run_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sched in (SCHED_RHO90, SCHED_ENTROPY):
        sub = run_df[run_df["scheduler"] == sched]
        if sub.empty:
            continue
        out[sched] = {
            "mean_depth": _nanmean(sub["mean_depth"].tolist()),
            "mean_max_depth": _nanmean(sub["max_depth"].tolist()),
        }
    return out


def _del_ball_impact(split_df: pd.DataFrame) -> dict[str, Any]:
    rho = split_df[split_df["scheduler"] == SCHED_RHO90]
    ent_succ = split_df[
        (split_df["scheduler"] == SCHED_ENTROPY) & (split_df["outcome"] == OUTCOME_SUCCESS)
    ]
    if rho.empty:
        return {}
    return {
        "rho90_frac_singleton_child": _nanmean(rho["either_child_singleton"].tolist()),
        "rho90_frac_both_survive": _nanmean(rho["both_children_survive_del_ball"].tolist()),
        "entropy_frac_singleton_child": _nanmean(ent_succ["either_child_singleton"].tolist())
        if not ent_succ.empty
        else float("nan"),
        "entropy_frac_both_survive": _nanmean(ent_succ["both_children_survive_del_ball"].tolist())
        if not ent_succ.empty
        else float("nan"),
        "rho90_accepts_singleton_splits": True,
        "entropy_rejects_singleton_splits": True,
    }


def _answer_questions(
    run_df: pd.DataFrame,
    split_df: pd.DataFrame,
    by_sched: list[dict[str, Any]],
    failure: dict[str, Any],
    parent_geom: dict[str, Any],
    candidates: dict[str, Any],
    depth_cmp: dict[str, Any],
    del_ball: dict[str, Any],
) -> dict[str, str]:
    by_name = {a["scheduler"]: a for a in by_sched}
    rho = by_name.get(SCHED_RHO90, {})
    ent = by_name.get(SCHED_ENTROPY, {})

    q1 = (
        f"Online entropy stops after **{ent.get('mean_successful_splits', float('nan')):.1f}** "
        f"splits on average vs rho90 **{rho.get('mean_successful_splits', float('nan')):.1f}** "
        f"(budget = rho90 count). "
        f"**{ent.get('frac_terminated_no_candidates', 0) * 100:.0f}%** of runs hit "
        f"`no_remaining_candidates` before budget is met; only "
        f"**{ent.get('frac_budget_met', 0) * 100:.0f}%** meet budget. "
        f"Singleton failures contribute **{ent.get('mean_singleton_failures', 0):.2f}** "
        f"failed splits/run vs **{ent.get('mean_failed_splits', 0):.2f}** total failures. "
    )
    if ent.get("frac_terminated_no_candidates", 0) > 0.5:
        q1 += (
            "**Primary cause: candidate exhaustion** — entropy-priority deep-splits "
            "high-entropy branches until no ball remains below ρ=0.90, not split rejection."
        )
    else:
        q1 += "**Mixed** — both candidate exhaustion and split rejection play a role."

    ent_depth = parent_geom.get("parent_depth", {})
    ent_entropy = parent_geom.get("parent_entropy", {})
    q2 = (
        f"Failed adaptive splits have parent depth "
        f"{ent_depth.get('mean_failed', float('nan')):.2f} vs success "
        f"{ent_depth.get('mean_depth', ent_depth.get('mean_success', float('nan'))):.2f}, "
        f"parent entropy {ent_entropy.get('mean_failed', float('nan')):.3f} vs "
        f"{ent_entropy.get('mean_success', float('nan')):.3f}. "
    )
    depth_corr = parent_geom.get("failed_parent_depth_vs_success_corr", {})
    if depth_corr.get("spearman", 0) > 0.2:
        q2 += (
            "**Yes — entropy selects deeper, smaller parents** that are more likely to "
            "produce singleton children and terminate refinement paths on failure."
        )
    else:
        q2 += (
            "**Partially** — failures are rare; instability comes mainly from exhausting "
            "shallow-wide candidate pools after prioritising high-entropy mixed balls."
        )

    q3 = (
        f"rho90 accepts splits with singleton children in "
        f"**{del_ball.get('rho90_frac_singleton_child', float('nan')) * 100:.1f}%** of splits; "
        f"entropy rejects them pre-commit (only "
        f"**{del_ball.get('entropy_frac_singleton_child', float('nan')) * 100:.1f}%** "
        f"of its successful splits produce singleton children). "
        f"rho90 both-survive rate: **{del_ball.get('rho90_frac_both_survive', float('nan')) * 100:.1f}%**; "
        f"entropy: **{del_ball.get('entropy_frac_both_survive', float('nan')) * 100:.1f}%**. "
    )
    if del_ball.get("rho90_frac_singleton_child", 0) > 0.3:
        q3 += (
            "**Yes — del_ball would remove refinement paths** that rho90 preserves by "
            "accepting singleton-producing splits; adaptive pre-rejection plus `split_done` "
            "on failure permanently removes branches."
        )
    else:
        q3 += "**Minor role** for del_ball on this dataset mix; exhaustion dominates."

    ent_cand = candidates.get(SCHED_ENTROPY, {})
    q4a = (
        f"After last successful split, entropy leaves "
        f"**{ent_cand.get('mean_splittable_after_last', float('nan')):.1f}** splittable balls "
        f"(heap candidates **{ent_cand.get('mean_candidates_after_last', float('nan')):.1f}**). "
        "**Candidate preservation** — do not mark failed-split parents `split_done`; "
        "re-queue siblings and relax purity only after genuine exhaustion."
    )
    q4b = (
        f"Singleton rejection blocks "
        f"**{failure.get('frac_singleton', 0) * 100:.1f}%** of entropy split attempts. "
        "**Softer deletion** — accept singleton splits like rho90 and defer to del_ball, "
        "or retry with alternate split geometry before abandoning the parent."
    )
    q4c = (
        f"Entropy mean depth **{depth_cmp.get(SCHED_ENTROPY, {}).get('mean_depth', float('nan')):.2f}** "
        f"vs rho90 **{depth_cmp.get(SCHED_RHO90, {}).get('mean_depth', float('nan')):.2f}**; "
        f"max depth {depth_cmp.get(SCHED_ENTROPY, {}).get('mean_max_depth', float('nan')):.1f} "
        f"vs {depth_cmp.get(SCHED_RHO90, {}).get('mean_max_depth', float('nan')):.1f}. "
        "**Different split selection** alone is insufficient — BFS breadth-first maintains "
        "more parallel refinement paths; pure entropy depth-first exhausts candidates early. "
        "Combine entropy priority with breadth caps or minimum parent-size guards."
    )

    return {
        "q1_why_stops_early": q1,
        "q2_unstable_branches": q2,
        "q3_del_ball_removal": q3,
        "q4a_candidate_preservation": q4a,
        "q4b_softer_deletion": q4b,
        "q4c_split_selection": q4c,
    }


def write_report(path: Path, *, summary: dict[str, Any]) -> None:
    by_sched = summary["by_scheduler"]
    answers = summary["answers"]
    failure = summary["failure_breakdown"]
    candidates = summary["candidate_trajectory"]

    lines = [
        "# Entropy scheduler failure diagnostic",
        "",
        "Read-only comparison of rho=0.90 BFS vs online entropy-priority scheduling. "
        "Traces every attempted split, child del_ball survival, and candidate exhaustion.",
        "",
        f"Runs: **{summary['n_dataset_seed_pairs']}** dataset×seed pairs | "
        f"Split records: **{summary['n_split_records']}**",
        "",
        "## Scheduler summary",
        "",
        "| scheduler | success splits | failed | singleton fail | no-cand stop | "
        "budget met | final balls | mean depth | max depth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in by_sched:
        lines.append(
            f"| {a['scheduler']} | {a['mean_successful_splits']:.1f} | "
            f"{a['mean_failed_splits']:.2f} | {a['mean_singleton_failures']:.2f} | "
            f"{a['frac_terminated_no_candidates']:.2f} | {a['frac_budget_met']:.2f} | "
            f"{a['mean_final_n_balls']:.1f} | {a['mean_depth']:.2f} | "
            f"{a['mean_max_depth']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Adaptive failure breakdown (split records)",
            "",
            f"- success: {failure.get('frac_success', float('nan')):.1%}",
            f"- singleton deletion: {failure.get('frac_singleton', float('nan')):.1%}",
            f"- invalid geometry: {failure.get('frac_invalid_geometry', float('nan')):.1%}",
            f"- no remaining candidates: {failure.get('frac_no_candidates', float('nan')):.1%}",
            "",
            "## Candidate trajectory",
            "",
        ]
    )
    for sched, vals in candidates.items():
        lines.append(
            f"- **{sched}**: after last split — "
            f"candidates={vals.get('mean_candidates_after_last', float('nan')):.1f}, "
            f"splittable={vals.get('mean_splittable_after_last', float('nan')):.1f}"
        )

    lines.extend(
        [
            "",
            "## 1. Why does online entropy stop after few splits?",
            "",
            answers["q1_why_stops_early"],
            "",
            "## 2. Is entropy selecting unstable branches?",
            "",
            answers["q2_unstable_branches"],
            "",
            "## 3. Is del_ball removing useful refinement paths?",
            "",
            answers["q3_del_ball_removal"],
            "",
            "## 4. What minimal change is required?",
            "",
            "### a) Candidate preservation",
            "",
            answers["q4a_candidate_preservation"],
            "",
            "### b) Softer deletion",
            "",
            answers["q4b_softer_deletion"],
            "",
            "### c) Different split selection",
            "",
            answers["q4c_split_selection"],
            "",
            "## Outputs",
            "",
            f"- `{OUT_CSV.name}`",
            f"- `{OUT_SUMMARY.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(
    split_df: pd.DataFrame,
    run_df: pd.DataFrame,
    failed: list[dict[str, str]],
    smoke: bool,
) -> dict[str, Any]:
    by_sched = _aggregate_by_scheduler(run_df)
    failure = _failure_breakdown(split_df)
    parent_geom = _compare_parent_geometry(split_df)
    candidates = _candidate_trajectory(split_df)
    depth_cmp = _depth_comparison(run_df)
    del_ball = _del_ball_impact(split_df)
    answers = _answer_questions(
        run_df, split_df, by_sched, failure, parent_geom, candidates, depth_cmp, del_ball
    )
    return {
        "protocol": {
            "read_only": True,
            "schedulers": [SCHED_RHO90, SCHED_ENTROPY],
            "rho_reference": RHO_REF,
            "min_ball_count_del_ball": MIN_BALL_COUNT,
            "entropy_budget": "100% of rho90 split count per dataset×seed",
            "seeds": SEEDS,
            "smoke_mode": smoke,
            "failed": failed,
        },
        "n_split_records": int(len(split_df)),
        "n_run_summaries": int(len(run_df)),
        "n_dataset_seed_pairs": int(run_df.groupby(["dataset", "seed"]).ngroups)
        if not run_df.empty
        else 0,
        "by_scheduler": by_sched,
        "failure_breakdown": failure,
        "parent_geometry_success_vs_failed": parent_geom,
        "candidate_trajectory": candidates,
        "depth_comparison": depth_cmp,
        "del_ball_impact": del_ball,
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        split_df = pd.read_csv(OUT_CSV)
        run_path = OUT_DIR / "run_summary.csv"
        run_df = pd.read_csv(run_path) if run_path.exists() else pd.DataFrame()
        summary = _build_summary(split_df, run_df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print("SMOKE: 1 dataset × seed 42")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds")

    split_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
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
                s_rows, r_rows = analyze_run(split, seed)
                split_rows.extend(s_rows)
                run_rows.extend(r_rows)
                done += 1
                rho = next(r for r in r_rows if r["scheduler"] == SCHED_RHO90)
                ent = next(r for r in r_rows if r["scheduler"] == SCHED_ENTROPY)
                print(
                    f"{split.name} s{seed}: rho90 splits={rho['n_successful_splits']} "
                    f"depth={rho['mean_depth']:.1f} | "
                    f"entropy splits={ent['n_successful_splits']}/{ent['split_budget']} "
                    f"failed={ent['n_failed_splits']} "
                    f"no_cand={ent['terminated_no_candidates']}"
                )
                if done % SAVE_EVERY == 0:
                    pd.DataFrame(split_rows).to_csv(OUT_CSV, index=False)
                    pd.DataFrame(run_rows).to_csv(OUT_DIR / "run_summary.csv", index=False)
                    print(f"  [checkpoint] {done} runs")
            except Exception as exc:
                failed.append({"dataset": split.name, "seed": str(seed), "error": str(exc)})
                print(f"FAIL {desc['name']} seed={seed}: {exc}")

    split_df = pd.DataFrame(split_rows)
    run_df = pd.DataFrame(run_rows)
    split_df.to_csv(OUT_CSV, index=False)
    run_df.to_csv(OUT_DIR / "run_summary.csv", index=False)
    summary = _build_summary(split_df, run_df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(split_df)} split records)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
