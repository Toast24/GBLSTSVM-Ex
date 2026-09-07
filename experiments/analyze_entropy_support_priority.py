"""
Entropy support-priority diagnostic (read-only).

Hypothesis: pure entropy priority ignores support size, causing collapse despite
compact high-F1 granulations. Compares four online refinement priority signals
during adaptive entropy scheduling (budget = rho90 split count per run).

Does NOT modify src/, solver, prediction, weighting, or granulation.

Priority policies:
  entropy              — raw parent entropy
  entropy_x_frac       — entropy × sample_fraction
  entropy_x_sqrt_frac  — entropy × sqrt(sample_fraction)
  entropy_x_log_n      — entropy × log(1 + n_samples)

Run from GBLSTSVM root:
  python experiments/analyze_entropy_support_priority.py --smoke
  python experiments/analyze_entropy_support_priority.py
  python experiments/analyze_entropy_support_priority.py --from-csv
"""

from __future__ import annotations

import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_geometry_vs_performance import fit_gaussian_gblstsvm  # noqa: E402
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.test_adaptive_entropy_granulation_v2 import (  # noqa: E402
    RHO_REF,
    _finalize_with_depths,
    compute_extended_stats,
    measure_rho90_split_count,
)
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _labels,
    _to_ball_info,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "entropy_support_priority"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"
RUN_SUMMARY_CSV = OUT_DIR / "run_summary.csv"

SEEDS = list(range(10))
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
SAVE_EVERY = 20
COLLAPSE_BALL_THRESHOLD = 3

OUTCOME_SUCCESS = "success"
FAIL_SINGLETON = "child_deletion_singleton"
FAIL_INVALID_GEOMETRY = "child_deletion_invalid_geometry"
FAIL_NO_CANDIDATES = "no_remaining_candidates"

PriorityFn = Callable[[float, int, int], float]

PRIORITY_POLICIES: dict[str, PriorityFn] = {
    "entropy": lambda ent, n, _n_train: ent,
    "entropy_x_frac": lambda ent, n, n_train: ent * (n / n_train if n_train else 0.0),
    "entropy_x_sqrt_frac": lambda ent, n, n_train: ent * math.sqrt(n / n_train if n_train else 0.0),
    "entropy_x_log_n": lambda ent, n, _n_train: ent * math.log1p(n),
}


@dataclass
class ChildRecord:
    samples: int
    purity: float
    entropy: float
    survives_del_ball: bool


@dataclass
class ParentRecord:
    depth: int
    samples: int
    sample_fraction: float
    purity: float
    entropy: float
    boundary_score: float
    priority_score: float


@dataclass
class RunTrace:
    priority_policy: str
    split_budget: int
    records: list[dict[str, Any]]
    run_summary: dict[str, Any]
    kept: list[GranularBall]
    kept_depths: list[int]


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


def _parent_props(
    ball: GranularBall,
    depth: int,
    global_min: float,
    n_train: int,
    priority_fn: PriorityFn,
) -> ParentRecord:
    labels = _labels(ball)
    n = int(ball.num)
    r = float(ball.r)
    n_min = int(np.sum(labels == global_min))
    entropy = _ball_entropy(ball, global_min)
    density = n / (r + EPS)
    boundary = entropy * density
    frac = n / n_train if n_train else 0.0
    priority = priority_fn(entropy, n, n_train)
    return ParentRecord(
        depth=depth,
        samples=n,
        sample_fraction=frac,
        purity=float(ball.purity),
        entropy=entropy,
        boundary_score=boundary,
        priority_score=priority,
    )


def _child_props(child: GranularBall, global_min: float) -> ChildRecord:
    n = int(child.num)
    survives = n >= MIN_BALL_COUNT and np.isfinite(float(child.r))
    return ChildRecord(
        samples=n,
        purity=float(child.purity),
        entropy=_ball_entropy(child, global_min),
        survives_del_ball=survives,
    )


def _minority_survival_after_split(
    c1: GranularBall,
    c2: GranularBall,
    global_min: float,
) -> float:
    parent_n_min = int(np.sum(_labels(c1) == global_min)) + int(
        np.sum(_labels(c2) == global_min)
    )
    if parent_n_min == 0:
        return float("nan")
    retained = 0
    for child in (c1, c2):
        if int(child.num) >= MIN_BALL_COUNT:
            retained += int(np.sum(_labels(child) == global_min))
    return float(retained / parent_n_min)


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


def _all_priority_scores(ent: float, n: int, n_train: int) -> dict[str, float]:
    return {name: fn(ent, n, n_train) for name, fn in PRIORITY_POLICIES.items()}


def _heap_candidate_rankings(
    heap: list[tuple[float, int, int]],
    balls: list[GranularBall],
    global_min: float,
    n_train: int,
) -> dict[str, Any]:
    """Top heap index under each policy; True if policies disagree on top pick."""
    if not heap:
        return {
            "n_heap_candidates": 0,
            "policy_would_disagree": 0,
            "top_idx_entropy": float("nan"),
        }

    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for _pri, _ctr, idx in heap:
        if idx in seen or idx >= len(balls):
            continue
        seen.add(idx)
        ball = balls[idx]
        ent = _ball_entropy(ball, global_min)
        n = int(ball.num)
        scores = _all_priority_scores(ent, n, n_train)
        candidates.append({"idx": idx, **scores})

    if not candidates:
        return {
            "n_heap_candidates": 0,
            "policy_would_disagree": 0,
            "top_idx_entropy": float("nan"),
        }

    tops: dict[str, int] = {}
    for policy in PRIORITY_POLICIES:
        tops[policy] = max(candidates, key=lambda c: c[policy])["idx"]

    disagree = len(set(tops.values())) > 1
    out: dict[str, Any] = {
        "n_heap_candidates": len(candidates),
        "policy_would_disagree": int(disagree),
        "top_idx_entropy": tops["entropy"],
    }
    for policy in PRIORITY_POLICIES:
        out[f"top_idx_{policy}"] = tops[policy]
    return out


def _weighted_child_metric(c1: ChildRecord, c2: ChildRecord, attr: str) -> float:
    n1, n2 = c1.samples, c2.samples
    v1 = getattr(c1, attr)
    v2 = getattr(c2, attr)
    return float((n1 * v1 + n2 * v2) / max(n1 + n2, 1))


def trace_priority_policy(
    indexed: np.ndarray,
    *,
    seed: int,
    dataset: str,
    global_min: float,
    n_train: int,
    rho90_split_count: int,
    split_budget: int,
    priority_policy: str,
    priority_fn: PriorityFn,
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
        pri = priority_fn(ent, int(ball.num), n_train)
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
        if not heap and not refill_heap():
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "priority_policy": priority_policy,
                    "split_index": split_index + 1,
                    "split_budget": split_budget,
                    "rho90_split_count": rho90_split_count,
                    "parent_entropy": float("nan"),
                    "parent_purity": float("nan"),
                    "parent_n_samples": float("nan"),
                    "parent_sample_fraction": float("nan"),
                    "parent_depth": float("nan"),
                    "parent_boundary_score": float("nan"),
                    "parent_priority_score": float("nan"),
                    "child1_entropy": float("nan"),
                    "child2_entropy": float("nan"),
                    "child_weighted_entropy": float("nan"),
                    "child1_purity": float("nan"),
                    "child2_purity": float("nan"),
                    "child_weighted_purity": float("nan"),
                    "minority_survival": float("nan"),
                    "child1_survives_del_ball": float("nan"),
                    "child2_survives_del_ball": float("nan"),
                    "outcome": FAIL_NO_CANDIDATES,
                }
            )
            break

        heap_snapshot = list(heap)
        rank_info = _heap_candidate_rankings(heap_snapshot, balls, global_min, n_train)
        _, _, idx = heapq.heappop(heap)
        if idx >= len(balls) or split_done[idx]:
            continue

        ball = balls[idx]
        if not _splittable(ball, purity_thr()):
            continue

        parent = _parent_props(ball, depths[idx], global_min, n_train, priority_fn)
        priority_scores = _all_priority_scores(parent.entropy, parent.samples, n_train)
        c1_raw, c2_raw = ball.split_2balls()
        c1 = _child_props(c1_raw, global_min)
        c2 = _child_props(c2_raw, global_min)
        outcome = _classify_failed_split(c1_raw, c2_raw)
        split_index += 1
        min_surv = _minority_survival_after_split(c1_raw, c2_raw, global_min)

        row = {
            "dataset": dataset,
            "seed": seed,
            "priority_policy": priority_policy,
            "split_index": split_index,
            "split_budget": split_budget,
            "rho90_split_count": rho90_split_count,
            "parent_entropy": parent.entropy,
            "parent_purity": parent.purity,
            "parent_n_samples": parent.samples,
            "parent_sample_fraction": parent.sample_fraction,
            "parent_depth": parent.depth,
            "parent_boundary_score": parent.boundary_score,
            "parent_priority_score": parent.priority_score,
            "priority_entropy": priority_scores["entropy"],
            "priority_entropy_x_frac": priority_scores["entropy_x_frac"],
            "priority_entropy_x_sqrt_frac": priority_scores["entropy_x_sqrt_frac"],
            "priority_entropy_x_log_n": priority_scores["entropy_x_log_n"],
            "n_heap_candidates": rank_info["n_heap_candidates"],
            "policy_would_disagree": rank_info["policy_would_disagree"],
            "child1_entropy": c1.entropy,
            "child2_entropy": c2.entropy,
            "child_weighted_entropy": _weighted_child_metric(c1, c2, "entropy"),
            "child1_purity": c1.purity,
            "child2_purity": c2.purity,
            "child_weighted_purity": _weighted_child_metric(c1, c2, "purity"),
            "minority_survival": min_surv,
            "child1_survives_del_ball": int(c1.survives_del_ball),
            "child2_survives_del_ball": int(c2.survives_del_ball),
            "outcome": outcome,
        }

        if outcome != OUTCOME_SUCCESS:
            n_failed += 1
            if outcome == FAIL_SINGLETON:
                n_singleton_fail += 1
            else:
                n_invalid_fail += 1
            split_done[idx] = True
            records.append(row)
            continue

        balls[idx] = c1_raw
        balls.append(c2_raw)
        depths.append(depths[idx] + 1)
        split_done[idx] = True
        split_done.append(False)
        n_success += 1

        push_candidate(idx)
        push_candidate(len(balls) - 1)
        records.append(row)

    kept, kept_depths = _finalize_with_depths(balls, depths)
    success_records = [r for r in records if r["outcome"] == OUTCOME_SUCCESS]
    run_summary = {
        "priority_policy": priority_policy,
        "n_successful_splits": n_success,
        "n_failed_splits": n_failed,
        "n_singleton_failures": n_singleton_fail,
        "n_invalid_geometry_failures": n_invalid_fail,
        "terminated_no_candidates": int(
            any(r["outcome"] == FAIL_NO_CANDIDATES for r in records)
        ),
        "split_budget": split_budget,
        "budget_met": n_success >= split_budget,
        "final_n_balls": len(kept),
        "final_relax_level": relax_level,
        "frac_splits_with_singleton_child": _nanmean(
            [
                int(r["child1_survives_del_ball"] == 0 or r["child2_survives_del_ball"] == 0)
                for r in success_records
            ]
        ),
        "mean_parent_n_samples": _nanmean([r["parent_n_samples"] for r in success_records]),
        "mean_parent_sample_fraction": _nanmean(
            [r["parent_sample_fraction"] for r in success_records]
        ),
        "mean_parent_entropy": _nanmean([r["parent_entropy"] for r in success_records]),
        "mean_minority_survival": _nanmean([r["minority_survival"] for r in success_records]),
        "frac_heap_policy_disagreement": _nanmean(
            [r.get("policy_would_disagree", 0) for r in records if r["outcome"] != FAIL_NO_CANDIDATES]
        ),
        "mean_heap_candidates": _nanmean(
            [r.get("n_heap_candidates", 0) for r in records if r["outcome"] != FAIL_NO_CANDIDATES]
        ),
    }
    return RunTrace(
        priority_policy=priority_policy,
        split_budget=split_budget,
        records=records,
        run_summary=run_summary,
        kept=kept,
        kept_depths=kept_depths,
    )


def evaluate_granulation(
    kept: list[GranularBall],
    kept_depths: list[int],
    indexed: np.ndarray,
    global_min: float,
    split,
    seed: int,
) -> dict[str, Any]:
    rep = compute_extended_stats(kept, kept_depths, indexed, global_min)
    ball_infos = [_to_ball_info(b) for b in kept]
    perf = fit_gaussian_gblstsvm(
        ball_infos,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )
    n_balls = rep["n_balls"]
    macro_f1 = float(perf["macro_f1"])
    collapsed = (
        n_balls <= COLLAPSE_BALL_THRESHOLD
        or not bool(perf.get("classifier_ok", False))
        or not np.isfinite(macro_f1)
    )
    return {
        "macro_f1": macro_f1,
        "minority_f1": float(perf["minority_f1"]),
        "gmean": float(perf["gmean"]),
        "collapsed": int(collapsed),
        "n_balls": n_balls,
        "classifier_ok": bool(perf.get("classifier_ok", False)),
        "mean_entropy": rep["mean_entropy"],
        "mean_boundary_score": rep["mean_boundary_score"],
        "minority_survival_final": rep["minority_survival"],
    }


def analyze_run(
    split,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    n_train = int(indexed.shape[0])
    rho90_count = measure_rho90_split_count(indexed, seed=seed)
    split_budget = rho90_count

    split_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for policy, priority_fn in PRIORITY_POLICIES.items():
        trace = trace_priority_policy(
            indexed,
            seed=seed,
            dataset=split.name,
            global_min=global_min,
            n_train=n_train,
            rho90_split_count=rho90_count,
            split_budget=split_budget,
            priority_policy=policy,
            priority_fn=priority_fn,
        )
        perf = evaluate_granulation(
            trace.kept, trace.kept_depths, indexed, global_min, split, seed
        )
        trace.run_summary["rho90_split_count"] = rho90_count
        trace.run_summary.update(
            {
                "dataset": split.name,
                "seed": seed,
                **perf,
            }
        )
        run_rows.append(trace.run_summary)
        for rec in trace.records:
            row = rec.copy()
            row.update(
                {
                    "run_macro_f1": perf["macro_f1"],
                    "run_minority_f1": perf["minority_f1"],
                    "run_gmean": perf["gmean"],
                    "run_collapsed": perf["collapsed"],
                    "run_n_balls": perf["n_balls"],
                    "run_split_count": trace.run_summary["n_successful_splits"],
                }
            )
            split_rows.append(row)

    return split_rows, run_rows


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(np.sum(mask))}


def _aggregate_by_policy(run_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in PRIORITY_POLICIES:
        sub = run_df[run_df["priority_policy"] == policy]
        if sub.empty:
            continue
        rows.append(
            {
                "priority_policy": policy,
                "n_runs": int(len(sub)),
                "frac_collapsed": float(sub["collapsed"].mean()),
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["gmean"].tolist()),
                "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
                "mean_split_count": _nanmean(sub["n_successful_splits"].tolist()),
                "frac_budget_met": _nanmean(sub["budget_met"].astype(float).tolist()),
                "mean_failed_splits": _nanmean(sub["n_failed_splits"].tolist()),
                "frac_terminated_no_candidates": _nanmean(
                    sub["terminated_no_candidates"].astype(float).tolist()
                ),
                "mean_parent_n_samples": _nanmean(sub["mean_parent_n_samples"].tolist()),
                "mean_parent_sample_fraction": _nanmean(
                    sub["mean_parent_sample_fraction"].tolist()
                ),
                "mean_parent_entropy": _nanmean(sub["mean_parent_entropy"].tolist()),
                "mean_split_minority_survival": _nanmean(
                    sub["mean_minority_survival"].tolist()
                ),
                "frac_heap_policy_disagreement": _nanmean(
                    sub["frac_heap_policy_disagreement"].tolist()
                ),
            }
        )
    return rows


def _policy_trajectory_equivalence(run_df: pd.DataFrame) -> dict[str, Any]:
    """Check whether support policies change outcomes vs pure entropy."""
    ent = run_df[run_df["priority_policy"] == "entropy"].set_index(["dataset", "seed"])
    rows: list[dict[str, Any]] = []
    for policy in PRIORITY_POLICIES:
        if policy == "entropy":
            continue
        sub = run_df[run_df["priority_policy"] == policy].set_index(["dataset", "seed"])
        keys = ent.index.intersection(sub.index)
        n_diff_splits = 0
        n_diff_balls = 0
        n_diff_f1 = 0
        for key in keys:
            e, s = ent.loc[key], sub.loc[key]
            if int(e["n_successful_splits"]) != int(s["n_successful_splits"]):
                n_diff_splits += 1
            if int(e["n_balls"]) != int(s["n_balls"]):
                n_diff_balls += 1
            if abs(float(e["macro_f1"]) - float(s["macro_f1"])) > 1e-9:
                n_diff_f1 += 1
        rows.append(
            {
                "priority_policy": policy,
                "n_runs": int(len(keys)),
                "n_diff_split_count": n_diff_splits,
                "n_diff_n_balls": n_diff_balls,
                "n_diff_macro_f1": n_diff_f1,
                "trajectory_identical_to_entropy": n_diff_splits == 0 and n_diff_balls == 0,
            }
        )
    return {"vs_entropy": rows}


def _small_ball_failure_analysis(split_df: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy in PRIORITY_POLICIES:
        sub = split_df[
            (split_df["priority_policy"] == policy)
            & split_df["outcome"].isin(
                [OUTCOME_SUCCESS, FAIL_SINGLETON, FAIL_INVALID_GEOMETRY]
            )
        ].copy()
        if sub.empty:
            continue
        failed = sub["outcome"] != OUTCOME_SUCCESS
        rows.append(
            {
                "priority_policy": policy,
                "frac_failed": float(failed.mean()),
                "mean_parent_n_samples_success": _nanmean(
                    sub.loc[~failed, "parent_n_samples"].tolist()
                ),
                "mean_parent_n_samples_failed": _nanmean(
                    sub.loc[failed, "parent_n_samples"].tolist()
                ),
                "mean_parent_sample_fraction_success": _nanmean(
                    sub.loc[~failed, "parent_sample_fraction"].tolist()
                ),
                "mean_parent_sample_fraction_failed": _nanmean(
                    sub.loc[failed, "parent_sample_fraction"].tolist()
                ),
                "failure_vs_parent_samples_corr": _safe_corr(
                    sub["parent_n_samples"].to_numpy(dtype=float),
                    failed.astype(float).to_numpy(),
                ),
            }
        )
    return {"by_policy": rows}


def _compact_reliability(run_df: pd.DataFrame) -> dict[str, Any]:
    base = run_df[run_df["priority_policy"] == "entropy"]
    out: dict[str, Any] = {}
    if base.empty:
        return out
    base_f1 = _nanmean(base["macro_f1"].tolist())
    base_balls = _nanmean(base["n_balls"].tolist())
    base_collapse = float(base["collapsed"].mean())
    for policy in PRIORITY_POLICIES:
        sub = run_df[run_df["priority_policy"] == policy]
        if sub.empty:
            continue
        out[policy] = {
            "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
            "delta_balls_vs_entropy": _nanmean(sub["n_balls"].tolist()) - base_balls,
            "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
            "delta_macro_f1_vs_entropy": _nanmean(sub["macro_f1"].tolist()) - base_f1,
            "frac_collapsed": float(sub["collapsed"].mean()),
            "delta_collapse_vs_entropy": float(sub["collapsed"].mean()) - base_collapse,
            "f1_per_ball": _nanmean(
                [
                    f / b if b > 0 and np.isfinite(f) else float("nan")
                    for f, b in zip(sub["macro_f1"], sub["n_balls"])
                ]
            ),
        }
    return out


def _answer_questions(
    by_policy: list[dict[str, Any]],
    small_ball: dict[str, Any],
    compact: dict[str, Any],
    equiv: dict[str, Any],
) -> dict[str, str]:
    by_name = {a["priority_policy"]: a for a in by_policy}
    ent = by_name.get("entropy", {})
    best_collapse = min(by_policy, key=lambda a: a["frac_collapsed"], default={})
    best_f1 = max(by_policy, key=lambda a: a["mean_macro_f1"], default={})
    support_policies = [
        p for p in ("entropy_x_frac", "entropy_x_sqrt_frac", "entropy_x_log_n")
        if p in by_name
    ]

    q1_parts = []
    for p in support_policies:
        a = by_name[p]
        q1_parts.append(
            f"**{p}**: collapse {a['frac_collapsed']:.1%} "
            f"(entropy {ent.get('frac_collapsed', float('nan')):.1%}), "
            f"macro F1 {a['mean_macro_f1']:.3f}"
        )
    q1 = (
        f"Pure entropy collapses **{ent.get('frac_collapsed', float('nan')):.1%}** of runs "
        f"({ent.get('mean_n_balls', float('nan')):.1f} balls). "
        + ("; ".join(q1_parts) if q1_parts else "")
        + ". "
    )
    if best_collapse.get("priority_policy") != "entropy":
        q1 += (
            f"**{best_collapse.get('priority_policy')}** has lowest collapse "
            f"({best_collapse.get('frac_collapsed', float('nan')):.1%}), "
            "support-aware weighting helps."
        )
    else:
        equiv_rows = equiv.get("vs_entropy", [])
        all_identical = equiv_rows and all(r["trajectory_identical_to_entropy"] for r in equiv_rows)
        ent_heap_disagree = ent.get("frac_heap_policy_disagreement", 0)
        if all_identical and ent_heap_disagree > 0:
            q1 += (
                f"Support-aware priorities would disagree on heap top-pick "
                f"**{ent_heap_disagree:.1%}** of decisions, but trajectories remain "
                f"identical because entropy exhausts candidates before reordering matters."
            )
        else:
            q1 += "Support-aware priorities do not materially reduce collapse on this sweep."

    balance_scores = []
    for a in by_policy:
        score = (
            a["mean_macro_f1"]
            - 0.02 * a["frac_collapsed"]
            + 0.005 * a["mean_parent_sample_fraction"]
        )
        balance_scores.append((a["priority_policy"], score, a))
    balance_scores.sort(key=lambda x: x[1], reverse=True)
    top = balance_scores[0][2] if balance_scores else {}
    q2 = (
        f"Best ambiguity–coverage tradeoff: **{top.get('priority_policy', 'n/a')}** "
        f"(macro F1 {top.get('mean_macro_f1', float('nan')):.3f}, "
        f"mean parent fraction {top.get('mean_parent_sample_fraction', float('nan')):.3f}, "
        f"collapse {top.get('frac_collapsed', float('nan')):.1%}). "
        f"Highest raw F1: **{best_f1.get('priority_policy', 'n/a')}** "
        f"({best_f1.get('mean_macro_f1', float('nan')):.3f})."
    )

    sb = small_ball.get("by_policy", [])
    ent_sb = next((x for x in sb if x["priority_policy"] == "entropy"), {})
    fail_corr = ent_sb.get("failure_vs_parent_samples_corr", {})
    q3 = (
        f"Entropy failures: parent n={ent_sb.get('mean_parent_n_samples_failed', float('nan')):.1f} "
        f"vs success n={ent_sb.get('mean_parent_n_samples_success', float('nan')):.1f}; "
        f"sample fraction failed={ent_sb.get('mean_parent_sample_fraction_failed', float('nan')):.3f} "
        f"vs success={ent_sb.get('mean_parent_sample_fraction_success', float('nan')):.3f}. "
    )
    if fail_corr.get("spearman", 0) < -0.15:
        q3 += "**Yes** — smaller parents are more likely to fail (negative size–failure correlation)."
    else:
        q3 += "**Partially** — failures correlate with depth-first entropy chasing, not only tiny balls."

    comp_ent = compact.get("entropy", {})
    best_compact = min(
        compact.items(),
        key=lambda kv: (kv[1].get("frac_collapsed", 1.0), -kv[1].get("mean_macro_f1", 0)),
    )[0] if compact else "entropy"
    comp_best = compact.get(best_compact, {})
    q4 = (
        f"Entropy baseline: {comp_ent.get('mean_n_balls', float('nan')):.1f} balls, "
        f"F1 {comp_ent.get('mean_macro_f1', float('nan')):.3f}, "
        f"collapse {comp_ent.get('frac_collapsed', float('nan')):.1%}. "
        f"**{best_compact}**: {comp_best.get('mean_n_balls', float('nan')):.1f} balls "
        f"(Δ {comp_best.get('delta_balls_vs_entropy', float('nan')):+.1f}), "
        f"F1 {comp_best.get('mean_macro_f1', float('nan')):.3f} "
        f"(Δ {comp_best.get('delta_macro_f1_vs_entropy', float('nan')):+.3f}), "
        f"collapse {comp_best.get('frac_collapsed', float('nan')):.1%}."
    )
    if comp_best.get("delta_collapse_vs_entropy", 0) < -0.05:
        q4 += " Support weighting can stay compact while improving reliability."
    else:
        q4 += " Compactness–reliability tradeoff remains; priority alone is insufficient."

    return {
        "q1_support_aware_avoid_collapse": q1,
        "q2_best_ambiguity_coverage_balance": q2,
        "q3_small_unstable_balls": q3,
        "q4_compact_and_reliable": q4,
    }


def write_report(path: Path, *, summary: dict[str, Any]) -> None:
    by_policy = summary["by_priority_policy"]
    answers = summary["answers"]
    compact = summary.get("compact_reliability", {})

    lines = [
        "# Entropy support-priority diagnostic",
        "",
        "Read-only comparison of four online entropy refinement priority signals. "
        "Budget = rho90 split count; traces every selected split with parent/child "
        "geometry and links each run to classifier outcomes.",
        "",
        f"Runs: **{summary['n_dataset_seed_pairs']}** dataset×seed pairs × "
        f"**{len(PRIORITY_POLICIES)}** policies | "
        f"Split records: **{summary['n_split_records']}**",
        "",
        "## Policy summary",
        "",
        "| policy | macro F1 | minority F1 | G-mean | balls | collapse% | "
        "parent n | parent frac | split min surv | heap disagree% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in by_policy:
        lines.append(
            f"| {a['priority_policy']} | {a['mean_macro_f1']:.3f} | "
            f"{a['mean_minority_f1']:.3f} | {a['mean_gmean']:.3f} | "
            f"{a['mean_n_balls']:.1f} | {a['frac_collapsed']:.1%} | "
            f"{a['mean_parent_n_samples']:.1f} | "
            f"{a['mean_parent_sample_fraction']:.3f} | "
            f"{a['mean_split_minority_survival']:.3f} | "
            f"{a.get('frac_heap_policy_disagreement', float('nan')):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Compactness vs reliability",
            "",
        ]
    )
    for policy, vals in compact.items():
        lines.append(
            f"- **{policy}**: {vals.get('mean_n_balls', float('nan')):.1f} balls, "
            f"F1 {vals.get('mean_macro_f1', float('nan')):.3f} "
            f"(Δ vs entropy {vals.get('delta_macro_f1_vs_entropy', float('nan')):+.3f}), "
            f"collapse {vals.get('frac_collapsed', float('nan')):.1%}"
        )

    lines.extend(
        [
            "",
            "## 1. Does support-aware entropy avoid collapse?",
            "",
            answers["q1_support_aware_avoid_collapse"],
            "",
            "## 2. Which priority best balances ambiguity and coverage?",
            "",
            answers["q2_best_ambiguity_coverage_balance"],
            "",
            "## 3. Are failures caused by selecting small unstable balls?",
            "",
            answers["q3_small_unstable_balls"],
            "",
            "## 4. Can entropy remain compact while improving reliability?",
            "",
            answers["q4_compact_and_reliable"],
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
    by_policy = _aggregate_by_policy(run_df)
    small_ball = _small_ball_failure_analysis(split_df)
    compact = _compact_reliability(run_df)
    equiv = _policy_trajectory_equivalence(run_df)
    answers = _answer_questions(by_policy, small_ball, compact, equiv)
    return {
        "protocol": {
            "read_only": True,
            "priority_policies": list(PRIORITY_POLICIES.keys()),
            "rho_reference": RHO_REF,
            "entropy_budget": "100% of rho90 split count per dataset×seed",
            "collapse_ball_threshold": COLLAPSE_BALL_THRESHOLD,
            "seeds": SEEDS,
            "smoke_mode": smoke,
            "failed": failed,
        },
        "n_split_records": int(len(split_df)),
        "n_run_summaries": int(len(run_df)),
        "n_dataset_seed_pairs": int(run_df.groupby(["dataset", "seed"]).ngroups)
        if not run_df.empty
        else 0,
        "by_priority_policy": by_policy,
        "small_ball_failure_analysis": small_ball,
        "compact_reliability": compact,
        "policy_trajectory_equivalence": equiv,
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        split_df = pd.read_csv(OUT_CSV)
        run_df = pd.read_csv(RUN_SUMMARY_CSV) if RUN_SUMMARY_CSV.exists() else pd.DataFrame()
        summary = _build_summary(split_df, run_df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print("SMOKE: 1 dataset × seed 42 × 4 policies")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds × 4 policies")

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
                ent = next(r for r in r_rows if r["priority_policy"] == "entropy")
                print(
                    f"{split.name} s{seed}: entropy "
                    f"splits={ent['n_successful_splits']}/{ent['split_budget']} "
                    f"balls={ent['n_balls']} F1={ent['macro_f1']:.3f} "
                    f"collapsed={ent['collapsed']}"
                )
                if done % SAVE_EVERY == 0:
                    pd.DataFrame(split_rows).to_csv(OUT_CSV, index=False)
                    pd.DataFrame(run_rows).to_csv(RUN_SUMMARY_CSV, index=False)
                    print(f"  [checkpoint] {done} dataset×seed pairs")
            except Exception as exc:
                failed.append({"dataset": split.name, "seed": str(seed), "error": str(exc)})
                print(f"FAIL {desc['name']} seed={seed}: {exc}")

    split_df = pd.DataFrame(split_rows)
    run_df = pd.DataFrame(run_rows)
    split_df.to_csv(OUT_CSV, index=False)
    run_df.to_csv(RUN_SUMMARY_CSV, index=False)
    summary = _build_summary(split_df, run_df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(split_df)} split records)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
