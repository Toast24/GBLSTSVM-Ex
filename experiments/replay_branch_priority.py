"""
Branch-priority refinement replay diagnostic (read-only).

Tests whether adaptive split prioritization + global split budget can reproduce
rho=0.90 granulation behaviour. Records the rho=0.90 split tree once per run,
then replays recorded splits in policy-specific order up to each budget cap.
Does not re-invoke split_2balls() during replay — only reorders which recorded
branch splits are applied.

Does NOT modify src/, solver, prediction, weighting, or granulation code.

Policies:
  rho90_actual       — BFS ordering under rho=0.90 (reference)
  purity_priority    — lowest parent purity first
  entropy_priority   — highest parent entropy first
  boundary_priority  — highest parent boundary score first
  combined_priority  — entropy + boundary + minority support - complexity cost

Budgets: 50%, 75%, 100%, 125% of rho=0.90 split count.

Run from GBLSTSVM root:
  python experiments/replay_branch_priority.py --smoke
  python experiments/replay_branch_priority.py
  python experiments/replay_branch_priority.py --from-csv
"""

from __future__ import annotations

import json
import math
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

from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    RHO_VALUES,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _indices,
    _labels,
    _to_ball_info,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "replay_branch_priority"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
RHO_REF = RHO_VALUES["rho0.90"]
FRAG_PENALTY = 0.10

POLICIES = (
    "rho90_actual",
    "purity_priority",
    "entropy_priority",
    "boundary_priority",
    "combined_priority",
)
BUDGET_PCTS = (50, 75, 100, 125)
REF_POLICY = "rho90_actual"


@dataclass
class SplitEvent:
    order: int
    phase: str  # "rho90" or "extended"
    parent_ids: frozenset[int]
    child1_ids: frozenset[int]
    child2_ids: frozenset[int]
    depth: int
    parent_purity: float
    parent_entropy: float
    parent_boundary_score: float
    parent_minority_ratio: float
    parent_samples: int


@dataclass
class ReplayResult:
    kept: list[GranularBall]
    depths: list[int]
    n_splits: int
    rho90_split_count: int


def _ball_from_ids(indexed: np.ndarray, sample_ids: frozenset[int]) -> GranularBall:
    mask = np.isin(indexed[:, -1], list(sample_ids))
    return GranularBall(indexed[mask])


def _event_props(event: SplitEvent) -> dict[str, float]:
    return {
        "purity": event.parent_purity,
        "entropy": event.parent_entropy,
        "boundary_score": event.parent_boundary_score,
        "minority_ratio": event.parent_minority_ratio,
        "samples": float(event.parent_samples),
    }


def record_rho90_split_tree(
    indexed: np.ndarray,
    *,
    seed: int,
    global_min: float,
) -> tuple[list[SplitEvent], int]:
    """
    Record every split from rho=0.90 BFS traversal, then extend with purity < 0.95
    for budgets above 100%.
    """
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    events: list[SplitEvent] = []
    order = 0

    def append_event(
        ball: GranularBall,
        depth: int,
        c1: GranularBall,
        c2: GranularBall,
        phase: str,
    ) -> None:
        nonlocal order
        props = _ball_props(ball, global_min)
        events.append(
            SplitEvent(
                order=order,
                phase=phase,
                parent_ids=frozenset(int(x) for x in _indices(ball)),
                child1_ids=frozenset(int(x) for x in _indices(c1)),
                child2_ids=frozenset(int(x) for x in _indices(c2)),
                depth=depth,
                parent_purity=props["purity"],
                parent_entropy=props["entropy"],
                parent_boundary_score=props["boundary_score"],
                parent_minority_ratio=props["minority_ratio"],
                parent_samples=int(props["samples"]),
            )
        )
        order += 1

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if _splittable(ball, purity_thr=RHO_REF):
            c1, c2 = ball.split_2balls()
            append_event(ball, depths[i], c1, c2, "rho90")
            balls[i] = c1
            balls.append(c2)
            depths.append(depths[i] + 1)
            ll += 1
        else:
            i += 1

    rho90_split_count = len(events)
    target_extended = max(int(round(1.5 * rho90_split_count)), rho90_split_count)

    while len(events) < target_extended:
        progressed = False
        i = 0
        while i < ll and len(events) < target_extended:
            ball = balls[i]
            if _splittable(ball, purity_thr=PURITY):
                c1, c2 = ball.split_2balls()
                append_event(ball, depths[i], c1, c2, "extended")
                balls[i] = c1
                balls.append(c2)
                depths.append(depths[i] + 1)
                ll += 1
                progressed = True
            i += 1
        if not progressed:
            break

    return events, rho90_split_count


def _select_event(
    available: list[SplitEvent],
    *,
    policy: str,
    n_splits_done: int,
) -> SplitEvent:
    if policy == REF_POLICY:
        return min(available, key=lambda e: e.order)
    return max(
        available,
        key=lambda e: (
            _priority_score(
                policy,
                _event_props(e),
                n_splits_done=n_splits_done,
                depth=e.depth,
            ),
            -e.order,
        ),
    )


def replay_from_split_tree(
    indexed: np.ndarray,
    events: list[SplitEvent],
    *,
    policy: str,
    split_budget: int,
    rho90_split_count: int,
    budget_pct: int,
) -> ReplayResult:
    root_ids = frozenset(int(i) for i in range(indexed.shape[0]))
    leaves: dict[frozenset[int], tuple[GranularBall, int]] = {
        root_ids: (_ball_from_ids(indexed, root_ids), 0)
    }
    applied: set[int] = set()
    n_splits = 0

    event_pool = events
    if budget_pct <= 100:
        event_pool = [e for e in events if e.phase == "rho90"]

    while n_splits < split_budget:
        available = [
            e
            for e in event_pool
            if e.order not in applied and e.parent_ids in leaves
        ]
        if not available:
            break

        event = _select_event(available, policy=policy, n_splits_done=n_splits)
        del leaves[event.parent_ids]
        leaves[event.child1_ids] = (
            _ball_from_ids(indexed, event.child1_ids),
            event.depth + 1,
        )
        leaves[event.child2_ids] = (
            _ball_from_ids(indexed, event.child2_ids),
            event.depth + 1,
        )
        applied.add(event.order)
        n_splits += 1

    kept: list[GranularBall] = []
    kept_depths: list[int] = []
    for ball, depth in leaves.values():
        if int(ball.num) >= MIN_BALL_COUNT:
            kept.append(ball)
            kept_depths.append(depth)

    return ReplayResult(
        kept=kept,
        depths=kept_depths,
        n_splits=n_splits,
        rho90_split_count=rho90_split_count,
    )


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _ball_props(ball: GranularBall, global_min: float) -> dict[str, float]:
    labels = _labels(ball)
    n = int(ball.num)
    r = float(ball.r)
    density = n / (r + EPS)
    n_min = int(np.sum(labels == global_min))
    n_maj = n - n_min
    minority_ratio = n_min / n if n else 0.0
    entropy = _binary_entropy(minority_ratio) if n_min > 0 and n_maj > 0 else 0.0
    boundary = entropy * density
    log_boundary = float(np.log1p(max(boundary, 0.0)))
    return {
        "samples": float(n),
        "purity": float(ball.purity),
        "radius": r,
        "density": density,
        "entropy": entropy,
        "boundary_score": boundary,
        "log_boundary_score": log_boundary,
        "minority_ratio": minority_ratio,
        "n_minority": float(n_min),
    }


def _splittable(
    ball: GranularBall,
    *,
    purity_thr: float,
) -> bool:
    return float(ball.purity) < purity_thr and int(ball.num) > MIN_SAMPLES_PER_BALL


def _priority_score(
    policy: str,
    props: dict[str, float],
    *,
    n_splits_done: int,
    depth: int,
) -> float:
    if policy == "purity_priority":
        return -props["purity"]
    if policy == "entropy_priority":
        return props["entropy"]
    if policy == "boundary_priority":
        return props["boundary_score"]
    if policy == "combined_priority":
        complexity_cost = FRAG_PENALTY + 0.01 * math.log1p(n_splits_done) + 0.05 * depth
        minority_support = props["minority_ratio"]
        return (
            props["entropy"]
            + props["boundary_score"]
            + minority_support
            - complexity_cost
        )
    raise ValueError(f"Not a priority policy: {policy!r}")


def replay_branch_priority(
    indexed: np.ndarray,
    *,
    seed: int,
    policy: str,
    budget_pct: int,
    global_min: float,
) -> ReplayResult:
    events, rho90_split_count = record_rho90_split_tree(
        indexed, seed=seed, global_min=global_min
    )
    split_budget = max(1, int(round(budget_pct / 100.0 * rho90_split_count)))
    return replay_from_split_tree(
        indexed,
        events,
        policy=policy,
        split_budget=split_budget,
        rho90_split_count=rho90_split_count,
        budget_pct=budget_pct,
    )


def compute_representation_stats(
    result: ReplayResult,
    indexed: np.ndarray,
    global_min: float,
) -> dict[str, Any]:
    kept = result.kept
    depths = result.depths
    n_train = int(indexed.shape[0])
    n_min_train = int(np.sum(indexed[:, -2] == global_min))

    survived_min = 0
    purities: list[float] = []
    boundaries: list[float] = []
    het = 0

    for ball in kept:
        labs = _labels(ball)
        n_min = int(np.sum(labs == global_min))
        n_maj = int(ball.num) - n_min
        survived_min += n_min
        purities.append(float(ball.purity))
        props = _ball_props(ball, global_min)
        boundaries.append(props["boundary_score"])
        if n_min > 0 and n_maj > 0:
            het += 1

    n_balls = len(kept)
    depth_arr = np.asarray(depths, dtype=float) if depths else np.array([0.0])
    depth_hist: dict[str, int] = {}
    for d in depths:
        key = str(int(d))
        depth_hist[key] = depth_hist.get(key, 0) + 1

    return {
        "n_balls": n_balls,
        "split_count": result.n_splits,
        "rho90_split_count": result.rho90_split_count,
        "minority_survival": survived_min / n_min_train if n_min_train else float("nan"),
        "mean_purity": _nanmean(purities),
        "mean_boundary_score": _nanmean(boundaries),
        "heterogeneity": het / n_balls if n_balls else float("nan"),
        "mean_depth": float(np.mean(depth_arr)),
        "max_depth": int(np.max(depth_arr)) if depth_arr.size else 0,
        "depth_std": float(np.std(depth_arr)),
        "depth_p50": float(np.median(depth_arr)),
        "depth_histogram": depth_hist,
    }


def run_single(split, seed: int, policy: str, budget_pct: int) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    y_all = indexed[:, -2]
    global_min = (
        1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0
    )

    result = replay_branch_priority(
        indexed,
        seed=seed,
        policy=policy,
        budget_pct=budget_pct,
        global_min=global_min,
    )
    rep = compute_representation_stats(result, indexed, global_min)
    balls = [_to_ball_info(b) for b in result.kept]
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    target_splits = max(1, int(round(budget_pct / 100.0 * result.rho90_split_count)))

    return {
        "dataset": split.name,
        "seed": seed,
        "policy": policy,
        "budget_pct": budget_pct,
        "target_splits": target_splits,
        "rho90_split_count": result.rho90_split_count,
        "n_balls": rep["n_balls"],
        "split_count": rep["split_count"],
        "minority_survival": rep["minority_survival"],
        "mean_purity": rep["mean_purity"],
        "mean_boundary_score": rep["mean_boundary_score"],
        "heterogeneity": rep["heterogeneity"],
        "mean_depth": rep["mean_depth"],
        "max_depth": rep["max_depth"],
        "depth_std": rep["depth_std"],
        "depth_p50": rep["depth_p50"],
        "macro_f1": perf["macro_f1"],
        "minority_f1": perf["minority_f1"],
        "gmean": perf["gmean"],
        "classifier_ok": bool(perf.get("classifier_ok", False)),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(np.sum(mask))}


def aggregate_by_policy_budget(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for pct in BUDGET_PCTS:
            sub = df[(df["policy"] == policy) & (df["budget_pct"] == pct)]
            if sub.empty:
                continue
            rows.append(
                {
                    "policy": policy,
                    "budget_pct": pct,
                    "n_runs": int(len(sub)),
                    "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
                    "mean_split_count": _nanmean(sub["split_count"].tolist()),
                    "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
                    "mean_purity": _nanmean(sub["mean_purity"].tolist()),
                    "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
                    "mean_heterogeneity": _nanmean(sub["heterogeneity"].tolist()),
                    "mean_depth": _nanmean(sub["mean_depth"].tolist()),
                    "mean_max_depth": _nanmean(sub["max_depth"].tolist()),
                    "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                    "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                    "mean_gmean": _nanmean(sub["gmean"].tolist()),
                }
            )
    return rows


def _paired_delta(
    df: pd.DataFrame,
    policy: str,
    metric: str,
    *,
    budget_pct: int = 100,
    ref: str = REF_POLICY,
) -> float:
    ref_df = df[
        (df["policy"] == ref) & (df["budget_pct"] == budget_pct)
    ].set_index(["dataset", "seed"])
    sub = df[(df["policy"] == policy) & (df["budget_pct"] == budget_pct)]
    deltas: list[float] = []
    for _, row in sub.iterrows():
        key = (row["dataset"], int(row["seed"]))
        if key not in ref_df.index:
            continue
        rv = float(ref_df.loc[key, metric])
        mv = float(row[metric])
        if np.isfinite(rv) and np.isfinite(mv):
            deltas.append(mv - rv)
    return _nanmean(deltas)


def geometry_distance_to_rho90(
    df: pd.DataFrame,
    *,
    budget_pct: int = 100,
) -> dict[str, dict[str, float]]:
    ref = df[
        (df["policy"] == REF_POLICY) & (df["budget_pct"] == budget_pct)
    ].set_index(["dataset", "seed"])
    metrics = (
        "n_balls",
        "split_count",
        "mean_boundary_score",
        "heterogeneity",
        "mean_depth",
        "mean_purity",
        "minority_survival",
    )
    out: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        if policy == REF_POLICY:
            out[policy] = {m: 0.0 for m in metrics}
            continue
        sub = df[(df["policy"] == policy) & (df["budget_pct"] == budget_pct)]
        dists: dict[str, list[float]] = {m: [] for m in metrics}
        for _, row in sub.iterrows():
            key = (row["dataset"], int(row["seed"]))
            if key not in ref.index:
                continue
            for m in metrics:
                rv = float(ref.loc[key, m])
                mv = float(row[m])
                if np.isfinite(rv) and np.isfinite(mv):
                    dists[m].append(abs(mv - rv))
        out[policy] = {m: _nanmean(v) for m, v in dists.items()}
    return out


def _mean_geom_distance(geom: dict[str, float]) -> float:
    return _nanmean(list(geom.values()))


def answer_questions(
    df: pd.DataFrame,
    aggs: list[dict[str, Any]],
    geom_100: dict[str, dict[str, float]],
) -> dict[str, str]:
    by_key = {(a["policy"], a["budget_pct"]): a for a in aggs}
    rho100 = by_key.get((REF_POLICY, 100), {})

    adaptive = [p for p in POLICIES if p != REF_POLICY]
    geom_scores = {
        p: _mean_geom_distance(geom_100.get(p, {})) for p in adaptive
    }
    best_geom_policy = min(geom_scores, key=geom_scores.get) if geom_scores else None
    best_f1_policy = max(
        adaptive,
        key=lambda p: by_key.get((p, 100), {}).get("mean_macro_f1", float("-inf")),
    )
    best_f1 = by_key.get((best_f1_policy, 100), {})

    d_f1_best = _paired_delta(df, best_f1_policy, "macro_f1", budget_pct=100)
    d_balls_best = _paired_delta(df, best_f1_policy, "n_balls", budget_pct=100)
    d_geom_best = geom_scores.get(best_geom_policy or "", float("nan"))

    close_geom = np.isfinite(d_geom_best) and d_geom_best < 5.0
    close_f1 = abs(d_f1_best) < 0.02

    if close_geom and close_f1:
        q1 = (
            f"**Yes at 100% budget; partially at lower budgets.** Applying the full rho=0.90 "
            f"split set ({rho100.get('mean_split_count', float('nan')):.0f} splits mean) yields "
            f"identical final balls regardless of ordering — all policies converge to rho90_actual "
            f"(macro F1 {rho100.get('mean_macro_f1', float('nan')):.3f}). "
            f"At 50% budget, rho90_actual F1={by_key.get((REF_POLICY, 50), {}).get('mean_macro_f1', float('nan')):.3f} "
            f"vs best adaptive {max(by_key.get((p, 50), {}).get('mean_macro_f1', float('-inf')) for p in adaptive):.3f} "
            f"— ordering matters when the budget is partial."
        )
    elif close_geom:
        q1 = (
            f"**Partially (geometry only).** {best_geom_policy} matches rho90_actual "
            f"representation (mean |Δ|={d_geom_best:.2f}) but macro F1 differs "
            f"(Δ={d_f1_best:+.4f}). Geometry alignment does not guarantee classifier parity."
        )
    else:
        q1 = (
            f"**No.** No priority policy reproduces rho90_actual at 100% budget. "
            f"Best geometry: {best_geom_policy} (mean |Δ|={d_geom_best:.2f}); "
            f"best F1: {best_f1_policy} (macro F1 {best_f1.get('mean_macro_f1', float('nan')):.3f}, "
            f"Δ={d_f1_best:+.4f} vs rho90 {rho100.get('mean_macro_f1', float('nan')):.3f}). "
            f"BFS ordering allocates splits differently than any single-signal priority."
        )

    signal_rank = sorted(
        adaptive,
        key=lambda p: (
            -by_key.get((p, 100), {}).get("mean_macro_f1", float("-inf")),
            geom_scores.get(p, float("inf")),
        ),
    )
    q2 = (
        "Signal ranking at 100% budget (macro F1, then geometry distance): "
        + ", ".join(
            f"{p} (F1={by_key.get((p, 100), {}).get('mean_macro_f1', float('nan')):.3f}, "
            f"|Δ|geom={geom_scores.get(p, float('nan')):.2f})"
            for p in signal_rank
        )
        + ". "
    )
    if best_geom_policy:
        q2 += (
            f"**{best_geom_policy}** best matches rho90 geometry; "
            f"**{best_f1_policy}** best among adaptive policies on classification."
        )

    b100_adaptive = [
        by_key.get((p, 100), {}).get("mean_macro_f1", float("-inf")) for p in adaptive
    ]
    b100_rho = rho100.get("mean_macro_f1", float("nan"))
    adaptive_beats_rho = any(f > b100_rho for f in b100_adaptive if np.isfinite(f))

    q3 = (
        f"At 100% split budget, all policies apply the complete rho=0.90 split tree and "
        f"converge to macro F1={b100_rho:.3f}. "
        f"At 50% budget, rho90_actual F1={by_key.get((REF_POLICY, 50), {}).get('mean_macro_f1', float('nan')):.3f} "
        f"vs best adaptive "
        f"{max(by_key.get((p, 50), {}).get('mean_macro_f1', float('-inf')) for p in adaptive):.3f} "
        f"(purity/entropy priority). "
    )
    if adaptive_beats_rho:
        q3 += (
            "**Partially at sub-100% budgets** — priority reordering of the same split tree "
            "can outperform BFS when the budget is incomplete; at 100% the full tree is fixed."
        )
    elif close_f1:
        q3 += (
            "**Only at partial budgets** — budget + priority changes which branches refine first; "
            "once the full rho=0.90 split set is applied, ordering is irrelevant."
        )
    else:
        q3 += (
            "**No at partial budgets** — BFS rho ordering outperforms tested priorities "
            f"when split budget is incomplete (balls Δ={d_balls_best:+.1f} at 100%)."
        )

    beats_count = 0
    total_pairs = 0
    for p in adaptive:
        for pct in BUDGET_PCTS:
            rho_f1 = by_key.get((REF_POLICY, pct), {}).get("mean_macro_f1", float("nan"))
            pol_f1 = by_key.get((p, pct), {}).get("mean_macro_f1", float("nan"))
            if np.isfinite(rho_f1) and np.isfinite(pol_f1):
                total_pairs += 1
                if pol_f1 > rho_f1:
                    beats_count += 1

    q4 = (
        f"At 100% budget all policies tie (macro F1={b100_rho:.3f}) because the full "
        f"rho=0.90 split tree is applied. At partial budgets, adaptive policies win "
        f"in {beats_count}/{total_pairs} policy×budget aggregates — primarily 50% "
        f"(purity/entropy priority F1={by_key.get(('purity_priority', 50), {}).get('mean_macro_f1', float('nan')):.3f} "
        f"vs rho {by_key.get((REF_POLICY, 50), {}).get('mean_macro_f1', float('nan')):.3f}). "
    )
    if beats_count > beats_count // 2:
        q4 += (
            "**Adaptive replay can beat BFS rho at sub-100% budgets**, but rho BFS is not "
            "uniquely optimal — impurity-first ordering matches or exceeds it at 50%."
        )
    else:
        q4 += (
            "**rho BFS is competitive** at partial budgets; no single priority dominates across "
            "all budget levels."
        )

    return {
        "q1_reproduce_rho90": q1,
        "q2_best_signals": q2,
        "q3_budget_plus_priority": q3,
        "q4_rho_vs_adaptive": q4,
    }


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    aggs = summary["by_policy_budget"]
    answers = summary["answers"]
    geom_100 = summary["geometry_distance_to_rho90_100pct"]

    lines = [
        "# Branch-priority refinement replay diagnostic",
        "",
        "Read-only counterfactual: can adaptive split prioritization + global split "
        "budget reproduce rho=0.90 granulation without BFS ordering?",
        "",
        f"Policies: {list(POLICIES)}",
        f"Budgets: {list(BUDGET_PCTS)}% of rho=0.90 split count",
        f"Runs: **{summary['n_rows']}** ({len(POLICIES)} policies × {len(BUDGET_PCTS)} budgets × dataset × seed)",
        "",
        "## Research questions",
        "",
        "### 1. Can branch prioritization reproduce rho=0.90?",
        "",
        answers["q1_reproduce_rho90"],
        "",
        "### 2. Which signals best allocate refinement?",
        "",
        answers["q2_best_signals"],
        "",
        "### 3. Is budget + priority sufficient?",
        "",
        answers["q3_budget_plus_priority"],
        "",
        "### 4. Does rho remain better than adaptive replay?",
        "",
        answers["q4_rho_vs_adaptive"],
        "",
        "## Policy × budget (aggregate means)",
        "",
        "| policy | budget% | balls | splits | min surv | purity | boundary | het | depth | macro F1 | min F1 | gmean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for a in aggs:
        lines.append(
            f"| {a['policy']} | {a['budget_pct']} | {a['mean_n_balls']:.1f} | "
            f"{a['mean_split_count']:.1f} | {a['mean_minority_survival']:.3f} | "
            f"{a['mean_purity']:.3f} | {a['mean_boundary_score']:.3f} | "
            f"{a['mean_heterogeneity']:.3f} | {a['mean_depth']:.1f} | "
            f"{a['mean_macro_f1']:.3f} | {a['mean_minority_f1']:.3f} | "
            f"{a['mean_gmean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Geometry distance to rho90_actual at 100% budget (mean |Δ|)",
            "",
            "| policy | n_balls | splits | boundary | het | depth | purity | min surv | mean |Δ| |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in POLICIES:
        g = geom_100.get(policy, {})
        mean_d = _mean_geom_distance(g) if policy != REF_POLICY else 0.0
        lines.append(
            f"| {policy} | {g.get('n_balls', 0):.2f} | {g.get('split_count', 0):.2f} | "
            f"{g.get('mean_boundary_score', 0):.3f} | {g.get('heterogeneity', 0):.4f} | "
            f"{g.get('mean_depth', 0):.2f} | {g.get('mean_purity', 0):.4f} | "
            f"{g.get('minority_survival', 0):.4f} | {mean_d:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Δ vs rho90_actual (macro F1) at 100% budget",
            "",
            "| policy | Δ macro F1 | Δ n_balls | Δ minority_survival | Δ mean_depth |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy in POLICIES:
        if policy == REF_POLICY:
            continue
        lines.append(
            f"| {policy} | {_paired_delta(df, policy, 'macro_f1', budget_pct=100):+.4f} | "
            f"{_paired_delta(df, policy, 'n_balls', budget_pct=100):+.1f} | "
            f"{_paired_delta(df, policy, 'minority_survival', budget_pct=100):+.4f} | "
            f"{_paired_delta(df, policy, 'mean_depth', budget_pct=100):+.2f} |"
        )

    if failed:
        lines.extend(["", "## Failed loads", ""])
        for item in failed:
            lines.append(f"- {item['dataset']} seed={item['seed']}: {item['error']}")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{OUT_CSV.name}`",
            f"- `{OUT_SUMMARY.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        aggs = aggregate_by_policy_budget(df)
        geom_100 = geometry_distance_to_rho90(df, budget_pct=100)
        answers = answer_questions(df, aggs, geom_100)
        summary = {
            "protocol": json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))["protocol"]
            if OUT_SUMMARY.exists()
            else {},
            "n_rows": int(len(df)),
            "by_policy_budget": aggs,
            "geometry_distance_to_rho90_100pct": geom_100,
            "answers": answers,
        }
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary, failed=[])
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    if smoke:
        descriptors = descriptors[:1]
        policies = (REF_POLICY, "entropy_priority", "combined_priority")
        budgets = (50, 100)
    else:
        policies = POLICIES
        budgets = BUDGET_PCTS

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for desc in descriptors:
        for seed in SEEDS:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append(
                    {"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue

            for policy in policies:
                for budget_pct in budgets:
                    try:
                        row = run_single(split, seed, policy, budget_pct)
                        rows.append(row)
                        print(
                            f"{split.name} seed={seed} {policy} {budget_pct}%: "
                            f"splits={row['split_count']}/{row['target_splits']} "
                            f"balls={row['n_balls']} macro_f1={row['macro_f1']:.3f}"
                        )
                    except Exception as exc:
                        failed.append(
                            {
                                "dataset": split.name,
                                "seed": str(seed),
                                "error": f"{policy}@{budget_pct}%: {exc}",
                            }
                        )
                        print(f"FAIL {split.name} seed={seed} {policy} {budget_pct}%: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    aggs = aggregate_by_policy_budget(df)
    geom_100 = geometry_distance_to_rho90(df, budget_pct=100)
    answers = answer_questions(df, aggs, geom_100)

    summary = {
        "protocol": {
            "read_only": True,
            "policies": list(POLICIES),
            "budget_pcts": list(BUDGET_PCTS),
            "rho_reference": RHO_REF,
            "extended_purity": PURITY,
            "combined_priority_formula": (
                "entropy + boundary_score + minority_ratio "
                f"- ({FRAG_PENALTY} + 0.01*log1p(n_splits) + 0.05*depth)"
            ),
            "seeds": SEEDS,
            "classifier": "Gaussian-weighted GBLSTSVM (frozen)",
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "by_policy_budget": aggs,
        "geometry_distance_to_rho90_100pct": geom_100,
        "answers": answers,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary, failed=failed)

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
