"""
Genuine online adaptive entropy granulation experiment.

First end-to-end validation of the adaptive refinement hypothesis using LIVE
KMeans splits and an entropy-priority queue — no rho=0.90 tree recording,
reordering, or replay.

Does NOT modify src/, solver, prediction, CW, CA, Topo, Gaussian membership,
or existing init_granular_balls().

Policies:
  baseline-rho90              — existing BFS rho=0.90 purity stopping
  baseline-rho90-{pct}pct     — BFS rho capped at pct% of rho90 split count
  adaptive-entropy-{pct}pct   — online entropy-priority queue, split budget
  adaptive-entropy-sqrt-k{k}  — online entropy queue, budget = k * sqrt(n)

Run from GBLSTSVM root:
  python experiments/test_adaptive_entropy_granulation.py --smoke
  python experiments/test_adaptive_entropy_granulation.py
  python experiments/test_adaptive_entropy_granulation.py --from-csv
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

OUT_DIR = Path(__file__).resolve().parent / "output" / "adaptive_entropy_granulation"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
RHO_REF = RHO_VALUES["rho0.90"]
SAVE_EVERY = 10

BUDGET_PCTS = (50, 75, 100, 125)
SQRT_K_VALUES = (2, 4, 6, 8)
BASELINE_FULL = "baseline-rho90"
REPLAY_REF_F1_50 = 0.544
REPLAY_REF_F1_RHO_50 = 0.514


@dataclass(frozen=True)
class GranulationConfig:
    name: str
    family: str  # baseline_full | baseline_budget | adaptive_pct | adaptive_sqrt
    split_budget: int | None = None
    budget_pct: int | float | None = None
    sqrt_k: float | None = None


@dataclass
class GranulationResult:
    kept: list[GranularBall]
    depths: list[int]
    n_successful_splits: int
    n_failed_splits: int
    n_budget_iterations: int
    split_budget: int | None
    budget_met: bool


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


def _del_ball_kept(balls: list[GranularBall]) -> list[GranularBall]:
    return [b for b in balls if int(b.num) >= MIN_BALL_COUNT]


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


def measure_rho90_split_count(indexed: np.ndarray, *, seed: int) -> int:
    """Count rho=0.90 splits for budget calibration only (no tree stored)."""
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
    """Existing BFS rho=0.90 granulation; optional split-budget cap."""
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    n_success = 0
    n_failed = 0
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
    budget_met = split_budget is None or n_success >= split_budget
    return GranulationResult(
        kept=kept,
        depths=kept_depths,
        n_successful_splits=n_success,
        n_failed_splits=n_failed,
        n_budget_iterations=n_iterations,
        split_budget=split_budget,
        budget_met=budget_met,
    )


def build_adaptive_entropy_granulation(
    indexed: np.ndarray,
    *,
    seed: int,
    split_budget: int,
    global_min: float,
    rho90_split_count: int,
) -> GranulationResult:
    """
    Genuine online entropy-priority granulation.

    Maintains a live priority queue over splittable balls; each iteration
    selects the highest-entropy candidate and splits via KMeans(k=2).
    """
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    split_done = [False]
    counter = 0
    heap: list[tuple[float, int, int]] = []
    relax_level = 0
    n_success = 0
    n_failed = 0
    n_iterations = 0

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
        if int(c1.num) < MIN_BALL_COUNT or int(c2.num) < MIN_BALL_COUNT:
            n_failed += 1
            split_done[idx] = True
            continue

        balls[idx] = c1
        balls.append(c2)
        depths.append(depths[idx] + 1)
        split_done[idx] = True
        split_done.append(False)
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


def _make_configs(
    rho90_split_count: int,
    n_samples: int,
) -> list[GranulationConfig]:
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
                name=f"adaptive-entropy-{pct}pct",
                family="adaptive_pct",
                split_budget=budget,
                budget_pct=pct,
            )
        )
    for k in SQRT_K_VALUES:
        budget = max(1, int(round(k * np.sqrt(n_samples))))
        configs.append(
            GranulationConfig(
                name=f"adaptive-entropy-sqrt-k{k}",
                family="adaptive_sqrt",
                split_budget=budget,
                sqrt_k=float(k),
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
        return build_baseline_rho90(
            indexed, seed=seed, split_budget=config.split_budget
        )
    return build_adaptive_entropy_granulation(
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
        "budget_pct": config.budget_pct,
        "sqrt_k": config.sqrt_k,
        "rho90_split_count": rho90_split_count,
        "split_budget": result.split_budget,
        "split_count": result.n_successful_splits,
        "n_failed_splits": result.n_failed_splits,
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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(np.sum(mask))}


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
                "n_runs": int(len(sub)),
                "mean_split_count": _nanmean(sub["split_count"].tolist()),
                "mean_split_budget": _nanmean(
                    [x for x in sub["split_budget"].tolist() if np.isfinite(x)]
                ),
                "frac_budget_met": _nanmean(sub["budget_met"].tolist()),
                "mean_failed_splits": _nanmean(sub["n_failed_splits"].tolist()),
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


def answer_questions(df: pd.DataFrame, aggs: list[dict[str, Any]]) -> dict[str, str]:
    by_name = {a["config"]: a for a in aggs}

    adapt_50 = by_name.get("adaptive-entropy-50pct", {})
    base_50 = by_name.get("baseline-rho90-50pct", {})
    adapt_75 = by_name.get("adaptive-entropy-75pct", {})
    adapt_100 = by_name.get("adaptive-entropy-100pct", {})
    base_100 = by_name.get("baseline-rho90-100pct", {})
    base_full = by_name.get(BASELINE_FULL, {})

    d50_online = _paired_delta(df, "adaptive-entropy-50pct", "baseline-rho90-50pct", "macro_f1")
    d100_online = _paired_delta(df, "adaptive-entropy-100pct", "baseline-rho90-100pct", "macro_f1")
    replay_gap = adapt_50.get("mean_macro_f1", float("nan")) - base_50.get(
        "mean_macro_f1", float("nan")
    )

    q1 = (
        f"Replay diagnostic at 50% budget: entropy replay +0.030 vs BFS rho "
        f"({REPLAY_REF_F1_50:.3f} vs {REPLAY_REF_F1_RHO_50:.3f}). "
        f"Online adaptive-entropy-50pct: F1={adapt_50.get('mean_macro_f1', float('nan')):.3f} "
        f"vs baseline-rho90-50pct {base_50.get('mean_macro_f1', float('nan')):.3f} "
        f"(paired Δ={d50_online:+.4f}, budget met {adapt_50.get('frac_budget_met', float('nan')):.1%}). "
    )
    if abs(d50_online - 0.030) < 0.015 and d50_online > 0.01:
        q1 += "**Yes** — genuine online entropy scheduling reproduces the replay gain direction and magnitude."
    elif d50_online > 0.005:
        q1 += "**Partially** — online scheduling shows the same directional gain but magnitude differs from replay (live KMeans tree differs from recorded-tree reordering)."
    else:
        q1 += (
            "**No** — online scheduling does not reproduce the replay gain; "
            "live priority exhausts splittable candidates before spending the full budget "
            f"(mean splits {adapt_50.get('mean_split_count', float('nan')):.1f} "
            f"vs budget {adapt_50.get('mean_split_budget', float('nan')):.1f})."
        )

    q2 = (
        f"At 50% budget: adaptive F1={adapt_50.get('mean_macro_f1', float('nan')):.3f} "
        f"vs BFS rho {base_50.get('mean_macro_f1', float('nan')):.3f} (Δ={d50_online:+.4f}). "
        f"At 100% budget: adaptive {adapt_100.get('mean_macro_f1', float('nan')):.3f} "
        f"vs BFS {base_100.get('mean_macro_f1', float('nan')):.3f} (Δ={d100_online:+.4f}). "
    )
    if d50_online > 0.01:
        q2 += "**Yes at 50%** — entropy scheduling outperforms BFS rho at equal split budget."
    elif abs(d100_online) < 0.01:
        q2 += "**Tie at 100%** — both reach similar F1 when budgets are fully spent."
    else:
        q2 += "**Mixed** — advantage is budget-dependent."

    q3 = (
        f"Full baseline-rho90: F1={base_full.get('mean_macro_f1', float('nan')):.3f} "
        f"at {base_full.get('mean_split_count', float('nan')):.0f} splits. "
        f"Adaptive-75pct: F1={adapt_75.get('mean_macro_f1', float('nan')):.3f} "
        f"at {adapt_75.get('mean_split_count', float('nan')):.0f} splits "
        f"({adapt_75.get('mean_split_count', 0) / max(base_full.get('mean_split_count', 1), 1):.0%} of full rho splits). "
    )
    if adapt_75.get("mean_macro_f1", 0) >= base_full.get("mean_macro_f1", 0) - 0.015:
        q3 += "**Yes** — 75% adaptive budget approaches full-rho performance with fewer splits."
    else:
        gap = base_full.get("mean_macro_f1", 0) - adapt_75.get("mean_macro_f1", 0)
        q3 += f"**Partially** — 25% split savings costs ~{gap:.3f} macro F1 vs full rho."

    q4 = (
        "This experiment tests entropy-only scheduling with no boundary or minority terms. "
        f"Adaptive-entropy-50pct macro F1={adapt_50.get('mean_macro_f1', float('nan')):.3f}. "
        f"If this exceeds baseline-rho90-50pct ({base_50.get('mean_macro_f1', float('nan')):.3f}), "
        "entropy alone is sufficient at partial budgets. "
    )
    if d50_online > 0.01:
        q4 += "**Yes** — entropy-only scheduling provides the improvement without extra terms."
    else:
        q4 += "**Unclear** — online entropy alone does not clearly exceed BFS at equal budget."

    sqrt_rows = [a for a in aggs if a["config"].startswith("adaptive-entropy-sqrt-k")]
    best_sqrt = max(sqrt_rows, key=lambda a: a.get("mean_macro_f1", float("-inf")), default={})
    q5 = (
        f"Best sqrt-n config: **{best_sqrt.get('config', 'n/a')}** "
        f"(F1={best_sqrt.get('mean_macro_f1', float('nan')):.3f}, "
        f"splits={best_sqrt.get('mean_split_count', float('nan')):.1f}, "
        f"balls={best_sqrt.get('mean_n_balls', float('nan')):.1f}). "
        f"Compare to full rho F1={base_full.get('mean_macro_f1', float('nan')):.3f}. "
    )
    if best_sqrt.get("mean_macro_f1", 0) >= base_full.get("mean_macro_f1", 0) - 0.01:
        q5 += "Dataset-independent sqrt scaling is competitive with rho-calibrated budgets."
    else:
        q5 += "Rho-calibrated split fractions remain better than fixed sqrt-n coefficients."

    online_real = d50_online > 0.005 and adapt_50.get("frac_budget_met", 0) < 0.95
    q6 = (
        f"Budget-met rate for adaptive-50pct: {adapt_50.get('frac_budget_met', float('nan')):.1%}; "
        f"mean failed splits {adapt_50.get('mean_failed_splits', float('nan')):.2f}. "
    )
    if d50_online > 0.01 and adapt_50.get("frac_budget_met", 1) > 0.8:
        q6 += (
            "**Yes** — improvement persists with live KMeans splits and near-full budget "
            "utilization, not a replay-tree artifact."
        )
    elif d50_online > 0.01:
        q6 += (
            "**Partially genuine** — directional gain exists online but under-spends budget "
            "when entropy-priority deep-splits exhaust rho-eligible candidates early."
        )
    else:
        q6 += (
            "**Replay artifact risk** — online scheduling cannot match replay gains; "
            "recorded-tree reordering and live priority produce different trees."
        )

    return {
        "q1_reproduce_replay": q1,
        "q2_outperform_rho_equal_budget": q2,
        "q3_reduce_splits": q3,
        "q4_entropy_only": q4,
        "q5_best_normalized_budget": q5,
        "q6_genuine_vs_artifact": q6,
    }


def write_report(path: Path, *, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    aggs = summary["by_config"]
    answers = summary["answers"]

    lines = [
        "# Adaptive entropy granulation (online)",
        "",
        "First genuine online validation: live KMeans splits + entropy-priority queue. "
        "No rho-tree recording or replay.",
        "",
        f"Runs: **{summary['n_rows']}**",
        "",
        "## Research questions",
        "",
        "### 1. Does genuine online entropy-priority granulation reproduce the replay result?",
        "",
        answers["q1_reproduce_replay"],
        "",
        "### 2. Does it outperform rho90 at equal refinement budgets?",
        "",
        answers["q2_outperform_rho_equal_budget"],
        "",
        "### 3. Does it reduce the number of splits required for comparable performance?",
        "",
        answers["q3_reduce_splits"],
        "",
        "### 4. Does entropy-only scheduling already provide the improvement?",
        "",
        answers["q4_entropy_only"],
        "",
        "### 5. What is the best normalized budget?",
        "",
        answers["q5_best_normalized_budget"],
        "",
        "### 6. Genuine improvement vs replay artifact?",
        "",
        answers["q6_genuine_vs_artifact"],
        "",
        "## Configuration summary (aggregate means)",
        "",
        "| config | splits | budget | met% | balls | entropy | depth | macro F1 | min F1 | gmean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for a in aggs:
        lines.append(
            f"| {a['config']} | {a['mean_split_count']:.1f} | "
            f"{a.get('mean_split_budget', float('nan')):.1f} | "
            f"{a.get('frac_budget_met', float('nan')):.2f} | "
            f"{a['mean_n_balls']:.1f} | {a['mean_entropy']:.3f} | "
            f"{a['mean_depth']:.1f} | {a['mean_macro_f1']:.3f} | "
            f"{a['mean_minority_f1']:.3f} | {a['mean_gmean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Δ macro F1 (adaptive − baseline BFS at same budget)",
            "",
            "| budget | Δ at pct budget |",
            "|---|---:|",
            f"| 50% | {_paired_delta(df, 'adaptive-entropy-50pct', 'baseline-rho90-50pct', 'macro_f1'):+.4f} |",
            f"| 75% | {_paired_delta(df, 'adaptive-entropy-75pct', 'baseline-rho90-75pct', 'macro_f1'):+.4f} |",
            f"| 100% | {_paired_delta(df, 'adaptive-entropy-100pct', 'baseline-rho90-100pct', 'macro_f1'):+.4f} |",
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
            "no_tree_replay": True,
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
                best = max(
                    (r for r in recent if r["family"].startswith("adaptive")),
                    key=lambda r: r["macro_f1"],
                )
                a50 = next((r for r in recent if r["config"] == "adaptive-entropy-50pct"), None)
                b50 = next((r for r in recent if r["config"] == "baseline-rho90-50pct"), None)
                a50_f1 = a50["macro_f1"] if a50 else float("nan")
                a50_splits = f"{a50['split_count']}/{a50['split_budget']}" if a50 else "n/a"
                print(
                    f"{split.name} s{seed}: rho90 F1={ref['macro_f1']:.3f} "
                    f"splits={ref['split_count']} | "
                    f"adapt50 F1={a50_f1:.3f} splits={a50_splits} "
                    f"| best {best['config']} F1={best['macro_f1']:.3f}"
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
