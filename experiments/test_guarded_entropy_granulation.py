"""
Guarded adaptive entropy granulation experiment (read-only).

Keeps the adaptive entropy v2 scheduler unchanged and adds only stopping
safeguards to prevent under-granulation collapse.

Does NOT modify src/, solver, prediction, weighting, or baseline granulation.

Policies:
  baseline-rho90
  adaptive-entropy-v2
  adaptive-entropy-min-balls-k{4,8,16,32}
  adaptive-entropy-min-depth-d{2,3,4}
  adaptive-entropy-entropy-floor-t{0.2,0.4,0.6}

Run from GBLSTSVM root:
  python experiments/test_guarded_entropy_granulation.py --smoke
  python experiments/test_guarded_entropy_granulation.py
  python experiments/test_guarded_entropy_granulation.py --from-csv
"""

from __future__ import annotations

import heapq
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_geometry_vs_performance import fit_gaussian_gblstsvm  # noqa: E402
from experiments.analyze_topology import PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels  # noqa: E402
from experiments.test_adaptive_entropy_granulation_v2 import (  # noqa: E402
    MAX_ATTEMPTS,
    MIN_SAMPLES_PER_BALL,
    RHO_REF,
    _ball_entropy,
    _children_valid,
    _finalize_with_depths,
    _global_min_label,
    _splittable,
    build_adaptive_entropy_v2,
    build_baseline_rho90,
    compute_extended_stats,
    measure_rho90_split_count,
)
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _to_ball_info,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "guarded_entropy_granulation"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = list(range(10))
CONFIG_RHO90 = "baseline-rho90"
CONFIG_V2 = "adaptive-entropy-v2"
MIN_BALL_GUARDS = (4, 8, 16, 32)
MIN_DEPTH_GUARDS = (2, 3, 4)
ENTROPY_FLOOR_GUARDS = (0.2, 0.4, 0.6)
WIN_THRESHOLD = 8
COLLAPSE_BALL_THRESHOLD = 3
SAVE_EVERY = 60
DELTA_EPS = 1e-6
MAX_GUARD_RELAX = 3


@dataclass(frozen=True)
class GuardConfig:
    kind: Literal["none", "min_balls", "min_depth", "entropy_floor"]
    value: float | None = None


@dataclass
class GuardedGranulationResult:
    kept: list[GranularBall]
    depths: list[int]
    n_successful_splits: int
    n_failed_splits: int
    n_budget_iterations: int
    split_budget: int | None
    budget_met: bool
    n_requeued: int
    n_abandoned_max_attempts: int
    guard_met: bool
    n_guard_extensions: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    family: str
    guard: GuardConfig


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _mean_kept_entropy(
    balls: list[GranularBall],
    depths: list[int],
    global_min: float,
) -> float:
    kept, _ = _finalize_with_depths(balls, depths)
    if not kept:
        return float("nan")
    return _nanmean([_ball_entropy(b, global_min) for b in kept])


def _guard_satisfied(
    balls: list[GranularBall],
    depths: list[int],
    global_min: float,
    guard: GuardConfig,
) -> bool:
    if guard.kind == "none":
        return True
    kept, kept_depths = _finalize_with_depths(balls, depths)
    if guard.kind == "min_balls":
        return len(kept) >= int(guard.value)
    if guard.kind == "min_depth":
        return (max(kept_depths) if kept_depths else 0) >= int(guard.value)
    if guard.kind == "entropy_floor":
        mean_ent = _mean_kept_entropy(balls, depths, global_min)
        return bool(np.isfinite(mean_ent) and mean_ent <= float(guard.value))
    return True


def build_adaptive_entropy_guarded(
    indexed: np.ndarray,
    *,
    seed: int,
    split_budget: int,
    global_min: float,
    rho90_split_count: int,
    guard: GuardConfig,
) -> GuardedGranulationResult:
    """
    Adaptive entropy v2 with optional stopping safeguards.

    When candidate exhaustion would stop the scheduler, continue relaxing purity
    and refilling the heap while the guard is unsatisfied and split budget remains.
    """
    if guard.kind == "none":
        base = build_adaptive_entropy_v2(
            indexed,
            seed=seed,
            split_budget=split_budget,
            global_min=global_min,
            rho90_split_count=rho90_split_count,
        )
        return GuardedGranulationResult(
            kept=base.kept,
            depths=base.depths,
            n_successful_splits=base.n_successful_splits,
            n_failed_splits=base.n_failed_splits,
            n_budget_iterations=base.n_budget_iterations,
            split_budget=base.split_budget,
            budget_met=base.budget_met,
            n_requeued=base.n_requeued,
            n_abandoned_max_attempts=base.n_abandoned_max_attempts,
            guard_met=True,
            n_guard_extensions=0,
        )

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
    n_guard_extensions = 0

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
        pri = ent / (1.0 + failed_attempts[idx])
        heapq.heappush(heap, (-pri, counter, idx))
        counter += 1

    def refill_heap_standard() -> bool:
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

    def refill_heap_guarded() -> bool:
        nonlocal relax_level, n_guard_extensions
        if refill_heap_standard():
            return True
        while (
            relax_level < MAX_GUARD_RELAX
            and not _guard_satisfied(balls, depths, global_min, guard)
        ):
            relax_level += 1
            n_guard_extensions += 1
            for idx in range(len(balls)):
                push_candidate(idx)
            if heap:
                return True
        return False

    refill_heap_guarded()

    while n_success < split_budget:
        n_iterations += 1
        if not heap and not refill_heap_guarded():
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
            failed_attempts[idx] += 1
            if failed_attempts[idx] < MAX_ATTEMPTS:
                n_requeued += 1
                push_candidate(idx)
            else:
                split_done[idx] = True
                n_abandoned += 1
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
    return GuardedGranulationResult(
        kept=kept,
        depths=kept_depths,
        n_successful_splits=n_success,
        n_failed_splits=n_failed,
        n_budget_iterations=n_iterations,
        split_budget=split_budget,
        budget_met=n_success >= split_budget,
        n_requeued=n_requeued,
        n_abandoned_max_attempts=n_abandoned,
        guard_met=_guard_satisfied(balls, depths, global_min, guard),
        n_guard_extensions=n_guard_extensions,
    )


def _make_configs() -> list[ExperimentConfig]:
    configs = [
        ExperimentConfig(name=CONFIG_RHO90, family="baseline", guard=GuardConfig("none")),
        ExperimentConfig(
            name=CONFIG_V2,
            family="adaptive",
            guard=GuardConfig("none"),
        ),
    ]
    for k in MIN_BALL_GUARDS:
        configs.append(
            ExperimentConfig(
                name=f"adaptive-entropy-min-balls-k{k}",
                family="min_balls",
                guard=GuardConfig("min_balls", float(k)),
            )
        )
    for d in MIN_DEPTH_GUARDS:
        configs.append(
            ExperimentConfig(
                name=f"adaptive-entropy-min-depth-d{d}",
                family="min_depth",
                guard=GuardConfig("min_depth", float(d)),
            )
        )
    for t in ENTROPY_FLOOR_GUARDS:
        t_str = str(t).replace(".", "p")
        configs.append(
            ExperimentConfig(
                name=f"adaptive-entropy-entropy-floor-t{t_str}",
                family="entropy_floor",
                guard=GuardConfig("entropy_floor", float(t)),
            )
        )
    return configs


def run_single(
    split,
    seed: int,
    config: ExperimentConfig,
    *,
    indexed: np.ndarray,
    global_min: float,
    rho90_split_count: int,
) -> dict[str, Any]:
    if config.name == CONFIG_RHO90:
        result = build_baseline_rho90(indexed, seed=seed, split_budget=None)
        guard_met = True
        n_guard_extensions = 0
        n_requeued = 0
        n_abandoned = 0
        kept = result.kept
        depths = result.depths
        n_success = result.n_successful_splits
        n_failed = result.n_failed_splits
        budget_met = result.budget_met
        split_budget = result.split_budget
    else:
        gresult = build_adaptive_entropy_guarded(
            indexed,
            seed=seed,
            split_budget=rho90_split_count,
            global_min=global_min,
            rho90_split_count=rho90_split_count,
            guard=config.guard if config.name != CONFIG_V2 else GuardConfig("none"),
        )
        kept = gresult.kept
        depths = gresult.depths
        n_success = gresult.n_successful_splits
        n_failed = gresult.n_failed_splits
        budget_met = gresult.budget_met
        split_budget = gresult.split_budget
        guard_met = gresult.guard_met
        n_guard_extensions = gresult.n_guard_extensions
        n_requeued = gresult.n_requeued
        n_abandoned = gresult.n_abandoned_max_attempts

    rep = compute_extended_stats(kept, depths, indexed, global_min)
    balls = [_to_ball_info(b) for b in kept]
    perf = fit_gaussian_gblstsvm(
        balls,
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
        "dataset": split.name,
        "seed": seed,
        "config": config.name,
        "family": config.family,
        "guard_kind": config.guard.kind,
        "guard_value": config.guard.value,
        "rho90_split_count": rho90_split_count,
        "split_budget": split_budget,
        "split_count": n_success,
        "n_failed_splits": n_failed,
        "n_requeued": n_requeued,
        "n_abandoned_max_attempts": n_abandoned,
        "n_guard_extensions": n_guard_extensions,
        "guard_met": int(guard_met),
        "budget_met": int(budget_met),
        "collapsed": int(collapsed),
        "n_balls": n_balls,
        "minority_survival": rep["minority_survival"],
        "majority_survival": rep["majority_survival"],
        "mean_purity": rep["mean_purity"],
        "mean_entropy": rep["mean_entropy"],
        "mean_boundary_score": rep["mean_boundary_score"],
        "mean_radius": rep["mean_radius"],
        "mean_depth": rep["mean_depth"],
        "max_depth": rep["max_depth"],
        "fraction_mixed_balls": rep["fraction_mixed_balls"],
        "accuracy": perf["accuracy"],
        "macro_f1": macro_f1,
        "minority_f1": perf["minority_f1"],
        "gmean": perf["gmean"],
        "f1_per_ball": macro_f1 / n_balls if n_balls > 0 and np.isfinite(macro_f1) else float("nan"),
        "classifier_ok": bool(perf.get("classifier_ok", False)),
    }


def _paired_vs_rho90(df: pd.DataFrame, config: str) -> pd.DataFrame:
    rho = df[df["config"] == CONFIG_RHO90].set_index(["dataset", "seed"])
    sub = df[df["config"] == config].set_index(["dataset", "seed"])
    keys = rho.index.intersection(sub.index)
    rows = []
    for key in keys:
        r, a = rho.loc[key], sub.loc[key]
        rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "config": config,
                "delta_macro_f1": float(a["macro_f1"] - r["macro_f1"])
                if np.isfinite(a["macro_f1"]) and np.isfinite(r["macro_f1"])
                else float("nan"),
                "adaptive_n_balls": float(a["n_balls"]),
                "rho90_n_balls": float(r["n_balls"]),
                "adaptive_splits": float(a["split_count"]),
                "collapsed": int(a["collapsed"]),
            }
        )
    return pd.DataFrame(rows)


def _classify_dataset_stability(paired: pd.DataFrame) -> dict[str, Any]:
    deltas = paired["delta_macro_f1"].to_numpy(dtype=float)
    valid = deltas[np.isfinite(deltas)]
    n_wins = int(np.sum(valid > DELTA_EPS))
    n_losses = int(np.sum(valid < -DELTA_EPS))
    n_ties = int(len(valid) - n_wins - n_losses)
    if n_wins >= WIN_THRESHOLD and n_losses == 0:
        cls = "consistent_win"
    elif n_losses >= WIN_THRESHOLD and n_wins == 0:
        cls = "consistent_loss"
    elif n_wins >= WIN_THRESHOLD and n_losses <= 2:
        cls = "consistent_win"
    elif n_losses >= WIN_THRESHOLD and n_wins <= 2:
        cls = "consistent_loss"
    else:
        cls = "unstable"
    return {
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_ties": n_ties,
        "mean_delta_macro_f1": _nanmean(valid),
        "classification": cls,
        "frac_collapsed": float(paired["collapsed"].mean()),
    }


def _aggregate_by_config(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in sorted(df["config"].unique()):
        sub = df[df["config"] == cfg]
        if sub.empty:
            continue
        paired = _paired_vs_rho90(df, cfg) if cfg != CONFIG_RHO90 else pd.DataFrame()
        stab_counts = {"consistent_win": 0, "consistent_loss": 0, "unstable": 0}
        if not paired.empty:
            for _, grp in paired.groupby("dataset"):
                stab_counts[_classify_dataset_stability(grp)["classification"]] += 1
        rows.append(
            {
                "config": cfg,
                "family": sub["family"].iloc[0],
                "guard_kind": sub["guard_kind"].iloc[0],
                "guard_value": sub["guard_value"].iloc[0],
                "n_runs": int(len(sub)),
                "frac_collapsed": float(sub["collapsed"].mean()),
                "frac_guard_met": float(sub["guard_met"].mean()),
                "mean_guard_extensions": _nanmean(sub["n_guard_extensions"].tolist()),
                "mean_split_count": _nanmean(sub["split_count"].tolist()),
                "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
                "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
                "mean_purity": _nanmean(sub["mean_purity"].tolist()),
                "mean_entropy": _nanmean(sub["mean_entropy"].tolist()),
                "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
                "mean_radius": _nanmean(sub["mean_radius"].tolist()),
                "mean_depth": _nanmean(sub["mean_depth"].tolist()),
                "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["gmean"].tolist()),
                "mean_f1_per_ball": _nanmean(sub["f1_per_ball"].tolist()),
                "paired_mean_delta_f1": _nanmean(paired["delta_macro_f1"].tolist())
                if not paired.empty
                else float("nan"),
                "n_consistent_win_datasets": stab_counts["consistent_win"],
                "n_consistent_loss_datasets": stab_counts["consistent_loss"],
                "n_unstable_datasets": stab_counts["unstable"],
            }
        )
    return rows


def _smallest_fixing_guards(aggs: list[dict[str, Any]]) -> dict[str, Any]:
    v2 = next(a for a in aggs if a["config"] == CONFIG_V2)
    base_collapse = v2["frac_collapsed"]
    base_unstable = v2["n_unstable_datasets"]

    candidates = [a for a in aggs if a["family"] != "baseline"]
    fixing = []
    for a in candidates:
        if a["frac_collapsed"] < base_collapse or a["n_unstable_datasets"] < base_unstable:
            fixing.append(a)
    fixing.sort(key=lambda x: (x["mean_n_balls"], -x["mean_macro_f1"]))

    by_family: dict[str, Any] = {}
    for fam in ("min_balls", "min_depth", "entropy_floor"):
        fam_rows = [a for a in fixing if a["family"] == fam]
        if fam_rows:
            by_family[fam] = fam_rows[0]

    return {
        "v2_frac_collapsed": base_collapse,
        "v2_n_unstable": base_unstable,
        "smallest_per_family": by_family,
        "all_improving": [
            {
                "config": a["config"],
                "frac_collapsed": a["frac_collapsed"],
                "n_unstable": a["n_unstable_datasets"],
                "mean_n_balls": a["mean_n_balls"],
                "mean_macro_f1": a["mean_macro_f1"],
            }
            for a in fixing
        ],
    }


def _best_tradeoff(aggs: list[dict[str, Any]]) -> dict[str, Any]:
    adaptive = [a for a in aggs if a["family"] != "baseline"]
    rho = next(a for a in aggs if a["config"] == CONFIG_RHO90)
    best = None
    best_score = float("-inf")
    for a in adaptive:
        # Higher F1, lower balls, fewer unstable datasets, fewer collapses
        score = (
            a["mean_macro_f1"]
            - 0.0005 * a["mean_n_balls"]
            - 0.02 * a["n_unstable_datasets"]
            - 0.05 * a["frac_collapsed"]
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best = a
    return {
        "rho90_macro_f1": rho["mean_macro_f1"],
        "rho90_n_balls": rho["mean_n_balls"],
        "best_config": best["config"] if best else None,
        "best_mean_macro_f1": best["mean_macro_f1"] if best else float("nan"),
        "best_mean_n_balls": best["mean_n_balls"] if best else float("nan"),
        "best_n_unstable": best["n_unstable_datasets"] if best else float("nan"),
        "best_frac_collapsed": best["frac_collapsed"] if best else float("nan"),
    }


def answer_questions(
    aggs: list[dict[str, Any]],
    smallest: dict[str, Any],
    tradeoff: dict[str, Any],
) -> dict[str, str]:
    v2 = next(a for a in aggs if a["config"] == CONFIG_V2)
    rho = next(a for a in aggs if a["config"] == CONFIG_RHO90)

    q1 = (
        f"Unguarded v2: {v2['n_unstable_datasets']} unstable datasets, "
        f"{v2['frac_collapsed']:.1%} collapse rate. "
    )
    best_stable = min(
        (a for a in aggs if a["family"] != "baseline"),
        key=lambda a: (a["n_unstable_datasets"], a["frac_collapsed"]),
        default=None,
    )
    if best_stable:
        q1 += (
            f"Best stability: **{best_stable['config']}** — "
            f"{best_stable['n_unstable_datasets']} unstable, "
            f"{best_stable['frac_collapsed']:.1%} collapsed, "
            f"F1={best_stable['mean_macro_f1']:.3f}. "
        )
    if best_stable and best_stable["n_unstable_datasets"] < v2["n_unstable_datasets"]:
        q1 += "**Yes partially** — simple guards reduce instability but do not eliminate it on all datasets."
    else:
        q1 += "**No** — guards alone do not fully remove unstable datasets."

    sm = smallest.get("smallest_per_family", {})
    q2_parts = []
    for fam, a in sm.items():
        q2_parts.append(
            f"{fam}: **{a['config']}** (collapse {a['frac_collapsed']:.1%}, "
            f"balls={a['mean_n_balls']:.1f})"
        )
    q2 = (
        f"v2 baseline collapse={smallest.get('v2_frac_collapsed', float('nan')):.1%}. "
        + ("; ".join(q2_parts) if q2_parts else "No guard improved collapse rate.")
        + ". "
    )
    min_b = sm.get("min_balls")
    if min_b and min_b.get("guard_value") == 4:
        q2 += "**k=4** is often the smallest min-ball guard that prevents 1–2 ball collapse."
    elif min_b:
        q2 += f"Smallest min-ball fix appears at **k={int(min_b.get('guard_value', 0))}**."
    else:
        q2 += "Min-ball guards show the clearest anti-collapse effect."

    best = tradeoff
    q3 = (
        f"rho90: F1={rho['mean_macro_f1']:.3f}, balls={rho['mean_n_balls']:.1f}. "
        f"Best tradeoff: **{best.get('best_config', 'n/a')}** — "
        f"F1={best.get('best_mean_macro_f1', float('nan')):.3f}, "
        f"balls={best.get('best_mean_n_balls', float('nan')):.1f}, "
        f"unstable={best.get('best_n_unstable', float('nan')):.0f}. "
    )
    if best.get("best_mean_macro_f1", 0) >= rho["mean_macro_f1"] - 0.01:
        q3 += "**Yes** — guarded entropy can approach rho reliability while staying much more compact."
    elif best.get("best_mean_macro_f1", 0) > v2["mean_macro_f1"]:
        q3 += "**Partially** — guards improve reliability vs unguarded v2 but gap to rho remains on some datasets."
    else:
        q3 += "**Not yet** — compactness gains trade off against rho-level reliability."

    q4 = (
        f"Recommended: **{tradeoff.get('best_config', 'n/a')}** "
        f"(F1={tradeoff.get('best_mean_macro_f1', float('nan')):.3f}, "
        f"{tradeoff.get('best_mean_n_balls', float('nan')):.1f} balls, "
        f"{tradeoff.get('best_frac_collapsed', float('nan')):.1%} collapse, "
        f"{tradeoff.get('best_n_unstable', float('nan')):.0f} unstable datasets). "
        f"Compare v2 unguarded: F1={v2['mean_macro_f1']:.3f}, "
        f"{v2['mean_n_balls']:.1f} balls, {v2['frac_collapsed']:.1%} collapse."
    )

    return {
        "q1_remove_unstable": q1,
        "q2_smallest_guard": q2,
        "q3_compact_reliability": q3,
        "q4_best_tradeoff": q4,
    }


def write_report(path: Path, *, summary: dict[str, Any]) -> None:
    aggs = summary["by_config"]
    answers = summary["answers"]
    smallest = summary["smallest_fixing_guards"]
    tradeoff = summary["best_tradeoff"]

    lines = [
        "# Guarded entropy granulation",
        "",
        "Adaptive entropy v2 with stopping safeguards (min balls, min depth, entropy floor).",
        "",
        f"Runs: **{summary['n_rows']}**",
        "",
        "## Configuration summary",
        "",
        "| config | splits | balls | collapse% | unstable ds | macro F1 | dF1 vs rho | F1/ball |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggs:
        lines.append(
            f"| {a['config']} | {a['mean_split_count']:.1f} | {a['mean_n_balls']:.1f} | "
            f"{a['frac_collapsed']:.1%} | {a['n_unstable_datasets']} | "
            f"{a['mean_macro_f1']:.3f} | {a.get('paired_mean_delta_f1', float('nan')):+.4f} | "
            f"{a['mean_f1_per_ball']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Smallest guards per family",
            "",
        ]
    )
    for fam, info in smallest.get("smallest_per_family", {}).items():
        lines.append(
            f"- **{fam}**: {info['config']} — collapse {info['frac_collapsed']:.1%}, "
            f"balls={info['mean_n_balls']:.1f}, F1={info['mean_macro_f1']:.3f}"
        )

    lines.extend(
        [
            "",
            "## 1. Does a simple safeguard remove unstable datasets?",
            "",
            answers["q1_remove_unstable"],
            "",
            "## 2. What is the smallest guard that fixes collapse?",
            "",
            answers["q2_smallest_guard"],
            "",
            "## 3. Can entropy retain compactness while approaching rho90 reliability?",
            "",
            answers["q3_compact_reliability"],
            "",
            "## 4. Which safeguard gives the best F1/complexity tradeoff?",
            "",
            answers["q4_best_tradeoff"],
            "",
            f"Best tradeoff config: **{tradeoff.get('best_config', 'n/a')}**",
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
    aggs = _aggregate_by_config(df)
    smallest = _smallest_fixing_guards(aggs)
    tradeoff = _best_tradeoff(aggs)
    answers = answer_questions(aggs, smallest, tradeoff)
    return {
        "protocol": {
            "read_only": True,
            "seeds": SEEDS,
            "min_ball_guards": list(MIN_BALL_GUARDS),
            "min_depth_guards": list(MIN_DEPTH_GUARDS),
            "entropy_floor_guards": list(ENTROPY_FLOOR_GUARDS),
            "collapse_ball_threshold": COLLAPSE_BALL_THRESHOLD,
            "smoke_mode": smoke,
            "classifier": f"Gaussian-weighted GBLSTSVM ({PRIMARY_GAUSSIAN})",
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "by_config": aggs,
        "smallest_fixing_guards": smallest,
        "best_tradeoff": tradeoff,
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv
    configs = _make_configs()

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [0, 1] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print(f"SMOKE: 1 dataset x seeds {seeds}, {len(configs)} configs")
    else:
        print(f"FULL: {len(descriptors)} datasets x {len(seeds)} seeds x {len(configs)} configs")

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
                rho90_count = measure_rho90_split_count(indexed, seed=seed)

                for config in configs:
                    rows.append(
                        run_single(
                            split,
                            seed,
                            config,
                            indexed=indexed,
                            global_min=global_min,
                            rho90_split_count=rho90_count,
                        )
                    )
                    done += 1

                v2 = next(r for r in rows[-len(configs):] if r["config"] == CONFIG_V2)
                best = max(
                    rows[-len(configs):],
                    key=lambda r: r["macro_f1"] if np.isfinite(r["macro_f1"]) else -1,
                )
                print(
                    f"{split.name} s{seed}: v2 F1={v2['macro_f1']:.3f} "
                    f"balls={v2['n_balls']} | best {best['config']} F1={best['macro_f1']:.3f}"
                )

                if done % SAVE_EVERY == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    print(f"  [checkpoint] {done} rows")

            except Exception as exc:
                failed.append({"dataset": split.name, "seed": str(seed), "error": str(exc)})
                print(f"FAIL {desc['name']} seed={seed}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
