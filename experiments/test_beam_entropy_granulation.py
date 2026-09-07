"""
Beam-search entropy granulation experiment (read-only).

Tests whether preserving a global top-k refinement frontier fixes the
candidate-exhaustion collapse of online entropy scheduling.

Does NOT modify src/, solver, prediction, weighting, or baseline granulation.

Configs:
  baseline-rho90     — full BFS rho=0.90 granulation
  beam-entropy-k1    — beam width 1 (entropy baseline frontier)
  beam-entropy-k2    — beam width 2
  beam-entropy-k4    — beam width 4
  beam-entropy-k8    — beam width 8
  beam-entropy-k16   — beam width 16

Beam budget = rho90 split count per dataset×seed.

Run from GBLSTSVM root:
  python experiments/test_beam_entropy_granulation.py --smoke
  python experiments/test_beam_entropy_granulation.py
  python experiments/test_beam_entropy_granulation.py --from-csv
"""

from __future__ import annotations

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
from experiments.test_adaptive_entropy_granulation_v2 import (  # noqa: E402
    _finalize_with_depths,
    build_baseline_rho90,
    compute_extended_stats,
    measure_rho90_split_count,
)
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _labels,
    _to_ball_info,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "beam_entropy_granulation"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = list(range(10))
BEAM_WIDTHS = (1, 2, 4, 8, 16)
CONFIG_RHO90 = "baseline-rho90"
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
RHO_REF = RHO_VALUES["rho0.90"]
COLLAPSE_BALL_THRESHOLD = 3
SAVE_EVERY = 30
DELTA_EPS = 1e-6

CONFIGS: tuple[str, ...] = (CONFIG_RHO90,) + tuple(
    f"beam-entropy-k{k}" for k in BEAM_WIDTHS
)


@dataclass
class BeamGranulationResult:
    kept: list[GranularBall]
    depths: list[int]
    n_successful_splits: int
    n_failed_splits: int
    split_budget: int
    budget_met: bool
    beam_k: int
    pool_trajectory: list[int]
    terminated_no_candidates: bool
    final_relax_level: int


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


def _prune_pool(
    candidates: list[int],
    beam_k: int,
    balls: list[GranularBall],
    global_min: float,
) -> list[int]:
    unique = list(dict.fromkeys(candidates))
    unique.sort(key=lambda i: -_ball_entropy(balls[i], global_min))
    return unique[:beam_k]


def build_beam_entropy(
    indexed: np.ndarray,
    *,
    seed: int,
    beam_k: int,
    split_budget: int,
    global_min: float,
    rho90_split_count: int,
) -> BeamGranulationResult:
    """
    Beam-search entropy refinement.

    Maintains a global top-k pool of splittable leaf balls ranked by entropy.
    At each iteration: pick highest-entropy pool member, split, add children,
    prune pool to top-k. Relaxes purity threshold when pool empties before budget.
    """
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    is_leaf = [True]
    exhausted: set[int] = set()
    relax_level = 0
    n_success = 0
    n_failed = 0
    pool_trajectory: list[int] = []
    terminated_no_candidates = False

    def purity_thr() -> float:
        if relax_level <= 0:
            return RHO_REF
        if relax_level == 1:
            return PURITY
        return 1.0

    def splittable_leaf_indices() -> list[int]:
        return [
            i
            for i in range(len(balls))
            if is_leaf[i] and i not in exhausted and _splittable(balls[i], purity_thr())
        ]

    def repopulate_pool() -> list[int]:
        return _prune_pool(splittable_leaf_indices(), beam_k, balls, global_min)

    pool = repopulate_pool()
    max_relax = 2 if split_budget > rho90_split_count else 1

    while n_success < split_budget:
        if not pool:
            progressed = False
            while relax_level < max_relax:
                relax_level += 1
                pool = repopulate_pool()
                if pool:
                    progressed = True
                    break
            if not progressed:
                terminated_no_candidates = True
                break

        pool_trajectory.append(len(pool))
        pool.sort(key=lambda i: -_ball_entropy(balls[i], global_min))
        idx = pool.pop(0)

        if idx in exhausted or not is_leaf[idx] or not _splittable(balls[idx], purity_thr()):
            pool = _prune_pool(pool, beam_k, balls, global_min)
            continue

        parent = balls[idx]
        c1, c2 = parent.split_2balls()
        if not _children_valid(c1, c2):
            n_failed += 1
            exhausted.add(idx)
            pool = _prune_pool(pool, beam_k, balls, global_min)
            continue

        balls[idx] = c1
        balls.append(c2)
        depths.append(depths[idx] + 1)
        is_leaf[idx] = True
        is_leaf.append(True)
        n_success += 1

        new_entries = [i for i in pool if i not in exhausted]
        if _splittable(c1, purity_thr()) and idx not in exhausted:
            new_entries.append(idx)
        c2_idx = len(balls) - 1
        if _splittable(c2, purity_thr()) and c2_idx not in exhausted:
            new_entries.append(c2_idx)
        pool = _prune_pool(new_entries, beam_k, balls, global_min)

    kept, kept_depths = _finalize_with_depths(balls, depths)
    return BeamGranulationResult(
        kept=kept,
        depths=kept_depths,
        n_successful_splits=n_success,
        n_failed_splits=n_failed,
        split_budget=split_budget,
        budget_met=n_success >= split_budget,
        beam_k=beam_k,
        pool_trajectory=pool_trajectory,
        terminated_no_candidates=terminated_no_candidates,
        final_relax_level=relax_level,
    )


def _beam_config_name(beam_k: int) -> str:
    return f"beam-entropy-k{beam_k}"


def _parse_beam_k(config: str) -> int | None:
    if config == CONFIG_RHO90:
        return None
    if config.startswith("beam-entropy-k"):
        return int(config.replace("beam-entropy-k", ""))
    raise ValueError(f"Unknown config {config!r}")


def run_single(
    split,
    seed: int,
    config: str,
    *,
    indexed: np.ndarray,
    global_min: float,
    rho90_split_count: int,
) -> dict[str, Any]:
    beam_k = _parse_beam_k(config)
    if config == CONFIG_RHO90:
        result = build_baseline_rho90(indexed, seed=seed, split_budget=None)
        pool_trajectory: list[int] = []
        terminated_no_candidates = False
        final_relax_level = 0
        split_budget = rho90_split_count
    else:
        assert beam_k is not None
        beam_result = build_beam_entropy(
            indexed,
            seed=seed,
            beam_k=beam_k,
            split_budget=rho90_split_count,
            global_min=global_min,
            rho90_split_count=rho90_split_count,
        )
        result = beam_result
        pool_trajectory = beam_result.pool_trajectory
        terminated_no_candidates = beam_result.terminated_no_candidates
        final_relax_level = beam_result.final_relax_level
        split_budget = beam_result.split_budget

    rep = compute_extended_stats(result.kept, result.depths, indexed, global_min)
    ball_infos = [_to_ball_info(b) for b in result.kept]
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
    traj = pool_trajectory or [0]
    return {
        "dataset": split.name,
        "seed": seed,
        "config": config,
        "beam_k": beam_k if beam_k is not None else float("nan"),
        "rho90_split_count": rho90_split_count,
        "split_budget": split_budget,
        "split_count": result.n_successful_splits,
        "n_failed_splits": result.n_failed_splits,
        "budget_met": int(result.budget_met),
        "terminated_no_candidates": int(terminated_no_candidates),
        "final_relax_level": final_relax_level,
        "collapsed": int(collapsed),
        "n_balls": n_balls,
        "max_depth": rep["max_depth"],
        "mean_depth": rep["mean_depth"],
        "mean_entropy": rep["mean_entropy"],
        "mean_boundary_score": rep["mean_boundary_score"],
        "mean_purity": rep["mean_purity"],
        "minority_survival": rep["minority_survival"],
        "majority_survival": rep["majority_survival"],
        "fraction_mixed_balls": rep["fraction_mixed_balls"],
        "mean_pool_size": _nanmean(traj),
        "max_pool_size": float(max(traj)) if traj else 0.0,
        "final_pool_size": float(traj[-1]) if traj else 0.0,
        "pool_trajectory_len": len(traj),
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
    rows: list[dict[str, Any]] = []
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
                "delta_minority_f1": float(a["minority_f1"] - r["minority_f1"])
                if np.isfinite(a["minority_f1"]) and np.isfinite(r["minority_f1"])
                else float("nan"),
                "delta_gmean": float(a["gmean"] - r["gmean"])
                if np.isfinite(a["gmean"]) and np.isfinite(r["gmean"])
                else float("nan"),
                "delta_n_balls": float(a["n_balls"] - r["n_balls"]),
                "delta_splits": float(a["split_count"] - r["split_count"]),
                "beam_collapsed": int(a["collapsed"]),
                "rho_collapsed": int(r["collapsed"]),
                "beam_macro_f1": float(a["macro_f1"]),
                "rho_macro_f1": float(r["macro_f1"]),
                "beam_n_balls": float(a["n_balls"]),
                "rho_n_balls": float(r["n_balls"]),
                "mean_pool_size": float(a["mean_pool_size"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_config(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        sub = df[df["config"] == cfg]
        if sub.empty:
            continue
        rows.append(
            {
                "config": cfg,
                "beam_k": _parse_beam_k(cfg),
                "n_runs": int(len(sub)),
                "frac_collapsed": float(sub["collapsed"].mean()),
                "frac_budget_met": _nanmean(sub["budget_met"].tolist()),
                "frac_terminated_no_candidates": _nanmean(
                    sub["terminated_no_candidates"].tolist()
                ),
                "mean_split_count": _nanmean(sub["split_count"].tolist()),
                "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
                "mean_max_depth": _nanmean(sub["max_depth"].tolist()),
                "mean_entropy": _nanmean(sub["mean_entropy"].tolist()),
                "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
                "mean_purity": _nanmean(sub["mean_purity"].tolist()),
                "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
                "mean_pool_size": _nanmean(sub["mean_pool_size"].tolist()),
                "max_pool_size_mean": _nanmean(sub["max_pool_size"].tolist()),
                "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["gmean"].tolist()),
                "mean_f1_per_ball": _nanmean(sub["f1_per_ball"].tolist()),
            }
        )
    return rows


def _best_beam_tradeoff(aggs: list[dict[str, Any]]) -> dict[str, Any]:
    rho = next((a for a in aggs if a["config"] == CONFIG_RHO90), {})
    beam_aggs = [a for a in aggs if a["config"] != CONFIG_RHO90]
    if not beam_aggs:
        return {}
    scored = []
    for a in beam_aggs:
        score = (
            a["mean_macro_f1"]
            - 0.02 * a["frac_collapsed"]
            + 0.001 * a["mean_pool_size"]
        )
        scored.append((score, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    k1 = next((a for a in beam_aggs if a["beam_k"] == 1), {})
    return {
        "best_config": best["config"],
        "best_beam_k": best["beam_k"],
        "best_mean_macro_f1": best["mean_macro_f1"],
        "best_frac_collapsed": best["frac_collapsed"],
        "best_mean_n_balls": best["mean_n_balls"],
        "best_mean_pool_size": best["mean_pool_size"],
        "rho_mean_macro_f1": rho.get("mean_macro_f1", float("nan")),
        "rho_frac_collapsed": rho.get("frac_collapsed", float("nan")),
        "k1_frac_collapsed": k1.get("frac_collapsed", float("nan")),
        "k1_mean_pool_size": k1.get("mean_pool_size", float("nan")),
        "collapse_reduction_k1_to_best": k1.get("frac_collapsed", 1) - best["frac_collapsed"],
        "f1_gain_k1_to_best": best["mean_macro_f1"] - k1.get("mean_macro_f1", 0),
    }


def answer_questions(
    df: pd.DataFrame,
    aggs: list[dict[str, Any]],
    tradeoff: dict[str, Any],
) -> dict[str, str]:
    by_name = {a["config"]: a for a in aggs}
    rho = by_name.get(CONFIG_RHO90, {})
    k1 = by_name.get("beam-entropy-k1", {})
    best_cfg = tradeoff.get("best_config", "beam-entropy-k1")
    best = by_name.get(best_cfg, k1)

    q1 = (
        f"rho90 collapse **{rho.get('frac_collapsed', float('nan')):.1%}**; "
        f"beam-k1 **{k1.get('frac_collapsed', float('nan')):.1%}** "
        f"(mean pool {k1.get('mean_pool_size', float('nan')):.2f}); "
        f"beam-k{best.get('beam_k', '?')} **{best.get('frac_collapsed', float('nan')):.1%}** "
        f"(mean pool {best.get('mean_pool_size', float('nan')):.2f}). "
    )
    if best.get("frac_collapsed", 1) < k1.get("frac_collapsed", 1) - 0.05:
        q1 += (
            f"**Yes** — widening the beam from k=1 to k={best.get('beam_k')} "
            f"reduces collapse by "
            f"{(k1.get('frac_collapsed', 0) - best.get('frac_collapsed', 0)) * 100:.1f}pp."
        )
    elif best.get("frac_collapsed", 1) < k1.get("frac_collapsed", 1):
        q1 += "**Marginal** — beam widening slightly reduces collapse but does not eliminate it."
    else:
        q1 += "**No** — preserving a wider frontier does not materially reduce collapse."

    q2 = (
        f"Best tradeoff: **{best_cfg}** — macro F1 {best.get('mean_macro_f1', float('nan')):.3f}, "
        f"{best.get('mean_n_balls', float('nan')):.1f} balls, "
        f"collapse {best.get('frac_collapsed', float('nan')):.1%}, "
        f"mean pool {best.get('mean_pool_size', float('nan')):.2f}. "
    )
    if best.get("beam_k", 1) <= 4:
        q2 += "Modest beam widths suffice; very wide beams add pool overhead without proportional gain."
    else:
        q2 += "Wider beams are needed to maintain frontier diversity on this dataset mix."

    delta_f1 = best.get("mean_macro_f1", 0) - rho.get("mean_macro_f1", 0)
    delta_collapse = best.get("frac_collapsed", 0) - rho.get("frac_collapsed", 0)
    q3 = (
        f"rho90: F1 {rho.get('mean_macro_f1', float('nan')):.3f}, "
        f"collapse {rho.get('frac_collapsed', float('nan')):.1%}, "
        f"{rho.get('mean_n_balls', float('nan')):.1f} balls. "
        f"Best beam: F1 {best.get('mean_macro_f1', float('nan')):.3f} "
        f"(Δ {delta_f1:+.3f}), collapse {best.get('frac_collapsed', float('nan')):.1%} "
        f"(Δ {delta_collapse:+.1%}). "
    )
    if abs(delta_f1) < 0.01 and delta_collapse < -0.05:
        q3 += "Beam search matches rho90 F1 with lower collapse — reliability improves."
    elif delta_f1 >= 0 and delta_collapse <= 0:
        q3 += "Beam approaches rho90 on both F1 and collapse."
    elif delta_f1 < -0.02:
        q3 += "Beam does not yet match rho90 reliability; BFS breadth still wins."
    else:
        q3 += "Partial convergence — beam improves frontier maintenance but gap to rho90 remains."

    pool_gain = best.get("mean_pool_size", 0) - k1.get("mean_pool_size", 0)
    f1_gain = tradeoff.get("f1_gain_k1_to_best", 0)
    q4 = (
        f"k=1 mean pool {k1.get('mean_pool_size', float('nan')):.2f} vs "
        f"k={best.get('beam_k')} pool {best.get('mean_pool_size', float('nan')):.2f} "
        f"(+{pool_gain:.2f}). "
        f"F1 gain k1→best: {f1_gain:+.3f}. "
    )
    if pool_gain > 0.5 and f1_gain > 0.005:
        q4 += (
            "**Yes** — gains track frontier width, not a different scoring function; "
            "the bottleneck is search/scheduler structure."
        )
    elif pool_gain > 0.1:
        q4 += (
            "**Mostly search** — wider pool correlates with improvement; "
            "entropy scoring unchanged across beam widths."
        )
    else:
        q4 += "**Unclear** — pool size barely increases; collapse may need structural changes beyond beam width."

    return {
        "q1_frontier_removes_collapse": q1,
        "q2_best_beam_width": q2,
        "q3_approach_rho90": q3,
        "q4_search_vs_scoring": q4,
    }


def write_report(path: Path, *, summary: dict[str, Any]) -> None:
    aggs = summary["by_config"]
    answers = summary["answers"]
    tradeoff = summary["beam_tradeoff"]

    lines = [
        "# Beam entropy granulation",
        "",
        "Read-only experiment: beam-search entropy refinement with global top-k "
        "candidate pool. Budget = rho90 split count.",
        "",
        f"Runs: **{summary['n_rows']}** ({summary['n_dataset_seed_pairs']} dataset×seed × "
        f"{len(CONFIGS)} configs)",
        "",
        "## Config summary",
        "",
        "| config | macro F1 | minority F1 | G-mean | balls | splits | collapse% | "
        "budget met% | no-cand% | mean pool | max pool |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggs:
        lines.append(
            f"| {a['config']} | {a['mean_macro_f1']:.3f} | "
            f"{a['mean_minority_f1']:.3f} | {a['mean_gmean']:.3f} | "
            f"{a['mean_n_balls']:.1f} | {a['mean_split_count']:.1f} | "
            f"{a['frac_collapsed']:.1%} | {a['frac_budget_met']:.1%} | "
            f"{a['frac_terminated_no_candidates']:.1%} | "
            f"{a['mean_pool_size']:.2f} | {a['max_pool_size_mean']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Paired deltas vs rho90 (best beam)",
            "",
            f"- Best beam: **{tradeoff.get('best_config', 'n/a')}** (k={tradeoff.get('best_beam_k', 'n/a')})",
            f"- Δ macro F1 vs rho90: "
            f"{tradeoff.get('best_mean_macro_f1', float('nan')) - tradeoff.get('rho_mean_macro_f1', 0):+.3f}",
            f"- Collapse reduction k1→best: "
            f"{tradeoff.get('collapse_reduction_k1_to_best', float('nan')) * 100:.1f}pp",
            "",
            "## 1. Does preserving a refinement frontier remove collapse?",
            "",
            answers["q1_frontier_removes_collapse"],
            "",
            "## 2. What beam width gives the best compactness/performance tradeoff?",
            "",
            answers["q2_best_beam_width"],
            "",
            "## 3. Does beam search approach rho90 reliability?",
            "",
            answers["q3_approach_rho90"],
            "",
            "## 4. Is the improvement from better search rather than better split scoring?",
            "",
            answers["q4_search_vs_scoring"],
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
    tradeoff = _best_beam_tradeoff(aggs)
    paired: dict[str, Any] = {}
    for k in BEAM_WIDTHS:
        cfg = _beam_config_name(k)
        p = _paired_vs_rho90(df, cfg)
        if not p.empty:
            paired[cfg] = {
                "mean_delta_macro_f1": _nanmean(p["delta_macro_f1"].tolist()),
                "mean_delta_n_balls": _nanmean(p["delta_n_balls"].tolist()),
                "frac_beam_collapsed": float(p["beam_collapsed"].mean()),
                "frac_rho_collapsed": float(p["rho_collapsed"].mean()),
            }
    answers = answer_questions(df, aggs, tradeoff)
    return {
        "protocol": {
            "read_only": True,
            "configs": list(CONFIGS),
            "beam_widths": list(BEAM_WIDTHS),
            "entropy_budget": "rho90_split_count (100%)",
            "collapse_ball_threshold": COLLAPSE_BALL_THRESHOLD,
            "seeds": SEEDS,
            "smoke_mode": smoke,
            "classifier": f"Gaussian-weighted GBLSTSVM ({PRIMARY_GAUSSIAN})",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups)
        if not df.empty
        else 0,
        "by_config": aggs,
        "paired_vs_rho90": paired,
        "beam_tradeoff": tradeoff,
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print(f"SMOKE: 1 dataset × seed 42 × {len(CONFIGS)} configs")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds × {len(CONFIGS)} configs")

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

                for config in CONFIGS:
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
                rho = next(r for r in rows[-len(CONFIGS):] if r["config"] == CONFIG_RHO90)
                k1 = next(r for r in rows[-len(CONFIGS):] if r["config"] == "beam-entropy-k1")
                k16 = next(r for r in rows[-len(CONFIGS):] if r["config"] == "beam-entropy-k16")
                print(
                    f"{split.name} s{seed}: rho F1={rho['macro_f1']:.3f} "
                    f"({rho['n_balls']} balls) | k1 F1={k1['macro_f1']:.3f} "
                    f"pool={k1['mean_pool_size']:.1f} coll={k1['collapsed']} | "
                    f"k16 F1={k16['macro_f1']:.3f} pool={k16['mean_pool_size']:.1f}"
                )

                if done % SAVE_EVERY == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    print(f"  [checkpoint] {done} dataset×seed pairs")

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
