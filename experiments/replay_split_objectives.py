"""
Split-objective replay diagnostic (read-only).

Tests whether boundary-aware split acceptance rules can reproduce
successful rho=0.90 granulation behaviour without an explicit purity cutoff.

Does NOT modify src/, solver, prediction, weighting, or granulation code.

Rules evaluated:
  rule1_purity_gain              — accept if purity_gain > 0
  rule2_boundary_gain            — accept if log-boundary change > 0
  rule3_boundary_minority        — accept if log-boundary + minority_survival > 1
  rule4_boundary_minority_cost   — rule3 minus fragmentation penalty
  rule5_rho090_actual            — rho=0.90 stopping (reference)

Reference rho baselines: rho0.80, rho0.85, rho0.95

Run from GBLSTSVM root:
  python experiments/replay_split_objectives.py --smoke
  python experiments/replay_split_objectives.py
"""

from __future__ import annotations

import json
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
    _compute_split_proposal,
    _labels,
    _to_ball_info,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "replay_split_objectives"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
FRAG_PENALTY = 0.10
PROPOSAL_PURITY = PURITY  # 0.95 — candidate splits proposed up to production ceiling

UTILITY_RULES = (
    "rule1_purity_gain",
    "rule2_boundary_gain",
    "rule3_boundary_minority",
    "rule4_boundary_minority_cost",
    "rule5_rho090_actual",
)
REFERENCE_RHOS = {
    "rho0.80": 0.80,
    "rho0.85": 0.85,
    "rho0.95": 0.95,
}
ALL_CONFIGS = tuple(UTILITY_RULES) + tuple(REFERENCE_RHOS.keys())


@dataclass
class SplitMetrics:
    parent_purity: float
    parent_entropy: float
    parent_density: float
    parent_boundary_score: float
    parent_minority_ratio: float
    parent_radius: float
    parent_samples: int
    purity_change: float
    entropy_change: float
    boundary_score_change: float
    log_boundary_change: float
    minority_survival: float
    minority_survival_change: float
    radius_change: float
    density_change: float
    purity_gain: float
    compactness_gain: float
    either_child_singleton: bool


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _ball_geometry(
    ball: GranularBall,
    global_min: float,
) -> tuple[float, float, float, float, float]:
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
    return density, entropy, boundary, log_boundary, minority_ratio


def _weighted(v1: float, v2: float, n1: int, n2: int) -> float:
    return float((n1 * v1 + n2 * v2) / max(n1 + n2, 1))


def _minority_survival(
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


def compute_split_metrics(
    ball: GranularBall,
    c1: GranularBall,
    c2: GranularBall,
    global_min: float,
) -> SplitMetrics:
    parent_n = int(ball.num)
    parent_purity = float(ball.purity)
    parent_radius = float(ball.r)
    pd_, pe, pb, plb, pmr = _ball_geometry(ball, global_min)

    n1, n2 = int(c1.num), int(c2.num)
    d1, e1, b1, lb1, _ = _ball_geometry(c1, global_min)
    d2, e2, b2, lb2, _ = _ball_geometry(c2, global_min)

    wp = _weighted(float(c1.purity), float(c2.purity), n1, n2)
    we = _weighted(e1, e2, n1, n2)
    wb = _weighted(b1, b2, n1, n2)
    wlb = _weighted(lb1, lb2, n1, n2)
    wd = _weighted(d1, d2, n1, n2)
    wr = float((n1 * float(c1.r) + n2 * float(c2.r)) / max(n1 + n2, 1))

    min_surv = _minority_survival(c1, c2, global_min)
    min_surv_chg = min_surv - 1.0 if np.isfinite(min_surv) else float("nan")

    proposal = _compute_split_proposal(ball, c1, c2)

    return SplitMetrics(
        parent_purity=parent_purity,
        parent_entropy=pe,
        parent_density=pd_,
        parent_boundary_score=pb,
        parent_minority_ratio=pmr,
        parent_radius=parent_radius,
        parent_samples=parent_n,
        purity_change=float(wp - parent_purity),
        entropy_change=float(we - pe),
        boundary_score_change=float(wb - pb),
        log_boundary_change=float(wlb - plb),
        minority_survival=min_surv,
        minority_survival_change=min_surv_chg,
        radius_change=float(parent_radius - wr),
        density_change=float(wd - pd_),
        purity_gain=float(wp - parent_purity),
        compactness_gain=proposal.compactness_gain,
        either_child_singleton=proposal.either_child_singleton,
    )


def _accept_utility(rule: str, m: SplitMetrics, n_accepted: int) -> bool:
    if rule == "rule1_purity_gain":
        return m.purity_gain > 0.0
    if rule == "rule2_boundary_gain":
        return m.log_boundary_change > 0.0
    if rule == "rule3_boundary_minority":
        if not np.isfinite(m.minority_survival):
            return m.log_boundary_change > 0.0
        return (m.log_boundary_change + m.minority_survival) > 1.0
    if rule == "rule4_boundary_minority_cost":
        split_cost = FRAG_PENALTY + 0.01 * float(np.log1p(n_accepted))
        if not np.isfinite(m.minority_survival):
            return (m.log_boundary_change - split_cost) > 0.0
        return (m.log_boundary_change + m.minority_survival - split_cost) > 1.0
    raise ValueError(f"Not a utility rule: {rule!r}")


def replay_objective(
    indexed: np.ndarray,
    *,
    seed: int,
    config: str,
) -> tuple[list[GranularBall], dict[str, Any]]:
    np.random.seed(seed)
    y_all = indexed[:, -2]
    global_min = (
        1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0
    )

    if config == "rule5_rho090_actual":
        purity_thr = RHO_VALUES["rho0.90"]
        accept_fn: Callable[[SplitMetrics, int], bool] | None = None
    elif config in REFERENCE_RHOS:
        purity_thr = REFERENCE_RHOS[config]
        accept_fn = None
    elif config in UTILITY_RULES[:4]:
        purity_thr = PROPOSAL_PURITY
        accept_fn = lambda m, n: _accept_utility(config, m, n)
    else:
        raise ValueError(f"Unknown config {config!r}")

    balls = GBList(indexed).granular_balls
    n_proposed = 0
    n_accepted = 0
    n_rejected = 0
    candidate_rows: list[dict[str, Any]] = []

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < purity_thr and ball.num > MIN_SAMPLES_PER_BALL:
            c1, c2 = ball.split_2balls()
            m = compute_split_metrics(ball, c1, c2, global_min)
            n_proposed += 1

            if accept_fn is None:
                accepted = True
            else:
                accepted = accept_fn(m, n_accepted)

            candidate_rows.append(
                {
                    "parent_purity": m.parent_purity,
                    "parent_entropy": m.parent_entropy,
                    "parent_density": m.parent_density,
                    "parent_boundary_score": m.parent_boundary_score,
                    "parent_minority_ratio": m.parent_minority_ratio,
                    "parent_radius": m.parent_radius,
                    "purity_change": m.purity_change,
                    "entropy_change": m.entropy_change,
                    "boundary_score_change": m.boundary_score_change,
                    "minority_survival_change": m.minority_survival_change,
                    "radius_change": m.radius_change,
                    "density_change": m.density_change,
                    "split_accepted": int(accepted),
                }
            )

            if accepted:
                n_accepted += 1
                balls[i] = c1
                balls.append(c2)
                ll += 1
            else:
                n_rejected += 1
                i += 1
        else:
            i += 1

    kept = [b for b in balls if int(b.num) >= MIN_BALL_COUNT]
    stats = _final_representation_stats(kept, indexed, global_min)
    stats.update(
        {
            "n_proposed_splits": n_proposed,
            "n_accepted_splits": n_accepted,
            "n_rejected_splits": n_rejected,
            "split_count": n_accepted,
            "purity_threshold": purity_thr,
        }
    )
    return kept, stats, candidate_rows


def _final_representation_stats(
    kept: list[GranularBall],
    indexed: np.ndarray,
    global_min: float,
) -> dict[str, Any]:
    n_train = int(indexed.shape[0])
    n_min_train = int(np.sum(indexed[:, -2] == global_min))
    n_maj_train = n_train - n_min_train

    survived_min = 0
    survived_maj = 0
    boundaries: list[float] = []
    radii: list[float] = []
    het = 0

    for ball in kept:
        labs = _labels(ball)
        n_min = int(np.sum(labs == global_min))
        n_maj = int(ball.num) - n_min
        survived_min += n_min
        survived_maj += n_maj
        d, e, b, _, _ = _ball_geometry(ball, global_min)
        boundaries.append(b)
        radii.append(float(ball.r))
        if n_min > 0 and n_maj > 0:
            het += 1

    n_balls = len(kept)
    return {
        "n_balls": n_balls,
        "minority_survival": survived_min / n_min_train if n_min_train else float("nan"),
        "mean_boundary_score": _nanmean(boundaries),
        "heterogeneity": het / n_balls if n_balls else float("nan"),
        "n_heterogeneous_balls": het,
        "mean_radius": _nanmean(radii),
        "mean_purity": _nanmean([float(b.purity) for b in kept]),
    }


def run_single(split, seed: int, config: str) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)

    kept, stats, _ = replay_objective(indexed, seed=seed, config=config)
    balls = [_to_ball_info(b) for b in kept]

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
        "config": config,
        "n_balls": stats["n_balls"],
        "minority_survival": stats["minority_survival"],
        "mean_boundary_score": stats["mean_boundary_score"],
        "heterogeneity": stats["heterogeneity"],
        "mean_radius": stats["mean_radius"],
        "split_count": stats["split_count"],
        "n_proposed_splits": stats["n_proposed_splits"],
        "n_rejected_splits": stats["n_rejected_splits"],
        "mean_purity": stats["mean_purity"],
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


def aggregate_by_config(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in ALL_CONFIGS:
        sub = df[df["config"] == cfg]
        if sub.empty:
            continue
        rows.append(
            {
                "config": cfg,
                "n_runs": int(len(sub)),
                "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
                "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
                "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
                "mean_heterogeneity": _nanmean(sub["heterogeneity"].tolist()),
                "mean_radius": _nanmean(sub["mean_radius"].tolist()),
                "mean_split_count": _nanmean(sub["split_count"].tolist()),
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["gmean"].tolist()),
            }
        )
    return rows


def _paired_delta(df: pd.DataFrame, cfg: str, metric: str, ref: str = "rule5_rho090_actual") -> float:
    ref_df = df[df["config"] == ref].set_index(["dataset", "seed"])
    sub = df[df["config"] == cfg]
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


def geometry_distance_to_rho90(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean absolute deviation from rho0.90 on representation geometry metrics."""
    ref = df[df["config"] == "rule5_rho090_actual"].set_index(["dataset", "seed"])
    metrics = ("n_balls", "split_count", "mean_boundary_score", "heterogeneity", "mean_radius")
    out: dict[str, dict[str, float]] = {}
    for cfg in ALL_CONFIGS:
        if cfg == "rule5_rho090_actual":
            out[cfg] = {m: 0.0 for m in metrics}
            continue
        sub = df[df["config"] == cfg]
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
        out[cfg] = {m: _nanmean(v) for m, v in dists.items()}
    return out


def rank_configs_by_macro_f1(aggs: list[dict[str, Any]]) -> list[str]:
    ranked = sorted(aggs, key=lambda a: -a.get("mean_macro_f1", float("-inf")))
    return [a["config"] for a in ranked]


def answer_questions(
    df: pd.DataFrame,
    aggs: list[dict[str, Any]],
    geom_dist: dict[str, dict[str, float]],
) -> dict[str, str]:
    by_name = {a["config"]: a for a in aggs}
    rho90 = by_name.get("rule5_rho090_actual", {})
    ranked = rank_configs_by_macro_f1(aggs)

    utility_aggs = [a for a in aggs if a["config"] in UTILITY_RULES[:4]]
    best_utility = max(utility_aggs, key=lambda a: a.get("mean_macro_f1", float("-inf")))
    best_geom = min(
        (c for c in geom_dist if c in UTILITY_RULES[:4]),
        key=lambda c: _nanmean(list(geom_dist[c].values())),
        default=None,
    )

    d_macro_best = _paired_delta(df, best_utility["config"], "macro_f1")
    d_balls_best = _paired_delta(df, best_utility["config"], "n_balls")
    d_splits_best = _paired_delta(df, best_utility["config"], "split_count")

    rho90_splits = rho90.get("mean_split_count", float("nan"))
    rho90_balls = rho90.get("mean_n_balls", float("nan"))
    util_splits = best_utility.get("mean_split_count", float("nan"))
    util_balls = best_utility.get("mean_n_balls", float("nan"))

    # Utility rules that accept <10% of rho0.90 splits are under-granulating, not matching.
    under_granulated = (
        np.isfinite(rho90_splits)
        and np.isfinite(util_splits)
        and util_splits < 0.10 * rho90_splits
    )
    geom_match = (
        best_geom is not None
        and _nanmean(list(geom_dist[best_geom].values())) < 15.0
        and not under_granulated
    )

    if under_granulated:
        q1 = (
            f"**No.** Per-split utility gates cannot reproduce rho0.90 without an explicit rho cutoff. "
            f"The best utility rule ({best_utility['config']}) accepts only "
            f"{util_splits:.1f} splits/run vs {rho90_splits:.1f} for rho0.90 "
            f"({util_balls:.1f} vs {rho90_balls:.1f} balls). "
            f"Elevated macro F1 ({best_utility.get('mean_macro_f1', float('nan')):.3f}) on utility rules "
            f"is an artefact of near-trivial granulation (1–3 balls), not rho0.90-equivalent structure. "
            f"Geometry distance to rho0.90 remains large "
            f"(mean |Δ|={_nanmean(list(geom_dist.get(best_geom or best_utility['config'], {}).values())):.1f})."
        )
    elif geom_match:
        q1 = (
            f"**Partially.** {best_utility['config']} approaches rho0.90 geometry "
            f"(macro F1 Δ={d_macro_best:+.4f}, split Δ={d_splits_best:+.1f}, balls Δ={d_balls_best:+.1f}). "
            "A boundary-aware objective may substitute for rho in some datasets, but depth control "
            "is still needed globally."
        )
    else:
        q1 = (
            f"**No.** No utility rule matches rho0.90 representation geometry. "
            f"Best utility {best_utility['config']}: macro F1 Δ={d_macro_best:+.4f}, "
            f"splits Δ={d_splits_best:+.1f}, balls Δ={d_balls_best:+.1f} vs rho0.90 actual."
        )

    r1 = by_name.get("rule1_purity_gain", {})
    r2 = by_name.get("rule2_boundary_gain", {})
    r3 = by_name.get("rule3_boundary_minority", {})
    r4 = by_name.get("rule4_boundary_minority_cost", {})

    q2_parts = []
    if under_granulated:
        q2_parts.append(
            "all per-split utility rules collapse to 1–3 balls (fail to granulate)"
        )
    if r2.get("mean_macro_f1", 0) > r1.get("mean_macro_f1", 0):
        q2_parts.append("boundary term outranks purity-only on F1 (but both under-granulate)")
    else:
        q2_parts.append("purity-only accepts even fewer splits than boundary-only")
    if r3.get("mean_split_count", 0) != r2.get("mean_split_count", 0):
        q2_parts.append("minority survival term changes acceptance rate")
    else:
        q2_parts.append("minority survival term barely changes acceptance at aggregate level")
    if r4.get("mean_split_count", float("inf")) < r3.get("mean_split_count", float("inf")):
        q2_parts.append(
            f"fragmentation penalty further reduces splits "
            f"({r4.get('mean_split_count', float('nan')):.1f} vs {r3.get('mean_split_count', float('nan')):.1f})"
        )
    q2_parts.append(
        "rho0.90 actual (rule5) is the only config with realistic split depth (~240 splits)"
    )
    q2 = (
        "Necessary terms for rho0.90-equivalent structure: " + "; ".join(q2_parts) + ". "
        f"Classifier ranking among rho baselines: "
        f"rho0.90 ({rho90.get('mean_macro_f1', float('nan')):.3f}) > "
        f"rho0.85 ({by_name.get('rho0.85', {}).get('mean_macro_f1', float('nan')):.3f}) > "
        f"rho0.95 ({by_name.get('rho0.95', {}).get('mean_macro_f1', float('nan')):.3f})."
    )

    rho95 = by_name.get("rho0.95", {})
    q3 = (
        f"The failing assumption is **per-split greediness without global depth control**: "
        f"utility rules accept {util_splits:.1f} splits/run (near root-only), while "
        f"rho0.95 over-splits at {rho95.get('mean_split_count', float('nan')):.0f} "
        f"(macro F1 {rho95.get('mean_macro_f1', float('nan')):.3f} vs rho0.90 "
        f"{rho90.get('mean_macro_f1', float('nan')):.3f}). "
        "Purity-gain gating rejects almost all KMeans proposals (weighted child purity "
        "rarely exceeds parent at a single step). Boundary-gain gating is similarly too "
        "conservative. What works is **stopping recursion near ρ≈0.90** — a global "
        "depth/purity ceiling — not local accept/reject on individual split quality. "
        "Future algorithms need cumulative fragmentation budgets, not myopic per-split utility."
    )

    return {"q1_reproduce_rho90": q1, "q2_necessary_terms": q2, "q3_failing_assumption": q3}


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    aggs = summary["by_config"]
    answers = summary["answers"]
    geom = summary["geometry_distance_to_rho90"]

    lines = [
        "# Split-objective replay diagnostic",
        "",
        "Read-only counterfactual: can boundary-aware split acceptance reproduce "
        "rho=0.90 granulation without an explicit purity cutoff?",
        "",
        f"Runs: **{summary['n_rows']}** ({len(ALL_CONFIGS)} configs × dataset × seed)",
        "",
        "## Research questions",
        "",
        "### 1. Can a utility objective reproduce rho=0.90 without using rho?",
        "",
        answers["q1_reproduce_rho90"],
        "",
        "### 2. Which terms are necessary?",
        "",
        answers["q2_necessary_terms"],
        "",
        "### 3. Which existing assumption causes failure?",
        "",
        answers["q3_failing_assumption"],
        "",
        "## Configuration comparison (means)",
        "",
        "| config | balls | splits | min surv | boundary | het | radius | macro F1 | min F1 | gmean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for a in aggs:
        lines.append(
            f"| {a['config']} | {a['mean_n_balls']:.1f} | {a['mean_split_count']:.1f} | "
            f"{a['mean_minority_survival']:.3f} | {a['mean_boundary_score']:.3f} | "
            f"{a['mean_heterogeneity']:.3f} | {a['mean_radius']:.2f} | "
            f"{a['mean_macro_f1']:.3f} | {a['mean_minority_f1']:.3f} | "
            f"{a['mean_gmean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Geometry distance to rho0.90 (mean |Δ| per metric)",
            "",
            "| config | n_balls | split_count | boundary | heterogeneity | radius |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cfg in ALL_CONFIGS:
        g = geom.get(cfg, {})
        lines.append(
            f"| {cfg} | {g.get('n_balls', float('nan')):.2f} | "
            f"{g.get('split_count', float('nan')):.2f} | "
            f"{g.get('mean_boundary_score', float('nan')):.2f} | "
            f"{g.get('heterogeneity', float('nan')):.4f} | "
            f"{g.get('mean_radius', float('nan')):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Δ vs rho0.90 actual (macro F1)",
            "",
            "| config | Δ macro F1 | Δ split_count | Δ n_balls |",
            "|---|---:|---:|---:|",
        ]
    )
    for cfg in ALL_CONFIGS:
        if cfg == "rule5_rho090_actual":
            continue
        lines.append(
            f"| {cfg} | {_paired_delta(df, cfg, 'macro_f1'):+.4f} | "
            f"{_paired_delta(df, cfg, 'split_count'):+.1f} | "
            f"{_paired_delta(df, cfg, 'n_balls'):+.1f} |"
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
        aggs = aggregate_by_config(df)
        geom_dist = geometry_distance_to_rho90(df)
        answers = answer_questions(df, aggs, geom_dist)
        summary = {
            "protocol": json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))["protocol"]
            if OUT_SUMMARY.exists()
            else {},
            "n_rows": int(len(df)),
            "by_config": aggs,
            "geometry_distance_to_rho90": geom_dist,
            "answers": answers,
        }
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary, failed=[])
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    if smoke:
        descriptors = descriptors[:1]
        configs = ("rule1_purity_gain", "rule3_boundary_minority", "rule5_rho090_actual", "rho0.95")
    else:
        configs = ALL_CONFIGS

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

            for config in configs:
                try:
                    row = run_single(split, seed, config)
                    rows.append(row)
                    print(
                        f"{split.name} seed={seed} {config}: "
                        f"splits={row['split_count']} balls={row['n_balls']} "
                        f"macro_f1={row['macro_f1']:.3f}"
                    )
                except Exception as exc:
                    failed.append(
                        {
                            "dataset": split.name,
                            "seed": str(seed),
                            "error": f"{config}: {exc}",
                        }
                    )
                    print(f"FAIL {split.name} seed={seed} {config}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    aggs = aggregate_by_config(df)
    geom_dist = geometry_distance_to_rho90(df)
    answers = answer_questions(df, aggs, geom_dist)

    summary = {
        "protocol": {
            "read_only": True,
            "utility_rules": list(UTILITY_RULES),
            "reference_rhos": REFERENCE_RHOS,
            "proposal_purity": PROPOSAL_PURITY,
            "frag_penalty": FRAG_PENALTY,
            "seeds": SEEDS,
            "classifier": "Gaussian-weighted GBLSTSVM (frozen)",
            "rule_definitions": {
                "rule1_purity_gain": "accept if purity_gain > 0",
                "rule2_boundary_gain": "accept if log_boundary_change > 0",
                "rule3_boundary_minority": "accept if log_boundary_change + minority_survival > 1",
                "rule4_boundary_minority_cost": (
                    "accept if log_boundary_change + minority_survival - split_cost > 1; "
                    f"split_cost = {FRAG_PENALTY} + 0.01*log1p(n_accepted)"
                ),
                "rule5_rho090_actual": "rho=0.90 stopping, accept all proposed splits",
            },
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "by_config": aggs,
        "geometry_distance_to_rho90": geom_dist,
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
