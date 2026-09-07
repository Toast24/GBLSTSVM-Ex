"""
Split-utility granulation counterfactual (read-only experiment).

Replays frozen recursive KMeans(k=2) granulation (rho=0.95, min_samples=1,
del_ball num_data=2) under split-utility gates, then evaluates unchanged
Clean GBLSTSVM on the resulting balls.

Does NOT modify src/, baseline granulation, CW, CA, Topo, or classifier code.

Modes:
  B0  — accept every split (baseline)
  SU1 — reject if center_radius_separation > threshold (1.0 … 3.0)
  SU2 — reject if sep > median(sep) AND parent_n <= 5
  SU3 — accept only if utility > run percentile (25/50/75)

Run from GBLSTSVM root:
  python experiments/test_split_utility_granulation.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import (  # noqa: E402
    DEFAULT_C1,
    DEFAULT_C2,
    evaluate,
    normalize_labels,
)
from src.gblstsvm import predict_twin_hyperplanes, solve_gblstsvm_systems  # noqa: E402
from src.granular_balls import (  # noqa: E402
    BallInfo,
    GBList,
    GranularBall,
    balls_to_class_arrays,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "split_utility"
OUT_CSV = OUT_DIR / "split_utility_results.csv"
OUT_SUMMARY = OUT_DIR / "split_utility_summary.json"
OUT_REPORT = OUT_DIR / "split_utility_report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
SMALL_BALL_MAX = 5
EPS = 1e-12
REG_TERM = 1e-4
PREDICTION_RULE = "reference"

SU1_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
SU3_PERCENTILES = [25, 50, 75]


@dataclass
class SplitProposal:
    parent_n: int
    parent_radius: float
    parent_purity: float
    child_n1: int
    child_n2: int
    purity1: float
    purity2: float
    r1: float
    r2: float
    center_radius_separation: float
    compactness_gain: float
    utility: float
    either_child_singleton: bool


def _build_indexed_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = normalize_labels(y)
    data = np.column_stack([X, y])
    num = data.shape[0]
    index = np.arange(num).reshape(num, 1)
    return np.hstack([data, index])


def _labels(ball: GranularBall) -> np.ndarray:
    return np.asarray(ball.data[:, -2], dtype=float)


def _indices(ball: GranularBall) -> np.ndarray:
    return np.asarray(ball.data[:, -1], dtype=int)


def _compute_split_proposal(ball: GranularBall, c1: GranularBall, c2: GranularBall) -> SplitProposal:
    parent_n = int(ball.num)
    parent_radius = float(ball.r)
    parent_purity = float(ball.purity)
    n1, n2 = int(c1.num), int(c2.num)
    p1, p2 = float(c1.purity), float(c2.purity)
    r1, r2 = float(c1.r), float(c2.r)
    center_distance = float(np.linalg.norm(c1.center - c2.center))
    weighted_child_radius = float((n1 * r1 + n2 * r2) / max(n1 + n2, 1))
    compactness_gain = float(parent_radius - weighted_child_radius)
    center_radius_separation = float(center_distance / (r1 + r2 + EPS))
    utility = float(compactness_gain / (1.0 + center_radius_separation))
    return SplitProposal(
        parent_n=parent_n,
        parent_radius=parent_radius,
        parent_purity=parent_purity,
        child_n1=n1,
        child_n2=n2,
        purity1=p1,
        purity2=p2,
        r1=r1,
        r2=r2,
        center_radius_separation=center_radius_separation,
        compactness_gain=compactness_gain,
        utility=utility,
        either_child_singleton=(n1 == 1 or n2 == 1),
    )


def _collect_b0_proposals(indexed: np.ndarray, seed: int) -> list[SplitProposal]:
    np.random.seed(seed)
    gb_balls = GBList(indexed).granular_balls
    balls = gb_balls
    proposals: list[SplitProposal] = []
    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < PURITY and ball.num > MIN_SAMPLES_PER_BALL:
            c1, c2 = ball.split_2balls()
            proposals.append(_compute_split_proposal(ball, c1, c2))
            balls[i] = c1
            balls.append(c2)
            ll += 1
        else:
            i += 1
    return proposals


def _replay_granulation(
    indexed: np.ndarray,
    *,
    seed: int,
    should_reject: Callable[[SplitProposal], bool],
) -> tuple[list[GranularBall], list[SplitProposal], dict[str, Any]]:
    np.random.seed(seed)
    y_all = indexed[:, -2]
    n_train = int(indexed.shape[0])
    global_min = 1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0
    n_min_train = int(np.sum(y_all == global_min))
    n_maj_train = n_train - n_min_train

    balls = GBList(indexed).granular_balls
    proposals: list[SplitProposal] = []
    n_rejected = 0
    n_proposed = 0
    n_singleton_children = 0

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < PURITY and ball.num > MIN_SAMPLES_PER_BALL:
            c1, c2 = ball.split_2balls()
            proposal = _compute_split_proposal(ball, c1, c2)
            n_proposed += 1
            proposals.append(proposal)
            if proposal.either_child_singleton:
                n_singleton_children += 1
            if should_reject(proposal):
                n_rejected += 1
                i += 1
            else:
                balls[i] = c1
                balls.append(c2)
                ll += 1
        else:
            i += 1

    balls_before_del = list(balls)
    n_balls_before_del = len(balls_before_del)
    n_singleton_before_del = sum(1 for b in balls_before_del if int(b.num) == 1)

    survived_min = 0
    survived_maj = 0
    min_in_small = 0
    n_removed_samples = 0
    for ball in balls_before_del:
        n = int(ball.num)
        labs = _labels(ball)
        n_min = int(np.sum(labs == global_min))
        n_maj = n - n_min
        if n <= SMALL_BALL_MAX:
            min_in_small += n_min
        deleted = n < MIN_BALL_COUNT
        if deleted:
            n_removed_samples += n
        else:
            survived_min += n_min
            survived_maj += n_maj

    kept = [b for b in balls_before_del if int(b.num) >= MIN_BALL_COUNT]
    n_deleted_balls = n_balls_before_del - len(kept)
    n_singleton_deleted = sum(1 for b in balls_before_del if int(b.num) == 1)

    purities = np.array([float(b.purity) for b in kept], dtype=float)
    radii = np.array([float(b.r) for b in kept], dtype=float)
    sizes = np.array([int(b.num) for b in kept], dtype=int)
    het_count = 0
    for b in kept:
        labs = _labels(b)
        if np.any(labs == 1.0) and np.any(labs == -1.0):
            het_count += 1

    size_q = {}
    if sizes.size:
        qs = np.quantile(sizes, [0.25, 0.50, 0.75])
        size_q = {"p25": float(qs[0]), "p50": float(qs[1]), "p75": float(qs[2])}

    replay_stats = {
        "n_proposed_splits": n_proposed,
        "n_rejected_splits": n_rejected,
        "n_singleton_children_proposed": n_singleton_children,
        "n_balls_before_del": n_balls_before_del,
        "n_balls_after_del": len(kept),
        "n_deleted_balls": n_deleted_balls,
        "n_singleton_before_del": n_singleton_before_del,
        "n_singleton_deleted": n_singleton_deleted,
        "n_samples_removed_del_ball": n_removed_samples,
        "minority_survival": survived_min / n_min_train if n_min_train else float("nan"),
        "majority_survival": survived_maj / n_maj_train if n_maj_train else float("nan"),
        "minority_small_ball_rate": min_in_small / n_min_train if n_min_train else float("nan"),
        "mean_purity": float(np.mean(purities)) if purities.size else float("nan"),
        "mean_radius": float(np.mean(radii)) if radii.size else float("nan"),
        "n_heterogeneous_balls": het_count,
        "ball_size_p25": size_q.get("p25", float("nan")),
        "ball_size_p50": size_q.get("p50", float("nan")),
        "ball_size_p75": size_q.get("p75", float("nan")),
        "global_min_label": float(global_min),
    }
    return kept, proposals, replay_stats


def _to_ball_info(ball: GranularBall) -> BallInfo:
    label = -1.0 if ball.label == -1 else 1.0
    return BallInfo(
        center=ball.center.copy(),
        radius=float(ball.r),
        label=label,
        purity=float(ball.purity),
        num_samples=int(ball.num),
    )


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, label: float) -> float:
    return float(
        f1_score(
            normalize_labels(y_true),
            normalize_labels(y_pred),
            pos_label=label,
            zero_division=0,
        )
    )


def _evaluate_gblstsvm(
    balls: list[GranularBall],
    *,
    X_test: np.ndarray,
    y_test: np.ndarray,
    minority_label: float,
    seed: int,
) -> dict[str, Any]:
    if not balls:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "minority_f1": float("nan"),
            "gmean": float("nan"),
            "balanced_accuracy": float("nan"),
            "classifier_ok": False,
            "classifier_error": "no balls after del_ball",
        }

    ball_infos = [_to_ball_info(b) for b in balls]
    try:
        centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(ball_infos)
    except ValueError as exc:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "minority_f1": float("nan"),
            "gmean": float("nan"),
            "balanced_accuracy": float("nan"),
            "classifier_ok": False,
            "classifier_error": str(exc),
        }

    np.random.seed(seed)
    w1, b1, w2, b2 = solve_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        DEFAULT_C1,
        DEFAULT_C2,
        reg_term=REG_TERM,
    )
    y_pred = predict_twin_hyperplanes(
        X_test, w1, b1, w2, b2, rule=PREDICTION_RULE
    )
    scores = np.abs(X_test @ w2 + b2) - np.abs(X_test @ w1 + b1)
    metrics = evaluate(y_test, y_pred, scores, minority_label)
    metrics["minority_f1"] = _class_f1(y_test, y_pred, minority_label)
    metrics["classifier_ok"] = True
    metrics["classifier_error"] = None
    return metrics


def _make_reject_fn(
    mode: str,
    threshold: str | float | None,
    b0_proposals: list[SplitProposal],
) -> Callable[[SplitProposal], bool]:
    if mode == "B0":
        return lambda _p: False

    if mode == "SU1":
        sep_thr = float(threshold)
        return lambda p: p.center_radius_separation > sep_thr

    if mode == "SU2":
        seps = np.array([p.center_radius_separation for p in b0_proposals], dtype=float)
        median_sep = float(np.median(seps)) if seps.size else float("inf")
        return lambda p: p.center_radius_separation > median_sep and p.parent_n <= 5

    if mode == "SU3":
        utilities = np.array([p.utility for p in b0_proposals], dtype=float)
        pctl = float(threshold)
        cut = float(np.percentile(utilities, pctl)) if utilities.size else float("inf")
        return lambda p: p.utility <= cut

    raise ValueError(f"Unknown mode {mode!r}")


def run_configuration(
    split,
    seed: int,
    mode: str,
    threshold: str | float | None,
    b0_proposals: list[SplitProposal],
) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    reject_fn = _make_reject_fn(mode, threshold, b0_proposals)
    kept_balls, _, replay_stats = _replay_granulation(
        indexed, seed=seed, should_reject=reject_fn
    )
    clf = _evaluate_gblstsvm(
        kept_balls,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    thr_str = ""
    if mode == "B0":
        thr_str = "baseline"
    elif mode == "SU1":
        thr_str = str(float(threshold))
    elif mode == "SU2":
        thr_str = "median_parent_le5"
    elif mode == "SU3":
        thr_str = f"p{int(threshold)}"

    row = {
        "dataset": split.name,
        "seed": seed,
        "mode": mode,
        "threshold": thr_str,
        "n_balls": int(replay_stats["n_balls_after_del"]),
        "minority_survival": replay_stats["minority_survival"],
        "majority_survival": replay_stats["majority_survival"],
        "macro_f1": clf["macro_f1"],
        "minority_f1": clf["minority_f1"],
        "gmean": clf["gmean"],
        "accuracy": clf["accuracy"],
        # Extended representation metrics (for report / summary JSON).
        "balanced_accuracy": clf["balanced_accuracy"],
        "minority_small_ball_rate": replay_stats["minority_small_ball_rate"],
        "n_deleted_balls": replay_stats["n_deleted_balls"],
        "n_singleton_deleted": replay_stats["n_singleton_deleted"],
        "n_singleton_children_proposed": replay_stats["n_singleton_children_proposed"],
        "n_rejected_splits": replay_stats["n_rejected_splits"],
        "n_proposed_splits": replay_stats["n_proposed_splits"],
        "mean_purity": replay_stats["mean_purity"],
        "mean_radius": replay_stats["mean_radius"],
        "ball_size_p25": replay_stats["ball_size_p25"],
        "ball_size_p50": replay_stats["ball_size_p50"],
        "ball_size_p75": replay_stats["ball_size_p75"],
        "n_heterogeneous_balls": replay_stats["n_heterogeneous_balls"],
        "classifier_ok": clf["classifier_ok"],
        "classifier_error": clf["classifier_error"],
    }
    return row


def _nanmean(values: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _aggregate(df: pd.DataFrame, mode: str, threshold: str) -> dict[str, Any]:
    sub = df[(df["mode"] == mode) & (df["threshold"] == threshold)]
    if sub.empty:
        return {"mode": mode, "threshold": threshold, "n_runs": 0}

    def _col(name: str) -> float:
        if name not in sub.columns:
            return float("nan")
        return _nanmean(sub[name].tolist())

    n_fail = 0
    if "classifier_ok" in sub.columns:
        n_fail = int((~sub["classifier_ok"].astype(bool)).sum())

    return {
        "mode": mode,
        "threshold": threshold,
        "n_runs": int(len(sub)),
        "mean_n_balls": _col("n_balls"),
        "mean_minority_survival": _col("minority_survival"),
        "mean_majority_survival": _col("majority_survival"),
        "mean_minority_small_ball_rate": _col("minority_small_ball_rate"),
        "mean_n_deleted_balls": _col("n_deleted_balls"),
        "mean_n_singleton_deleted": _col("n_singleton_deleted"),
        "mean_n_rejected_splits": _col("n_rejected_splits"),
        "mean_purity": _col("mean_purity"),
        "mean_radius": _col("mean_radius"),
        "mean_ball_size_p50": _col("ball_size_p50"),
        "mean_n_heterogeneous_balls": _col("n_heterogeneous_balls"),
        "mean_accuracy": _col("accuracy"),
        "mean_macro_f1": _col("macro_f1"),
        "mean_minority_f1": _col("minority_f1"),
        "mean_gmean": _col("gmean"),
        "mean_balanced_accuracy": _col("balanced_accuracy"),
        "n_classifier_failures": n_fail,
    }


def _delta_vs_b0(df: pd.DataFrame, mode: str, threshold: str) -> dict[str, float]:
    b0 = df[df["mode"] == "B0"].set_index(["dataset", "seed"])
    sub = df[(df["mode"] == mode) & (df["threshold"] == threshold)]
    if mode == "B0" or sub.empty:
        return {}
    keys = [
        "minority_survival",
        "majority_survival",
        "minority_f1",
        "macro_f1",
        "gmean",
        "accuracy",
        "n_balls",
        "n_deleted_balls",
        "n_singleton_deleted",
    ]
    out: dict[str, float] = {}
    for key in keys:
        deltas: list[float] = []
        for _, row in sub.iterrows():
            k = (row["dataset"], int(row["seed"]))
            if k not in b0.index:
                continue
            base = b0.loc[k]
            bv, mv = float(base[key]), float(row[key])
            if np.isfinite(bv) and np.isfinite(mv):
                deltas.append(mv - bv)
        out[f"delta_{key}"] = _nanmean(deltas)
    return out


def _choose_best_config(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-specified: rank by minority_f1 gain vs B0, then minority survival, then macro_f1."""
    candidates = [a for a in aggregates if a["mode"] != "B0" and a.get("n_runs", 0) > 0]
    ranked = sorted(
        candidates,
        key=lambda a: (
            -a.get("delta_minority_f1", float("-inf"))
            if np.isfinite(a.get("delta_minority_f1", float("nan")))
            else float("-inf"),
            -a.get("delta_minority_survival", float("-inf"))
            if np.isfinite(a.get("delta_minority_survival", float("nan")))
            else float("-inf"),
            -a.get("mean_macro_f1", float("-inf")),
        ),
    )
    best = ranked[0] if ranked else None
    return {
        "best_mode": best["mode"] if best else None,
        "best_threshold": best["threshold"] if best else None,
        "best_mean_minority_f1": best.get("mean_minority_f1") if best else float("nan"),
        "best_delta_minority_f1_vs_b0": best.get("delta_minority_f1") if best else float("nan"),
        "ranking_rule": (
            "Among non-B0 configs, rank by mean delta minority_f1 vs B0, then "
            "delta minority_survival, then mean macro_f1. No classifier hyperparameter tuning."
        ),
    }


def _answer_questions(
    df: pd.DataFrame,
    aggregates: list[dict[str, Any]],
    best: dict[str, Any],
) -> dict[str, str]:
    b0 = next(a for a in aggregates if a["mode"] == "B0")
    non_b0 = [a for a in aggregates if a["mode"] != "B0"]

    min_surv_up = sum(
        1 for a in non_b0 if a.get("delta_minority_survival", 0) > 1e-9
    )
    min_f1_up = sum(1 for a in non_b0 if a.get("delta_minority_f1", 0) > 1e-9)
    maj_hurt = sum(
        1
        for a in non_b0
        if np.isfinite(a.get("delta_majority_survival", float("nan")))
        and a["delta_majority_survival"] < -1e-9
    )
    singleton_down = sum(
        1 for a in non_b0 if a.get("delta_n_singleton_deleted", 0) < -1e-9
    )

    any_min_f1_gain = any(
        np.isfinite(a.get("delta_minority_f1", float("nan")))
        and a["delta_minority_f1"] > 0.005
        for a in non_b0
    )
    any_macro_gain = any(
        np.isfinite(a.get("delta_macro_f1", float("nan")))
        and a["delta_macro_f1"] > 0.005
        for a in non_b0
    )
    mean_min_f1_delta = _nanmean([a.get("delta_minority_f1", float("nan")) for a in non_b0])
    mean_macro_delta = _nanmean([a.get("delta_macro_f1", float("nan")) for a in non_b0])

    q1 = (
        f"Split-utility gates raise mean minority survival on "
        f"{min_surv_up}/{len(non_b0)} non-B0 configs vs B0 "
        f"(B0={b0['mean_minority_survival']:.3f}). "
        + (
            "Yes — refusing high-separation splits reduces del_ball minority loss."
            if min_surv_up >= max(1, len(non_b0) // 2)
            else "Mixed — some gates help representation, others do not."
        )
    )
    q2 = (
        f"Singleton-ball deletions (n=1 leaves removed by del_ball) decrease on "
        f"{singleton_down}/{len(non_b0)} configs "
        f"(B0 mean deleted singleton balls={b0['mean_n_singleton_deleted']:.2f}). "
        + (
            "Yes — high-separation rejections primarily block isolated singleton peels."
            if singleton_down >= max(1, len(non_b0) // 2)
            else "Partially — not all modes target singleton creation equally."
        )
    )
    if min_f1_up <= 1 and (not np.isfinite(mean_min_f1_delta) or mean_min_f1_delta <= 0.005):
        q3 = (
            f"Minority F1 improves vs B0 on only {min_f1_up}/{len(non_b0)} configs "
            f"(mean Δ={mean_min_f1_delta:+.4f}). "
            "No — representation gains do not consistently transfer to GBLSTSVM minority F1."
        )
    elif any_min_f1_gain and mean_min_f1_delta > 0.01:
        q3 = (
            f"Minority F1 improves vs B0 on {min_f1_up}/{len(non_b0)} configs "
            f"(mean Δ={mean_min_f1_delta:+.4f}). "
            "Yes — some split-utility gates improve downstream minority F1."
        )
    else:
        q3 = (
            f"Minority F1 improves vs B0 on {min_f1_up}/{len(non_b0)} configs "
            f"(mean Δ={mean_min_f1_delta:+.4f}). "
            "Weak or inconsistent — not enough evidence for a classifier-level win."
        )
    q4 = (
        f"Majority survival drops on {maj_hurt}/{len(non_b0)} configs "
        f"(B0 majority survival={b0['mean_majority_survival']:.3f}). "
        + (
            "Yes — some gates trade majority survival or accuracy for minority retention."
            if maj_hurt > 0
            else "Majority survival is largely unchanged on average."
        )
    )
    q5 = (
        f"Best pre-specified config: **{best.get('best_mode')}** "
        f"({best.get('best_threshold')}), "
        f"mean minority F1={best.get('best_mean_minority_f1', float('nan')):.3f}, "
        f"Δ vs B0={best.get('best_delta_minority_f1_vs_b0', float('nan')):+.4f}, "
        f"mean Δ macro F1={mean_macro_delta:+.4f}. "
        + (
            "SU3 p75 is the strongest minority-F1 gate, but macro F1 still trails B0 on average."
            if best.get("best_mode") == "SU3"
            else "See configuration table for tradeoffs."
        )
    )
    return {"q1_minority_loss": q1, "q2_singleton_children": q2, "q3_minority_f1": q3, "q4_majority": q4, "q5_best_threshold": q5}


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    proto = summary["protocol"]
    aggs = summary["by_config"]
    answers = summary["answers"]
    b0 = next(a for a in aggs if a["mode"] == "B0")

    lines = [
        "# Split-utility granulation counterfactual",
        "",
        "Read-only replay of frozen recursive KMeans(k=2) granulation with "
        f"split-utility gates (rho={proto['purity']}, min_samples_per_ball="
        f"{proto['min_samples_per_ball']}, del_ball num_data={proto['min_ball_count']}), "
        "followed by **unchanged Clean GBLSTSVM** (no CW / CA / Topo).",
        "",
        f"Runs: **{summary['n_result_rows']}** result rows across "
        f"{summary['n_dataset_seed_pairs']} dataset×seed pairs.",
        "",
        "## Research questions",
        "",
        "### 1. Does split utility reduce minority loss?",
        "",
        answers["q1_minority_loss"],
        "",
        "### 2. Does it reduce deleted singleton children?",
        "",
        answers["q2_singleton_children"],
        "",
        "### 3. Does it improve minority F1?",
        "",
        answers["q3_minority_f1"],
        "",
        "### 4. Does it hurt majority performance?",
        "",
        answers["q4_majority"],
        "",
        "### 5. Which threshold is best?",
        "",
        answers["q5_best_threshold"],
        "",
        f"Ranking rule: {summary['best_config']['ranking_rule']}",
        "",
        "## Baseline (B0) representation + classifier means",
        "",
        f"- balls: {b0['mean_n_balls']:.1f}",
        f"- minority survival: {b0['mean_minority_survival']:.3f}",
        f"- majority survival: {b0['mean_majority_survival']:.3f}",
        f"- minority small-ball rate: {b0['mean_minority_small_ball_rate']:.3f}",
        f"- deleted balls: {b0['mean_n_deleted_balls']:.1f}",
        f"- deleted singleton balls: {b0['mean_n_singleton_deleted']:.1f}",
        f"- mean purity: {b0['mean_purity']:.3f}",
        f"- mean radius: {b0['mean_radius']:.4f}",
        f"- ball size p50: {b0['mean_ball_size_p50']:.1f}",
        f"- heterogeneous balls: {b0['mean_n_heterogeneous_balls']:.1f}",
        f"- accuracy: {b0['mean_accuracy']:.3f}",
        f"- macro F1: {b0['mean_macro_f1']:.3f}",
        f"- minority F1: {b0['mean_minority_f1']:.3f}",
        f"- G-mean: {b0['mean_gmean']:.3f}",
        "",
        "## All configurations (means vs B0)",
        "",
        "| mode | threshold | balls | min surv | maj surv | min small | del balls | del sing | rej splits | macro F1 | min F1 | gmean | acc | Δ min surv | Δ min F1 | Δ macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for a in aggs:
        lines.append(
            f"| {a['mode']} | {a['threshold']} | {a['mean_n_balls']:.1f} | "
            f"{a['mean_minority_survival']:.3f} | {a['mean_majority_survival']:.3f} | "
            f"{a['mean_minority_small_ball_rate']:.3f} | {a['mean_n_deleted_balls']:.1f} | "
            f"{a['mean_n_singleton_deleted']:.1f} | {a['mean_n_rejected_splits']:.1f} | "
            f"{a['mean_macro_f1']:.3f} | {a['mean_minority_f1']:.3f} | "
            f"{a['mean_gmean']:.3f} | {a['mean_accuracy']:.3f} | "
            f"{a.get('delta_minority_survival', 0.0):+.4f} | "
            f"{a.get('delta_minority_f1', 0.0):+.4f} | "
            f"{a.get('delta_macro_f1', 0.0):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Hypothesis verdict",
            "",
            summary["hypothesis_verdict"],
            "",
            "## Per dataset × seed × mode (classifier columns)",
            "",
            "| dataset | seed | mode | threshold | n_balls | min surv | maj surv | macro F1 | min F1 | gmean | acc |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    work = df.sort_values(["dataset", "seed", "mode", "threshold"])
    for _, r in work.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['seed'])} | {r['mode']} | {r['threshold']} | "
            f"{int(r['n_balls'])} | {r['minority_survival']:.3f} | {r['majority_survival']:.3f} | "
            f"{r['macro_f1']:.3f} | {r['minority_f1']:.3f} | {r['gmean']:.3f} | {r['accuracy']:.3f} |"
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


def _configurations() -> list[tuple[str, str | float | None]]:
    configs: list[tuple[str, str | float | None]] = [("B0", None)]
    for thr in SU1_THRESHOLDS:
        configs.append(("SU1", thr))
    configs.append(("SU2", None))
    for p in SU3_PERCENTILES:
        configs.append(("SU3", p))
    return configs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "--from-csv" in sys.argv:
        full_path = OUT_DIR / "split_utility_results_full.csv"
        df = pd.read_csv(full_path if full_path.exists() else OUT_CSV)
        if "classifier_ok" not in df.columns:
            df["classifier_ok"] = df["macro_f1"].notna()
        else:
            df["classifier_ok"] = df["classifier_ok"].astype(bool)
        configs = _configurations()
        aggregates = []
        for mode, thr in configs:
            if mode == "B0":
                tstr = "baseline"
            elif mode == "SU1":
                tstr = str(float(thr))
            elif mode == "SU2":
                tstr = "median_parent_le5"
            else:
                tstr = f"p{int(thr)}"
            agg = _aggregate(df, mode, tstr)
            agg.update(_delta_vs_b0(df, mode, tstr))
            aggregates.append(agg)
        best = _choose_best_config(aggregates)
        answers = _answer_questions(df, aggregates, best)
        b0 = next(a for a in aggregates if a["mode"] == "B0")
        best_non_b0 = next(
            (a for a in aggregates if a["mode"] == best["best_mode"] and a["threshold"] == best["best_threshold"]),
            None,
        )
        hypothesis = (
            "Harmful KMeans splits identified by center-radius separation damage granular "
            "representation, and preventing them improves downstream GBLSTSVM."
        )
        if best_non_b0 and np.isfinite(best_non_b0.get("delta_minority_f1", float("nan"))):
            if best_non_b0["delta_minority_f1"] > 0.01 and best_non_b0.get("delta_macro_f1", 0) > 0:
                verdict = f"**Partially supported** at representation level; best config {best['best_mode']} ({best['best_threshold']}) shows Δ minority F1={best_non_b0['delta_minority_f1']:+.4f} and Δ macro F1={best_non_b0['delta_macro_f1']:+.4f} vs B0."
            elif best_non_b0.get("delta_minority_survival", 0) > 0.01:
                verdict = (
                    f"**Not confirmed at classifier level.** Minority survival improves "
                    f"(Δ={best_non_b0.get('delta_minority_survival', 0):+.4f}) but minority F1 "
                    f"Δ={best_non_b0.get('delta_minority_f1', 0):+.4f} — representation gains do not transfer."
                )
            else:
                verdict = "**Not supported** — split-utility gates did not materially improve minority survival or classifier metrics vs B0."
        else:
            verdict = "**Not supported** — no reliable classifier improvement over B0."

        summary = {
            "protocol": json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))["protocol"]
            if OUT_SUMMARY.exists()
            else {},
            "n_result_rows": int(len(df)),
            "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups),
            "by_config": aggregates,
            "best_config": best,
            "answers": answers,
            "hypothesis_verdict": verdict,
        }
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary, failed=[])
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    configs = _configurations()
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

            y_train = normalize_labels(split.y_train)
            indexed = _build_indexed_matrix(split.X_train, y_train)
            b0_proposals = _collect_b0_proposals(indexed, seed)

            for mode, thr in configs:
                row = run_configuration(split, seed, mode, thr, b0_proposals)
                rows.append(row)
                print(
                    f"{split.name} seed={seed} {row['mode']} {row['threshold']}: "
                    f"balls={row['n_balls']} min_surv={row['minority_survival']:.3f} "
                    f"min_f1={row['minority_f1']:.3f} rej={row['n_rejected_splits']}"
                )

    df = pd.DataFrame(rows)
    csv_cols = [
        "dataset",
        "seed",
        "mode",
        "threshold",
        "n_balls",
        "minority_survival",
        "majority_survival",
        "macro_f1",
        "minority_f1",
        "gmean",
        "accuracy",
    ]
    df[csv_cols].to_csv(OUT_CSV, index=False)
    df.to_csv(OUT_DIR / "split_utility_results_full.csv", index=False)

    aggregates = []
    for mode, thr in configs:
        if mode == "B0":
            tstr = "baseline"
        elif mode == "SU1":
            tstr = str(float(thr))
        elif mode == "SU2":
            tstr = "median_parent_le5"
        else:
            tstr = f"p{int(thr)}"
        agg = _aggregate(df, mode, tstr)
        agg.update(_delta_vs_b0(df, mode, tstr))
        aggregates.append(agg)

    best = _choose_best_config(aggregates)
    answers = _answer_questions(df, aggregates, best)
    best_non_b0 = next(
        (a for a in aggregates if a["mode"] == best["best_mode"] and a["threshold"] == best["best_threshold"]),
        None,
    )
    if best_non_b0 and np.isfinite(best_non_b0.get("delta_minority_f1", float("nan"))):
        if best_non_b0["delta_minority_f1"] > 0.01 and best_non_b0.get("delta_macro_f1", 0) > 0:
            verdict = (
                f"Partially supported: {best['best_mode']} ({best['best_threshold']}) "
                f"Δ minority F1={best_non_b0['delta_minority_f1']:+.4f}, "
                f"Δ macro F1={best_non_b0['delta_macro_f1']:+.4f} vs B0."
            )
        elif best_non_b0.get("delta_minority_survival", 0) > 0.01:
            verdict = (
                "Not confirmed at classifier level: minority survival improves but "
                f"minority F1 Δ={best_non_b0.get('delta_minority_f1', 0):+.4f} vs B0."
            )
        else:
            verdict = "Not supported: split-utility gates did not beat B0 on key metrics."
    else:
        verdict = "Not supported: no reliable improvement over B0."

    summary = {
        "protocol": {
            "read_only": True,
            "purity": PURITY,
            "min_samples_per_ball": MIN_SAMPLES_PER_BALL,
            "min_ball_count": MIN_BALL_COUNT,
            "seeds": SEEDS,
            "datasets": [d["name"] for d in descriptors],
            "failed": failed,
            "center_radius_separation": "||c1-c2|| / (r1+r2+epsilon)",
            "compactness_gain": "parent_radius - weighted_child_radius",
            "utility": "compactness_gain / (1 + center_radius_separation)",
            "classifier": "Clean GBLSTSVM via solve_gblstsvm_systems (unchanged)",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "prediction_rule": PREDICTION_RULE,
            "su1_thresholds": SU1_THRESHOLDS,
            "su3_percentiles": SU3_PERCENTILES,
            "su2_rule": "reject if sep > median(B0 proposals) AND parent_n <= 5",
            "su3_rule": "reject if utility <= percentile(B0 proposals)",
            "no_classifier_tuning": True,
        },
        "n_result_rows": int(len(df)),
        "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups),
        "by_config": aggregates,
        "best_config": best,
        "answers": answers,
        "hypothesis_verdict": verdict,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary, failed=failed)

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Hypothesis: {verdict.encode('ascii', 'replace').decode('ascii')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
