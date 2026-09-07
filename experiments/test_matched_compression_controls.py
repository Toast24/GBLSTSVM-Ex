"""
Matched compression controls diagnostic (read-only).

Tests whether online entropy's F1 advantage comes from selecting structurally
useful balls versus mere ball-count regularization. Subsamples frozen rho90
balls to the same count as online entropy using several ranking rules.

Does NOT modify src/, solver, prediction, weighting, or baseline granulation.

Methods:
  random              — random rho90 subset (deterministic seed)
  purity              — highest-purity rho90 balls
  entropy             — highest-entropy rho90 balls
  boundary            — highest boundary-score rho90 balls
  stratified_random   — class-stratified random rho90 subset
  full_rho90          — all rho90 balls
  online_entropy      — actual online entropy v2 representation

Run from GBLSTSVM root:
  python experiments/test_matched_compression_controls.py --smoke
  python experiments/test_matched_compression_controls.py
  python experiments/test_matched_compression_controls.py --from-csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_ball_utility_landscape import compute_ball_features  # noqa: E402
from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    build_representation,
    fit_gaussian_gblstsvm,
)
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels  # noqa: E402
from experiments.test_adaptive_entropy_granulation_v2 import (  # noqa: E402
    build_adaptive_entropy_v2,
    compute_extended_stats,
    measure_rho90_split_count,
    _global_min_label,
)
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _to_ball_info,
)
from src.granular_balls import BallInfo, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "matched_compression_controls"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = list(range(10))
MATCHED_METHODS = ("random", "purity", "entropy", "boundary", "stratified_random")
BASELINE_METHODS = ("full_rho90", "online_entropy")
ALL_METHODS = MATCHED_METHODS + BASELINE_METHODS
CONTROL_METHODS = MATCHED_METHODS + ("full_rho90",)
WIN_THRESHOLD = 8
COLLAPSE_BALL_THRESHOLD = 3
SAVE_EVERY = 35
DELTA_EPS = 1e-6
REPRESENTATION = "rho0.90"


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _ranking_scores(
    features: list[dict[str, Any]],
    criterion: str,
    *,
    rng_seed: int,
) -> np.ndarray:
    n = len(features)
    if criterion == "purity":
        return np.array([f["purity"] for f in features], dtype=float)
    if criterion == "boundary":
        return np.array([f["boundary_score"] for f in features], dtype=float)
    if criterion == "entropy":
        return np.array([f["entropy"] for f in features], dtype=float)
    if criterion == "random":
        rng = np.random.RandomState(rng_seed)
        return rng.random(n)
    raise ValueError(f"Unknown criterion {criterion!r}")


def _ensure_class_balance(
    balls: list[BallInfo],
    selected: list[int],
    k: int,
    minority_label: float,
    *,
    scores: np.ndarray | None,
    rng_seed: int,
) -> list[int]:
    if k < 2:
        return selected[:k]
    majority_label = -minority_label
    labels = {balls[i].label for i in selected}
    needed: list[float] = []
    if minority_label not in labels:
        needed.append(minority_label)
    if majority_label not in labels:
        needed.append(majority_label)
    if not needed:
        return selected[:k]

    out = list(selected[:k])
    for req in needed:
        pool = [i for i in range(len(balls)) if balls[i].label == req and i not in out]
        if not pool:
            continue
        if scores is not None:
            best = max(pool, key=lambda i: scores[i])
        else:
            rng = np.random.RandomState(rng_seed + int(req))
            best = int(rng.choice(pool))
        if len(out) >= k:
            droppable = [
                i
                for i in out
                if sum(1 for j in out if balls[j].label == balls[i].label) > 1
            ]
            if droppable:
                if scores is not None:
                    worst = min(droppable, key=lambda i: scores[i])
                else:
                    rng = np.random.RandomState(rng_seed + 99)
                    worst = int(rng.choice(droppable))
                out.remove(worst)
        if best not in out:
            out.append(best)
    return sorted(set(out))[:k]


def _select_top_k(
    balls: list[BallInfo],
    features: list[dict[str, Any]],
    k: int,
    criterion: str,
    minority_label: float,
    rng_seed: int,
) -> list[int]:
    n = len(balls)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    k = max(1, k)

    scores = _ranking_scores(features, criterion, rng_seed=rng_seed)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    selected = _ensure_class_balance(balls, order, k, minority_label, scores=scores, rng_seed=rng_seed)
    return selected


def _select_stratified_random(
    balls: list[BallInfo],
    k: int,
    minority_label: float,
    rng_seed: int,
) -> list[int]:
    n = len(balls)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    k = max(1, k)

    rng = np.random.RandomState(rng_seed)
    majority_label = -minority_label
    min_idx = [i for i, b in enumerate(balls) if b.label == minority_label]
    maj_idx = [i for i, b in enumerate(balls) if b.label == majority_label]

    k_min = int(round(k * len(min_idx) / n)) if n else 0
    k_maj = k - k_min
    if k >= 2:
        if min_idx and k_min == 0:
            k_min = 1
            k_maj = k - 1
        if maj_idx and k_maj == 0:
            k_maj = 1
            k_min = k - 1
    k_min = min(k_min, len(min_idx))
    k_maj = min(k_maj, len(maj_idx))

    while k_min + k_maj < k and (k_min < len(min_idx) or k_maj < len(maj_idx)):
        if k_maj < len(maj_idx) and k_maj <= k_min:
            k_maj += 1
        elif k_min < len(min_idx):
            k_min += 1
        else:
            break

    selected: list[int] = []
    if min_idx and k_min > 0:
        selected.extend(rng.choice(min_idx, size=k_min, replace=False).tolist())
    if maj_idx and k_maj > 0:
        selected.extend(rng.choice(maj_idx, size=k_maj, replace=False).tolist())
    return sorted(set(selected))


def _representation_stats(
    kept: list[GranularBall],
    indexed: np.ndarray,
    global_min: float,
    minority_label: float,
) -> dict[str, Any]:
    depths = [0] * len(kept)
    rep = compute_extended_stats(kept, depths, indexed, global_min)
    feats = compute_ball_features(kept, minority_label)
    n_samples = int(sum(f["sample_count"] for f in feats)) if feats else 0
    return {
        "n_balls": len(kept),
        "n_samples_represented": n_samples,
        "minority_survival": rep["minority_survival"],
        "majority_survival": rep["majority_survival"],
        "mean_purity": rep["mean_purity"],
        "mean_entropy": rep["mean_entropy"],
        "mean_radius": rep["mean_radius"],
        "mean_boundary_score": _nanmean([f["boundary_score"] for f in feats]),
        "mean_density": _nanmean([f["density"] for f in feats]),
        "fraction_mixed_balls": rep["fraction_mixed_balls"],
    }


def _evaluate_method(
    balls: list[BallInfo],
    kept: list[GranularBall],
    *,
    split,
    seed: int,
    method: str,
    target_n_balls: int,
    indexed: np.ndarray,
    global_min: float,
) -> dict[str, Any]:
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )
    stats = _representation_stats(kept, indexed, global_min, split.minority_label)
    n_balls = stats["n_balls"]
    macro_f1 = float(perf["macro_f1"])
    collapsed = (
        n_balls <= COLLAPSE_BALL_THRESHOLD
        or not bool(perf.get("classifier_ok", False))
        or not np.isfinite(macro_f1)
    )
    return {
        "method": method,
        "target_n_balls": target_n_balls,
        "n_balls": n_balls,
        "n_samples_represented": stats["n_samples_represented"],
        "minority_survival": stats["minority_survival"],
        "majority_survival": stats["majority_survival"],
        "mean_purity": stats["mean_purity"],
        "mean_entropy": stats["mean_entropy"],
        "mean_radius": stats["mean_radius"],
        "mean_boundary_score": stats["mean_boundary_score"],
        "mean_density": stats["mean_density"],
        "fraction_mixed_balls": stats["fraction_mixed_balls"],
        "accuracy": perf["accuracy"],
        "macro_f1": macro_f1,
        "minority_f1": perf["minority_f1"],
        "gmean": perf["gmean"],
        "f1_per_ball": macro_f1 / n_balls if n_balls > 0 and np.isfinite(macro_f1) else float("nan"),
        "collapsed": int(collapsed),
        "classifier_ok": bool(perf.get("classifier_ok", False)),
    }


def run_single(split, seed: int) -> list[dict[str, Any]]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    rho90_count = measure_rho90_split_count(indexed, seed=seed)

    balls_rho90, kept_rho90, _ = build_representation(split, seed, REPRESENTATION)
    features = compute_ball_features(kept_rho90, split.minority_label)

    entropy_result = build_adaptive_entropy_v2(
        indexed,
        seed=seed,
        split_budget=rho90_count,
        global_min=global_min,
        rho90_split_count=rho90_count,
    )
    balls_entropy = [_to_ball_info(b) for b in entropy_result.kept]
    target_n = len(balls_entropy)

    rows: list[dict[str, Any]] = []
    by_method: dict[str, dict[str, Any]] = {}

    for method in MATCHED_METHODS:
        if method == "stratified_random":
            indices = _select_stratified_random(
                balls_rho90, target_n, split.minority_label, rng_seed=seed
            )
        else:
            indices = _select_top_k(
                balls_rho90,
                features,
                target_n,
                method,
                split.minority_label,
                rng_seed=seed,
            )
        subset_balls = [balls_rho90[i] for i in indices]
        subset_kept = [kept_rho90[i] for i in indices]
        row = _evaluate_method(
            subset_balls,
            subset_kept,
            split=split,
            seed=seed,
            method=method,
            target_n_balls=target_n,
            indexed=indexed,
            global_min=global_min,
        )
        rows.append({"dataset": split.name, "seed": seed, **row})
        by_method[method] = row

    full_row = _evaluate_method(
        balls_rho90,
        kept_rho90,
        split=split,
        seed=seed,
        method="full_rho90",
        target_n_balls=target_n,
        indexed=indexed,
        global_min=global_min,
    )
    rows.append({"dataset": split.name, "seed": seed, **full_row})
    by_method["full_rho90"] = full_row

    online_row = _evaluate_method(
        balls_entropy,
        entropy_result.kept,
        split=split,
        seed=seed,
        method="online_entropy",
        target_n_balls=target_n,
        indexed=indexed,
        global_min=global_min,
    )
    online_f1 = float(online_row["macro_f1"])
    online_row.update(
        {
            "minus_random": online_f1 - float(by_method["random"]["macro_f1"])
            if np.isfinite(online_f1) and np.isfinite(by_method["random"]["macro_f1"])
            else float("nan"),
            "minus_purity": online_f1 - float(by_method["purity"]["macro_f1"])
            if np.isfinite(online_f1) and np.isfinite(by_method["purity"]["macro_f1"])
            else float("nan"),
            "minus_entropy_selection": online_f1 - float(by_method["entropy"]["macro_f1"])
            if np.isfinite(online_f1) and np.isfinite(by_method["entropy"]["macro_f1"])
            else float("nan"),
            "minus_boundary_selection": online_f1 - float(by_method["boundary"]["macro_f1"])
            if np.isfinite(online_f1) and np.isfinite(by_method["boundary"]["macro_f1"])
            else float("nan"),
            "minus_stratified_random": online_f1
            - float(by_method["stratified_random"]["macro_f1"])
            if np.isfinite(online_f1)
            and np.isfinite(by_method["stratified_random"]["macro_f1"])
            else float("nan"),
            "minus_full_rho90": online_f1 - float(by_method["full_rho90"]["macro_f1"])
            if np.isfinite(online_f1) and np.isfinite(by_method["full_rho90"]["macro_f1"])
            else float("nan"),
        }
    )
    rows.append({"dataset": split.name, "seed": seed, **online_row})
    return rows


def _paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    online = df[df["method"] == "online_entropy"].copy()
    if online.empty:
        return pd.DataFrame()
    delta_cols = [
        "minus_random",
        "minus_purity",
        "minus_entropy_selection",
        "minus_boundary_selection",
        "minus_stratified_random",
        "minus_full_rho90",
    ]
    return online[["dataset", "seed", "macro_f1", "n_balls", "collapsed"] + delta_cols]


def aggregate_by_method(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "method": method,
            "n_runs": int(len(sub)),
            "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
            "mean_target_n_balls": _nanmean(sub["target_n_balls"].tolist()),
            "frac_collapsed": float(sub["collapsed"].mean()),
            "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
            "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
            "mean_gmean": _nanmean(sub["gmean"].tolist()),
            "mean_f1_per_ball": _nanmean(sub["f1_per_ball"].tolist()),
            "mean_purity": _nanmean(sub["mean_purity"].tolist()),
            "mean_entropy": _nanmean(sub["mean_entropy"].tolist()),
            "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
            "mean_fraction_mixed": _nanmean(sub["fraction_mixed_balls"].tolist()),
        }
        if method == "online_entropy":
            for col in (
                "minus_random",
                "minus_purity",
                "minus_entropy_selection",
                "minus_boundary_selection",
                "minus_stratified_random",
                "minus_full_rho90",
            ):
                row[f"mean_{col}"] = _nanmean(sub[col].tolist())
                row[f"frac_positive_{col}"] = float(np.mean(sub[col].to_numpy() > DELTA_EPS))
        rows.append(row)
    return rows


def _f1_vs_n_balls(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ALL_METHODS:
        sub = df[df["method"] == method]
        if len(sub) < 3:
            continue
        x = sub["n_balls"].to_numpy(dtype=float)
        y = sub["macro_f1"].to_numpy(dtype=float)
        pr, _ = pearsonr(x, y)
        sr, _ = spearmanr(x, y)
        out[method] = {"pearson": float(pr), "spearman": float(sr), "n": int(len(sub))}
    return out


def _classify_datasets(paired: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, grp in paired.groupby("dataset"):
        d_random = grp["minus_random"].to_numpy(dtype=float)
        d_full = grp["minus_full_rho90"].to_numpy(dtype=float)
        n = len(grp)
        n_win_random = int(np.sum(d_random > DELTA_EPS))
        n_loss_random = int(np.sum(d_random < -DELTA_EPS))
        n_win_full = int(np.sum(d_full > DELTA_EPS))
        n_loss_full = int(np.sum(d_full < -DELTA_EPS))

        if n_win_random >= WIN_THRESHOLD and n_loss_random <= 1:
            vs_random = "consistent_win"
        elif n_loss_random >= WIN_THRESHOLD and n_win_random <= 1:
            vs_random = "consistent_loss"
        else:
            vs_random = "unstable"

        if n_win_full >= WIN_THRESHOLD and n_loss_full <= 1:
            vs_full = "consistent_win"
        elif n_loss_full >= WIN_THRESHOLD and n_win_full <= 1:
            vs_full = "consistent_loss"
        else:
            vs_full = "unstable"

        rows.append(
            {
                "dataset": dataset,
                "n_seeds": n,
                "mean_minus_random": _nanmean(d_random),
                "mean_minus_full_rho90": _nanmean(d_full),
                "n_win_vs_random": n_win_random,
                "n_loss_vs_random": n_loss_random,
                "vs_random_class": vs_random,
                "vs_full_rho90_class": vs_full,
            }
        )
    return rows


def _control_comparison(paired: pd.DataFrame, control: str) -> dict[str, Any]:
    col = {
        "random": "minus_random",
        "purity": "minus_purity",
        "entropy": "minus_entropy_selection",
        "boundary": "minus_boundary_selection",
        "stratified_random": "minus_stratified_random",
        "full_rho90": "minus_full_rho90",
    }[control]
    deltas = paired[col].to_numpy(dtype=float)
    finite = deltas[np.isfinite(deltas)]
    return {
        "control": control,
        "mean_delta": _nanmean(finite),
        "frac_online_wins": float(np.mean(finite > DELTA_EPS)) if finite.size else float("nan"),
        "frac_online_losses": float(np.mean(finite < -DELTA_EPS)) if finite.size else float("nan"),
        "n_pairs": int(finite.size),
    }


def answer_questions(
    by_method: list[dict[str, Any]],
    paired: pd.DataFrame,
    comparisons: list[dict[str, Any]],
    by_dataset: list[dict[str, Any]],
) -> dict[str, str]:
    by_name = {a["method"]: a for a in by_method}
    online = by_name.get("online_entropy", {})
    rho = by_name.get("full_rho90", {})

    def _cmp(control: str) -> dict[str, Any]:
        return next((c for c in comparisons if c["control"] == control), {})

    c_random = _cmp("random")
    c_purity = _cmp("purity")
    c_entropy = _cmp("entropy")
    c_boundary = _cmp("boundary")
    c_strat = _cmp("stratified_random")

    all_similar = all(
        abs(c.get("mean_delta", 0)) < 0.01
        for c in (c_random, c_purity, c_entropy, c_boundary, c_strat)
    )
    entropy_wins_random = c_random.get("frac_online_wins", 0) > 0.55 and c_random.get(
        "mean_delta", 0
    ) > 0.01
    entropy_wins_all = all(
        _cmp(c).get("mean_delta", -1) > 0.005
        for c in ("random", "purity", "entropy", "boundary", "stratified_random")
    )

    n_consistent = sum(1 for d in by_dataset if d["vs_random_class"] == "consistent_win")
    n_unstable = sum(1 for d in by_dataset if d["vs_random_class"] == "unstable")

    q1 = (
        f"Paired online − random: mean ΔF1 **{c_random.get('mean_delta', float('nan')):+.3f}**, "
        f"online wins **{c_random.get('frac_online_wins', float('nan')):.1%}** of pairs. "
    )
    if entropy_wins_random:
        q1 += "**Yes** — online entropy beats size-matched random rho90 compression."
    elif all_similar:
        q1 += "**No** — matched random performs similarly; advantage is not ball-specific selection."
    else:
        q1 += "**Mixed** — online entropy does not consistently dominate random at matched size."

    q2 = (
        f"vs purity: ΔF1 **{c_purity.get('mean_delta', float('nan')):+.3f}**, "
        f"wins **{c_purity.get('frac_online_wins', float('nan')):.1%}**. "
    )
    q2 += (
        "**Yes** — entropy granulation beats purity-based subsampling."
        if c_purity.get("mean_delta", 0) > 0.01 and c_purity.get("frac_online_wins", 0) > 0.5
        else "**No** — purity subsampling is competitive with online entropy."
    )

    q3 = (
        f"vs boundary: ΔF1 **{c_boundary.get('mean_delta', float('nan')):+.3f}**, "
        f"wins **{c_boundary.get('frac_online_wins', float('nan')):.1%}**. "
    )
    q3 += (
        "**Yes** — online entropy beats boundary-ranked rho90 subsets."
        if c_boundary.get("mean_delta", 0) > 0.01
        else "**No** — boundary selection matches online entropy at equal ball counts."
    )

    q4 = (
        f"vs stratified random: ΔF1 **{c_strat.get('mean_delta', float('nan')):+.3f}**, "
        f"wins **{c_strat.get('frac_online_wins', float('nan')):.1%}**. "
    )
    q4 += (
        "**Yes** — online entropy beats class-stratified random subsampling."
        if c_strat.get("mean_delta", 0) > 0.01
        else "**No** — stratified random is as strong as online entropy."
    )

    q5 = ""
    if all_similar and not entropy_wins_random:
        q5 = (
            "**Conclusion: compression/regularization.** Matched-size rho90 subsets "
            f"(random F1 {by_name.get('random', {}).get('mean_macro_f1', float('nan')):.3f}, "
            f"online {online.get('mean_macro_f1', float('nan')):.3f}) perform similarly. "
            "The gain is primarily from using fewer balls, not which balls are kept."
        )
    else:
        q5 = (
            "**Not solely compression** — controls do not fully match online entropy; "
            "some selection signal remains beyond ball-count reduction."
        )

    q6 = ""
    if entropy_wins_all:
        q6 = (
            "**Conclusion: structurally useful regions.** Online entropy consistently "
            "outperforms all rho90 subsampling rules at matched ball counts — "
            "the online process selects different geometry than any static ranking."
        )
    elif entropy_wins_random:
        q6 = (
            "**Partial structural selection** — online entropy beats random/stratified "
            "controls but not all static rankings; advantage is mixed."
        )
    else:
        q6 = (
            "**No consistent structural advantage** — online entropy does not reliably "
            "beat matched rho90 subsets; compactness drives the effect."
        )

    q7 = (
        f"**{n_consistent}** datasets show consistent online advantage vs random "
        f"(≥{WIN_THRESHOLD}/10 seeds); **{n_unstable}** are unstable. "
    )
    top = sorted(by_dataset, key=lambda d: d["mean_minus_random"], reverse=True)[:3]
    if top:
        q7 += "Largest mean ΔF1 vs random: " + ", ".join(
            f"{d['dataset']} ({d['mean_minus_random']:+.3f})" for d in top
        ) + "."

    return {
        "q1_vs_random": q1,
        "q2_vs_purity": q2,
        "q3_vs_boundary": q3,
        "q4_vs_stratified_random": q4,
        "q5_compression_conclusion": q5,
        "q6_structural_selection_conclusion": q6,
        "q7_dataset_consistency": q7,
        "paired_vs_full_rho90": (
            f"Online − full rho90: mean ΔF1 "
            f"{_cmp('full_rho90').get('mean_delta', float('nan')):+.3f} "
            f"(online {online.get('mean_macro_f1', float('nan')):.3f} vs "
            f"rho {rho.get('mean_macro_f1', float('nan')):.3f}); "
            f"collapse online {online.get('frac_collapsed', float('nan')):.1%} vs "
            f"rho {rho.get('frac_collapsed', float('nan')):.1%}."
        ),
    }


def write_report(path: Path, *, summary: dict[str, Any]) -> None:
    aggs = summary["by_method"]
    answers = summary["answers"]
    comparisons = summary["control_comparisons"]

    lines = [
        "# Matched compression controls",
        "",
        "Read-only diagnostic: subsample frozen rho90 to the online-entropy ball count "
        "and compare classifier outcomes under identical Gaussian evaluation.",
        "",
        f"Runs: **{summary['n_rows']}** rows "
        f"({summary['n_dataset_seed_pairs']} dataset×seed × {len(ALL_METHODS)} methods)",
        "",
        "## Method summary",
        "",
        "| method | macro F1 | minority F1 | G-mean | balls | F1/ball | collapse% | "
        "purity | entropy | boundary | mixed% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggs:
        lines.append(
            f"| {a['method']} | {a['mean_macro_f1']:.3f} | "
            f"{a['mean_minority_f1']:.3f} | {a['mean_gmean']:.3f} | "
            f"{a['mean_n_balls']:.1f} | {a['mean_f1_per_ball']:.3f} | "
            f"{a['frac_collapsed']:.1%} | {a['mean_purity']:.3f} | "
            f"{a['mean_entropy']:.3f} | {a['mean_boundary_score']:.2f} | "
            f"{a['mean_fraction_mixed']:.2f} |"
        )

    lines.extend(["", "## Paired online − control (macro F1)", ""])
    for c in comparisons:
        lines.append(
            f"- **{c['control']}**: mean Δ {c['mean_delta']:+.3f}, "
            f"online wins {c['frac_online_wins']:.1%}, "
            f"losses {c['frac_online_losses']:.1%}"
        )

    lines.extend(
        [
            "",
            answers["paired_vs_full_rho90"],
            "",
            "## 1. Does entropy beat random matched compression?",
            "",
            answers["q1_vs_random"],
            "",
            "## 2. Does entropy beat purity-based compression?",
            "",
            answers["q2_vs_purity"],
            "",
            "## 3. Does entropy beat boundary-based compression?",
            "",
            answers["q3_vs_boundary"],
            "",
            "## 4. Does entropy beat random stratified compression?",
            "",
            answers["q4_vs_stratified_random"],
            "",
            "## 5. Compression vs selection",
            "",
            answers["q5_compression_conclusion"],
            "",
            "## 6. Structural region selection",
            "",
            answers["q6_structural_selection_conclusion"],
            "",
            "## 7. Dataset consistency",
            "",
            answers["q7_dataset_consistency"],
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
    paired = _paired_deltas(df)
    by_method = aggregate_by_method(df)
    comparisons = [_control_comparison(paired, c) for c in CONTROL_METHODS]
    by_dataset = _classify_datasets(paired) if not paired.empty else []
    answers = answer_questions(by_method, paired, comparisons, by_dataset)
    return {
        "protocol": {
            "read_only": True,
            "methods": list(ALL_METHODS),
            "matched_controls": list(MATCHED_METHODS),
            "target_n_balls": "online entropy v2 ball count per dataset×seed",
            "rho90_source": REPRESENTATION,
            "online_entropy": "adaptive entropy v2, budget=rho90 split count",
            "seeds": SEEDS,
            "collapse_ball_threshold": COLLAPSE_BALL_THRESHOLD,
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
        "by_method": by_method,
        "control_comparisons": comparisons,
        "f1_vs_n_balls_correlation": _f1_vs_n_balls(df),
        "by_dataset": by_dataset,
        "dataset_classification_counts": {
            "consistent_win_vs_random": sum(
                1 for d in by_dataset if d["vs_random_class"] == "consistent_win"
            ),
            "consistent_loss_vs_random": sum(
                1 for d in by_dataset if d["vs_random_class"] == "consistent_loss"
            ),
            "unstable_vs_random": sum(
                1 for d in by_dataset if d["vs_random_class"] == "unstable"
            ),
        },
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
        print(f"SMOKE: 1 dataset × seed 42 × {len(ALL_METHODS)} methods")
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
                run_rows = run_single(split, seed)
                rows.extend(run_rows)
                done += 1
                online = next(r for r in run_rows if r["method"] == "online_entropy")
                random = next(r for r in run_rows if r["method"] == "random")
                rho = next(r for r in run_rows if r["method"] == "full_rho90")
                print(
                    f"{split.name} s{seed}: target={online['target_n_balls']} | "
                    f"online F1={online['macro_f1']:.3f} | random={random['macro_f1']:.3f} "
                    f"d={online['macro_f1']-random['macro_f1']:+.3f} | "
                    f"rho={rho['macro_f1']:.3f} ({rho['n_balls']} balls)"
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
