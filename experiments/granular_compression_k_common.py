"""
Shared helpers for granular compression K-sensitivity experiments.

Selection uses frozen rho90 balls only (train-derived). Test labels are never used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.analyze_ball_utility_landscape import compute_ball_features
from experiments.analyze_granular_geometry_vs_performance import (
    _build_rho_granulation,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT
from experiments.phase8_common import normalize_labels
from experiments.test_adaptive_entropy_granulation_v2 import compute_extended_stats
from experiments.test_split_utility_granulation import _labels, _to_ball_info
from src.granular_balls import BallInfo, GranularBall

EPS = 1e-12
RHO_PURITY = 0.90
COLLAPSE_BALL_THRESHOLD = 3
K_VALUES = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48]
RANDOM_REPS = 10
WIN_THRESHOLD = 8
DELTA_EPS = 1e-6

BALL_METHODS = (
    "random_k",
    "entropy_k",
    "boundary_k",
    "entropy_boundary_k",
    "purity_k",
    "stratified_random_k",
)
SAMPLE_METHODS = ("sample_random_k", "sample_stratified_k")
REFERENCE_METHOD = "full_rho90"


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def deterministic_rng_seed(
    dataset: str,
    seed: int,
    k: int,
    *,
    method: str = "",
    repetition: int = 0,
) -> int:
    """Stable 31-bit seed from dataset + seed + K (+ method/rep)."""
    key = f"{dataset}|{seed}|{k}|{method}|{repetition}"
    return int(abs(hash(key)) % (2**31 - 1))


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
    if criterion == "entropy_boundary":
        ent = np.array([f["entropy"] for f in features], dtype=float)
        bnd = np.array([f["boundary_score"] for f in features], dtype=float)
        ent_n = (ent - ent.min()) / (ent.max() - ent.min() + EPS)
        bnd_n = (bnd - bnd.min()) / (bnd.max() - bnd.min() + EPS)
        return ent_n * bnd_n
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
        return sorted(set(selected))[:k]

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


def select_top_k_indices(
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
    return _ensure_class_balance(balls, order, k, minority_label, scores=scores, rng_seed=rng_seed)


def select_stratified_random_indices(
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


def samples_represented(kept: list[GranularBall], minority_label: float) -> dict[str, int]:
    n_min = 0
    n_maj = 0
    for ball in kept:
        labs = _labels(ball)
        n_min += int(np.sum(labs == minority_label))
        n_maj += int(np.sum(labs != minority_label))
    return {
        "n_samples_represented": n_min + n_maj,
        "minority_samples_represented": n_min,
        "majority_samples_represented": n_maj,
    }


def representation_stats(
    kept: list[GranularBall],
    indexed: np.ndarray,
    global_min: float,
    minority_label: float,
) -> dict[str, Any]:
    depths = [0] * len(kept)
    rep = compute_extended_stats(kept, depths, indexed, global_min)
    feats = compute_ball_features(kept, minority_label)
    samp = samples_represented(kept, minority_label)
    return {
        "n_balls": len(kept),
        **samp,
        "minority_survival": rep["minority_survival"],
        "majority_survival": rep["majority_survival"],
        "mean_purity": rep["mean_purity"],
        "mean_entropy": rep["mean_entropy"],
        "mean_radius": rep["mean_radius"],
        "mean_boundary_score": _nanmean([f["boundary_score"] for f in feats]),
        "fraction_mixed_balls": rep["fraction_mixed_balls"],
    }


def evaluate_balls(
    ball_infos: list[BallInfo],
    kept: list[GranularBall],
    *,
    split,
    seed: int,
    indexed: np.ndarray,
    global_min: float,
    minority_label: float,
) -> dict[str, Any]:
    perf = fit_gaussian_gblstsvm(
        ball_infos,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=minority_label,
        seed=seed,
    )
    stats = representation_stats(kept, indexed, global_min, minority_label)
    macro_f1 = float(perf["macro_f1"])
    n_balls = stats["n_balls"]
    collapsed = (
        n_balls <= COLLAPSE_BALL_THRESHOLD
        or not bool(perf.get("classifier_ok", False))
        or not np.isfinite(macro_f1)
    )
    reason = ""
    if n_balls <= COLLAPSE_BALL_THRESHOLD:
        reason = f"n_balls<={COLLAPSE_BALL_THRESHOLD}"
    elif not bool(perf.get("classifier_ok", False)):
        reason = str(perf.get("classifier_error") or "classifier_failed")
    elif not np.isfinite(macro_f1):
        reason = "invalid_macro_f1"

    return {
        **stats,
        "accuracy": float(perf["accuracy"]),
        "macro_f1": macro_f1,
        "minority_f1": float(perf["minority_f1"]),
        "gmean": float(perf["gmean"]),
        "classifier_ok": bool(perf.get("classifier_ok", False)),
        "collapsed": int(collapsed),
        "collapse_reason": reason,
        "invalid_run": int(not bool(perf.get("classifier_ok", False)) or not np.isfinite(macro_f1)),
    }


def select_training_sample_indices(
    y_train: np.ndarray,
    n_target: int,
    minority_label: float,
    rng_seed: int,
    *,
    stratified: bool,
) -> np.ndarray:
    y = normalize_labels(y_train)
    n = len(y)
    if n_target >= n:
        return np.arange(n, dtype=int)
    n_target = max(1, n_target)

    rng = np.random.RandomState(rng_seed)
    min_idx = np.flatnonzero(y == minority_label)
    maj_idx = np.flatnonzero(y != minority_label)

    if not stratified:
        return np.sort(rng.choice(n, size=n_target, replace=False))

    n_min = int(round(n_target * len(min_idx) / n)) if n else 0
    n_maj = n_target - n_min
    if n_target >= 2:
        if len(min_idx) and n_min == 0:
            n_min = 1
            n_maj = n_target - 1
        if len(maj_idx) and n_maj == 0:
            n_maj = 1
            n_min = n_target - 1
    n_min = min(n_min, len(min_idx))
    n_maj = min(n_maj, len(maj_idx))
    while n_min + n_maj < n_target and (n_min < len(min_idx) or n_maj < len(maj_idx)):
        if n_maj < len(maj_idx) and n_maj <= n_min:
            n_maj += 1
        elif n_min < len(min_idx):
            n_min += 1
        else:
            break

    chosen: list[int] = []
    if len(min_idx) and n_min > 0:
        chosen.extend(rng.choice(min_idx, size=n_min, replace=False).tolist())
    if len(maj_idx) and n_maj > 0:
        chosen.extend(rng.choice(maj_idx, size=n_maj, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=int))


def build_rho90_from_train_indices(
    indexed: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[list[BallInfo], list[GranularBall]]:
    sub = indexed[train_indices]
    kept, _ = _build_rho_granulation(sub, seed=seed, purity=RHO_PURITY)
    kept = [b for b in kept if int(b.num) >= MIN_BALL_COUNT]
    balls = [_to_ball_info(b) for b in kept]
    return balls, kept
