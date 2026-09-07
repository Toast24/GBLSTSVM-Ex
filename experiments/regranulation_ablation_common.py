"""
Shared helpers for re-granulation ablation experiment.

Holds sample selection fixed (sample_stratified_k semantics) and compares
singleton-ball vs re-granulation at multiple rho thresholds on the same subset.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.analyze_ball_utility_landscape import compute_ball_features
from experiments.analyze_granular_geometry_vs_performance import build_representation
from experiments.analyze_topology import MIN_BALL_COUNT
from experiments.granular_compression_k_common import COLLAPSE_BALL_THRESHOLD, evaluate_balls
from experiments.phase8_common import normalize_labels
from experiments.sample_control_reconciliation_common import (
    derive_n_target_from_entropy_k,
    run_sample_stratified_k_original,
    run_singleton_ball_subset,
    select_indices_k_sensitivity,
    selection_hash,
)
from experiments.test_adaptive_entropy_granulation_v2 import (
    _global_min_label,
    compute_extended_stats,
)
from experiments.test_split_utility_granulation import (
    MIN_SAMPLES_PER_BALL,
    _build_indexed_matrix,
    _labels,
    _to_ball_info,
)
from experiments.validation_selected_k_common import samples_to_ball_infos
from src.granular_balls import BallInfo, GBList, GranularBall

REPRESENTATION = "rho0.90"
K_VALUES = [2, 4, 6, 8, 10, 12]
SEEDS = list(range(10))
DELTA_EPS = 1e-6

REGRANULATE_RHOS: dict[str, float] = {
    "regranulate_rho80": 0.80,
    "regranulate_rho85": 0.85,
    "regranulate_rho90": 0.90,
    "regranulate_rho95": 0.95,
    "regranulate_rho98": 0.98,
}

MATCHED_METHODS = ("singleton",) + tuple(REGRANULATE_RHOS.keys())
SECONDARY_BASELINE = "sample_stratified_k_original"
FULL_REFERENCE = "full_rho90"


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _nanmedian(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def build_rho_from_train_indices(
    indexed: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
    purity: float,
) -> tuple[list[BallInfo], list[GranularBall], dict[str, Any]]:
    """
    Re-granulate a fixed training subset at ``purity`` (rho).

    Mirrors ``_build_rho_granulation`` / ``build_rho90_from_train_indices`` but
    accepts arbitrary rho and records split/deletion counters.
    """
    sub = indexed[train_indices]
    np.random.seed(seed)
    balls = GBList(sub).granular_balls
    ll = len(balls)
    i = 0
    successful_splits = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < purity and ball.num > MIN_SAMPLES_PER_BALL:
            c1, c2 = ball.split_2balls()
            balls[i] = c1
            balls.append(c2)
            ll += 1
            successful_splits += 1
        else:
            i += 1

    pre_del = list(balls)
    kept = [b for b in pre_del if int(b.num) >= MIN_BALL_COUNT]
    singleton_deletions = sum(int(b.num) for b in pre_del if int(b.num) < MIN_BALL_COUNT)

    ball_infos = [_to_ball_info(b) for b in kept]
    trace = {
        "successful_splits": successful_splits,
        "failed_splits": 0,
        "singleton_deletions": int(singleton_deletions),
        "rho": float(purity),
    }
    return ball_infos, kept, trace


def compute_ball_geometry(
    kept: list[GranularBall],
    indexed_subset: np.ndarray,
    global_min: float,
    minority_label: float,
    *,
    gran_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extended geometry metrics for re-granulated representations."""
    if not kept:
        rep = compute_extended_stats([], [], indexed_subset, global_min)
        feats: list[dict[str, Any]] = []
    else:
        depths = [0] * len(kept)
        rep = compute_extended_stats(kept, depths, indexed_subset, global_min)
        feats = compute_ball_features(kept, minority_label)

    sizes = np.array([int(b.num) for b in kept], dtype=int)
    n_singletons = int(np.sum(sizes == 1)) if sizes.size else 0
    n_non_singletons = int(len(kept) - n_singletons)

    purities = np.array([float(b.purity) for b in kept], dtype=float)
    radii = np.array([float(b.r) for b in kept], dtype=float)
    entropies = np.array([f["entropy"] for f in feats], dtype=float) if feats else np.array([])

    n_min_balls = sum(1 for b in kept if float(b.label) == minority_label)
    n_maj_balls = len(kept) - n_min_balls

    trace = gran_trace or {}
    collapsed_geom = int(len(kept) <= COLLAPSE_BALL_THRESHOLD)

    return {
        "n_balls": len(kept),
        "n_singletons": n_singletons,
        "n_non_singletons": n_non_singletons,
        "mean_ball_size": float(np.mean(sizes)) if sizes.size else float("nan"),
        "median_ball_size": float(np.median(sizes)) if sizes.size else float("nan"),
        "max_ball_size": int(np.max(sizes)) if sizes.size else 0,
        "mean_purity": float(np.mean(purities)) if purities.size else float("nan"),
        "median_purity": float(np.median(purities)) if purities.size else float("nan"),
        "mean_radius": float(np.mean(radii)) if radii.size else float("nan"),
        "median_radius": float(np.median(radii)) if radii.size else float("nan"),
        "mean_entropy": float(np.mean(entropies)) if entropies.size else float("nan"),
        "median_entropy": float(np.median(entropies)) if entropies.size else float("nan"),
        "minority_balls": n_min_balls,
        "majority_balls": n_maj_balls,
        "minority_sample_survival": rep.get("minority_survival", float("nan")),
        "majority_sample_survival": rep.get("majority_survival", float("nan")),
        "successful_splits": int(trace.get("successful_splits", 0)),
        "failed_splits": int(trace.get("failed_splits", 0)),
        "singleton_deletions": int(trace.get("singleton_deletions", 0)),
        "collapse_flag": collapsed_geom,
        "max_depth": int(rep.get("max_depth", 0)),
        "mean_depth": float(rep.get("mean_depth", float("nan"))),
        "rho": float(trace.get("rho", float("nan"))),
    }


def compute_singleton_geometry(
    ball_infos: list[BallInfo],
    y_sub: np.ndarray,
    minority_label: float,
) -> dict[str, Any]:
    n_min = int(np.sum(normalize_labels(y_sub) == minority_label))
    n_maj = len(y_sub) - n_min
    n_balls = len(ball_infos)
    return {
        "n_balls": n_balls,
        "n_singletons": n_balls,
        "n_non_singletons": 0,
        "mean_ball_size": 1.0 if n_balls else float("nan"),
        "median_ball_size": 1.0 if n_balls else float("nan"),
        "max_ball_size": 1,
        "mean_purity": 1.0 if n_balls else float("nan"),
        "median_purity": 1.0 if n_balls else float("nan"),
        "mean_radius": 1.0 if n_balls else float("nan"),
        "median_radius": 1.0 if n_balls else float("nan"),
        "mean_entropy": 0.0 if n_balls else float("nan"),
        "median_entropy": 0.0 if n_balls else float("nan"),
        "minority_balls": sum(1 for b in ball_infos if b.label == minority_label),
        "majority_balls": sum(1 for b in ball_infos if b.label != minority_label),
        "minority_sample_survival": n_min / n_min if n_min else float("nan"),
        "majority_sample_survival": n_maj / n_maj if n_maj else float("nan"),
        "successful_splits": 0,
        "failed_splits": 0,
        "singleton_deletions": 0,
        "collapse_flag": int(n_balls <= COLLAPSE_BALL_THRESHOLD),
        "max_depth": 0,
        "mean_depth": 0.0,
        "rho": float("nan"),
    }


def verify_selection_hashes(hash_map: dict[str, str], context: str) -> None:
    """Raise if any branch diverges from the reference hash."""
    if not hash_map:
        return
    ref_method = "singleton" if "singleton" in hash_map else next(iter(hash_map))
    ref = hash_map[ref_method]
    for method, h in hash_map.items():
        if h != ref:
            raise ValueError(
                f"selected_sample_hash mismatch in {context}: "
                f"{ref_method}={ref} vs {method}={h}"
            )


def select_shared_subset(split, seed: int, k: int) -> dict[str, Any]:
    """Deterministic stratified sample subset (sample_stratified_k semantics)."""
    y_train = normalize_labels(split.y_train)
    minority_label = split.minority_label

    balls_rho90, kept_rho90, _ = build_representation(split, seed, REPRESENTATION)
    from experiments.analyze_ball_utility_landscape import compute_ball_features

    features = compute_ball_features(kept_rho90, minority_label)
    if len(balls_rho90) < k:
        return {"skipped": 1, "skip_reason": f"pool<{k}"}

    n_target, entropy_idx = derive_n_target_from_entropy_k(
        balls_rho90, kept_rho90, features, k, minority_label, split.name, seed
    )
    train_idx = select_indices_k_sensitivity(
        y_train, n_target, minority_label, split.name, seed, k
    )
    return {
        "train_idx": train_idx,
        "n_target_samples": n_target,
        "requested_k": k,
        "actual_retained_samples": int(len(train_idx)),
        "selection_hash": selection_hash(train_idx),
        "entropy_ball_indices": entropy_idx,
        "n_rho90_pool": len(balls_rho90),
    }


def run_regranulate_method(
    split,
    seed: int,
    k: int,
    train_idx: np.ndarray,
    method: str,
    rho: float,
    *,
    indexed: np.ndarray | None = None,
    global_min: float | None = None,
) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    if indexed is None:
        indexed = _build_indexed_matrix(split.X_train, y_train)
    if global_min is None:
        global_min = _global_min_label(indexed)
    minority_label = split.minority_label

    sub_indexed = indexed[train_idx]
    ball_infos, kept, trace = build_rho_from_train_indices(
        indexed, train_idx, seed=seed, purity=rho
    )
    ev = evaluate_balls(
        ball_infos,
        kept,
        split=split,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        minority_label=minority_label,
    )
    geom = compute_ball_geometry(
        kept, sub_indexed, global_min, minority_label, gran_trace=trace
    )
    h = selection_hash(train_idx)
    return {
        "method": method,
        "rho": rho,
        "K": k,
        "requested_k": k,
        "actual_retained_samples": int(len(train_idx)),
        "selected_indices": train_idx,
        "selection_hash": h,
        "full_data_reference": False,
        "representation_type": f"rho{rho:.2f}_rebuild_on_subsample",
        "training_path": "evaluate_balls/full_y_train_weights",
        **ev,
        **geom,
    }


def run_singleton_method(
    split,
    seed: int,
    k: int,
    train_idx: np.ndarray,
) -> dict[str, Any]:
    r = run_singleton_ball_subset(split, seed, k, train_idx)
    y_sub = normalize_labels(split.y_train)[train_idx]
    balls = samples_to_ball_infos(split.X_train[train_idx], y_sub)
    geom = compute_singleton_geometry(balls, y_sub, split.minority_label)
    return {
        **r,
        **geom,
        "method": "singleton",
        "rho": float("nan"),
        "actual_retained_samples": int(len(train_idx)),
        "full_data_reference": False,
    }


def run_full_rho90_reference(split, seed: int) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    minority_label = split.minority_label

    balls_rho90, kept_rho90, _ = build_representation(split, seed, REPRESENTATION)
    ev = evaluate_balls(
        balls_rho90,
        kept_rho90,
        split=split,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        minority_label=minority_label,
    )
    geom = compute_ball_geometry(kept_rho90, indexed, global_min, minority_label)
    return {
        "method": FULL_REFERENCE,
        "rho": 0.90,
        "K": float("nan"),
        "requested_k": float("nan"),
        "actual_retained_samples": int(len(y_train)),
        "selected_indices": np.arange(len(y_train), dtype=int),
        "selection_hash": "",
        "full_data_reference": True,
        "representation_type": "full_train_rho90",
        "training_path": "evaluate_balls/full_y_train_weights",
        **ev,
        **geom,
    }


def run_matched_k_block(split, seed: int, k: int) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    """Run all matched-subset methods for one dataset×seed×K."""
    sel = select_shared_subset(split, seed, k)
    if sel.get("skipped"):
        return [], {"skip_reason": sel["skip_reason"]}

    train_idx = sel["train_idx"]
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)

    results: list[dict[str, Any]] = []
    hash_map: dict[str, str] = {}

    r_singleton = run_singleton_method(split, seed, k, train_idx)
    hash_map["singleton"] = r_singleton["selection_hash"]
    results.append({**sel, **r_singleton})

    for method, rho in REGRANULATE_RHOS.items():
        r = run_regranulate_method(
            split, seed, k, train_idx, method, rho, indexed=indexed, global_min=global_min
        )
        hash_map[method] = r["selection_hash"]
        results.append({**sel, **r})

    verify_selection_hashes(hash_map, context=f"{split.name} seed={seed} K={k}")

    r_korig = run_sample_stratified_k_original(split, seed, k)
    if not r_korig.get("skipped"):
        k_hash = r_korig.get("selection_hash", "")
        if k_hash != sel["selection_hash"]:
            raise ValueError(
                f"sample_stratified_k_original hash mismatch at {split.name} s={seed} K={k}"
            )
        _, sub_kept, trace = build_rho_from_train_indices(
            indexed, train_idx, seed=seed, purity=0.90
        )
        geom_k = compute_ball_geometry(
            sub_kept,
            indexed[train_idx],
            global_min,
            split.minority_label,
            gran_trace=trace,
        )
        results.append(
            {
                **sel,
                **r_korig,
                **geom_k,
                "rho": 0.90,
                "full_data_reference": False,
                "method": SECONDARY_BASELINE,
            }
        )

    return results, None
