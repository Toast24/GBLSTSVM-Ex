"""
Shared helpers for sample-control reconciliation experiment.

Documents and reproduces the two previous sample-control semantics without
modifying either original experiment.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.metrics import recall_score

from experiments.analyze_ball_utility_landscape import compute_ball_features
from experiments.analyze_granular_geometry_vs_performance import build_representation
from experiments.granular_compression_k_common import (
    COLLAPSE_BALL_THRESHOLD,
    build_rho90_from_train_indices,
    deterministic_rng_seed,
    evaluate_balls,
    select_top_k_indices,
    select_training_sample_indices,
    samples_represented,
)
from experiments.phase8_common import normalize_labels
from experiments.test_adaptive_entropy_granulation_v2 import _global_min_label
from experiments.test_split_utility_granulation import _build_indexed_matrix
from experiments.validation_selected_k_common import (
    evaluate_representation_full,
    sample_control_seed,
    samples_to_ball_infos,
)

REPRESENTATION = "rho0.90"
K_VALUES = [2, 4, 6, 8, 10, 12]
SEEDS = list(range(10))

# Documented semantics of the two previous controls (see README).
CONTROL_SEMANTICS = {
    "sample_stratified_k_original": {
        "source_experiment": "test_granular_compression_k_sensitivity.py",
        "k_meaning": "K = number of rho90 balls selected by entropy_k (not sample count)",
        "n_target_derivation": "sum of training samples inside entropy_k ball subset",
        "sample_selection_rng": "deterministic_rng_seed(dataset, seed, K, method='sample_stratified_k')",
        "selected_objects": "original training sample row indices",
        "post_selection_representation": "rebuild rho90 granulation on subsampled train indices",
        "y_train_for_classifier_weights": "full benchmark train labels (split.y_train)",
        "training_path": "ball-based Gaussian GBLSTSVM via evaluate_balls/fit_gaussian_gblstsvm",
        "membership": "Gaussian on rebuilt rho90 balls (non-singleton radii)",
    },
    "sample_stratified_matched_original": {
        "source_experiment": "test_validation_selected_k.py",
        "k_meaning": "K not used directly; n_target = samples represented by granular compression",
        "n_target_derivation": "n_samples_represented from entropy_k at K (reconciliation uses same K)",
        "sample_selection_rng": "sample_control_seed(dataset, seed, n_target, rep=0, method='sample_stratified_matched')",
        "selected_objects": "original training sample row indices",
        "post_selection_representation": "one singleton BallInfo per sample (radius=1.0, purity=1.0)",
        "y_train_for_classifier_weights": "subset labels only (y_sub)",
        "training_path": "ball-based Gaussian GBLSTSVM via evaluate_representation_full",
        "membership": "Gaussian on singleton balls",
    },
    "singleton_ball_subset": {
        "source_experiment": "reconciliation semantic control",
        "k_meaning": "same K as entropy_k; shares sample indices with k_original selection",
        "n_target_derivation": "same as sample_stratified_k_original",
        "sample_selection_rng": "same as sample_stratified_k_original",
        "selected_objects": "identical indices to k_original",
        "post_selection_representation": "singleton BallInfo per selected sample",
        "y_train_for_classifier_weights": "subset labels (y_sub)",
        "training_path": "evaluate_representation_full",
        "membership": "Gaussian on singleton balls",
    },
    "raw_sample_subset": {
        "source_experiment": "reconciliation semantic control",
        "limitation": (
            "Repository has no non-ball raw-sample Gaussian GBLSTSVM path without modifying src/. "
            "Proxy: singleton balls on identical indices with FULL-TRAIN y for class weights "
            "(isolates weighting vs singleton_ball_subset; still uses ball+Gaussian path)."
        ),
        "y_train_for_classifier_weights": "full benchmark train labels (split.y_train)",
    },
    "granular_compressed_entropy_k": {
        "source_experiment": "K-sensitivity entropy_k ball subsampling",
        "k_meaning": "K rho90 balls selected by entropy ranking",
        "post_selection_representation": "subset of frozen full-train rho90 balls (no sample removal)",
        "y_train_for_classifier_weights": "full benchmark train labels",
    },
}


def selection_hash(indices: np.ndarray) -> str:
    idx = np.sort(np.asarray(indices, dtype=int))
    return hashlib.md5(idx.tobytes()).hexdigest()[:16]


def class_counts(y: np.ndarray, minority_label: float) -> dict[str, int | float]:
    y_n = normalize_labels(y)
    n_min = int(np.sum(y_n == minority_label))
    n_maj = int(np.sum(y_n != minority_label))
    n = n_min + n_maj
    ratio = n_min / n if n else float("nan")
    return {
        "actual_samples": n,
        "class0_count": n_maj if minority_label == 1.0 else n_min,
        "class1_count": n_min if minority_label == 1.0 else n_maj,
        "minority_count": n_min,
        "majority_count": n_maj,
        "class_ratio_minority": ratio,
    }


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else float("nan")


def derive_n_target_from_entropy_k(
    balls_rho90,
    kept_rho90,
    features,
    k: int,
    minority_label: float,
    dataset: str,
    seed: int,
) -> tuple[int, list[int]]:
    idx = select_top_k_indices(
        balls_rho90,
        features,
        k,
        "entropy",
        minority_label,
        deterministic_rng_seed(dataset, seed, k, method="entropy_k"),
    )
    kept = [kept_rho90[i] for i in idx]
    n_target = samples_represented(kept, minority_label)["n_samples_represented"]
    return int(n_target), idx


def select_indices_k_sensitivity(
    y_train: np.ndarray,
    n_target: int,
    minority_label: float,
    dataset: str,
    seed: int,
    k: int,
) -> np.ndarray:
    rng_seed = deterministic_rng_seed(dataset, seed, k, method="sample_stratified_k")
    return select_training_sample_indices(
        y_train, n_target, minority_label, rng_seed, stratified=True
    )


def select_indices_validation_matched(
    y_train: np.ndarray,
    n_target: int,
    minority_label: float,
    dataset: str,
    seed: int,
    repetition: int = 0,
) -> np.ndarray:
    rng_seed = sample_control_seed(
        dataset, seed, n_target, repetition, method="sample_stratified_matched"
    )
    return select_training_sample_indices(
        y_train, n_target, minority_label, rng_seed, stratified=True
    )


def _recall(y_true, y_pred, label):
    y_t = normalize_labels(y_true)
    y_p = normalize_labels(y_pred)
    y_p[y_p == 0.0] = -1.0
    return float(recall_score(y_t, y_p, pos_label=label, zero_division=0))


def run_sample_stratified_k_original(split, seed: int, k: int) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    minority_label = split.minority_label

    balls_rho90, kept_rho90, _ = build_representation(split, seed, REPRESENTATION)
    features = compute_ball_features(kept_rho90, minority_label)
    if len(balls_rho90) < k:
        return {"skipped": 1, "skip_reason": f"pool<{k}"}

    n_target, entropy_idx = derive_n_target_from_entropy_k(
        balls_rho90, kept_rho90, features, k, minority_label, split.name, seed
    )
    train_idx = select_indices_k_sensitivity(y_train, n_target, minority_label, split.name, seed, k)
    sub_balls, sub_kept = build_rho90_from_train_indices(indexed, train_idx, seed=seed)

    ev = evaluate_balls(
        sub_balls,
        sub_kept,
        split=split,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        minority_label=minority_label,
    )
    cc = class_counts(y_train[train_idx], minority_label)
    return {
        "method": "sample_stratified_k_original",
        "K": k,
        "requested_k": k,
        "n_target_samples": n_target,
        "entropy_ball_indices": entropy_idx,
        "selected_indices": train_idx,
        "selection_hash": selection_hash(train_idx),
        "representation_type": "rho90_rebuild_on_subsample",
        "training_path": "evaluate_balls/full_y_train_weights",
        "n_singleton_balls": 0,
        "n_non_singleton_balls": ev.get("n_balls", 0),
        **cc,
        **ev,
    }


def run_sample_stratified_matched_original(
    split, seed: int, k: int, repetition: int = 0
) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    minority_label = split.minority_label

    balls_rho90, kept_rho90, _ = build_representation(split, seed, REPRESENTATION)
    features = compute_ball_features(kept_rho90, minority_label)
    if len(balls_rho90) < k:
        return {"skipped": 1, "skip_reason": f"pool<{k}"}

    n_target, _ = derive_n_target_from_entropy_k(
        balls_rho90, kept_rho90, features, k, minority_label, split.name, seed
    )
    train_idx = select_indices_validation_matched(
        y_train, n_target, minority_label, split.name, seed, repetition
    )
    X_sub = split.X_train[train_idx]
    y_sub = y_train[train_idx]
    balls = samples_to_ball_infos(X_sub, y_sub)

    ev = evaluate_representation_full(
        balls,
        None,
        y_train=y_sub,
        X_eval=split.X_test,
        y_eval=split.y_test,
        minority_label=minority_label,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        phase="final_test",
    )
    cc = class_counts(y_sub, minority_label)
    maj_label = -minority_label
    y_pred = ev.get("y_pred") if "y_pred" in ev else None
    return {
        "method": "sample_stratified_matched_original",
        "K": k,
        "requested_k": k,
        "n_target_samples": n_target,
        "selected_indices": train_idx,
        "selection_hash": selection_hash(train_idx),
        "representation_type": "singleton_balls",
        "training_path": "evaluate_representation_full/y_sub_weights",
        "n_singleton_balls": len(balls),
        "n_non_singleton_balls": 0,
        **cc,
        **ev,
        "minority_recall": ev.get("minority_recall", float("nan")),
        "majority_recall": float("nan"),
    }


def run_singleton_ball_subset(split, seed: int, k: int, train_idx: np.ndarray) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    minority_label = split.minority_label
    y_sub = y_train[train_idx]
    balls = samples_to_ball_infos(split.X_train[train_idx], y_sub)

    ev = evaluate_representation_full(
        balls,
        None,
        y_train=y_sub,
        X_eval=split.X_test,
        y_eval=split.y_test,
        minority_label=minority_label,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        phase="final_test",
    )
    cc = class_counts(y_sub, minority_label)
    return {
        "method": "singleton_ball_subset",
        "K": k,
        "requested_k": k,
        "selected_indices": train_idx,
        "selection_hash": selection_hash(train_idx),
        "representation_type": "singleton_balls",
        "training_path": "evaluate_representation_full/y_sub_weights",
        "n_singleton_balls": len(balls),
        "n_non_singleton_balls": 0,
        **cc,
        **ev,
    }


def run_raw_sample_subset_proxy(split, seed: int, k: int, train_idx: np.ndarray) -> dict[str, Any]:
    """Proxy only — see CONTROL_SEMANTICS['raw_sample_subset']['limitation']."""
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    minority_label = split.minority_label
    y_sub = y_train[train_idx]
    balls = samples_to_ball_infos(split.X_train[train_idx], y_sub)

    ev = evaluate_representation_full(
        balls,
        None,
        y_train=y_train,
        X_eval=split.X_test,
        y_eval=split.y_test,
        minority_label=minority_label,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        phase="final_test",
    )
    cc = class_counts(y_sub, minority_label)
    return {
        "method": "raw_sample_subset",
        "K": k,
        "requested_k": k,
        "selected_indices": train_idx,
        "selection_hash": selection_hash(train_idx),
        "representation_type": "singleton_balls_proxy_full_y_weights",
        "training_path": "evaluate_representation_full/y_full_weights_PROXY",
        "raw_path_available": False,
        "n_singleton_balls": len(balls),
        "n_non_singleton_balls": 0,
        **cc,
        **ev,
    }


def run_granular_compressed_entropy_k(split, seed: int, k: int) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    minority_label = split.minority_label

    balls_rho90, kept_rho90, _ = build_representation(split, seed, REPRESENTATION)
    features = compute_ball_features(kept_rho90, minority_label)
    if len(balls_rho90) < k:
        return {"skipped": 1, "skip_reason": f"pool<{k}"}

    idx = select_top_k_indices(
        balls_rho90,
        features,
        k,
        "entropy",
        minority_label,
        deterministic_rng_seed(split.name, seed, k, method="entropy_k"),
    )
    kept = [kept_rho90[i] for i in idx]
    ev = evaluate_balls(
        [balls_rho90[i] for i in idx],
        kept,
        split=split,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        minority_label=minority_label,
    )
    return {
        "method": "granular_compressed_entropy_k",
        "K": k,
        "requested_k": k,
        "selected_indices": np.array([], dtype=int),
        "selection_hash": "",
        "representation_type": "rho90_ball_subset",
        "training_path": "evaluate_balls/full_y_train_weights",
        "n_singleton_balls": 0,
        "n_non_singleton_balls": ev.get("n_balls", 0),
        "actual_samples": ev.get("n_samples_represented", 0),
        "minority_count": ev.get("minority_samples_represented", 0),
        "majority_count": ev.get("majority_samples_represented", 0),
        **ev,
    }


def pack_indices_for_csv(indices: np.ndarray) -> str:
    if indices is None or len(indices) == 0:
        return ""
    return ",".join(str(int(i)) for i in np.sort(indices))
