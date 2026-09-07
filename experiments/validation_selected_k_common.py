"""
Shared helpers for validation-based K-selection experiments.

Leakage policy (enforced by call structure, documented in README):
  - rho90 granulation, ball ranking, compression, and K selection use train_selection only.
  - validation (internal 20% of benchmark train) is used ONLY to choose K.
  - benchmark test is used ONLY for final evaluation after K is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from experiments.analyze_ball_utility_landscape import compute_ball_features
from experiments.analyze_granular_geometry_vs_performance import (
    _class_f1,
    build_representation,
    fit_gaussian_gblstsvm,
)
from experiments.granular_compression_k_common import (
    COLLAPSE_BALL_THRESHOLD,
    build_rho90_from_train_indices,
    deterministic_rng_seed,
    representation_stats,
    select_top_k_indices,
    select_training_sample_indices,
    samples_represented,
)
from experiments.phase8_common import evaluate, normalize_labels
from experiments.test_adaptive_entropy_granulation_v2 import _global_min_label
from experiments.test_split_utility_granulation import _build_indexed_matrix, _labels
from src.granular_balls import BallInfo, GranularBall

EPS = 1e-12
REPRESENTATION = "rho0.90"
K_CANDIDATES_DEFAULT = [4, 6, 8, 10, 12, 16, 20, 24, 32]
COMPRESSION_REPS = 5
FIXED_K_VALUES = [6, 8, 10]
TRAIN_VAL_FRAC = 0.20
DELTA_EPS = 1e-6
SEEDS_DEFAULT = list(range(10))

Phase = Literal["selection", "final_test"]


@dataclass(frozen=True)
class TrainValPartition:
    """Internal split of benchmark train only. Test data must never appear here."""

    dataset: str
    benchmark_seed: int
    selection_indices: np.ndarray
    validation_indices: np.ndarray
    split_seed: int


def train_val_split_seed(dataset: str, benchmark_seed: int) -> int:
    """Deterministic seed for stratified 80/20 split of benchmark train."""
    return deterministic_rng_seed(dataset, benchmark_seed, 0, method="train_val_split")


def compression_seed(
    dataset: str,
    benchmark_seed: int,
    k: int,
    repetition: int,
    *,
    phase: Phase = "selection",
) -> int:
    return deterministic_rng_seed(
        dataset, benchmark_seed, k, method=f"random_compress_{phase}", repetition=repetition
    )


def sample_control_seed(
    dataset: str,
    benchmark_seed: int,
    n_target: int,
    repetition: int,
    *,
    method: str = "sample_stratified",
) -> int:
    return deterministic_rng_seed(
        dataset, benchmark_seed, n_target, method=method, repetition=repetition
    )


def make_train_val_partition(
    dataset: str,
    benchmark_seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> TrainValPartition:
    """Stratified 80/20 split of benchmark train. No test leakage."""
    y = normalize_labels(y_train)
    n = len(y)
    indices = np.arange(n, dtype=int)
    split_seed = train_val_split_seed(dataset, benchmark_seed)
    sel_idx, val_idx = train_test_split(
        indices,
        test_size=TRAIN_VAL_FRAC,
        random_state=split_seed,
        stratify=y,
    )
    sel_idx = np.sort(np.asarray(sel_idx, dtype=int))
    val_idx = np.sort(np.asarray(val_idx, dtype=int))
    assert len(set(sel_idx) & set(val_idx)) == 0
    assert len(sel_idx) + len(val_idx) == n
    return TrainValPartition(
        dataset=dataset,
        benchmark_seed=benchmark_seed,
        selection_indices=sel_idx,
        validation_indices=val_idx,
        split_seed=split_seed,
    )


def assert_partition_disjoint_from_test(
    partition: TrainValPartition,
    n_train: int,
    n_test: int,
    *,
    context: str,
) -> None:
    """Sanity: internal indices are within train only."""
    assert partition.selection_indices.max() < n_train, context
    assert partition.validation_indices.max() < n_train, context
    assert n_test >= 0


def build_rho90_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[list[BallInfo], list[GranularBall], np.ndarray, float]:
    y = normalize_labels(y_train)
    indexed = _build_indexed_matrix(X_train, y)
    global_min = _global_min_label(indexed)
    balls, kept = build_rho90_from_train_indices(indexed, train_indices, seed=seed)
    return balls, kept, indexed, global_min


def random_compress_indices(
    balls: list[BallInfo],
    features: list[dict[str, Any]],
    k: int,
    minority_label: float,
    rng_seed: int,
) -> list[int]:
    return select_top_k_indices(balls, features, k, "random", minority_label, rng_seed)


def samples_to_ball_infos(X: np.ndarray, y: np.ndarray) -> list[BallInfo]:
    """One ball per training sample — sample-level GBLSTSVM without granulation."""
    y_n = normalize_labels(y)
    balls: list[BallInfo] = []
    for i in range(len(y_n)):
        balls.append(
            BallInfo(
                center=np.asarray(X[i], dtype=float),
                radius=1.0,
                label=float(y_n[i]),
                purity=1.0,
                num_samples=1,
            )
        )
    return balls


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _nanstd(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if v.size > 1 else (0.0 if v.size == 1 else float("nan"))


def _nanmedian(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def evaluate_representation(
    ball_infos: list[BallInfo],
    kept: list[GranularBall] | None,
    *,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    minority_label: float,
    seed: int,
    indexed: np.ndarray,
    global_min: float,
    phase: Phase,
) -> dict[str, Any]:
    """
    Fit Gaussian GBLSTSVM on ball_infos (trained with y_train) and evaluate on (X_eval, y_eval).

    phase='selection' must use internal validation features only.
    phase='final_test' must use benchmark test features only.
    """
    if phase not in ("selection", "final_test"):
        raise ValueError(f"Invalid phase {phase!r}")

    perf = fit_gaussian_gblstsvm(
        ball_infos,
        y_train=y_train,
        X_test=X_eval,
        y_test=y_eval,
        minority_label=minority_label,
        seed=seed,
    )
    majority_label = -minority_label

    if kept is not None and len(kept) > 0:
        stats = representation_stats(kept, indexed, global_min, minority_label)
    elif ball_infos:
        n_min = sum(1 for b in ball_infos if b.label == minority_label)
        n_maj = len(ball_infos) - n_min
        samp = samples_represented_from_ball_infos(ball_infos, minority_label)
        stats = {
            "n_balls": len(ball_infos),
            "minority_balls": n_min,
            "majority_balls": n_maj,
            **samp,
            "minority_survival": float("nan"),
            "majority_survival": float("nan"),
            "mean_purity": 1.0,
            "mean_entropy": 0.0,
            "mean_radius": _nanmean([b.radius for b in ball_infos]),
            "mean_boundary_score": float("nan"),
            "fraction_mixed_balls": 0.0,
        }
    else:
        stats = {
            "n_balls": 0,
            "minority_balls": 0,
            "majority_balls": 0,
            "n_samples_represented": 0,
            "minority_samples_represented": 0,
            "majority_samples_represented": 0,
            "minority_survival": float("nan"),
            "majority_survival": float("nan"),
            "mean_purity": float("nan"),
            "mean_entropy": float("nan"),
            "mean_radius": float("nan"),
            "mean_boundary_score": float("nan"),
            "fraction_mixed_balls": float("nan"),
        }

    macro_f1 = float(perf.get("macro_f1", float("nan")))
    n_balls = int(stats["n_balls"])
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

    y_eval_n = normalize_labels(y_eval)
    y_pred = np.asarray(perf.get("y_pred", []), dtype=float) if "y_pred" in perf else None
    if y_pred is None or y_pred.size != len(y_eval_n):
        extra = {
            "balanced_accuracy": float("nan"),
            "minority_recall": float("nan"),
            "minority_precision": float("nan"),
            "majority_f1": float("nan"),
        }
    else:
        y_pred_eval = y_pred.copy()
        y_pred_eval[y_pred_eval == 0.0] = -1.0
        extra = {
            "balanced_accuracy": float(
                evaluate(y_eval_n, y_pred, perf.get("scores"), minority_label).get(
                    "balanced_accuracy", float("nan")
                )
            ),
            "minority_recall": float(
                recall_score(
                    y_eval_n, y_pred_eval, pos_label=minority_label, zero_division=0
                )
            ),
            "minority_precision": float(
                precision_score(
                    y_eval_n, y_pred_eval, pos_label=minority_label, zero_division=0
                )
            ),
            "majority_f1": _class_f1(y_eval_n, y_pred_eval, majority_label),
        }

    return {
        **stats,
        "accuracy": float(perf.get("accuracy", float("nan"))),
        "macro_f1": macro_f1,
        "minority_f1": float(perf.get("minority_f1", float("nan"))),
        "gmean": float(perf.get("gmean", float("nan"))),
        **extra,
        "classifier_ok": bool(perf.get("classifier_ok", False)),
        "collapsed": int(collapsed),
        "collapse_reason": reason,
        "invalid_run": int(
            not bool(perf.get("classifier_ok", False)) or not np.isfinite(macro_f1)
        ),
        "phase": phase,
    }


def samples_represented_from_ball_infos(
    ball_infos: list[BallInfo], minority_label: float
) -> dict[str, int]:
    n_min = sum(int(b.num_samples) for b in ball_infos if b.label == minority_label)
    n_maj = sum(int(b.num_samples) for b in ball_infos if b.label != minority_label)
    return {
        "n_samples_represented": n_min + n_maj,
        "minority_samples_represented": n_min,
        "majority_samples_represented": n_maj,
    }


def fit_gaussian_with_predictions(
    ball_infos: list[BallInfo],
    *,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    minority_label: float,
    seed: int,
) -> dict[str, Any]:
    """Wrapper returning y_pred for extended metrics."""
    from experiments.gaussian_membership_screen import compute_gaussian_memberships
    from experiments.analyze_granular_geometry_vs_performance import (
        PREDICTION_RULE,
        REG_TERM,
        WEIGHT_PLACEMENT,
    )
    from src.cbcm import split_membership_by_class
    from src.class_weights import ball_class_weights, compute_binary_class_weights
    from src.cw_gblstsvm import solve_cw_gblstsvm_systems
    from src.gblstsvm import predict_twin_hyperplanes
    from src.granular_balls import balls_to_class_arrays
    from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2
    from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN

    y = normalize_labels(y_train)
    if not ball_infos:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "minority_f1": float("nan"),
            "gmean": float("nan"),
            "classifier_ok": False,
            "classifier_error": "no balls",
            "y_pred": np.array([]),
            "scores": np.array([]),
        }
    try:
        balls_to_class_arrays(ball_infos)
    except ValueError as exc:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "minority_f1": float("nan"),
            "gmean": float("nan"),
            "classifier_ok": False,
            "classifier_error": str(exc),
            "y_pred": np.array([]),
            "scores": np.array([]),
        }

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y)
    kappa_pos_b, kappa_neg_b = ball_class_weights(ball_infos, kappa_pos, kappa_neg)
    mu_g = compute_gaussian_memberships(ball_infos, PRIMARY_GAUSSIAN)
    mu_pos, mu_neg = split_membership_by_class(ball_infos, mu_g)
    weights_pos = kappa_pos_b * mu_pos
    weights_neg = kappa_neg_b * mu_neg
    centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(ball_infos)
    np.random.seed(seed)
    w1, b1, w2, b2 = solve_cw_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        DEFAULT_C1,
        DEFAULT_C2,
        weights_pos,
        weights_neg,
        reg_term=REG_TERM,
        weight_placement=WEIGHT_PLACEMENT,
    )
    y_pred = predict_twin_hyperplanes(X_eval, w1, b1, w2, b2, rule=PREDICTION_RULE)
    scores = np.abs(X_eval @ w2 + b2) - np.abs(X_eval @ w1 + b1)
    metrics = evaluate(y_eval, y_pred, scores, minority_label)
    metrics["minority_f1"] = _class_f1(y_eval, y_pred, minority_label)
    metrics["classifier_ok"] = True
    metrics["classifier_error"] = None
    metrics["y_pred"] = y_pred
    metrics["scores"] = scores
    return metrics


def evaluate_representation_full(
    ball_infos: list[BallInfo],
    kept: list[GranularBall] | None,
    *,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    minority_label: float,
    seed: int,
    indexed: np.ndarray,
    global_min: float,
    phase: Phase,
) -> dict[str, Any]:
    perf = fit_gaussian_with_predictions(
        ball_infos,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        minority_label=minority_label,
        seed=seed,
    )
    majority_label = -minority_label
    y_eval_n = normalize_labels(y_eval)

    if kept is not None and len(kept) > 0:
        stats = representation_stats(kept, indexed, global_min, minority_label)
        n_min_balls = sum(1 for b in kept if _labels(b)[0] == minority_label or float(b.label) == minority_label)
        # count by ball majority label from BallInfo mapping
        n_min_balls = sum(1 for bi in ball_infos if bi.label == minority_label)
        n_maj_balls = len(ball_infos) - n_min_balls
        stats["minority_balls"] = n_min_balls
        stats["majority_balls"] = n_maj_balls
    elif ball_infos:
        n_min_balls = sum(1 for b in ball_infos if b.label == minority_label)
        samp = samples_represented_from_ball_infos(ball_infos, minority_label)
        stats = {
            "n_balls": len(ball_infos),
            "minority_balls": n_min_balls,
            "majority_balls": len(ball_infos) - n_min_balls,
            **samp,
            "minority_survival": float("nan"),
            "majority_survival": float("nan"),
            "mean_purity": 1.0,
            "mean_entropy": 0.0,
            "mean_radius": _nanmean([b.radius for b in ball_infos]),
            "mean_boundary_score": float("nan"),
            "fraction_mixed_balls": 0.0,
        }
    else:
        stats = {
            "n_balls": 0,
            "minority_balls": 0,
            "majority_balls": 0,
            "n_samples_represented": 0,
            "minority_samples_represented": 0,
            "majority_samples_represented": 0,
            "minority_survival": float("nan"),
            "majority_survival": float("nan"),
            "mean_purity": float("nan"),
            "mean_entropy": float("nan"),
            "mean_radius": float("nan"),
            "mean_boundary_score": float("nan"),
            "fraction_mixed_balls": float("nan"),
        }

    macro_f1 = float(perf.get("macro_f1", float("nan")))
    n_balls = int(stats["n_balls"])
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

    y_pred = np.asarray(perf.get("y_pred", []), dtype=float)
    if (
        y_pred.size == len(y_eval_n)
        and bool(perf.get("classifier_ok", False))
        and np.isfinite(macro_f1)
    ):
        y_pred_eval = y_pred.copy()
        y_pred_eval[y_pred_eval == 0.0] = -1.0
        extra_metrics = {
            "balanced_accuracy": float(perf.get("balanced_accuracy", float("nan"))),
            "minority_recall": float(perf.get("minority_recall", float("nan"))),
            "minority_precision": float(
                precision_score(
                    y_eval_n, y_pred_eval, pos_label=minority_label, zero_division=0
                )
            ),
            "majority_f1": _class_f1(y_eval_n, y_pred_eval, majority_label),
        }
    else:
        extra_metrics = {
            "balanced_accuracy": float("nan"),
            "minority_recall": float("nan"),
            "minority_precision": float("nan"),
            "majority_f1": float("nan"),
        }

    return {
        **stats,
        "accuracy": float(perf.get("accuracy", float("nan"))),
        "macro_f1": macro_f1,
        "minority_f1": float(perf.get("minority_f1", float("nan"))),
        "gmean": float(perf.get("gmean", float("nan"))),
        **extra_metrics,
        "classifier_ok": bool(perf.get("classifier_ok", False)),
        "collapsed": int(collapsed),
        "collapse_reason": reason,
        "invalid_run": int(
            not bool(perf.get("classifier_ok", False)) or not np.isfinite(macro_f1)
        ),
        "phase": phase,
    }


@dataclass
class KValidationSummary:
    K: int
    mean_macro_f1: float
    std_macro_f1: float
    median_macro_f1: float
    n_successful: int
    mean_collapsed: float
    mean_n_balls: float
    mean_n_samples: float
    skipped: bool
    skip_reason: str


def summarize_k_validation(rep_rows: list[dict[str, Any]]) -> KValidationSummary:
    k = int(rep_rows[0]["K"]) if rep_rows else -1
    if not rep_rows:
        return KValidationSummary(
            k, float("nan"), float("nan"), float("nan"), 0, float("nan"), float("nan"), float("nan"), True, "no_rows"
        )
    if rep_rows[0].get("skipped"):
        return KValidationSummary(
            k,
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            True,
            str(rep_rows[0].get("skip_reason", "")),
        )
    f1s = [float(r["macro_f1"]) for r in rep_rows if np.isfinite(r.get("macro_f1", float("nan")))]
    ok = [r for r in rep_rows if r.get("classifier_ok")]
    return KValidationSummary(
        K=k,
        mean_macro_f1=_nanmean(f1s),
        std_macro_f1=_nanstd(f1s),
        median_macro_f1=_nanmedian(f1s),
        n_successful=len(ok),
        mean_collapsed=float(np.mean([r.get("collapsed", 0) for r in rep_rows])),
        mean_n_balls=_nanmean([r.get("n_balls", float("nan")) for r in rep_rows]),
        mean_n_samples=_nanmean([r.get("n_samples_represented", float("nan")) for r in rep_rows]),
        skipped=False,
        skip_reason="",
    )


def select_k_max(summaries: dict[int, KValidationSummary]) -> tuple[int | None, str]:
    """Argmax mean validation macro F1; tie-break: lower collapse, then smaller K."""
    eligible = {
        k: s
        for k, s in summaries.items()
        if not s.skipped and np.isfinite(s.mean_macro_f1)
    }
    if not eligible:
        return None, "no_eligible_K"

    best_k = sorted(
        eligible.keys(),
        key=lambda k: (
            -eligible[k].mean_macro_f1,
            eligible[k].mean_collapsed,
            k,
        ),
    )[0]
    return int(best_k), ""


def select_k_1se(summaries: dict[int, KValidationSummary]) -> tuple[int | None, str, float]:
    """
    1-SE rule on compression repetitions (documented limitation: reps are not independent).

    SE = std(best) / sqrt(n_successful_reps); threshold = best_mean - SE;
    choose smallest K with mean >= threshold.
    """
    k_max, reason = select_k_max(summaries)
    if k_max is None:
        return None, reason, float("nan")

    best = summaries[k_max]
    if best.n_successful <= 1:
        se = float("nan")
        threshold = best.mean_macro_f1
    else:
        se = best.std_macro_f1 / np.sqrt(best.n_successful)
        threshold = best.mean_macro_f1 - se

    eligible = []
    for k in sorted(summaries.keys()):
        s = summaries[k]
        if s.skipped or not np.isfinite(s.mean_macro_f1):
            continue
        if s.mean_macro_f1 >= threshold - EPS:
            eligible.append(k)
    if not eligible:
        return k_max, "1se_fallback_to_max", float(threshold)
    return int(min(eligible)), "", float(threshold)

def compress_and_evaluate(
    balls: list[BallInfo],
    kept: list[GranularBall],
    features: list[dict[str, Any]],
    k: int,
    *,
    minority_label: float,
    rng_seed: int,
    y_train_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    indexed: np.ndarray,
    global_min: float,
    phase: Phase,
) -> dict[str, Any]:
    idx = random_compress_indices(balls, features, k, minority_label, rng_seed)
    sub_balls = [balls[i] for i in idx]
    sub_kept = [kept[i] for i in idx]
    ev = evaluate_representation_full(
        sub_balls,
        sub_kept,
        y_train=y_train_fit,
        X_eval=X_eval,
        y_eval=y_eval,
        minority_label=minority_label,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        phase=phase,
    )
    ev["K"] = k
    ev["compression_indices"] = idx
    return ev


def eval_full_rho90(
    split,
    *,
    seed: int,
    phase: Phase,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    balls, kept, _ = build_representation(split, seed, REPRESENTATION)
    return evaluate_representation_full(
        balls,
        kept,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        minority_label=split.minority_label,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        phase=phase,
    )


def eval_full_sample(
    split,
    *,
    seed: int,
    phase: Phase,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    balls = samples_to_ball_infos(split.X_train, y_train)
    return evaluate_representation_full(
        balls,
        None,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        minority_label=split.minority_label,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        phase=phase,
    )


def eval_sample_stratified_matched(
    split,
    n_target: int,
    *,
    seed: int,
    repetition: int,
    phase: Phase,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    dataset: str,
    benchmark_seed: int,
) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    rng_seed = sample_control_seed(
        dataset, benchmark_seed, n_target, repetition, method="sample_stratified_matched"
    )
    train_idx = select_training_sample_indices(
        y_train, n_target, split.minority_label, rng_seed, stratified=True
    )
    X_sub = split.X_train[train_idx]
    y_sub = y_train[train_idx]
    balls = samples_to_ball_infos(X_sub, y_sub)
    ev = evaluate_representation_full(
        balls,
        None,
        y_train=y_sub,
        X_eval=X_eval,
        y_eval=y_eval,
        minority_label=split.minority_label,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        phase=phase,
    )
    ev["n_train_samples_retained"] = int(len(train_idx))
    ev["n_target_samples"] = int(n_target)
    ev["repetition"] = repetition
    return ev
