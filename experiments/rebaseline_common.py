"""R1 factorial re-baseline helpers (rho90 arm)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.analyze_granular_geometry_vs_performance import (
    build_representation,
    fit_gaussian_gblstsvm,
)
from experiments.granular_compression_k_common import COLLAPSE_BALL_THRESHOLD
from experiments.phase8_common import DatasetSplit, DEFAULT_C1, DEFAULT_C2, normalize_labels
from experiments.validation_selected_k_common import REPRESENTATION, make_train_val_partition
from src.gblstsvm import predict_twin_hyperplanes

SCALE_DEFAULT = True
PREDICT_RULE_DEFAULT: Literal["raw", "norm"] = "raw"
TUNE_C_DEFAULT = False

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SEEDS = list(range(10))

PredictRule = Literal["raw", "norm"]


@dataclass(frozen=True)
class RebaselineFlags:
    scale: bool = SCALE_DEFAULT
    predict_rule: PredictRule = PREDICT_RULE_DEFAULT
    tune_c: bool = TUNE_C_DEFAULT


def scale_benchmark_split(split: DatasetSplit, scale: bool) -> DatasetSplit:
    """StandardScaler fit on benchmark train only; transform train/val/test."""
    if not scale:
        return split
    scaler = StandardScaler()
    X_train = np.asarray(split.X_train, dtype=float)
    X_val = np.asarray(split.X_val, dtype=float)
    X_test = np.asarray(split.X_test, dtype=float)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    X_train_s = scaler.fit_transform(X_train)
    assert scaler.n_samples_seen_ == X_train.shape[0]
    X_val_s = scaler.transform(X_val) if X_val.shape[0] > 0 else X_val
    X_test_s = scaler.transform(X_test)
    return replace(
        split,
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
    )


def failure_flags(
    n_balls: int,
    classifier_ok: bool,
    macro_f1: float,
) -> tuple[int, int]:
    """degenerate = solver failed OR non-finite F1; tiny = n_balls <= 3."""
    tiny = int(n_balls <= COLLAPSE_BALL_THRESHOLD)
    degenerate = int(not classifier_ok or not np.isfinite(macro_f1))
    return degenerate, tiny


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_n = normalize_labels(y_true)
    y_pred_n = y_pred.copy()
    y_pred_n[y_pred_n == 0.0] = -1.0
    return float(f1_score(y_true_n, y_pred_n, average="macro", zero_division=0))


def tune_c1_c2(
    balls: list,
    split: DatasetSplit,
    seed: int,
    *,
    prediction_rule: PredictRule = "raw",
) -> tuple[float, float]:
    """Grid-search (c1,c2) on internal validation macro F1; fit uses full train."""
    y_train = normalize_labels(split.y_train)
    partition = make_train_val_partition(split.name, seed, split.X_train, y_train)
    val_idx = partition.validation_indices
    X_val = split.X_train[val_idx]
    y_val = y_train[val_idx]

    best_c1, best_c2 = DEFAULT_C1, DEFAULT_C2
    best_f1 = -np.inf
    for c1 in C_GRID:
        for c2 in C_GRID:
            perf = fit_gaussian_gblstsvm(
                balls,
                y_train=y_train,
                X_test=X_val,
                y_test=y_val,
                minority_label=split.minority_label,
                seed=seed,
                c1=c1,
                c2=c2,
                prediction_rule=prediction_rule,
            )
            f1 = float(perf.get("macro_f1", float("nan")))
            if perf.get("classifier_ok") and np.isfinite(f1) and f1 > best_f1:
                best_f1 = f1
                best_c1, best_c2 = c1, c2
    return best_c1, best_c2


def tune_c1_c2_cv(
    balls: list,
    split: DatasetSplit,
    seed: int,
    *,
    prediction_rule: PredictRule = "norm",
    n_splits: int = 5,
) -> tuple[float, float]:
    """5-fold CV macro F1 on benchmark train; fit uses full train."""
    y_train = normalize_labels(split.y_train)
    X_train = split.X_train
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_c1, best_c2 = DEFAULT_C1, DEFAULT_C2
    best_mean = -np.inf
    for c1 in C_GRID:
        for c2 in C_GRID:
            fold_f1: list[float] = []
            for tr_idx, va_idx in kf.split(X_train):
                perf = fit_gaussian_gblstsvm(
                    balls,
                    y_train=y_train[tr_idx],
                    X_test=X_train[va_idx],
                    y_test=y_train[va_idx],
                    minority_label=split.minority_label,
                    seed=seed,
                    c1=c1,
                    c2=c2,
                    prediction_rule=prediction_rule,
                )
                f1 = float(perf.get("macro_f1", float("nan")))
                if perf.get("classifier_ok") and np.isfinite(f1):
                    fold_f1.append(f1)
            if fold_f1 and float(np.mean(fold_f1)) > best_mean:
                best_mean = float(np.mean(fold_f1))
                best_c1, best_c2 = c1, c2
    return best_c1, best_c2


def run_rho90_protocol(
    split: DatasetSplit,
    seed: int,
    *,
    scale: bool = True,
    predict_rule: PredictRule = "norm",
    tune_c: bool = True,
    tune_cv: bool = False,
) -> dict[str, Any]:
    """rho90 run with optional 5-fold CV c-tuning on full train."""
    scaled = scale_benchmark_split(split, scale)
    balls, _, _ = build_representation(scaled, seed, REPRESENTATION)
    c1, c2 = DEFAULT_C1, DEFAULT_C2
    if tune_c:
        if tune_cv:
            c1, c2 = tune_c1_c2_cv(balls, scaled, seed, prediction_rule=predict_rule)
        else:
            c1, c2 = tune_c1_c2(balls, scaled, seed, prediction_rule=predict_rule)
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=normalize_labels(scaled.y_train),
        X_test=scaled.X_test,
        y_test=scaled.y_test,
        minority_label=scaled.minority_label,
        seed=seed,
        c1=c1,
        c2=c2,
        prediction_rule=predict_rule,
    )
    return {
        "accuracy": float(perf.get("accuracy", float("nan"))),
        "macro_f1": float(perf.get("macro_f1", float("nan"))),
        "c1": float(c1),
        "c2": float(c2),
        "classifier_ok": bool(perf.get("classifier_ok", False)),
    }


def disagree_fraction(X: np.ndarray, w1, b1, w2, b2) -> float:
    pred_raw = predict_twin_hyperplanes(X, w1, b1, w2, b2, rule="raw")
    pred_norm = predict_twin_hyperplanes(X, w1, b1, w2, b2, rule="norm")
    pred_raw = pred_raw.copy()
    pred_norm = pred_norm.copy()
    pred_raw[pred_raw == 0.0] = -1.0
    pred_norm[pred_norm == 0.0] = -1.0
    if pred_raw.size == 0:
        return float("nan")
    return float(np.mean(pred_raw != pred_norm))


def run_rho90_cell(
    split: DatasetSplit,
    seed: int,
    flags: RebaselineFlags,
) -> dict[str, Any]:
    """Single rho90 run for one dataset×seed under given flags."""
    scaled = scale_benchmark_split(split, flags.scale)
    balls, kept, _ = build_representation(scaled, seed, REPRESENTATION)
    n_balls = len(balls)

    c1, c2 = DEFAULT_C1, DEFAULT_C2
    if flags.tune_c:
        c1, c2 = tune_c1_c2(balls, scaled, seed, prediction_rule=flags.predict_rule)

    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=normalize_labels(scaled.y_train),
        X_test=scaled.X_test,
        y_test=scaled.y_test,
        minority_label=scaled.minority_label,
        seed=seed,
        c1=c1,
        c2=c2,
        prediction_rule=flags.predict_rule,
    )

    macro_f1 = float(perf.get("macro_f1", float("nan")))
    accuracy = float(perf.get("accuracy", float("nan")))
    classifier_ok = bool(perf.get("classifier_ok", False))
    degenerate, tiny = failure_flags(n_balls, classifier_ok, macro_f1)

    w1 = perf.get("w1")
    w2 = perf.get("w2")
    b1 = perf.get("b1")
    b2 = perf.get("b2")
    w_ratio = float("nan")
    disagree = float("nan")
    if w1 is not None and w2 is not None:
        n1 = float(np.linalg.norm(w1))
        n2 = float(np.linalg.norm(w2))
        w_ratio = n1 / n2 if n2 > 0 else float("nan")
        if b1 is not None and b2 is not None:
            disagree = disagree_fraction(scaled.X_test, w1, b1, w2, b2)

    return {
        "dataset": split.name,
        "seed": int(seed),
        "scale": flags.scale,
        "predict_rule": flags.predict_rule,
        "tune_c": flags.tune_c,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "degenerate": degenerate,
        "tiny": tiny,
        "n_balls": n_balls,
        "c1": float(c1),
        "c2": float(c2),
        "w_norm_ratio": w_ratio,
        "disagree_frac": disagree,
    }


def all_flag_cells() -> list[RebaselineFlags]:
    cells: list[RebaselineFlags] = []
    for scale in (False, True):
        for predict_rule in ("raw", "norm"):
            for tune_c in (False, True):
                cells.append(RebaselineFlags(scale=scale, predict_rule=predict_rule, tune_c=tune_c))
    return cells
