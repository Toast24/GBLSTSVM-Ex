"""
Shared utilities for Phase 8 ablation experiments.

Does not modify frozen model implementations.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.cr_gblstsvm import CRGBLSTSVM
from src.cw_gblstsvm import CWGBLSTSVM
from src.gblstsvm import GBLSTSVM, solve_gblstsvm_systems
from src.r_gblstsvm import RGBLSTSVM
from src.reference_pipeline import (
    balls_to_training_matrix,
    reference_gen_balls,
    reference_predict_from_matrix,
)
from src.reliability import DEFAULT_EPSILON, DEFAULT_GAMMA, DEFAULT_LAMBDA_R

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data" / "Data"
OUT_DIR = Path(__file__).resolve().parent / "output" / "phase8"
DATASETS = ["diabetes.csv", "ecoli2.csv", "vehicle0.csv"]

DEFAULT_SEED = 42
DEFAULT_C1 = 100.0
DEFAULT_C2 = 1e-2
DEFAULT_PURITY = 0.95
DEFAULT_MIN_BALL_COUNT = 2
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"
RELIABILITY_VARIANT = "full"

CLEAN_MODELS = ["Clean GBLSTSVM", "CW-GBLSTSVM", "R-GBLSTSVM", "CR-GBLSTSVM"]
ALL_MODELS = ["Original/reference GBLSTSVM"] + CLEAN_MODELS

ScalingName = Literal["none", "standard", "minmax"]
METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "minority_recall",
    "gmean",
    "auroc",
)


@dataclass
class ExperimentConfig:
    scaling: ScalingName = "none"
    purity_threshold: float = DEFAULT_PURITY
    min_ball_count: int = DEFAULT_MIN_BALL_COUNT
    c1: float = DEFAULT_C1
    c2: float = DEFAULT_C2
    seed: int = DEFAULT_SEED
    prediction_rule: str = PREDICTION_RULE
    weight_placement: str = WEIGHT_PLACEMENT


@dataclass
class DatasetSplit:
    name: str
    info: dict
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    minority_label: float
    label_mapping: str


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_status_snapshot() -> dict:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        diff = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "status_porcelain": status.stdout.strip(),
            "diff_stat": diff.stdout.strip(),
            "clean": status.stdout.strip() == "",
        }
    except OSError:
        return {"error": "git not available"}


def inspect_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(path, header=None)
    n_cols = df.shape[1]
    if n_cols < 2:
        raise ValueError(f"Expected at least 2 columns; got {n_cols}.")

    feature_df = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]
    n_features = int(feature_df.shape[1])
    counts = y_raw.value_counts().sort_index().to_dict()
    counts = {int(k) if not isinstance(k, str) else k: int(v) for k, v in counts.items()}
    label_values = sorted(pd.unique(y_raw).tolist(), key=lambda v: (str(type(v)), v))

    info = {
        "name": path.stem,
        "path": str(path),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "n_samples": int(df.shape[0]),
        "n_features": n_features,
        "n_classes": len(label_values),
        "class_counts": counts,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "label_values": [int(v) if not isinstance(v, str) else v for v in label_values],
        "imbalance_ratio": float(max(counts.values()) / min(counts.values())),
        "duplicates_retained": True,
    }
    X = feature_df.to_numpy(dtype=float)
    y = y_raw.to_numpy(dtype=float)
    return X, y, info


def normalize_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel().copy()
    y[y == 0] = -1.0
    return y


def map_to_binary(y: np.ndarray, info: dict) -> tuple[np.ndarray, str]:
    y = np.asarray(y, dtype=float).ravel()
    if info["n_classes"] != 2:
        raise ValueError(f"Expected binary dataset; got {info['n_classes']} classes.")
    if set(info["label_values"]) <= {0, 1}:
        return normalize_labels(y), "CSV 0 -> -1, CSV 1 -> +1"
    counts = info["class_counts"]
    minority_raw = max(counts, key=counts.get)
    majority_raw = min(counts, key=counts.get)
    y_mapped = y.copy()
    y_mapped[y == minority_raw] = 1.0
    y_mapped[y == majority_raw] = -1.0
    return y_mapped, (
        f"raw {majority_raw} -> -1, raw {minority_raw} -> +1"
    )


def stratified_three_way_split(X, y, seed: int):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_dataset_split(name: str, seed: int = DEFAULT_SEED) -> DatasetSplit:
    path = DATA_DIR / name
    if not path.name.endswith(".csv"):
        path = DATA_DIR / f"{name}.csv"
    X, y_raw, info = inspect_dataset(path)
    y, label_mapping = map_to_binary(y_raw, info)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(X, y, seed)
    y_train_n = normalize_labels(y_train)
    n_pos = int(np.sum(y_train_n == 1.0))
    n_neg = int(np.sum(y_train_n == -1.0))
    minority_label = 1.0 if n_pos <= n_neg else -1.0
    return DatasetSplit(
        name=info["name"],
        info=info,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        minority_label=minority_label,
        label_mapping=label_mapping,
    )


def distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return {}
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": mean,
        "median": float(np.median(values)),
        "std": std,
    }


def minority_class(y: np.ndarray) -> float:
    y = normalize_labels(y)
    return 1.0 if int(np.sum(y == 1.0)) <= int(np.sum(y == -1.0)) else -1.0


def twin_scores(model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    f1 = X @ model.w1_ + model.b1_
    f2 = X @ model.w2_ + model.b2_
    return np.abs(f2) - np.abs(f1)


def original_twin_scores(train_matrix: np.ndarray, X: np.ndarray, c1: float, c2: float):
    C1 = train_matrix[train_matrix[:, -1] == 1, :-1]
    C2 = train_matrix[train_matrix[:, -1] != 1, :-1]
    A = C1[:, :-1]
    B = C2[:, :-1]
    R_pos = C1[:, -1]
    R_neg = C2[:, -1]
    w1, b1, w2, b2 = solve_gblstsvm_systems(A, R_pos, B, R_neg, c1, c2)
    X = np.asarray(X, dtype=float)
    scores = np.abs(X @ w2 + b2) - np.abs(X @ w1 + b1)
    return scores, w1, b1, w2, b2


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    minority_label: float,
) -> dict:
    y_true = normalize_labels(y_true)
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_pred_eval = y_pred.copy()
    y_pred_eval[y_pred_eval == 0.0] = -1.0

    rec_pos = float(recall_score(y_true, y_pred_eval, pos_label=1.0, zero_division=0))
    rec_neg = float(recall_score(y_true, y_pred_eval, pos_label=-1.0, zero_division=0))
    gmean = float(np.sqrt(rec_pos * rec_neg))

    cm = confusion_matrix(y_true, y_pred_eval, labels=[-1.0, 1.0])
    auroc = None
    if scores is not None:
        try:
            auroc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auroc = None

    unique = sorted(np.unique(y_pred_eval).tolist())
    collapsed = len(unique) == 1

    return {
        "accuracy": float(accuracy_score(y_true, y_pred_eval)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_eval)),
        "macro_f1": float(f1_score(y_true, y_pred_eval, average="macro", zero_division=0)),
        "minority_recall": float(
            recall_score(y_true, y_pred_eval, pos_label=minority_label, zero_division=0)
        ),
        "gmean": gmean,
        "auroc": auroc,
        "confusion_matrix": cm.astype(int).tolist(),
        "prediction_collapse": collapsed,
        "unique_predicted_labels": unique,
    }


def delta_metrics(a: dict, b: dict, keys: tuple[str, ...] = METRIC_KEYS) -> dict:
    out = {}
    for k in keys:
        av, bv = a.get(k), b.get(k)
        if av is None or bv is None:
            out[k] = None
        else:
            out[k] = float(av - bv)
    return out


def fit_scaler(
    scaling: ScalingName,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
    if scaling == "none":
        return X_train, X_val, X_test, None
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(scaling)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def fit_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ExperimentConfig,
) -> Any:
    base = {
        "c1": config.c1,
        "c2": config.c2,
        "purity_threshold": config.purity_threshold,
        "min_ball_count": config.min_ball_count,
        "prediction_rule": config.prediction_rule,
        "random_state": config.seed,
    }
    r_extra = {
        "epsilon": DEFAULT_EPSILON,
        "gamma": DEFAULT_GAMMA,
        "lambda_r": DEFAULT_LAMBDA_R,
        "reliability_variant": RELIABILITY_VARIANT,
        "weight_placement": config.weight_placement,
    }
    if name == "Clean GBLSTSVM":
        model = GBLSTSVM(**base)
    elif name == "CW-GBLSTSVM":
        model = CWGBLSTSVM(**base, weight_placement=config.weight_placement)
    elif name == "R-GBLSTSVM":
        model = RGBLSTSVM(**base, **r_extra)
    elif name == "CR-GBLSTSVM":
        model = CRGBLSTSVM(**base, **r_extra)
    else:
        raise ValueError(name)
    model.fit(X_train, y_train)
    return model


class OriginalReferenceGBLSTSVM:
    def __init__(
        self,
        c1: float,
        c2: float,
        purity_threshold: float,
        min_ball_count: int,
        seed: int,
    ):
        self.c1 = c1
        self.c2 = c2
        self.purity_threshold = purity_threshold
        self.min_ball_count = min_ball_count
        self.seed = seed
        self.balls_ = None
        self.train_matrix_ = None
        self.w1_ = None
        self.b1_ = None
        self.w2_ = None
        self.b2_ = None
        self.supported = False
        self.skip_reason = None

    def fit(self, X, y):
        y = normalize_labels(y)
        np.random.seed(self.seed)
        try:
            self.balls_ = reference_gen_balls(
                np.asarray(X, dtype=float),
                y,
                self.purity_threshold,
                self.min_ball_count,
            )
            n_pos = sum(1 for b in self.balls_ if b.label == 1)
            n_neg = sum(1 for b in self.balls_ if b.label != 1)
            if n_pos == 0 or n_neg == 0:
                self.skip_reason = f"Empty ball class pos={n_pos} neg={n_neg}"
                return self
            self.train_matrix_ = balls_to_training_matrix(self.balls_)
            _, w1, b1, w2, b2 = original_twin_scores(
                self.train_matrix_, np.asarray(X, dtype=float)[:1], self.c1, self.c2
            )
            self.w1_, self.b1_, self.w2_, self.b2_ = w1, b1, w2, b2
            self.supported = True
        except Exception as exc:  # noqa: BLE001
            self.skip_reason = f"{type(exc).__name__}: {exc}"
        return self

    def predict(self, X):
        if not self.supported:
            raise RuntimeError(self.skip_reason or "not fitted")
        return reference_predict_from_matrix(
            self.train_matrix_, np.asarray(X, dtype=float), self.c1, self.c2
        )

    def decision_scores(self, X):
        scores, _, _, _, _ = original_twin_scores(
            self.train_matrix_, np.asarray(X, dtype=float), self.c1, self.c2
        )
        return scores


def fit_original(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ExperimentConfig,
) -> OriginalReferenceGBLSTSVM:
    model = OriginalReferenceGBLSTSVM(
        config.c1, config.c2, config.purity_threshold, config.min_ball_count, config.seed
    )
    model.fit(X_train, y_train)
    return model


def predict_and_evaluate(
    model: Any,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    minority_label: float,
) -> tuple[dict, np.ndarray, Optional[np.ndarray]]:
    if name == "Original/reference GBLSTSVM":
        if not model.supported:
            raise RuntimeError(model.skip_reason)
        y_pred = model.predict(X)
        scores = model.decision_scores(X)
    else:
        y_pred = model.predict(X)
        scores = twin_scores(model, X)
    metrics = evaluate(y, y_pred, scores, minority_label)
    return metrics, np.asarray(y_pred), scores


def ball_stats(balls) -> dict:
    purities = np.array([float(b.purity) for b in balls], dtype=float)
    radii = np.array([float(b.radius) for b in balls], dtype=float)
    n_pos = sum(1 for b in balls if b.label == 1)
    n_neg = sum(1 for b in balls if b.label != 1)
    samples = (
        int(sum(int(b.num_samples) for b in balls))
        if balls and all(hasattr(b, "num_samples") for b in balls)
        else None
    )
    return {
        "n_balls_total": len(balls),
        "n_balls_pos": n_pos,
        "n_balls_neg": n_neg,
        "training_samples_in_balls": samples,
        "purity_min": float(np.min(purities)) if purities.size else None,
        "purity_mean": float(np.mean(purities)) if purities.size else None,
        "purity_std": float(np.std(purities)) if purities.size else None,
        "purity_max": float(np.max(purities)) if purities.size else None,
        "radius_min": float(np.min(radii)) if radii.size else None,
        "radius_mean": float(np.mean(radii)) if radii.size else None,
        "radius_max": float(np.max(radii)) if radii.size else None,
    }


def reliability_stats(reliabilities: np.ndarray) -> dict:
    rel = np.asarray(reliabilities, dtype=float).ravel()
    if rel.size == 0:
        return {}
    return {
        "reliability_min": float(np.min(rel)),
        "reliability_mean": float(np.mean(rel)),
        "reliability_std": float(np.std(rel)),
        "reliability_max": float(np.max(rel)),
        "frac_reliability_ge_0_99": float(np.mean(rel >= 0.99)),
        "frac_reliability_lt_0_95": float(np.mean(rel < 0.95)),
        "frac_reliability_ge_0_95": float(np.mean(rel >= 0.95)),
        "frac_reliability_lt_0_90": float(np.mean(rel < 0.90)),
    }


def cr_cw_weight_stats(cr_model: CRGBLSTSVM) -> dict:
    final = np.concatenate([cr_model.weights_pos_, cr_model.weights_neg_])
    cw_like = np.concatenate([cr_model.kappa_pos_balls_, cr_model.kappa_neg_balls_])
    diff = np.abs(final - cw_like)
    return {
        "frac_cr_weights_differ_from_cw": float(np.mean(diff > 1e-12)),
        "mean_abs_cr_minus_cw_weight": float(np.mean(diff)),
        "max_abs_cr_minus_cw_weight": float(np.max(diff)),
    }


def model_ball_and_reliability_stats(name: str, model: Any) -> dict:
    stats = ball_stats(model.balls_)
    if hasattr(model, "reliabilities_") and model.reliabilities_ is not None:
        stats.update(reliability_stats(model.reliabilities_))
    if name == "CR-GBLSTSVM":
        stats.update(cr_cw_weight_stats(model))
    if name in ("CW-GBLSTSVM", "CR-GBLSTSVM"):
        stats["kappa_positive"] = float(model.kappa_pos_)
        stats["kappa_negative"] = float(model.kappa_neg_)
    return stats


def run_clean_family(
    split: DatasetSplit,
    config: ExperimentConfig,
    models: list[str] = CLEAN_MODELS,
    eval_splits: tuple[str, ...] = ("val", "test"),
) -> dict:
    X_train, X_val, X_test = split.X_train, split.X_val, split.X_test
    if config.scaling != "none":
        X_train, X_val, X_test, _ = fit_scaler(
            config.scaling, X_train, X_val, X_test
        )
    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name in models:
        try:
            model = fit_model(name, X_train, split.y_train, config)
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        fitted[name] = model
        entry = {"ball_stats": model_ball_and_reliability_stats(name, model)}
        for sp in eval_splits:
            X = X_val if sp == "val" else X_test
            y = split.y_val if sp == "val" else split.y_test
            m, pred, _ = predict_and_evaluate(model, name, X, y, split.minority_label)
            entry[sp] = m
            entry[f"{sp}_predictions"] = pred.tolist()
        results[name] = entry
    return {"models": results, "fitted": fitted}


def mechanism_diagnostics(
    cw_model: CWGBLSTSVM,
    cr_model: CRGBLSTSVM,
    y_val: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    minority_label: float,
) -> dict:
    cw_weights = np.concatenate([cw_model.weights_pos_, cw_model.weights_neg_])
    cr_weights = np.concatenate([cr_model.weights_pos_, cr_model.weights_neg_])
    weight_diff = np.abs(cr_weights - cw_weights)

    val_pred_cw = cw_model.predict(X_val)
    val_pred_cr = cr_model.predict(X_val)
    test_pred_cw = cw_model.predict(X_test)
    test_pred_cr = cr_model.predict(X_test)

    rel = cr_model.reliabilities_
    rel_stats = reliability_stats(rel) if rel is not None else {}

    return {
        **rel_stats,
        "frac_cr_weights_differ_from_cw": float(np.mean(weight_diff > 1e-12)),
        "mean_abs_cr_minus_cw_weight": float(np.mean(weight_diff)),
        "max_abs_cr_minus_cw_weight": float(np.max(weight_diff)),
        "w1_norm_diff": float(np.linalg.norm(cr_model.w1_ - cw_model.w1_)),
        "w2_norm_diff": float(np.linalg.norm(cr_model.w2_ - cw_model.w2_)),
        "b1_abs_diff": float(abs(cr_model.b1_ - cw_model.b1_)),
        "b2_abs_diff": float(abs(cr_model.b2_ - cw_model.b2_)),
        "val_frac_pred_cr_neq_cw": float(np.mean(val_pred_cr != val_pred_cw)),
        "test_frac_pred_cr_neq_cw": float(np.mean(test_pred_cr != test_pred_cw)),
        "val_metrics_cw": evaluate(y_val, val_pred_cw, twin_scores(cw_model, X_val), minority_label),
        "val_metrics_cr": evaluate(y_val, val_pred_cr, twin_scores(cr_model, X_val), minority_label),
        "test_metrics_cw": evaluate(y_test, test_pred_cw, twin_scores(cw_model, X_test), minority_label),
        "test_metrics_cr": evaluate(y_test, test_pred_cr, twin_scores(cr_model, X_test), minority_label),
    }


def config_dict(config: ExperimentConfig) -> dict:
    return {
        "scaling": config.scaling,
        "purity_threshold": config.purity_threshold,
        "min_ball_count": config.min_ball_count,
        "c1": config.c1,
        "c2": config.c2,
        "seed": config.seed,
        "prediction_rule": config.prediction_rule,
        "weight_placement": config.weight_placement,
    }
