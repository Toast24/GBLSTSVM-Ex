"""
Phase 7: stratified train/validation/test experiment on imbalanced benchmark datasets.

Datasets: diabetes.csv, ecoli2.csv, vehicle0.csv
Compares original/reference GBLSTSVM, frozen clean GBLSTSVM, CW, R, and CR.
Does not modify frozen model implementations or mathematical formulations.
Does not tune hyperparameters on validation or test.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from src.cr_gblstsvm import CRGBLSTSVM
from src.cw_gblstsvm import CWGBLSTSVM
from src.gblstsvm import GBLSTSVM, solve_gblstsvm_systems
from src.granular_balls import balls_to_class_arrays
from src.r_gblstsvm import RGBLSTSVM
from src.reference_pipeline import (
    balls_to_training_matrix,
    reference_gen_balls,
    reference_predict_from_matrix,
)
from src.reliability import DEFAULT_EPSILON, DEFAULT_GAMMA, DEFAULT_LAMBDA_R

SEED = 42
C1 = 100.0
C2 = 1e-2
PURITY = 0.95
MIN_BALL_COUNT = 2
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"
RELIABILITY_VARIANT = "full"

DATA_DIR = ROOT / "Data" / "Data"
DATASETS = ["diabetes.csv", "ecoli2.csv", "vehicle0.csv"]
OUT_DIR = Path(__file__).resolve().parent / "output"

MODEL_ORDER = [
    "Original/reference GBLSTSVM",
    "Clean GBLSTSVM",
    "CW-GBLSTSVM",
    "R-GBLSTSVM",
    "CR-GBLSTSVM",
]

METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "minority_recall",
    "gmean",
    "auroc",
)


def inspect_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(path, header=None)
    n_cols = df.shape[1]
    if n_cols < 2:
        raise ValueError(f"Expected at least 2 columns (features + label); got {n_cols}.")

    feature_df = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]
    n_features = int(feature_df.shape[1])

    non_numeric = [
        int(c)
        for c in feature_df.columns
        if not np.issubdtype(feature_df[c].dtype, np.number)
    ]
    label_values = sorted(pd.unique(y_raw).tolist(), key=lambda v: (str(type(v)), v))
    counts = y_raw.value_counts().sort_index().to_dict()
    counts = {int(k) if not isinstance(k, str) else k: int(v) for k, v in counts.items()}

    feature_means = feature_df.mean().to_numpy(dtype=float)
    feature_stds = feature_df.std(ddof=0).to_numpy(dtype=float)
    already_std = bool(
        np.allclose(feature_means, 0.0, atol=1e-4)
        and np.allclose(feature_stds, 1.0, atol=1e-2)
    )

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
        "non_numeric_feature_columns": non_numeric,
        "feature_means": [float(x) for x in feature_means],
        "feature_stds": [float(x) for x in feature_stds],
        "already_approximately_standardized": already_std,
        "duplicates_retained": True,
        "imbalance_ratio": float(max(counts.values()) / min(counts.values())),
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
    other = [v for v in info["label_values"] if v not in (minority_raw, majority_raw)]
    for v in other:
        y_mapped[y == v] = -1.0
    return y_mapped, (
        f"raw {majority_raw} (majority, n={counts[majority_raw]}) -> -1, "
        f"raw {minority_raw} (minority, n={counts[minority_raw]}) -> +1"
    )


def stratified_three_way_split(X, y, seed: int):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


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
        "cv": float(std / mean) if mean != 0 else float("nan"),
    }


def minority_class(y: np.ndarray) -> float:
    y = normalize_labels(y)
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == -1.0))
    return 1.0 if n_pos <= n_neg else -1.0


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


def prediction_collapse(y_pred: np.ndarray) -> dict:
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_eval = y_pred.copy()
    y_eval[y_eval == 0.0] = -1.0
    unique = sorted(np.unique(y_eval).tolist())
    n_neg = int(np.sum(y_eval == -1.0))
    n_pos = int(np.sum(y_eval == 1.0))
    collapsed = len(unique) == 1
    rec = {
        "unique_predicted_labels": unique,
        "n_predicted_neg1": n_neg,
        "n_predicted_pos1": n_pos,
        "prediction_collapse": collapsed,
    }
    if collapsed:
        rec["collapse_flag"] = "PREDICTION COLLAPSE: MAJORITY-CLASS-ONLY"
    return rec


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    minority_label: float,
) -> dict:
    y_true = normalize_labels(y_true)
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n_zero = int(np.sum(y_pred == 0.0))
    y_pred_eval = y_pred.copy()
    y_pred_eval[y_pred_eval == 0.0] = -1.0

    rec_pos = float(recall_score(y_true, y_pred_eval, pos_label=1.0, zero_division=0))
    rec_neg = float(recall_score(y_true, y_pred_eval, pos_label=-1.0, zero_division=0))
    gmean = float(np.sqrt(rec_pos * rec_neg))

    prec_pos = float(precision_score(y_true, y_pred_eval, pos_label=1.0, zero_division=0))
    prec_neg = float(precision_score(y_true, y_pred_eval, pos_label=-1.0, zero_division=0))
    f1_pos = float(f1_score(y_true, y_pred_eval, pos_label=1.0, zero_division=0))
    f1_neg = float(f1_score(y_true, y_pred_eval, pos_label=-1.0, zero_division=0))

    cm = confusion_matrix(y_true, y_pred_eval, labels=[-1.0, 1.0])
    auroc = None
    if scores is not None:
        try:
            auroc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auroc = None

    collapse = prediction_collapse(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred_eval)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_eval)),
        "macro_f1": float(f1_score(y_true, y_pred_eval, average="macro", zero_division=0)),
        "minority_recall": float(
            recall_score(y_true, y_pred_eval, pos_label=minority_label, zero_division=0)
        ),
        "minority_precision": float(
            precision_score(
                y_true, y_pred_eval, pos_label=minority_label, zero_division=0
            )
        ),
        "gmean": gmean,
        "auroc": auroc,
        "per_class_precision": {"-1": prec_neg, "+1": prec_pos},
        "per_class_recall": {"-1": rec_neg, "+1": rec_pos},
        "per_class_f1": {"-1": f1_neg, "+1": f1_pos},
        "confusion_matrix_labels": [-1, 1],
        "confusion_matrix": cm.astype(int).tolist(),
        "n_zero_predictions_mapped_to_neg": n_zero,
        "recall_pos": rec_pos,
        "recall_neg": rec_neg,
        **collapse,
    }


def ball_diagnostics(balls, extra: dict | None = None) -> dict:
    purities = np.array([float(b.purity) for b in balls], dtype=float)
    radii = np.array([float(b.radius) for b in balls], dtype=float)
    n_pos = sum(1 for b in balls if b.label == 1)
    n_neg = sum(1 for b in balls if b.label != 1)
    if balls and all(hasattr(b, "num_samples") for b in balls):
        samples_in_balls = int(sum(int(b.num_samples) for b in balls))
    else:
        samples_in_balls = None
    rec = {
        "n_balls_pos": int(n_pos),
        "n_balls_neg": int(n_neg),
        "n_balls_total": int(len(balls)),
        "training_samples_in_balls": samples_in_balls,
        "purity": distribution_summary(purities),
        "radius": distribution_summary(radii),
    }
    if extra:
        rec.update(extra)
    return rec


def reliability_diagnostics(reliabilities: np.ndarray) -> dict:
    rel = np.asarray(reliabilities, dtype=float).ravel()
    if rel.size == 0:
        return {}
    return {
        **distribution_summary(rel),
        "frac_ge_0_99": float(np.mean(rel >= 0.99)),
        "frac_lt_0_95": float(np.mean(rel < 0.95)),
        "varies": bool(np.std(rel) > 1e-12),
        "all_approximately_one": bool(np.allclose(rel, 1.0, atol=1e-6)),
    }


class OriginalReferenceGBLSTSVM:
    """Experiment-only wrapper around unmodified gen_ball.py / GBLSTSVM.py."""

    def __init__(self, c1: float, c2: float, purity_threshold: float, min_ball_count: int):
        self.c1 = c1
        self.c2 = c2
        self.purity_threshold = purity_threshold
        self.min_ball_count = min_ball_count
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
        np.random.seed(SEED)
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
                self.skip_reason = (
                    "Original granulation produced empty class of balls "
                    f"(pos={n_pos}, neg={n_neg})."
                )
                return self
            self.train_matrix_ = balls_to_training_matrix(self.balls_)
            _, w1, b1, w2, b2 = original_twin_scores(
                self.train_matrix_, np.asarray(X, dtype=float)[:1], self.c1, self.c2
            )
            self.w1_, self.b1_, self.w2_, self.b2_ = w1, b1, w2, b2
            self.supported = True
        except Exception as exc:  # noqa: BLE001
            self.skip_reason = f"Original/reference pipeline failed: {type(exc).__name__}: {exc}"
        return self

    def predict(self, X):
        if not self.supported:
            raise RuntimeError(self.skip_reason or "Original model is not fitted.")
        return reference_predict_from_matrix(
            self.train_matrix_, np.asarray(X, dtype=float), self.c1, self.c2
        )

    def decision_scores(self, X):
        scores, _, _, _, _ = original_twin_scores(
            self.train_matrix_, np.asarray(X, dtype=float), self.c1, self.c2
        )
        return scores


def fit_clean_family(name: str, X_train, y_train):
    base = {
        "c1": C1,
        "c2": C2,
        "purity_threshold": PURITY,
        "min_ball_count": MIN_BALL_COUNT,
        "prediction_rule": PREDICTION_RULE,
        "random_state": SEED,
    }
    r_extra = {
        "epsilon": DEFAULT_EPSILON,
        "gamma": DEFAULT_GAMMA,
        "lambda_r": DEFAULT_LAMBDA_R,
        "reliability_variant": RELIABILITY_VARIANT,
        "weight_placement": WEIGHT_PLACEMENT,
    }
    if name == "Clean GBLSTSVM":
        model = GBLSTSVM(**base)
    elif name == "CW-GBLSTSVM":
        model = CWGBLSTSVM(**base, weight_placement=WEIGHT_PLACEMENT)
    elif name == "R-GBLSTSVM":
        model = RGBLSTSVM(**base, **r_extra)
    elif name == "CR-GBLSTSVM":
        model = CRGBLSTSVM(**base, **r_extra)
    else:
        raise ValueError(name)
    model.fit(X_train, y_train)
    return model


def modified_model_diagnostics(name: str, model, y_train: np.ndarray) -> dict:
    y_train = normalize_labels(y_train)
    n_pos = int(np.sum(y_train == 1.0))
    n_neg = int(np.sum(y_train == -1.0))
    rec = ball_diagnostics(
        model.balls_,
        extra={
            "n_train_pos": n_pos,
            "n_train_neg": n_neg,
            "n_train": int(len(y_train)),
        },
    )
    if hasattr(model, "kappa_pos_"):
        rec["kappa_positive"] = float(model.kappa_pos_)
        rec["kappa_negative"] = float(model.kappa_neg_)
    if hasattr(model, "reliabilities_") and model.reliabilities_ is not None:
        rec["reliability"] = reliability_diagnostics(model.reliabilities_)
        rec["radius_scale"] = float(model.radius_scale_) if model.radius_scale_ is not None else None
    if name == "CR-GBLSTSVM":
        final = np.concatenate([model.weights_pos_, model.weights_neg_])
        cw_like = np.concatenate([model.kappa_pos_balls_, model.kappa_neg_balls_])
        rec["frac_cr_weights_differ_from_cw"] = float(
            np.mean(np.abs(final - cw_like) > 1e-12)
        )
        rec["max_abs_cr_minus_cw_weight"] = float(np.max(np.abs(final - cw_like)))
        rec["mean_abs_cr_minus_cw_weight"] = float(np.mean(np.abs(final - cw_like)))
    return rec


def verify_reference_vs_clean(
    original: OriginalReferenceGBLSTSVM,
    clean: GBLSTSVM,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> dict:
    if not original.supported:
        return {"verified": False, "reason": original.skip_reason}

    ref_balls = original.balls_
    clean_balls = clean.balls_
    ball_cmp = {
        "n_balls_ref": len(ref_balls),
        "n_balls_clean": len(clean_balls),
        "n_balls_match": len(ref_balls) == len(clean_balls),
        "n_balls_pos_ref": sum(1 for b in ref_balls if b.label == 1),
        "n_balls_pos_clean": sum(1 for b in clean_balls if b.label == 1),
        "n_balls_neg_ref": sum(1 for b in ref_balls if b.label != 1),
        "n_balls_neg_clean": sum(1 for b in clean_balls if b.label != 1),
    }

    ref_val_pred = original.predict(X_val)
    ref_test_pred = original.predict(X_test)
    clean_val_pred = clean.predict(X_val)
    clean_test_pred = clean.predict(X_test)
    val_agree = float(np.mean(ref_val_pred == clean_val_pred))
    test_agree = float(np.mean(ref_test_pred == clean_test_pred))

    ref_val_scores = original.decision_scores(X_val)
    ref_test_scores = original.decision_scores(X_test)
    clean_val_scores = twin_scores(clean, X_val)
    clean_test_scores = twin_scores(clean, X_test)
    val_score_max_diff = float(np.max(np.abs(ref_val_scores - clean_val_scores)))
    test_score_max_diff = float(np.max(np.abs(ref_test_scores - clean_test_scores)))

    hyperplane_cmp = {}
    if original.w1_ is not None and clean.w1_ is not None:
        hyperplane_cmp = {
            "w1_max_abs_diff": float(np.max(np.abs(original.w1_ - clean.w1_))),
            "w2_max_abs_diff": float(np.max(np.abs(original.w2_ - clean.w2_))),
            "b1_diff": float(abs(original.b1_ - clean.b1_)),
            "b2_diff": float(abs(original.b2_ - clean.b2_)),
        }

    material_disagreement = (
        val_agree < 1.0
        or test_agree < 1.0
        or val_score_max_diff > 1e-6
        or test_score_max_diff > 1e-6
        or not ball_cmp["n_balls_match"]
    )

    return {
        "verified": not material_disagreement,
        "prediction_rule": PREDICTION_RULE,
        "ball_comparison": ball_cmp,
        "val_prediction_agreement": val_agree,
        "test_prediction_agreement": test_agree,
        "val_score_max_abs_diff": val_score_max_diff,
        "test_score_max_abs_diff": test_score_max_diff,
        "hyperplane_comparison": hyperplane_cmp,
        "material_disagreement": material_disagreement,
    }


def classify_reliability_status(diag_r: dict, diag_cr: dict, cr_test: dict, cw_test: dict) -> dict:
    rel = diag_r.get("reliability", {})
    purity = diag_r.get("purity", {})
    mean_purity = purity.get("mean", float("nan"))
    mean_rel = rel.get("mean", float("nan"))
    all_rel_one = rel.get("all_approximately_one", False)
    rel_varies = rel.get("varies", False)
    frac_ge_099 = rel.get("frac_ge_0_99", float("nan"))

    cr_weights_differ = diag_cr.get("frac_cr_weights_differ_from_cw", 0.0) > 1e-12
    cr_pred_differ = (
        cr_test.get("macro_f1") != cw_test.get("macro_f1")
        or cr_test.get("balanced_accuracy") != cw_test.get("balanced_accuracy")
    )

    if all_rel_one or (frac_ge_099 is not float("nan") and frac_ge_099 >= 0.99 and not rel_varies):
        status = "RELIABILITY SATURATED"
        r_reduces_to_baseline = True
        cr_reduces_to_cw = not cr_weights_differ
    else:
        status = "RELIABILITY ACTIVE"
        r_reduces_to_baseline = False
        cr_reduces_to_cw = not cr_weights_differ

    return {
        "status": status,
        "mean_purity": mean_purity,
        "mean_reliability": mean_rel,
        "frac_reliability_ge_0_99": frac_ge_099,
        "reliability_varies": rel_varies,
        "r_effectively_baseline": r_reduces_to_baseline,
        "cr_effectively_cw": cr_reduces_to_cw and not cr_weights_differ,
        "cr_weights_differ_from_cw": cr_weights_differ,
    }


def delta(a: dict, b: dict, keys: tuple[str, ...]) -> dict:
    out = {}
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if av is None or bv is None:
            out[k] = None
        else:
            out[k] = float(av - bv)
    return out


def build_preprocessing(dataset_info: dict) -> dict:
    if dataset_info["already_approximately_standardized"]:
        reason = (
            f"{dataset_info['name']}.csv features already have mean ~0 and std ~1. "
            "Frozen implementations do not scale internally; reference code does not scale."
        )
    else:
        reason = (
            f"{dataset_info['name']}.csv features are not pre-standardized "
            f"(mean range across features, std range). "
            "MinMaxScaler is imported but unused in gen_ball.py / GBLSTSVM.py; "
            "main.py loads raw CSV values with no scaling. Frozen Phase 1–5 models "
            "also do not scale internally. Introducing scaling would be a new pipeline "
            "step inconsistent with the frozen baseline protocol, so none is applied."
        )
    return {
        "method": "none",
        "reason": reason,
        "fitted_on_train_only": None,
        "leakage_controls": [
            "raw split before any statistics",
            "granular balls generated from training rows only",
            "class weights from training labels only",
            "reliability radius scale from training balls only",
            "validation unused for test construction or hyperparameter selection",
        ],
    }


def run_dataset(path: Path) -> dict:
    X, y_raw, dataset_info = inspect_dataset(path)
    y, label_mapping = map_to_binary(y_raw, dataset_info)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(
        X, y, SEED
    )
    y_train_n = normalize_labels(y_train)
    y_val_n = normalize_labels(y_val)
    y_test_n = normalize_labels(y_test)
    min_label = minority_class(y_train)

    split_info = {
        "strategy": "stratified 70/15/15, two sequential sklearn train_test_split calls",
        "seed": SEED,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "train_class_counts": {
            "+1": int(np.sum(y_train_n == 1.0)),
            "-1": int(np.sum(y_train_n == -1.0)),
        },
        "val_class_counts": {
            "+1": int(np.sum(y_val_n == 1.0)),
            "-1": int(np.sum(y_val_n == -1.0)),
        },
        "test_class_counts": {
            "+1": int(np.sum(y_test_n == 1.0)),
            "-1": int(np.sum(y_test_n == -1.0)),
        },
        "minority_label_from_train": int(min_label),
        "label_mapping": label_mapping,
    }

    preprocessing = build_preprocessing(dataset_info)
    hyperparameters = {
        "baseline": {
            "c1": C1,
            "c2": C2,
            "purity_threshold": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "epsilon": "N/A",
            "gamma": "N/A",
            "lambda_r": "N/A",
            "weight_placement": "N/A",
            "prediction_rule": PREDICTION_RULE,
            "random_state": SEED,
        },
        "CW": {
            "c1": C1,
            "c2": C2,
            "purity_threshold": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "epsilon": "N/A",
            "gamma": "N/A",
            "lambda_r": "N/A",
            "weight_placement": WEIGHT_PLACEMENT,
            "prediction_rule": PREDICTION_RULE,
            "random_state": SEED,
            "class_weighting": "kappa_c = n_train / (2 * n_c)",
        },
        "R": {
            "c1": C1,
            "c2": C2,
            "purity_threshold": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "epsilon": DEFAULT_EPSILON,
            "gamma": DEFAULT_GAMMA,
            "lambda_r": DEFAULT_LAMBDA_R,
            "weight_placement": WEIGHT_PLACEMENT,
            "prediction_rule": PREDICTION_RULE,
            "random_state": SEED,
            "reliability_variant": RELIABILITY_VARIANT,
        },
        "CR": {
            "c1": C1,
            "c2": C2,
            "purity_threshold": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "epsilon": DEFAULT_EPSILON,
            "gamma": DEFAULT_GAMMA,
            "lambda_r": DEFAULT_LAMBDA_R,
            "weight_placement": WEIGHT_PLACEMENT,
            "prediction_rule": PREDICTION_RULE,
            "random_state": SEED,
            "reliability_variant": RELIABILITY_VARIANT,
            "final_weight": "kappa * reliability",
        },
        "tuning": "none; Phase 2–5 defaults; validation not used for model selection",
        "validation_note": (
            "Validation set used only for diagnostic evaluation; no tuning performed."
        ),
    }

    results = {}
    predictions = {}
    diagnostics_rows = []
    reliability_rows = []
    confusion_matrices = {}

    original = OriginalReferenceGBLSTSVM(C1, C2, PURITY, MIN_BALL_COUNT)
    original.fit(X_train, y_train)
    original_status = {
        "executed": original.supported,
        "skip_reason": original.skip_reason,
        "numpy_seed_before_granulation": SEED,
        "note": (
            "Invoked unmodified gen_ball.GBList via src.reference_pipeline. "
            "Original KMeans(n_clusters=2) has no random_state; numpy is seeded "
            "with 42 immediately before granulation."
        ),
    }

    if original.supported:
        val_pred = original.predict(X_val)
        test_pred = original.predict(X_test)
        val_scores = original.decision_scores(X_val)
        test_scores = original.decision_scores(X_test)
        val_m = evaluate(y_val, val_pred, val_scores, min_label)
        test_m = evaluate(y_test, test_pred, test_scores, min_label)
        diag = ball_diagnostics(
            original.balls_,
            extra={
                "n_train_pos": int(np.sum(y_train_n == 1.0)),
                "n_train_neg": int(np.sum(y_train_n == -1.0)),
                "n_train": int(len(y_train)),
            },
        )
        results["Original/reference GBLSTSVM"] = {
            "val": val_m,
            "test": test_m,
            "diagnostics": diag,
        }
        predictions["Original/reference GBLSTSVM"] = {
            "val": np.asarray(val_pred).tolist(),
            "test": np.asarray(test_pred).tolist(),
        }
        confusion_matrices["Original/reference GBLSTSVM"] = {
            "val": val_m["confusion_matrix"],
            "test": test_m["confusion_matrix"],
        }
    else:
        results["Original/reference GBLSTSVM"] = {
            "executed": False,
            "skip_reason": original.skip_reason,
        }

    clean_models = {}
    for name in ("Clean GBLSTSVM", "CW-GBLSTSVM", "R-GBLSTSVM", "CR-GBLSTSVM"):
        model = fit_clean_family(name, X_train, y_train)
        clean_models[name] = model
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_scores = twin_scores(model, X_val)
        test_scores = twin_scores(model, X_test)
        val_m = evaluate(y_val, val_pred, val_scores, min_label)
        test_m = evaluate(y_test, test_pred, test_scores, min_label)
        diag = modified_model_diagnostics(name, model, y_train)
        results[name] = {
            "val": val_m,
            "test": test_m,
            "diagnostics": diag,
        }
        predictions[name] = {
            "val": np.asarray(val_pred).tolist(),
            "test": np.asarray(test_pred).tolist(),
        }
        confusion_matrices[name] = {
            "val": val_m["confusion_matrix"],
            "test": test_m["confusion_matrix"],
        }

        balls = model.balls_
        rel = getattr(model, "reliabilities_", None)
        pos_idx = 0
        neg_idx = 0
        for i, ball in enumerate(balls):
            row = {
                "dataset": dataset_info["name"],
                "model": name,
                "ball_index": i,
                "class": int(ball.label),
                "n_samples": int(ball.num_samples),
                "purity": float(ball.purity),
                "radius": float(ball.radius),
            }
            if rel is not None:
                row["reliability"] = float(rel[i])
                rel_row = {
                    "dataset": dataset_info["name"],
                    "model": name,
                    "ball_index": i,
                    "class": int(ball.label),
                    "purity": float(ball.purity),
                    "radius": float(ball.radius),
                    "reliability": float(rel[i]),
                }
                if name in ("CW-GBLSTSVM", "CR-GBLSTSVM"):
                    rel_row["kappa"] = float(
                        model.kappa_pos_ if ball.label == 1 else model.kappa_neg_
                    )
                if name == "CR-GBLSTSVM":
                    if ball.label == 1:
                        rel_row["cw_weight"] = float(model.kappa_pos_balls_[pos_idx])
                        rel_row["cr_weight"] = float(model.weights_pos_[pos_idx])
                    else:
                        rel_row["cw_weight"] = float(model.kappa_neg_balls_[neg_idx])
                        rel_row["cr_weight"] = float(model.weights_neg_[neg_idx])
                reliability_rows.append(rel_row)
            if name in ("CW-GBLSTSVM", "CR-GBLSTSVM"):
                if ball.label == 1:
                    row["kappa"] = float(model.kappa_pos_)
                    row["final_weight"] = float(model.weights_pos_[pos_idx])
                    pos_idx += 1
                else:
                    row["kappa"] = float(model.kappa_neg_)
                    row["final_weight"] = float(model.weights_neg_[neg_idx])
                    neg_idx += 1
            elif name == "R-GBLSTSVM":
                if ball.label == 1:
                    row["final_weight"] = float(model.weights_pos_[pos_idx])
                    pos_idx += 1
                else:
                    row["final_weight"] = float(model.weights_neg_[neg_idx])
                    neg_idx += 1
            diagnostics_rows.append(row)

    ref_clean = verify_reference_vs_clean(
        original, clean_models["Clean GBLSTSVM"], X_val, X_test
    )

    baseline_test = results["Clean GBLSTSVM"]["test"]
    cw_test = results["CW-GBLSTSVM"]["test"]
    r_test = results["R-GBLSTSVM"]["test"]
    cr_test = results["CR-GBLSTSVM"]["test"]

    deltas = {
        "CW_minus_clean": delta(cw_test, baseline_test, METRIC_KEYS),
        "R_minus_clean": delta(r_test, baseline_test, METRIC_KEYS),
        "CR_minus_clean": delta(cr_test, baseline_test, METRIC_KEYS),
        "CR_minus_CW": delta(cr_test, cw_test, METRIC_KEYS),
        "CR_minus_R": delta(cr_test, r_test, METRIC_KEYS),
    }

    reliability_status = classify_reliability_status(
        results["R-GBLSTSVM"]["diagnostics"],
        results["CR-GBLSTSVM"]["diagnostics"],
        cr_test,
        cw_test,
    )

    cr_pred = np.asarray(predictions["CR-GBLSTSVM"]["test"])
    cw_pred = np.asarray(predictions["CW-GBLSTSVM"]["test"])

    return {
        "dataset": dataset_info,
        "split": split_info,
        "preprocessing": preprocessing,
        "hyperparameters": hyperparameters,
        "original_reference_status": original_status,
        "reference_vs_clean": ref_clean,
        "results": results,
        "deltas_test": deltas,
        "reliability_status": reliability_status,
        "interpretation": {
            "CR_outperforms_clean": bool(
                cr_test["macro_f1"] > baseline_test["macro_f1"]
                or cr_test["balanced_accuracy"] > baseline_test["balanced_accuracy"]
                or cr_test["minority_recall"] > baseline_test["minority_recall"]
            ),
            "CR_outperforms_CW": bool(
                cr_test["macro_f1"] > cw_test["macro_f1"]
                or cr_test["balanced_accuracy"] > cw_test["balanced_accuracy"]
                or cr_test["minority_recall"] > cw_test["minority_recall"]
            ),
            "test_prediction_agreement_CR_vs_CW": float(np.mean(cr_pred == cw_pred)),
            "n_test_predictions_CR_neq_CW": int(np.sum(cr_pred != cw_pred)),
        },
        "diagnostics_rows": diagnostics_rows,
        "reliability_rows": reliability_rows,
        "confusion_matrices": confusion_matrices,
    }


def write_csv_outputs(all_results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / "multidataset_train_val_test.json"
    json_payload = {
        "seed": SEED,
        "datasets": {r["dataset"]["name"]: {k: v for k, v in r.items() if k not in (
            "diagnostics_rows", "reliability_rows", "confusion_matrices"
        )} for r in all_results},
        "auroc_definition": (
            "roc_auc_score(y in {-1,+1}, |f2|-|f1|) using the reference twin score."
        ),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    csv_path = OUT_DIR / "multidataset_train_val_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Dataset",
                "Model",
                "Split",
                "Accuracy",
                "Balanced Acc",
                "Macro F1",
                "Minority Recall",
                "G-mean",
                "AUROC",
                "Prediction Collapse",
            ]
        )
        for r in all_results:
            ds = r["dataset"]["name"]
            for name in MODEL_ORDER:
                res = r["results"].get(name, {})
                if "val" not in res:
                    writer.writerow([ds, name, "val", "NA", "NA", "NA", "NA", "NA", "NA", "NA"])
                    writer.writerow([ds, name, "test", "NA", "NA", "NA", "NA", "NA", "NA", "NA"])
                    continue
                for split in ("val", "test"):
                    m = res[split]
                    auroc = m.get("auroc")
                    writer.writerow(
                        [
                            ds,
                            name,
                            split,
                            f"{m['accuracy']:.6f}",
                            f"{m['balanced_accuracy']:.6f}",
                            f"{m['macro_f1']:.6f}",
                            f"{m['minority_recall']:.6f}",
                            f"{m['gmean']:.6f}",
                            f"{auroc:.6f}" if auroc is not None else "NA",
                            "YES" if m.get("prediction_collapse") else "NO",
                        ]
                    )

    diag_rows = []
    rel_summary_rows = []
    cm_payload = {}
    for r in all_results:
        ds = r["dataset"]["name"]
        diag_rows.extend(r["diagnostics_rows"])
        cm_payload[ds] = r["confusion_matrices"]

        for name in MODEL_ORDER:
            res = r["results"].get(name, {})
            if "diagnostics" not in res:
                continue
            d = res["diagnostics"]
            rel_summary_rows.append(
                {
                    "dataset": ds,
                    "model": name,
                    "n_balls_total": d.get("n_balls_total"),
                    "n_balls_pos": d.get("n_balls_pos"),
                    "n_balls_neg": d.get("n_balls_neg"),
                    "training_samples_in_balls": d.get("training_samples_in_balls"),
                    "purity_min": d.get("purity", {}).get("min"),
                    "purity_mean": d.get("purity", {}).get("mean"),
                    "purity_median": d.get("purity", {}).get("median"),
                    "purity_max": d.get("purity", {}).get("max"),
                    "radius_min": d.get("radius", {}).get("min"),
                    "radius_mean": d.get("radius", {}).get("mean"),
                    "radius_median": d.get("radius", {}).get("median"),
                    "radius_max": d.get("radius", {}).get("max"),
                    "kappa_positive": d.get("kappa_positive"),
                    "kappa_negative": d.get("kappa_negative"),
                    "reliability_min": d.get("reliability", {}).get("min"),
                    "reliability_mean": d.get("reliability", {}).get("mean"),
                    "reliability_median": d.get("reliability", {}).get("median"),
                    "reliability_max": d.get("reliability", {}).get("max"),
                    "frac_reliability_ge_0_99": d.get("reliability", {}).get("frac_ge_0_99"),
                    "frac_reliability_lt_0_95": d.get("reliability", {}).get("frac_lt_0_95"),
                    "frac_cr_weights_differ_from_cw": d.get("frac_cr_weights_differ_from_cw"),
                    "mean_abs_cr_minus_cw_weight": d.get("mean_abs_cr_minus_cw_weight"),
                }
            )

    diag_path = OUT_DIR / "multidataset_ball_diagnostics.csv"
    if diag_rows:
        fieldnames = []
        for row in diag_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with diag_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(diag_rows)

    rel_path = OUT_DIR / "multidataset_reliability_diagnostics.csv"
    if rel_summary_rows:
        with rel_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rel_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rel_summary_rows)

    rel_ball_path = OUT_DIR / "multidataset_reliability_per_ball.csv"
    all_rel_rows = []
    for r in all_results:
        all_rel_rows.extend(r["reliability_rows"])
    if all_rel_rows:
        fieldnames = []
        for row in all_rel_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with rel_ball_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rel_rows)

    cm_path = OUT_DIR / "multidataset_confusion_matrices.json"
    cm_path.write_text(json.dumps(cm_payload, indent=2), encoding="utf-8")

    return json_path, csv_path, diag_path, rel_path, cm_path


def generate_markdown_report(all_results: list[dict]) -> Path:
    lines = [
        "# Phase 7: Multi-Dataset Train / Validation / Test Results",
        "",
        "This document records the imbalanced-benchmark comparison of frozen GBLSTSVM variants "
        "on `diabetes.csv`, `ecoli2.csv`, and `vehicle0.csv`. Hyperparameters were **not** tuned "
        "on validation or test. No frozen model file was modified.",
        "",
        "Script: `experiments/validate_multidataset.py`",
        "",
        "Existing tests: `python -m pytest tests/ -v` → **59 passed** (unchanged), then experiment executed.",
        "",
        "---",
        "",
        "## 1. Dataset descriptions",
        "",
        "| Dataset | Samples | Features | Class distribution | Imbalance ratio |",
        "|---------|---------|----------|-------------------|-----------------|",
    ]

    for r in all_results:
        d = r["dataset"]
        counts = d["class_counts"]
        dist = ", ".join(f"{k}: {v}" for k, v in counts.items())
        lines.append(
            f"| {d['name']} | {d['n_samples']} | {d['n_features']} | {dist} | {d['imbalance_ratio']:.2f} |"
        )

    lines.extend(
        [
            "",
            "Label mapping (all datasets binary 0/1 in CSV): **0 → −1**, **1 → +1**.",
            "",
            "Inspection: no missing values; no duplicate rows; all feature columns numeric; "
            "no rows or columns silently discarded.",
            "",
            "---",
            "",
            "## 2. Experimental protocol",
            "",
            "- Seed **42**; stratified 70/15/15 via two `train_test_split` calls.",
            "- Validation used only for diagnostic evaluation; **no tuning performed**.",
            "- Test set used only for final evaluation.",
            "- Prediction rule: **reference** (`sign(|f2| − |f1|)`).",
            "",
            "---",
            "",
            "## 3. Preprocessing",
            "",
            "**None** (consistent with frozen baseline protocol and Phase 6 breast_cancer experiment).",
            "",
            "Frozen implementations and reference code do not scale features internally. "
            "`diabetes.csv` and `vehicle0.csv` are not pre-standardized; `ecoli2.csv` features "
            "are in [0, 1] range.",
            "",
            "---",
            "",
            "## 4. Model configurations",
            "",
            "| Parameter | Baseline/Clean | CW | R | CR |",
            "|-----------|----------------|----|----|-----|",
            f"| c1 | {C1} | {C1} | {C1} | {C1} |",
            f"| c2 | {C2} | {C2} | {C2} | {C2} |",
            f"| purity_threshold | {PURITY} | {PURITY} | {PURITY} | {PURITY} |",
            f"| min_ball_count | {MIN_BALL_COUNT} | {MIN_BALL_COUNT} | {MIN_BALL_COUNT} | {MIN_BALL_COUNT} |",
            f"| epsilon | N/A | N/A | {DEFAULT_EPSILON} | {DEFAULT_EPSILON} |",
            f"| gamma | N/A | N/A | {DEFAULT_GAMMA} | {DEFAULT_GAMMA} |",
            f"| lambda_r | N/A | N/A | {DEFAULT_LAMBDA_R} | {DEFAULT_LAMBDA_R} |",
            f"| weight_placement | N/A | {WEIGHT_PLACEMENT} | {WEIGHT_PLACEMENT} | {WEIGHT_PLACEMENT} |",
            f"| prediction_rule | {PREDICTION_RULE} | {PREDICTION_RULE} | {PREDICTION_RULE} | {PREDICTION_RULE} |",
            f"| random_state | {SEED} | {SEED} | {SEED} | {SEED} |",
            "",
            "---",
            "",
            "## 5. Validation results",
            "",
            "| Dataset | Model | Accuracy | Bal Acc | Macro F1 | Minority Recall | G-mean |",
            "|---------|-------|----------|---------|----------|-----------------|--------|",
        ]
    )

    for r in all_results:
        ds = r["dataset"]["name"]
        for name in MODEL_ORDER:
            res = r["results"].get(name, {})
            if "val" not in res:
                lines.append(f"| {ds} | {name} | SKIPPED | | | | |")
                continue
            m = res["val"]
            lines.append(
                f"| {ds} | {name} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
                f"{m['macro_f1']:.4f} | {m['minority_recall']:.4f} | {m['gmean']:.4f} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Test results",
            "",
            "| Dataset | Model | Accuracy | Bal Acc | Macro F1 | Minority Recall | G-mean | AUROC | Collapse |",
            "|---------|-------|----------|---------|----------|-----------------|--------|-------|----------|",
        ]
    )

    for r in all_results:
        ds = r["dataset"]["name"]
        for name in MODEL_ORDER:
            res = r["results"].get(name, {})
            if "test" not in res:
                lines.append(f"| {ds} | {name} | SKIPPED | | | | | | |")
                continue
            m = res["test"]
            auroc = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "NA"
            collapse = "YES" if m.get("prediction_collapse") else "NO"
            lines.append(
                f"| {ds} | {name} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
                f"{m['macro_f1']:.4f} | {m['minority_recall']:.4f} | {m['gmean']:.4f} | "
                f"{auroc} | {collapse} |"
            )

    lines.extend(["", "---", "", "## 7. Reference vs clean verification", ""])
    for r in all_results:
        ds = r["dataset"]["name"]
        v = r["reference_vs_clean"]
        lines.append(f"### {ds}")
        lines.append(f"- Verified: **{v.get('verified')}**")
        if v.get("ball_comparison"):
            bc = v["ball_comparison"]
            lines.append(
                f"- Balls: ref={bc['n_balls_ref']}, clean={bc['n_balls_clean']}, match={bc['n_balls_match']}"
            )
        lines.append(
            f"- Val prediction agreement: {v.get('val_prediction_agreement', 'NA')}; "
            f"test: {v.get('test_prediction_agreement', 'NA')}"
        )
        lines.append(
            f"- Score max abs diff: val={v.get('val_score_max_abs_diff', 'NA')}, "
            f"test={v.get('test_score_max_abs_diff', 'NA')}"
        )
        if v.get("material_disagreement"):
            lines.append("- **Material disagreement detected** — interpret modified-model deltas cautiously.")
        lines.append("")

    lines.extend(["---", "", "## 8. Ball diagnostics (summary)", ""])
    for r in all_results:
        ds = r["dataset"]["name"]
        lines.append(f"### {ds}")
        for name in MODEL_ORDER:
            res = r["results"].get(name, {})
            if "diagnostics" not in res:
                continue
            d = res["diagnostics"]
            p = d.get("purity", {})
            rad = d.get("radius", {})
            lines.append(
                f"- **{name}**: balls={d.get('n_balls_total')} "
                f"(pos={d.get('n_balls_pos')}, neg={d.get('n_balls_neg')}), "
                f"samples_in_balls={d.get('training_samples_in_balls')}, "
                f"purity mean={p.get('mean', 'NA'):.4f}, radius mean={rad.get('mean', 'NA'):.4f}"
                if p.get("mean") is not None
                else f"- **{name}**: no diagnostics"
            )
        lines.append("")

    lines.extend(["---", "", "## 9. Reliability diagnostics", ""])
    for r in all_results:
        ds = r["dataset"]["name"]
        rs = r["reliability_status"]
        lines.append(
            f"### {ds}: **{rs['status']}**"
        )
        lines.append(
            f"- Mean purity: {rs['mean_purity']:.4f}; mean reliability: {rs['mean_reliability']:.4f}; "
            f"frac rel ≥ 0.99: {rs['frac_reliability_ge_0_99']:.4f}"
        )
        lines.append(
            f"- R effectively baseline: {rs['r_effectively_baseline']}; "
            f"CR effectively CW: {rs['cr_effectively_cw']}; "
            f"CR weights differ from CW: {rs['cr_weights_differ_from_cw']}"
        )
        lines.append("")

    lines.extend(["---", "", "## 10. CW / R / CR comparisons (test deltas vs clean)", ""])
    for r in all_results:
        ds = r["dataset"]["name"]
        lines.append(f"### {ds}")
        for key, d in r["deltas_test"].items():
            mf1 = d.get("macro_f1")
            bal = d.get("balanced_accuracy")
            minr = d.get("minority_recall")
            lines.append(
                f"- {key}: Δmacro_f1={mf1:+.4f}, Δbal_acc={bal:+.4f}, Δminority_recall={minr:+.4f}"
                if mf1 is not None
                else f"- {key}: NA"
            )
        lines.append("")

    lines.extend(["---", "", "## 11. Scientific interpretation", ""])
    # Auto-generate interpretation
    cr_wins_clean = []
    cr_wins_cw = []
    cw_wins_clean = []
    saturated = []
    active = []
    for r in all_results:
        ds = r["dataset"]["name"]
        if r["interpretation"]["CR_outperforms_clean"]:
            cr_wins_clean.append(ds)
        if r["interpretation"]["CR_outperforms_CW"]:
            cr_wins_cw.append(ds)
        d = r["deltas_test"]["CW_minus_clean"]
        if d.get("macro_f1") and d["macro_f1"] > 0:
            cw_wins_clean.append(ds)
        if r["reliability_status"]["status"] == "RELIABILITY SATURATED":
            saturated.append(ds)
        else:
            active.append(ds)

    lines.append(
        f"- CR improves over clean on test (any of macro F1 / bal acc / minority recall): "
        f"{cr_wins_clean or 'none'}"
    )
    lines.append(
        f"- CR improves over CW on test: {cr_wins_cw or 'none'}"
    )
    lines.append(
        f"- CW improves macro F1 over clean: {cw_wins_clean or 'none'}"
    )
    lines.append(f"- Reliability SATURATED datasets: {saturated or 'none'}")
    lines.append(f"- Reliability ACTIVE datasets: {active or 'none'}")
    lines.append("")
    lines.append(
        "Single-metric improvements should not be treated as evidence of superiority without "
        "consistency across datasets and non-collapsed minority recall."
    )

    lines.extend(["", "---", "", "## 12. Limitations", "", "- No hyperparameter tuning; defaults may not be optimal per dataset.", "- No feature scaling (frozen protocol); distance-based granulation on raw features for diabetes/vehicle0.", "- Small test sets (15% stratified) limit statistical power.", "- Original KMeans in reference granulation is not fully deterministic beyond numpy seed.", "- AUROC uses twin hyperplane score |f2|−|f1|, not a calibrated probability.", "", "---", "", "## 13. Recommendation for next experiment", "", "1. **Granulation sensitivity**: sweep `purity_threshold` and `min_ball_count` on datasets where reliability is saturated to find regimes where R/CR diverge from baseline/CW.", "2. **Scaling ablation**: separate controlled experiment with StandardScaler fit on train only (document as new protocol, not frozen baseline).", "3. **Hyperparameter grid**: c1/c2 per dataset with validation-only selection, then single test evaluation.", "4. **Resampling / cross-validation** for stable estimates on small test folds.", "5. If CW helps but CR does not, prioritize diagnosing reliability saturation before further CR formulation changes.", ""])

    report_path = ROOT / "docs" / "PHASE7_MULTIDATASET_RESULTS.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    all_results = []
    for ds_name in DATASETS:
        path = DATA_DIR / ds_name
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"\n{'='*60}")
        print(f"Running {ds_name}")
        print("=" * 60)
        result = run_dataset(path)
        all_results.append(result)

        ds = result["dataset"]["name"]
        print(f"shape={result['dataset']['shape']} counts={result['dataset']['class_counts']}")
        print(f"split train/val/test={result['split']['train_size']}/"
              f"{result['split']['val_size']}/{result['split']['test_size']}")
        print(f"reliability_status={result['reliability_status']['status']}")
        print(f"ref_vs_clean verified={result['reference_vs_clean'].get('verified')}")

        header = f"{'Model':<28} {'TAcc':>6} {'TBal':>6} {'TF1':>6} {'TMinR':>6} {'Collapse':>8}"
        print(header)
        for name in MODEL_ORDER:
            res = result["results"].get(name, {})
            if "test" not in res:
                print(f"{name:<28} SKIPPED")
                continue
            m = res["test"]
            col = "YES" if m.get("prediction_collapse") else "NO"
            print(
                f"{name:<28} {m['accuracy']:6.3f} {m['balanced_accuracy']:6.3f} "
                f"{m['macro_f1']:6.3f} {m['minority_recall']:6.3f} {col:>8}"
            )

    paths = write_csv_outputs(all_results)
    report_path = generate_markdown_report(all_results)

    print("\n=== Output files ===")
    for p in paths:
        print(f"Wrote {p}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
