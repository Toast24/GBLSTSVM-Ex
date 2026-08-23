"""
Phase 6: stratified train/validation/test experiment on breast_cancer.csv.

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

DATA_PATH = ROOT / "Data" / "Data" / "breast_cancer.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"

MODEL_ORDER = [
    "Original/reference GBLSTSVM",
    "Clean GBLSTSVM",
    "CW-GBLSTSVM",
    "R-GBLSTSVM",
    "CR-GBLSTSVM",
]


def inspect_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(path, header=None)
    n_cols = df.shape[1]
    if n_cols != 10:
        raise ValueError(f"Expected 10 columns (A:I features + J label); got {n_cols}.")

    feature_df = df.iloc[:, :9]
    y_raw = df.iloc[:, 9]
    non_numeric = [
        int(c)
        for c in feature_df.columns
        if not np.issubdtype(feature_df[c].dtype, np.number)
    ]
    label_values = sorted(pd.unique(y_raw).tolist(), key=lambda v: (str(type(v)), v))
    counts = y_raw.value_counts().sort_index().to_dict()
    counts = {int(k) if not isinstance(k, str) else k: int(v) for k, v in counts.items()}

    info = {
        "path": str(path),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "n_samples": int(df.shape[0]),
        "n_features": 9,
        "class_counts": counts,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "label_values": [int(v) if not isinstance(v, str) else v for v in label_values],
        "non_numeric_feature_columns": non_numeric,
        "feature_means": [float(x) for x in feature_df.mean().to_numpy()],
        "feature_stds": [float(x) for x in feature_df.std(ddof=0).to_numpy()],
        "already_approximately_standardized": bool(
            np.allclose(feature_df.mean().to_numpy(), 0.0, atol=1e-4)
            and np.allclose(feature_df.std(ddof=0).to_numpy(), 1.0, atol=1e-2)
        ),
        "duplicates_retained": True,
    }
    X = feature_df.to_numpy(dtype=float)
    y = y_raw.to_numpy(dtype=float)
    return X, y, info


def normalize_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel().copy()
    y[y == 0] = -1.0
    return y


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
        "std": std,
        "cv": float(std / mean) if mean != 0 else float("nan"),
    }


def minority_class(y: np.ndarray) -> float:
    y = normalize_labels(y)
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == -1.0))
    return 1.0 if n_pos <= n_neg else -1.0


def twin_scores(model, X: np.ndarray) -> np.ndarray:
    """Reference-rule continuous score: |f2| - |f1|. Positive favors class +1."""
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
    n_zero = int(np.sum(y_pred == 0.0))
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
        "confusion_matrix_labels": [-1, 1],
        "confusion_matrix": cm.astype(int).tolist(),
        "n_zero_predictions_mapped_to_neg": n_zero,
        "recall_pos": rec_pos,
        "recall_neg": rec_neg,
    }


def ball_diagnostics(balls, extra: dict | None = None) -> dict:
    purities = np.array([float(b.purity) for b in balls], dtype=float)
    radii = np.array([float(b.radius) for b in balls], dtype=float)
    n_pos = sum(1 for b in balls if b.label == 1)
    n_neg = sum(1 for b in balls if b.label != 1)
    rec = {
        "n_balls_pos": int(n_pos),
        "n_balls_neg": int(n_neg),
        "n_balls_total": int(len(balls)),
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
        except Exception as exc:  # noqa: BLE001 — document original-pipeline failures
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
        rec["kappa_pos"] = float(model.kappa_pos_)
        rec["kappa_neg"] = float(model.kappa_neg_)
    if hasattr(model, "reliabilities_") and model.reliabilities_ is not None:
        rec["reliability"] = reliability_diagnostics(model.reliabilities_)
        rec["radius_scale"] = float(model.radius_scale_) if model.radius_scale_ is not None else None
    if name == "CR-GBLSTSVM":
        kappa_all = []
        for b in model.balls_:
            kappa_all.append(model.kappa_pos_ if b.label == 1 else model.kappa_neg_)
        kappa_all = np.asarray(kappa_all, dtype=float)
        # CR weights are stored in class-array order, not original ball order.
        final = np.concatenate([model.weights_pos_, model.weights_neg_])
        rec["final_cr_weight"] = distribution_summary(final)
        cw_like = np.concatenate([model.kappa_pos_balls_, model.kappa_neg_balls_])
        rec["frac_cr_weights_differ_from_cw"] = float(
            np.mean(np.abs(final - cw_like) > 1e-12)
        )
        rec["max_abs_cr_minus_cw_weight"] = float(np.max(np.abs(final - cw_like)))
        rec["mean_abs_cr_minus_cw_weight"] = float(np.mean(np.abs(final - cw_like)))
    return rec


def metric_row(val_m: dict, test_m: dict) -> dict:
    return {
        "val_accuracy": val_m["accuracy"],
        "val_balanced_accuracy": val_m["balanced_accuracy"],
        "val_macro_f1": val_m["macro_f1"],
        "val_minority_recall": val_m["minority_recall"],
        "test_accuracy": test_m["accuracy"],
        "test_balanced_accuracy": test_m["balanced_accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "test_minority_recall": test_m["minority_recall"],
    }


def delta(a: dict, b: dict, keys: tuple[str, ...]) -> dict:
    return {k: float(a[k] - b[k]) for k in keys}


def main() -> None:
    X, y, dataset_info = inspect_dataset(DATA_PATH)
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
        "label_mapping": "CSV 0 -> -1, CSV 1 -> +1",
    }

    preprocessing = {
        "method": "none",
        "reason": (
            "MinMaxScaler is imported but unused in gen_ball.py / GBLSTSVM.py; "
            "main.py loads raw CSV values with no scaling. Frozen Phase 1–5 models "
            "also do not scale internally. breast_cancer.csv features already have "
            "mean ~0 and std ~1. Introducing MinMaxScaler would be a new pipeline "
            "step, so none is applied. Split is performed on raw rows first."
        ),
        "fitted_on_train_only": None,
        "leakage_controls": [
            "raw split before any statistics",
            "granular balls generated from training rows only",
            "class weights from training labels only",
            "reliability radius scale from training balls only",
            "validation unused for test construction or hyperparameter selection",
        ],
    }

    hyperparameters = {
        "baseline": {
            "c1": C1,
            "c2": C2,
            "purity_threshold": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "prediction_rule": PREDICTION_RULE,
            "random_state": SEED,
        },
        "CW": {
            "class_weighting": "kappa_c = n_train / (2 * n_c)",
            "weight_placement": WEIGHT_PLACEMENT,
        },
        "R": {
            "epsilon": DEFAULT_EPSILON,
            "gamma": DEFAULT_GAMMA,
            "lambda_r": DEFAULT_LAMBDA_R,
            "reliability_variant": RELIABILITY_VARIANT,
            "weight_placement": WEIGHT_PLACEMENT,
        },
        "CR": {
            "final_weight": "kappa * reliability",
            "weight_placement": WEIGHT_PLACEMENT,
        },
        "tuning": "none; Phase 2–5 defaults; validation not used for model selection",
    }

    results = {}
    predictions = {}
    diagnostics_rows = []

    original = OriginalReferenceGBLSTSVM(C1, C2, PURITY, MIN_BALL_COUNT)
    original.fit(X_train, y_train)
    original_status = {
        "executed": original.supported,
        "skip_reason": original.skip_reason,
        "note": (
            "Invoked unmodified gen_ball.GBList via src.reference_pipeline. "
            "Original KMeans(n_clusters=2) has no random_state; numpy is seeded "
            "with 42 immediately before granulation. Test labels are not passed "
            "into training. Predictions use reference_predict_from_matrix, which "
            "reproduces LSTWGBSVM's sign(|f2|-|f1|) rule without editing GBLSTSVM.py."
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
                "num_samples_per_ball": "not exposed by ReferenceBall",
            },
        )
        results["Original/reference GBLSTSVM"] = {
            **metric_row(val_m, test_m),
            "val": val_m,
            "test": test_m,
            "diagnostics": diag,
        }
        predictions["Original/reference GBLSTSVM"] = {
            "val": val_pred.tolist(),
            "test": test_pred.tolist(),
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
            **metric_row(val_m, test_m),
            "val": val_m,
            "test": test_m,
            "diagnostics": diag,
        }
        predictions[name] = {
            "val": np.asarray(val_pred).tolist(),
            "test": np.asarray(test_pred).tolist(),
        }

        balls = model.balls_
        rel = getattr(model, "reliabilities_", None)
        pos_idx = 0
        neg_idx = 0
        for i, ball in enumerate(balls):
            row = {
                "model": name,
                "ball_index": i,
                "class": int(ball.label),
                "n_samples": int(ball.num_samples),
                "purity": float(ball.purity),
                "radius": float(ball.radius),
            }
            if rel is not None:
                row["reliability"] = float(rel[i])
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

    keys = ("balanced_accuracy", "macro_f1", "minority_recall")
    baseline_test = results["Clean GBLSTSVM"]["test"]
    cw_test = results["CW-GBLSTSVM"]["test"]
    r_test = results["R-GBLSTSVM"]["test"]
    cr_test = results["CR-GBLSTSVM"]["test"]

    deltas = {
        "CW_minus_baseline": delta(cw_test, baseline_test, keys),
        "R_minus_baseline": delta(r_test, baseline_test, keys),
        "CR_minus_baseline": delta(cr_test, baseline_test, keys),
        "CR_minus_CW": delta(cr_test, cw_test, keys),
        "CR_minus_R": delta(cr_test, r_test, keys),
    }

    cr_pred = np.asarray(predictions["CR-GBLSTSVM"]["test"])
    cw_pred = np.asarray(predictions["CW-GBLSTSVM"]["test"])
    r_pred = np.asarray(predictions["R-GBLSTSVM"]["test"])
    clean_pred = np.asarray(predictions["Clean GBLSTSVM"]["test"])
    cr_cw_agree = float(np.mean(cr_pred == cw_pred))
    n_cr_cw_diff = int(np.sum(cr_pred != cw_pred))

    cr_diag = results["CR-GBLSTSVM"]["diagnostics"]
    rel_info = cr_diag.get("reliability", {})
    interpretation = {
        "A_CR_outperforms_clean_GBLSTSVM": bool(
            cr_test["macro_f1"] > baseline_test["macro_f1"]
            or cr_test["balanced_accuracy"] > baseline_test["balanced_accuracy"]
            or cr_test["minority_recall"] > baseline_test["minority_recall"]
        ),
        "B_CR_outperforms_CW": bool(
            cr_test["macro_f1"] > cw_test["macro_f1"]
            or cr_test["balanced_accuracy"] > cw_test["balanced_accuracy"]
            or cr_test["minority_recall"] > cw_test["minority_recall"]
        ),
        "C_R_outperforms_baseline": bool(
            r_test["macro_f1"] > baseline_test["macro_f1"]
            or r_test["balanced_accuracy"] > baseline_test["balanced_accuracy"]
            or r_test["minority_recall"] > baseline_test["minority_recall"]
        ),
        "D_reliability_varies": bool(rel_info.get("varies", False)),
        "E_CR_predictions_differ_from_CW": bool(n_cr_cw_diff > 0),
        "test_prediction_agreement_CR_vs_CW": cr_cw_agree,
        "n_test_predictions_CR_neq_CW": n_cr_cw_diff,
        "n_test_predictions_CR_neq_R": int(np.sum(cr_pred != r_pred)),
        "n_test_predictions_CR_neq_clean": int(np.sum(cr_pred != clean_pred)),
        "frac_cr_weights_differ_from_cw": cr_diag.get("frac_cr_weights_differ_from_cw"),
        "reliability_frac_ge_0_99": rel_info.get("frac_ge_0_99"),
        "reliability_frac_lt_0_95": rel_info.get("frac_lt_0_95"),
        "F_gains_vs_numerical_noise_note": (
            "Deltas smaller than one validation/test sample (~1/n) should be "
            "treated as numerical noise pending resampling studies."
        ),
        "test_n": int(len(y_test)),
        "one_test_sample_accuracy_delta": float(1.0 / len(y_test)),
    }

    payload = {
        "dataset": dataset_info,
        "split": split_info,
        "preprocessing": preprocessing,
        "hyperparameters": hyperparameters,
        "original_reference_status": original_status,
        "results": results,
        "deltas_test": deltas,
        "interpretation_flags": interpretation,
        "auroc_definition": (
            "roc_auc_score(y in {-1,+1}, |f2|-|f1|) using the reference twin "
            "score; not an exposed model API."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "breast_cancer_train_val_test.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = OUT_DIR / "breast_cancer_train_val_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Val Accuracy",
                "Val Bal Acc",
                "Val Macro F1",
                "Val Minority Recall",
                "Test Accuracy",
                "Test Bal Acc",
                "Test Macro F1",
                "Test Minority Recall",
            ]
        )
        for name in MODEL_ORDER:
            row = results[name]
            if "val_accuracy" not in row:
                writer.writerow([name, "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"])
                continue
            writer.writerow(
                [
                    name,
                    f"{row['val_accuracy']:.6f}",
                    f"{row['val_balanced_accuracy']:.6f}",
                    f"{row['val_macro_f1']:.6f}",
                    f"{row['val_minority_recall']:.6f}",
                    f"{row['test_accuracy']:.6f}",
                    f"{row['test_balanced_accuracy']:.6f}",
                    f"{row['test_macro_f1']:.6f}",
                    f"{row['test_minority_recall']:.6f}",
                ]
            )

    diag_path = OUT_DIR / "breast_cancer_ball_diagnostics.csv"
    if diagnostics_rows:
        fieldnames = []
        for row in diagnostics_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with diag_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(diagnostics_rows)

    print("=== Dataset ===")
    print(
        f"shape={dataset_info['shape']} counts={dataset_info['class_counts']} "
        f"missing={dataset_info['missing_values']} dups={dataset_info['duplicate_rows']}"
    )
    print(
        f"split train/val/test={split_info['train_size']}/"
        f"{split_info['val_size']}/{split_info['test_size']}"
    )
    print(f"preprocessing={preprocessing['method']}")
    print(f"minority_label={min_label}")
    print(f"original executed={original.supported} reason={original.skip_reason}")
    print()
    header = (
        f"{'Model':<28} {'VAcc':>6} {'VBal':>6} {'VF1':>6} {'VMinR':>6} "
        f"{'TAcc':>6} {'TBal':>6} {'TF1':>6} {'TMinR':>6}"
    )
    print(header)
    for name in MODEL_ORDER:
        row = results[name]
        if "val_accuracy" not in row:
            print(f"{name:<28} SKIPPED ({row.get('skip_reason')})")
            continue
        print(
            f"{name:<28} {row['val_accuracy']:6.3f} {row['val_balanced_accuracy']:6.3f} "
            f"{row['val_macro_f1']:6.3f} {row['val_minority_recall']:6.3f} "
            f"{row['test_accuracy']:6.3f} {row['test_balanced_accuracy']:6.3f} "
            f"{row['test_macro_f1']:6.3f} {row['test_minority_recall']:6.3f}"
        )
    print("\n=== Test deltas ===")
    for name, d in deltas.items():
        print(
            f"{name}: bal_acc={d['balanced_accuracy']:+.4f} "
            f"macro_f1={d['macro_f1']:+.4f} min_recall={d['minority_recall']:+.4f}"
        )
    print("\n=== CR vs CW ===")
    print(
        f"test prediction agreement={cr_cw_agree:.4f} "
        f"n_diff={n_cr_cw_diff} "
        f"reliability_varies={rel_info.get('varies')} "
        f"frac>=0.99={rel_info.get('frac_ge_0_99')} "
        f"frac_cr_w_diff={cr_diag.get('frac_cr_weights_differ_from_cw')}"
    )
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {diag_path}")


if __name__ == "__main__":
    main()
