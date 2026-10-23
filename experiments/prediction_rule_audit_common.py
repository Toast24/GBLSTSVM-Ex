"""
Shared helpers for experiment A3 — prediction-rule audit.

Compares RULE_REPO vs RULE_REFERENCE on identical fitted twin hyperplanes.
Does not modify src/ or existing experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

from experiments.analyze_granular_geometry_vs_performance import build_representation
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN, compute_gaussian_memberships
from experiments.granular_compression_k_common import COLLAPSE_BALL_THRESHOLD
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, DatasetSplit, evaluate, normalize_labels
from experiments.validation_selected_k_common import REPRESENTATION
from src.cbcm import split_membership_by_class
from src.class_weights import ball_class_weights, compute_binary_class_weights
from src.cw_gblstsvm import solve_cw_gblstsvm_systems
from src.gblstsvm import predict_twin_hyperplanes
from src.granular_balls import BallInfo, balls_to_class_arrays
from src.prediction_diagnostics import (
    compute_prediction_diagnostics,
    summarize_disagreements,
)

RuleName = Literal["RULE_REPO", "RULE_REFERENCE"]

# Maps audit names to src.gblstsvm.predict_twin_hyperplanes rule argument.
RULE_TO_SRC: dict[RuleName, str] = {
    "RULE_REPO": "reference",
    "RULE_REFERENCE": "paper",
}

SEEDS = list(range(10))

# Deterministic synthetic diagnostic (Session 1 / diagnose_prediction_discrepancy.py).
SYNTHETIC_SEED = 0
SYNTHETIC_N = 100
SYNTHETIC_D = 4
SYNTHETIC_PURITY = 0.985
SYNTHETIC_MIN_BALL_COUNT = 2
SYNTHETIC_C1 = 1000.0
SYNTHETIC_C2 = 1e-5


@dataclass(frozen=True)
class RuleConventionDoc:
    """Documented mapping for README / verdict checks."""

    f_definition: str = "f_i(x) = w_i^T x + b_i (bias included in the affine score, not in ||w_i||)"
    w_definition: str = "w_i are the weight vectors from solve_cw_gblstsvm_systems (bias b_i separate)"
    norm_definition: str = "||w_i|| = Euclidean L2 norm of w_i only; bias excluded from normalization"
    label_encoding: str = "Binary labels mapped to +1 (minority or CSV 1) and -1 (majority or CSV 0)"
    hyperplane_1: str = "w1,b1: first twin subproblem (positive ball centers A, opposite class B + R-)"
    hyperplane_2: str = "w2,b2: second twin subproblem (negative ball centers B, opposite class A + R+)"
    rule_repo: str = "sign(|f2(x)| - |f1(x)|); +1 if |f2|>|f1|, -1 if |f2|<|f1|, 0 if equal"
    rule_reference: str = "argmin_i |f_i(x)|/||w_i||; predict +1 if d1<=d2 else -1 (ties -> +1)"
    repo_zero_handling: str = "predictions 0 mapped to -1 at metric time (phase8_common.evaluate)"
    ambiguity: str = "None for this implementation; rules map directly to src predict_twin_hyperplanes."


CONVENTION = RuleConventionDoc()


def build_synthetic_dataset(seed: int = SYNTHETIC_SEED, n: int = SYNTHETIC_N, d: int = SYNTHETIC_D):
    np.random.seed(seed)
    X = np.random.randn(n, d)
    y = np.sign(X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n))
    y[y == 0] = 1
    return X, y


def fit_synthetic_clean_gblstsvm(seed: int = SYNTHETIC_SEED) -> tuple[Any, np.ndarray, np.ndarray]:
    """Reproduce Session 1 diagnostic fit (unweighted clean GBLSTSVM)."""
    from src.gblstsvm import GBLSTSVM

    X, y = build_synthetic_dataset(seed=seed)
    model = GBLSTSVM(
        c1=SYNTHETIC_C1,
        c2=SYNTHETIC_C2,
        purity_threshold=SYNTHETIC_PURITY,
        min_ball_count=SYNTHETIC_MIN_BALL_COUNT,
        prediction_rule="reference",
        random_state=seed,
    )
    model.fit(X, y)
    return model, X, y


def run_deterministic_disagreement_check() -> dict[str, Any]:
    """Part 1: reproduce synthetic ~29% diagnostic on training points."""
    model, X, _y = fit_synthetic_clean_gblstsvm()
    diag = compute_prediction_diagnostics(X, model.w1_, model.b1_, model.w2_, model.b2_)
    summary = summarize_disagreements(diag)
    summary["context"] = "synthetic_train_all_points"
    summary["c1"] = SYNTHETIC_C1
    summary["c2"] = SYNTHETIC_C2
    return summary


@dataclass
class FitResult:
    w1: np.ndarray | None
    b1: float | None
    w2: np.ndarray | None
    b2: float | None
    n_balls: int
    classifier_ok: bool
    classifier_error: str | None
    status: str  # finite | nan | exception


def fit_rho90_hyperplanes(split: DatasetSplit, seed: int) -> FitResult:
    """
    Fit twin hyperplanes once using the FULL_RHO90 evaluation path:
    rho=0.90 granulation + CW class weights + Gaussian memberships + opposite_only placement.
    Same solver as validation_selected_k_common.fit_gaussian_with_predictions (fixed c1/c2).
    """
    from experiments.analyze_granular_geometry_vs_performance import REG_TERM, WEIGHT_PLACEMENT

    try:
        balls, _kept, rep_stats = build_representation(split, seed, REPRESENTATION)
        n_balls = int(rep_stats.get("n_balls_after_del", len(balls)))
        if not balls:
            return FitResult(
                None, None, None, None, n_balls, False, "no balls after del_ball", "nan"
            )
        y = normalize_labels(split.y_train)
        try:
            balls_to_class_arrays(balls)
        except ValueError as exc:
            return FitResult(None, None, None, None, n_balls, False, str(exc), "nan")

        kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y)
        kappa_pos_b, kappa_neg_b = ball_class_weights(balls, kappa_pos, kappa_neg)
        mu_g = compute_gaussian_memberships(balls, PRIMARY_GAUSSIAN)
        mu_pos, mu_neg = split_membership_by_class(balls, mu_g)
        weights_pos = kappa_pos_b * mu_pos
        weights_neg = kappa_neg_b * mu_neg
        centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(balls)
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
        return FitResult(w1, b1, w2, b2, n_balls, True, None, "finite")
    except Exception as exc:  # noqa: BLE001
        return FitResult(None, None, None, None, 0, False, f"{type(exc).__name__}: {exc}", "exception")


def predict_with_rule(
    X: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    rule: RuleName,
) -> np.ndarray:
    return predict_twin_hyperplanes(X, w1, b1, w2, b2, rule=RULE_TO_SRC[rule])


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_n = normalize_labels(y_true)
    y_pred_n = y_pred.copy()
    y_pred_n[y_pred_n == 0.0] = -1.0
    return {
        "f1_pos": float(f1_score(y_true_n, y_pred_n, pos_label=1.0, zero_division=0)),
        "f1_neg": float(f1_score(y_true_n, y_pred_n, pos_label=-1.0, zero_division=0)),
    }


def collapse_flags(
    n_balls: int,
    classifier_ok: bool,
    macro_f1: float,
    metrics: dict[str, Any],
) -> tuple[int, str]:
    """Project collapse definition (validation_selected_k_common.evaluate_representation_full)."""
    if n_balls <= COLLAPSE_BALL_THRESHOLD:
        return 1, f"n_balls<={COLLAPSE_BALL_THRESHOLD}"
    if not classifier_ok:
        return 1, "classifier_failed"
    if not np.isfinite(macro_f1):
        return 1, "invalid_macro_f1"
    if metrics.get("prediction_collapse"):
        return 1, "prediction_collapse"
    return 0, ""


def metrics_for_rule(
    split: DatasetSplit,
    fit: FitResult,
    rule: RuleName,
) -> dict[str, Any]:
    if not fit.classifier_ok or fit.w1 is None or fit.w2 is None:
        return {
            "prediction_rule": rule,
            "status": fit.status,
            "classifier_ok": False,
            "classifier_error": fit.classifier_error,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1_pos": float("nan"),
            "f1_neg": float("nan"),
            "confusion_matrix": None,
            "prediction_collapse": float("nan"),
            "collapsed": 1,
            "collapse_reason": fit.classifier_error or fit.status,
            "norm_w1": float("nan"),
            "norm_w2": float("nan"),
            "norm_ratio": float("nan"),
        }

    w1, b1, w2, b2 = fit.w1, fit.b1, fit.w2, fit.b2
    X_test = split.X_test
    y_pred = predict_with_rule(X_test, w1, b1, w2, b2, rule)
    scores = np.abs(X_test @ w2 + b2) - np.abs(X_test @ w1 + b1)
    ev = evaluate(split.y_test, y_pred, scores, split.minority_label)
    pc = per_class_f1(split.y_test, y_pred)
    n1 = float(np.linalg.norm(w1))
    n2 = float(np.linalg.norm(w2))
    ratio = n1 / n2 if n2 > 0 else float("inf")
    collapsed, reason = collapse_flags(fit.n_balls, True, ev["macro_f1"], ev)
    return {
        "prediction_rule": rule,
        "status": "finite",
        "classifier_ok": True,
        "classifier_error": None,
        "accuracy": ev["accuracy"],
        "macro_f1": ev["macro_f1"],
        "balanced_accuracy": ev["balanced_accuracy"],
        "f1_pos": pc["f1_pos"],
        "f1_neg": pc["f1_neg"],
        "confusion_matrix": ev["confusion_matrix"],
        "prediction_collapse": int(ev["prediction_collapse"]),
        "collapsed": collapsed,
        "collapse_reason": reason,
        "norm_w1": n1,
        "norm_w2": n2,
        "norm_ratio": ratio,
        "y_pred": y_pred,
    }


def disagreement_stats(
    y_repo: np.ndarray,
    y_ref: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    y_true_n = normalize_labels(y_true)
    agree = y_repo == y_ref
    n = int(y_repo.size)
    n_dis = int(np.sum(~agree))
    cm = confusion_matrix(y_repo, y_ref, labels=[-1.0, 1.0])
    per_true: dict[str, Any] = {}
    for label in (-1.0, 1.0):
        mask = y_true_n == label
        if not np.any(mask):
            per_true[f"disagree_rate_true_{label:g}"] = float("nan")
            continue
        per_true[f"disagree_rate_true_{label:g}"] = float(np.mean(y_repo[mask] != y_ref[mask]))
    return {
        "n_predictions": n,
        "n_agree": int(np.sum(agree)),
        "n_disagree": n_dis,
        "disagree_rate": n_dis / n if n else float("nan"),
        "cm_repo_rows_ref_cols": cm.astype(int).tolist(),
        **per_true,
    }


def classify_disagreement_examples(
    X: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    y_repo: np.ndarray,
    y_ref: np.ndarray,
    max_examples: int = 5,
) -> list[dict[str, Any]]:
    f1 = X @ w1 + b1
    f2 = X @ w2 + b2
    n1 = float(np.linalg.norm(w1))
    n2 = float(np.linalg.norm(w2))
    d1 = np.abs(f1) / (n1 if n1 > 0 else 1.0)
    d2 = np.abs(f2) / (n2 if n2 > 0 else 1.0)
    mask = y_repo != y_ref
    idxs = np.flatnonzero(mask)
    examples: list[dict[str, Any]] = []
    for i in idxs[:max_examples]:
        i = int(i)
        raw_order = "+1" if np.abs(f2[i]) > np.abs(f1[i]) else "-1"
        if np.isclose(np.abs(f1[i]), np.abs(f2[i])):
            raw_order = "tie"
        norm_order = "+1" if d1[i] <= d2[i] else "-1"
        examples.append(
            {
                "sample_index": i,
                "abs_f1": float(np.abs(f1[i])),
                "abs_f2": float(np.abs(f2[i])),
                "d1_norm": float(d1[i]),
                "d2_norm": float(d2[i]),
                "norm_w1": n1,
                "norm_w2": n2,
                "raw_winner_class": raw_order,
                "norm_winner_class": norm_order,
                "repo_pred": float(y_repo[i]),
                "ref_pred": float(y_ref[i]),
                "normalization_flipped_winner": raw_order != norm_order and raw_order != "tie",
            }
        )
    n_flip = int(
        np.sum(
            mask
            & (
                (np.abs(f2) > np.abs(f1))
                != (np.abs(f1) / (n1 if n1 > 0 else 1.0) <= np.abs(f2) / (n2 if n2 > 0 else 1.0))
            )
        )
    )
    return examples


def _cohens_d(deltas: np.ndarray) -> float:
    if deltas.size < 2:
        return float("nan")
    return float(np.mean(deltas) / (np.std(deltas, ddof=1) + 1e-12))


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        samp = rng.choice(v, size=v.size, replace=True)
        means.append(float(np.mean(samp)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def dataset_level_stats(deltas: np.ndarray) -> dict[str, Any]:
    d = np.asarray(deltas, dtype=float)
    d_fin = d[np.isfinite(d)]
    out: dict[str, Any] = {
        "n_datasets": int(d_fin.size),
        "mean_delta": float(np.mean(d_fin)) if d_fin.size else float("nan"),
        "median_delta": float(np.median(d_fin)) if d_fin.size else float("nan"),
        "std_delta": float(np.std(d_fin, ddof=1)) if d_fin.size > 1 else float("nan"),
        "cohens_d": _cohens_d(d_fin),
    }
    if d_fin.size:
        wins = int(np.sum(d_fin > 0))
        losses = int(np.sum(d_fin < 0))
        ties = int(np.sum(d_fin == 0))
        out.update({"wins": wins, "ties": ties, "losses": losses})
        try:
            stat, p = stats.wilcoxon(d_fin, alternative="two-sided", zero_method="wilcox")
            out["wilcoxon_stat"] = float(stat)
            out["wilcoxon_p"] = float(p)
        except ValueError:
            out["wilcoxon_stat"] = float("nan")
            out["wilcoxon_p"] = float("nan")
        lo, hi = bootstrap_ci_mean(d_fin)
        out["bootstrap_95_ci_low"] = lo
        out["bootstrap_95_ci_high"] = hi
    return out


def spearman_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)
