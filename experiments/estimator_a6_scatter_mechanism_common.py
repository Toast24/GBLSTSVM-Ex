"""
A6 — Scatter mechanism (directional vs isotropic vs ridge control).

Experiment-only. Scatter enters H[:d,:d] exactly as A4 L2:
  H += n_j u_j u_j^T + S_j^{(arm)}
L1 uses S=0. Ridge arms use L1 moments + lambda I_d on the same block (once per class H).

Trace-matched ridge (per class c):
  lambda_c = (sum_{j in class c} tr(S_j)) / d
so tr(lambda_c I_d) = sum_j tr(S_j), matching total diagonal trace from ISOTROPIC arm.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

from experiments.estimator_information_ladder_common import (
    REG_BASE,
    build_baseline_rho90_balls,
    split_balls,
)
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, DatasetSplit, normalize_labels
from experiments.prediction_rule_audit_common import RULE_TO_SRC, RuleName, predict_with_rule
from experiments.rebaseline_common import C_GRID
from experiments.validation_selected_k_common import make_train_val_partition
from src.granular_balls import BallInfo
from src.ladder_gblstsvm import ridge_scale

SEEDS = list(range(10))

ScatterArm = Literal[
    "L1_POPULATION",
    "FULL_SCATTER",
    "DIAG_SCATTER",
    "ISOTROPIC_SCATTER",
    "RIDGE_TRACE_MATCH",
    "RIDGE_VAL_TUNED",
]

TuningMode = Literal["MODE_A_SHARED_C", "MODE_B_PER_ARM_C"]

ARMS: list[ScatterArm] = [
    "L1_POPULATION",
    "FULL_SCATTER",
    "DIAG_SCATTER",
    "ISOTROPIC_SCATTER",
    "RIDGE_TRACE_MATCH",
    "RIDGE_VAL_TUNED",
]

RULES: list[RuleName] = ["RULE_REPO", "RULE_REFERENCE"]
LAMBDA_MULT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _raw_scatter(ball: BallInfo, dim: int) -> np.ndarray:
    if ball.scatter is not None:
        return np.asarray(ball.scatter, dtype=float)
    return np.zeros((dim, dim), dtype=float)


def scatter_variant(S: np.ndarray, arm: ScatterArm, dim: int) -> np.ndarray | None:
    """Return S block to add to H[:d,:d], or None for pure L1."""
    if arm == "L1_POPULATION":
        return None
    if arm == "FULL_SCATTER":
        return S
    if arm == "DIAG_SCATTER":
        return np.diag(np.diag(S))
    if arm == "ISOTROPIC_SCATTER":
        tr = float(np.trace(S))
        return (tr / dim) * np.eye(dim, dtype=float)
    if arm in ("RIDGE_TRACE_MATCH", "RIDGE_VAL_TUNED"):
        return None
    raise ValueError(arm)


def class_trace_sum(balls: Sequence[BallInfo], dim: int) -> float:
    return float(sum(np.trace(_raw_scatter(b, dim)) for b in balls))


def trace_matched_lambda(balls: Sequence[BallInfo], dim: int) -> float:
    """lambda such that tr(lambda I_d) equals sum_j tr(S_j) for this class."""
    return class_trace_sum(balls, dim) / dim


def accumulate_class_scatter_arm(
    balls: Sequence[BallInfo],
    arm: ScatterArm,
    *,
    ridge_lambda: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not balls:
        raise ValueError("empty balls")
    dim = int(balls[0].center.shape[0])
    design = np.zeros((dim + 1, dim + 1), dtype=float)
    rhs = np.zeros(dim + 1, dtype=float)
    trace_added = 0.0
    for ball in balls:
        u = np.concatenate([np.asarray(ball.center, dtype=float), [1.0]])
        n_j = float(ball.num_samples)
        design += n_j * np.outer(u, u)
        rhs += n_j * u
        S = _raw_scatter(ball, dim)
        block = scatter_variant(S, arm, dim)
        if block is not None:
            design[:dim, :dim] += block
            trace_added += float(np.trace(block))
    if arm == "RIDGE_TRACE_MATCH":
        lam = trace_matched_lambda(balls, dim)
        design[:dim, :dim] += lam * np.eye(dim, dtype=float)
        trace_added += lam * dim
        ridge_lambda = lam
    elif arm == "RIDGE_VAL_TUNED" and ridge_lambda > 0:
        design[:dim, :dim] += ridge_lambda * np.eye(dim, dtype=float)
        trace_added += ridge_lambda * dim
    return design, rhs, ridge_lambda


def solve_twin_scatter_arm(
    pos_balls: Sequence[BallInfo],
    neg_balls: Sequence[BallInfo],
    arm: ScatterArm,
    c1: float,
    c2: float,
    *,
    ridge_lambda_pos: float = 0.0,
    ridge_lambda_neg: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray, float, float, float]:
    da, ra, lp = accumulate_class_scatter_arm(
        pos_balls, arm, ridge_lambda=ridge_lambda_pos
    )
    db, rb, ln = accumulate_class_scatter_arm(
        neg_balls, arm, ridge_lambda=ridge_lambda_neg
    )
    dim = da.shape[0]
    eye = np.identity(dim, dtype=float)

    comb1 = db + (1.0 / c1) * da
    reg1 = ridge_scale(comb1, REG_BASE)
    inv_p_1 = np.linalg.inv(comb1 + reg1 * eye)
    hyper_p_1 = -inv_p_1 @ rb.reshape(-1, 1)

    comb2 = da + (1.0 / c2) * db
    reg2 = ridge_scale(comb2, REG_BASE)
    inv_p_2 = np.linalg.inv(comb2 + reg2 * eye)
    hyper_p_2 = inv_p_2 @ ra.reshape(-1, 1)

    w1 = hyper_p_1[:-1, 0]
    b1 = float(hyper_p_1[-1, 0])
    w2 = hyper_p_2[:-1, 0]
    b2 = float(hyper_p_2[-1, 0])
    lam_rep = float(np.nanmean([lp, ln]))
    return w1, b1, w2, b2, lam_rep, float(np.linalg.cond(comb1))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = normalize_labels(y_true)
    yp = y_pred.copy()
    yp[yp == 0.0] = -1.0
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def _val_metrics(
    split: DatasetSplit,
    balls: list[BallInfo],
    arm: ScatterArm,
    c1: float,
    c2: float,
    seed: int,
    ridge_mult: float = 1.0,
) -> float:
    y_train = normalize_labels(split.y_train)
    partition = make_train_val_partition(split.name, seed, split.X_train, y_train)
    val_idx = partition.validation_indices
    X_val = split.X_train[val_idx]
    y_val = y_train[val_idx]
    pos, neg = split_balls(balls)
    dim = int(balls[0].center.shape[0])
    lp = trace_matched_lambda(pos, dim) * ridge_mult
    ln = trace_matched_lambda(neg, dim) * ridge_mult
    try:
        w1, b1, w2, b2, _, _ = solve_twin_scatter_arm(
            pos,
            neg,
            arm,
            c1,
            c2,
            ridge_lambda_pos=lp if arm == "RIDGE_VAL_TUNED" else 0.0,
            ridge_lambda_neg=ln if arm == "RIDGE_VAL_TUNED" else 0.0,
        )
        from src.gblstsvm import predict_twin_hyperplanes

        preds = predict_twin_hyperplanes(X_val, w1, b1, w2, b2, rule="reference")
        return _macro_f1(y_val, preds)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def tune_c1_c2_for_arm(
    split: DatasetSplit,
    balls: list[BallInfo],
    arm: ScatterArm,
    seed: int,
) -> tuple[float, float, float]:
    best_c1, best_c2, best_f1 = DEFAULT_C1, DEFAULT_C2, -np.inf
    for c1 in C_GRID:
        for c2 in C_GRID:
            f1 = _val_metrics(split, balls, arm, c1, c2, seed, ridge_mult=1.0)
            if np.isfinite(f1) and f1 > best_f1:
                best_f1, best_c1, best_c2 = f1, c1, c2
    return best_c1, best_c2, float(best_f1 if np.isfinite(best_f1) else float("nan"))


def tune_ridge_multiplier(
    split: DatasetSplit,
    balls: list[BallInfo],
    seed: int,
    c1: float,
    c2: float,
) -> tuple[float, float, float]:
    """Validation-tuned multiplier on trace-matched lambda (per class)."""
    best_mult = 1.0
    best_f1 = -np.inf
    for mult in LAMBDA_MULT_GRID:
        f1 = _val_metrics(
            split, balls, "RIDGE_VAL_TUNED", c1, c2, seed, ridge_mult=mult
        )
        if np.isfinite(f1) and f1 > best_f1:
            best_f1, best_mult = f1, mult
    pos, neg = split_balls(balls)
    dim = int(balls[0].center.shape[0])
    lp = trace_matched_lambda(pos, dim) * best_mult
    ln = trace_matched_lambda(neg, dim) * best_mult
    return lp, ln, float(best_f1 if np.isfinite(best_f1) else float("nan"))


def scatter_geometry(balls: list[BallInfo]) -> dict[str, float]:
    if not balls:
        return {}
    dim = int(balls[0].center.shape[0])
    full_tr = diag_tr = iso_tr = 0.0
    frob_fd = frob_fi = frob_di = 0.0
    for ball in balls:
        S = _raw_scatter(ball, dim)
        Sd = np.diag(np.diag(S))
        Si = (np.trace(S) / dim) * np.eye(dim)
        full_tr += float(np.trace(S))
        diag_tr += float(np.trace(Sd))
        iso_tr += float(np.trace(Si))
        frob_fd += float(np.linalg.norm(S - Sd, ord="fro"))
        frob_fi += float(np.linalg.norm(S - Si, ord="fro"))
        frob_di += float(np.linalg.norm(Sd - Si, ord="fro"))
    n = len(balls)
    return {
        "trace_full": full_tr,
        "trace_diagonal": diag_tr,
        "trace_isotropic": iso_tr,
        "fro_full_minus_diag": frob_fd,
        "fro_full_minus_isotropic": frob_fi,
        "fro_diag_minus_isotropic": frob_di,
        "n_balls_geom": n,
    }


def fit_and_predict(
    split: DatasetSplit,
    balls: list[BallInfo],
    arm: ScatterArm,
    c1: float,
    c2: float,
    ridge_lambda_pos: float,
    ridge_lambda_neg: float,
) -> tuple[np.ndarray, float, np.ndarray, float, bool, str | None, float, float]:
    pos, neg = split_balls(balls)
    try:
        t0 = time.perf_counter()
        w1, b1, w2, b2, lam_eff, cond = solve_twin_scatter_arm(
            pos,
            neg,
            arm,
            c1,
            c2,
            ridge_lambda_pos=ridge_lambda_pos if arm == "RIDGE_VAL_TUNED" else 0.0,
            ridge_lambda_neg=ridge_lambda_neg if arm == "RIDGE_VAL_TUNED" else 0.0,
        )
        train_time = time.perf_counter() - t0
        return w1, b1, w2, b2, True, None, lam_eff, cond
    except (ValueError, np.linalg.LinAlgError) as exc:
        d = split.X_train.shape[1]
        return np.zeros(d), 0.0, np.zeros(d), 0.0, False, str(exc), float("nan"), float("nan")


def evaluate_cell(
    split: DatasetSplit,
    seed: int,
    arm: ScatterArm,
    tuning_mode: TuningMode,
    shared_c1: float | None,
    shared_c2: float | None,
) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    balls = build_baseline_rho90_balls(split, seed)
    n_balls = len(balls)
    geom = scatter_geometry(balls)
    dim = int(split.X_train.shape[1])

    ridge_lp = ridge_ln = 0.0
    val_f1 = float("nan")
    if tuning_mode == "MODE_A_SHARED_C":
        c1 = shared_c1 if shared_c1 is not None else DEFAULT_C1
        c2 = shared_c2 if shared_c2 is not None else DEFAULT_C2
        if arm == "RIDGE_VAL_TUNED":
            ridge_lp, ridge_ln, val_f1 = tune_ridge_multiplier(split, balls, seed, c1, c2)
    else:
        c1, c2, val_f1 = tune_c1_c2_for_arm(split, balls, arm, seed)
        if arm == "RIDGE_VAL_TUNED":
            ridge_lp, ridge_ln, val_f1 = tune_ridge_multiplier(split, balls, seed, c1, c2)

    w1, b1, w2, b2, ok, err, lam_eff, cond = fit_and_predict(
        split, balls, arm, c1, c2, ridge_lp, ridge_ln
    )
    trace_total = geom.get("trace_full", float("nan"))
    trace_contrib = trace_total
    if arm == "ISOTROPIC_SCATTER":
        trace_contrib = geom.get("trace_isotropic", trace_total)
    elif arm == "DIAG_SCATTER":
        trace_contrib = geom.get("trace_diagonal", trace_total)
    elif arm in ("RIDGE_TRACE_MATCH", "RIDGE_VAL_TUNED"):
        trace_contrib = lam_eff * dim if np.isfinite(lam_eff) else float("nan")

    rows: list[dict[str, Any]] = []
    for rule in RULES:
        t_pred = time.perf_counter()
        if ok:
            y_pred = predict_with_rule(split.X_test, w1, b1, w2, b2, rule)
            macro = _macro_f1(split.y_test, y_pred)
            yt = normalize_labels(split.y_test)
            yp = np.where(y_pred == 0, -1, y_pred)
            acc = float(np.mean(yt == yp))
            bacc = float(balanced_accuracy_score(yt, yp))
        else:
            macro = acc = bacc = float("nan")
        pred_time = time.perf_counter() - t_pred
        rows.append(
            {
                "dataset": split.name,
                "seed": seed,
                "arm": arm,
                "tuning_mode": tuning_mode,
                "prediction_rule": rule,
                "c1": float(c1),
                "c2": float(c2),
                "lambda": float(lam_eff) if np.isfinite(lam_eff) else float("nan"),
                "ridge_lambda_pos": ridge_lp,
                "ridge_lambda_neg": ridge_ln,
                "validation_macro_f1": val_f1,
                "n_balls": n_balls,
                "macro_f1": macro,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "classifier_ok": ok,
                "finite_macro_f1": int(np.isfinite(macro)),
                "classifier_error": err or "",
                "train_time_seconds": time.perf_counter() - t0,
                "predict_time_seconds": pred_time,
                "trace_S_total": trace_total,
                "trace_contribution": trace_contrib,
                "condition_number_comb1": cond,
                **geom,
            }
        )
    return rows


def dataset_level_stats(deltas: np.ndarray) -> dict[str, Any]:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    out: dict[str, Any] = {"n_datasets": int(d.size)}
    if d.size == 0:
        return {**out, "mean_delta": float("nan"), "wilcoxon_p": float("nan")}
    out.update(
        {
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "wins": int(np.sum(d > 0)),
            "ties": int(np.sum(d == 0)),
            "losses": int(np.sum(d < 0)),
        }
    )
    if d.size >= 2:
        try:
            stat, p = stats.wilcoxon(d, zero_method="wilcox")
            out["wilcoxon_stat"] = float(stat)
            out["wilcoxon_p"] = float(p)
        except ValueError:
            out["wilcoxon_stat"] = float("nan")
            out["wilcoxon_p"] = float("nan")
        rng = np.random.default_rng(0)
        boots = [float(np.mean(d[rng.integers(0, d.size, d.size)])) for _ in range(3000)]
        out["bootstrap_95_ci_low"] = float(np.percentile(boots, 2.5))
        out["bootstrap_95_ci_high"] = float(np.percentile(boots, 97.5))
    return out


def paired_delta(
    df: "pd.DataFrame",
    arm_a: str,
    arm_b: str,
    rule: str,
    tuning_mode: str,
) -> np.ndarray:
    import pandas as pd

    sub = df[
        (df["prediction_rule"] == rule)
        & (df["tuning_mode"] == tuning_mode)
        & (df["arm"].isin([arm_a, arm_b]))
    ]
    wide = sub.pivot_table(index=["dataset", "seed"], columns="arm", values="macro_f1")
    if arm_a not in wide.columns or arm_b not in wide.columns:
        return np.array([])
    return (wide[arm_a] - wide[arm_b]).groupby(level=0).mean().to_numpy()


import pandas as pd  # noqa: E402
