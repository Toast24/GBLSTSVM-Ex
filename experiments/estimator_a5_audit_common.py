"""
A5 audit helpers — k-invariance (partition-preserving L3) and A4 rule/collapse rescoring.

Experiment-only; does not modify src/ or existing experiment scripts.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from experiments.estimator_information_ladder_common import (
    IDENTITY_TOL,
    PREDICTION_RULE,
    SEEDS,
    _balanced_acc,
    _macro_f1,
    _matrix_errors,
    balls_from_samples,
    build_baseline_rho90_balls,
    class_moments_from_balls,
    class_moments_from_samples,
    partition_coverage,
    solve_twin_from_balls,
    split_balls,
)
from experiments.granular_compression_k_common import COLLAPSE_BALL_THRESHOLD
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, DatasetSplit, normalize_labels
from experiments.prediction_rule_audit_common import (
    RULE_TO_SRC,
    RuleName,
    dataset_level_stats,
    predict_with_rule,
)
from experiments.test_split_utility_granulation import _build_indexed_matrix
from src.gblstsvm import predict_twin_hyperplanes
from src.granular_balls import BallInfo, GranularBall

K_VALUES = [1, 2, 4, 8, 16, 81]

MethodArm = Literal[
    "RAW_LSTSVM",
    "L0_CENTER_UNIT",
    "L1_POPULATION",
    "L2_POPULATION_SCATTER",
    "L3_MOMENT_EXACT",
]

RESCORE_METHODS: list[MethodArm] = [
    "RAW_LSTSVM",
    "L0_CENTER_UNIT",
    "L1_POPULATION",
    "L2_POPULATION_SCATTER",
    "L3_MOMENT_EXACT",
]

ARM_MAP: dict[MethodArm, str] = {
    "RAW_LSTSVM": "L3",
    "L0_CENTER_UNIT": "L0",
    "L1_POPULATION": "L1",
    "L2_POPULATION_SCATTER": "L2",
    "L3_MOMENT_EXACT": "L3",
}

RULE_REPO: RuleName = "RULE_REPO"
RULE_REF: RuleName = "RULE_REFERENCE"


def allocate_class_ball_counts(
    requested_k: int,
    n_pos: int,
    n_neg: int,
) -> tuple[int, int, int]:
    """
    Split requested total ball count across classes (>=1 each).
    Returns (k_pos, k_neg, capped_target).
    """
    n = n_pos + n_neg
    if n <= 0:
        return 0, 0, 0
    if requested_k < 2:
        return min(1, n_pos), min(1, n_neg), 2
    target = min(int(requested_k), n)
    k_pos = max(1, int(round(target * n_pos / n)))
    k_neg = target - k_pos
    if k_neg < 1:
        k_neg = 1
        k_pos = target - 1
    k_pos = min(k_pos, n_pos)
    k_neg = min(k_neg, n_neg)
    while k_pos + k_neg < target and (k_pos < n_pos or k_neg < n_neg):
        if k_pos < n_pos:
            k_pos += 1
        elif k_neg < n_neg:
            k_neg += 1
        else:
            break
    return k_pos, k_neg, target


def _granular_to_ball_info(ball: GranularBall, label: float) -> BallInfo:
    return BallInfo(
        center=ball.center.copy(),
        radius=float(ball.r),
        label=label,
        purity=float(ball.purity),
        num_samples=int(ball.num),
        scatter=ball.scatter.copy(),
    )


def _partition_indexed_to_k(indexed: np.ndarray, k_target: int, seed: int) -> list[GranularBall]:
    """Complete partition of indexed rows into k_target GranularBalls (deterministic)."""
    if k_target <= 0:
        return []
    if indexed.shape[0] == 0:
        return []
    np.random.seed(seed)
    balls: list[GranularBall] = [GranularBall(indexed)]
    if k_target == 1:
        return balls
    while len(balls) < k_target:
        splittable = [i for i, b in enumerate(balls) if b.num > 1]
        if not splittable:
            break
        idx = max(splittable, key=lambda i: balls[i].num)
        left, right = balls[idx].split_2balls()
        balls[idx] = left
        balls.append(right)
    while len(balls) > k_target:
        if len(balls) < 2:
            break
        best_i = 0
        best_size = balls[0].num + balls[1].num
        for i in range(len(balls) - 1):
            sz = balls[i].num + balls[i + 1].num
            if sz < best_size:
                best_size = sz
                best_i = i
        merged = GranularBall(np.vstack([balls[best_i].data, balls[best_i + 1].data]))
        balls = balls[:best_i] + [merged] + balls[best_i + 2 :]
    return balls


def build_partition_preserving_at_k(
    split: DatasetSplit,
    seed: int,
    requested_k: int,
) -> tuple[list[BallInfo], int, int, int, str]:
    """
    Deterministic complete partition into ~requested_k balls (per-class binary splits/merges).
    Does not use baseline singleton deletion.
    """
    y = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y)
    idx_pos = np.flatnonzero(y == 1.0)
    idx_neg = np.flatnonzero(y == -1.0)
    n_pos, n_neg = int(idx_pos.size), int(idx_neg.size)
    k_pos, k_neg, capped = allocate_class_ball_counts(requested_k, n_pos, n_neg)

    pos_granular = _partition_indexed_to_k(indexed[idx_pos], k_pos, seed + 17)
    neg_granular = _partition_indexed_to_k(indexed[idx_neg], k_neg, seed + 29)

    balls: list[BallInfo] = []
    for g in pos_granular:
        balls.append(_granular_to_ball_info(g, 1.0))
    for g in neg_granular:
        balls.append(_granular_to_ball_info(g, -1.0))

    actual_k = len(balls)
    note = ""
    if requested_k < 2:
        note = "requested_k<2_binary_minimum_2_balls"
    elif actual_k != requested_k:
        if actual_k != capped:
            note = f"achieved_{actual_k}_requested_{requested_k}_capped_{capped}"
        else:
            note = f"achieved_{actual_k}_requested_{requested_k}"

    return balls, requested_k, actual_k, capped, note


def collapse_a4_definition(
    n_balls: int,
    classifier_ok: bool,
    macro_f1: float,
    method: MethodArm,
) -> tuple[int, str]:
    """Exact A5 collapse definition (no prediction-collapse term)."""
    if not classifier_ok:
        return 1, "classifier_failed"
    if not np.isfinite(macro_f1):
        return 1, "invalid_macro_f1"
    if method != "RAW_LSTSVM" and n_balls <= COLLAPSE_BALL_THRESHOLD:
        return 1, f"n_balls<={COLLAPSE_BALL_THRESHOLD}"
    return 0, ""


def run_k_invariance_cell(
    split: DatasetSplit,
    seed: int,
    requested_k: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    y = normalize_labels(split.y_train)
    n_train = int(y.size)
    balls, req_k, actual_k, capped, k_note = build_partition_preserving_at_k(
        split, seed, requested_k
    )
    complete, covered = partition_coverage(balls, n_train)
    pos_b, neg_b = split_balls(balls)
    n_pos_balls = len(pos_b)
    n_neg_balls = len(neg_b)

    X = split.X_train
    X_pos, X_neg = X[y == 1.0], X[y == -1.0]
    Hp_raw, gp_raw = class_moments_from_samples(X_pos)
    Hn_raw, gn_raw = class_moments_from_samples(X_neg)
    Hp_l3, gp_l3 = class_moments_from_balls(pos_b)
    Hn_l3, gn_l3 = class_moments_from_balls(neg_b)

    H_rel = max(_matrix_errors(Hp_raw, Hp_l3)[1], _matrix_errors(Hn_raw, Hn_l3)[1])
    g_rel = max(
        float(np.linalg.norm(gp_raw - gp_l3) / (np.linalg.norm(gp_raw) + 1e-12)),
        float(np.linalg.norm(gn_raw - gn_l3) / (np.linalg.norm(gn_raw) + 1e-12)),
    )

    c1, c2 = DEFAULT_C1, DEFAULT_C2
    classifier_ok = False
    solver_error = ""
    try:
        raw_pos, raw_neg = balls_from_samples(X, y)
        w_raw = solve_twin_from_balls(raw_pos, raw_neg, c1, c2, "L3")
        w_l3 = solve_twin_from_balls(pos_b, neg_b, c1, c2, "L3")
        param_rel = max(
            float(np.linalg.norm(w_raw[0] - w_l3[0]) / (np.linalg.norm(w_raw[0]) + 1e-12)),
            abs(w_raw[1] - w_l3[1]) / (abs(w_raw[1]) + 1e-12),
            float(np.linalg.norm(w_raw[2] - w_l3[2]) / (np.linalg.norm(w_raw[2]) + 1e-12)),
            abs(w_raw[3] - w_l3[3]) / (abs(w_raw[3]) + 1e-12),
        )
        preds_raw = predict_twin_hyperplanes(
            split.X_test, *w_raw, rule=PREDICTION_RULE
        )
        preds_l3 = predict_twin_hyperplanes(
            split.X_test, *w_l3, rule=PREDICTION_RULE
        )
        pred_dis = float(np.mean(preds_raw != preds_l3))
        macro_f1 = _macro_f1(split.y_test, preds_l3)
        acc = float(
            np.mean(
                normalize_labels(split.y_test)
                == np.where(preds_l3 == 0, -1, preds_l3)
            )
        )
        bacc = _balanced_acc(split.y_test, preds_l3)
        classifier_ok = True
    except (ValueError, np.linalg.LinAlgError) as exc:
        param_rel = float("nan")
        pred_dis = float("nan")
        macro_f1 = acc = bacc = float("nan")
        solver_error = f"{type(exc).__name__}: {exc}"

    identity_ok = (
        complete
        and classifier_ok
        and H_rel <= IDENTITY_TOL
        and g_rel <= IDENTITY_TOL
        and param_rel <= IDENTITY_TOL
        and pred_dis <= 1e-12
    )

    return {
        "dataset": split.name,
        "seed": seed,
        "requested_k": req_k,
        "actual_k": actual_k,
        "capped_target_k": capped,
        "k_construction_note": k_note,
        "n_pos_balls": n_pos_balls,
        "n_neg_balls": n_neg_balls,
        "partition_complete": int(complete),
        "n_covered": covered,
        "H_relative_error": H_rel,
        "g_relative_error": g_rel,
        "parameter_relative_error": param_rel,
        "prediction_disagreement_vs_raw": pred_dis,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "classifier_ok": classifier_ok,
        "solver_error": solver_error,
        "identity_pass": int(identity_ok),
        "runtime_seconds": time.perf_counter() - t0,
    }


def balls_for_a4_method(split: DatasetSplit, seed: int, method: MethodArm) -> tuple[list[BallInfo], int]:
    y = normalize_labels(split.y_train)
    if method == "RAW_LSTSVM":
        pos, neg = balls_from_samples(split.X_train, y)
        balls = pos + neg
        return balls, len(balls)
    balls = build_baseline_rho90_balls(split, seed)
    return balls, len(balls)


def solve_a4_arm(
    split: DatasetSplit,
    seed: int,
    method: MethodArm,
    c1: float,
    c2: float,
) -> tuple[np.ndarray, float, np.ndarray, float, int, bool, str | None]:
    arm = ARM_MAP[method]
    try:
        balls, n_balls = balls_for_a4_method(split, seed, method)
        pos, neg = split_balls(balls)
        w1, b1, w2, b2 = solve_twin_from_balls(pos, neg, c1, c2, arm)  # type: ignore[arg-type]
        return w1, b1, w2, b2, n_balls, True, None
    except (ValueError, np.linalg.LinAlgError) as exc:
        return (
            np.zeros(split.X_train.shape[1]),
            0.0,
            np.zeros(split.X_train.shape[1]),
            0.0,
            0,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def metrics_for_hyperplanes(
    split: DatasetSplit,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    rule: RuleName,
) -> dict[str, Any]:
    y_pred = predict_with_rule(split.X_test, w1, b1, w2, b2, rule)
    yt = normalize_labels(split.y_test)
    yp = np.where(y_pred == 0, -1, y_pred)
    macro = float(f1_score(yt, yp, average="macro", zero_division=0))
    acc = float(np.mean(yt == yp))
    bacc = float(balanced_accuracy_score(yt, yp))
    return {
        "macro_f1": macro,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "y_pred": y_pred,
    }


def run_rule_rescore_cell(
    split: DatasetSplit,
    seed: int,
    method: MethodArm,
    c1: float,
    c2: float,
) -> list[dict[str, Any]]:
    w1, b1, w2, b2, n_balls, ok, err = solve_a4_arm(split, seed, method, c1, c2)
    preds_by_rule: dict[RuleName, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for rule in (RULE_REPO, RULE_REF):
        if ok:
            m = metrics_for_hyperplanes(split, w1, b1, w2, b2, rule)
            macro = m["macro_f1"]
            preds_by_rule[rule] = m["y_pred"]
        else:
            macro = float("nan")
            preds_by_rule[rule] = np.zeros_like(split.y_test)
            m = {
                "macro_f1": float("nan"),
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
            }
        collapsed, creason = collapse_a4_definition(n_balls, ok, macro, method)
        rows.append(
            {
                "dataset": split.name,
                "seed": seed,
                "method": method,
                "prediction_rule": rule,
                "src_rule": RULE_TO_SRC[rule],
                "c1": float(c1),
                "c2": float(c2),
                "n_balls": n_balls,
                "classifier_ok": ok,
                "classifier_failure": int(not ok),
                "classifier_error": err or "",
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "collapsed": collapsed,
                "collapse_reason": creason,
                "finite_macro_f1": int(np.isfinite(macro)),
            }
        )
    if ok:
        dis = float(np.mean(preds_by_rule[RULE_REPO] != preds_by_rule[RULE_REF]))
    else:
        dis = float("nan")
    for r in rows:
        r["rule_disagreement_rate"] = dis
    return rows


def paired_macro_f1_deltas(
    df: "pd.DataFrame",
    method_a: str,
    method_b: str,
    rule: str,
) -> np.ndarray:
    import pandas as pd

    sub = df[(df["prediction_rule"] == rule)]
    wide = sub.pivot_table(
        index=["dataset", "seed"], columns="method", values="macro_f1"
    )
    if method_a not in wide.columns or method_b not in wide.columns:
        return np.array([])
    delta_seed = wide[method_a] - wide[method_b]
    return delta_seed.groupby(level=0).mean().to_numpy()


import pandas as pd  # noqa: E402
