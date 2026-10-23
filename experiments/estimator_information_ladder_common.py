"""
A4 — Estimator-information ladder (experiment-only; does not modify src/).

Twin LSTSVM normal equations follow src/ladder_gblstsvm.py assembly, but:
  - radius target shift is OFF for L0–L3 (information-preservation ladder)
  - RAW / L3 identity uses per-sample or exact ball moments

Prediction: RULE_REPO -> predict_twin_hyperplanes(..., rule='reference').
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

from experiments.granular_compression_k_common import COLLAPSE_BALL_THRESHOLD
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, DatasetSplit, normalize_labels
from experiments.rebaseline_common import C_GRID
from experiments.test_split_utility_granulation import _build_indexed_matrix
from experiments.validation_selected_k_common import make_train_val_partition
from src.gblstsvm import predict_twin_hyperplanes
from src.granular_balls import BallInfo, GBList, decompose_granular_balls
from src.ladder_gblstsvm import balls_from_samples, ridge_scale

RHO = 0.90
REG_BASE = 1e-4
PREDICTION_RULE = "reference"
PAPER_RULE_SENSITIVITY = "paper"

SEEDS = list(range(10))
IDENTITY_TOL = 2e-5
SCATTER_REL_TOL = 1e-5

AuditArm = Literal["L0", "L1", "L2", "L3"]
MethodArm = Literal[
    "RAW_LSTSVM",
    "L0_CENTER_UNIT",
    "L1_POPULATION",
    "L2_POPULATION_SCATTER",
    "L3_MOMENT_EXACT",
    "MATCHED_K_SAMPLES",
]

LADDER_ARMS: dict[MethodArm, AuditArm | None] = {
    "RAW_LSTSVM": "L3",
    "L0_CENTER_UNIT": "L0",
    "L1_POPULATION": "L1",
    "L2_POPULATION_SCATTER": "L2",
    "L3_MOMENT_EXACT": "L3",
    "MATCHED_K_SAMPLES": "L3",
}

MIN_BALL_COUNT_BASELINE = 2


@dataclass(frozen=True)
class ProtocolDoc:
    split: str = "70/15/15 stratified (phase8_common); test never used for tuning"
    granulation: str = "rho=0.90 recursive KMeans(k=2) split"
    baseline_path: str = "min_ball_count=2 deletion (baseline-compatible)"
    partition_path: str = "min_ball_count=0 (partition-preserving for identity)"
    radius_shift: str = "OFF for all ladder arms in this experiment"
    tuning: str = "C_GRID per arm on internal 20% train val; refit on full train"
    prediction: str = "RULE_REPO = reference = sign(|f2|-|f1|)"


PROTOCOL = ProtocolDoc()


def _ball_scatter(ball: BallInfo, dim: int) -> np.ndarray:
    if ball.scatter is not None:
        return np.asarray(ball.scatter, dtype=float)
    return np.zeros((dim, dim), dtype=float)


def accumulate_class_no_radius(
    balls: Sequence[BallInfo],
    arm: AuditArm,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class design (H) and rhs (g) without radius target shift."""
    if not balls:
        raise ValueError("accumulate_class requires at least one ball.")
    dim = int(balls[0].center.shape[0])
    design = np.zeros((dim + 1, dim + 1), dtype=float)
    rhs = np.zeros(dim + 1, dtype=float)
    for ball in balls:
        u = np.concatenate([np.asarray(ball.center, dtype=float), [1.0]])
        n_j = float(ball.num_samples)
        if arm == "L0":
            w = 1.0
            design += w * np.outer(u, u)
            rhs += w * u
        elif arm == "L1":
            design += n_j * np.outer(u, u)
            rhs += n_j * u
        elif arm in ("L2", "L3"):
            design += n_j * np.outer(u, u)
            design[:dim, :dim] += _ball_scatter(ball, dim)
            rhs += n_j * u
        else:
            raise ValueError(arm)
    return design, rhs


def solve_twin_from_balls(
    pos_balls: Sequence[BallInfo],
    neg_balls: Sequence[BallInfo],
    c1: float,
    c2: float,
    arm: AuditArm,
    reg_base: float = REG_BASE,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    if c1 <= 0 or c2 <= 0:
        raise ValueError("c1 and c2 must be positive.")
    design_a, rhs_a = accumulate_class_no_radius(pos_balls, arm)
    design_b, rhs_b = accumulate_class_no_radius(neg_balls, arm)
    dim = design_a.shape[0]
    eye = np.identity(dim, dtype=float)

    comb1 = design_b + (1.0 / c1) * design_a
    reg1 = ridge_scale(comb1, reg_base)
    inv_p_1 = np.linalg.inv(comb1 + reg1 * eye)
    hyper_p_1 = -inv_p_1 @ rhs_b.reshape(-1, 1)

    comb2 = design_a + (1.0 / c2) * design_b
    reg2 = ridge_scale(comb2, reg_base)
    inv_p_2 = np.linalg.inv(comb2 + reg2 * eye)
    hyper_p_2 = inv_p_2 @ rhs_a.reshape(-1, 1)

    w1 = hyper_p_1[:-1, 0]
    b1 = float(hyper_p_1[-1, 0])
    w2 = hyper_p_2[:-1, 0]
    b2 = float(hyper_p_2[-1, 0])
    return w1, b1, w2, b2


def split_balls(balls: list[BallInfo]) -> tuple[list[BallInfo], list[BallInfo]]:
    pos = [b for b in balls if b.label == 1.0]
    neg = [b for b in balls if b.label != 1.0]
    if not pos or not neg:
        raise ValueError("Need at least one ball per class.")
    return pos, neg


def _split_rho_granulation(
    indexed: np.ndarray,
    *,
    seed: int,
    purity: float,
    min_ball_count: int,
) -> list:
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < purity and ball.num > 1:
            left, right = ball.split_2balls()
            balls[i] = left
            balls.append(right)
            ll += 1
        else:
            i += 1
    if min_ball_count > 0:
        return [b for b in balls if int(b.num) >= min_ball_count]
    return balls


def build_baseline_rho90_balls(split: DatasetSplit, seed: int) -> list[BallInfo]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    kept = _split_rho_granulation(
        indexed, seed=seed, purity=RHO, min_ball_count=MIN_BALL_COUNT_BASELINE
    )
    return decompose_granular_balls(kept)


def build_partition_rho90_balls(split: DatasetSplit, seed: int) -> list[BallInfo]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    kept = _split_rho_granulation(indexed, seed=seed, purity=RHO, min_ball_count=0)
    return decompose_granular_balls(kept)


def partition_coverage(balls: list[BallInfo], n_train: int) -> tuple[bool, int]:
    covered = int(sum(b.num_samples for b in balls))
    return covered == n_train, covered


def class_moments_from_samples(X_class: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_class = np.asarray(X_class, dtype=float)
    if X_class.size == 0:
        d = X_class.shape[1] if X_class.ndim == 2 else 0
        p = d + 1
        return np.zeros((p, p)), np.zeros(p)
    aug = np.column_stack([X_class, np.ones(X_class.shape[0])])
    return aug.T @ aug, aug.sum(axis=0)


def class_moments_from_balls(balls: Sequence[BallInfo]) -> tuple[np.ndarray, np.ndarray]:
    if not balls:
        raise ValueError("empty balls")
    dim = int(balls[0].center.shape[0])
    H = np.zeros((dim + 1, dim + 1))
    g = np.zeros(dim + 1)
    for ball in balls:
        u = np.concatenate([ball.center, [1.0]])
        n_j = float(ball.num_samples)
        H += n_j * np.outer(u, u)
        H[:dim, :dim] += _ball_scatter(ball, dim)
        g += n_j * u
    return H, g


def _matrix_errors(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    abs_err = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(a, ord="fro"))
    rel_err = float(np.linalg.norm(diff, ord="fro") / (denom + 1e-12))
    return abs_err, rel_err


def verify_scatter_splits(indexed: np.ndarray, seed: int) -> tuple[bool, float]:
    """Check S_parent = S1+S2 + (n1*n2/n)(c1-c2)(c1-c2)^T on one split."""
    np.random.seed(seed)
    gbl = GBList(indexed)
    ball = gbl.granular_balls[0]
    if ball.num <= 1:
        return True, 0.0
    left, right = ball.split_2balls()
    max_err = 0.0
    abs_err = 0.0
    parent = ball
    c1, c2 = left, right
    feats_p = parent.data_no_label
    c_p = parent.center
    S_p = (feats_p - c_p).T @ (feats_p - c_p)
    n1, n2 = c1.num, c2.num
    if n1 > 0 and n2 > 0:
        f1, f2 = c1.data_no_label, c2.data_no_label
        cen1, cen2 = c1.center, c2.center
        S1 = (f1 - cen1).T @ (f1 - cen1)
        S2 = (f2 - cen2).T @ (f2 - cen2)
        dc = np.asarray(cen1 - cen2).reshape(-1, 1)
        recon = S1 + S2 + (n1 * n2 / parent.num) * (dc @ dc.T)
        abs_err = float(np.max(np.abs(S_p - recon)))
        rel_err = float(np.linalg.norm(S_p - recon) / (np.linalg.norm(S_p) + 1e-12))
        max_err = max(abs_err, rel_err)
    ok = max_err <= SCATTER_REL_TOL or abs_err <= 2e-5
    return ok, max_err


def run_moment_identity_audit(
    split: DatasetSplit,
    seed: int,
) -> dict[str, Any]:
    y = normalize_labels(split.y_train)
    X = split.X_train
    n_train = X.shape[0]
    balls = build_partition_rho90_balls(split, seed)
    complete, covered = partition_coverage(balls, n_train)
    indexed = _build_indexed_matrix(X, y)
    scatter_ok, scatter_err = verify_scatter_splits(indexed, seed)

    pos_b, neg_b = split_balls(balls)
    X_pos, X_neg = X[y == 1.0], X[y == -1.0]

    Hp_raw, gp_raw = class_moments_from_samples(X_pos)
    Hn_raw, gn_raw = class_moments_from_samples(X_neg)
    Hp_l3, gp_l3 = class_moments_from_balls(pos_b)
    Hn_l3, gn_l3 = class_moments_from_balls(neg_b)

    H_abs = max(
        _matrix_errors(Hp_raw, Hp_l3)[0],
        _matrix_errors(Hn_raw, Hn_l3)[0],
    )
    H_rel = max(
        _matrix_errors(Hp_raw, Hp_l3)[1],
        _matrix_errors(Hn_raw, Hn_l3)[1],
    )
    g_abs = max(
        float(np.max(np.abs(gp_raw - gp_l3))),
        float(np.max(np.abs(gn_raw - gn_l3))),
    )
    g_rel = max(
        float(np.linalg.norm(gp_raw - gp_l3) / (np.linalg.norm(gp_raw) + 1e-12)),
        float(np.linalg.norm(gn_raw - gn_l3) / (np.linalg.norm(gn_raw) + 1e-12)),
    )

    moment_ok = complete and H_abs <= IDENTITY_TOL and g_abs <= IDENTITY_TOL

    c1, c2 = DEFAULT_C1, DEFAULT_C2
    raw_pos, raw_neg = balls_from_samples(X, y)
    try:
        w_raw = solve_twin_from_balls(raw_pos, raw_neg, c1, c2, "L3")
        w_l3 = solve_twin_from_balls(pos_b, neg_b, c1, c2, "L3")
        param_abs = max(
            float(np.max(np.abs(w_raw[0] - w_l3[0]))),
            abs(w_raw[1] - w_l3[1]),
            float(np.max(np.abs(w_raw[2] - w_l3[2]))),
            abs(w_raw[3] - w_l3[3]),
        )
        preds_raw = predict_twin_hyperplanes(
            split.X_test, *w_raw, rule=PREDICTION_RULE
        )
        preds_l3 = predict_twin_hyperplanes(
            split.X_test, *w_l3, rule=PREDICTION_RULE
        )
        pred_dis = float(np.mean(preds_raw != preds_l3))
        f1_raw = _macro_f1(split.y_test, preds_raw)
        f1_l3 = _macro_f1(split.y_test, preds_l3)
    except (ValueError, np.linalg.LinAlgError) as exc:
        param_abs = float("nan")
        pred_dis = float("nan")
        f1_raw = f1_l3 = float("nan")
        moment_ok = False
        exc_msg = str(exc)
    else:
        exc_msg = ""

    identity_status = "PASS" if moment_ok and param_abs <= IDENTITY_TOL else "FAIL"
    if not moment_ok and param_abs > IDENTITY_TOL:
        identity_status = "IDENTITY_IMPLEMENTATION_FAILURE"

    return {
        "dataset": split.name,
        "seed": seed,
        "partition_complete": int(complete),
        "n_train": n_train,
        "n_covered": covered,
        "scatter_identity_pass": int(scatter_ok),
        "scatter_max_abs_error": scatter_err,
        "moment_identity_pass": int(moment_ok),
        "H_error_abs": H_abs,
        "H_error_rel": H_rel,
        "g_error_abs": g_abs,
        "g_error_rel": g_rel,
        "parameter_error_abs": param_abs,
        "prediction_disagreement": pred_dis,
        "macro_f1_raw": f1_raw,
        "macro_f1_l3": f1_l3,
        "macro_f1_diff": float(f1_l3 - f1_raw) if np.isfinite(f1_l3) and np.isfinite(f1_raw) else float("nan"),
        "accuracy_diff": float("nan"),
        "identity_status": identity_status,
        "error": exc_msg,
        "IDENTITY_CHECK": "PASS" if identity_status == "PASS" else "FAIL",
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = normalize_labels(y_true)
    yp = y_pred.copy()
    yp[yp == 0.0] = -1.0
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def _balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = normalize_labels(y_true)
    yp = y_pred.copy()
    yp[yp == 0.0] = -1.0
    return float(balanced_accuracy_score(yt, yp))


def tune_c1_c2(
    split: DatasetSplit,
    balls: list[BallInfo],
    arm: AuditArm,
    seed: int,
) -> tuple[float, float, float]:
    y_train = normalize_labels(split.y_train)
    partition = make_train_val_partition(split.name, seed, split.X_train, y_train)
    val_idx = partition.validation_indices
    X_val = split.X_train[val_idx]
    y_val = y_train[val_idx]
    pos, neg = split_balls(balls)
    best_c1, best_c2 = DEFAULT_C1, DEFAULT_C2
    best_f1 = -np.inf
    for c1 in C_GRID:
        for c2 in C_GRID:
            try:
                w1, b1, w2, b2 = solve_twin_from_balls(pos, neg, c1, c2, arm)
                preds = predict_twin_hyperplanes(X_val, w1, b1, w2, b2, rule=PREDICTION_RULE)
                f1 = _macro_f1(y_val, preds)
            except (ValueError, np.linalg.LinAlgError):
                f1 = float("nan")
            if np.isfinite(f1) and f1 > best_f1:
                best_f1 = f1
                best_c1, best_c2 = c1, c2
    return best_c1, best_c2, float(best_f1 if np.isfinite(best_f1) else float("nan"))


def matched_k_sample_indices(
    y_train: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Stratified subsample of size k (deterministic) for matched compression baseline."""
    y = normalize_labels(y_train)
    n = y.size
    k = min(k, n)
    rng = np.random.default_rng(seed + 99173)
    idx_pos = np.flatnonzero(y == 1.0)
    idx_neg = np.flatnonzero(y == -1.0)
    n_pos = idx_pos.size
    n_neg = idx_neg.size
    if n_pos == 0 or n_neg == 0:
        return rng.choice(n, size=k, replace=False)
    k_pos = max(1, int(round(k * n_pos / n)))
    k_pos = min(k_pos, n_pos)
    k_neg = min(k - k_pos, n_neg)
    if k_neg <= 0:
        k_neg = 1
        k_pos = k - 1
    chosen = np.concatenate(
        [
            rng.choice(idx_pos, size=k_pos, replace=False),
            rng.choice(idx_neg, size=k_neg, replace=False),
        ]
    )
    if chosen.size < k:
        rest = np.setdiff1d(np.arange(n), chosen)
        extra = rng.choice(rest, size=k - chosen.size, replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def evaluate_method(
    split: DatasetSplit,
    seed: int,
    method: MethodArm,
    *,
    tune: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    y_train = normalize_labels(split.y_train)
    n_train = int(y_train.size)
    d = int(split.X_train.shape[1])

    classifier_error = None
    classifier_ok = False
    n_balls = 0
    compression_ratio = float("nan")
    arm: AuditArm = "L3"
    balls: list[BallInfo] = []

    try:
        if method == "RAW_LSTSVM":
            pos, neg = balls_from_samples(split.X_train, y_train)
            n_balls = len(pos) + len(neg)
            balls = pos + neg
            arm: AuditArm = "L3"
            compression_ratio = 1.0
        elif method == "MATCHED_K_SAMPLES":
            baseline = build_baseline_rho90_balls(split, seed)
            k = max(len(baseline), 2)
            idx = matched_k_sample_indices(y_train, k, seed)
            X_sub = split.X_train[idx]
            y_sub = y_train[idx]
            pos, neg = balls_from_samples(X_sub, y_sub)
            balls = pos + neg
            n_balls = k
            arm = "L3"
            compression_ratio = k / n_train
        else:
            balls = build_baseline_rho90_balls(split, seed)
            n_balls = len(balls)
            arm_key = LADDER_ARMS[method]
            assert arm_key is not None
            arm = arm_key
            compression_ratio = n_balls / n_train

        if tune:
            c1, c2, val_f1 = tune_c1_c2(split, balls, arm, seed)
        else:
            c1, c2, val_f1 = DEFAULT_C1, DEFAULT_C2, float("nan")

        pos, neg = split_balls(balls)
        w1, b1, w2, b2 = solve_twin_from_balls(pos, neg, c1, c2, arm)
        y_pred = predict_twin_hyperplanes(
            split.X_test, w1, b1, w2, b2, rule=PREDICTION_RULE
        )
        classifier_ok = True
        status = "finite"
    except (ValueError, np.linalg.LinAlgError) as exc:
        c1, c2, val_f1 = DEFAULT_C1, DEFAULT_C2, float("nan")
        w1 = w2 = None
        b1 = b2 = 0.0
        y_pred = np.zeros_like(split.y_test)
        classifier_error = f"{type(exc).__name__}: {exc}"
        status = "classifier_failure"

    macro = _macro_f1(split.y_test, y_pred) if classifier_ok else float("nan")
    acc = (
        float(np.mean(normalize_labels(split.y_test) == np.where(y_pred == 0, -1, y_pred)))
        if classifier_ok
        else float("nan")
    )
    bacc = _balanced_acc(split.y_test, y_pred) if classifier_ok else float("nan")

    collapsed = int(
        (not classifier_ok)
        or (not np.isfinite(macro))
        or (n_balls <= COLLAPSE_BALL_THRESHOLD and method != "RAW_LSTSVM")
        or (len(np.unique(np.where(y_pred == 0, -1, y_pred))) == 1)
    )
    collapse_reason = ""
    if not classifier_ok:
        collapse_reason = classifier_error or "classifier_failed"
    elif not np.isfinite(macro):
        collapse_reason = "invalid_macro_f1"
    elif n_balls <= COLLAPSE_BALL_THRESHOLD and method != "RAW_LSTSVM":
        collapse_reason = f"n_balls<={COLLAPSE_BALL_THRESHOLD}"
    elif len(np.unique(np.where(y_pred == 0, -1, y_pred))) == 1:
        collapse_reason = "prediction_collapse"

    storage_k_d2 = n_balls * (d + 1) ** 2
    storage_n_d = n_train * d

    return {
        "dataset": split.name,
        "seed": seed,
        "method": method,
        "arm_internal": arm if classifier_ok or method != "RAW_LSTSVM" else "L3",
        "c1": float(c1),
        "c2": float(c2),
        "validation_macro_f1": float(val_f1),
        "accuracy": acc,
        "macro_f1": macro,
        "balanced_accuracy": bacc,
        "n_train": n_train,
        "n_features": d,
        "n_balls": n_balls,
        "compression_ratio": compression_ratio,
        "storage_n_d": storage_n_d,
        "storage_k_d2": storage_k_d2,
        "storage_ratio_kd2_over_nd": storage_k_d2 / storage_n_d if storage_n_d else float("nan"),
        "classifier_ok": classifier_ok,
        "status": status,
        "collapsed": collapsed,
        "collapse_reason": collapse_reason,
        "runtime_seconds": time.perf_counter() - t0,
        "prediction_rule": PREDICTION_RULE,
        "radius_shift": "absent",
    }


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
            "std_delta": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
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
        out["cohens_d"] = float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
    return out


def paired_comparison(df: "pd.DataFrame", a: str, b: str) -> dict[str, Any]:
    wide = df.pivot_table(index=["dataset", "seed"], columns="method", values="macro_f1")
    if a not in wide.columns or b not in wide.columns:
        return {"comparison": f"{a}_minus_{b}", "n_pairs": 0}
    delta_seed = wide[a] - wide[b]
    ds = delta_seed.groupby(level=0).mean()
    stats_row = dataset_level_stats(ds.to_numpy())
    return {"comparison": f"{a}_minus_{b}", **stats_row}


import pandas as pd  # noqa: E402
