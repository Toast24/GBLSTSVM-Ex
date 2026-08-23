"""
Ball-level margin-based reliability uncertainty (BMRU) weights.

BMRU down-weights granular balls whose spatial extent is large relative to
their distance from the twin-hyperplane classification decision boundary.

**FROZEN (Stages 4–6).** This module is frozen for research reproducibility.
The formulation u_i = 2 r_i / (|s_i| + epsilon), mu_i = exp(-alpha * u_i)
must not be altered without a new research phase. Boundary-straddling
diagnostics (2026-08) falsified the motivating hypothesis; see
docs/BMRU_GBLSTSVM.md §15–16. No replacement formula is implemented.

See docs/BMRU_GBLSTSVM.md for motivation, derivation notes, and diagnostic results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from .granular_balls import BallInfo

MarginMode = Literal["reference", "paper"]

DEFAULT_ALPHA = 1.0
DEFAULT_EPSILON = 1e-6
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_WEIGHT_TOLERANCE = 1e-8


def _safe_norm(w: np.ndarray) -> float:
    n = float(np.linalg.norm(np.asarray(w, dtype=float)))
    return n if n > 0.0 else 1.0


def hyperplane_function_values(
    center: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
) -> tuple[float, float]:
    """Return f1(c) and f2(c) for one ball center."""
    c = np.asarray(center, dtype=float).ravel()
    f1 = float(c @ np.asarray(w1, dtype=float) + b1)
    f2 = float(c @ np.asarray(w2, dtype=float) + b2)
    return f1, f2


def decision_margin_at_center(
    center: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    *,
    margin_mode: MarginMode = "reference",
) -> tuple[float, float, float]:
    """
    Compute the decision-margin quantity s_i at a ball center.

    reference (repository prediction rule):
        s = |f2(c)| - |f1(c)|   boundary at s = 0

    paper (normalized twin distance rule):
        s = d1(c) - d2(c)       boundary at s = 0
        d_k = |f_k(c)| / ||w_k||

    Returns
    -------
    s, d1_or_abs_f1, d2_or_abs_f2
        For reference mode the last two values are |f1|, |f2|.
        For paper mode they are normalized distances d1, d2.
    """
    f1, f2 = hyperplane_function_values(center, w1, b1, w2, b2)
    abs_f1 = abs(f1)
    abs_f2 = abs(f2)

    if margin_mode == "reference":
        return abs_f2 - abs_f1, abs_f1, abs_f2

    if margin_mode == "paper":
        d1 = abs_f1 / _safe_norm(w1)
        d2 = abs_f2 / _safe_norm(w2)
        return d1 - d2, d1, d2

    raise ValueError(f"Unknown margin_mode: {margin_mode!r}")


def bmru_uncertainty(
    radius: float,
    margin: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """
    u_i = 2 r_i / (|s_i| + epsilon)

    Uses raw ball radius (frozen mean-distance definition).
    """
    r = float(radius)
    if r < 0.0:
        raise ValueError("radius must be non-negative.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if r == 0.0:
        return 0.0
    return (2.0 * r) / (abs(float(margin)) + epsilon)


def bmru_weight(
    radius: float,
    margin: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[float, float]:
    """
    Compute (u_i, mu_i) with mu_i = exp(-alpha * u_i).

    Higher mu => more reliable / confident ball.
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    u = bmru_uncertainty(radius, margin, epsilon=epsilon)
    mu = float(np.exp(-alpha * u))
    if not np.isfinite(mu):
        raise FloatingPointError("BMRU weight overflow/underflow.")
    return u, mu


def ball_bmru_weights(
    balls: list[BallInfo],
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    margin_mode: MarginMode = "reference",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute BMRU weights aligned with ``balls_to_class_arrays`` row order.

    Returns
    -------
    weights_pos, weights_neg, u_all, mu_all, margins_all
        u_all, mu_all, margins_all follow ``balls`` list order.
    """
    u_all = np.zeros(len(balls), dtype=float)
    mu_all = np.zeros(len(balls), dtype=float)
    margins_all = np.zeros(len(balls), dtype=float)

    for i, ball in enumerate(balls):
        s, _, _ = decision_margin_at_center(
            ball.center, w1, b1, w2, b2, margin_mode=margin_mode
        )
        u, mu = bmru_weight(ball.radius, s, alpha=alpha, epsilon=epsilon)
        u_all[i] = u
        mu_all[i] = mu
        margins_all[i] = s

    weights_pos = np.array(
        [mu_all[i] for i, b in enumerate(balls) if b.label == 1],
        dtype=float,
    )
    weights_neg = np.array(
        [mu_all[i] for i, b in enumerate(balls) if b.label != 1],
        dtype=float,
    )
    return weights_pos, weights_neg, u_all, mu_all, margins_all


@dataclass
class BMRUIterationRecord:
    iteration: int
    max_abs_weight_change: float
    mean_weight: float
    min_weight: float
    max_weight: float
    mean_u: float
    median_u: float
    min_u: float
    max_u: float
    n_u_lt_1: int
    n_u_ge_1: int
    w1_norm_change: float
    w2_norm_change: float
    b1_abs_change: float
    b2_abs_change: float
    cond_hp1: float
    cond_hp2: float


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN when undefined (zero variance or <2 points)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_bmru_iteration(
    mu_all: np.ndarray,
    u_all: np.ndarray,
    iteration: int,
    previous_mu: np.ndarray | None = None,
    *,
    w1_prev: np.ndarray | None = None,
    b1_prev: float | None = None,
    w2_prev: np.ndarray | None = None,
    b2_prev: float | None = None,
    w1_new: np.ndarray | None = None,
    b1_new: float | None = None,
    w2_new: np.ndarray | None = None,
    b2_new: float | None = None,
    cond_hp1: float = float("nan"),
    cond_hp2: float = float("nan"),
) -> BMRUIterationRecord:
    mu = np.asarray(mu_all, dtype=float).ravel()
    u = np.asarray(u_all, dtype=float).ravel()
    if previous_mu is None:
        max_change = float("inf")
    else:
        prev = np.asarray(previous_mu, dtype=float).ravel()
        max_change = float(np.max(np.abs(mu - prev))) if mu.size else 0.0

    if w1_prev is not None and w1_new is not None:
        w1_change = float(np.linalg.norm(np.asarray(w1_new) - np.asarray(w1_prev)))
        w2_change = float(np.linalg.norm(np.asarray(w2_new) - np.asarray(w2_prev)))
        b1_change = float(abs(b1_new - b1_prev))
        b2_change = float(abs(b2_new - b2_prev))
    else:
        w1_change = w2_change = b1_change = b2_change = float("nan")

    return BMRUIterationRecord(
        iteration=iteration,
        max_abs_weight_change=max_change,
        mean_weight=float(np.mean(mu)) if mu.size else 1.0,
        min_weight=float(np.min(mu)) if mu.size else 1.0,
        max_weight=float(np.max(mu)) if mu.size else 1.0,
        mean_u=float(np.mean(u)) if u.size else 0.0,
        median_u=float(np.median(u)) if u.size else 0.0,
        min_u=float(np.min(u)) if u.size else 0.0,
        max_u=float(np.max(u)) if u.size else 0.0,
        n_u_lt_1=int(np.sum(u < 1.0)),
        n_u_ge_1=int(np.sum(u >= 1.0)),
        w1_norm_change=w1_change,
        w2_norm_change=w2_change,
        b1_abs_change=b1_change,
        b2_abs_change=b2_change,
        cond_hp1=cond_hp1,
        cond_hp2=cond_hp2,
    )


def bmru_diagnostics_summary(
    balls: list[BallInfo],
    u_all: np.ndarray,
    mu_all: np.ndarray,
    *,
    reliabilities: np.ndarray | None = None,
    class_weights: np.ndarray | None = None,
) -> dict:
    """Aggregate diagnostics for experiment reporting."""
    radii = np.array([b.radius for b in balls], dtype=float)
    purities = np.array([b.purity for b in balls], dtype=float)
    u = np.asarray(u_all, dtype=float).ravel()
    mu = np.asarray(mu_all, dtype=float).ravel()

    summary = {
        "n_balls": len(balls),
        "mean_radius": float(np.mean(radii)) if radii.size else 0.0,
        "median_radius": float(np.median(radii)) if radii.size else 0.0,
        "mean_purity": float(np.mean(purities)) if purities.size else 0.0,
        "mean_u": float(np.mean(u)) if u.size else 0.0,
        "median_u": float(np.median(u)) if u.size else 0.0,
        "min_u": float(np.min(u)) if u.size else 0.0,
        "max_u": float(np.max(u)) if u.size else 0.0,
        "mean_bmru_weight": float(np.mean(mu)) if mu.size else 1.0,
        "median_bmru_weight": float(np.median(mu)) if mu.size else 1.0,
        "min_bmru_weight": float(np.min(mu)) if mu.size else 1.0,
        "max_bmru_weight": float(np.max(mu)) if mu.size else 1.0,
        "frac_u_lt_1": float(np.mean(u < 1.0)) if u.size else 0.0,
        "frac_u_ge_1": float(np.mean(u >= 1.0)) if u.size else 0.0,
        "frac_mu_lt_0_1": float(np.mean(mu < 0.1)) if mu.size else 0.0,
        "frac_weights_approx_1": float(np.mean(np.abs(mu - 1.0) < 1e-6)) if mu.size else 1.0,
        "frac_very_small_weights": float(np.mean(mu < 0.1)) if mu.size else 0.0,
        "corr_purity_bmru_weight": safe_correlation(purities, mu),
        "corr_radius_bmru_weight": safe_correlation(radii, mu),
    }
    if reliabilities is not None:
        rel = np.asarray(reliabilities, dtype=float).ravel()
        summary["corr_reliability_bmru_weight"] = safe_correlation(rel, mu)
    if class_weights is not None:
        cw = np.asarray(class_weights, dtype=float).ravel()
        summary["corr_class_weight_bmru_weight"] = safe_correlation(cw, mu)
    return summary


def margin_ablation_weight_stats(u_all: np.ndarray, mu_all: np.ndarray) -> dict:
    """u/mu distribution stats for margin-mode ablation."""
    u = np.asarray(u_all, dtype=float).ravel()
    mu = np.asarray(mu_all, dtype=float).ravel()
    return {
        "mean_u": float(np.mean(u)) if u.size else 0.0,
        "median_u": float(np.median(u)) if u.size else 0.0,
        "min_u": float(np.min(u)) if u.size else 0.0,
        "max_u": float(np.max(u)) if u.size else 0.0,
        "frac_u_lt_1": float(np.mean(u < 1.0)) if u.size else 0.0,
        "frac_u_ge_1": float(np.mean(u >= 1.0)) if u.size else 0.0,
        "mean_mu": float(np.mean(mu)) if mu.size else 1.0,
        "median_mu": float(np.median(mu)) if mu.size else 1.0,
        "min_mu": float(np.min(mu)) if mu.size else 1.0,
        "max_mu": float(np.max(mu)) if mu.size else 1.0,
        "frac_mu_lt_0_1": float(np.mean(mu < 0.1)) if mu.size else 0.0,
        "frac_mu_approx_1": float(np.mean(np.abs(mu - 1.0) < 1e-6)) if mu.size else 1.0,
    }


def prediction_distribution(y_pred: np.ndarray) -> dict:
    """Count positive/negative predictions for collapse detection."""
    y = np.asarray(y_pred, dtype=float).ravel()
    y_eval = y.copy()
    y_eval[y_eval == 0.0] = -1.0
    n_pos = int(np.sum(y_eval == 1.0))
    n_neg = int(np.sum(y_eval == -1.0))
    n_total = int(y_eval.size)
    return {
        "n_predictions": n_total,
        "n_positive_predictions": n_pos,
        "n_negative_predictions": n_neg,
        "positive_prediction_fraction": float(n_pos / n_total) if n_total else float("nan"),
        "prediction_collapse": n_pos == 0 or n_neg == 0,
    }


def _weight_mass_stats(weights: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "sum": 0.0,
            "frac_lt_0_1": float("nan"),
            "frac_approx_1": float("nan"),
        }
    return {
        "mean": float(np.mean(w)),
        "median": float(np.median(w)),
        "sum": float(np.sum(w)),
        "frac_lt_0_1": float(np.mean(w < 0.1)),
        "frac_approx_1": float(np.mean(np.abs(w - 1.0) < 1e-6)),
    }


def class_weight_mass_diagnostics(
    balls: list[BallInfo],
    mu_all: np.ndarray,
    combined_weights: np.ndarray,
) -> list[dict]:
    """
    Per-class weight-mass diagnostics.

    combined_weights must follow ``balls`` list order (one entry per ball).
    """
    mu = np.asarray(mu_all, dtype=float).ravel()
    combined = np.asarray(combined_weights, dtype=float).ravel()
    rows = []
    for class_label in (1.0, -1.0):
        mask = np.array([b.label == class_label for b in balls], dtype=bool)
        mu_cls = mu[mask]
        comb_cls = combined[mask]
        mu_stats = _weight_mass_stats(mu_cls)
        comb_stats = _weight_mass_stats(comb_cls)
        rows.append(
            {
                "class_label": float(class_label),
                "number_of_balls": int(mask.sum()),
                "mean_bmru_weight": mu_stats["mean"],
                "median_bmru_weight": mu_stats["median"],
                "sum_bmru_weight": mu_stats["sum"],
                "mean_combined_weight": comb_stats["mean"],
                "median_combined_weight": comb_stats["median"],
                "sum_combined_weight": comb_stats["sum"],
                "frac_bmru_weight_lt_0_1": mu_stats["frac_lt_0_1"],
                "frac_bmru_weight_approx_1": mu_stats["frac_approx_1"],
                "frac_combined_weight_lt_0_1": comb_stats["frac_lt_0_1"],
                "frac_combined_weight_approx_1": comb_stats["frac_approx_1"],
            }
        )
    return rows


def combined_weights_per_ball(model) -> np.ndarray:
    """Final training weight per ball in ``model.balls_`` list order."""
    weights = np.zeros(len(model.balls_), dtype=float)
    pos_idx = neg_idx = 0
    for i, ball in enumerate(model.balls_):
        if ball.label == 1:
            weights[i] = model.weights_pos_[pos_idx]
            pos_idx += 1
        else:
            weights[i] = model.weights_neg_[neg_idx]
            neg_idx += 1
    return weights


DEFAULT_BOUNDARY_TOLERANCE = DEFAULT_EPSILON
ZERO_TOLERANCE = 1e-8


def decision_scores_at_points(
    X: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    *,
    margin_mode: MarginMode = "reference",
) -> np.ndarray:
    """
    Repository decision score s(x) for each row of X.

    reference: s(x) = |f2(x)| - |f1(x)|
    paper: s(x) = d1(x) - d2(x)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    f1 = X @ np.asarray(w1, dtype=float) + b1
    f2 = X @ np.asarray(w2, dtype=float) + b2
    abs_f1 = np.abs(f1)
    abs_f2 = np.abs(f2)

    if margin_mode == "reference":
        return abs_f2 - abs_f1

    if margin_mode == "paper":
        d1 = abs_f1 / _safe_norm(w1)
        d2 = abs_f2 / _safe_norm(w2)
        return d1 - d2

    raise ValueError(f"Unknown margin_mode: {margin_mode!r}")


def center_distances_to_members(
    center: np.ndarray,
    X_members: np.ndarray,
) -> np.ndarray:
    """Euclidean distances from ball center to member feature rows."""
    center = np.asarray(center, dtype=float).ravel()
    X_members = np.asarray(X_members, dtype=float)
    diff = X_members - center
    return np.sqrt(np.sum(np.square(diff), axis=1))


def ball_boundary_diagnostics(
    ball: BallInfo,
    member_indices: np.ndarray,
    X: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    u: float,
    mu: float,
    *,
    margin_mode: MarginMode = "reference",
    epsilon: float = DEFAULT_EPSILON,
    boundary_near_threshold: float = DEFAULT_BOUNDARY_TOLERANCE,
    zero_tolerance: float = ZERO_TOLERANCE,
) -> dict:
    """
    Per-ball empirical boundary diagnostics for BMRU falsification.

    Uses the same hyperplanes and margin definition as BMRU weighting.
    """
    member_indices = np.asarray(member_indices, dtype=int).ravel()
    X_members = X[member_indices]
    scores = decision_scores_at_points(
        X_members, w1, b1, w2, b2, margin_mode=margin_mode
    )
    signed = np.asarray(scores, dtype=float).ravel()
    abs_scores = np.abs(signed)

    center_score = float(
        decision_scores_at_points(
            ball.center, w1, b1, w2, b2, margin_mode=margin_mode
        )[0]
    )
    center_abs_margin = abs(center_score)
    min_abs_sample_margin = float(np.min(abs_scores)) if abs_scores.size else float("nan")
    median_abs_sample_margin = float(np.median(abs_scores)) if abs_scores.size else float("nan")
    max_abs_sample_margin = float(np.max(abs_scores)) if abs_scores.size else float("nan")
    min_signed_sample_margin = float(np.min(signed)) if signed.size else float("nan")
    max_signed_sample_margin = float(np.max(signed)) if signed.size else float("nan")

    distances = center_distances_to_members(ball.center, X_members)
    mean_center_distance = float(ball.radius)
    max_center_distance = float(np.max(distances)) if distances.size else 0.0
    median_center_distance = float(np.median(distances)) if distances.size else 0.0
    max_radius_ratio = max_center_distance / (mean_center_distance + epsilon)

    boundary_straddles = (
        min_signed_sample_margin <= zero_tolerance
        and max_signed_sample_margin >= -zero_tolerance
    )
    boundary_near = min_abs_sample_margin <= boundary_near_threshold

    n_near = int(np.sum(abs_scores <= boundary_near_threshold))
    n_positive = int(np.sum(signed > zero_tolerance))
    n_negative = int(np.sum(signed < -zero_tolerance))

    margin_reduction = center_abs_margin - min_abs_sample_margin

    return {
        "class_label": float(ball.label),
        "center": ball.center.tolist(),
        "radius": float(ball.radius),
        "purity": float(ball.purity),
        "u": float(u),
        "mu": float(mu),
        "min_abs_sample_margin": min_abs_sample_margin,
        "median_abs_sample_margin": median_abs_sample_margin,
        "max_abs_sample_margin": max_abs_sample_margin,
        "sample_margin_at_center": center_abs_margin,
        "min_signed_sample_margin": min_signed_sample_margin,
        "max_signed_sample_margin": max_signed_sample_margin,
        "n_samples": int(member_indices.size),
        "n_samples_near_boundary": n_near,
        "n_samples_positive_side": n_positive,
        "n_samples_negative_side": n_negative,
        "mean_center_distance": mean_center_distance,
        "max_center_distance": max_center_distance,
        "median_center_distance": median_center_distance,
        "max_radius_ratio": max_radius_ratio,
        "center_abs_margin": center_abs_margin,
        "min_abs_margin": min_abs_sample_margin,
        "margin_reduction": margin_reduction,
        "boundary_straddles": bool(boundary_straddles),
        "boundary_near": bool(boundary_near),
    }


def summarize_boundary_groups(per_ball_rows: list[dict]) -> list[dict]:
    """Aggregate boundary metrics by u and mu groups."""
    summaries: list[dict] = []

    def _summarize(rows: list[dict], group_name: str) -> dict:
        return {
            "group": group_name,
            "number_of_balls": len(rows),
            "fraction_boundary_straddling": float(
                np.mean([r["boundary_straddles"] for r in rows])
            ),
            "fraction_boundary_near": float(np.mean([r["boundary_near"] for r in rows])),
            "mean_radius": float(np.mean([r["radius"] for r in rows])),
            "mean_max_center_distance": float(
                np.mean([r["max_center_distance"] for r in rows])
            ),
            "mean_u": float(np.mean([r["u"] for r in rows])),
            "mean_mu": float(np.mean([r["mu"] for r in rows])),
            "mean_center_abs_margin": float(
                np.mean([r["center_abs_margin"] for r in rows])
            ),
            "mean_min_abs_sample_margin": float(
                np.mean([r["min_abs_sample_margin"] for r in rows])
            ),
        }

    u_groups = {
        "u_lt_1": [r for r in per_ball_rows if r["u"] < 1.0],
        "u_ge_1": [r for r in per_ball_rows if r["u"] >= 1.0],
    }
    mu_groups = {
        "mu_lt_0_1": [r for r in per_ball_rows if r["mu"] < 0.1],
        "mu_ge_0_1": [r for r in per_ball_rows if r["mu"] >= 0.1],
    }

    for name, rows in u_groups.items():
        if rows:
            summaries.append(_summarize(rows, name))
    for name, rows in mu_groups.items():
        if rows:
            summaries.append(_summarize(rows, name))
    return summaries


def boundary_u_detector_confusion(
    per_ball_rows: list[dict],
    *,
    empirical_field: str,
    empirical_label: str,
) -> dict:
    """Mechanism diagnostic: u >= 1 as uncertainty detector."""
    flagged: list[dict] = []
    for row in per_ball_rows:
        flagged.append(
            {
                **row,
                "u_ge_1": row["u"] >= 1.0,
            }
        )
    return boundary_mechanism_confusion(
        flagged,
        detector_field="u_ge_1",
        empirical_field=empirical_field,
        detector_label="u >= 1",
        empirical_label=empirical_label,
    )


def boundary_correlations(per_ball_rows: list[dict]) -> dict:
    """Correlations between BMRU quantities and empirical boundary metrics."""
    if not per_ball_rows:
        return {}

    u = np.array([r["u"] for r in per_ball_rows], dtype=float)
    mu = np.array([r["mu"] for r in per_ball_rows], dtype=float)
    min_abs = np.array([r["min_abs_sample_margin"] for r in per_ball_rows], dtype=float)
    max_dist = np.array([r["max_center_distance"] for r in per_ball_rows], dtype=float)
    radius = np.array([r["radius"] for r in per_ball_rows], dtype=float)
    purity = np.array([r["purity"] for r in per_ball_rows], dtype=float)
    straddles = np.array(
        [1.0 if r["boundary_straddles"] else 0.0 for r in per_ball_rows], dtype=float
    )

    return {
        "corr_u_min_abs_sample_margin": safe_correlation(u, min_abs),
        "corr_mu_min_abs_sample_margin": safe_correlation(mu, min_abs),
        "corr_u_boundary_straddles": safe_correlation(u, straddles),
        "corr_u_max_center_distance": safe_correlation(u, max_dist),
        "corr_radius_min_abs_sample_margin": safe_correlation(radius, min_abs),
        "corr_u_purity": safe_correlation(u, purity),
    }


def boundary_mechanism_confusion(
    per_ball_rows: list[dict],
    *,
    detector_field: str,
    empirical_field: str,
    detector_label: str,
    empirical_label: str,
) -> dict:
    """
    Confusion-style table for BMRU mechanism diagnostics (not ML classifier metrics).

    ``detector_field`` True => flagged uncertain (e.g. u >= 1).
  """
    tp = fp = tn = fn = 0
    for row in per_ball_rows:
        detected = bool(row[detector_field])
        empirical = bool(row[empirical_field])
        if detected and empirical:
            tp += 1
        elif detected and not empirical:
            fp += 1
        elif not detected and not empirical:
            tn += 1
        else:
            fn += 1

    n_pos = tp + fp
    n_emp_pos = tp + fn
    precision = tp / n_pos if n_pos else float("nan")
    recall = tp / n_emp_pos if n_emp_pos else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    return {
        "detector": detector_label,
        "empirical_condition": empirical_label,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_mechanism": precision,
        "recall_mechanism": recall,
        "false_positive_rate_mechanism": fpr,
        "false_negative_rate_mechanism": fnr,
        "u_lt_1_empirical_no": tn,
        "u_lt_1_empirical_yes": fn,
        "u_ge_1_empirical_no": fp,
        "u_ge_1_empirical_yes": tp,
    }


def per_ball_boundary_records(
    balls: list[BallInfo],
    memberships: list[np.ndarray],
    X: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    u_all: np.ndarray,
    mu_all: np.ndarray,
    *,
    margin_mode: MarginMode = "reference",
    epsilon: float = DEFAULT_EPSILON,
    boundary_near_threshold: float = DEFAULT_BOUNDARY_TOLERANCE,
    metadata: dict | None = None,
) -> list[dict]:
    """Compute per-ball boundary diagnostics aligned with ``balls`` order."""
    rows: list[dict] = []
    for i, ball in enumerate(balls):
        row = ball_boundary_diagnostics(
            ball,
            memberships[i],
            X,
            w1,
            b1,
            w2,
            b2,
            float(u_all[i]),
            float(mu_all[i]),
            margin_mode=margin_mode,
            epsilon=epsilon,
            boundary_near_threshold=boundary_near_threshold,
        )
        row["ball_id"] = i
        if metadata:
            row.update(metadata)
        rows.append(row)
    return rows


def iteration_record_to_dict(rec: BMRUIterationRecord) -> dict:
    return {
        "iteration": rec.iteration,
        "max_abs_weight_change": rec.max_abs_weight_change,
        "mean_weight": rec.mean_weight,
        "min_weight": rec.min_weight,
        "max_weight": rec.max_weight,
        "mean_u": rec.mean_u,
        "median_u": rec.median_u,
        "min_u": rec.min_u,
        "max_u": rec.max_u,
        "n_u_lt_1": rec.n_u_lt_1,
        "n_u_ge_1": rec.n_u_ge_1,
        "w1_norm_change": rec.w1_norm_change,
        "w2_norm_change": rec.w2_norm_change,
        "b1_abs_change": rec.b1_abs_change,
        "b2_abs_change": rec.b2_abs_change,
        "cond_hp1": rec.cond_hp1,
        "cond_hp2": rec.cond_hp2,
    }
