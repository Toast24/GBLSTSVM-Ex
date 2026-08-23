"""
Inter-class overlap proxy weights for OBRU-GBLSTSVM (Stages 7–9).

Uses frozen ball center, mean-distance radius, and label only.
O_ij is an overlap PROXY, not guaranteed geometric intersection.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .class_weights import ball_class_weights
from .granular_balls import BallInfo
from .reliability import (
    DEFAULT_EPSILON as REL_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_R,
    ReliabilityVariant,
    ball_reliability_weights,
)

DEFAULT_ALPHA = 1.0
DEFAULT_EPSILON = 1e-6
DEFAULT_TOP_K = 3


def minority_label_from_training(y_train: np.ndarray) -> float:
    """Return +1 or -1 label of the minority class from training counts."""
    y = np.asarray(y_train, dtype=float).ravel().copy()
    y[y == 0] = -1.0
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == -1.0))
    return 1.0 if n_pos <= n_neg else -1.0


def pair_overlap_proxy(
    r_i: float,
    r_j: float,
    d_ij: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """
    Inter-class overlap proxy for one opposite-class pair.

    O_ij = max(0, (r_i + r_j - d_ij) / (r_i + r_j + epsilon))
    """
    ri = float(r_i)
    rj = float(r_j)
    d = float(d_ij)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    denom = ri + rj + epsilon
    return float(max(0.0, (ri + rj - d) / denom))


def aggregate_top_k_overlaps(
    overlaps: np.ndarray,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> float:
    """
    Mean of the top-k overlap values.

    If fewer than top_k values exist, average all available values.
    If no values exist, return 0.
    """
    values = np.asarray(overlaps, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    k = min(int(top_k), values.size)
    top = np.partition(values, -k)[-k:]
    return float(np.mean(top))


def ball_overlap_scores(
    balls: list[BallInfo],
    *,
    epsilon: float = DEFAULT_EPSILON,
    top_k: int = DEFAULT_TOP_K,
) -> np.ndarray:
    """
    Compute O_i for each ball using top-k mean among opposite-class pairs.
    """
    n = len(balls)
    if n == 0:
        return np.array([], dtype=float)

    centers = np.array([b.center for b in balls], dtype=float)
    radii = np.array([b.radius for b in balls], dtype=float)
    labels = np.array([b.label for b in balls], dtype=float)
    O = np.zeros(n, dtype=float)

    for i in range(n):
        opp_mask = (labels != labels[i]) & (np.arange(n) != i)
        opp_idx = np.flatnonzero(opp_mask)
        if opp_idx.size == 0:
            O[i] = 0.0
            continue
        dists = np.linalg.norm(centers[i] - centers[opp_idx], axis=1)
        overlaps = np.array(
            [
                pair_overlap_proxy(radii[i], radii[j], d, epsilon=epsilon)
                for j, d in zip(opp_idx, dists)
            ],
            dtype=float,
        )
        O[i] = aggregate_top_k_overlaps(overlaps, top_k=top_k)
    return O


def overlap_mu_from_scores(
    balls: list[BallInfo],
    overlap_scores: np.ndarray,
    minority_label: float,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """
    Class-conditional overlap weights.

    Minority: mu_i = 1 + alpha * O_i
    Majority: mu_i = 1 - alpha * O_i
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    O = np.asarray(overlap_scores, dtype=float).reshape(-1)
    if O.shape[0] != len(balls):
        raise ValueError("overlap_scores length must match number of balls.")

    mu = np.zeros(len(balls), dtype=float)
    for i, ball in enumerate(balls):
        if ball.label == minority_label:
            mu[i] = 1.0 + alpha * O[i]
        else:
            mu[i] = 1.0 - alpha * O[i]
    return mu


def validate_overlap_weights(mu: np.ndarray) -> None:
    """Raise on non-finite or negative training weights (no clipping)."""
    mu = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu)):
        raise FloatingPointError("OBRU weight overflow/underflow.")
    if np.any(mu < 0.0):
        raise ValueError(
            "OBRU weight became negative "
            f"(min mu={float(np.min(mu))}). No clipping applied."
        )


def split_mu_by_class(
    balls: list[BallInfo],
    mu_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split mu aligned with ``balls_to_class_arrays`` row order."""
    mu_all = np.asarray(mu_all, dtype=float).reshape(-1)
    weights_pos = np.array(
        [mu_all[i] for i, b in enumerate(balls) if b.label == 1], dtype=float
    )
    weights_neg = np.array(
        [mu_all[i] for i, b in enumerate(balls) if b.label != 1], dtype=float
    )
    return weights_pos, weights_neg


def ball_obru_weights(
    balls: list[BallInfo],
    minority_label: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OBRU-only weights: W_i = mu_i.

    Returns weights_pos, weights_neg, overlap_all, mu_all.
    """
    O_all = ball_overlap_scores(balls, epsilon=epsilon, top_k=top_k)
    mu_all = overlap_mu_from_scores(
        balls, O_all, minority_label, alpha=alpha
    )
    validate_overlap_weights(mu_all)
    weights_pos, weights_neg = split_mu_by_class(balls, mu_all)
    return weights_pos, weights_neg, O_all, mu_all


def ball_cw_obru_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    minority_label: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combined CW + OBRU: W_i = kappa_{y_i} * mu_i."""
    kappa_pos_balls, kappa_neg_balls = ball_class_weights(
        balls, kappa_pos, kappa_neg
    )
    obru_pos, obru_neg, O_all, mu_all = ball_obru_weights(
        balls,
        minority_label,
        alpha=alpha,
        epsilon=epsilon,
        top_k=top_k,
    )
    weights_pos = kappa_pos_balls * obru_pos
    weights_neg = kappa_neg_balls * obru_neg
    return (
        weights_pos,
        weights_neg,
        O_all,
        mu_all,
        kappa_pos_balls,
        kappa_neg_balls,
    )


def ball_cr_obru_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    minority_label: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    top_k: int = DEFAULT_TOP_K,
    rel_epsilon: float = REL_EPSILON,
    gamma: float = DEFAULT_GAMMA,
    lambda_r: float = DEFAULT_LAMBDA_R,
    variant: ReliabilityVariant = "full",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Combined CW + R + OBRU: W_i = kappa_{y_i} * a_i * mu_i."""
    kappa_pos_balls, kappa_neg_balls = ball_class_weights(
        balls, kappa_pos, kappa_neg
    )
    rel_pos, rel_neg, radius_scale, reliabilities = ball_reliability_weights(
        balls,
        epsilon=rel_epsilon,
        gamma=gamma,
        lambda_r=lambda_r,
        variant=variant,
    )
    obru_pos, obru_neg, O_all, mu_all = ball_obru_weights(
        balls,
        minority_label,
        alpha=alpha,
        epsilon=epsilon,
        top_k=top_k,
    )
    weights_pos = kappa_pos_balls * rel_pos * obru_pos
    weights_neg = kappa_neg_balls * rel_neg * obru_neg
    return (
        weights_pos,
        weights_neg,
        radius_scale,
        reliabilities,
        O_all,
        mu_all,
        kappa_pos_balls,
        kappa_neg_balls,
    )


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def overlap_diagnostics_summary(
    balls: list[BallInfo],
    overlap_scores: np.ndarray,
    mu_all: np.ndarray,
    minority_label: float,
    *,
    purities: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    cw_weights: np.ndarray | None = None,
    rel_weights: np.ndarray | None = None,
    cr_weights: np.ndarray | None = None,
) -> dict:
    """Dataset-level overlap statistics and correlations."""
    O = np.asarray(overlap_scores, dtype=float).ravel()
    mu = np.asarray(mu_all, dtype=float).ravel()
    if purities is None:
        purities = np.array([b.purity for b in balls], dtype=float)
    if radii is None:
        radii = np.array([b.radius for b in balls], dtype=float)

    minority_mask = np.array([b.label == minority_label for b in balls])
    majority_mask = ~minority_mask

    def _stats(values: np.ndarray) -> dict:
        if values.size == 0:
            return {}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary = {
        "n_balls": len(balls),
        "radius": _stats(radii),
        "overlap": _stats(O),
        "frac_overlap_gt_0": float(np.mean(O > 0.0)) if O.size else 0.0,
        "frac_overlap_ge_0_5": float(np.mean(O >= 0.5)) if O.size else 0.0,
        "minority_mean_overlap": float(np.mean(O[minority_mask]))
        if np.any(minority_mask)
        else float("nan"),
        "majority_mean_overlap": float(np.mean(O[majority_mask]))
        if np.any(majority_mask)
        else float("nan"),
        "minority_mean_mu": float(np.mean(mu[minority_mask]))
        if np.any(minority_mask)
        else float("nan"),
        "majority_mean_mu": float(np.mean(mu[majority_mask]))
        if np.any(majority_mask)
        else float("nan"),
    }

    correlations = {
        "overlap_vs_purity": safe_correlation(O, purities),
        "overlap_vs_radius": safe_correlation(O, radii),
        "overlap_vs_cw_weight": safe_correlation(O, cw_weights)
        if cw_weights is not None
        else float("nan"),
        "overlap_vs_r_weight": safe_correlation(O, rel_weights)
        if rel_weights is not None
        else float("nan"),
        "overlap_vs_cr_weight": safe_correlation(O, cr_weights)
        if cr_weights is not None
        else float("nan"),
    }
    return {"summary": summary, "correlations": correlations}
