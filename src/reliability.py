"""
Ball-level reliability weights for R-GBLSTSVM.

The reliability score is empirically motivated — not a theoretically derived
fuzzy membership. See docs/RELIABILITY_WEIGHTING.md.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .granular_balls import BallInfo

ReliabilityVariant = Literal["full", "purity_only", "radius_impurity_only"]
WeightPlacement = Literal["opposite_only", "both"]

# Documented Phase 3 defaults (not tuned).
DEFAULT_EPSILON = 0.05
DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA_R = 1.0


def compute_radius_scale(
    radii: np.ndarray,
    *,
    method: str = "median_positive",
) -> float:
    """
    Training-fold radius scale for normalization.

    Uses the median of strictly positive ball radii. Balls with r_i = 0
    (singletons under the frozen mean-distance definition) do not enter the
    scale. If no positive radius exists, returns 1.0 to avoid division by zero.
    """
    radii = np.asarray(radii, dtype=float).ravel()
    positive = radii[radii > 0.0]

    if positive.size == 0:
        return 1.0

    if method == "median_positive":
        return float(np.median(positive))

    raise ValueError(f"Unknown radius scale method: {method!r}")


def normalize_radius(radius: float, scale: float) -> float:
    """r_tilde = r / scale with safe denominator."""
    if scale <= 0.0:
        return 0.0
    return float(radius) / scale


def ball_reliability_score(
    purity: float,
    radius: float,
    radius_scale: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gamma: float = DEFAULT_GAMMA,
    lambda_r: float = DEFAULT_LAMBDA_R,
    variant: ReliabilityVariant = "full",
) -> float:
    """
    Compute reliability a_i for one granular ball.

    Full score:
        a_i = epsilon + (1-epsilon) * p^gamma * exp(-lambda_r * r_tilde * (1-p))

    Ablation variants omit the purity or radius/impurity factor respectively.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1).")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if lambda_r < 0.0:
        raise ValueError("lambda_r must be non-negative.")

    p = float(np.clip(purity, 0.0, 1.0))
    r_tilde = normalize_radius(radius, radius_scale)

    if variant == "purity_only":
        core = p**gamma
    elif variant == "radius_impurity_only":
        core = float(np.exp(-lambda_r * r_tilde * (1.0 - p)))
    elif variant == "full":
        core = (p**gamma) * float(np.exp(-lambda_r * r_tilde * (1.0 - p)))
    else:
        raise ValueError(f"Unknown reliability variant: {variant!r}")

    return epsilon + (1.0 - epsilon) * core


def ball_reliability_weights(
    balls: list[BallInfo],
    *,
    epsilon: float = DEFAULT_EPSILON,
    gamma: float = DEFAULT_GAMMA,
    lambda_r: float = DEFAULT_LAMBDA_R,
    variant: ReliabilityVariant = "full",
    radius_scale: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Map reliability scores to per-ball weights aligned with ``balls_to_class_arrays``.

    Returns
    -------
    weights_pos, weights_neg, radius_scale, reliabilities_all
        reliabilities_all has one entry per ball in ``balls`` list order.
    """
    radii = np.array([b.radius for b in balls], dtype=float)
    if radius_scale is None:
        radius_scale = compute_radius_scale(radii)

    reliabilities = np.array(
        [
            ball_reliability_score(
                b.purity,
                b.radius,
                radius_scale,
                epsilon=epsilon,
                gamma=gamma,
                lambda_r=lambda_r,
                variant=variant,
            )
            for b in balls
        ],
        dtype=float,
    )

    weights_pos = np.array(
        [reliabilities[i] for i, b in enumerate(balls) if b.label == 1],
        dtype=float,
    )
    weights_neg = np.array(
        [reliabilities[i] for i, b in enumerate(balls) if b.label != 1],
        dtype=float,
    )
    return weights_pos, weights_neg, radius_scale, reliabilities
