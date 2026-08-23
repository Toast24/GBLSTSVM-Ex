"""
Class-level cost weights for CW-GBLSTSVM.

kappa_c = n_train / (2 * n_c)

Counts n_c are taken from the original training labels, not from
granular-ball statistics.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .granular_balls import BallInfo

WeightPlacement = Literal["opposite_only", "both"]


def compute_binary_class_weights(
    y_train: np.ndarray,
) -> Tuple[float, float, int, int, int]:
    """
    Compute kappa for positive (+1) and negative (-1) classes.

    Returns
    -------
    kappa_pos, kappa_neg, n_pos, n_neg, n_train
    """
    y = np.asarray(y_train, dtype=float).ravel()
    y = y.copy()
    y[y == 0] = -1

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == -1))
    n_train = n_pos + n_neg

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Binary class weighting requires both classes in training data.")

    kappa_pos = n_train / (2.0 * n_pos)
    kappa_neg = n_train / (2.0 * n_neg)
    return kappa_pos, kappa_neg, n_pos, n_neg, n_train


def ball_class_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map class-level costs to per-ball weights.

    Each ball receives kappa_{y_i} where y_i is the ball's assigned label.
    Ball multiplicity (num_samples) is NOT multiplied into the weight.

    Returns
    -------
    weights_pos_balls, weights_neg_balls
        One weight per positive (+1) ball and per negative (-1) ball,
        in the same order as ``balls_to_class_arrays``.
    """
    weights_pos = np.array(
        [kappa_pos if b.label == 1 else kappa_neg for b in balls if b.label == 1],
        dtype=float,
    )
    weights_neg = np.array(
        [kappa_pos if b.label == 1 else kappa_neg for b in balls if b.label != 1],
        dtype=float,
    )
    return weights_pos, weights_neg
