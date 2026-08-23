"""
Class-weighted GBLSTSVM (CW-GBLSTSVM).

Extends the frozen GBLSTSVM optimization with per-ball class costs
kappa_c = n_train / (2 * n_c), computed from original training labels.

By default only the opposite-class separation squared-loss term is
weighted in each twin subproblem (weight_placement="opposite_only").
See docs/CLASS_WEIGHTING.md for the full mathematical placement.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .class_weights import (
    WeightPlacement,
    ball_class_weights,
    compute_binary_class_weights,
)
from .gblstsvm import GBLSTSVM, PredictionRule, _normalize_labels
from .granular_balls import balls_to_class_arrays


def _weighted_gram(mat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return mat^T W mat for diagonal W = diag(weights)."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    return mat.T @ (w[:, None] * mat)


def _weighted_matvec(mat: np.ndarray, weights: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Return mat^T W vec for diagonal W = diag(weights)."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    v = np.asarray(vec, dtype=float).reshape(-1, 1)
    return (mat.T @ (w[:, None] * v)).ravel()


def solve_cw_gblstsvm_systems(
    centers_pos: np.ndarray,
    radii_pos: np.ndarray,
    centers_neg: np.ndarray,
    radii_neg: np.ndarray,
    c1: float,
    c2: float,
    weights_pos: np.ndarray,
    weights_neg: np.ndarray,
    reg_term: float = 1e-4,
    weight_placement: WeightPlacement = "opposite_only",
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """
    Solve CW-GBLSTSVM twin hyperplanes with explicit diagonal weighting.

    Frozen baseline (all weights = 1, opposite_only) reproduces
    ``solve_gblstsvm_systems`` when the same inputs are used.

    Parameters
    ----------
    weights_pos : (m1,)
        Class weight kappa_{y_i} for each positive (+1) ball.
    weights_neg : (m2,)
        Class weight kappa_{y_i} for each negative (-1) ball.
    weight_placement : 'opposite_only' | 'both'
        Where class weights enter the objective (see docs/CLASS_WEIGHTING.md).
    """
    if c1 <= 0 or c2 <= 0:
        raise ValueError("c1 and c2 must be positive.")

    weights_pos = np.asarray(weights_pos, dtype=float).reshape(-1)
    weights_neg = np.asarray(weights_neg, dtype=float).reshape(-1)

    A = np.asarray(centers_pos, dtype=float)
    B = np.asarray(centers_neg, dtype=float)
    R_pos = np.asarray(radii_pos, dtype=float).reshape(-1, 1)
    R_neg = np.asarray(radii_neg, dtype=float).reshape(-1, 1)

    if weights_pos.shape[0] != A.shape[0]:
        raise ValueError("weights_pos length must match number of positive balls.")
    if weights_neg.shape[0] != B.shape[0]:
        raise ValueError("weights_neg length must match number of negative balls.")

    mat_e1 = np.ones((A.shape[0], 1))
    mat_e2 = np.ones((B.shape[0], 1))

    mat_H = np.column_stack((A, mat_e1))
    mat_G = np.column_stack((B, mat_e2))

    dim_aug = mat_H.shape[1]
    eye = reg_term * np.identity(dim_aug)

    ones_pos = np.ones(A.shape[0])
    ones_neg = np.ones(B.shape[0])

    if weight_placement == "opposite_only":
        w_own_hp1 = ones_pos
        w_opp_hp1 = weights_neg
        w_own_hp2 = ones_neg
        w_opp_hp2 = weights_pos
    elif weight_placement == "both":
        w_own_hp1 = weights_pos
        w_opp_hp1 = weights_neg
        w_own_hp2 = weights_neg
        w_opp_hp2 = weights_pos
    else:
        raise ValueError(f"Unknown weight_placement: {weight_placement!r}")

    # Hyperplane 1:
    # min 1/2 ||W_own^{1/2} H theta1||^2 + (c1/2) ||W_opp^{1/2}(G theta1 + e2 + R-)||^2
    gram_hp1 = _weighted_gram(mat_G, w_opp_hp1) + (1.0 / c1) * _weighted_gram(
        mat_H, w_own_hp1
    )
    inv_p_1 = np.linalg.inv(gram_hp1 + eye)
    e2_R_neg = mat_e2 + R_neg
    rhs_1_top = _weighted_matvec(B, w_opp_hp1, e2_R_neg)
    rhs_1_bottom = _weighted_matvec(mat_e2, w_opp_hp1, mat_e2 + R_neg)
    rhs_1 = np.concatenate([rhs_1_top, rhs_1_bottom])
    hyper_p_1 = -1.0 * inv_p_1 @ rhs_1.reshape(-1, 1)
    w1 = hyper_p_1[:-1, 0]
    b1 = float(hyper_p_1[-1, 0])

    # Hyperplane 2:
    # min 1/2 ||W_own^{1/2} G theta2||^2 + (c2/2) ||W_opp^{1/2}(H theta2 - (e1 + R+))||^2
    gram_hp2 = _weighted_gram(mat_H, w_opp_hp2) + (1.0 / c2) * _weighted_gram(
        mat_G, w_own_hp2
    )
    inv_p_2 = np.linalg.inv(gram_hp2 + eye)
    e1_R_pos = mat_e1 + R_pos
    rhs_2_top = _weighted_matvec(A, w_opp_hp2, e1_R_pos)
    rhs_2_bottom = _weighted_matvec(mat_e1, w_opp_hp2, mat_e1 + R_pos)
    rhs_2 = np.concatenate([rhs_2_top, rhs_2_bottom])
    hyper_p_2 = inv_p_2 @ rhs_2.reshape(-1, 1)
    w2 = hyper_p_2[:-1, 0]
    b2 = float(hyper_p_2[-1, 0])

    return w1, b1, w2, b2


class CWGBLSTSVM(GBLSTSVM):
    """
    Class-weighted granular-ball LSTSVM.

    Inherits granular-ball generation, prediction rules, and hyperparameters
    from the frozen GBLSTSVM baseline. Only the optimization uses class costs.
    """

    def __init__(
        self,
        c1: float = 1.0,
        c2: float = 0.1,
        purity_threshold: float = 1.0,
        min_samples_per_ball: int = 1,
        min_ball_count: int = 2,
        reg_term: float = 1e-4,
        prediction_rule: PredictionRule = "reference",
        random_state: Optional[int] = None,
        weight_placement: WeightPlacement = "opposite_only",
    ) -> None:
        super().__init__(
            c1=c1,
            c2=c2,
            purity_threshold=purity_threshold,
            min_samples_per_ball=min_samples_per_ball,
            min_ball_count=min_ball_count,
            reg_term=reg_term,
            prediction_rule=prediction_rule,
            random_state=random_state,
        )
        self.weight_placement = weight_placement
        self.kappa_pos_: float | None = None
        self.kappa_neg_: float | None = None
        self.n_train_: int | None = None
        self.n_pos_: int | None = None
        self.n_neg_: int | None = None
        self.weights_pos_: np.ndarray | None = None
        self.weights_neg_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CWGBLSTSVM":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = _normalize_labels(y)

        kappa_pos, kappa_neg, n_pos, n_neg, n_train = compute_binary_class_weights(
            y
        )
        self.kappa_pos_ = kappa_pos
        self.kappa_neg_ = kappa_neg
        self.n_train_ = n_train
        self.n_pos_ = n_pos
        self.n_neg_ = n_neg

        self.balls_ = self._generate_balls(X, y)
        weights_pos, weights_neg = ball_class_weights(
            self.balls_, kappa_pos, kappa_neg
        )
        self.weights_pos_ = weights_pos
        self.weights_neg_ = weights_neg

        centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(
            self.balls_
        )
        self.centers_pos_ = centers_pos
        self.radii_pos_ = radii_pos
        self.centers_neg_ = centers_neg
        self.radii_neg_ = radii_neg

        w1, b1, w2, b2 = solve_cw_gblstsvm_systems(
            centers_pos,
            radii_pos,
            centers_neg,
            radii_neg,
            self.c1,
            self.c2,
            weights_pos,
            weights_neg,
            reg_term=self.reg_term,
            weight_placement=self.weight_placement,
        )

        self.w1_ = w1
        self.b1_ = b1
        self.w2_ = w2
        self.b2_ = b2
        self.coef_ = np.vstack([w1, w2])
        self.intercept_ = np.array([b1, b2])

        return self

    def _generate_balls(self, X: np.ndarray, y: np.ndarray):
        """Delegate to frozen granular-ball generation (unchanged)."""
        from .granular_balls import generate_granular_balls

        return generate_granular_balls(
            X,
            y,
            purity_threshold=self.purity_threshold,
            min_samples_per_ball=self.min_samples_per_ball,
            min_ball_count=self.min_ball_count,
        )
