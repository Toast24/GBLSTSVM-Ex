"""
Class- and reliability-weighted GBLSTSVM (CR-GBLSTSVM).

Combines class-level imbalance costs (Phase 2) with ball-level reliability
weights (Phase 3) as a compositional product:

    final_weight_i = kappa_{y_i} * a_i

Does not modify frozen GBLSTSVM, CW-GBLSTSVM, or R-GBLSTSVM.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .class_weights import (
    WeightPlacement,
    ball_class_weights,
    compute_binary_class_weights,
)
from .cw_gblstsvm import solve_cw_gblstsvm_systems
from .gblstsvm import GBLSTSVM, PredictionRule, _normalize_labels
from .granular_balls import BallInfo, balls_to_class_arrays, generate_granular_balls
from .reliability import (
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_R,
    ReliabilityVariant,
    ball_reliability_weights,
)


def ball_cr_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gamma: float = DEFAULT_GAMMA,
    lambda_r: float = DEFAULT_LAMBDA_R,
    variant: ReliabilityVariant = "full",
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute combined CR weights: kappa_{y_i} * a_i for each ball.

    Class weights use original training counts (via kappa_pos/kappa_neg).
    Reliability is computed independently from ball purity and radius.
    Ball multiplicity is never multiplied into the weight.

    Returns
    -------
    weights_pos, weights_neg, radius_scale, reliabilities_all,
    kappa_pos_balls, kappa_neg_balls
        Final weights and intermediate factors aligned with
        ``balls_to_class_arrays`` row order.
    """
    kappa_pos_balls, kappa_neg_balls = ball_class_weights(
        balls, kappa_pos, kappa_neg
    )
    rel_pos, rel_neg, radius_scale, reliabilities = ball_reliability_weights(
        balls,
        epsilon=epsilon,
        gamma=gamma,
        lambda_r=lambda_r,
        variant=variant,
    )

    weights_pos = kappa_pos_balls * rel_pos
    weights_neg = kappa_neg_balls * rel_neg
    return (
        weights_pos,
        weights_neg,
        radius_scale,
        reliabilities,
        kappa_pos_balls,
        kappa_neg_balls,
    )


def solve_cr_gblstsvm_systems(
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
    Solve CR-GBLSTSVM twin hyperplanes with combined diagonal weights.

    Delegates to the shared weighted solver from CW-GBLSTSVM; CR differs
    only in how ``weights_pos`` and ``weights_neg`` are constructed.
    """
    return solve_cw_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        c1,
        c2,
        weights_pos,
        weights_neg,
        reg_term=reg_term,
        weight_placement=weight_placement,
    )


class CRGBLSTSVM(GBLSTSVM):
    """
    Granular-ball LSTSVM with class- and reliability-weighted optimization.

    CR-GBLSTSVM is an empirical combination of class-level imbalance
    weighting and ball-level reliability weighting. The product
    ``kappa_{y_i} * a_i`` is not claimed to be theoretically optimal.
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
        epsilon: float = DEFAULT_EPSILON,
        gamma: float = DEFAULT_GAMMA,
        lambda_r: float = DEFAULT_LAMBDA_R,
        reliability_variant: ReliabilityVariant = "full",
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
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambda_r = lambda_r
        self.reliability_variant = reliability_variant
        self.weight_placement = weight_placement
        self.kappa_pos_: float | None = None
        self.kappa_neg_: float | None = None
        self.n_train_: int | None = None
        self.n_pos_: int | None = None
        self.n_neg_: int | None = None
        self.radius_scale_: float | None = None
        self.reliabilities_: np.ndarray | None = None
        self.kappa_pos_balls_: np.ndarray | None = None
        self.kappa_neg_balls_: np.ndarray | None = None
        self.weights_pos_: np.ndarray | None = None
        self.weights_neg_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CRGBLSTSVM":
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

        self.balls_ = generate_granular_balls(
            X,
            y,
            purity_threshold=self.purity_threshold,
            min_samples_per_ball=self.min_samples_per_ball,
            min_ball_count=self.min_ball_count,
        )

        (
            weights_pos,
            weights_neg,
            radius_scale,
            reliabilities,
            kappa_pos_balls,
            kappa_neg_balls,
        ) = ball_cr_weights(
            self.balls_,
            kappa_pos,
            kappa_neg,
            epsilon=self.epsilon,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            variant=self.reliability_variant,
        )
        self.radius_scale_ = radius_scale
        self.reliabilities_ = reliabilities
        self.kappa_pos_balls_ = kappa_pos_balls
        self.kappa_neg_balls_ = kappa_neg_balls
        self.weights_pos_ = weights_pos
        self.weights_neg_ = weights_neg

        centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(
            self.balls_
        )
        self.centers_pos_ = centers_pos
        self.radii_pos_ = radii_pos
        self.centers_neg_ = centers_neg
        self.radii_neg_ = radii_neg

        w1, b1, w2, b2 = solve_cr_gblstsvm_systems(
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
