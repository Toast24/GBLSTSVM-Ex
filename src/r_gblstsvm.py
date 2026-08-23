"""
Reliability-weighted GBLSTSVM (R-GBLSTSVM).

Applies ball-level reliability weights to the optimization only.
No class-level weighting (kappa = 1 for every ball).

Reuses the weighted normal-equation solver from the CW implementation
without importing the class-weighting module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .cw_gblstsvm import solve_cw_gblstsvm_systems
from .gblstsvm import GBLSTSVM, PredictionRule, _normalize_labels
from .granular_balls import balls_to_class_arrays, generate_granular_balls
from .reliability import (
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_R,
    ReliabilityVariant,
    WeightPlacement,
    ball_reliability_weights,
)


class RGBLSTSVM(GBLSTSVM):
    """
    Granular-ball LSTSVM with ball-level reliability weighting.

    Class weights are not used. Reliability enters the opposite-class
    separation term by default (weight_placement="opposite_only").
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
        self.radius_scale_: float | None = None
        self.reliabilities_: np.ndarray | None = None
        self.weights_pos_: np.ndarray | None = None
        self.weights_neg_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RGBLSTSVM":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = _normalize_labels(y)

        self.balls_ = generate_granular_balls(
            X,
            y,
            purity_threshold=self.purity_threshold,
            min_samples_per_ball=self.min_samples_per_ball,
            min_ball_count=self.min_ball_count,
        )

        weights_pos, weights_neg, radius_scale, reliabilities = ball_reliability_weights(
            self.balls_,
            epsilon=self.epsilon,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            variant=self.reliability_variant,
        )
        self.radius_scale_ = radius_scale
        self.reliabilities_ = reliabilities
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
