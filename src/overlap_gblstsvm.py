"""
Overlap-aware weighted GBLSTSVM variants (Stages 7–9).

Stage 7: OBRU-GBLSTSVM          — mu_i only (fixed from ball geometry)
Stage 8: CW+OBRU-GBLSTSVM       — kappa * mu
Stage 9: CW+R+OBRU-GBLSTSVM     — kappa * reliability * mu

Overlap weights depend only on fixed granular-ball geometry; no iterative
recomputation is required. Frozen baseline/CW/R/CR implementations are reused.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .class_weights import (
    WeightPlacement,
    compute_binary_class_weights,
)
from .cw_gblstsvm import solve_cw_gblstsvm_systems
from .gblstsvm import GBLSTSVM, PredictionRule, _normalize_labels
from .granular_balls import balls_to_class_arrays, generate_granular_balls
from .overlap import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_TOP_K,
    ball_cr_obru_weights,
    ball_cw_obru_weights,
    ball_obru_weights,
    minority_label_from_training,
    split_mu_by_class,
)
from .reliability import (
    DEFAULT_EPSILON as REL_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_R,
    ReliabilityVariant,
    ball_reliability_weights,
)


class OBRUGBLSTSVM(GBLSTSVM):
    """Stage 7: OBRU-GBLSTSVM with W_i = mu_i."""

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
        alpha: float = DEFAULT_ALPHA,
        epsilon: float = DEFAULT_EPSILON,
        top_k: int = DEFAULT_TOP_K,
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
        self.alpha = alpha
        self.epsilon = epsilon
        self.top_k = top_k
        self.weight_placement = weight_placement
        self.minority_label_: float | None = None
        self.overlap_all_: np.ndarray | None = None
        self.mu_all_: np.ndarray | None = None
        self.weights_pos_: np.ndarray | None = None
        self.weights_neg_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OBRUGBLSTSVM":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = _normalize_labels(y)
        minority_label = minority_label_from_training(y)
        self.minority_label_ = minority_label

        self.balls_ = generate_granular_balls(
            X,
            y,
            purity_threshold=self.purity_threshold,
            min_samples_per_ball=self.min_samples_per_ball,
            min_ball_count=self.min_ball_count,
        )

        weights_pos, weights_neg, O_all, mu_all = ball_obru_weights(
            self.balls_,
            minority_label,
            alpha=self.alpha,
            epsilon=self.epsilon,
            top_k=self.top_k,
        )
        self.overlap_all_ = O_all
        self.mu_all_ = mu_all
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


class CWOBRUGBLSTSVM(OBRUGBLSTSVM):
    """Stage 8: CW + OBRU with W_i = kappa_{y_i} * mu_i."""

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
        alpha: float = DEFAULT_ALPHA,
        epsilon: float = DEFAULT_EPSILON,
        top_k: int = DEFAULT_TOP_K,
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
            alpha=alpha,
            epsilon=epsilon,
            top_k=top_k,
            weight_placement=weight_placement,
        )
        self.kappa_pos_: float | None = None
        self.kappa_neg_: float | None = None
        self.n_train_: int | None = None
        self.n_pos_: int | None = None
        self.n_neg_: int | None = None
        self.kappa_pos_balls_: np.ndarray | None = None
        self.kappa_neg_balls_: np.ndarray | None = None
        self.obru_pos_: np.ndarray | None = None
        self.obru_neg_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CWOBRUGBLSTSVM":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = _normalize_labels(y)

        kappa_pos, kappa_neg, n_pos, n_neg, n_train = compute_binary_class_weights(y)
        self.kappa_pos_ = kappa_pos
        self.kappa_neg_ = kappa_neg
        self.n_train_ = n_train
        self.n_pos_ = n_pos
        self.n_neg_ = n_neg
        minority_label = minority_label_from_training(y)
        self.minority_label_ = minority_label

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
            O_all,
            mu_all,
            k_pos,
            k_neg,
        ) = ball_cw_obru_weights(
            self.balls_,
            kappa_pos,
            kappa_neg,
            minority_label,
            alpha=self.alpha,
            epsilon=self.epsilon,
            top_k=self.top_k,
        )
        obru_pos, obru_neg = split_mu_by_class(self.balls_, mu_all)

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
        self.overlap_all_ = O_all
        self.mu_all_ = mu_all
        self.weights_pos_ = weights_pos
        self.weights_neg_ = weights_neg
        self.kappa_pos_balls_ = k_pos
        self.kappa_neg_balls_ = k_neg
        self.obru_pos_ = obru_pos
        self.obru_neg_ = obru_neg
        return self


class CROBRUGBLSTSVM(OBRUGBLSTSVM):
    """Stage 9: CW + R + OBRU with W_i = kappa_{y_i} * a_i * mu_i."""

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
        alpha: float = DEFAULT_ALPHA,
        epsilon: float = DEFAULT_EPSILON,
        top_k: int = DEFAULT_TOP_K,
        weight_placement: WeightPlacement = "opposite_only",
        rel_epsilon: float = REL_EPSILON,
        gamma: float = DEFAULT_GAMMA,
        lambda_r: float = DEFAULT_LAMBDA_R,
        reliability_variant: ReliabilityVariant = "full",
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
            alpha=alpha,
            epsilon=epsilon,
            top_k=top_k,
            weight_placement=weight_placement,
        )
        self.rel_epsilon = rel_epsilon
        self.gamma = gamma
        self.lambda_r = lambda_r
        self.reliability_variant = reliability_variant
        self.kappa_pos_: float | None = None
        self.kappa_neg_: float | None = None
        self.n_train_: int | None = None
        self.n_pos_: int | None = None
        self.n_neg_: int | None = None
        self.radius_scale_: float | None = None
        self.reliabilities_: np.ndarray | None = None
        self.kappa_pos_balls_: np.ndarray | None = None
        self.kappa_neg_balls_: np.ndarray | None = None
        self.obru_pos_: np.ndarray | None = None
        self.obru_neg_: np.ndarray | None = None
        self.rel_pos_: np.ndarray | None = None
        self.rel_neg_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CROBRUGBLSTSVM":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = _normalize_labels(y)

        kappa_pos, kappa_neg, n_pos, n_neg, n_train = compute_binary_class_weights(y)
        self.kappa_pos_ = kappa_pos
        self.kappa_neg_ = kappa_neg
        self.n_train_ = n_train
        self.n_pos_ = n_pos
        self.n_neg_ = n_neg
        minority_label = minority_label_from_training(y)
        self.minority_label_ = minority_label

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
            O_all,
            mu_all,
            k_pos,
            k_neg,
        ) = ball_cr_obru_weights(
            self.balls_,
            kappa_pos,
            kappa_neg,
            minority_label,
            alpha=self.alpha,
            epsilon=self.epsilon,
            top_k=self.top_k,
            rel_epsilon=self.rel_epsilon,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            variant=self.reliability_variant,
        )
        rel_pos, rel_neg, _, _ = ball_reliability_weights(
            self.balls_,
            epsilon=self.rel_epsilon,
            gamma=self.gamma,
            lambda_r=self.lambda_r,
            variant=self.reliability_variant,
        )
        obru_pos, obru_neg = split_mu_by_class(self.balls_, mu_all)

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
        self.overlap_all_ = O_all
        self.mu_all_ = mu_all
        self.weights_pos_ = weights_pos
        self.weights_neg_ = weights_neg
        self.radius_scale_ = radius_scale
        self.reliabilities_ = reliabilities
        self.kappa_pos_balls_ = k_pos
        self.kappa_neg_balls_ = k_neg
        self.obru_pos_ = obru_pos
        self.obru_neg_ = obru_neg
        self.rel_pos_ = rel_pos
        self.rel_neg_ = rel_neg
        return self
