"""
BMRU-weighted GBLSTSVM variants (Stages 4–6).

Stage 4: BMRU-GBLSTSVM          — BMRU only, iterative
Stage 5: CW+BMRU-GBLSTSVM       — kappa * mu
Stage 6: CW+R+BMRU-GBLSTSVM     — kappa * reliability * mu

**FROZEN.** Model classes and iterative fitting loop are frozen alongside
``src/bmru.py``. Frozen baseline/CW/R/CR implementations are not modified.
The current BMRU formulation is retained as a documented failed hypothesis;
see docs/BMRU_GBLSTSVM.md §15–16.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .bmru import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_WEIGHT_TOLERANCE,
    BMRUIterationRecord,
    MarginMode,
    ball_bmru_weights,
    summarize_bmru_iteration,
)
from .class_weights import (
    WeightPlacement,
    ball_class_weights,
    compute_binary_class_weights,
)
from .cw_gblstsvm import _weighted_gram, solve_cw_gblstsvm_systems
from .gblstsvm import GBLSTSVM, PredictionRule, _normalize_labels, solve_gblstsvm_systems
from .granular_balls import BallInfo, balls_to_class_arrays, generate_granular_balls
from .reliability import (
    DEFAULT_EPSILON as REL_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_R,
    ReliabilityVariant,
    ball_reliability_weights,
)


def ball_cw_bmru_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    margin_mode: MarginMode = "reference",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combined CW + BMRU: W_i = kappa_{y_i} * mu_i."""
    kappa_pos_balls, kappa_neg_balls = ball_class_weights(
        balls, kappa_pos, kappa_neg
    )
    bmru_pos, bmru_neg, u_all, mu_all, margins = ball_bmru_weights(
        balls,
        w1,
        b1,
        w2,
        b2,
        alpha=alpha,
        epsilon=epsilon,
        margin_mode=margin_mode,
    )
    weights_pos = kappa_pos_balls * bmru_pos
    weights_neg = kappa_neg_balls * bmru_neg
    return (
        weights_pos,
        weights_neg,
        u_all,
        mu_all,
        margins,
        kappa_pos_balls,
        kappa_neg_balls,
    )


def ball_cr_bmru_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    margin_mode: MarginMode = "reference",
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
    np.ndarray,
]:
    """Combined CW + R + BMRU: W_i = kappa_{y_i} * a_i * mu_i."""
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
    bmru_pos, bmru_neg, u_all, mu_all, margins = ball_bmru_weights(
        balls,
        w1,
        b1,
        w2,
        b2,
        alpha=alpha,
        epsilon=epsilon,
        margin_mode=margin_mode,
    )
    weights_pos = kappa_pos_balls * rel_pos * bmru_pos
    weights_neg = kappa_neg_balls * rel_neg * bmru_neg
    return (
        weights_pos,
        weights_neg,
        radius_scale,
        reliabilities,
        u_all,
        mu_all,
        margins,
        kappa_pos_balls,
        kappa_neg_balls,
    )


def _weighted_system_condition_numbers(
    centers_pos: np.ndarray,
    radii_pos: np.ndarray,
    centers_neg: np.ndarray,
    radii_neg: np.ndarray,
    c1: float,
    c2: float,
    weights_pos: np.ndarray,
    weights_neg: np.ndarray,
    reg_term: float,
    weight_placement: WeightPlacement,
) -> tuple[float, float]:
    """Condition numbers of the two weighted normal-equation matrices."""
    A = np.asarray(centers_pos, dtype=float)
    B = np.asarray(centers_neg, dtype=float)
    weights_pos = np.asarray(weights_pos, dtype=float).reshape(-1)
    weights_neg = np.asarray(weights_neg, dtype=float).reshape(-1)

    mat_e1 = np.ones((A.shape[0], 1))
    mat_e2 = np.ones((B.shape[0], 1))
    mat_H = np.column_stack((A, mat_e1))
    mat_G = np.column_stack((B, mat_e2))
    dim_aug = mat_H.shape[1]
    eye = reg_term * np.identity(dim_aug)

    ones_pos = np.ones(A.shape[0])
    ones_neg = np.ones(B.shape[0])
    if weight_placement == "opposite_only":
        w_own_hp1, w_opp_hp1 = ones_pos, weights_neg
        w_own_hp2, w_opp_hp2 = ones_neg, weights_pos
    else:
        w_own_hp1, w_opp_hp1 = weights_pos, weights_neg
        w_own_hp2, w_opp_hp2 = weights_neg, weights_pos

    gram_hp1 = _weighted_gram(mat_G, w_opp_hp1) + (1.0 / c1) * _weighted_gram(
        mat_H, w_own_hp1
    ) + eye
    gram_hp2 = _weighted_gram(mat_H, w_opp_hp2) + (1.0 / c2) * _weighted_gram(
        mat_G, w_own_hp2
    ) + eye

    def _cond(mat: np.ndarray) -> float:
        try:
            c = float(np.linalg.cond(mat))
            return c if np.isfinite(c) else float("nan")
        except (np.linalg.LinAlgError, ValueError):
            return float("nan")

    return _cond(gram_hp1), _cond(gram_hp2)


def _iterative_weighted_fit(
    balls: list[BallInfo],
    c1: float,
    c2: float,
    reg_term: float,
    weight_placement: WeightPlacement,
    weight_fn,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
) -> tuple[np.ndarray, float, np.ndarray, float, list[BMRUIterationRecord], np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic iterative BMRU reweighting loop.

    weight_fn(w1, b1, w2, b2) -> (weights_pos, weights_neg, u_all, mu_all, ...)
    """
    centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(balls)

    w1, b1, w2, b2 = solve_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        c1,
        c2,
        reg_term=reg_term,
    )

    history: list[BMRUIterationRecord] = []
    previous_mu: np.ndarray | None = None
    u_all = np.ones(len(balls))
    mu_all = np.ones(len(balls))
    margins_all = np.zeros(len(balls))

    for it in range(1, max_iterations + 1):
        w1_before, b1_before = w1.copy(), b1
        w2_before, b2_before = w2.copy(), b2

        result = weight_fn(w1, b1, w2, b2)
        weights_pos, weights_neg = result[0], result[1]
        u_all, mu_all = result[2], result[3]
        if len(result) > 4:
            margins_all = result[4]

        cond_hp1, cond_hp2 = _weighted_system_condition_numbers(
            centers_pos,
            radii_pos,
            centers_neg,
            radii_neg,
            c1,
            c2,
            weights_pos,
            weights_neg,
            reg_term,
            weight_placement,
        )

        w1, b1, w2, b2 = solve_cw_gblstsvm_systems(
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

        record = summarize_bmru_iteration(
            mu_all,
            u_all,
            it,
            previous_mu,
            w1_prev=w1_before,
            b1_prev=b1_before,
            w2_prev=w2_before,
            b2_prev=b2_before,
            w1_new=w1,
            b1_new=b1,
            w2_new=w2,
            b2_new=b2,
            cond_hp1=cond_hp1,
            cond_hp2=cond_hp2,
        )
        history.append(record)
        previous_mu = mu_all.copy()

        if it > 1 and record.max_abs_weight_change < weight_tolerance:
            break

    return w1, b1, w2, b2, history, u_all, mu_all, margins_all


class BMRUGBLSTSVM(GBLSTSVM):
    """
    Stage 4: BMRU-GBLSTSVM without class or reliability weighting.

    Iteratively recomputes mu_i from current twin hyperplanes.
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
        alpha: float = DEFAULT_ALPHA,
        epsilon: float = DEFAULT_EPSILON,
        margin_mode: MarginMode = "reference",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
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
        self.margin_mode = margin_mode
        self.max_iterations = max_iterations
        self.weight_tolerance = weight_tolerance
        self.weight_placement = weight_placement
        self.u_all_: np.ndarray | None = None
        self.mu_all_: np.ndarray | None = None
        self.margins_all_: np.ndarray | None = None
        self.weights_pos_: np.ndarray | None = None
        self.weights_neg_: np.ndarray | None = None
        self.iteration_history_: list[BMRUIterationRecord] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BMRUGBLSTSVM":
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

        def weight_fn(w1, b1, w2, b2):
            pos, neg, u, mu, margins = ball_bmru_weights(
                self.balls_,
                w1,
                b1,
                w2,
                b2,
                alpha=self.alpha,
                epsilon=self.epsilon,
                margin_mode=self.margin_mode,
            )
            return pos, neg, u, mu, margins

        w1, b1, w2, b2, history, u_all, mu_all, margins = _iterative_weighted_fit(
            self.balls_,
            self.c1,
            self.c2,
            self.reg_term,
            self.weight_placement,
            weight_fn,
            max_iterations=self.max_iterations,
            weight_tolerance=self.weight_tolerance,
        )

        pos, neg, _, _, _ = ball_bmru_weights(
            self.balls_,
            w1,
            b1,
            w2,
            b2,
            alpha=self.alpha,
            epsilon=self.epsilon,
            margin_mode=self.margin_mode,
        )

        centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(
            self.balls_
        )
        self.centers_pos_ = centers_pos
        self.radii_pos_ = radii_pos
        self.centers_neg_ = centers_neg
        self.radii_neg_ = radii_neg

        self.w1_ = w1
        self.b1_ = b1
        self.w2_ = w2
        self.b2_ = b2
        self.coef_ = np.vstack([w1, w2])
        self.intercept_ = np.array([b1, b2])
        self.u_all_ = u_all
        self.mu_all_ = mu_all
        self.margins_all_ = margins
        self.weights_pos_ = pos
        self.weights_neg_ = neg
        self.iteration_history_ = history

        return self


class CWBMRUGBLSTSVM(BMRUGBLSTSVM):
    """Stage 5: CW + BMRU with W_i = kappa_{y_i} * mu_i."""

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
        margin_mode: MarginMode = "reference",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
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
            margin_mode=margin_mode,
            max_iterations=max_iterations,
            weight_tolerance=weight_tolerance,
            weight_placement=weight_placement,
        )
        self.kappa_pos_: float | None = None
        self.kappa_neg_: float | None = None
        self.n_train_: int | None = None
        self.n_pos_: int | None = None
        self.n_neg_: int | None = None
        self.kappa_pos_balls_: np.ndarray | None = None
        self.kappa_neg_balls_: np.ndarray | None = None
        self.bmru_pos_: np.ndarray | None = None
        self.bmru_neg_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CWBMRUGBLSTSVM":
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

        self.balls_ = generate_granular_balls(
            X,
            y,
            purity_threshold=self.purity_threshold,
            min_samples_per_ball=self.min_samples_per_ball,
            min_ball_count=self.min_ball_count,
        )

        def weight_fn(w1, b1, w2, b2):
            return ball_cw_bmru_weights(
                self.balls_,
                kappa_pos,
                kappa_neg,
                w1,
                b1,
                w2,
                b2,
                alpha=self.alpha,
                epsilon=self.epsilon,
                margin_mode=self.margin_mode,
            )

        w1, b1, w2, b2, history, u_all, mu_all, margins = _iterative_weighted_fit(
            self.balls_,
            self.c1,
            self.c2,
            self.reg_term,
            self.weight_placement,
            weight_fn,
            max_iterations=self.max_iterations,
            weight_tolerance=self.weight_tolerance,
        )

        (
            w_pos,
            w_neg,
            u_all,
            mu_all,
            margins,
            k_pos,
            k_neg,
        ) = ball_cw_bmru_weights(
            self.balls_,
            kappa_pos,
            kappa_neg,
            w1,
            b1,
            w2,
            b2,
            alpha=self.alpha,
            epsilon=self.epsilon,
            margin_mode=self.margin_mode,
        )
        bmru_pos, bmru_neg, _, _, _ = ball_bmru_weights(
            self.balls_,
            w1,
            b1,
            w2,
            b2,
            alpha=self.alpha,
            epsilon=self.epsilon,
            margin_mode=self.margin_mode,
        )

        centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(
            self.balls_
        )
        self.centers_pos_ = centers_pos
        self.radii_pos_ = radii_pos
        self.centers_neg_ = centers_neg
        self.radii_neg_ = radii_neg

        self.w1_ = w1
        self.b1_ = b1
        self.w2_ = w2
        self.b2_ = b2
        self.coef_ = np.vstack([w1, w2])
        self.intercept_ = np.array([b1, b2])
        self.u_all_ = u_all
        self.mu_all_ = mu_all
        self.margins_all_ = margins
        self.weights_pos_ = w_pos
        self.weights_neg_ = w_neg
        self.kappa_pos_balls_ = k_pos
        self.kappa_neg_balls_ = k_neg
        self.bmru_pos_ = bmru_pos
        self.bmru_neg_ = bmru_neg
        self.iteration_history_ = history

        return self


class CRBMRUGBLSTSVM(BMRUGBLSTSVM):
    """Stage 6: CW + R + BMRU with W_i = kappa_{y_i} * a_i * mu_i."""

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
        margin_mode: MarginMode = "reference",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
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
            margin_mode=margin_mode,
            max_iterations=max_iterations,
            weight_tolerance=weight_tolerance,
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
        self.bmru_pos_: np.ndarray | None = None
        self.bmru_neg_: np.ndarray | None = None
        self.rel_pos_: np.ndarray | None = None
        self.rel_neg_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CRBMRUGBLSTSVM":
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

        self.balls_ = generate_granular_balls(
            X,
            y,
            purity_threshold=self.purity_threshold,
            min_samples_per_ball=self.min_samples_per_ball,
            min_ball_count=self.min_ball_count,
        )

        def weight_fn(w1, b1, w2, b2):
            return ball_cr_bmru_weights(
                self.balls_,
                kappa_pos,
                kappa_neg,
                w1,
                b1,
                w2,
                b2,
                alpha=self.alpha,
                epsilon=self.epsilon,
                margin_mode=self.margin_mode,
                rel_epsilon=self.rel_epsilon,
                gamma=self.gamma,
                lambda_r=self.lambda_r,
                variant=self.reliability_variant,
            )

        w1, b1, w2, b2, history, u_all, mu_all, margins = _iterative_weighted_fit(
            self.balls_,
            self.c1,
            self.c2,
            self.reg_term,
            self.weight_placement,
            weight_fn,
            max_iterations=self.max_iterations,
            weight_tolerance=self.weight_tolerance,
        )

        (
            w_pos,
            w_neg,
            radius_scale,
            reliabilities,
            u_all,
            mu_all,
            margins,
            k_pos,
            k_neg,
        ) = ball_cr_bmru_weights(
            self.balls_,
            kappa_pos,
            kappa_neg,
            w1,
            b1,
            w2,
            b2,
            alpha=self.alpha,
            epsilon=self.epsilon,
            margin_mode=self.margin_mode,
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
        bmru_pos, bmru_neg, _, _, _ = ball_bmru_weights(
            self.balls_,
            w1,
            b1,
            w2,
            b2,
            alpha=self.alpha,
            epsilon=self.epsilon,
            margin_mode=self.margin_mode,
        )

        centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(
            self.balls_
        )
        self.centers_pos_ = centers_pos
        self.radii_pos_ = radii_pos
        self.centers_neg_ = centers_neg
        self.radii_neg_ = radii_neg

        self.w1_ = w1
        self.b1_ = b1
        self.w2_ = w2
        self.b2_ = b2
        self.coef_ = np.vstack([w1, w2])
        self.intercept_ = np.array([b1, b2])
        self.u_all_ = u_all
        self.mu_all_ = mu_all
        self.margins_all_ = margins
        self.weights_pos_ = w_pos
        self.weights_neg_ = w_neg
        self.radius_scale_ = radius_scale
        self.reliabilities_ = reliabilities
        self.kappa_pos_balls_ = k_pos
        self.kappa_neg_balls_ = k_neg
        self.bmru_pos_ = bmru_pos
        self.bmru_neg_ = bmru_neg
        self.rel_pos_ = rel_pos
        self.rel_neg_ = rel_neg
        self.iteration_history_ = history

        return self
