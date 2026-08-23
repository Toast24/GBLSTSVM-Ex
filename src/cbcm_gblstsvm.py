"""
Competition-Connectivity GBLSTSVM variants (CA / CB / CC).

Structurally identical to CR-GBLSTSVM except reliability weight R_i is
replaced by CBCM fuzzy membership mu_i:

  CA-GBLSTSVM: w_i = CW(y_i) * muA_i
  CB-GBLSTSVM: w_i = CW(y_i) * muB_i
  CC-GBLSTSVM: w_i = CW(y_i) * muC_i   (muC = muA * muB)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .cbcm import (
    CBCMMembershipResult,
    CBCMVariant,
    DEFAULT_SIGMA_EPS,
    ball_cbcm_weights,
)
from .class_weights import WeightPlacement, compute_binary_class_weights
from .cw_gblstsvm import solve_cw_gblstsvm_systems
from .gblstsvm import GBLSTSVM, PredictionRule, _normalize_labels
from .granular_balls import balls_to_class_arrays, generate_granular_balls


def solve_cbcm_gblstsvm_systems(
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
    """Delegate to shared weighted solver; CBCM differs only in weight construction."""
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


class _CBCMGBLSTSVMBase(GBLSTSVM):
    """Shared fit logic for CA / CB / CC variants."""

    _variant: CBCMVariant = "combined"

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
        sigma_eps: float = DEFAULT_SIGMA_EPS,
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
        self.sigma_eps = sigma_eps
        self.weight_placement = weight_placement
        self.kappa_pos_: float | None = None
        self.kappa_neg_: float | None = None
        self.n_train_: int | None = None
        self.n_pos_: int | None = None
        self.n_neg_: int | None = None
        self.kappa_pos_balls_: np.ndarray | None = None
        self.kappa_neg_balls_: np.ndarray | None = None
        self.weights_pos_: np.ndarray | None = None
        self.weights_neg_: np.ndarray | None = None
        self.membership_: CBCMMembershipResult | None = None
        self.k_neighbors_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CBCMGBLSTSVMBase":
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

        (
            weights_pos,
            weights_neg,
            membership,
            kappa_pos_balls,
            kappa_neg_balls,
        ) = ball_cbcm_weights(
            self.balls_,
            kappa_pos,
            kappa_neg,
            variant=self._variant,
            sigma_eps=self.sigma_eps,
        )
        self.membership_ = membership
        self.k_neighbors_ = membership.k
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

        w1, b1, w2, b2 = solve_cbcm_gblstsvm_systems(
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


class CAGBLSTSVM(_CBCMGBLSTSVMBase):
    """CA-GBLSTSVM: w_i = CW(y_i) * muA_i (competition only)."""

    _variant = "competition"


class CBGBLSTSVM(_CBCMGBLSTSVMBase):
    """CB-GBLSTSVM: w_i = CW(y_i) * muB_i (connectivity only)."""

    _variant = "connectivity"


class CCGBLSTSVM(_CBCMGBLSTSVMBase):
    """CC-GBLSTSVM: w_i = CW(y_i) * muC_i (competition x connectivity)."""

    _variant = "combined"
