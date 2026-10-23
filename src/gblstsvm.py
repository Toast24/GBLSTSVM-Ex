"""
Paper-faithful linear GBLSTSVM (Granular Ball LSTSVM).

Optimization follows the closed-form twin hyperplane systems in the reference
repository (GBLSTSVM.py / LSTWGBSVM) and the published GBLSTSVM formulation:
  min 1/2 ||C w1 + e1 b1||^2 + (c1/2) ||D w1 + e2 b1 + e2 + R-||^2
  min 1/2 ||D w2 + e2 b2||^2 + (c2/2) ||C w2 + e1 b2 - e1 + R+||^2

c1 and c2 are trade-off parameters between the two squared-loss terms (not
class weights). The reference code uses +e1 + R+ in the second RHS; see
docs/BASELINE_RECONCILIATION.md for the §8 vs §13 sign analysis.

Numerical stabilization: small diagonal Tikhonov term (reg_term) matching the
reference implementation (default 1e-4).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .granular_balls import (
    BallInfo,
    balls_to_class_arrays,
    generate_granular_balls,
)

PredictionRule = Literal["paper", "reference", "raw", "norm"]


def resolve_prediction_rule(rule: PredictionRule | str) -> Literal["paper", "reference"]:
    """Map rebaseline aliases: raw=reference, norm=paper."""
    if rule in ("raw", "reference"):
        return "reference"
    if rule in ("norm", "paper"):
        return "paper"
    raise ValueError(f"Unknown prediction rule: {rule!r}")


def _normalize_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    unique = np.unique(y)
    if len(unique) != 2:
        raise ValueError(f"GBLSTSVM is binary; got {len(unique)} classes.")
    y_norm = y.copy()
    y_norm[y_norm == 0] = -1
    if not np.all(np.isin(y_norm, [-1, 1])):
        raise ValueError("Labels must be binary (+1/-1 or 0/1).")
    return y_norm


def solve_gblstsvm_systems(
    centers_pos: np.ndarray,
    radii_pos: np.ndarray,
    centers_neg: np.ndarray,
    radii_neg: np.ndarray,
    c1: float,
    c2: float,
    reg_term: float = 1e-4,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """
    Solve the two GBLSTSVM hyperplanes (reference LSTWGBSVM linear algebra).

    Returns w1, b1, w2, b2 with shapes (d,), scalar biases.
    """
    if c1 <= 0 or c2 <= 0:
        raise ValueError("c1 and c2 must be positive.")

    A = np.asarray(centers_pos, dtype=float)
    B = np.asarray(centers_neg, dtype=float)
    R_pos = np.asarray(radii_pos, dtype=float).reshape(-1, 1)
    R_neg = np.asarray(radii_neg, dtype=float).reshape(-1, 1)

    mat_e1 = np.ones((A.shape[0], 1))
    mat_e2 = np.ones((B.shape[0], 1))

    mat_H = np.column_stack((A, mat_e1))
    mat_G = np.column_stack((B, mat_e2))

    dim_aug = mat_H.shape[1]
    eye = reg_term * np.identity(dim_aug)

    # First hyperplane: theta1 = [w1; b1]
    inv_p_1 = np.linalg.inv(mat_G.T @ mat_G + (1.0 / c1) * (mat_H.T @ mat_H) + eye)
    e2_R_neg = mat_e2 + R_neg
    rhs_1_top = (B.T @ e2_R_neg).ravel()
    rhs_1_bottom = (mat_e2.T @ mat_e2 + mat_e2.T @ R_neg).ravel()
    rhs_1 = np.concatenate([rhs_1_top, rhs_1_bottom])
    hyper_p_1 = -1.0 * inv_p_1 @ rhs_1.reshape(-1, 1)
    w1 = hyper_p_1[:-1, 0]
    b1 = float(hyper_p_1[-1, 0])

    # Second hyperplane: theta2 = [w2; b2]
    inv_p_2 = np.linalg.inv((mat_H.T @ mat_H) + (1.0 / c2) * (mat_G.T @ mat_G) + eye)
    e1_R_pos = mat_e1 + R_pos
    rhs_2_top = (A.T @ e1_R_pos).ravel()
    rhs_2_bottom = (mat_e1.T @ mat_e1 + mat_e1.T @ R_pos).ravel()
    rhs_2 = np.concatenate([rhs_2_top, rhs_2_bottom])
    hyper_p_2 = inv_p_2 @ rhs_2.reshape(-1, 1)
    w2 = hyper_p_2[:-1, 0]
    b2 = float(hyper_p_2[-1, 0])

    return w1, b1, w2, b2


def predict_twin_hyperplanes(
    X: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
    rule: PredictionRule = "paper",
) -> np.ndarray:
    """
    Twin-hyperplane decision.

    paper / norm: argmin_i |w_i^T x + b_i| / ||w_i||  (published rule)
    reference / raw: sign(|w2^T x + b2| - |w1^T x + b1|)  (repository rule)
    """
    X = np.asarray(X, dtype=float)
    y1 = X @ w1 + b1
    y2 = X @ w2 + b2

    resolved = resolve_prediction_rule(rule)
    if resolved == "reference":
        return np.sign(np.abs(y2) - np.abs(y1))

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    d1 = np.abs(y1) / (norm1 if norm1 > 0 else 1.0)
    d2 = np.abs(y2) / (norm2 if norm2 > 0 else 1.0)
    return np.where(d1 <= d2, 1.0, -1.0)


class GBLSTSVM(BaseEstimator, ClassifierMixin):
    """
    sklearn-style linear GBLSTSVM classifier.

    Parameters
    ----------
    c1, c2 : float
        Trade-off between the two squared-loss terms in each twin subproblem.
    purity_threshold : float
        Granular-ball splitting stops when purity >= threshold.
    min_samples_per_ball : int
        Minimum ball size required to continue splitting.
    min_ball_count : int
        Balls with fewer samples are removed after splitting (delbals).
    reg_term : float
        Diagonal Tikhonov stabilization added to normal equations (reference: 1e-4).
    prediction_rule : 'paper' | 'reference'
        Distance rule for classification at predict time.
    random_state : int | None
        Seed for NumPy RNG before ball generation (KMeans in reference code is
        unseeded; set for partial reproducibility of splits).
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
    ) -> None:
        self.c1 = c1
        self.c2 = c2
        self.purity_threshold = purity_threshold
        self.min_samples_per_ball = min_samples_per_ball
        self.min_ball_count = min_ball_count
        self.reg_term = reg_term
        self.prediction_rule = prediction_rule
        self.random_state = random_state
        self.w1_: np.ndarray | None = None
        self.b1_: float | None = None
        self.w2_: np.ndarray | None = None
        self.b2_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBLSTSVM":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = _normalize_labels(y)

        self.balls_: list[BallInfo] = generate_granular_balls(
            X,
            y,
            purity_threshold=self.purity_threshold,
            min_samples_per_ball=self.min_samples_per_ball,
            min_ball_count=self.min_ball_count,
        )

        centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(
            self.balls_
        )

        self.centers_pos_ = centers_pos
        self.radii_pos_ = radii_pos
        self.centers_neg_ = centers_neg
        self.radii_neg_ = radii_neg

        w1, b1, w2, b2 = solve_gblstsvm_systems(
            centers_pos,
            radii_pos,
            centers_neg,
            radii_neg,
            self.c1,
            self.c2,
            reg_term=self.reg_term,
        )

        self.w1_ = w1
        self.b1_ = b1
        self.w2_ = w2
        self.b2_ = b2
        self.coef_ = np.vstack([w1, w2])
        self.intercept_ = np.array([b1, b2])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w1_ is None or self.w2_ is None:
            raise ValueError("Model is not fitted.")
        return predict_twin_hyperplanes(
            X,
            self.w1_,
            self.b1_,
            self.w2_,
            self.b2_,
            rule=self.prediction_rule,
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score

        y = _normalize_labels(y)
        return float(accuracy_score(y, self.predict(X)))
