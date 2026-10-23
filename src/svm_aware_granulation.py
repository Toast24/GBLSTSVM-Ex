"""
SVM-aware (decision-boundary-aware) granular-ball generation — Proposal 1.

Idea: the baseline generator splits a ball only while its class purity is below
a threshold. Here, granularity is instead allocated by an estimate of how much
a ball's abstraction matters to the *downstream learner's* decision boundary.

Cheap proxy for margin sensitivity (no per-split retraining):

1. Coarse start: generate balls with a relaxed purity threshold.
2. For a few rounds: fit the downstream learner once on the current balls
   (closed form for GBLSTSVM, so this is cheap), evaluate its decision score
   s(x) on the *member samples* of every ball (O(n) per round) and compute

       S(B_i) = alpha*P_i + beta*D_i + gamma*M_i + delta*V_i

   P_i  impurity               1 - purity
   D_i  discriminative error   fraction of members the pilot misclassifies
   M_i  margin sensitivity     fraction of members inside the band
                               |s(x)| < tau * median|s|  (+ sign disagreement)
   V_i  geometric variation    (r_i / median r) gated on M_i > 0, i.e. wide
                               balls only matter when they touch the boundary

   Balls with S > theta and enough samples are split (2-means, as baseline).
3. Final fit on the refined balls.

Interior balls (M = D = 0) are never refined, so they stay coarse even when
slightly impure; boundary balls get refined even when pure.

The learner is pluggable through ``BallLearner`` (fit on balls -> decision
function), so the same generator can feed GBLSTSVM, a ball-weighted SVM, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .gblstsvm import (
    _normalize_labels,
    predict_twin_hyperplanes,
    solve_gblstsvm_systems,
)
from .granular_balls import BallInfo, GBList, GranularBall, balls_to_class_arrays

DecisionFn = Callable[[np.ndarray], np.ndarray]


class BallLearner(Protocol):
    def fit_balls(self, balls: List[BallInfo]) -> DecisionFn:
        """Fit on balls; return signed decision score (>0 => class +1)."""


# --------------------------------------------------------------------------- #
# Learner adapters
# --------------------------------------------------------------------------- #


@dataclass
class GBLSTSVMLearner:
    c1: float = 1.0
    c2: float = 0.1
    reg_term: float = 1e-4

    def fit_balls(self, balls: List[BallInfo]) -> DecisionFn:
        cp, rp, cn, rn = balls_to_class_arrays(balls)
        w1, b1, w2, b2 = solve_gblstsvm_systems(cp, rp, cn, rn, self.c1, self.c2, self.reg_term)
        n1 = np.linalg.norm(w1) or 1.0
        n2 = np.linalg.norm(w2) or 1.0

        def decision(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=float)
            # >0 when closer to the positive hyperplane (paper distance rule)
            return np.abs(X @ w2 + b2) / n2 - np.abs(X @ w1 + b1) / n1

        return decision


@dataclass
class BallSVMLearner:
    """GBSVM-style proxy: SVM on ball centers weighted by ball size."""

    C: float = 1.0
    kernel: str = "linear"
    gamma: str | float = "scale"

    def fit_balls(self, balls: List[BallInfo]) -> DecisionFn:
        from sklearn.svm import SVC

        X = np.vstack([b.center for b in balls])
        y = np.array([b.label for b in balls])
        w = np.array([b.num_samples for b in balls], dtype=float)
        clf = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        clf.fit(X, y, sample_weight=w / w.mean())
        return clf.decision_function


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


@dataclass
class SplitScoreWeights:
    alpha: float = 0.5  # impurity
    beta: float = 1.0  # pilot misclassification inside ball
    gamma: float = 1.0  # margin-band occupancy
    delta: float = 0.25  # width, only when touching the boundary


@dataclass
class GenerationTrace:
    rounds: list = field(default_factory=list)
    gen_time: float = 0.0


def _to_info(ball: GranularBall) -> BallInfo:
    return BallInfo(
        center=ball.center.copy(),
        radius=ball.r,
        label=-1.0 if ball.label == -1 else 1.0,
        purity=ball.purity,
        num_samples=ball.num,
        scatter=ball.scatter.copy(),
    )


def score_balls(
    balls: List[GranularBall],
    decision: DecisionFn,
    weights: SplitScoreWeights,
    tau: float,
) -> tuple[np.ndarray, dict]:
    """Return S(B_i) for each ball plus the component arrays."""
    X = np.vstack([b.data_no_label for b in balls])
    sizes = np.array([b.num for b in balls])
    owner = np.repeat(np.arange(len(balls)), sizes)
    y = np.concatenate([b.data[:, -2] for b in balls])

    s = decision(X)
    scale = np.median(np.abs(s)) or 1.0
    in_band = np.abs(s) < tau * scale
    wrong = np.sign(s) != y

    pos_side = np.bincount(owner, weights=(s > 0).astype(float), minlength=len(balls))
    straddle = (pos_side > 0) & (pos_side < sizes)  # boundary passes through ball

    P = 1.0 - np.array([b.purity for b in balls])
    D = np.bincount(owner, weights=wrong.astype(float), minlength=len(balls)) / sizes
    M = np.bincount(owner, weights=in_band.astype(float), minlength=len(balls)) / sizes
    M = np.maximum(M, straddle.astype(float))
    radii = np.array([b.r for b in balls])
    V = radii / (np.median(radii) or 1.0) * (M > 0)

    S = weights.alpha * P + weights.beta * D + weights.gamma * M + weights.delta * V
    return S, {"P": P, "D": D, "M": M, "V": V}


def generate_svm_aware_balls(
    X: np.ndarray,
    y: np.ndarray,
    learner: BallLearner,
    coarse_purity: float = 0.8,
    rounds: int = 6,
    theta: float = 0.6,
    tau: float = 0.25,
    min_samples_per_ball: int = 4,
    weights: Optional[SplitScoreWeights] = None,
) -> tuple[List[BallInfo], GenerationTrace]:
    weights = weights or SplitScoreWeights()
    t0 = time.perf_counter()
    data = np.column_stack([X, y, np.arange(len(y))])
    # Coarse start must not swallow the minority class: require purity above
    # the majority share so the root is always split at least to class level.
    minority = min(np.mean(y == 1), np.mean(y != 1))
    start_purity = max(coarse_purity, 1.0 - 0.5 * minority)
    gb = GBList(data)
    gb.init_granular_balls(purity=start_purity, min_sample=min_samples_per_ball)
    balls: List[GranularBall] = gb.granular_balls
    trace = GenerationTrace()

    for r in range(rounds):
        infos = [_to_info(b) for b in balls]
        if len({b.label for b in infos}) < 2:
            break
        decision = learner.fit_balls(infos)
        S, comps = score_balls(balls, decision, weights, tau)
        to_split = [i for i, b in enumerate(balls) if S[i] > theta and b.num > min_samples_per_ball]
        trace.rounds.append(
            {"round": r, "n_balls": len(balls), "n_split": len(to_split),
             "mean_M": float(comps["M"].mean())}
        )
        if not to_split:
            break
        split_set = set(to_split)
        new: List[GranularBall] = [b for i, b in enumerate(balls) if i not in split_set]
        for i in to_split:
            new.extend(b for b in balls[i].split_2balls() if b.num > 0)
        balls = new

    trace.gen_time = time.perf_counter() - t0
    return [_to_info(b) for b in balls], trace


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #


class SVMAwareGBLSTSVM(BaseEstimator, ClassifierMixin):
    """GBLSTSVM trained on SVM-aware granular balls."""

    def __init__(
        self,
        c1: float = 1.0,
        c2: float = 0.1,
        reg_term: float = 1e-4,
        coarse_purity: float = 0.8,
        rounds: int = 6,
        theta: float = 0.6,
        tau: float = 0.25,
        min_samples_per_ball: int = 4,
        min_ball_count: int = 2,
        prediction_rule: str = "reference",
        random_state: Optional[int] = None,
    ) -> None:
        self.c1 = c1
        self.c2 = c2
        self.reg_term = reg_term
        self.coarse_purity = coarse_purity
        self.rounds = rounds
        self.theta = theta
        self.tau = tau
        self.min_samples_per_ball = min_samples_per_ball
        self.min_ball_count = min_ball_count
        self.prediction_rule = prediction_rule
        self.random_state = random_state

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=float)
        y = _normalize_labels(y)
        learner = GBLSTSVMLearner(self.c1, self.c2, self.reg_term)
        balls, self.trace_ = generate_svm_aware_balls(
            X, y, learner,
            coarse_purity=self.coarse_purity, rounds=self.rounds, theta=self.theta,
            tau=self.tau, min_samples_per_ball=self.min_samples_per_ball,
        )
        kept = [b for b in balls if b.num_samples >= self.min_ball_count]
        if len({b.label for b in kept}) == 2:
            balls = kept
        self.balls_ = balls
        cp, rp, cn, rn = balls_to_class_arrays(balls)
        self.w1_, self.b1_, self.w2_, self.b2_ = solve_gblstsvm_systems(
            cp, rp, cn, rn, self.c1, self.c2, self.reg_term
        )
        return self

    def predict(self, X):
        return predict_twin_hyperplanes(
            X, self.w1_, self.b1_, self.w2_, self.b2_, rule=self.prediction_rule
        )
