"""
Margin-aware adaptive multi-granularity (Proposal 2).

Builds an explicit hierarchy G_0 -> G_1 -> ... -> G_k in which each level
subdivides only the balls whose estimated importance to the downstream
learner's decision boundary is high. Compared with the single-score refiner in
``svm_aware_granulation`` this adds region-specific treatment:

  region      rule                                              treatment
  interior    margin occupancy M = 0 and pure                   keep coarse
  boundary    S(B) > theta                                      split
  noise       small ball whose members' k-NN are mostly enemies frozen, weight
              and which the pilot misclassifies                 down to w_noise
  minority    ball labelled with the minority class             threshold
                                                                scaled by
                                                                minority_relax

S(B) = P-mix of impurity, pilot error, margin occupancy, width (as in
Proposal 1) plus a neighbour-structure term N = mean fraction of each member's
k nearest training neighbours carrying the other label.

Every ball records the level at which it was created and a weight; learners
consume the weights (weighted GBLSTSVM / weighted ball-SVM).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .granular_balls import GBList, GranularBall
from .svm_aware_granulation import SplitScoreWeights, score_balls

DecisionFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class WeightedBall:
    center: np.ndarray
    radius: float
    label: float
    purity: float
    num_samples: int
    level: int
    weight: float = 1.0
    region: str = "interior"


@dataclass
class Hierarchy:
    levels: List[List[WeightedBall]] = field(default_factory=list)
    gen_time: float = 0.0

    @property
    def final(self) -> List[WeightedBall]:
        return self.levels[-1]


# --------------------------------------------------------------------------- #
# Weighted learners
# --------------------------------------------------------------------------- #


def _split(balls: List[WeightedBall]):
    pos = [b for b in balls if b.label == 1]
    neg = [b for b in balls if b.label != 1]
    if not pos or not neg:
        raise ValueError("need balls of both classes")
    arr = lambda bs: (
        np.vstack([b.center for b in bs]),
        np.array([b.radius for b in bs]),
        np.array([getattr(b, "weight", 1.0) * b.num_samples for b in bs], dtype=float),
    )
    return arr(pos), arr(neg)


@dataclass
class WeightedGBLSTSVMLearner:
    """GBLSTSVM with each ball's squared residual multiplied by its weight."""

    c1: float = 1.0
    c2: float = 0.1
    reg_term: float = 1e-4
    use_counts: bool = False  # False: weight only (matches unweighted GBLSTSVM when w=1)

    def fit_balls(self, balls: List[WeightedBall]) -> DecisionFn:
        (A, Rp, wp), (B, Rn, wn) = _split(balls)
        if not self.use_counts:
            wp = np.array([getattr(b, "weight", 1.0) for b in balls if b.label == 1])
            wn = np.array([getattr(b, "weight", 1.0) for b in balls if b.label != 1])
        sp, sn = np.sqrt(wp)[:, None], np.sqrt(wn)[:, None]
        H = np.column_stack([A, np.ones(len(A))]) * sp
        G = np.column_stack([B, np.ones(len(B))]) * sn
        eye = self.reg_term * np.eye(H.shape[1])
        t1 = -np.linalg.solve(G.T @ G + H.T @ H / self.c1 + eye, G.T @ (sn * (1 + Rn[:, None])))
        t2 = np.linalg.solve(H.T @ H + G.T @ G / self.c2 + eye, H.T @ (sp * (1 + Rp[:, None])))
        w1, b1, w2, b2 = t1[:-1, 0], t1[-1, 0], t2[:-1, 0], t2[-1, 0]
        n1, n2 = np.linalg.norm(w1) or 1.0, np.linalg.norm(w2) or 1.0
        return lambda X: np.abs(X @ w2 + b2) / n2 - np.abs(X @ w1 + b1) / n1


@dataclass
class WeightedBallSVMLearner:
    C: float = 1.0
    kernel: str = "linear"

    def fit_balls(self, balls: List[WeightedBall]) -> DecisionFn:
        from sklearn.svm import SVC

        X = np.vstack([b.center for b in balls])
        y = np.array([b.label for b in balls])
        w = np.array([getattr(b, "weight", 1.0) * b.num_samples for b in balls], dtype=float)
        return SVC(C=self.C, kernel=self.kernel).fit(X, y, sample_weight=w / w.mean()).decision_function


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


def _enemy_ratio(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(y) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    return (y[idx] != y[:, None]).mean(1)


def _wrap(ball: GranularBall, level: int, weight: float, region: str) -> WeightedBall:
    return WeightedBall(
        center=ball.center.copy(), radius=ball.r,
        label=-1.0 if ball.label == -1 else 1.0, purity=ball.purity,
        num_samples=ball.num, level=level, weight=weight, region=region,
    )


def build_margin_hierarchy(
    X: np.ndarray,
    y: np.ndarray,
    learner,
    max_levels: int = 6,
    coarse_purity: float = 0.8,
    theta: float = 0.6,
    tau: float = 0.25,
    nu: float = 0.5,
    k: int = 7,
    noise_size: int = 3,
    noise_enemy: float = 0.7,
    w_noise: float = 0.2,
    minority_relax: float = 0.7,
    min_samples_per_ball: int = 4,
    weights: Optional[SplitScoreWeights] = None,
) -> Hierarchy:
    """Build G_0..G_k. ``nu`` weights the neighbour-structure term."""
    weights = weights or SplitScoreWeights()
    t0 = time.perf_counter()
    y = np.asarray(y, dtype=float)
    minority = min(np.mean(y == 1), np.mean(y != 1))
    minority_label = 1.0 if np.mean(y == 1) <= 0.5 else -1.0
    enemy = _enemy_ratio(X, y, k)

    gb = GBList(np.column_stack([X, y, np.arange(len(y))]))
    gb.init_granular_balls(purity=max(coarse_purity, 1 - 0.5 * minority),
                           min_sample=min_samples_per_ball)
    active = gb.granular_balls
    level = [0] * len(active)
    w = [1.0] * len(active)
    region = ["interior"] * len(active)
    frozen = [False] * len(active)
    h = Hierarchy()
    snap = lambda: [_wrap(b, l, ww, r) for b, l, ww, r in zip(active, level, w, region)]
    h.levels.append(snap())

    for lev in range(1, max_levels + 1):
        decision = learner.fit_balls(h.levels[-1])
        S, comps = score_balls(active, decision, weights, tau)
        N = np.array([enemy[b.data[:, -1].astype(int)].mean() for b in active])
        S = S + nu * N

        nxt, nl, nw, nr, nf = [], [], [], [], []
        n_split = 0
        for i, b in enumerate(active):
            lbl = -1.0 if b.label == -1 else 1.0
            thr = theta * (minority_relax if lbl == minority_label else 1.0)
            is_noise = b.num <= noise_size and N[i] >= noise_enemy and comps["D"][i] > 0.5
            if frozen[i] or is_noise:
                nxt.append(b); nl.append(level[i]); nf.append(True)
                nw.append(w_noise if is_noise else w[i])
                nr.append("noise" if is_noise else region[i])
            elif S[i] > thr and b.num > min_samples_per_ball:
                n_split += 1
                for c in b.split_2balls():
                    if c.num:
                        nxt.append(c); nl.append(lev); nw.append(1.0); nf.append(False)
                        nr.append("minority" if lbl == minority_label else "boundary")
            else:
                nxt.append(b); nl.append(level[i]); nw.append(w[i]); nf.append(frozen[i])
                nr.append(region[i] if comps["M"][i] > 0 else "interior")
        active, level, w, region, frozen = nxt, nl, nw, nr, nf
        h.levels.append(snap())
        if n_split == 0:
            break

    h.gen_time = time.perf_counter() - t0
    return h
