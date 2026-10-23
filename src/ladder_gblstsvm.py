"""Ladder arms L0–L3 for GBLSTSVM normal-equation assembly."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .granular_balls import BallInfo

LadderArm = Literal["L0", "L1", "L2", "L3"]
ARMS: tuple[LadderArm, ...] = ("L0", "L1", "L2", "L3")
L3_K2_ARM: LadderArm = "L3"


def _ball_scatter(ball: BallInfo, dim: int) -> np.ndarray:
    if ball.scatter is not None:
        return ball.scatter
    return np.zeros((dim, dim), dtype=float)


def accumulate_class(
    balls: Sequence[BallInfo],
    arm: LadderArm,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (design, rhs) for one class under the given ladder arm."""
    if not balls:
        raise ValueError("accumulate_class requires at least one ball.")
    dim = int(balls[0].center.shape[0])
    design = np.zeros((dim + 1, dim + 1), dtype=float)
    rhs = np.zeros(dim + 1, dtype=float)
    for ball in balls:
        u = np.concatenate([np.asarray(ball.center, dtype=float), [1.0]])
        weight = 1.0 if arm == "L0" else float(ball.num_samples)
        design += weight * np.outer(u, u)
        if arm in ("L2", "L3"):
            design[:dim, :dim] += _ball_scatter(ball, dim)
        if arm in ("L0", "L1", "L2"):
            rhs += weight * u * (1.0 + float(ball.radius))
        else:
            rhs += float(ball.num_samples) * u
    return design, rhs


def ridge_scale(combined_design: np.ndarray, reg_base: float) -> float:
    """Ridge λ = reg_base × mean diagonal of the combined design block."""
    dim = combined_design.shape[0]
    return reg_base * float(np.trace(combined_design)) / dim


def solve_ladder_systems(
    pos_balls: Sequence[BallInfo],
    neg_balls: Sequence[BallInfo],
    c1: float,
    c2: float,
    arm: LadderArm,
    reg_base: float = 1e-4,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Solve twin hyperplanes under ladder arm (per-class design/RHS assembly)."""
    if c1 <= 0 or c2 <= 0:
        raise ValueError("c1 and c2 must be positive.")

    design_a, rhs_a = accumulate_class(pos_balls, arm)
    design_b, rhs_b = accumulate_class(neg_balls, arm)
    dim = design_a.shape[0]
    eye = np.identity(dim, dtype=float)

    comb1 = design_b + (1.0 / c1) * design_a
    reg1 = ridge_scale(comb1, reg_base)
    inv_p_1 = np.linalg.inv(comb1 + reg1 * eye)
    hyper_p_1 = -inv_p_1 @ rhs_b.reshape(-1, 1)

    comb2 = design_a + (1.0 / c2) * design_b
    reg2 = ridge_scale(comb2, reg_base)
    inv_p_2 = np.linalg.inv(comb2 + reg2 * eye)
    hyper_p_2 = inv_p_2 @ rhs_a.reshape(-1, 1)

    w1 = hyper_p_1[:-1, 0]
    b1 = float(hyper_p_1[-1, 0])
    w2 = hyper_p_2[:-1, 0]
    b2 = float(hyper_p_2[-1, 0])
    return w1, b1, w2, b2


def balls_from_samples(X: np.ndarray, y: np.ndarray) -> tuple[list[BallInfo], list[BallInfo]]:
    """One ball per training sample (raw LSTSVM sufficient statistics)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    y[y == 0.0] = -1.0
    dim = X.shape[1]
    pos: list[BallInfo] = []
    neg: list[BallInfo] = []
    zscatter = np.zeros((dim, dim), dtype=float)
    for i in range(X.shape[0]):
        ball = BallInfo(
            center=X[i].copy(),
            radius=0.0,
            label=float(y[i]),
            purity=1.0,
            num_samples=1,
            scatter=zscatter,
        )
        if y[i] == 1.0:
            pos.append(ball)
        else:
            neg.append(ball)
    return pos, neg


def _class_ball(X: np.ndarray, label: float) -> BallInfo:
    n_j = int(X.shape[0])
    center = X.mean(0)
    diff = X - center
    scatter = diff.T @ diff
    dist = np.sqrt(np.sum(np.square(diff), axis=1))
    radius = float(np.sum(dist) / max(n_j, 1))
    return BallInfo(
        center=center.copy(),
        radius=radius,
        label=label,
        purity=1.0,
        num_samples=n_j,
        scatter=scatter,
    )


def balls_k2_from_train(X: np.ndarray, y: np.ndarray) -> tuple[list[BallInfo], list[BallInfo]]:
    """Exactly one ball per class: (n_j, c_j, S_j) over all same-class train points."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    y[y == 0.0] = -1.0
    pos = _class_ball(X[y == 1.0], 1.0)
    neg = _class_ball(X[y == -1.0], -1.0)
    if pos.num_samples == 0 or neg.num_samples == 0:
        raise ValueError("balls_k2_from_train requires both classes in training data.")
    return [pos], [neg]


def solve_l3_k2_systems(
    X: np.ndarray,
    y: np.ndarray,
    c1: float,
    c2: float,
    reg_base: float = 1e-4,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """L3_k2: k=2 class-aggregated balls, L3 normal equations."""
    pos, neg = balls_k2_from_train(X, y)
    return solve_ladder_systems(pos, neg, c1=c1, c2=c2, arm=L3_K2_ARM, reg_base=reg_base)
