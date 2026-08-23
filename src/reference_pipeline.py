"""
Thin wrapper around the original reference implementation modules.

The reference code lives at the repository root (gen_ball.py, GBLSTSVM.py)
and is not modified. This module reproduces the main.py data pipeline for
validation comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from gen_ball import GBList
from GBLSTSVM import LSTWGBSVM


@dataclass
class ReferenceBall:
    center: np.ndarray
    radius: float
    label: float
    purity: float


def reference_gen_balls(
    X: np.ndarray,
    y: np.ndarray,
    purity_threshold: float,
    min_ball_count: int,
) -> List[ReferenceBall]:
    """Run reference granular-ball generation (gen_ball.GBList pipeline)."""
    data = np.column_stack([X, y])
    num = data.shape[0]
    index = np.arange(num).reshape(num, 1)
    indexed = np.hstack([data, index])

    gb = GBList(indexed)
    gb.init_granular_balls(purity=purity_threshold)
    if min_ball_count > 0:
        gb.del_ball(num_data=min_ball_count)

    balls: List[ReferenceBall] = []
    for ball in gb.granular_balls:
        label = -1.0 if ball.label == -1 else 1.0
        balls.append(
            ReferenceBall(
                center=ball.center.copy(),
                radius=float(ball.r),
                label=label,
                purity=float(ball.purity),
            )
        )
    return balls


def balls_to_training_matrix(balls: List[ReferenceBall]) -> np.ndarray:
    """Format balls as [center..., radius, label] matrix for LSTWGBSVM."""
    rows = []
    for ball in balls:
        row = np.concatenate([ball.center, [ball.radius, ball.label]])
        rows.append(row)
    return np.vstack(rows)


def reference_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c1: float,
    c2: float,
    purity_threshold: float = 0.985,
    min_ball_count: int = 2,
) -> Tuple[np.ndarray, float, List[ReferenceBall]]:
    """
    End-to-end reference pipeline: gen_balls + LSTWGBSVM.

    Returns predictions on X_test (via internal LSTWGBSVM logic), accuracy,
    and the generated balls.
    """
    balls = reference_gen_balls(
        X_train, y_train, purity_threshold, min_ball_count
    )
    train_matrix = balls_to_training_matrix(balls)

    test_matrix = np.column_stack([X_test, y_test])
    accuracy, _elapsed = LSTWGBSVM(train_matrix, test_matrix, c1, c2)

    # Recompute predictions explicitly for return value
    predictions = reference_predict_from_matrix(train_matrix, X_test, c1, c2)
    return predictions, accuracy, balls


def reference_predict_from_matrix(
    train_matrix: np.ndarray,
    X_test: np.ndarray,
    c1: float,
    c2: float,
    reg_term: float = 1e-4,
) -> np.ndarray:
    """Apply reference hyperplane solve + prediction rule."""
    from src.gblstsvm import solve_gblstsvm_systems

    C1 = train_matrix[train_matrix[:, -1] == 1, :-1]
    C2 = train_matrix[train_matrix[:, -1] != 1, :-1]
    A = C1[:, :-1]
    B = C2[:, :-1]
    R_pos = C1[:, -1]
    R_neg = C2[:, -1]

    w1, b1, w2, b2 = solve_gblstsvm_systems(
        A, R_pos, B, R_neg, c1, c2, reg_term=reg_term
    )

    y1 = X_test @ w1 + b1
    y2 = X_test @ w2 + b2
    return np.sign(np.abs(y2) - np.abs(y1))
