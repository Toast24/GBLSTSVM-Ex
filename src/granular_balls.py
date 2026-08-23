"""
Granular-ball generation aligned with the reference repository (gen_ball.py).

Data layout inside ball construction:
  features | class label (-2) | sample index (-1)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class BallInfo:
    """Summary of one granular ball."""

    center: np.ndarray
    radius: float
    label: float
    purity: float
    num_samples: int


class GranularBall:
    """Single granular ball (reference: gen_ball.GranularBall)."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data[:, :]
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.label, self.purity, self.r = self._label_purity_radius()

    def _label_purity_radius(self) -> Tuple[float, float, float]:
        count = Counter(self.data[:, -2])
        label = max(count, key=count.get)
        purity = count[label] / self.num
        arr = np.array(self.data_no_label) - self.center
        distances = np.sqrt(np.sum(np.square(arr), axis=1))
        radius = float(np.sum(distances) / len(self.data_no_label))
        return float(label), float(purity), radius

    def split_2balls(self) -> Tuple["GranularBall", "GranularBall"]:
        clu = KMeans(n_clusters=2).fit(self.data_no_label)
        label_cluster = clu.labels_
        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :])
            ball2 = GranularBall(self.data[label_cluster == 1, :])
        else:
            ball1 = GranularBall(self.data[0:1, :])
            ball2 = GranularBall(self.data[1:, :])
        return ball1, ball2


class GBList:
    """List of granular balls (reference: gen_ball.GBList)."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data[:, :]
        self.granular_balls: List[GranularBall] = [GranularBall(self.data)]

    def init_granular_balls(self, purity: float = 1.0, min_sample: int = 1) -> None:
        ll = len(self.granular_balls)
        i = 0
        while True:
            ball = self.granular_balls[i]
            if ball.purity < purity and ball.num > min_sample:
                split_balls = ball.split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        self.data = self.get_data()

    def get_center(self) -> np.ndarray:
        return np.array([ball.center for ball in self.granular_balls])

    def get_r(self) -> np.ndarray:
        return np.array([ball.r for ball in self.granular_balls])

    def get_purity(self) -> List[float]:
        return [ball.purity for ball in self.granular_balls]

    def get_data(self) -> np.ndarray:
        return np.vstack([ball.data for ball in self.granular_balls])

    def del_ball(self, purty: float = 0.0, num_data: int = 0) -> None:
        kept = [
            ball
            for ball in self.granular_balls
            if ball.purity >= purty and ball.num >= num_data
        ]
        self.granular_balls = kept
        self.data = self.get_data()


def generate_granular_balls(
    X: np.ndarray,
    y: np.ndarray,
    purity_threshold: float = 1.0,
    min_samples_per_ball: int = 1,
    min_ball_count: int = 0,
) -> List[BallInfo]:
    """
    Build granular balls from feature matrix X and labels y.

    Parameters mirror gen_ball.generate_ball_data / main.py usage:
      purity_threshold — stop splitting when purity >= threshold
      min_samples_per_ball — minimum samples required to keep splitting
      min_ball_count — remove balls with fewer than this many samples (delbals)
    """
    data = np.column_stack([X, y])
    num = data.shape[0]
    index = np.arange(num).reshape(num, 1)
    indexed = np.hstack([data, index])

    gb = GBList(indexed)
    gb.init_granular_balls(purity=purity_threshold, min_sample=min_samples_per_ball)
    if min_ball_count > 0:
        gb.del_ball(num_data=min_ball_count)

    balls: List[BallInfo] = []
    for ball in gb.granular_balls:
        label = -1.0 if ball.label == -1 else 1.0
        balls.append(
            BallInfo(
                center=ball.center.copy(),
                radius=ball.r,
                label=label,
                purity=ball.purity,
                num_samples=ball.num,
            )
        )
    return balls


def generate_granular_balls_with_membership(
    X: np.ndarray,
    y: np.ndarray,
    purity_threshold: float = 1.0,
    min_samples_per_ball: int = 1,
    min_ball_count: int = 0,
) -> Tuple[List[BallInfo], List[np.ndarray]]:
    """
    Build granular balls and return per-ball training sample indices.

    Index arrays align with the returned ``BallInfo`` list (same order as
    ``generate_granular_balls``).
    """
    data = np.column_stack([X, y])
    num = data.shape[0]
    index = np.arange(num).reshape(num, 1)
    indexed = np.hstack([data, index])

    gb = GBList(indexed)
    gb.init_granular_balls(purity=purity_threshold, min_sample=min_samples_per_ball)
    if min_ball_count > 0:
        gb.del_ball(num_data=min_ball_count)

    balls: List[BallInfo] = []
    memberships: List[np.ndarray] = []
    for ball in gb.granular_balls:
        label = -1.0 if ball.label == -1 else 1.0
        balls.append(
            BallInfo(
                center=ball.center.copy(),
                radius=ball.r,
                label=label,
                purity=ball.purity,
                num_samples=ball.num,
            )
        )
        memberships.append(np.asarray(ball.data[:, -1], dtype=int))
    return balls, memberships


def balls_to_class_arrays(
    balls: Sequence[BallInfo],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ball summaries into positive (+1) and negative (-1) center/radius arrays."""
    pos = [b for b in balls if b.label == 1]
    neg = [b for b in balls if b.label != 1]

    if not pos or not neg:
        raise ValueError("GBLSTSVM requires at least one ball per class.")

    centers_pos = np.vstack([b.center for b in pos])
    radii_pos = np.array([b.radius for b in pos], dtype=float)
    centers_neg = np.vstack([b.center for b in neg])
    radii_neg = np.array([b.radius for b in neg], dtype=float)
    return centers_pos, radii_pos, centers_neg, radii_neg
