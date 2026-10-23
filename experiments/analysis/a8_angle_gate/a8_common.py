"""
A8 — Angle-pipeline gate and clean compression geometry.

Experiment-only. Reuses the A4/A6 assembly conventions exactly:
  per-class H = sum_j n_j u_j u_j^T + blockdiag(S_j^(arm), 0),  g = sum_j n_j u_j
  (L0: unit weights), twin solve with ridge_scale(comb, REG_BASE).

Paths (same granulation run, same seed):
  PARTITION : rho-granulation, no deletion (every train sample covered)
  DELETION  : same balls, drop balls with n < 2 (A4/A6 baseline path)

RAW is the per-sample LSTSVM; RAW_RETAINED is per-sample LSTSVM on exactly the
samples that survive deletion (FULL on DELETION must equal it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from experiments.estimator_information_ladder_common import (
    MIN_BALL_COUNT_BASELINE,
    REG_BASE,
    _split_rho_granulation,
    split_balls,
)
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels
from experiments.rebaseline_common import C_GRID
from experiments.test_split_utility_granulation import _build_indexed_matrix
from sklearn.metrics import f1_score
from src.gblstsvm import predict_twin_hyperplanes
from src.granular_balls import BallInfo, decompose_granular_balls
from src.ladder_gblstsvm import ridge_scale

Arm = Literal["L0", "L1", "DIAG", "ISO", "FULL"]
ARMS: tuple[Arm, ...] = ("L0", "L1", "DIAG", "ISO", "FULL")


@dataclass
class Granulation:
    partition: list[BallInfo]
    deletion: list[BallInfo]
    retained_idx: np.ndarray
    n_raw_balls: int
    n_kept_raw_balls: int


def granulate(X: np.ndarray, y: np.ndarray, seed: int, rho: float) -> Granulation:
    """One granulation run; deletion path is a filter of the same balls."""
    indexed = _build_indexed_matrix(X, y)
    balls = _split_rho_granulation(indexed, seed=seed, purity=rho, min_ball_count=0)
    kept = [b for b in balls if int(b.num) >= MIN_BALL_COUNT_BASELINE]
    retained = np.sort(np.concatenate([b.data[:, -1] for b in kept]).astype(int))
    return Granulation(
        partition=decompose_granular_balls(balls),
        deletion=decompose_granular_balls(kept),
        retained_idx=retained,
        n_raw_balls=len(balls),
        n_kept_raw_balls=len(kept),
    )


def partition_members(X: np.ndarray, y: np.ndarray, seed: int, rho: float) -> list[tuple[float, np.ndarray]]:
    """Class-pure partition-path balls as (label, train-row indices); same order as
    granulate(...).partition (decompose_granular_balls iterates np.unique(labels))."""
    indexed = _build_indexed_matrix(X, y)
    out: list[tuple[float, np.ndarray]] = []
    for ball in _split_rho_granulation(indexed, seed=seed, purity=rho, min_ball_count=0):
        labels = ball.data[:, -2]
        idx = ball.data[:, -1].astype(int)
        for cls in np.unique(labels):
            out.append((-1.0 if cls == -1 else 1.0, idx[labels == cls]))
    return out


def _scatter_block(S: np.ndarray, arm: Arm) -> np.ndarray | None:
    if arm in ("L0", "L1"):
        return None
    if arm == "FULL":
        return S
    if arm == "DIAG":
        return np.diag(np.diag(S))
    if arm == "ISO":
        return (float(np.trace(S)) / S.shape[0]) * np.eye(S.shape[0])
    raise ValueError(arm)


def class_moments(balls: Sequence[BallInfo], arm: Arm) -> tuple[np.ndarray, np.ndarray]:
    dim = int(balls[0].center.shape[0])
    C = np.vstack([b.center for b in balls])
    U = np.hstack([C, np.ones((C.shape[0], 1))])
    n = np.array([1.0 if arm == "L0" else float(b.num_samples) for b in balls])
    H = (U * n[:, None]).T @ U
    g = U.T @ n
    if arm not in ("L0", "L1"):
        S_tot = np.zeros((dim, dim))
        for b in balls:
            S_tot += _scatter_block(np.asarray(b.scatter, dtype=float), arm)
        H[:dim, :dim] += S_tot
    return H, g


def sample_moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U = np.hstack([X, np.ones((X.shape[0], 1))])
    return U.T @ U, U.sum(0)


def twin_solve(
    Ha: np.ndarray, ga: np.ndarray, Hb: np.ndarray, gb: np.ndarray, c1: float, c2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return augmented z1=[w1;b1], z2=[w2;b2] (A4/A6 solve, identical ridge)."""
    eye = np.identity(Ha.shape[0])
    comb1 = Hb + (1.0 / c1) * Ha
    z1 = -np.linalg.solve(comb1 + ridge_scale(comb1, REG_BASE) * eye, gb)
    comb2 = Ha + (1.0 / c2) * Hb
    z2 = np.linalg.solve(comb2 + ridge_scale(comb2, REG_BASE) * eye, ga)
    return z1, z2


def fit_balls(balls: Sequence[BallInfo], arm: Arm, c1: float, c2: float):
    pos, neg = split_balls(list(balls))
    Ha, ga = class_moments(pos, arm)
    Hb, gb = class_moments(neg, arm)
    return twin_solve(Ha, ga, Hb, gb, c1, c2)


def fit_raw(X: np.ndarray, y: np.ndarray, c1: float, c2: float):
    y = normalize_labels(y)
    Ha, ga = sample_moments(X[y == 1.0])
    Hb, gb = sample_moments(X[y != 1.0])
    return twin_solve(Ha, ga, Hb, gb, c1, c2)


def predict(X: np.ndarray, z1: np.ndarray, z2: np.ndarray, rule: str = "reference") -> np.ndarray:
    return predict_twin_hyperplanes(X, z1[:-1], float(z1[-1]), z2[:-1], float(z2[-1]), rule=rule)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yp = np.asarray(y_pred, dtype=float).copy()
    yp[yp == 0.0] = -1.0
    return float(f1_score(normalize_labels(y_true), yp, average="macro", zero_division=0))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """A7 definition: unsigned angle between normal vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not (np.isfinite(na) and np.isfinite(nb)):
        return float("nan")
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b)) / (na * nb)))))


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def tune_c(fit_fn, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float, float]:
    """Grid over C_GRID^2; fit_fn(c1, c2) -> (z1, z2) trained WITHOUT validation rows."""
    best = (DEFAULT_C1, DEFAULT_C2, -np.inf)
    for c1 in C_GRID:
        for c2 in C_GRID:
            try:
                z1, z2 = fit_fn(c1, c2)
                f = macro_f1(y_val, predict(X_val, z1, z2))
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(f) and f > best[2]:
                best = (c1, c2, f)
    return best


def within_fraction(balls: Sequence[BallInfo], X_class: np.ndarray) -> float:
    """tr(sum_j S_j) / tr(total class scatter): variance discarded by center-only."""
    tot = X_class - X_class.mean(0)
    denom = float(np.sum(tot * tot))
    return float(sum(np.trace(b.scatter) for b in balls) / denom) if denom > 0 else float("nan")


def iso_mismatch(balls: Sequence[BallInfo]) -> float:
    """||S_W - (tr S_W / d) I||_F / ||S_W||_F for pooled within-ball scatter S_W."""
    S = sum(np.asarray(b.scatter, dtype=float) for b in balls)
    d = S.shape[0]
    nrm = np.linalg.norm(S, "fro")
    if nrm == 0:
        return float("nan")
    return float(np.linalg.norm(S - (np.trace(S) / d) * np.eye(d), "fro") / nrm)
