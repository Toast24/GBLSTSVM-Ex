"""
Isolation-Penalized Ambiguity (IPA) fuzzy ball membership for GBLSTSVM.

Combines local class ambiguity (A2 competition remap) with a within-class
isolation penalty. Uses only ball centers and labels; no radius, margin,
overlap, or inverse-density weighting.

  ambiguity_i = 2 * min(comp_i, 1 - comp_i)
  muIPA_i     = ambiguity_i / (1 + lambda * max(0, q_i/Q_y - 1))

where comp_i is the Gaussian-weighted opposite-class k-NN fraction and
q_i is the nearest same-class center distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .cbcm import (
    DEFAULT_SIGMA_EPS,
    compute_competition_membership,
    resolve_k,
    split_membership_by_class,
)
from .class_weights import ball_class_weights
from .granular_balls import BallInfo


DEFAULT_LAMBDA = 1.0


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=float).ravel().copy()
    y[y == 0] = -1.0
    return y


def compute_ambiguity(comp: np.ndarray) -> np.ndarray:
    """A2 ambiguity remap: 2 * min(comp, 1 - comp)."""
    c = np.asarray(comp, dtype=float).ravel()
    return 2.0 * np.minimum(c, 1.0 - c)


def compute_nearest_same_class_distance(
    centers: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-ball Euclidean distance to the nearest other same-class center."""
    centers = np.asarray(centers, dtype=float)
    labels = _normalize_labels(labels)
    n = centers.shape[0]
    nearest_same = np.full(n, np.inf, dtype=float)

    for i in range(n):
        same_mask = (labels == labels[i]) & (np.arange(n) != i)
        if not np.any(same_mask):
            continue
        dists = np.linalg.norm(centers[i] - centers[same_mask], axis=1)
        nearest_same[i] = float(np.min(dists))

    return nearest_same


def compute_isolation(
    nearest_same: np.ndarray,
    labels: np.ndarray,
    *,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Within-class normalized isolation.

    Returns
    -------
    isolation_clipped : min(1, q_i / Q_y) for diagnostics
    raw_isolation     : q_i / Q_y (unclipped, used for penalty)
    """
    nearest_same = np.asarray(nearest_same, dtype=float).ravel()
    labels = _normalize_labels(labels)
    n = nearest_same.shape[0]
    raw = np.zeros(n, dtype=float)
    clipped = np.zeros(n, dtype=float)

    for y in (-1.0, 1.0):
        mask = labels == y
        if not np.any(mask):
            continue
        q = nearest_same[mask]
        finite_q = q[np.isfinite(q)]
        if finite_q.size == 0:
            continue
        q_y = float(np.median(finite_q))
        if q_y <= 0.0:
            q_y = sigma_eps
        ratio = q / q_y
        ratio[~np.isfinite(q)] = 0.0
        raw[mask] = ratio
        clipped[mask] = np.minimum(1.0, ratio)

    return clipped, raw


def compute_ipa_membership(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
    lam: float = DEFAULT_LAMBDA,
) -> "IPAMembershipResult":
    """Compute IPA membership for ball centers and labels."""
    centers = np.asarray(centers, dtype=float)
    labels = _normalize_labels(labels)
    n = centers.shape[0]
    if labels.shape[0] != n:
        raise ValueError("centers and labels length mismatch.")

    comp_a, _, k_resolved = compute_competition_membership(
        centers, labels, k=k, sigma_eps=sigma_eps
    )
    ambiguity = compute_ambiguity(comp_a)
    nearest_same = compute_nearest_same_class_distance(centers, labels)
    isolation, raw_isolation = compute_isolation(
        nearest_same, labels, sigma_eps=sigma_eps
    )
    excess = np.maximum(0.0, raw_isolation - 1.0)
    mu_ipa = ambiguity / (1.0 + lam * excess)

    return IPAMembershipResult(
        comp_a=comp_a,
        ambiguity=ambiguity,
        nearest_same_dist=nearest_same,
        isolation=isolation,
        raw_isolation=raw_isolation,
        mu_ipa=mu_ipa,
        k=k_resolved,
        lam=lam,
    )


@dataclass
class IPAMembershipResult:
    comp_a: np.ndarray
    ambiguity: np.ndarray
    nearest_same_dist: np.ndarray
    isolation: np.ndarray
    raw_isolation: np.ndarray
    mu_ipa: np.ndarray
    k: int
    lam: float


def ball_ipa_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
    lam: float = DEFAULT_LAMBDA,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    IPAMembershipResult,
    np.ndarray,
    np.ndarray,
]:
    """
    CW x IPA membership weights for GBLSTSVM.

    w_i = CW(y_i) * muIPA_i

    Returns weights_pos, weights_neg, membership_result, kappa_pos_balls, kappa_neg_balls.
    """
    if not balls:
        empty = IPAMembershipResult(
            comp_a=np.array([]),
            ambiguity=np.array([]),
            nearest_same_dist=np.array([]),
            isolation=np.array([]),
            raw_isolation=np.array([]),
            mu_ipa=np.array([]),
            k=0,
            lam=lam,
        )
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            empty,
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    centers = np.vstack([b.center for b in balls])
    labels = np.array([b.label for b in balls], dtype=float)
    kappa_pos_balls, kappa_neg_balls = ball_class_weights(
        balls, kappa_pos, kappa_neg
    )

    result = compute_ipa_membership(
        centers, labels, k=k, sigma_eps=sigma_eps, lam=lam
    )
    mu_pos, mu_neg = split_membership_by_class(balls, result.mu_ipa)

    weights_pos = kappa_pos_balls * mu_pos
    weights_neg = kappa_neg_balls * mu_neg
    return weights_pos, weights_neg, result, kappa_pos_balls, kappa_neg_balls
