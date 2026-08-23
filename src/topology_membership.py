"""
Topology-supported competition membership for GBLSTSVM.

  A_i      = 2 * min(comp_i, 1 - comp_i)     (A2 ambiguity; unchanged)
  R_opp(i) = n_reciprocal_opp / (n_opp + eps)
  muTopo   = A_i * R_opp(i)

Uses only ball centers and labels. No radius, margin, density, overlap,
purity, IPA, SP, or CBCM muB terms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .cbcm import (
    DEFAULT_SIGMA_EPS,
    compute_competition_membership,
    resolve_k,
    split_membership_by_class,
)
from .class_weights import ball_class_weights
from .granular_balls import BallInfo
from .ipa import compute_ambiguity

logger = logging.getLogger(__name__)

DEFAULT_RECIP_EPS = 1e-8
# Unit-information Laplace/credibility prior: not dataset-tuned.
# Shrink R_opp toward 1 (no CA penalty) when n_opp is small.
DEFAULT_SUPPORT_LAMBDA = 1.0
DEFAULT_SUPPORT_PRIOR = 1.0


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=float).ravel().copy()
    y[y == 0] = -1.0
    return y


def knn_graph(centers: np.ndarray, k: int) -> np.ndarray:
    """
    Directed kNN indices over ball centers, excluding self.

    Returns array of shape (N, k_eff) with k_eff = min(k, N-1).
    Duplicate centers of *other* balls are valid neighbors.
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    if n <= 1 or k <= 0:
        return np.zeros((n, 0), dtype=int)

    k_eff = min(int(k), n - 1)
    nn = NearestNeighbors(
        n_neighbors=n,
        algorithm="auto",
        metric="euclidean",
    )
    nn.fit(centers)
    _, indices = nn.kneighbors(centers, return_distance=True)

    out = np.full((n, k_eff), -1, dtype=int)
    for i in range(n):
        others = indices[i][indices[i] != i]
        out[i] = others[:k_eff]
    return out


def compute_competition_fraction(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> Tuple[np.ndarray, int]:
    """Distance-weighted opposite-class fraction (same definition as CA/CBCM)."""
    comp, _, k_resolved = compute_competition_membership(
        centers, labels, k=k, sigma_eps=sigma_eps
    )
    return comp, k_resolved


def compute_reciprocal_opposite_support(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int | None = None,
    neighbor_idx: np.ndarray | None = None,
    eps: float = DEFAULT_RECIP_EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    R_opp(i) = (# reciprocal opposite-class neighbors) / (# opposite-class neighbors + eps).

    Returns R_opp, n_opp, n_reciprocal_opp, resolved k.
    """
    centers = np.asarray(centers, dtype=float)
    labels = _normalize_labels(labels)
    n = centers.shape[0]
    if labels.shape[0] != n:
        raise ValueError("centers and labels length mismatch.")

    k_resolved = resolve_k(n) if k is None else int(k)
    r_opp = np.zeros(n, dtype=float)
    n_opp = np.zeros(n, dtype=int)
    n_recip = np.zeros(n, dtype=int)

    if n <= 1:
        return r_opp, n_opp, n_recip, k_resolved

    if neighbor_idx is None:
        neighbor_idx = knn_graph(centers, min(k_resolved, n - 1))

    neighbor_sets = [set(int(j) for j in neighbor_idx[i] if j >= 0) for i in range(n)]

    for i in range(n):
        opp = [j for j in neighbor_sets[i] if labels[j] != labels[i]]
        n_opp[i] = len(opp)
        if n_opp[i] == 0:
            r_opp[i] = 0.0
            continue
        n_recip[i] = sum(1 for j in opp if i in neighbor_sets[j])
        r_opp[i] = float(n_recip[i]) / (float(n_opp[i]) + eps)

    return r_opp, n_opp, n_recip, k_resolved


def compute_topology_membership(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
    recip_eps: float = DEFAULT_RECIP_EPS,
) -> "TopologyMembershipResult":
    """Compute A2, R_opp, and muTopo for ball centers and labels."""
    centers = np.asarray(centers, dtype=float)
    labels = _normalize_labels(labels)
    n = centers.shape[0]
    if labels.shape[0] != n:
        raise ValueError("centers and labels length mismatch.")

    k_resolved = resolve_k(n) if k is None else int(k)
    logger.info("Topo membership: N=%d, k=%d", n, k_resolved)

    comp, k_comp = compute_competition_fraction(
        centers, labels, k=k_resolved, sigma_eps=sigma_eps
    )
    ambiguity = compute_ambiguity(comp)
    neighbor_idx = knn_graph(centers, min(k_resolved, max(0, n - 1)))
    r_opp, n_opp, n_recip, _ = compute_reciprocal_opposite_support(
        centers,
        labels,
        k=k_resolved,
        neighbor_idx=neighbor_idx,
        eps=recip_eps,
    )
    mu_topo = ambiguity * r_opp
    np.clip(mu_topo, 0.0, 1.0, out=mu_topo)

    return TopologyMembershipResult(
        neighbor_idx=neighbor_idx,
        comp=comp,
        ambiguity=ambiguity,
        r_opp=r_opp,
        n_opp=n_opp,
        n_reciprocal_opp=n_recip,
        mu_topo=mu_topo,
        k=k_comp,
    )


@dataclass
class TopologyMembershipResult:
    neighbor_idx: np.ndarray
    comp: np.ndarray
    ambiguity: np.ndarray
    r_opp: np.ndarray
    n_opp: np.ndarray
    n_reciprocal_opp: np.ndarray
    mu_topo: np.ndarray
    k: int


def support_confidence(
    n_opp: np.ndarray,
    *,
    lam: float = DEFAULT_SUPPORT_LAMBDA,
) -> np.ndarray:
    """
    Graph-support confidence for R_opp.

    c(n_opp) = n_opp / (n_opp + λ),  λ = 1 (unit information).

    low n_opp → c → 0; high n_opp → c → 1.
    """
    n = np.asarray(n_opp, dtype=float)
    return n / (n + float(lam))


def adjust_r_opp(
    r_opp: np.ndarray,
    n_opp: np.ndarray,
    *,
    lam: float = DEFAULT_SUPPORT_LAMBDA,
    prior: float = DEFAULT_SUPPORT_PRIOR,
) -> np.ndarray:
    """
    Support-aware reciprocal rate.

    R_opp^S = c(n_opp) * R_opp + (1 - c(n_opp)) * 1

    Equivalent Laplace/Dirichlet smoothing of the reciprocal rate toward 1:

    R_opp^S = (n_reciprocal_opp + λ) / (n_opp + λ)   when prior=1 and
    R_opp = n_reciprocal_opp / n_opp (the stored R_opp uses +eps in the
    denominator; the convex combination uses that stored value).

    n_opp = 0 ⇒ R_opp^S = 1 ⇒ μ = A (CA is not zeroed).
    """
    r = np.asarray(r_opp, dtype=float)
    c = support_confidence(n_opp, lam=lam)
    out = c * r + (1.0 - c) * float(prior)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def compute_support_aware_topology_membership(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
    recip_eps: float = DEFAULT_RECIP_EPS,
    support_lambda: float = DEFAULT_SUPPORT_LAMBDA,
    support_prior: float = DEFAULT_SUPPORT_PRIOR,
) -> "SupportAwareTopologyMembershipResult":
    """Original Topo membership plus support-attenuated R_opp / μ."""
    base = compute_topology_membership(
        centers,
        labels,
        k=k,
        sigma_eps=sigma_eps,
        recip_eps=recip_eps,
    )
    r_opp_s = adjust_r_opp(
        base.r_opp,
        base.n_opp,
        lam=support_lambda,
        prior=support_prior,
    )
    conf = support_confidence(base.n_opp, lam=support_lambda)
    mu_s = base.ambiguity * r_opp_s
    np.clip(mu_s, 0.0, 1.0, out=mu_s)
    return SupportAwareTopologyMembershipResult(
        neighbor_idx=base.neighbor_idx,
        comp=base.comp,
        ambiguity=base.ambiguity,
        r_opp=base.r_opp,
        n_opp=base.n_opp,
        n_reciprocal_opp=base.n_reciprocal_opp,
        mu_topo=base.mu_topo,
        k=base.k,
        support_confidence=conf,
        r_opp_adjusted=r_opp_s,
        mu_topo_s=mu_s,
        support_lambda=float(support_lambda),
        support_prior=float(support_prior),
    )


@dataclass
class SupportAwareTopologyMembershipResult(TopologyMembershipResult):
    support_confidence: np.ndarray = None  # type: ignore[assignment]
    r_opp_adjusted: np.ndarray = None  # type: ignore[assignment]
    mu_topo_s: np.ndarray = None  # type: ignore[assignment]
    support_lambda: float = DEFAULT_SUPPORT_LAMBDA
    support_prior: float = DEFAULT_SUPPORT_PRIOR


def ball_support_aware_topo_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
    recip_eps: float = DEFAULT_RECIP_EPS,
    support_lambda: float = DEFAULT_SUPPORT_LAMBDA,
    support_prior: float = DEFAULT_SUPPORT_PRIOR,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    SupportAwareTopologyMembershipResult,
    np.ndarray,
    np.ndarray,
]:
    """
    CW x support-aware topology membership.

    w_i = CW(y_i) * A_i * R_opp^S(i)
    """
    if not balls:
        empty = SupportAwareTopologyMembershipResult(
            neighbor_idx=np.zeros((0, 0), dtype=int),
            comp=np.array([]),
            ambiguity=np.array([]),
            r_opp=np.array([]),
            n_opp=np.array([], dtype=int),
            n_reciprocal_opp=np.array([], dtype=int),
            mu_topo=np.array([]),
            k=0,
            support_confidence=np.array([]),
            r_opp_adjusted=np.array([]),
            mu_topo_s=np.array([]),
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

    result = compute_support_aware_topology_membership(
        centers,
        labels,
        k=k,
        sigma_eps=sigma_eps,
        recip_eps=recip_eps,
        support_lambda=support_lambda,
        support_prior=support_prior,
    )
    mu_pos, mu_neg = split_membership_by_class(balls, result.mu_topo_s)

    weights_pos = kappa_pos_balls * mu_pos
    weights_neg = kappa_neg_balls * mu_neg
    return weights_pos, weights_neg, result, kappa_pos_balls, kappa_neg_balls


def ball_topo_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
    recip_eps: float = DEFAULT_RECIP_EPS,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    TopologyMembershipResult,
    np.ndarray,
    np.ndarray,
]:
    """
    CW x topology membership weights for GBLSTSVM.

    w_i = CW(y_i) * muTopo(i)
    """
    if not balls:
        empty = TopologyMembershipResult(
            neighbor_idx=np.zeros((0, 0), dtype=int),
            comp=np.array([]),
            ambiguity=np.array([]),
            r_opp=np.array([]),
            n_opp=np.array([], dtype=int),
            n_reciprocal_opp=np.array([], dtype=int),
            mu_topo=np.array([]),
            k=0,
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

    result = compute_topology_membership(
        centers, labels, k=k, sigma_eps=sigma_eps, recip_eps=recip_eps
    )
    mu_pos, mu_neg = split_membership_by_class(balls, result.mu_topo)

    weights_pos = kappa_pos_balls * mu_pos
    weights_neg = kappa_neg_balls * mu_neg
    return weights_pos, weights_neg, result, kappa_pos_balls, kappa_neg_balls
