"""
Competition-Connectivity fuzzy ball membership (CBCM) for GBLSTSVM.

Computes per-ball membership from inter-ball center neighborhood topology:
  Component A — soft k-NN class-competition (muA)
  Component B — within-class structural support (muB)
  Component C — multiplicative combination (muC = muA * muB)

Uses only ball centers and labels; no radius, margin, or volume overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .class_weights import ball_class_weights
from .granular_balls import BallInfo

logger = logging.getLogger(__name__)

DEFAULT_SIGMA_EPS = 1e-8
CBCMVariant = Literal["competition", "connectivity", "combined"]

_PERMUTATION_NULL_REPS = 200


def resolve_k(n_balls: int) -> int:
    """k = max(5, ceil(0.05 * N))."""
    if n_balls <= 0:
        raise ValueError("n_balls must be positive.")
    return max(5, int(np.ceil(0.05 * n_balls)))


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=float).ravel().copy()
    y[y == 0] = -1.0
    return y


def _adaptive_bandwidth(distances: np.ndarray, *, sigma_eps: float) -> float:
    """Median neighbor distance; fall back to sigma_eps when degenerate."""
    d = np.asarray(distances, dtype=float).ravel()
    if d.size == 0:
        return sigma_eps
    sigma = float(np.median(d))
    return sigma_eps if sigma == 0.0 else sigma


def _gaussian_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    d = np.asarray(distances, dtype=float).ravel()
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def _knn_indices_excluding_self(
    centers: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    For each point, return indices of k nearest *other* centers.

    Returns shape (N, k).
    """
    n = centers.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=int)
    k_query = min(k + 1, n)
    nn = NearestNeighbors(
        n_neighbors=k_query,
        algorithm="auto",
        metric="euclidean",
    )
    nn.fit(centers)
    dists, indices = nn.kneighbors(centers, return_distance=True)
    # Drop self (distance ~0 at column 0 when duplicate or exact match).
    neighbor_idx = indices[:, 1:]
    neighbor_dist = dists[:, 1:]
    if neighbor_idx.shape[1] < k:
        return neighbor_idx
    return neighbor_idx[:, :k]


def _median_top_quartile(values: np.ndarray) -> float:
    """Robust normalizer: median of the top 25% of values (descending sort)."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("Cannot compute top-quartile median of empty array.")
    v_sorted = np.sort(v)[::-1]
    n_top = max(1, int(np.ceil(0.25 * v.size)))
    return float(np.median(v_sorted[:n_top]))


def compute_competition_membership(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Component A: soft k-NN class-competition membership.

    Returns compA (raw competition ratio), muA (parabolic remap), and resolved k.
    """
    centers = np.asarray(centers, dtype=float)
    labels = _normalize_labels(labels)
    n = centers.shape[0]
    if labels.shape[0] != n:
        raise ValueError("centers and labels length mismatch.")

    k_resolved = resolve_k(n) if k is None else int(k)
    comp_a = np.zeros(n, dtype=float)

    if n <= 1:
        return comp_a, 4.0 * comp_a * (1.0 - comp_a), k_resolved

    k_eff = min(k_resolved, n - 1)
    neighbor_idx = _knn_indices_excluding_self(centers, k_eff)

    for i in range(n):
        idx = neighbor_idx[i]
        if idx.size == 0:
            comp_a[i] = 0.0
            continue
        dists = np.linalg.norm(centers[i] - centers[idx], axis=1)
        sigma_i = _adaptive_bandwidth(dists, sigma_eps=sigma_eps)
        weights = _gaussian_weights(dists, sigma_i)
        opp = labels[idx] != labels[i]
        w_sum = float(np.sum(weights))
        if w_sum == 0.0:
            comp_a[i] = 0.0
        else:
            comp_a[i] = float(np.sum(weights[opp]) / w_sum)

    mu_a = 4.0 * comp_a * (1.0 - comp_a)
    return comp_a, mu_a, k_resolved


def compute_connectivity_membership(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> Tuple[np.ndarray, int, dict]:
    """
    Component B: within-class structural support membership.

    Returns muB, resolved k, and diagnostic metadata (capping info).
    """
    centers = np.asarray(centers, dtype=float)
    labels = _normalize_labels(labels)
    n = centers.shape[0]
    if labels.shape[0] != n:
        raise ValueError("centers and labels length mismatch.")

    k_resolved = resolve_k(n) if k is None else int(k)
    mu_b = np.ones(n, dtype=float)
    densities = np.zeros(n, dtype=float)
    meta: dict = {"k": k_resolved, "class_k_capped": {}}

    for y in (-1.0, 1.0):
        class_mask = labels == y
        class_idx = np.flatnonzero(class_mask)
        n_y = class_idx.size
        if n_y == 0:
            continue
        k_y = min(k_resolved, n_y - 1)
        if k_y < k_resolved:
            meta["class_k_capped"][str(int(y))] = {
                "n_class": n_y,
                "k_requested": k_resolved,
                "k_used": k_y,
            }
            logger.info(
                "CBCM Component B: class y=%s capped k from %d to %d (n_y=%d)",
                int(y),
                k_resolved,
                k_y,
                n_y,
            )
        if k_y <= 0:
            densities[class_idx] = 0.0
            continue

        class_centers = centers[class_idx]
        neighbor_local = _knn_indices_excluding_self(class_centers, k_y)

        for local_i, global_i in enumerate(class_idx):
            nbr_local = neighbor_local[local_i]
            dists = np.linalg.norm(
                class_centers[local_i] - class_centers[nbr_local], axis=1
            )
            sigma_i = _adaptive_bandwidth(dists, sigma_eps=sigma_eps)
            weights = _gaussian_weights(dists, sigma_i)
            densities[global_i] = float(np.sum(weights))

        class_d = densities[class_idx]
        z_y = _median_top_quartile(class_d)
        if z_y <= 0.0:
            z_y = sigma_eps
        mu_b[class_idx] = np.minimum(1.0, class_d / z_y)

    meta["densities"] = densities
    return mu_b, k_resolved, meta


def compute_cbcm_membership(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> "CBCMMembershipResult":
    """Compute all CBCM components for ball centers and labels."""
    comp_a, mu_a, k_a = compute_competition_membership(
        centers, labels, k=k, sigma_eps=sigma_eps
    )
    mu_b, k_b, meta_b = compute_connectivity_membership(
        centers, labels, k=k, sigma_eps=sigma_eps
    )
    k_resolved = k_a if k is None else int(k)
    mu_c = mu_a * mu_b
    return CBCMMembershipResult(
        comp_a=comp_a,
        mu_a=mu_a,
        mu_b=mu_b,
        mu_c=mu_c,
        k=k_resolved,
        connectivity_meta=meta_b,
    )


def compute_naive_density_weight(
    centers: np.ndarray,
    labels: np.ndarray,
    kappa_per_ball: np.ndarray,
    *,
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> np.ndarray:
    """
    Naive CW x raw same-class density (no per-class normalization, no competition).

    Used only for synthetic-noise diagnostic contrast.
    """
    _, _, meta = compute_connectivity_membership(
        centers, labels, k=k, sigma_eps=sigma_eps
    )
    densities = meta["densities"]
    kappa = np.asarray(kappa_per_ball, dtype=float).ravel()
    d = np.asarray(densities, dtype=float).ravel()
    if d.size != kappa.size:
        raise ValueError("density and kappa length mismatch.")
    # Inverse density: lower density -> higher weight (naive minority-outlier reward).
    inv = 1.0 / np.maximum(d, sigma_eps)
    inv = inv / np.median(inv) if np.median(inv) > 0 else inv
    return kappa * inv


@dataclass
class CBCMMembershipResult:
    comp_a: np.ndarray
    mu_a: np.ndarray
    mu_b: np.ndarray
    mu_c: np.ndarray
    k: int
    connectivity_meta: dict = field(default_factory=dict)


def split_membership_by_class(
    balls: list[BallInfo],
    membership: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split per-ball membership aligned with ``balls_to_class_arrays`` row order."""
    membership = np.asarray(membership, dtype=float).ravel()
    weights_pos = np.array(
        [membership[i] for i, b in enumerate(balls) if b.label == 1], dtype=float
    )
    weights_neg = np.array(
        [membership[i] for i, b in enumerate(balls) if b.label != 1], dtype=float
    )
    return weights_pos, weights_neg


def _membership_for_variant(
    result: CBCMMembershipResult,
    variant: CBCMVariant,
) -> np.ndarray:
    if variant == "competition":
        return result.mu_a
    if variant == "connectivity":
        return result.mu_b
    if variant == "combined":
        return result.mu_c
    raise ValueError(f"Unknown CBCM variant: {variant}")


def ball_cbcm_weights(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
    *,
    variant: CBCMVariant = "combined",
    k: int | None = None,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    CBCMMembershipResult,
    np.ndarray,
    np.ndarray,
]:
    """
    CW x CBCM membership weights for GBLSTSVM.

    w_i = CW(y_i) * mu_i  where mu_i depends on variant:
      competition  -> muA
      connectivity -> muB
      combined     -> muC

    Returns weights_pos, weights_neg, membership_result, kappa_pos_balls, kappa_neg_balls.
    """
    if not balls:
        empty = CBCMMembershipResult(
            comp_a=np.array([]),
            mu_a=np.array([]),
            mu_b=np.array([]),
            mu_c=np.array([]),
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

    result = compute_cbcm_membership(centers, labels, k=k, sigma_eps=sigma_eps)
    mu = _membership_for_variant(result, variant)
    mu_pos, mu_neg = split_membership_by_class(balls, mu)

    weights_pos = kappa_pos_balls * mu_pos
    weights_neg = kappa_neg_balls * mu_neg
    return weights_pos, weights_neg, result, kappa_pos_balls, kappa_neg_balls


def kappa_per_ball_array(
    balls: list[BallInfo],
    kappa_pos: float,
    kappa_neg: float,
) -> np.ndarray:
    """Per-ball CW(y_i) in ball list order."""
    k_pos, k_neg = ball_class_weights(balls, kappa_pos, kappa_neg)
    out = np.zeros(len(balls), dtype=float)
    pos_i = neg_i = 0
    for i, b in enumerate(balls):
        if b.label == 1:
            out[i] = k_pos[pos_i]
            pos_i += 1
        else:
            out[i] = k_neg[neg_i]
            neg_i += 1
    return out


def safe_correlation(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    from scipy.stats import pearsonr, spearmanr

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    if method == "pearson":
        r, _ = pearsonr(a, b)
        return float(r)
    if method == "spearman":
        r, _ = spearmanr(a, b)
        return float(r)
    raise ValueError(method)


def label_permutation_null_ks(
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = _PERMUTATION_NULL_REPS,
    k: int | None = None,
    random_state: int = 42,
) -> dict:
    """
    Null-model check for Component A: permute labels preserving class counts.
    """
    from scipy.stats import ks_2samp

    labels = _normalize_labels(labels)
    n = labels.size
    rng = np.random.default_rng(random_state)

    comp_real, _, k_resolved = compute_competition_membership(
        centers, labels, k=k
    )
    n_pos = int(np.sum(labels == 1.0))
    n_neg = n - n_pos

    null_comps: list[float] = []
    for _ in range(n_permutations):
        perm_labels = np.full(n, -1.0)
        perm_pos = rng.choice(n, size=n_pos, replace=False)
        perm_labels[perm_pos] = 1.0
        comp_null, _, _ = compute_competition_membership(
            centers, perm_labels, k=k_resolved
        )
        null_comps.extend(comp_null.tolist())

    stat, p_value = ks_2samp(comp_real, np.array(null_comps))

    # Per-class KS (minority / majority)
    minority_label = 1.0 if n_pos <= n_neg else -1.0
    minority_mask = labels == minority_label
    majority_mask = ~minority_mask
    per_class = {}
    for name, mask in [("minority", minority_mask), ("majority", majority_mask)]:
        if not np.any(mask):
            per_class[name] = {"ks_statistic": float("nan"), "p_value": float("nan")}
            continue
        null_class: list[float] = []
        for _ in range(n_permutations):
            perm_labels = np.full(n, -1.0)
            perm_pos = rng.choice(n, size=n_pos, replace=False)
            perm_labels[perm_pos] = 1.0
            comp_null, _, _ = compute_competition_membership(
                centers, perm_labels, k=k_resolved
            )
            null_class.extend(comp_null[mask].tolist())
        s, p = ks_2samp(comp_real[mask], np.array(null_class))
        per_class[name] = {"ks_statistic": float(s), "p_value": float(p)}

    return {
        "k": k_resolved,
        "n_permutations": n_permutations,
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "per_class": per_class,
        "comp_real_summary": {
            "mean": float(np.mean(comp_real)),
            "std": float(np.std(comp_real)),
            "median": float(np.median(comp_real)),
        },
    }


def bonferroni_correct(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    return [min(1.0, p * m) for p in p_values]


def percentile_rank(value: float, population: np.ndarray) -> float:
    pop = np.asarray(population, dtype=float).ravel()
    if pop.size == 0:
        return float("nan")
    return float(100.0 * np.mean(pop <= value))
