"""Unit checks for margin-aware multi-granularity (Proposal 2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gblstsvm import solve_gblstsvm_systems  # noqa: E402
from src.margin_multigranularity import (  # noqa: E402
    WeightedBall,
    WeightedBallSVMLearner,
    WeightedGBLSTSVMLearner,
    build_margin_hierarchy,
)


def _data(seed=0, n=400, weights=(0.5, 0.5)):
    rng = np.random.default_rng(seed)
    n1 = int(n * weights[1])
    X = np.vstack([rng.normal([-1.2, 0], 0.9, (n - n1, 2)), rng.normal([1.2, 0], 0.9, (n1, 2))])
    y = np.r_[-np.ones(n - n1), np.ones(n1)]
    return X, y


def test_unit_weights_match_reference_solver():
    rng = np.random.default_rng(0)
    balls = [WeightedBall(rng.normal(size=3), rng.uniform(0.1, 1), lbl, 1.0, 5, 0)
             for lbl in [1.0] * 6 + [-1.0] * 7]
    dec = WeightedGBLSTSVMLearner().fit_balls(balls)
    A = np.vstack([b.center for b in balls[:6]]); Rp = np.array([b.radius for b in balls[:6]])
    B = np.vstack([b.center for b in balls[6:]]); Rn = np.array([b.radius for b in balls[6:]])
    w1, b1, w2, b2 = solve_gblstsvm_systems(A, Rp, B, Rn, 1.0, 0.1)
    Z = rng.normal(size=(20, 3))
    ref = np.abs(Z @ w2 + b2) / np.linalg.norm(w2) - np.abs(Z @ w1 + b1) / np.linalg.norm(w1)
    assert np.allclose(dec(Z), ref, atol=1e-8)


def test_levels_are_nested_partitions():
    np.random.seed(0)
    X, y = _data()
    h = build_margin_hierarchy(X, y, WeightedGBLSTSVMLearner())
    counts = [len(L) for L in h.levels]
    assert counts == sorted(counts) and len(h.levels) >= 2
    for L in h.levels:
        assert sum(b.num_samples for b in L) == len(y)
        assert max(b.level for b in L) <= len(h.levels) - 1


def test_boundary_finer_than_interior():
    np.random.seed(1)
    X, y = _data(1)
    final = build_margin_hierarchy(X, y, WeightedBallSVMLearner()).final
    near = np.mean([b.radius for b in final if abs(b.center[0]) < 0.7])
    far = np.mean([b.radius for b in final if abs(b.center[0]) > 1.5])
    assert near < far


def test_flipped_labels_become_noise_balls():
    np.random.seed(2)
    X, y = _data(2)
    far = np.where(np.abs(X[:, 0]) > 2.0)[0][:8]
    y = y.copy(); y[far] *= -1
    final = build_margin_hierarchy(X, y, WeightedBallSVMLearner(), max_levels=8).final
    # Flipped interior points are either absorbed by a majority ball or isolated
    # as down-weighted noise; no full-weight wrong-label ball survives there.
    wrong_side = [b for b in final if abs(b.center[0]) > 2.0 and np.sign(b.center[0]) != b.label]
    assert all(b.region == "noise" and b.weight < 1 for b in wrong_side)
    assert all(b.weight < 1 for b in final if b.region == "noise")


def test_minority_is_preserved():
    np.random.seed(3)
    X, y = _data(3, weights=(0.9, 0.1))
    final = build_margin_hierarchy(X, y, WeightedGBLSTSVMLearner()).final
    assert any(b.label == 1 for b in final)
