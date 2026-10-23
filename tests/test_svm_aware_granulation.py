"""Unit checks for SVM-aware granular-ball generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.granular_balls import GBList  # noqa: E402
from src.svm_aware_granulation import (  # noqa: E402
    BallSVMLearner,
    GBLSTSVMLearner,
    SplitScoreWeights,
    SVMAwareGBLSTSVM,
    generate_svm_aware_balls,
    score_balls,
)


def _data(seed=0):
    X, y = make_blobs(400, centers=[[-1.2, 0], [1.2, 0]], cluster_std=0.9, random_state=seed)
    return X, np.where(y == 1, 1.0, -1.0)


def test_partition_preserved():
    np.random.seed(0)
    X, y = _data()
    balls, trace = generate_svm_aware_balls(X, y, GBLSTSVMLearner())
    assert sum(b.num_samples for b in balls) == len(y)
    assert trace.rounds and trace.gen_time > 0


def test_boundary_balls_score_higher_than_interior():
    np.random.seed(0)
    X, y = _data()
    gb = GBList(np.column_stack([X, y, np.arange(len(y))]))
    gb.init_granular_balls(purity=0.8, min_sample=4)
    dec = lambda Z: Z[:, 0]  # true boundary x0 = 0
    S, comps = score_balls(gb.granular_balls, dec, SplitScoreWeights(), tau=0.5)
    dist = np.array([abs(b.center[0]) for b in gb.granular_balls])
    near, far = S[dist < np.median(dist)], S[dist >= np.median(dist)]
    assert near.mean() > far.mean()
    assert np.all(comps["M"][dist > 2.5] == 0)


def test_refines_near_boundary_only():
    np.random.seed(1)
    X, y = _data(1)
    balls, _ = generate_svm_aware_balls(X, y, BallSVMLearner(), rounds=6, tau=0.5, theta=0.5)
    r_near = np.mean([b.radius for b in balls if abs(b.center[0]) < 0.7])
    r_far = np.mean([b.radius for b in balls if abs(b.center[0]) > 1.5])
    assert r_near < r_far


def test_estimator_accuracy():
    X, y = _data(2)
    m = SVMAwareGBLSTSVM(random_state=0).fit(X[:300], y[:300])
    assert np.mean(m.predict(X[300:]) == y[300:]) > 0.8
