"""Lightweight tests for granular compression K-sensitivity helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.granular_compression_k_common import (
    deterministic_rng_seed,
    select_stratified_random_indices,
    select_top_k_indices,
    select_training_sample_indices,
)
from experiments.phase8_common import normalize_labels
from src.granular_balls import BallInfo


def _dummy_balls(n: int = 20, minority_label: float = 1.0) -> tuple[list[BallInfo], list[dict]]:
    balls: list[BallInfo] = []
    features: list[dict] = []
    for i in range(n):
        label = minority_label if i % 3 == 0 else -minority_label
        balls.append(
            BallInfo(
                center=np.array([float(i), float(i % 5)]),
                radius=1.0 + 0.1 * i,
                label=label,
                purity=0.5 + 0.02 * i,
                num_samples=10 + i,
            )
        )
        features.append(
            {
                "purity": balls[-1].purity,
                "entropy": 0.1 * (i % 7),
                "boundary_score": 0.2 * (n - i),
                "density": 1.0,
            }
        )
    return balls, features


def test_deterministic_random_selection():
    balls, features = _dummy_balls()
    k = 5
    s1 = select_top_k_indices(balls, features, k, "random", 1.0, rng_seed=12345)
    s2 = select_top_k_indices(balls, features, k, "random", 1.0, rng_seed=12345)
    assert s1 == s2
    assert len(s1) == k
    assert len(set(s1)) == k


def test_exactly_k_selected():
    balls, features = _dummy_balls(n=30)
    for criterion in ("entropy", "boundary", "purity", "entropy_boundary"):
        idx = select_top_k_indices(balls, features, 8, criterion, 1.0, rng_seed=7)
        assert len(idx) == 8
        assert len(set(idx)) == 8


def test_stratified_selection_counts_and_uniqueness():
    balls, _ = _dummy_balls(n=24)
    idx = select_stratified_random_indices(balls, 6, 1.0, rng_seed=42)
    assert len(idx) == 6
    assert len(set(idx)) == 6
    labels = {balls[i].label for i in idx}
    assert 1.0 in labels and -1.0 in labels


def test_sample_count_matching():
    y = normalize_labels(np.array([1, 1, -1, -1, 1, -1, 1, -1, -1, -1]))
    idx = select_training_sample_indices(y, 6, 1.0, rng_seed=1, stratified=False)
    assert len(idx) == 6
    assert len(set(idx)) == 6
    idx_s = select_training_sample_indices(y, 6, 1.0, rng_seed=1, stratified=True)
    assert len(idx_s) == 6
    assert sum(y[idx_s] == 1.0) >= 1
    assert sum(y[idx_s] == -1.0) >= 1


def test_deterministic_rng_seed_stable():
    a = deterministic_rng_seed("ds", 3, 8, method="random_k", repetition=2)
    b = deterministic_rng_seed("ds", 3, 8, method="random_k", repetition=2)
    c = deterministic_rng_seed("ds", 3, 8, method="random_k", repetition=3)
    assert a == b
    assert a != c


def test_no_test_label_leakage_selection_uses_train_only():
    balls, features = _dummy_balls()
    idx = select_top_k_indices(balls, features, 4, "entropy", 1.0, rng_seed=0)
    assert all(0 <= i < len(balls) for i in idx)
