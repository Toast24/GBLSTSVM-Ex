"""Unit tests for validation-based K-selection helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.granular_compression_k_common import select_training_sample_indices
from experiments.phase8_common import normalize_labels
from experiments.validation_selected_k_common import (
    KValidationSummary,
    compression_seed,
    make_train_val_partition,
    select_k_1se,
    select_k_max,
    summarize_k_validation,
    train_val_split_seed,
)
from src.granular_balls import BallInfo


def _dummy_xy(n: int = 40, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = np.array([1.0 if i % 4 == 0 else -1.0 for i in range(n)])
    return X, y


def test_train_val_split_no_overlap():
    X, y = _dummy_xy()
    part = make_train_val_partition("toy", 3, X, y)
    assert len(set(part.selection_indices) & set(part.validation_indices)) == 0
    assert len(part.selection_indices) + len(part.validation_indices) == len(y)
    assert part.selection_indices.max() < len(y)
    assert part.validation_indices.max() < len(y)


def test_train_val_split_deterministic():
    X, y = _dummy_xy()
    p1 = make_train_val_partition("toy", 5, X, y)
    p2 = make_train_val_partition("toy", 5, X, y)
    assert np.array_equal(p1.selection_indices, p2.selection_indices)
    assert train_val_split_seed("toy", 5) == train_val_split_seed("toy", 5)


def test_k_selection_max_f1():
    summaries = {
        4: KValidationSummary(4, 0.5, 0.01, 0.5, 5, 0.0, 4, 100, False, ""),
        6: KValidationSummary(6, 0.62, 0.02, 0.61, 5, 0.1, 6, 120, False, ""),
        8: KValidationSummary(8, 0.60, 0.01, 0.60, 5, 0.0, 8, 140, False, ""),
    }
    k, _ = select_k_max(summaries)
    assert k == 6


def test_k_selection_max_tiebreak_smaller_k():
    summaries = {
        6: KValidationSummary(6, 0.62, 0.01, 0.62, 5, 0.2, 6, 100, False, ""),
        8: KValidationSummary(8, 0.62, 0.01, 0.62, 5, 0.2, 8, 100, False, ""),
    }
    k, _ = select_k_max(summaries)
    assert k == 6


def test_k_selection_1se_smallest_eligible():
    summaries = {
        4: KValidationSummary(4, 0.58, 0.01, 0.58, 5, 0.0, 4, 80, False, ""),
        6: KValidationSummary(6, 0.62, 0.02, 0.61, 5, 0.0, 6, 100, False, ""),
        8: KValidationSummary(8, 0.60, 0.01, 0.60, 5, 0.0, 8, 120, False, ""),
        10: KValidationSummary(10, 0.55, 0.01, 0.55, 5, 0.0, 10, 140, False, ""),
    }
    k, _, thr = select_k_1se(summaries)
    assert k == 6
    assert thr < summaries[6].mean_macro_f1


def test_matched_sample_stratified_constraints():
    y = normalize_labels(np.array([1, 1, -1, -1, 1, -1, 1, -1, -1, -1]))
    idx = select_training_sample_indices(y, 6, 1.0, rng_seed=99, stratified=True)
    assert len(idx) == 6
    assert sum(y[idx] == 1.0) >= 1
    assert sum(y[idx] == -1.0) >= 1


def test_compression_seed_deterministic():
    a = compression_seed("ds", 1, 8, 2, phase="selection")
    b = compression_seed("ds", 1, 8, 2, phase="selection")
    c = compression_seed("ds", 1, 8, 3, phase="selection")
    assert a == b
    assert a != c


def test_summarize_k_validation_from_rows():
    rows = [
        {"K": 6, "macro_f1": 0.5, "classifier_ok": True, "collapsed": 0, "n_balls": 6, "n_samples_represented": 100},
        {"K": 6, "macro_f1": 0.6, "classifier_ok": True, "collapsed": 0, "n_balls": 6, "n_samples_represented": 100},
    ]
    s = summarize_k_validation(rows)
    assert s.K == 6
    assert abs(s.mean_macro_f1 - 0.55) < 1e-9


def test_candidate_k_skipped_when_pool_small():
    rows = [{"K": 32, "skipped": 1, "skip_reason": "pool<32"}]
    s = summarize_k_validation(rows)
    assert s.skipped
    k, reason = select_k_max({32: s})
    assert k is None
