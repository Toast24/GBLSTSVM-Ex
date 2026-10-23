"""Unit tests for principled K-selection decision experiment."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.granular_compression_k_common import deterministic_rng_seed
from experiments.principled_complexity_rule_common import (
    K_CANDIDATES,
    METHOD_FIXED_K10,
    METHOD_PREREG_K10,
    RUN_COLUMNS,
    fraction_of_rho90_k,
    pack_run_row,
    size_based_k,
)
from experiments.validation_selected_k_common import (
    KValidationSummary,
    compression_seed,
    make_train_val_partition,
    select_k_1se,
    select_k_max,
)


def test_k_candidates_exact():
    assert K_CANDIDATES == [2, 4, 6, 8, 10, 12]


def test_kmeans_remains_k2():
    import inspect

    import src.granular_balls as gb

    assert "n_clusters=2" in inspect.getsource(gb.GranularBall.split_2balls)


def test_size_rule_exact():
    assert size_based_k(100) == 10
    assert size_based_k(36) == 6
    assert size_based_k(200) == 12
    assert size_based_k(9) == 6


def test_fraction_rule_exact():
    assert fraction_of_rho90_k(40) == 6
    assert fraction_of_rho90_k(80) == 12
    assert fraction_of_rho90_k(100) == 12
    assert fraction_of_rho90_k(50) == 8


def test_max_chooses_highest_validation_score():
    summaries = {
        4: KValidationSummary(4, 0.55, 0.01, 0.55, 5, 0.0, 4, 100, False, ""),
        6: KValidationSummary(6, 0.62, 0.02, 0.61, 5, 0.0, 6, 100, False, ""),
        8: KValidationSummary(8, 0.60, 0.01, 0.60, 5, 0.0, 8, 100, False, ""),
    }
    k, _ = select_k_max(summaries)
    assert k == 6


def test_max_tie_chooses_smaller_k():
    summaries = {
        6: KValidationSummary(6, 0.62, 0.01, 0.62, 5, 0.0, 6, 100, False, ""),
        8: KValidationSummary(8, 0.62, 0.01, 0.62, 5, 0.0, 8, 100, False, ""),
    }
    k, _ = select_k_max(summaries)
    assert k == 6


def test_1se_chooses_smallest_acceptable_k():
    summaries = {
        4: KValidationSummary(4, 0.58, 0.01, 0.58, 5, 0.0, 4, 80, False, ""),
        6: KValidationSummary(6, 0.62, 0.02, 0.61, 5, 0.0, 6, 100, False, ""),
        8: KValidationSummary(8, 0.60, 0.01, 0.60, 5, 0.0, 8, 120, False, ""),
        10: KValidationSummary(10, 0.55, 0.01, 0.55, 5, 0.0, 10, 140, False, ""),
    }
    k, _, thr = select_k_1se(summaries)
    assert k == 6
    assert thr < summaries[6].mean_macro_f1


def test_final_rho90_rebuilt_on_full_train():
    """Full-train pool must be >= train_selection pool."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = np.array([1.0 if i % 5 == 0 else -1.0 for i in range(50)])
    part = make_train_val_partition("toy", 3, X, y)
    sel_idx = part.selection_indices
    all_idx = np.arange(len(y), dtype=int)
    assert len(all_idx) > len(sel_idx)
    assert set(sel_idx).issubset(set(all_idx))


def test_test_data_not_in_k_selection_partition():
    X = np.random.RandomState(1).randn(40, 3)
    y = np.array([1.0 if i % 4 == 0 else -1.0 for i in range(40)])
    part = make_train_val_partition("toy", 7, X, y)
    assert part.selection_indices.max() < len(y)
    assert part.validation_indices.max() < len(y)
    assert len(set(part.selection_indices) & set(part.validation_indices)) == 0


def test_fixed_k10_equals_preregistered_rule():
    ev = {
        "macro_f1": 0.75,
        "accuracy": 0.8,
        "balanced_accuracy": 0.78,
        "collapsed": 0,
        "n_balls": 10,
        "minority_survival": 0.9,
        "majority_survival": 0.95,
    }
    row_fixed = pack_run_row("ds", 0, METHOD_FIXED_K10, ev, n_train=100, n_balls_full_rho90=30, K_selected=10)
    row_pre = pack_run_row("ds", 0, METHOD_PREREG_K10, ev, n_train=100, n_balls_full_rho90=30, K_selected=10)
    assert row_fixed["macro_f1"] == row_pre["macro_f1"]
    assert row_fixed["K_selected"] == row_pre["K_selected"] == 10.0


def test_deterministic_rng():
    a = compression_seed("ds", 2, 8, 1, phase="selection")
    b = compression_seed("ds", 2, 8, 1, phase="selection")
    c = compression_seed("ds", 2, 8, 2, phase="selection")
    assert a == b
    assert a != c
    d = deterministic_rng_seed("ds", 2, 8, method="random_compress_selection", repetition=1)
    assert isinstance(d, int)


def test_required_output_columns_exist():
    ev = {
        "macro_f1": 0.5,
        "accuracy": 0.6,
        "balanced_accuracy": 0.55,
        "collapsed": 0,
        "n_balls": 8,
        "minority_survival": 1.0,
        "majority_survival": 0.9,
    }
    row = pack_run_row(
        "ds",
        0,
        METHOD_FIXED_K10,
        ev,
        n_train=70,
        n_balls_full_rho90=20,
        K_selected=10,
        n_splits=15,
        validation_best_k=8,
        validation_best_f1=0.55,
        runtime_seconds=1.2,
    )
    for col in RUN_COLUMNS:
        assert col in row
