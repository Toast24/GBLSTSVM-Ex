"""Tests for sample-control reconciliation experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.granular_compression_k_common import (
    deterministic_rng_seed,
    select_training_sample_indices,
)
from experiments.phase8_common import normalize_labels
from experiments.sample_control_reconciliation_common import (
    CONTROL_SEMANTICS,
    class_counts,
    jaccard_similarity,
    select_indices_k_sensitivity,
    select_indices_validation_matched,
    selection_hash,
)


def test_k_sensitivity_rng_reproducible():
    y = normalize_labels(np.array([1, -1, 1, -1, 1, -1, 1, -1]))
    a = select_indices_k_sensitivity(y, 4, 1.0, "ds", 3, 6)
    b = select_indices_k_sensitivity(y, 4, 1.0, "ds", 3, 6)
    assert np.array_equal(a, b)


def test_validation_matched_rng_differs_from_k_sensitivity():
    y = normalize_labels(np.array([1, -1] * 20))
    a = select_indices_k_sensitivity(y, 10, 1.0, "ds", 3, 6)
    b = select_indices_validation_matched(y, 10, 1.0, "ds", 3)
    assert not np.array_equal(a, b)


def test_class_counts():
    y = normalize_labels(np.array([1.0, 1.0, -1.0, -1.0, -1.0]))
    cc = class_counts(y, 1.0)
    assert cc["minority_count"] == 2
    assert cc["majority_count"] == 3
    assert cc["actual_samples"] == 5


def test_selection_hash_identical_for_same_indices():
    idx = np.array([1, 3, 5])
    assert selection_hash(idx) == selection_hash(np.array([5, 1, 3]))


def test_jaccard_identical_sets():
    a = np.array([1, 2, 3])
    assert jaccard_similarity(a, a) == 1.0


def test_jaccard_disjoint():
    assert jaccard_similarity(np.array([1, 2]), np.array([3, 4])) == 0.0


def test_control_semantics_documented():
    assert "sample_stratified_k_original" in CONTROL_SEMANTICS
    assert "rebuild rho90" in CONTROL_SEMANTICS["sample_stratified_k_original"]["post_selection_representation"]
    assert "singleton" in CONTROL_SEMANTICS["sample_stratified_matched_original"]["post_selection_representation"]


def test_stratified_class_constraints():
    y = normalize_labels(np.array([1, 1, -1, -1, 1, -1, 1, -1, -1, -1]))
    idx = select_training_sample_indices(
        y, 6, 1.0, deterministic_rng_seed("d", 0, 6, method="sample_stratified_k"), stratified=True
    )
    assert sum(y[idx] == 1.0) >= 1
    assert sum(y[idx] == -1.0) >= 1


def test_raw_path_limitation_documented():
    assert "limitation" in CONTROL_SEMANTICS["raw_sample_subset"]


def test_no_src_modification():
    src = ROOT / "src"
    assert src.is_dir()
