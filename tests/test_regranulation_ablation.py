"""Tests for re-granulation ablation experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.granular_compression_k_common import deterministic_rng_seed
from experiments.phase8_common import DatasetSplit, normalize_labels
from experiments.regranulation_ablation_common import (
    MATCHED_METHODS,
    REGRANULATE_RHOS,
    SECONDARY_BASELINE,
    build_rho_from_train_indices,
    compute_ball_geometry,
    compute_singleton_geometry,
    run_matched_k_block,
    select_shared_subset,
    selection_hash,
    verify_selection_hashes,
)
from experiments.sample_control_reconciliation_common import select_indices_k_sensitivity
from experiments.test_regranulation_ablation import (
    GEOMETRY_COLS,
    _geometry_row,
    _row_from_result,
    analyze_and_write,
    build_paired_comparisons,
    build_rho_k_summary,
)
from experiments.test_split_utility_granulation import _build_indexed_matrix
from experiments.validation_selected_k_common import samples_to_ball_infos


def _synthetic_split(n: int = 80, seed: int = 0) -> DatasetSplit:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = np.array([1.0 if i % 3 != 0 else -1.0 for i in range(n)])
    n_test = 20
    return DatasetSplit(
        name="synthetic_regran",
        info={},
        X_train=X[:-n_test],
        X_val=X[-n_test:][:5],
        X_test=X[-n_test:],
        y_train=y[:-n_test],
        y_val=y[-n_test:][:5],
        y_test=y[-n_test:],
        minority_label=1.0,
        label_mapping="synthetic",
    )


def test_sample_selection_reproducible():
    y = normalize_labels(np.array([1, -1] * 30))
    a = select_indices_k_sensitivity(y, 12, 1.0, "ds", 3, 6)
    b = select_indices_k_sensitivity(y, 12, 1.0, "ds", 3, 6)
    assert np.array_equal(a, b)


def test_identical_selection_hash_across_rho_branches():
    split = _synthetic_split()
    results, skip = run_matched_k_block(split, 0, k=2)
    assert skip is None
    hashes = {r["method"]: r["selection_hash"] for r in results if r["method"] in MATCHED_METHODS}
    verify_selection_hashes(hashes, context="test")


def test_rho_passed_to_granulation():
    split = _synthetic_split()
    y = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y)
    idx = np.arange(min(20, len(y)), dtype=int)
    _, _, trace_lo = build_rho_from_train_indices(indexed, idx, seed=0, purity=0.80)
    _, kept_hi, trace_hi = build_rho_from_train_indices(indexed, idx, seed=0, purity=0.98)
    assert trace_lo["rho"] == 0.80
    assert trace_hi["rho"] == 0.98
    assert len(kept_hi) <= len(idx)


def test_singleton_representation_unchanged():
    y = normalize_labels(np.array([1.0, -1.0, 1.0, -1.0]))
    X = np.arange(8, dtype=float).reshape(4, 2)
    balls = samples_to_ball_infos(X, y)
    assert len(balls) == 4
    assert all(b.radius == 1.0 for b in balls)
    assert all(b.purity == 1.0 for b in balls)
    geom = compute_singleton_geometry(balls, y, 1.0)
    assert geom["n_singletons"] == 4
    assert geom["n_non_singletons"] == 0


def test_class_counts_in_subset():
    split = _synthetic_split()
    sel = select_shared_subset(split, 0, k=2)
    if sel.get("skipped"):
        pytest.skip("pool too small for synthetic data")
    idx = sel["train_idx"]
    y_sub = normalize_labels(split.y_train)[idx]
    assert np.sum(y_sub == 1.0) >= 1
    assert np.sum(y_sub == -1.0) >= 1


def test_no_test_samples_in_granulation_indexing():
    split = _synthetic_split()
    y = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y)
    n_train = len(y)
    idx = np.arange(min(15, n_train), dtype=int)
    assert idx.max() < n_train
    _, kept, _ = build_rho_from_train_indices(indexed, idx, seed=1, purity=0.90)
    for ball in kept:
        for row in ball.data:
            assert int(row[-1]) < n_train


def test_collapse_detection():
    geom = compute_singleton_geometry([], np.array([]), 1.0)
    assert geom["collapse_flag"] == 1
    assert geom["n_balls"] == 0


def test_ball_statistics_internally_consistent():
    split = _synthetic_split()
    y = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y)
    idx = np.arange(min(25, len(y)), dtype=int)
    _, kept, trace = build_rho_from_train_indices(indexed, idx, seed=2, purity=0.85)
    geom = compute_ball_geometry(
        kept, indexed[idx], 1.0, split.minority_label, gran_trace=trace
    )
    assert geom["n_balls"] == geom["n_singletons"] + geom["n_non_singletons"]
    assert geom["minority_balls"] + geom["majority_balls"] == geom["n_balls"]
    assert geom["successful_splits"] == trace["successful_splits"]


def test_full_data_reference_not_mixed():
    split = _synthetic_split()
    results, _ = run_matched_k_block(split, 0, k=2)
    matched = [r for r in results if not r.get("full_data_reference")]
    assert all(r["selection_hash"] for r in matched)


def test_output_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.test_regranulation_ablation.OUT_DIR", tmp_path)
    rows = [
        _row_from_result(
            "ds",
            0,
            {
                "method": "singleton",
                "K": 2,
                "requested_k": 2,
                "actual_retained_samples": 10,
                "selection_hash": "abc",
                "full_data_reference": False,
                "macro_f1": 0.5,
                "collapsed": 0,
                "n_balls": 10,
                "selected_indices": np.arange(10),
            },
        )
    ]
    geom = [_geometry_row("ds", 0, {"method": "singleton", "K": 2, **{c: 1.0 for c in GEOMETRY_COLS}})]
    df = pd.DataFrame(rows)
    geom_df = pd.DataFrame(geom)
    analyze_and_write(df, geom_df, [], hash_ok=True)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "paired_comparisons.csv").exists()
    assert (tmp_path / "rho_k_summary.csv").exists()
    assert (tmp_path / "README.md").exists()


def test_selection_hash_stable():
    idx = np.array([3, 1, 7])
    assert selection_hash(idx) == selection_hash(np.array([7, 3, 1]))


def test_verify_selection_hashes_raises_on_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        verify_selection_hashes({"a": "1111", "b": "2222"}, context="unit")


def test_regranulate_rho90_matches_k_original_f1_on_synthetic():
    split = _synthetic_split(n=120, seed=1)
    results, skip = run_matched_k_block(split, 0, k=2)
    if skip:
        pytest.skip(skip["skip_reason"])
    rho90 = next(r for r in results if r["method"] == "regranulate_rho90")
    korig = next(r for r in results if r["method"] == SECONDARY_BASELINE)
    assert rho90["selection_hash"] == korig["selection_hash"]
    if not np.isfinite(rho90["macro_f1"]) or not np.isfinite(korig["macro_f1"]):
        pytest.skip("classifier did not produce finite F1 on synthetic split")
    assert abs(rho90["macro_f1"] - korig["macro_f1"]) < 1e-9


def test_paired_comparisons_schema():
    df = pd.DataFrame(
        [
            {
                "dataset": "d",
                "seed": 0,
                "K": 2,
                "method": "singleton",
                "macro_f1": 0.4,
                "full_data_reference": False,
                "collapsed": 0,
                "n_balls": 5,
            },
            {
                "dataset": "d",
                "seed": 0,
                "K": 2,
                "method": "regranulate_rho90",
                "macro_f1": 0.6,
                "full_data_reference": False,
                "collapsed": 0,
                "n_balls": 3,
                "rho": 0.9,
            },
        ]
    )
    paired = build_paired_comparisons(df)
    assert not paired.empty
    assert "wilcoxon_p" in paired.columns
