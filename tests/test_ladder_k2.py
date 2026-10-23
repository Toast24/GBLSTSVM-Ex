"""L3_k2 exactness: L3_k2 == L3 == raw LSTSVM to 1e-8."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.ladder_common import (  # noqa: E402
    K2_EXACTNESS_SEEDS,
    REG_BASE,
    build_k2_balls,
    build_l3_full_balls,
    param_deviation,
    split_balls,
)
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels
from experiments.rebaseline_common import scale_benchmark_split
from src.ladder_gblstsvm import balls_from_samples, solve_ladder_systems

TOL = 1e-8


def _solve_raw(X: np.ndarray, y: np.ndarray, c1: float, c2: float):
    pos, neg = balls_from_samples(X, y)
    return solve_ladder_systems(pos, neg, c1=c1, c2=c2, arm="L3", reg_base=REG_BASE)


def _solve_k2_balls(balls, c1: float, c2: float):
    pos, neg = split_balls(balls)
    return solve_ladder_systems(pos, neg, c1=c1, c2=c2, arm="L3", reg_base=REG_BASE)


def run_exactness_gate() -> tuple[bool, float, float]:
    max_dev1 = 0.0
    max_dev2 = 0.0
    for desc in discover_available_datasets():
        for seed in K2_EXACTNESS_SEEDS:
            split, _ = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                continue
            scaled = scale_benchmark_split(split, scale=True)
            y = normalize_labels(scaled.y_train)
            c1, c2 = DEFAULT_C1, DEFAULT_C2
            raw = _solve_raw(scaled.X_train, y, c1, c2)
            k2 = _solve_k2_balls(build_k2_balls(scaled), c1, c2)
            l3 = _solve_k2_balls(build_l3_full_balls(scaled, seed), c1, c2)
            for cand in (k2, l3):
                d1, d2 = param_deviation(cand, raw)
                max_dev1 = max(max_dev1, d1)
                max_dev2 = max(max_dev2, d2)
            d1, d2 = param_deviation(k2, l3)
            max_dev1 = max(max_dev1, d1)
            max_dev2 = max(max_dev2, d2)
    ok = max(max_dev1, max_dev2) <= TOL
    print(
        f"L3_k2 exactness: {'PASS' if ok else 'FAIL'} "
        f"max_dev[w1;b1]={max_dev1:.3e} max_dev[w2;b2]={max_dev2:.3e}"
    )
    return ok, max_dev1, max_dev2


def test_l3_k2_exactness() -> None:
    ok, d1, d2 = run_exactness_gate()
    assert ok, f"L3_k2/L3/raw mismatch: [w1;b1]={d1:.3e}, [w2;b2]={d2:.3e} (> {TOL})"


if __name__ == "__main__":
    ok, _, _ = run_exactness_gate()
    sys.exit(0 if ok else 1)
