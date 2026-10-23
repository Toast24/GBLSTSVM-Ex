"""Gate: L3 reproduces raw LSTSVM [w;b] to 1e-8."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ladder_common import (  # noqa: E402
    GATE_DATASETS,
    GATE_PURITIES,
    GATE_SEEDS,
    REG_BASE,
    build_gate_balls,
    split_balls,
)
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, load_dataset_split, normalize_labels
from experiments.rebaseline_common import scale_benchmark_split
from src.ladder_gblstsvm import balls_from_samples, solve_ladder_systems

TOL = 1e-8


def _raw_lstsvm_params(X: np.ndarray, y: np.ndarray, c1: float, c2: float):
    pos, neg = balls_from_samples(X, y)
    return solve_ladder_systems(pos, neg, c1=c1, c2=c2, arm="L3", reg_base=REG_BASE)


def _granulated_l3_params(balls, c1: float, c2: float):
    pos, neg = split_balls(balls)
    return solve_ladder_systems(pos, neg, c1=c1, c2=c2, arm="L3", reg_base=REG_BASE)


def _max_dev(raw, gran) -> float:
    parts = []
    for a, b in zip(raw, gran):
        parts.append(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    return max(parts)


def run_gate() -> tuple[bool, float]:
    max_dev = 0.0
    for name in GATE_DATASETS:
        for seed in GATE_SEEDS:
            split = scale_benchmark_split(load_dataset_split(f"{name}.csv", seed=seed), True)
            y = normalize_labels(split.y_train)
            raw = _raw_lstsvm_params(split.X_train, y, DEFAULT_C1, DEFAULT_C2)
            for rho in GATE_PURITIES:
                balls = build_gate_balls(split, seed=seed, purity=rho)
                gran = _granulated_l3_params(balls, DEFAULT_C1, DEFAULT_C2)
                max_dev = max(max_dev, _max_dev(raw, gran))
    ok = max_dev <= TOL
    print(f"GATE L3 vs raw LSTSVM: {'PASS' if ok else 'FAIL'} max_abs_dev={max_dev:.3e}")
    return ok, max_dev


def test_ladder_l3_matches_raw_lstsvm() -> None:
    ok, max_dev = run_gate()
    assert ok, f"L3 deviates from raw LSTSVM by {max_dev:.3e} (> {TOL})"


if __name__ == "__main__":
    ok, _ = run_gate()
    if not ok:
        print("ESCALATE")
        sys.exit(1)
