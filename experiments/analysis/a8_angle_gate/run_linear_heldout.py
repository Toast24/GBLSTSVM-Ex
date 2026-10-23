"""
A8-E2: linear classification ladder with HELD-OUT c tuning + cost accounting.

A4/A6 tuned c on validation rows that were also inside the fitted balls
(in-sample validation). Here every arm is tuned by fitting on the selection
rows only (re-granulated on those rows) and scoring the disjoint validation
rows, then refit on the full train split.

Arms:
  RAW          per-sample LSTSVM
  P_L0 / P_L1  partition path, center-only (unit / population weights)
  D_L1         deletion path, population-weighted centers
  D_FULL       deletion path, exact moments (== RAW on retained samples)
  PUB          published GBLSTSVM: deletion, majority-label balls, unit weights,
               radius-shifted RHS (src.gblstsvm.solve_gblstsvm_systems)

  python experiments/analysis/a8_angle_gate/run_linear_heldout.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.a8_angle_gate.a8_common import (  # noqa: E402
    fit_balls,
    fit_raw,
    granulate,
    macro_f1,
    predict,
    tune_c,
)
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.estimator_information_ladder_common import (  # noqa: E402
    MIN_BALL_COUNT_BASELINE,
    RHO,
    _split_rho_granulation,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import _build_indexed_matrix  # noqa: E402
from experiments.validation_selected_k_common import make_train_val_partition  # noqa: E402
from src.gblstsvm import solve_gblstsvm_systems  # noqa: E402

OUT = ROOT / "experiments" / "output" / "a8_angle_gate" / "linear_heldout"
SEEDS = list(range(10))
ARMS = ("RAW", "P_L0", "P_L1", "D_L1", "D_FULL", "PUB")


def _pub_fitter(X: np.ndarray, y: np.ndarray, seed: int):
    indexed = _build_indexed_matrix(X, y)
    balls = _split_rho_granulation(indexed, seed=seed, purity=RHO, min_ball_count=MIN_BALL_COUNT_BASELINE)
    pos = [b for b in balls if b.label == 1.0]
    neg = [b for b in balls if b.label != 1.0]
    if not pos or not neg:
        raise ValueError("PUB needs both classes")
    A = np.vstack([b.center for b in pos]); rA = np.array([b.r for b in pos])
    B = np.vstack([b.center for b in neg]); rB = np.array([b.r for b in neg])

    def fit(c1: float, c2: float):
        w1, b1, w2, b2 = solve_gblstsvm_systems(A, rA, B, rB, c1, c2)
        return np.append(w1, b1), np.append(w2, b2)

    return fit


def make_fitter(arm: str, X: np.ndarray, y: np.ndarray, seed: int, cache: dict):
    """Return (fit(c1,c2) -> (z1,z2), representation_seconds, n_units)."""
    t0 = time.perf_counter()
    if arm == "RAW":
        return (lambda a, b: fit_raw(X, y, a, b)), 0.0, len(y)
    if arm == "PUB":
        f = _pub_fitter(X, y, seed)
        return f, time.perf_counter() - t0, -1
    key = (X.shape[0], seed)
    if key not in cache:
        cache[key] = (granulate(X, y, seed, RHO), time.perf_counter() - t0)
    gran, rep_t = cache[key]
    if arm == "P_L0":
        return (lambda a, b: fit_balls(gran.partition, "L0", a, b)), rep_t, len(gran.partition)
    if arm == "P_L1":
        return (lambda a, b: fit_balls(gran.partition, "L1", a, b)), rep_t, len(gran.partition)
    if arm == "D_L1":
        return (lambda a, b: fit_balls(gran.deletion, "L1", a, b)), rep_t, len(gran.deletion)
    if arm == "D_FULL":
        return (lambda a, b: fit_balls(gran.deletion, "FULL", a, b)), rep_t, len(gran.deletion)
    raise ValueError(arm)


def run_cell(split, seed: int) -> list[dict]:
    X, y = split.X_train, normalize_labels(split.y_train)
    part = make_train_val_partition(split.name, seed, X, y)
    s, v = part.selection_indices, part.validation_indices
    rows = []
    cache: dict = {}
    for arm in ARMS:
        try:
            f_sel, _, _ = make_fitter(arm, X[s], y[s], seed, cache)
            c1, c2, val_f1 = tune_c(f_sel, X[v], y[v])
            f_full, rep_t, n_units = make_fitter(arm, X, y, seed, cache)
            t0 = time.perf_counter()
            z1, z2 = f_full(c1, c2)
            solve_t = time.perf_counter() - t0
            ok = True
        except (ValueError, np.linalg.LinAlgError):
            ok = False
        rec = {"dataset": split.name, "seed": seed, "arm": arm, "ok": ok}
        if ok:
            p_repo = predict(split.X_test, z1, z2, "reference")
            p_paper = predict(split.X_test, z1, z2, "paper")
            rec.update({
                "c1": c1, "c2": c2, "val_f1": val_f1, "n_units": n_units,
                "test_f1_repo": macro_f1(split.y_test, p_repo),
                "test_f1_paper": macro_f1(split.y_test, p_paper),
                "collapsed_repo": int(len(np.unique(np.where(p_repo == 0, -1, p_repo))) == 1),
                "representation_seconds": rep_t, "solve_seconds": solve_t,
                "n_train": len(y), "d": X.shape[1],
            })
        rows.append(rec)
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for desc in discover_available_datasets():
        t0 = time.perf_counter()
        for seed in SEEDS:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                print(f"skip {desc['name']}: {err}")
                break
            rows.extend(run_cell(split, seed))
        print(f"{desc['name']}: {time.perf_counter() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
