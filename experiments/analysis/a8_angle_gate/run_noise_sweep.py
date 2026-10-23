"""
A8-E6: label-noise sweep — is granular-ball *selection* (n<2 deletion) worth anything?

Train labels flipped at rate p per class (stratified, symmetric); test labels clean.
Held-out c tuning as in run_linear_heldout (validation rows carry the same noise).

Arms:
  RAW, P_L1, D_L1, D_FULL (= RAW on GB-retained rows), PUB     (from run_linear_heldout)
  ENN_RAW    RAW on rows kept by Edited Nearest Neighbours (k=3, drop if label != 3-NN majority)
  RDROP_RAW  RAW after deleting a random subset of the same size GB deletion removes (per class)

Also logs filter quality on the full train split: fraction removed, recall of flipped rows,
precision (flipped among removed).

  python experiments/analysis/a8_angle_gate/run_noise_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.a8_angle_gate.a8_common import fit_raw, granulate, macro_f1, predict, tune_c  # noqa: E402
from experiments.analysis.a8_angle_gate.run_linear_heldout import make_fitter  # noqa: E402
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.estimator_information_ladder_common import RHO  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.validation_selected_k_common import make_train_val_partition  # noqa: E402

OUT = ROOT / "experiments" / "output" / "a8_angle_gate" / "noise_sweep"
SEEDS = list(range(10))
NOISE = (0.0, 0.05, 0.10, 0.20, 0.30)
BASE_ARMS = ("RAW", "P_L1", "D_L1", "D_FULL", "PUB")
FILTER_ARMS = ("ENN_RAW", "RDROP_RAW")


def flip_labels(y: np.ndarray, p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    y_noisy = y.copy()
    flipped = np.zeros(y.size, dtype=bool)
    for cls in (1.0, -1.0):
        idx = np.flatnonzero(y == cls)
        pick = rng.choice(idx, int(round(p * idx.size)), replace=False)
        flipped[pick] = True
    y_noisy[flipped] = -y_noisy[flipped]
    return y_noisy, flipped


def enn_keep(X: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    votes = y[ind[:, 1:]].sum(1)
    keep = np.sign(votes) == y  # k odd -> no ties
    if len(np.unique(y[keep])) < 2:
        keep[:] = True
    return keep


def gb_keep(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    keep = np.zeros(y.size, dtype=bool)
    keep[granulate(X, y, seed, RHO).retained_idx] = True
    return keep


def rdrop_keep(y: np.ndarray, gb: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(20_000 + seed)
    keep = np.ones(y.size, dtype=bool)
    for cls in (1.0, -1.0):
        idx = np.flatnonzero(y == cls)
        n_drop = int(np.sum(~gb[idx]))
        keep[rng.choice(idx, n_drop, replace=False)] = False
    return keep


def filter_fitter(arm: str, X: np.ndarray, y: np.ndarray, seed: int):
    keep = enn_keep(X, y) if arm == "ENN_RAW" else rdrop_keep(y, gb_keep(X, y, seed), seed)
    Xk, yk = X[keep], y[keep]
    return lambda a, b: fit_raw(Xk, yk, a, b)


def filter_quality(name: str, keep: np.ndarray, flipped: np.ndarray) -> dict:
    removed = ~keep
    return {
        f"{name}_removed_frac": float(removed.mean()),
        f"{name}_recall": float(removed[flipped].mean()) if flipped.any() else float("nan"),
        f"{name}_precision": float(flipped[removed].mean()) if removed.any() else float("nan"),
    }


def run_cell(split, seed: int, p: float) -> list[dict]:
    X = split.X_train
    y_clean = normalize_labels(split.y_train)
    y, flipped = flip_labels(y_clean, p, seed)
    part = make_train_val_partition(split.name, seed, X, y)
    s, v = part.selection_indices, part.validation_indices
    gb = gb_keep(X, y, seed)
    quality = {**filter_quality("gb", gb, flipped), **filter_quality("enn", enn_keep(X, y), flipped)}
    rows = []
    cache: dict = {}
    for arm in BASE_ARMS + FILTER_ARMS:
        try:
            if arm in FILTER_ARMS:
                f_sel = filter_fitter(arm, X[s], y[s], seed)
                c1, c2, _ = tune_c(f_sel, X[v], y[v])
                z1, z2 = filter_fitter(arm, X, y, seed)(c1, c2)
            else:
                f_sel, _, _ = make_fitter(arm, X[s], y[s], seed, cache)
                c1, c2, _ = tune_c(f_sel, X[v], y[v])
                f_full, _, _ = make_fitter(arm, X, y, seed, cache)
                z1, z2 = f_full(c1, c2)
        except (ValueError, np.linalg.LinAlgError):
            rows.append({"dataset": split.name, "seed": seed, "noise": p, "arm": arm, "ok": False})
            continue
        rows.append({
            "dataset": split.name, "seed": seed, "noise": p, "arm": arm, "ok": True,
            "c1": c1, "c2": c2,
            "test_f1_repo": macro_f1(split.y_test, predict(split.X_test, z1, z2, "reference")),
            "test_f1_paper": macro_f1(split.y_test, predict(split.X_test, z1, z2, "paper")),
            **quality,
        })
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
            for p in NOISE:
                rows.extend(run_cell(split, seed, p))
        print(f"{desc['name']}: {time.perf_counter() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
