"""
A9-1: full information ladder with HELD-OUT c tuning, both coordinate systems.

Replaces the A4/A6 classification tables (which tuned c on rows inside the fitted
balls). Every arm is re-granulated and fit on the selection rows only, tuned on the
disjoint validation rows, then refit on the full train split (A8-E2 protocol), now
with the scatter arms (DIAG, ISO, FULL) on the partition path as well.

Arms:
  RAW                    per-sample LSTSVM
  P_L0 P_L1 P_ISO P_DIAG P_FULL   partition path (no deletion); P_FULL == RAW
  D_L1 D_FULL            deletion path (n < 2 balls dropped); D_FULL == RAW on kept rows
  PUB                    published GBLSTSVM (src.gblstsvm.solve_gblstsvm_systems)

  python experiments/analysis/a9_paper_evidence/run_heldout_ladder.py
  python experiments/analysis/a9_paper_evidence/run_heldout_ladder.py --standardize
"""

from __future__ import annotations

import argparse
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
from experiments.analysis.a8_angle_gate.run_angle_gate import _standardize  # noqa: E402
from experiments.analysis.a8_angle_gate.run_linear_heldout import _pub_fitter  # noqa: E402
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.estimator_information_ladder_common import RHO  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.granular_compression_k_common import deterministic_rng_seed  # noqa: E402
from experiments.validation_selected_k_common import TRAIN_VAL_FRAC, make_train_val_partition  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

OUT_ROOT = ROOT / "experiments" / "output" / "a9_paper_evidence"
SEEDS = list(range(10))
ARMS = ("RAW", "P_L0", "P_L1", "P_ISO", "P_DIAG", "P_FULL", "D_L1", "D_FULL", "PUB")
BALL_ARMS = {"P_L0": ("partition", "L0"), "P_L1": ("partition", "L1"), "P_ISO": ("partition", "ISO"),
             "P_DIAG": ("partition", "DIAG"), "P_FULL": ("partition", "FULL"),
             "D_L1": ("deletion", "L1"), "D_FULL": ("deletion", "FULL")}


def make_fitter(arm: str, X: np.ndarray, y: np.ndarray, seed: int, cache: dict):
    if arm == "RAW":
        return lambda a, b: fit_raw(X, y, a, b)
    if arm == "PUB":
        return _pub_fitter(X, y, seed)
    key = X.shape[0]
    if key not in cache:
        cache[key] = granulate(X, y, seed, RHO)
    path, rung = BALL_ARMS[arm]
    balls = getattr(cache[key], path)
    return lambda a, b: fit_balls(balls, rung, a, b)


def selection_split(name: str, seed: int, X: np.ndarray, y: np.ndarray, draw: int):
    """Draw 0 is the standard partition; draw k > 0 re-draws the 80/20 split."""
    if draw == 0:
        p = make_train_val_partition(name, seed, X, y)
        return p.selection_indices, p.validation_indices
    rs = deterministic_rng_seed(name, seed, 0, method="train_val_split", repetition=draw)
    s, v = train_test_split(np.arange(len(y)), test_size=TRAIN_VAL_FRAC, random_state=rs, stratify=y)
    return np.sort(s), np.sort(v)


def run_cell(split, seed: int, draw: int = 0) -> list[dict]:
    X, y = split.X_train, normalize_labels(split.y_train)
    s, v = selection_split(split.name, seed, X, y, draw)
    cache: dict = {}
    rows = []
    for arm in ARMS:
        rec = {"dataset": split.name, "seed": seed, "arm": arm, "ok": False}
        try:
            c1, c2, val_f1 = tune_c(make_fitter(arm, X[s], y[s], seed, cache), X[v], y[v])
            z1, z2 = make_fitter(arm, X, y, seed, cache)(c1, c2)
        except (ValueError, np.linalg.LinAlgError):
            rows.append(rec)
            continue
        p_repo = predict(split.X_test, z1, z2, "reference")
        p_paper = predict(split.X_test, z1, z2, "paper")
        rec.update({
            "ok": True, "c1": c1, "c2": c2, "val_f1": val_f1,
            "test_f1_repo": macro_f1(split.y_test, p_repo),
            "test_f1_paper": macro_f1(split.y_test, p_paper),
            "collapsed_repo": int(len(np.unique(np.where(p_repo == 0, -1, p_repo))) == 1),
        })
        rows.append(rec)
    gran = cache.get(X.shape[0]) or granulate(X, y, seed, RHO)
    kept_y = y[gran.retained_idx]
    drop = {
        "dropped_frac": 1.0 - len(gran.retained_idx) / len(y),
        "dropped_frac_pos": 1.0 - float(np.sum(kept_y == 1)) / float(np.sum(y == 1)),
        "dropped_frac_neg": 1.0 - float(np.sum(kept_y != 1)) / float(np.sum(y != 1)),
        "n_train": len(y), "d": X.shape[1],
    }
    for r in rows:
        r.update(drop)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--draw", type=int, default=0, help="train/val split draw (0 = standard)")
    args = ap.parse_args()
    out = OUT_ROOT / (("heldout_ladder_std" if args.standardize else "heldout_ladder")
                      + (f"_draw{args.draw}" if args.draw else ""))
    out.mkdir(parents=True, exist_ok=True)
    seeds = [0] if args.smoke else SEEDS
    rows: list[dict] = []
    for desc in discover_available_datasets():
        t0 = time.perf_counter()
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                print(f"skip {desc['name']}: {err}")
                break
            if args.standardize:
                split = _standardize(split)
            rows.extend(run_cell(split, seed, args.draw))
        print(f"{desc['name']}: {time.perf_counter() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
