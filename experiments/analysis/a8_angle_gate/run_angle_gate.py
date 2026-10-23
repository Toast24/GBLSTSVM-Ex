"""
A8 gate: does the angle pipeline return ~0 deg for a partition-preserving FULL arm
at c matched to RAW?  Decomposes A7's theta_FULL into deletion vs c-mismatch.

  python experiments/analysis/a8_angle_gate/run_angle_gate.py            # gate, rho=0.90
  python experiments/analysis/a8_angle_gate/run_angle_gate.py --sweep    # rho sweep
  python experiments/analysis/a8_angle_gate/run_angle_gate.py --standardize

c sources (all arms in a cell share the same c1, c2):
  C_A6_SHARED : A6 MODE_A c (L1 tuned on deletion balls; in-sample validation, as A6)
  C_RAW_HO    : RAW tuned on held-out validation (fit on selection rows only)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.a8_angle_gate.a8_common import (  # noqa: E402
    ARMS,
    angle_deg,
    fit_balls,
    fit_raw,
    granulate,
    iso_mismatch,
    macro_f1,
    predict,
    rel_err,
    tune_c,
    within_fraction,
)
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.estimator_information_ladder_common import split_balls  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.validation_selected_k_common import make_train_val_partition  # noqa: E402

A6_PARAMS = ROOT / "experiments" / "output" / "estimator_a6_scatter_mechanism" / "params"
OUT_ROOT = ROOT / "experiments" / "output" / "a8_angle_gate"
SEEDS = list(range(10))
SWEEP_RHOS = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--datasets", nargs="*")
    return p.parse_args()


def _standardize(split):
    mu = split.X_train.mean(0)
    sd = split.X_train.std(0)
    sd[sd == 0] = 1.0
    split.X_train = (split.X_train - mu) / sd
    split.X_test = (split.X_test - mu) / sd
    return split


def _a6_record(man: pd.DataFrame | None, name: str, seed: int, arm: str):
    if man is None:
        return None
    r = man[(man.dataset == name) & (man.seed == seed) & (man.arm == arm)
            & (man.tuning_mode == "MODE_A_SHARED_C")]
    return None if r.empty else r.iloc[0]


def run_cell(split, seed: int, rho: float, man: pd.DataFrame | None, c_sources: list[str]) -> list[dict]:
    X, Xt, yt_test = split.X_train, split.X_test, split.y_test
    y = normalize_labels(split.y_train)
    n = len(y)
    gran = granulate(X, y, seed, rho)
    part = gran.partition
    dele = gran.deletion
    X_ret, y_ret = X[gran.retained_idx], y[gran.retained_idx]

    # Geometry descriptors
    geo: dict[str, Any] = {
        "n_train": n,
        "d": X.shape[1],
        "n_balls_partition": len(part),
        "n_balls_deletion": len(dele),
        "dropped_frac": 1.0 - len(gran.retained_idx) / n,
        "dropped_frac_pos": 1.0 - float(np.sum(y_ret == 1)) / float(np.sum(y == 1)),
        "dropped_frac_neg": 1.0 - float(np.sum(y_ret != 1)) / float(np.sum(y != 1)),
    }
    for tag, balls in (("partition", part), ("deletion", dele)):
        try:
            pos, neg = split_balls(balls)
            geo[f"within_frac_pos_{tag}"] = within_fraction(pos, X[y == 1])
            geo[f"within_frac_neg_{tag}"] = within_fraction(neg, X[y != 1])
            geo[f"iso_mismatch_pos_{tag}"] = iso_mismatch(pos)
            geo[f"iso_mismatch_neg_{tag}"] = iso_mismatch(neg)
        except ValueError:
            pass

    cs: dict[str, tuple[float, float]] = {}
    a6_full = _a6_record(man, split.name, seed, "FULL_SCATTER")
    if "C_A6_SHARED" in c_sources and a6_full is not None:
        cs["C_A6_SHARED"] = (float(a6_full.c1), float(a6_full.c2))
    if "C_RAW_HO" in c_sources:
        part_idx = make_train_val_partition(split.name, seed, X, y)
        s, v = part_idx.selection_indices, part_idx.validation_indices
        c1, c2, _ = tune_c(lambda a, b: fit_raw(X[s], y[s], a, b), X[v], y[v])
        cs["C_RAW_HO"] = (c1, c2)

    rows: list[dict] = []
    for c_src, (c1, c2) in cs.items():
        z_raw = fit_raw(X, y, c1, c2)
        z_ret = fit_raw(X_ret, y_ret, c1, c2)
        fits: dict[tuple[str, str], tuple] = {("RAW", "RAW"): z_raw, ("DELETION", "RAW_RETAINED"): z_ret}
        for path, balls in (("PARTITION", part), ("DELETION", dele)):
            for arm in ARMS:
                try:
                    fits[(path, arm)] = fit_balls(balls, arm, c1, c2)
                except (ValueError, np.linalg.LinAlgError):
                    fits[(path, arm)] = None
        for (path, arm), z in fits.items():
            if z is None:
                continue
            z1, z2 = z
            rec = {
                "dataset": split.name, "seed": seed, "rho": rho, "c_source": c_src,
                "c1": c1, "c2": c2, "path": path, "arm": arm,
                "theta1_vs_raw": angle_deg(z1[:-1], z_raw[0][:-1]),
                "theta2_vs_raw": angle_deg(z2[:-1], z_raw[1][:-1]),
                "relerr1_vs_raw": rel_err(z1, z_raw[0]),
                "relerr2_vs_raw": rel_err(z2, z_raw[1]),
                "theta1_vs_retained": angle_deg(z1[:-1], z_ret[0][:-1]),
                "theta2_vs_retained": angle_deg(z2[:-1], z_ret[1][:-1]),
                "relerr1_vs_retained": rel_err(z1, z_ret[0]),
                "norm_w1_ratio_vs_raw": float(np.linalg.norm(z1[:-1]) / np.linalg.norm(z_raw[0][:-1])),
                "norm_w2_ratio_vs_raw": float(np.linalg.norm(z2[:-1]) / np.linalg.norm(z_raw[1][:-1])),
                "log_w1_over_w2": float(np.log(np.linalg.norm(z1[:-1]) / np.linalg.norm(z2[:-1]))),
                "test_f1_repo": macro_f1(yt_test, predict(Xt, z1, z2, "reference")),
                "test_f1_paper": macro_f1(yt_test, predict(Xt, z1, z2, "paper")),
                "rule_disagree": float(np.mean(predict(Xt, z1, z2, "reference") != predict(Xt, z1, z2, "paper"))),
                **geo,
            }
            if c_src == "C_A6_SHARED" and path == "DELETION" and arm == "FULL" and a6_full is not None:
                d = np.load(A6_PARAMS / a6_full.path)
                rec["repro_relerr_w1_vs_a6"] = rel_err(z1[:-1], d["w1"])
                rec["repro_relerr_w2_vs_a6"] = rel_err(z2[:-1], d["w2"])
                a6_raw = _a6_record(man, split.name, seed, "RAW_LSTSVM")
                if a6_raw is not None:
                    dr = np.load(A6_PARAMS / a6_raw.path)
                    rec["repro_relerr_raw_w1_vs_a6"] = rel_err(z_raw[0][:-1], dr["w1"])
            rows.append(rec)
    return rows


def main() -> int:
    args = parse_args()
    rhos = SWEEP_RHOS if args.sweep else (0.90,)
    tag = ("sweep" if args.sweep else "gate") + ("_std" if args.standardize else "")
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(A6_PARAMS / "manifest.csv") if (A6_PARAMS / "manifest.csv").exists() else None
    # A6 params live in unscaled coordinates; only compare in the unscaled gate.
    c_sources = ["C_RAW_HO"] if (args.sweep or args.standardize) else ["C_A6_SHARED", "C_RAW_HO"]
    if args.standardize:
        man = None

    descs = discover_available_datasets()
    if args.datasets:
        descs = [d for d in descs if d["name"] in args.datasets]
    seeds = [0] if args.smoke else SEEDS
    rows: list[dict] = []
    for desc in descs:
        t0 = time.perf_counter()
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                print(f"skip {desc['name']}: {err}")
                break
            if args.standardize:
                split = _standardize(split)
            for rho in rhos:
                rows.extend(run_cell(split, seed, rho, man, c_sources))
        print(f"{desc['name']}: {time.perf_counter() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
