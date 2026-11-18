"""
A9-2: test the closed-form account of the center-only error before fitting.

On the partition path, every rung shares the RHS r of the twin system, so with
M_arm = comb_arm + ridge(comb_arm) * I,

    z_arm = (I + M_arm^{-1} dM) z_FULL,   dM = M_FULL - M_arm   (exact)

and z_FULL == z_RAW. Hence ||z_arm - z_RAW|| / ||z_RAW|| <= q_arm := ||M_arm^{-1} dM||_2.
q_arm needs only ball moments and c: no solve against the RHS. We test
  (a) the identity and the bound numerically,
  (b) whether q predicts the angle theta(arm, RAW) better than eta (A8-E1),
across rho in {0.6..1.0}, unscaled and standardized, c = RAW held-out c.
Arms with the same RHS as FULL: L1, ISO, DIAG (L0 uses unit weights, so it is excluded).

  python experiments/analysis/a9_paper_evidence/run_closed_form.py [--standardize]
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
    angle_deg,
    class_moments,
    fit_raw,
    granulate,
    rel_err,
    tune_c,
    within_fraction,
)
from experiments.analysis.a8_angle_gate.run_angle_gate import SWEEP_RHOS, _standardize  # noqa: E402
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.estimator_information_ladder_common import REG_BASE, split_balls  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.validation_selected_k_common import make_train_val_partition  # noqa: E402
from src.ladder_gblstsvm import ridge_scale  # noqa: E402

OUT_ROOT = ROOT / "experiments" / "output" / "a9_paper_evidence"
SEEDS = list(range(10))
ARMS = ("L1", "ISO", "DIAG", "FULL")


def systems(pos, neg, arm: str, c1: float, c2: float):
    """(M1, r1, M2, r2) exactly as a8_common.twin_solve assembles them."""
    Ha, ga = class_moments(pos, arm)
    Hb, gb = class_moments(neg, arm)
    eye = np.identity(Ha.shape[0])
    comb1 = Hb + (1.0 / c1) * Ha
    comb2 = Ha + (1.0 / c2) * Hb
    return (comb1 + ridge_scale(comb1, REG_BASE) * eye, -gb,
            comb2 + ridge_scale(comb2, REG_BASE) * eye, ga)


def run_cell(split, seed: int, rho: float, c1: float, c2: float) -> list[dict]:
    X, y = split.X_train, normalize_labels(split.y_train)
    pos, neg = split_balls(granulate(X, y, seed, rho).partition)
    eta = 0.5 * (within_fraction(pos, X[y == 1]) + within_fraction(neg, X[y != 1]))
    z_raw = fit_raw(X, y, c1, c2)
    sysd = {arm: systems(pos, neg, arm, c1, c2) for arm in ARMS}
    MF1, _, MF2, _ = sysd["FULL"]
    rows = []
    for arm in ("L1", "ISO", "DIAG"):
        M1, r1, M2, r2 = sysd[arm]
        rec = {"dataset": split.name, "seed": seed, "rho": rho, "arm": arm, "c1": c1, "c2": c2,
               "eta": eta, "n_balls": len(pos) + len(neg)}
        th, q, re, ident = [], [], [], []
        for M, r, MF, zr in ((M1, r1, MF1, z_raw[0]), (M2, r2, MF2, z_raw[1])):
            z = np.linalg.solve(M, r)
            T = np.linalg.solve(M, MF - M)          # M_arm^{-1} dM
            q.append(float(np.linalg.norm(T, 2)))
            th.append(angle_deg(z[:-1], zr[:-1]))
            re.append(rel_err(z, zr))
            ident.append(rel_err(z, zr + T @ zr))   # identity residual
        rec.update({
            "theta": float(np.nanmean(th)), "theta1": th[0], "theta2": th[1],
            "q": float(np.mean(q)), "q1": q[0], "q2": q[1],
            "relerr": float(np.mean(re)), "relerr1": re[0], "relerr2": re[1],
            "bound_ok": int(re[0] <= q[0] * (1 + 1e-9) and re[1] <= q[1] * (1 + 1e-9)),
            "identity_resid": float(max(ident)),
        })
        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = OUT_ROOT / ("closed_form_std" if args.standardize else "closed_form")
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
            X, y = split.X_train, normalize_labels(split.y_train)
            p = make_train_val_partition(split.name, seed, X, y)
            s, v = p.selection_indices, p.validation_indices
            c1, c2, _ = tune_c(lambda a, b: fit_raw(X[s], y[s], a, b), X[v], y[v])
            for rho in SWEEP_RHOS:
                try:
                    rows.extend(run_cell(split, seed, rho, c1, c2))
                except (ValueError, np.linalg.LinAlgError) as e:
                    print(f"  {desc['name']} s{seed} rho{rho}: {e}")
        print(f"{desc['name']}: {time.perf_counter() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
