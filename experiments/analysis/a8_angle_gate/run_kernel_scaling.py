"""
A8-E5: wall-clock scaling of one kernel fit (no tuning) vs n on synthetic data.

Two overlapping Gaussian classes, d=20, standardized. Times K_RAW vs K_GB_EXACT /
K_KM_EXACT / K_GB_L1 (representation + moments + solve) at gamma=1/d, c=1.

  python experiments/analysis/a8_angle_gate/run_kernel_scaling.py
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

from experiments.analysis.a8_angle_gate.run_kernel_pilot import build_rep, fit  # noqa: E402

OUT = ROOT / "experiments" / "output" / "a8_angle_gate" / "kernel_scaling"
SIZES = (500, 1000, 2000, 4000, 8000)
ARMS = ("K_RAW", "K_GB_L1", "K_GB_EXACT", "K_KM_EXACT")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    d = 20
    for n in SIZES:
        rng = np.random.default_rng(n)
        y = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        X = rng.normal(size=(n, d)) + 0.25 * y[:, None] * (np.arange(d) < 5)
        X = (X - X.mean(0)) / X.std(0)
        for arm in ARMS:
            if arm == "K_RAW" and n > 8000:
                continue
            t0 = time.perf_counter()
            rep = build_rep(arm, X, y, seed=0)
            t1 = time.perf_counter()
            fit(rep, X, y, 1.0, 1.0 / d, {})
            t2 = time.perf_counter()
            rows.append({"n": n, "arm": arm, "k": rep["k"], "rep_seconds": t1 - t0,
                         "fit_seconds": t2 - t1, "total_seconds": t2 - t0})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "timing.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
