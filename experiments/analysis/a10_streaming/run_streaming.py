"""
A10: streaming / incremental training with mergeable ball moments.

The training split arrives in B batches. After each batch every arm updates and
is scored on the (clean) test split. Symmetric label noise is applied to the
training stream only.

Arms
  RAW_REFIT    store every sample, refit LSTSVM on all data seen (exact reference)
  CLASS_MOM    keep only per-class running sums H = sum u u^T, g = sum u;
               add the new batch's sums, solve. Exact, constant memory.
  BALL_MOM     granulate each batch (rho partition), add each ball's exact moments.
               Must equal CLASS_MOM (moments add); checked.
  BALL_MOM_W   as BALL_MOM, but each class-part of a granular ball is weighted by
               local label agreement w = (its count) / (parent ball size). A flipped
               label inside a ball of the other class gets a small weight. This is
               the only ball-specific ingredient; still constant memory, because the
               weighted sums fold into the class totals at arrival.
  PUB_STREAM   published GBLSTSVM: granulate each batch (deletion path), append
               centers/radii, solve on all stored balls
  PUB_REGRAN   published GBLSTSVM re-granulated on all data seen at every update

c1, c2 are fixed before streaming: moment arms share RAW's held-out c, the PUB
arms use PUB's held-out c (both tuned on the noisy training split).

  python experiments/analysis/a10_streaming/run_streaming.py [--smoke]
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
    fit_raw,
    macro_f1,
    predict,
    rel_err,
    sample_moments,
    tune_c,
    twin_solve,
)
from experiments.analysis.a8_angle_gate.run_angle_gate import _standardize  # noqa: E402
from experiments.analysis.a8_angle_gate.run_linear_heldout import _pub_fitter  # noqa: E402
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

OUT = ROOT / "experiments" / "output" / "a10_streaming"
SEEDS = list(range(5))
N_BATCHES = 10
NOISE = (0.0, 0.1, 0.2)


class Moments:
    """Per-class running sums of u u^T and u, u = [x; 1]."""

    def __init__(self, d: int):
        self.H = {1.0: np.zeros((d + 1, d + 1)), -1.0: np.zeros((d + 1, d + 1))}
        self.g = {1.0: np.zeros(d + 1), -1.0: np.zeros(d + 1)}

    def add(self, X: np.ndarray, cls: float, w: float = 1.0, sign: float = 1.0) -> None:
        H, g = sample_moments(X)
        self.H[cls] += sign * w * H
        self.g[cls] += sign * w * g

    def solve(self, c1: float, c2: float):
        return twin_solve(self.H[1.0], self.g[1.0], self.H[-1.0], self.g[-1.0], c1, c2)

    def floats(self) -> int:
        return 2 * (self.H[1.0].size + self.g[1.0].size)


def ball_parts(X: np.ndarray, y: np.ndarray, seed: int):
    """(class label, row indices, local label agreement) for every class-part of every ball."""
    balls = _split_rho_granulation(_build_indexed_matrix(X, y), seed=seed, purity=RHO, min_ball_count=0)
    out = []
    for b in balls:
        labels, idx = b.data[:, -2], b.data[:, -1].astype(int)
        for cls in np.unique(labels):
            m = labels == cls
            out.append((1.0 if cls == 1 else -1.0, idx[m], float(m.sum()) / len(idx)))
    return out


def pub_balls(X: np.ndarray, y: np.ndarray, seed: int):
    balls = _split_rho_granulation(_build_indexed_matrix(X, y), seed=seed, purity=RHO,
                                   min_ball_count=MIN_BALL_COUNT_BASELINE)
    return [(b.center, b.r, 1.0 if b.label == 1.0 else -1.0) for b in balls]


def pub_solve(balls, c1: float, c2: float):
    pos = [b for b in balls if b[2] == 1.0]
    neg = [b for b in balls if b[2] != 1.0]
    if not pos or not neg:
        raise ValueError("PUB needs both classes")
    w1, b1, w2, b2 = solve_gblstsvm_systems(np.vstack([b[0] for b in pos]), np.array([b[1] for b in pos]),
                                            np.vstack([b[0] for b in neg]), np.array([b[1] for b in neg]), c1, c2)
    return np.append(w1, b1), np.append(w2, b2)


def flip_labels(y: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    y = y.copy()
    k = int(round(rate * len(y)))
    if k:
        idx = rng.choice(len(y), size=k, replace=False)
        y[idx] = -y[idx]
    return y


def run_stream(split, seed: int, noise: float) -> list[dict]:
    rng = np.random.default_rng(1000 * seed + int(noise * 100))
    X = split.X_train
    y = flip_labels(normalize_labels(split.y_train), noise, rng)
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    d = X.shape[1]

    p = make_train_val_partition(split.name, seed, X, y)
    s, v = p.selection_indices, p.validation_indices
    c1, c2, _ = tune_c(lambda a, b: fit_raw(X[s], y[s], a, b), X[v], y[v])
    pc1, pc2, _ = tune_c(_pub_fitter(X[s], y[s], seed), X[v], y[v])

    batches = np.array_split(np.arange(len(y)), N_BATCHES)
    mom = {"CLASS_MOM": Moments(d), "BALL_MOM": Moments(d), "BALL_MOM_W": Moments(d)}
    batch_class_mom: list[Moments] = []
    pub_store: list = []
    rows = []
    for t, bidx in enumerate(batches, 1):
        Xb, yb = X[bidx], y[bidx]
        seen = np.concatenate(batches[:t])
        res: dict[str, tuple] = {}

        t0 = time.perf_counter()
        z_raw = fit_raw(X[seen], y[seen], c1, c2)
        res["RAW_REFIT"] = (z_raw, time.perf_counter() - t0, len(seen) * (d + 1))

        t0 = time.perf_counter()
        bm = Moments(d)
        for cls in (1.0, -1.0):
            if np.any(yb == cls):
                mom["CLASS_MOM"].add(Xb[yb == cls], cls)
                bm.add(Xb[yb == cls], cls)
        batch_class_mom.append(bm)
        res["CLASS_MOM"] = (mom["CLASS_MOM"].solve(c1, c2), time.perf_counter() - t0, mom["CLASS_MOM"].floats())

        t0 = time.perf_counter()
        parts = ball_parts(Xb, yb, seed)
        t_gran = time.perf_counter() - t0
        for arm, weighted in (("BALL_MOM", False), ("BALL_MOM_W", True)):
            t0 = time.perf_counter()
            for cls, idx, agree in parts:
                mom[arm].add(Xb[idx], cls, agree if weighted else 1.0)
            res[arm] = (mom[arm].solve(c1, c2), t_gran + time.perf_counter() - t0, mom[arm].floats())

        try:
            t0 = time.perf_counter()
            pub_store.extend(pub_balls(Xb, yb, seed))
            res["PUB_STREAM"] = (pub_solve(pub_store, pc1, pc2), time.perf_counter() - t0, len(pub_store) * (d + 1))
        except ValueError:
            pass
        try:
            t0 = time.perf_counter()
            regran = pub_balls(X[seen], y[seen], seed)
            res["PUB_REGRAN"] = (pub_solve(regran, pc1, pc2), time.perf_counter() - t0, len(seen) * (d + 1))
        except ValueError:
            pass

        for arm, (z, secs, floats) in res.items():
            rows.append({
                "dataset": split.name, "seed": seed, "noise": noise, "batch": t, "n_seen": len(seen), "d": d,
                "arm": arm, "update_seconds": secs, "stored_floats": floats,
                "test_f1": macro_f1(split.y_test, predict(split.X_test, z[0], z[1], "reference")),
                "theta_vs_refit": float(np.nanmean([angle_deg(z[0][:-1], z_raw[0][:-1]),
                                                    angle_deg(z[1][:-1], z_raw[1][:-1])])),
                "relerr_vs_refit": max(rel_err(z[0], z_raw[0]), rel_err(z[1], z_raw[1])),
            })

    # Unlearning: remove batch 1 by subtracting its moments; compare to refit without it.
    keep = np.concatenate(batches[1:])
    forget = Moments(d)
    for cls in (1.0, -1.0):
        forget.H[cls] = mom["CLASS_MOM"].H[cls] - batch_class_mom[0].H[cls]
        forget.g[cls] = mom["CLASS_MOM"].g[cls] - batch_class_mom[0].g[cls]
    z_f, z_ref = forget.solve(c1, c2), fit_raw(X[keep], y[keep], c1, c2)
    rows.append({"dataset": split.name, "seed": seed, "noise": noise, "batch": -1, "arm": "UNLEARN_CHECK",
                 "relerr_vs_refit": max(rel_err(z_f[0], z_ref[0]), rel_err(z_f[1], z_ref[1]))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [0] if args.smoke else SEEDS
    rows: list[dict] = []
    for desc in discover_available_datasets():
        t0 = time.perf_counter()
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                print(f"skip {desc['name']}: {err}")
                break
            split = _standardize(split)
            for noise in NOISE:
                rows.extend(run_stream(split, seed, noise))
        print(f"{desc['name']}: {time.perf_counter() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
