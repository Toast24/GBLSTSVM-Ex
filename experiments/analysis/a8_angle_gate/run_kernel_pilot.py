"""
A8-E4: kernel pilot — where does compression buy compute, and what must be kept?

Twin kernel LSTSVM (Kumar & Gopal) with empirical RBF map phi(x) = K(x, basis):
  per-class H = sum_rows w_r [phi_r,1][phi_r,1]^T, g = sum_rows w_r [phi_r,1]; same twin solve.

Arms (k = number of partition-path class-pure balls):
  K_RAW        basis = all train samples, exact sample moments        O(n^3)
  K_GB_L0      basis = ball centers, rows = phi(center), unit weights   (standard GB kernel, no radius)
  K_GB_L1      basis = ball centers, rows = phi(center), weight n_j
  K_GB_MEAN    basis = ball centers, rows = exact per-ball mean embedding, weight n_j
               (first feature-space moment kept, within-ball feature scatter dropped)
  K_GB_EXACT   basis = ball centers, exact sample moments               O(n k^2 + k^3)
  K_KM_EXACT   basis = per-class k-means centers (same per-class k), exact moments
  K_RND_EXACT  basis = stratified random k train samples, exact moments

Features standardized on train. Hyperparameters (c1=c2=c, gamma) tuned on held-out
validation rows (fit on selection rows only), then refit on full train. Agreement with
K_RAW measured at K_RAW's tuned hyperparameters (function-space analog of the angle).

  python experiments/analysis/a8_angle_gate/run_kernel_pilot.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.a8_angle_gate.a8_common import macro_f1, partition_members, twin_solve  # noqa: E402
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.estimator_information_ladder_common import RHO  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.rebaseline_common import C_GRID  # noqa: E402
from experiments.validation_selected_k_common import make_train_val_partition  # noqa: E402

OUT = ROOT / "experiments" / "output" / "a8_angle_gate" / "kernel_pilot"
GAMMA_MULT = (0.25, 1.0, 4.0)  # gamma = mult / d on standardized features
ARMS = ("K_RAW", "K_GB_L0", "K_GB_L1", "K_GB_MEAN", "K_GB_EXACT", "K_KM_EXACT", "K_RND_EXACT")


def rbf(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    sq = (X * X).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2.0 * X @ Z.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def _aug(P: np.ndarray) -> np.ndarray:
    return np.hstack([P, np.ones((P.shape[0], 1))])


def build_rep(arm: str, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Everything independent of (c, gamma). Returns dict with basis and row groups."""
    t0 = time.perf_counter()
    rep: dict = {"arm": arm}
    if arm == "K_RAW":
        rep["basis"] = X
    else:
        members = partition_members(X, y, seed, RHO)
        labels = np.array([lab for lab, _ in members])
        k_pos, k_neg = int((labels == 1.0).sum()), int((labels != 1.0).sum())
        if arm.startswith("K_GB"):
            rep["basis"] = np.vstack([X[idx].mean(0) for _, idx in members])
            rep["members"] = members
        elif arm == "K_KM_EXACT":
            parts = []
            for mask, k in ((y == 1.0, k_pos), (y != 1.0, k_neg)):
                km = KMeans(n_clusters=min(k, int(mask.sum())), n_init=1, random_state=seed).fit(X[mask])
                parts.append(km.cluster_centers_)
            rep["basis"] = np.vstack(parts)
        elif arm == "K_RND_EXACT":
            rng = np.random.default_rng(seed)
            pos, neg = np.flatnonzero(y == 1.0), np.flatnonzero(y != 1.0)
            idx = np.concatenate([rng.choice(pos, min(k_pos, pos.size), replace=False),
                                  rng.choice(neg, min(k_neg, neg.size), replace=False)])
            rep["basis"] = X[idx]
    rep["rep_seconds"] = time.perf_counter() - t0
    rep["k"] = rep["basis"].shape[0]
    return rep


def class_moments(rep: dict, X: np.ndarray, y: np.ndarray, gamma: float):
    arm, basis = rep["arm"], rep["basis"]
    res = []
    for cls in (1.0, -1.0):
        mask = (y == 1.0) if cls == 1.0 else (y != 1.0)
        if arm in ("K_RAW", "K_GB_EXACT", "K_KM_EXACT", "K_RND_EXACT"):
            U = _aug(rbf(X[mask], basis, gamma))
            res.append((U.T @ U, U.sum(0)))
            continue
        groups = [idx for lab, idx in rep["members"] if lab == cls]
        w = np.array([1.0 if arm == "K_GB_L0" else float(idx.size) for idx in groups])
        if arm in ("K_GB_L0", "K_GB_L1"):
            U = _aug(rbf(np.vstack([X[idx].mean(0) for idx in groups]), basis, gamma))
        else:  # K_GB_MEAN: exact per-ball mean embedding
            U = np.vstack([_aug(rbf(X[idx], basis, gamma)).mean(0) for idx in groups])
        res.append(((U * w[:, None]).T @ U, U.T @ w))
    return res


def fit(rep: dict, X: np.ndarray, y: np.ndarray, c: float, gamma: float, cache: dict):
    if gamma not in cache:
        cache[gamma] = class_moments(rep, X, y, gamma)
    (Ha, ga), (Hb, gb) = cache[gamma]
    return twin_solve(Ha, ga, Hb, gb, c, c)


def scores(rep: dict, Xq: np.ndarray, z1, z2, gamma: float) -> np.ndarray:
    P = _aug(rbf(Xq, rep["basis"], gamma))
    return np.abs(P @ z2) - np.abs(P @ z1)  # RULE_REPO score; sign = prediction


def run_cell(split, seed: int) -> list[dict]:
    X, y = split.X_train, normalize_labels(split.y_train)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    X, Xt = (X - mu) / sd, (split.X_test - mu) / sd
    d = X.shape[1]
    gammas = [m / d for m in GAMMA_MULT]
    part = make_train_val_partition(split.name, seed, X, y)
    s, v = part.selection_indices, part.validation_indices
    fitted: dict[str, dict] = {}
    for arm in ARMS:
        try:
            rep_s = build_rep(arm, X[s], y[s], seed)
            cache: dict = {}
            best = (C_GRID[0], gammas[0], -np.inf)
            for gamma in gammas:
                for c in C_GRID:
                    try:
                        z1, z2 = fit(rep_s, X[s], y[s], c, gamma, cache)
                        f = macro_f1(y[v], np.sign(scores(rep_s, X[v], z1, z2, gamma)))
                    except np.linalg.LinAlgError:
                        continue
                    if f > best[2]:
                        best = (c, gamma, f)
            c, gamma, val_f1 = best
            rep = build_rep(arm, X, y, seed)
            t0 = time.perf_counter()
            z1, z2 = fit(rep, X, y, c, gamma, {})
            fit_t = time.perf_counter() - t0
            fitted[arm] = {"rep": rep, "c": c, "gamma": gamma, "val_f1": val_f1, "z": (z1, z2),
                           "fit_seconds": fit_t}
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"  {split.name} s{seed} {arm} failed: {exc}")
    rows = []
    raw = fitted.get("K_RAW")
    for arm, f in fitted.items():
        sc = scores(f["rep"], Xt, *f["z"], f["gamma"])
        rec = {"dataset": split.name, "seed": seed, "arm": arm, "n_train": len(y), "d": d,
               "k": f["rep"]["k"], "c": f["c"], "gamma_mult": f["gamma"] * d, "val_f1": f["val_f1"],
               "test_f1": macro_f1(split.y_test, np.sign(sc)),
               "rep_seconds": f["rep"]["rep_seconds"], "fit_seconds": f["fit_seconds"]}
        if raw is not None:
            # function-space agreement with K_RAW at K_RAW's hyperparameters
            z1, z2 = fit(f["rep"], X, y, raw["c"], raw["gamma"], {})
            sa = scores(f["rep"], Xt, z1, z2, raw["gamma"])
            sr = scores(raw["rep"], Xt, *raw["z"], raw["gamma"])
            rec["agree_with_raw"] = float(np.mean(np.sign(sa) == np.sign(sr)))
            rec["score_spearman_raw"] = float(stats.spearmanr(sa, sr)[0])
            rec["test_f1_at_raw_hp"] = macro_f1(split.y_test, np.sign(sa))
        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    descs = discover_available_datasets()
    if args.datasets:
        descs = [x for x in descs if x["name"] in args.datasets]
    rows: list[dict] = []
    for desc in descs:
        t0 = time.perf_counter()
        for seed in range(args.seeds):
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                print(f"skip {desc['name']}: {err}")
                break
            rows.extend(run_cell(split, seed))
        print(f"{desc['name']}: {time.perf_counter() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / ("runs.csv" if not args.datasets else "runs_subset.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
