"""Shared helpers for ladder gate and noise sweep."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from experiments.analyze_topology import MIN_BALL_COUNT
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, DatasetSplit, normalize_labels
from experiments.rebaseline_common import C_GRID, scale_benchmark_split
from experiments.test_split_utility_granulation import _build_indexed_matrix
from experiments.validation_selected_k_common import make_train_val_partition
from src.gblstsvm import predict_twin_hyperplanes
from src.granular_balls import BallInfo, GBList, decompose_granular_balls
from src.ladder_gblstsvm import ARMS, LadderArm, balls_k2_from_train, solve_ladder_systems

RHO = 0.90
REPRESENTATION = "rho0.90"
NOISE_LEVELS = (0, 5, 10, 15, 20)
SEEDS = list(range(10))
GATE_DATASETS = ("diabetes", "ecoli2", "breast_cancer")
GATE_PURITIES = (0.85, 0.90, 0.95)
GATE_SEEDS = (0, 1, 2)
K2_EXACTNESS_SEEDS = (0, 1, 2)
REG_BASE = 1e-4
PREDICTION_RULE: Literal["raw"] = "raw"


def flip_training_labels(y: np.ndarray, noise_pct: float, seed: int) -> np.ndarray:
    """Flip noise_pct% of training labels after split, stratified by class."""
    y_out = normalize_labels(y.copy())
    if noise_pct <= 0:
        return y_out
    rate = noise_pct / 100.0
    rng = np.random.default_rng(seed)
    n = y_out.size
    n_flip = int(round(rate * n))
    if n_flip <= 0:
        return y_out
    classes, counts = np.unique(y_out, return_counts=True)
    flip_counts = np.round(rate * counts).astype(int)
    diff = n_flip - int(flip_counts.sum())
    if diff != 0:
        flip_counts[np.argmax(flip_counts)] += diff
    flip_idx: list[int] = []
    for cls, k in zip(classes, flip_counts):
        idx = np.flatnonzero(y_out == cls)
        if k <= 0 or idx.size == 0:
            continue
        k = min(int(k), idx.size)
        chosen = rng.choice(idx, size=k, replace=False)
        flip_idx.extend(chosen.tolist())
    other = {float(c): float(o) for c, o in zip(classes, classes[::-1])}
    for i in flip_idx:
        y_out[i] = other[float(y_out[i])]
    return y_out


def split_balls(balls: list[BallInfo]) -> tuple[list[BallInfo], list[BallInfo]]:
    pos = [b for b in balls if b.label == 1.0]
    neg = [b for b in balls if b.label != 1.0]
    if not pos or not neg:
        raise ValueError("Need at least one ball per class.")
    return pos, neg


def _split_rho_granulation(
    indexed: np.ndarray,
    *,
    seed: int,
    purity: float,
    min_ball_count: int,
) -> list:
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < purity and ball.num > 1:
            left, right = ball.split_2balls()
            balls[i] = left
            balls.append(right)
            ll += 1
        else:
            i += 1
    if min_ball_count > 0:
        return [b for b in balls if int(b.num) >= min_ball_count]
    return balls


def build_rho90_balls(split: DatasetSplit, seed: int) -> list[BallInfo]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    kept = _split_rho_granulation(
        indexed, seed=seed, purity=RHO, min_ball_count=MIN_BALL_COUNT
    )
    return decompose_granular_balls(kept)


def build_gate_balls(split: DatasetSplit, seed: int, purity: float) -> list[BallInfo]:
    """Granulation for L3 gate: retain all balls so every train sample is represented."""
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    kept = _split_rho_granulation(indexed, seed=seed, purity=purity, min_ball_count=0)
    return decompose_granular_balls(kept)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_n = normalize_labels(y_true)
    y_pred_n = y_pred.copy()
    y_pred_n[y_pred_n == 0.0] = -1.0
    return float(f1_score(y_true_n, y_pred_n, average="macro", zero_division=0))


def build_k2_balls(split: DatasetSplit) -> list[BallInfo]:
    y_train = normalize_labels(split.y_train)
    pos, neg = balls_k2_from_train(split.X_train, y_train)
    return pos + neg


def build_l3_full_balls(split: DatasetSplit, seed: int) -> list[BallInfo]:
    """L3 on full-coverage rho90 granulation (equivalent to raw LSTSVM)."""
    return build_gate_balls(split, seed=seed, purity=RHO)


def param_deviation(
    a: tuple[np.ndarray, float, np.ndarray, float],
    b: tuple[np.ndarray, float, np.ndarray, float],
) -> tuple[float, float]:
    w1, b1, w2, b2 = a
    w1b, b1b, w2b, b2b = b
    dev1 = max(float(np.max(np.abs(w1 - w1b))), abs(b1 - b1b))
    dev2 = max(float(np.max(np.abs(w2 - w2b))), abs(b2 - b2b))
    return dev1, dev2


def storage_table_rows(seed: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for desc in discover_available_datasets():
        split, err = try_load_descriptor(desc, seed=seed, onehot=True)
        if split is None:
            continue
        scaled = scale_benchmark_split(split, scale=True)
        d = int(scaled.X_train.shape[1])
        n_train = int(scaled.X_train.shape[0])
        rho_balls = build_rho90_balls(scaled, seed)
        k_rho90 = len(rho_balls)
        raw_store = n_train * d
        rho90_store = k_rho90 * (d + 1) ** 2
        k2_store = 2 * (d + 1) ** 2
        rows.append(
            {
                "dataset": desc["name"],
                "d": d,
                "n_train": n_train,
                "k_rho90": k_rho90,
                "raw_n_d": raw_store,
                "rho90_k_d1_sq": rho90_store,
                "L3_k2_2_d1_sq": k2_store,
                "ratio_rho90_to_raw": rho90_store / raw_store if raw_store else float("nan"),
                "ratio_k2_to_raw": k2_store / raw_store if raw_store else float("nan"),
            }
        )
    return rows


def fit_eval_ladder(
    split: DatasetSplit,
    balls: list[BallInfo],
    arm: LadderArm,
    c1: float,
    c2: float,
    *,
    prediction_rule: str = PREDICTION_RULE,
) -> dict[str, Any]:
    pos, neg = split_balls(balls)
    try:
        w1, b1, w2, b2 = solve_ladder_systems(
            pos, neg, c1=c1, c2=c2, arm=arm, reg_base=REG_BASE
        )
        preds = predict_twin_hyperplanes(
            split.X_test, w1, b1, w2, b2, rule=prediction_rule
        )
        ok = True
    except (ValueError, np.linalg.LinAlgError):
        w1 = b1 = w2 = b2 = None
        preds = np.zeros_like(split.y_test)
        ok = False
    macro = _macro_f1(split.y_test, preds) if ok else float("nan")
    acc = (
        float(accuracy_score(normalize_labels(split.y_test), preds))
        if ok
        else float("nan")
    )
    return {
        "macro_f1": macro,
        "accuracy": acc,
        "classifier_ok": ok,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }


def tune_c1_c2_ladder(
    split: DatasetSplit,
    balls: list[BallInfo],
    arm: LadderArm,
    seed: int,
    *,
    prediction_rule: str = PREDICTION_RULE,
) -> tuple[float, float]:
    y_train = normalize_labels(split.y_train)
    partition = make_train_val_partition(split.name, seed, split.X_train, y_train)
    val_idx = partition.validation_indices
    X_val = split.X_train[val_idx]
    y_val = y_train[val_idx]
    best_c1, best_c2 = DEFAULT_C1, DEFAULT_C2
    best_f1 = -np.inf
    for c1 in C_GRID:
        for c2 in C_GRID:
            pos, neg = split_balls(balls)
            try:
                w1, b1, w2, b2 = solve_ladder_systems(
                    pos, neg, c1=c1, c2=c2, arm=arm, reg_base=REG_BASE
                )
                preds = predict_twin_hyperplanes(
                    X_val, w1, b1, w2, b2, rule=prediction_rule
                )
                f1 = _macro_f1(y_val, preds)
            except (ValueError, np.linalg.LinAlgError):
                f1 = float("nan")
            if np.isfinite(f1) and f1 > best_f1:
                best_f1 = f1
                best_c1, best_c2 = c1, c2
    return best_c1, best_c2


def ball_geometry_stats(balls: list[BallInfo]) -> dict[str, float]:
    if not balls:
        return {
            "n_balls": 0.0,
            "mean_n_j": float("nan"),
            "mean_trace_S_over_n": float("nan"),
        }
    n_j = np.array([b.num_samples for b in balls], dtype=float)
    traces = []
    for b in balls:
        s = b.scatter
        if s is None:
            traces.append(0.0)
        else:
            traces.append(float(np.trace(s)) / max(b.num_samples, 1))
    return {
        "n_balls": float(len(balls)),
        "mean_n_j": float(np.mean(n_j)),
        "mean_trace_S_over_n": float(np.mean(traces)),
    }


def prepare_noisy_split(
    split: DatasetSplit,
    noise_pct: float,
    seed: int,
) -> DatasetSplit:
    y_noisy = flip_training_labels(split.y_train, noise_pct, seed + 10007 * noise_pct)
    return replace(split, y_train=y_noisy)


def run_l3_k2_cell(
    descriptor: dict,
    seed: int,
    noise_pct: float,
    *,
    tune_c: bool = True,
) -> dict[str, Any] | None:
    split, err = try_load_descriptor(descriptor, seed=seed, onehot=True)
    if split is None:
        return None
    scaled = scale_benchmark_split(split, scale=True)
    noisy = prepare_noisy_split(scaled, noise_pct, seed)
    try:
        balls = build_k2_balls(noisy)
    except ValueError:
        return {
            "dataset": descriptor["name"],
            "seed": seed,
            "noise_pct": noise_pct,
            "macro_f1": float("nan"),
            "accuracy": float("nan"),
            "classifier_ok": False,
        }
    c1, c2 = DEFAULT_C1, DEFAULT_C2
    if tune_c:
        c1, c2 = tune_c1_c2_ladder(noisy, balls, "L3", seed)
    perf = fit_eval_ladder(noisy, balls, "L3", c1, c2)
    return {
        "dataset": descriptor["name"],
        "seed": seed,
        "noise_pct": noise_pct,
        "c1": c1,
        "c2": c2,
        **perf,
    }


def run_ladder_cell(
    descriptor: dict,
    seed: int,
    arm: LadderArm,
    noise_pct: float,
    *,
    tune_c: bool = True,
) -> dict[str, Any] | None:
    split, err = try_load_descriptor(descriptor, seed=seed, onehot=True)
    if split is None:
        return None
    scaled = scale_benchmark_split(split, scale=True)
    noisy = prepare_noisy_split(scaled, noise_pct, seed)
    try:
        balls = build_rho90_balls(noisy, seed)
    except ValueError:
        return {
            "dataset": descriptor["name"],
            "seed": seed,
            "arm": arm,
            "noise_pct": noise_pct,
            "macro_f1": float("nan"),
            "classifier_ok": False,
            **ball_geometry_stats([]),
        }
    c1, c2 = DEFAULT_C1, DEFAULT_C2
    if tune_c:
        c1, c2 = tune_c1_c2_ladder(noisy, balls, arm, seed)
    perf = fit_eval_ladder(noisy, balls, arm, c1, c2)
    geom = ball_geometry_stats(balls)
    return {
        "dataset": descriptor["name"],
        "seed": seed,
        "arm": arm,
        "noise_pct": noise_pct,
        "c1": c1,
        "c2": c2,
        **perf,
        **geom,
    }


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [
        float(np.mean(arr[rng.integers(0, arr.size, arr.size)])) for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate_sweep_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pandas as pd

    df = pd.DataFrame(rows)
    datasets = discover_available_datasets()
    ds_names = [d["name"] for d in datasets if d["name"] in df["dataset"].unique()]
    per_ds = (
        df.groupby(["dataset", "arm", "noise_pct"], as_index=False)["macro_f1"]
        .mean()
        .rename(columns={"macro_f1": "mean_macro_f1"})
    )
    grid = (
        per_ds.groupby(["arm", "noise_pct"])["mean_macro_f1"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grid.columns = ["arm", "noise_pct", "macro_f1_mean", "n_datasets"]
    ci_rows = []
    for (arm, noise), sub in per_ds.groupby(["arm", "noise_pct"]):
        lo, hi = bootstrap_ci(sub["mean_macro_f1"].tolist())
        ci_rows.append(
            {
                "arm": arm,
                "noise_pct": noise,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    grid = grid.merge(ci_df, on=["arm", "noise_pct"], how="left")

    deltas = []
    for noise in NOISE_LEVELS:
        sub = per_ds[per_ds["noise_pct"] == noise]
        p = sub.pivot(index="dataset", columns="arm", values="mean_macro_f1")
        for a, b, name in (("L3", "L0", "L3-L0"), ("L3", "L1", "L3-L1"), ("L1", "L0", "L1-L0")):
            if a in p.columns and b in p.columns:
                d = (p[a] - p[b]).dropna().tolist()
                lo, hi = bootstrap_ci(d)
                deltas.append(
                    {
                        "delta": name,
                        "noise_pct": noise,
                        "mean": float(np.mean(d)) if d else float("nan"),
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
    per_ds_delta = []
    for noise in (0, 20):
        sub = per_ds[per_ds["noise_pct"] == noise]
        p = sub.pivot(index="dataset", columns="arm", values="mean_macro_f1")
        if "L3" in p.columns and "L0" in p.columns:
            for ds in p.index:
                per_ds_delta.append(
                    {
                        "dataset": ds,
                        "noise_pct": noise,
                        "L3_minus_L0": float(p.loc[ds, "L3"] - p.loc[ds, "L0"]),
                    }
                )
    geom = (
        df[df["noise_pct"] == 0]
        .groupby("dataset", as_index=False)[["n_balls", "mean_n_j", "mean_trace_S_over_n"]]
        .mean()
    )
    return {
        "grid": grid,
        "deltas": pd.DataFrame(deltas),
        "per_dataset_delta": pd.DataFrame(per_ds_delta),
        "geometry": geom,
        "datasets": ds_names,
    }
