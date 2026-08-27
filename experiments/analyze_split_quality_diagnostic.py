"""
Split-quality diagnostic (read-only).

Replays frozen recursive KMeans(k=2) granulation (rho=0.95, min_samples=1,
del_ball num_data=2) and records every proposed split B -> B1, B2.

Labels A/B/C/D from split geometry and del_ball fate only. Does NOT use
Topo, CW, or GBLSTSVM performance.

Does NOT modify granulation, del_ball, Topo, CW, or GBLSTSVM.

Run from GBLSTSVM root:
  python experiments/analyze_split_quality_diagnostic.py
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granulation"
OUT_CSV = OUT_DIR / "split_quality_diagnostic.csv"
OUT_SUMMARY = OUT_DIR / "split_quality_summary.json"
OUT_REPORT = OUT_DIR / "split_quality_report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
HD_EPS = 1e-12
AUROC_GATE = 0.80

# Split-time features used to predict harmful minority peels (no fate leakage).
PREDICTIVE = [
    "parent_n",
    "parent_purity",
    "parent_radius",
    "parent_n_min",
    "parent_n_maj",
    "n1",
    "n2",
    "purity1",
    "purity2",
    "r1",
    "r2",
    "center_distance",
    "S_geom",
    "size_ratio",
    "purity_gain",
    "small_is_global_minority",
    "small_is_pure_minority",
    "minority_fraction_parent",
    "minority_fraction_small_child",
    "minority_hd1",
    "minority_hd1_norm",
    "d_same_global",
    "d_opp",
    "d_same_global_over_hd1",
]

HIGHLIGHT = [
    "size_ratio",
    "S_geom",
    "purity_gain",
    "d_same_global_over_hd1",
    "minority_hd1_norm",
]


def _build_indexed_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = normalize_labels(y)
    data = np.column_stack([X, y])
    num = data.shape[0]
    index = np.arange(num).reshape(num, 1)
    return np.hstack([data, index])


def _labels(ball: GranularBall) -> np.ndarray:
    return np.asarray(ball.data[:, -2], dtype=float)


def _indices(ball: GranularBall) -> np.ndarray:
    return np.asarray(ball.data[:, -1], dtype=int)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float("nan") for k in ("n", "min", "p10", "p25", "p50", "p75", "p90", "max", "mean")}
    qs = np.quantile(v, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    return {
        "n": int(v.size),
        "min": float(qs[0]),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(np.mean(v)),
    }


def _auroc(scores: np.ndarray, y_true: np.ndarray) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y_true, dtype=int)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    if s.size == 0 or y.min() == y.max():
        return float("nan")
    order = np.argsort(s)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, y.size + 1, dtype=float)
    s_sorted = s[order]
    i = 0
    while i < y.size:
        j = i
        while j + 1 < y.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _median_nn(src: np.ndarray, dst: np.ndarray) -> float:
    if src.size == 0 or dst.size == 0 or src.ndim != 2 or dst.ndim != 2:
        return float("nan")
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return float("nan")
    dists = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
    return float(np.median(np.min(dists, axis=1)))


def _per_point_nn_exclude_self(
    pts: np.ndarray,
    idx: np.ndarray,
    pool_pts: np.ndarray,
    pool_idx: np.ndarray,
) -> np.ndarray:
    """Nearest neighbour in pool for each pt, excluding the same sample index."""
    n = pts.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if pool_pts.shape[0] == 0:
        return out
    dists = np.linalg.norm(pts[:, None, :] - pool_pts[None, :, :], axis=2)
    same = idx[:, None] == pool_idx[None, :]
    dists = np.where(same, np.inf, dists)
    mins = np.min(dists, axis=1)
    out[np.isfinite(mins)] = mins[np.isfinite(mins)]
    return out


def _pick_small_large(
    c1: GranularBall,
    c2: GranularBall,
    global_min_label: float,
) -> tuple[GranularBall, GranularBall, str, str]:
    n1, n2 = int(c1.num), int(c2.num)
    m1 = int(np.sum(_labels(c1) == global_min_label))
    m2 = int(np.sum(_labels(c2) == global_min_label))
    if n1 < n2:
        return c1, c2, "first", "second"
    if n2 < n1:
        return c2, c1, "second", "first"
    if m1 >= m2:
        return c1, c2, "first", "second"
    return c2, c1, "second", "first"


def classify_split(row: dict[str, Any]) -> str:
    """
    A: small child contains global minority and is fully deleted.
    B: singleton split whose small child is not a minority peel.
    C: both children have n>=2 and the small child is not fully deleted.
    D: other.
    """
    small_has_min = int(row["small_n_min"]) > 0
    small_deleted = int(row["small_all_deleted"]) == 1
    is_singleton_split = int(row["small_is_singleton"]) == 1
    if small_has_min and small_deleted:
        return "A"
    if is_singleton_split:
        return "B"
    if int(row["n_small"]) >= 2 and not small_deleted:
        return "C"
    return "D"


def collect_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    dataset: str,
    global_min_label: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np.random.seed(seed)
    indexed = _build_indexed_matrix(X_train, y_train)
    y_all = indexed[:, -2]
    feats_all = indexed[:, :-2]
    idx_all = np.arange(indexed.shape[0])
    n_train = indexed.shape[0]
    n_min_train = int(np.sum(y_all == global_min_label))

    pos_pool = feats_all[y_all == 1.0]
    pos_idx = idx_all[y_all == 1.0]
    neg_pool = feats_all[y_all == -1.0]
    neg_idx = idx_all[y_all == -1.0]

    gb = GBList(indexed)
    balls = gb.granular_balls
    splits: list[dict[str, Any]] = []
    split_id = 0

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < PURITY and ball.num > MIN_SAMPLES_PER_BALL:
            parent_feats = ball.data_no_label.copy()
            parent_labels = _labels(ball).copy()
            parent_idx = _indices(ball).copy()
            parent_n = int(ball.num)
            parent_purity = float(ball.purity)
            parent_radius = float(ball.r)
            parent_n_min = int(np.sum(parent_labels == global_min_label))
            parent_n_maj = parent_n - parent_n_min
            parent_n_pos = int(np.sum(parent_labels == 1.0))
            parent_n_neg = int(np.sum(parent_labels == -1.0))

            c1, c2 = ball.split_2balls()
            split_id += 1
            small, large, small_slot, large_slot = _pick_small_large(
                c1, c2, global_min_label
            )
            n1, n2 = int(c1.num), int(c2.num)
            p1, p2 = float(c1.purity), float(c2.purity)
            r1, r2 = float(c1.r), float(c2.r)
            center_distance = float(np.linalg.norm(c1.center - c2.center))
            denom = r1 + r2
            s_geom = float(center_distance / denom) if denom > 1e-12 else float("nan")
            n_small, n_large = int(small.num), int(large.num)
            size_ratio = float(n_small / max(n_large, 1))
            weighted_p = (n1 * p1 + n2 * p2) / max(n1 + n2, 1)
            purity_gain = float(weighted_p - parent_purity)

            small_labels = _labels(small)
            small_idx = _indices(small)
            small_feats = small.data_no_label
            small_n_min = int(np.sum(small_labels == global_min_label))
            small_n_maj = n_small - small_n_min
            small_label = float(-1.0 if small.label == -1 else 1.0)
            small_is_global_min = int(small_label == global_min_label)
            small_is_pure = float(small.purity) >= 1.0 - 1e-12
            small_is_pure_min = int(small_is_pure and small_is_global_min == 1)

            min_mask_small = small_labels == global_min_label
            min_mask_parent = parent_labels == global_min_label
            opp_mask_parent = ~min_mask_parent

            if np.any(min_mask_small) and np.any(opp_mask_parent):
                minority_hd1 = _median_nn(
                    small_feats[min_mask_small], parent_feats[opp_mask_parent]
                )
            elif np.any(min_mask_parent) and np.any(opp_mask_parent):
                minority_hd1 = _median_nn(
                    parent_feats[min_mask_parent], parent_feats[opp_mask_parent]
                )
            else:
                minority_hd1 = float("nan")
            minority_hd1_norm = (
                float(minority_hd1 / max(parent_radius, HD_EPS))
                if np.isfinite(minority_hd1)
                else float("nan")
            )

            # Global same-class / opposite-class distances for small-child samples.
            same_d = np.full(n_small, np.nan)
            opp_d = np.full(n_small, np.nan)
            pos_m = small_labels == 1.0
            neg_m = small_labels == -1.0
            if np.any(pos_m):
                same_d[pos_m] = _per_point_nn_exclude_self(
                    small_feats[pos_m], small_idx[pos_m], pos_pool, pos_idx
                )
                opp_d[pos_m] = _per_point_nn_exclude_self(
                    small_feats[pos_m], small_idx[pos_m], neg_pool, neg_idx
                )
            if np.any(neg_m):
                same_d[neg_m] = _per_point_nn_exclude_self(
                    small_feats[neg_m], small_idx[neg_m], neg_pool, neg_idx
                )
                opp_d[neg_m] = _per_point_nn_exclude_self(
                    small_feats[neg_m], small_idx[neg_m], pos_pool, pos_idx
                )

            if np.any(min_mask_small):
                d_same_global = float(np.nanmedian(same_d[min_mask_small]))
                d_opp = float(np.nanmedian(opp_d[min_mask_small]))
            else:
                d_same_global = float(np.nanmedian(same_d))
                d_opp = float(np.nanmedian(opp_d))

            hd1_for_ratio = minority_hd1 if np.isfinite(minority_hd1) else d_opp
            if (
                np.isfinite(d_same_global)
                and np.isfinite(hd1_for_ratio)
                and hd1_for_ratio > HD_EPS
            ):
                ratio = float(d_same_global / hd1_for_ratio)
            else:
                ratio = float("nan")

            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "split_id": split_id,
                "parent_n": parent_n,
                "parent_purity": parent_purity,
                "parent_radius": parent_radius,
                "parent_n_min": parent_n_min,
                "parent_n_maj": parent_n_maj,
                "parent_n_pos": parent_n_pos,
                "parent_n_neg": parent_n_neg,
                "n1": n1,
                "n2": n2,
                "purity1": p1,
                "purity2": p2,
                "r1": r1,
                "r2": r2,
                "center_distance": center_distance,
                "S_geom": s_geom,
                "size_ratio": size_ratio,
                "purity_gain": purity_gain,
                "weighted_child_purity": weighted_p,
                "n_small": n_small,
                "n_large": n_large,
                "small_slot": small_slot,
                "large_slot": large_slot,
                "small_label": small_label,
                "small_purity": float(small.purity),
                "small_radius": float(small.r),
                "small_n_min": small_n_min,
                "small_n_maj": small_n_maj,
                "small_is_global_minority": small_is_global_min,
                "small_is_pure_minority": small_is_pure_min,
                "small_is_singleton": int(n_small == 1),
                "minority_fraction_parent": float(parent_n_min / parent_n) if parent_n else float("nan"),
                "minority_fraction_small_child": float(small_n_min / n_small) if n_small else float("nan"),
                "minority_hd1": minority_hd1,
                "minority_hd1_norm": minority_hd1_norm,
                "d_same_global": d_same_global,
                "d_opp": d_opp,
                "d_same_global_over_hd1": ratio,
                "small_indices": ",".join(map(str, small_idx.tolist())),
                "large_indices": ",".join(map(str, _indices(large).tolist())),
            }
            splits.append(row)
            balls[i] = c1
            balls.append(c2)
            ll += 1
        else:
            i += 1

    sample_leaf_n: dict[int, int] = {}
    for b in balls:
        n = int(b.num)
        for s in _indices(b):
            sample_leaf_n[int(s)] = n

    def _parse_idx(s: str) -> list[int]:
        if not s:
            return []
        return [int(x) for x in s.split(",") if x != ""]

    n_min_lost_run = 0
    n_maj_lost_run = 0
    n_A = n_B = n_C = n_D = 0
    for row in splits:
        small_ids = _parse_idx(row["small_indices"])
        large_ids = _parse_idx(row["large_indices"])
        small_del = [sample_leaf_n.get(s, 1) < MIN_BALL_COUNT for s in small_ids]
        large_del = [sample_leaf_n.get(s, 1) < MIN_BALL_COUNT for s in large_ids]
        y_small = y_all[np.array(small_ids, dtype=int)] if small_ids else np.array([])
        y_large = y_all[np.array(large_ids, dtype=int)] if large_ids else np.array([])
        small_all_deleted = int(len(small_del) > 0 and all(small_del))
        n_min_lost_small = int(
            np.sum((y_small == global_min_label) & np.array(small_del, dtype=bool))
        ) if small_ids else 0
        n_maj_lost_small = int(
            np.sum((y_small != global_min_label) & np.array(small_del, dtype=bool))
        ) if small_ids else 0
        n_min_lost_large = int(
            np.sum((y_large == global_min_label) & np.array(large_del, dtype=bool))
        ) if large_ids else 0
        n_maj_lost_large = int(
            np.sum((y_large != global_min_label) & np.array(large_del, dtype=bool))
        ) if large_ids else 0
        row["small_all_deleted"] = small_all_deleted
        row["small_frac_deleted"] = float(np.mean(small_del)) if small_del else float("nan")
        row["n_min_lost_small"] = n_min_lost_small
        row["n_maj_lost_small"] = n_maj_lost_small
        row["n_min_lost_split"] = n_min_lost_small + n_min_lost_large
        row["n_maj_lost_split"] = n_maj_lost_small + n_maj_lost_large
        row["minority_loss"] = n_min_lost_small
        row["majority_loss"] = n_maj_lost_small
        row["split_class"] = classify_split(row)
        n_min_lost_run += n_min_lost_small
        n_maj_lost_run += n_maj_lost_small
        sc = row["split_class"]
        if sc == "A":
            n_A += 1
        elif sc == "B":
            n_B += 1
        elif sc == "C":
            n_C += 1
        else:
            n_D += 1
        del row["small_indices"]
        del row["large_indices"]

    gb.del_ball(num_data=MIN_BALL_COUNT)
    run = {
        "dataset": dataset,
        "seed": seed,
        "n_train": n_train,
        "n_global_min": n_min_train,
        "n_splits": len(splits),
        "n_A": n_A,
        "n_B": n_B,
        "n_C": n_C,
        "n_D": n_D,
        "n_min_lost_small": n_min_lost_run,
        "n_maj_lost_small": n_maj_lost_run,
        "n_balls_after_del": len(gb.granular_balls),
    }
    return splits, run


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    v = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(v) < 3:
        return float("nan")
    if v["x"].nunique() < 2 or v["y"].nunique() < 2:
        return float("nan")
    return float(v["x"].corr(v["y"], method=method))


def _zscore(s: pd.Series) -> np.ndarray:
    v = s.to_numpy(dtype=float)
    m = np.isfinite(v)
    out = np.full_like(v, np.nan, dtype=float)
    if m.sum() < 2:
        return out
    mu = float(np.mean(v[m]))
    sd = float(np.std(v[m]))
    if sd < 1e-15:
        out[m] = 0.0
        return out
    out[m] = (v[m] - mu) / sd
    return out


def _best_threshold(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Youden J on finite scores; higher score predicted as A."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    empty = {
        "threshold": float("nan"),
        "youden": float("nan"),
        "tpr": float("nan"),
        "fpr": float("nan"),
        "acc": float("nan"),
        "n": 0,
    }
    if s.size == 0 or y.min() == y.max():
        return empty
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    # Cumulative from high to low: treat threshold as s >= t
    best = empty
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    # Unique thresholds from high to low
    uniq = np.unique(s_sorted)
    for t in uniq[::-1]:
        pred = s >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        tn = n_neg - fp
        fn = n_pos - tp
        tpr = tp / n_pos if n_pos else float("nan")
        fpr = fp / n_neg if n_neg else float("nan")
        spec = 1.0 - fpr if np.isfinite(fpr) else float("nan")
        j = tpr + spec - 1.0 if np.isfinite(tpr) and np.isfinite(spec) else float("nan")
        acc = (tp + tn) / y.size
        if np.isfinite(j) and (not np.isfinite(best["youden"]) or j > best["youden"]):
            best = {
                "threshold": float(t),
                "youden": float(j),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "acc": float(acc),
                "n": int(y.size),
            }
    return best


def summarize(df: pd.DataFrame, runs: pd.DataFrame) -> dict[str, Any]:
    n = int(len(df))
    y_a = (df["split_class"] == "A").astype(int) if n else pd.Series(dtype=int)
    counts = df["split_class"].value_counts().to_dict() if n else {}

    by_class: dict[str, Any] = {}
    for cls in ("A", "B", "C", "D"):
        sub = df[df["split_class"] == cls] if n else df
        feats = {col: _quantiles(sub[col].to_numpy()) for col in PREDICTIVE if col in sub}
        by_class[cls] = {
            "n": int(len(sub)),
            "frac": float(len(sub) / n) if n else float("nan"),
            "mean_minority_loss": float(sub["minority_loss"].mean()) if len(sub) else float("nan"),
            "mean_majority_loss": float(sub["majority_loss"].mean()) if len(sub) else float("nan"),
            "frac_singleton": float(sub["small_is_singleton"].mean()) if len(sub) else float("nan"),
            "frac_small_deleted": float(sub["small_all_deleted"].mean()) if len(sub) else float("nan"),
            "features": feats,
        }

    corrs = []
    outcomes = ["minority_loss", "is_A"]
    work = df.copy() if n else df
    if n:
        work["is_A"] = y_a
    for feat in PREDICTIVE:
        if feat not in work:
            continue
        row = {"feature": feat}
        for out in outcomes:
            row[f"pearson_{out}"] = _corr(work[feat], work[out], "pearson")
            row[f"spearman_{out}"] = _corr(work[feat], work[out], "spearman")
        row["auroc_A"] = _auroc(work[feat].to_numpy(), y_a.to_numpy()) if n else float("nan")
        row["auroc_A_inverted"] = _auroc(-work[feat].to_numpy(), y_a.to_numpy()) if n else float("nan")
        best_dir = "higher" if (
            np.isfinite(row["auroc_A"])
            and (not np.isfinite(row["auroc_A_inverted"]) or row["auroc_A"] >= row["auroc_A_inverted"])
        ) else "lower"
        row["auroc_A_best"] = max(
            (v for v in (row["auroc_A"], row["auroc_A_inverted"]) if np.isfinite(v)),
            default=float("nan"),
        )
        row["auroc_direction"] = best_dir
        signed = work[feat].to_numpy() if best_dir == "higher" else -work[feat].to_numpy()
        row["youden"] = _best_threshold(signed, y_a.to_numpy()) if n else {}
        corrs.append(row)

    corrs_sorted = sorted(
        corrs,
        key=lambda r: -r["auroc_A_best"] if np.isfinite(r["auroc_A_best"]) else 0.0,
    )

    # 2-feature signed z-sum among highlight features, plus all PREDICTIVE pairs
    # restricted to HIGHLIGHT as requested; also scan all PREDICTIVE pairs.
    pair_rows = []
    z = {c: _zscore(work[c]) for c in PREDICTIVE if c in work} if n else {}
    y = y_a.to_numpy() if n else np.array([])

    def _pair_scan(names: list[str], tag: str) -> None:
        cols = [c for c in names if c in z]
        for a, b in itertools.combinations(cols, 2):
            best = {"auroc": float("nan"), "signs": "", "features": f"{a}+{b}"}
            for sa, sb in itertools.product((1.0, -1.0), (1.0, -1.0)):
                score = sa * z[a] + sb * z[b]
                auc = _auroc(score, y)
                if np.isfinite(auc) and (not np.isfinite(best["auroc"]) or auc > best["auroc"]):
                    best = {
                        "auroc": float(auc),
                        "signs": f"{'+' if sa > 0 else '-'}{a} {'+' if sb > 0 else '-'}{b}",
                        "features": f"{a}|{b}",
                        "set": tag,
                    }
            pair_rows.append(best)

    if n:
        _pair_scan(HIGHLIGHT, "highlight")
        _pair_scan(PREDICTIVE, "all_predictive")

    highlight_pairs = [p for p in pair_rows if p.get("set") == "highlight"]
    all_pairs = [p for p in pair_rows if p.get("set") == "all_predictive"]
    highlight_pairs = sorted(highlight_pairs, key=lambda p: -p["auroc"] if np.isfinite(p["auroc"]) else 0)
    all_pairs = sorted(all_pairs, key=lambda p: -p["auroc"] if np.isfinite(p["auroc"]) else 0)

    by_dataset = []
    if n:
        for ds, g in work.groupby("dataset"):
            ya = (g["split_class"] == "A").astype(int).to_numpy()
            rec: dict[str, Any] = {
                "dataset": ds,
                "n_splits": int(len(g)),
                "n_A": int((g["split_class"] == "A").sum()),
                "n_B": int((g["split_class"] == "B").sum()),
                "n_C": int((g["split_class"] == "C").sum()),
                "n_D": int((g["split_class"] == "D").sum()),
                "frac_A": float((g["split_class"] == "A").mean()),
                "mean_minority_loss": float(g["minority_loss"].mean()),
            }
            for feat in HIGHLIGHT:
                rec[f"auroc_{feat}"] = _auroc(g[feat].to_numpy(), ya)
                rec[f"auroc_{feat}_inv"] = _auroc(-g[feat].to_numpy(), ya)
                rec[f"auroc_{feat}_best"] = max(
                    (
                        v
                        for v in (rec[f"auroc_{feat}"], rec[f"auroc_{feat}_inv"])
                        if np.isfinite(v)
                    ),
                    default=float("nan"),
                )
            by_dataset.append(rec)
        by_dataset = sorted(by_dataset, key=lambda r: -r["frac_A"])

    single_best = corrs_sorted[0] if corrs_sorted else None
    pair_best = all_pairs[0] if all_pairs else None
    hi_pair_best = highlight_pairs[0] if highlight_pairs else None
    single_ok = bool(
        single_best and np.isfinite(single_best["auroc_A_best"]) and single_best["auroc_A_best"] >= AUROC_GATE
    )
    pair_ok = bool(pair_best and np.isfinite(pair_best["auroc"]) and pair_best["auroc"] >= AUROC_GATE)
    hi_pair_ok = bool(
        hi_pair_best and np.isfinite(hi_pair_best["auroc"]) and hi_pair_best["auroc"] >= AUROC_GATE
    )

    highlight_corrs = [r for r in corrs_sorted if r["feature"] in HIGHLIGHT]
    highlight_best_single = highlight_corrs[0] if highlight_corrs else None
    highlight_single_ok = bool(
        highlight_best_single
        and np.isfinite(highlight_best_single["auroc_A_best"])
        and highlight_best_single["auroc_A_best"] >= AUROC_GATE
    )

    hi_auc = float("nan")
    hi_name = "none"
    if highlight_best_single:
        hi_name = str(highlight_best_single["feature"])
        hi_auc = float(highlight_best_single["auroc_A_best"])

    if n == 0:
        verdict = "No splits recorded."
    elif highlight_single_ok:
        verdict = (
            f"YES: a highlighted geometry/distance feature reaches AUROC>={AUROC_GATE} "
            f"for A: {hi_name} (AUROC={hi_auc:.3f}). "
            "Descriptive on the same data, not held-out."
        )
    elif single_ok and not highlight_single_ok:
        verdict = (
            f"YES on class-structure / parent-size features, NO on the five highlighted "
            f"geometry scores. Best overall: {single_best['feature']} "
            f"(AUROC={single_best['auroc_A_best']:.3f}). "
            f"Best among size_ratio / S_geom / purity_gain / d_same_global/hd1 / "
            f"minority_hd1_norm: {hi_name} (AUROC={hi_auc:.3f}). "
            "A Youden threshold of 0.5 on minority_fraction_small_child marks most "
            "harmful peels (small child at least half minority). S_geom and size_ratio "
            "do not clear 0.80 on their own."
        )
    elif hi_pair_ok:
        verdict = (
            f"PARTIAL: no single feature hits {AUROC_GATE}, but a 2-feature signed z-sum "
            f"among the highlighted geometry/distance features does "
            f"({hi_pair_best['signs']}, AUROC={hi_pair_best['auroc']:.3f}). "
            "Signs were selected on the same data."
        )
    elif pair_ok:
        verdict = (
            f"PARTIAL: no highlighted single/pair hits {AUROC_GATE}, but some 2-feature "
            f"z-sum among all split-time features does ({pair_best['signs']}, "
            f"AUROC={pair_best['auroc']:.3f}). Treat as descriptive only."
        )
    else:
        verdict = (
            f"NO: no single split-time feature and no simple 2-feature signed z-sum "
            f"reaches AUROC>={AUROC_GATE} for A. size_ratio, S_geom, purity_gain, "
            "d_same_global/hd1, and minority hd1_norm do not by themselves mark "
            "harmful minority-peeling splits. Do not install a granulation gate yet."
        )

    return {
        "n_splits": n,
        "n_runs": int(len(runs)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_frac": {k: float(v / n) for k, v in counts.items()} if n else {},
        "label_definitions": {
            "A": "small child contains global minority and is fully deleted by del_ball",
            "B": "singleton split whose small child is not a minority peel",
            "C": "min(n1,n2)>=2 and small child is not fully deleted (retained split)",
            "D": "other",
        },
        "by_class": by_class,
        "correlations_and_auroc": corrs_sorted,
        "highlight_pair_auroc": highlight_pairs,
        "all_pair_auroc_top20": all_pairs[:20],
        "by_dataset": by_dataset,
        "auroc_gate": AUROC_GATE,
        "single_best": single_best,
        "highlight_pair_best": hi_pair_best,
        "any_pair_best": pair_best,
        "simple_rule_auroc_ge_0_80": bool(single_ok or hi_pair_ok or pair_ok),
        "single_feature_auroc_ge_0_80": single_ok,
        "highlight_feature_auroc_ge_0_80": bool(
            any(
                r["feature"] in HIGHLIGHT and np.isfinite(r["auroc_A_best"]) and r["auroc_A_best"] >= AUROC_GATE
                for r in corrs_sorted
            )
        ),
        "verdict": verdict,
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if not np.isfinite(x):
            return "nan"
        return f"{x:.3f}"
    return str(x)


def _fmt_q(q: dict[str, float]) -> str:
    if not q or not q.get("n"):
        return "empty"
    return (
        f"n={q['n']} mean={q['mean']:.3f} p10={q['p10']:.3f} "
        f"p50={q['p50']:.3f} p90={q['p90']:.3f}"
    )


def write_report(path: Path, summary: dict[str, Any], runs: pd.DataFrame, failed: list) -> None:
    lines = [
        "# Split quality diagnostic",
        "",
        "Read-only replay of frozen recursive KMeans(k=2) splits "
        f"(purity={PURITY}, min_samples_per_ball={MIN_SAMPLES_PER_BALL}, "
        f"del_ball num_data={MIN_BALL_COUNT}).",
        "",
        "A/B/C/D labels use split geometry and del_ball fate only. "
        "**Topo / classifier performance is not used.**",
        "",
        "## Label definitions",
        "",
        f"- **A** — {summary['label_definitions']['A']}",
        f"- **B** — {summary['label_definitions']['B']}",
        f"- **C** — {summary['label_definitions']['C']}",
        f"- **D** — {summary['label_definitions']['D']}",
        "",
        "## Headline",
        "",
        f"- Splits: {summary['n_splits']} across {summary['n_runs']} runs",
        f"- Class counts: {summary['class_counts']}",
        f"- Class fractions: {{{', '.join(f'{k}: {v:.3f}' for k, v in summary['class_frac'].items())}}}",
        f"- AUROC gate: {summary['auroc_gate']}",
        f"- simple_rule_auroc_ge_0.80: {summary['simple_rule_auroc_ge_0_80']}",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
    ]
    sb = summary.get("single_best") or {}
    if sb:
        lines.append(
            f"Best single feature: `{sb.get('feature')}` AUROC={_fmt(sb.get('auroc_A_best', float('nan')))} "
            f"({sb.get('auroc_direction')} => A)."
        )
        yod = sb.get("youden") or {}
        if yod:
            lines.append(
                f"Youden threshold on the oriented score: t={_fmt(yod.get('threshold', float('nan')))}, "
                f"J={_fmt(yod.get('youden', float('nan')))}, TPR={_fmt(yod.get('tpr', float('nan')))}, "
                f"FPR={_fmt(yod.get('fpr', float('nan')))}, acc={_fmt(yod.get('acc', float('nan')))}."
            )
        lines.append("")

    lines.extend(["## Highlighted features by class", ""])
    lines.append("| Feature | A mean/p50 | B mean/p50 | C mean/p50 | D mean/p50 |")
    lines.append("|---|---|---|---|---|")
    for feat in HIGHLIGHT:
        cells = [feat]
        for cls in ("A", "B", "C", "D"):
            q = summary["by_class"][cls]["features"].get(feat, {})
            cells.append(f"{_fmt(q.get('mean', float('nan')))}/{_fmt(q.get('p50', float('nan')))}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    for cls in ("A", "B", "C", "D"):
        pack = summary["by_class"][cls]
        if pack["n"] == 0:
            lines.append(f"### Class {cls}: none")
            lines.append("")
            continue
        lines.append(f"### Class {cls} (n={pack['n']}, frac={_fmt(pack['frac'])})")
        lines.append(
            f"- minority_loss mean={_fmt(pack['mean_minority_loss'])}, "
            f"majority_loss mean={_fmt(pack['mean_majority_loss'])}, "
            f"singleton={_fmt(pack['frac_singleton'])}, "
            f"small_deleted={_fmt(pack['frac_small_deleted'])}"
        )
        for feat in HIGHLIGHT + [
            "parent_n",
            "parent_purity",
            "size_ratio",
            "minority_fraction_small_child",
            "small_is_global_minority",
        ]:
            q = pack["features"].get(feat)
            if q:
                lines.append(f"- {feat}: {_fmt_q(q)}")
        lines.append("")

    lines.extend(
        [
            "## Correlations with minority loss and AUROC for A",
            "",
            "| Feature | Pearson loss | Spearman loss | Pearson A | Spearman A | AUROC A | dir |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in summary["correlations_and_auroc"]:
        lines.append(
            f"| {r['feature']} | {_fmt(r['pearson_minority_loss'])} | "
            f"{_fmt(r['spearman_minority_loss'])} | {_fmt(r['pearson_is_A'])} | "
            f"{_fmt(r['spearman_is_A'])} | {_fmt(r['auroc_A_best'])} | {r['auroc_direction']} |"
        )

    lines.extend(
        [
            "",
            "## 2-feature signed z-sum (highlighted pair)",
            "",
            "| Pair | AUROC A |",
            "|---|---:|",
        ]
    )
    for p in summary["highlight_pair_auroc"]:
        lines.append(f"| {p['signs']} | {_fmt(p['auroc'])} |")

    lines.extend(
        [
            "",
            "## Top 2-feature signed z-sums (all predictive features)",
            "",
            "| Pair | AUROC A |",
            "|---|---:|",
        ]
    )
    for p in summary["all_pair_auroc_top20"]:
        lines.append(f"| {p['signs']} | {_fmt(p['auroc'])} |")

    lines.extend(
        [
            "",
            "## Dataset-wise",
            "",
            "| dataset | splits | A | B | C | D | frac A | "
            + " | ".join(f"AUROC {f}" for f in HIGHLIGHT)
            + " |",
            "|---|---:|---:|---:|---:|---:|---:|"
            + "|---:|" * len(HIGHLIGHT),
        ]
    )
    for r in summary["by_dataset"]:
        aucs = " | ".join(_fmt(r[f"auroc_{f}_best"]) for f in HIGHLIGHT)
        lines.append(
            f"| {r['dataset']} | {r['n_splits']} | {r['n_A']} | {r['n_B']} | "
            f"{r['n_C']} | {r['n_D']} | {_fmt(r['frac_A'])} | {aucs} |"
        )

    lines.extend(
        [
            "",
            "## Per-run split counts",
            "",
            "| dataset | seed | n_train | splits | A | B | C | D | min lost |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if len(runs):
        for _, r in runs.sort_values(["dataset", "seed"]).iterrows():
            lines.append(
                f"| {r['dataset']} | {int(r['seed'])} | {int(r['n_train'])} | "
                f"{int(r['n_splits'])} | {int(r['n_A'])} | {int(r['n_B'])} | "
                f"{int(r['n_C'])} | {int(r['n_D'])} | {int(r['n_min_lost_small'])} |"
            )

    if failed:
        lines.extend(["", "## Failed loads", ""])
        for item in failed:
            lines.append(f"- {item['dataset']} seed={item['seed']}: {item['error']}")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{OUT_CSV.name}`",
            f"- `{OUT_SUMMARY.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--summarize-only" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        denom = df["r1"] + df["r2"]
        df["S_geom"] = np.where(denom > 1e-12, df["center_distance"] / denom, np.nan)
        df.to_csv(OUT_CSV, index=False)
        runs = pd.read_csv(OUT_DIR / "split_quality_runs.csv")
        summary = summarize(df, runs)
        proto = {}
        if OUT_SUMMARY.exists():
            proto = json.loads(OUT_SUMMARY.read_text(encoding="utf-8")).get("protocol", {})
        summary["protocol"] = proto
        write_report(OUT_REPORT, summary, runs, proto.get("failed", []))
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary["verdict"])
        return 0

    descriptors = discover_available_datasets()
    all_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for desc in descriptors:
        for seed in SEEDS:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append(
                    {"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue
            rows, run = collect_run(
                split.X_train,
                split.y_train,
                seed=seed,
                dataset=split.name,
                global_min_label=float(split.minority_label),
            )
            run_rows.append(run)
            all_rows.extend(rows)
            print(
                f"{split.name} seed={seed}: splits={run['n_splits']} "
                f"A={run['n_A']} B={run['n_B']} C={run['n_C']} D={run['n_D']} "
                f"min_lost={run['n_min_lost_small']}"
            )

    df = pd.DataFrame(all_rows)
    runs = pd.DataFrame(run_rows)
    df.to_csv(OUT_CSV, index=False)
    runs.to_csv(OUT_DIR / "split_quality_runs.csv", index=False)

    summary = summarize(df, runs)
    summary["protocol"] = {
        "purity": PURITY,
        "min_samples_per_ball": MIN_SAMPLES_PER_BALL,
        "min_ball_count": MIN_BALL_COUNT,
        "seeds": SEEDS,
        "datasets": [d["name"] for d in descriptors],
        "failed": failed,
        "purity_gain": "weighted_child_purity - parent_purity",
        "S_geom": "||c1-c2|| / (r1+r2+eps)",
        "size_ratio": "min(n1,n2)/max(n1,n2)",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, summary, runs, failed)
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
