"""
Split-utility diagnostic (read-only).

Replays frozen recursive KMeans(k=2) granulation (rho=0.95, min_samples=1,
del_ball num_data=2) on all 17 discoverable datasets x seeds [42, 0, 1].

Records every proposed split, classifies A/B/C/D from geometry and del_ball
fate only. Does NOT modify granulation, del_ball, Topo, CW, or GBLSTSVM.

Run from GBLSTSVM root:
  python experiments/analyze_split_utility.py
"""

from __future__ import annotations

import json
import sys
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
OUT_CSV = OUT_DIR / "split_utility.csv"
OUT_SUMMARY = OUT_DIR / "split_utility_summary.json"
OUT_REPORT = OUT_DIR / "split_utility_report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
AUROC_GATE = 0.80

# Numeric metrics for distribution / AUROC tables (split-time unless noted).
METRICS = [
    "parent_n",
    "child_n1",
    "child_n2",
    "size_ratio",
    "parent_purity",
    "purity1",
    "purity2",
    "purity_gain",
    "parent_radius",
    "r1",
    "r2",
    "weighted_child_radius",
    "radius_reduction",
    "center_distance",
    "center_radius_separation",
    "center_distance_over_parent_r",
    "minority_fraction_parent",
    "minority_fraction_child1",
    "minority_fraction_child2",
    "n_min_retained_child1",
    "n_min_retained_child2",
    "minority_concentrated_in_small",
    "parent_mean_d_opp",
    "parent_min_d_opp",
    "parent_mean_d_same",
    "parent_min_d_same",
    "minority_margin_mean",
    "minority_margin_min",
    "child_compactness_gain",
    "split_depth",
    "either_child_singleton",
    "child1_later_deleted",
    "child2_later_deleted",
    "either_child_deleted",
]

# Outcome-adjacent or definitional — excluded from the main non-tautological question.
TAUTOLOGICAL = {
    "n_min_retained_child1",
    "n_min_retained_child2",
    "minority_concentrated_in_small",
    "either_child_singleton",
    "child1_later_deleted",
    "child2_later_deleted",
    "either_child_deleted",
    "child_n1",
    "child_n2",
    "size_ratio",
    "minority_fraction_child1",
    "minority_fraction_child2",
}

# Recursion / ball-size proxies — informative but not principled split-quality geometry.
STRUCTURAL = {
    "parent_n",
    "split_depth",
}

COMPOSITE_SPECS: list[tuple[str, str]] = [
    ("composite_radius_reduction", "radius_reduction"),
    ("composite_center_separation", "center_radius_separation"),
    (
        "composite_separation_over_compactness",
        "center_distance / max(weighted_child_radius, eps)",
    ),
    ("composite_minority_margin", "minority_margin_mean"),
    (
        "composite_radius_plus_separation",
        "radius_reduction / max(parent_radius, eps) + center_radius_separation",
    ),
    (
        "composite_margin_plus_separation",
        "minority_margin_mean / max(parent_radius, eps) + center_radius_separation",
    ),
]

VERDICT_LABELS = {
    1: "principled split criterion supported",
    2: "split criterion not supported; filtering is the likely issue",
    3: "failure is primarily recursion/depth driven",
    4: "evidence inconclusive",
}


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
        return {
            k: float("nan")
            for k in ("n", "min", "p10", "p25", "p50", "p75", "p90", "max", "mean")
        }
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


def _best_auroc(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, str]:
    fwd = _auroc(scores, y_true)
    inv = _auroc(-scores, y_true)
    if np.isfinite(fwd) and (not np.isfinite(inv) or fwd >= inv):
        return fwd, "higher"
    return inv, "lower"


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


def _parent_distance_stats(
    parent_feats: np.ndarray,
    parent_labels: np.ndarray,
    global_min_label: float,
) -> dict[str, float]:
    n = parent_feats.shape[0]
    if n <= 1:
        nan = float("nan")
        return {
            "parent_mean_d_opp": nan,
            "parent_min_d_opp": nan,
            "parent_mean_d_same": nan,
            "parent_min_d_same": nan,
            "minority_margin_mean": nan,
            "minority_margin_min": nan,
        }

    d_opp = np.full(n, np.nan)
    d_same = np.full(n, np.nan)
    for i in range(n):
        dists = np.linalg.norm(parent_feats - parent_feats[i], axis=1)
        dists[i] = np.inf
        opp = parent_labels != parent_labels[i]
        same = parent_labels == parent_labels[i]
        if np.any(opp):
            d_opp[i] = float(np.min(dists[opp]))
        if np.sum(same) > 1:
            d_same[i] = float(np.min(dists[same & (np.arange(n) != i)]))

    min_mask = parent_labels == global_min_label
    margins = d_opp[min_mask] - d_same[min_mask]
    margins = margins[np.isfinite(margins)]

    return {
        "parent_mean_d_opp": float(np.nanmean(d_opp)),
        "parent_min_d_opp": float(np.nanmin(d_opp)),
        "parent_mean_d_same": float(np.nanmean(d_same)),
        "parent_min_d_same": float(np.nanmin(d_same)),
        "minority_margin_mean": float(np.mean(margins)) if margins.size else float("nan"),
        "minority_margin_min": float(np.min(margins)) if margins.size else float("nan"),
    }


def classify_split(row: dict[str, Any]) -> str:
    """
    A: small child contains global minority and is fully deleted by del_ball.
    B: singleton split whose small child is not a minority peel.
    C: both children survive del_ball (neither branch fully deleted).
    D: other.
    """
    small_has_min = int(row["small_n_min"]) > 0
    small_deleted = int(row["small_all_deleted"]) == 1
    is_singleton_split = int(row["n_small"]) == 1
    if small_has_min and small_deleted:
        return "A"
    if is_singleton_split:
        return "B"
    if not int(row["child1_later_deleted"]) and not int(row["child2_later_deleted"]):
        return "C"
    return "D"


def _child_all_deleted(
    child_ids: list[int],
    sample_leaf_n: dict[int, int],
) -> int:
    if not child_ids:
        return 0
    return int(all(sample_leaf_n.get(s, 1) < MIN_BALL_COUNT for s in child_ids))


def _retained_minority(
    child_ids: list[int],
    y_all: np.ndarray,
    sample_leaf_n: dict[int, int],
    global_min_label: float,
) -> int:
    if not child_ids:
        return 0
    total = 0
    for s in child_ids:
        if sample_leaf_n.get(s, 1) >= MIN_BALL_COUNT and y_all[s] == global_min_label:
            total += 1
    return total


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

    gb = GBList(indexed)
    balls = gb.granular_balls
    depths = [0]
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
            parent_depth = int(depths[i])

            c1, c2 = ball.split_2balls()
            split_id += 1
            small, large, small_slot, large_slot = _pick_small_large(
                c1, c2, global_min_label
            )
            n1, n2 = int(c1.num), int(c2.num)
            p1, p2 = float(c1.purity), float(c2.purity)
            r1, r2 = float(c1.r), float(c2.r)
            center_distance = float(np.linalg.norm(c1.center - c2.center))
            weighted_child_radius = float((n1 * r1 + n2 * r2) / max(n1 + n2, 1))
            radius_reduction = float(parent_radius - weighted_child_radius)
            center_radius_separation = float(
                center_distance / max(r1 + r2, EPS)
            )
            center_distance_over_parent_r = float(
                center_distance / max(parent_radius, EPS)
            )
            n_small, n_large = int(small.num), int(large.num)
            size_ratio = float(n_small / max(n_large, 1))
            weighted_p = (n1 * p1 + n2 * p2) / max(n1 + n2, 1)
            purity_gain = float(weighted_p - parent_purity)
            child_compactness_gain = float(
                1.0 - weighted_child_radius / max(parent_radius, EPS)
            )

            labels_c1 = _labels(c1)
            labels_c2 = _labels(c2)
            n_min_c1 = int(np.sum(labels_c1 == global_min_label))
            n_min_c2 = int(np.sum(labels_c2 == global_min_label))
            small_labels = _labels(small)
            small_n_min = int(np.sum(small_labels == global_min_label))
            large_n_min = int(np.sum(_labels(large) == global_min_label))

            dist_stats = _parent_distance_stats(
                parent_feats, parent_labels, global_min_label
            )

            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "split_id": split_id,
                "split_depth": parent_depth,
                "parent_n": parent_n,
                "child_n1": n1,
                "child_n2": n2,
                "size_ratio": size_ratio,
                "parent_purity": parent_purity,
                "purity1": p1,
                "purity2": p2,
                "purity_gain": purity_gain,
                "parent_radius": parent_radius,
                "r1": r1,
                "r2": r2,
                "weighted_child_radius": weighted_child_radius,
                "radius_reduction": radius_reduction,
                "center_distance": center_distance,
                "center_radius_separation": center_radius_separation,
                "center_distance_over_parent_r": center_distance_over_parent_r,
                "minority_fraction_parent": float(parent_n_min / parent_n)
                if parent_n
                else float("nan"),
                "minority_fraction_child1": float(n_min_c1 / n1) if n1 else float("nan"),
                "minority_fraction_child2": float(n_min_c2 / n2) if n2 else float("nan"),
                "minority_concentrated_in_small": int(
                    (small_n_min / max(n_small, 1))
                    > (large_n_min / max(n_large, 1))
                ),
                "n_small": n_small,
                "n_large": n_large,
                "small_slot": small_slot,
                "large_slot": large_slot,
                "small_n_min": small_n_min,
                "either_child_singleton": int(n1 == 1 or n2 == 1),
                "child1_is_singleton": int(n1 == 1),
                "child2_is_singleton": int(n2 == 1),
                "child1_indices": ",".join(map(str, _indices(c1).tolist())),
                "child2_indices": ",".join(map(str, _indices(c2).tolist())),
                "small_indices": ",".join(map(str, _indices(small).tolist())),
                **dist_stats,
                "child_compactness_gain": child_compactness_gain,
            }
            splits.append(row)
            balls[i] = c1
            balls.append(c2)
            depths[i] = parent_depth + 1
            depths.append(parent_depth + 1)
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

    n_A = n_B = n_C = n_D = 0
    for row in splits:
        c1_ids = _parse_idx(row["child1_indices"])
        c2_ids = _parse_idx(row["child2_indices"])
        small_ids = _parse_idx(row["small_indices"])

        row["child1_later_deleted"] = _child_all_deleted(c1_ids, sample_leaf_n)
        row["child2_later_deleted"] = _child_all_deleted(c2_ids, sample_leaf_n)
        row["either_child_deleted"] = int(
            row["child1_later_deleted"] or row["child2_later_deleted"]
        )
        row["n_min_retained_child1"] = _retained_minority(
            c1_ids, y_all, sample_leaf_n, global_min_label
        )
        row["n_min_retained_child2"] = _retained_minority(
            c2_ids, y_all, sample_leaf_n, global_min_label
        )

        small_del = [sample_leaf_n.get(s, 1) < MIN_BALL_COUNT for s in small_ids]
        row["small_all_deleted"] = int(len(small_del) > 0 and all(small_del))
        row["split_class"] = classify_split(row)

        sc = row["split_class"]
        if sc == "A":
            n_A += 1
        elif sc == "B":
            n_B += 1
        elif sc == "C":
            n_C += 1
        else:
            n_D += 1

        for k in ("child1_indices", "child2_indices", "small_indices"):
            del row[k]

    gb.del_ball(num_data=MIN_BALL_COUNT)
    run = {
        "dataset": dataset,
        "seed": seed,
        "n_splits": len(splits),
        "n_A": n_A,
        "n_B": n_B,
        "n_C": n_C,
        "n_D": n_D,
        "n_balls_after_del": len(gb.granular_balls),
    }
    return splits, run


def _add_composites(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["composite_radius_reduction"] = out["radius_reduction"]
    out["composite_center_separation"] = out["center_radius_separation"]
    out["composite_separation_over_compactness"] = out["center_distance"] / out[
        "weighted_child_radius"
    ].clip(lower=EPS)
    out["composite_minority_margin"] = out["minority_margin_mean"]
    out["composite_radius_plus_separation"] = (
        out["radius_reduction"] / out["parent_radius"].clip(lower=EPS)
        + out["center_radius_separation"]
    )
    out["composite_margin_plus_separation"] = (
        out["minority_margin_mean"] / out["parent_radius"].clip(lower=EPS)
        + out["center_radius_separation"]
    )
    return out


def _metric_auroc_table(
    df: pd.DataFrame,
    metrics: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_a = (df["split_class"] == "A").astype(int).to_numpy()
    ac_mask = df["split_class"].isin(["A", "C"])
    df_ac = df[ac_mask]
    y_c_not_a = (df_ac["split_class"] == "C").astype(int).to_numpy()

    for metric in metrics:
        if metric not in df.columns:
            continue
        scores = df[metric].to_numpy(dtype=float)
        auc_a, dir_a = _best_auroc(scores, y_a)
        scores_ac = df_ac[metric].to_numpy(dtype=float)
        auc_ca, dir_ca = _best_auroc(scores_ac, y_c_not_a)
        rows.append(
            {
                "metric": metric,
                "auroc_A": auc_a,
                "auroc_A_direction": dir_a,
                "auroc_C_vs_A": auc_ca,
                "auroc_C_vs_A_direction": dir_ca,
                "is_tautological": metric in TAUTOLOGICAL,
                "is_structural": metric in STRUCTURAL,
                "is_principled_geometry": metric not in TAUTOLOGICAL
                and metric not in STRUCTURAL,
            }
        )
    rows.sort(
        key=lambda r: (
            -r["auroc_C_vs_A"] if np.isfinite(r["auroc_C_vs_A"]) else 0.0,
            -r["auroc_A"] if np.isfinite(r["auroc_A"]) else 0.0,
        )
    )
    return rows


def _class_distributions(df: pd.DataFrame, metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in ("A", "B", "C", "D"):
        sub = df[df["split_class"] == cls]
        feats: dict[str, Any] = {}
        for m in metrics:
            if m in sub.columns:
                feats[m] = _quantiles(sub[m].to_numpy())
        out[cls] = {"n": int(len(sub)), "metrics": feats}
    return out


def _depth_parent_analysis(df: pd.DataFrame) -> dict[str, Any]:
    ac = df[df["split_class"].isin(["A", "C"])].copy()
    if len(ac) == 0:
        return {
            "n_A_vs_C": 0,
            "parent_n_auroc_C_vs_A": float("nan"),
            "split_depth_auroc_C_vs_A": float("nan"),
            "parent_n_auroc_A": float("nan"),
            "split_depth_auroc_A": float("nan"),
        }
    y_ca = (ac["split_class"] == "C").astype(int).to_numpy()
    y_a = (df["split_class"] == "A").astype(int).to_numpy()
    pn_ca, _ = _best_auroc(ac["parent_n"].to_numpy(), y_ca)
    sd_ca, _ = _best_auroc(ac["split_depth"].to_numpy(), y_ca)
    pn_a, pn_dir = _best_auroc(df["parent_n"].to_numpy(), y_a)
    sd_a, sd_dir = _best_auroc(df["split_depth"].to_numpy(), y_a)
    return {
        "n_A_vs_C": int(len(ac)),
        "parent_n_auroc_C_vs_A": pn_ca,
        "split_depth_auroc_C_vs_A": sd_ca,
        "parent_n_auroc_A": pn_a,
        "parent_n_direction_A": pn_dir,
        "split_depth_auroc_A": sd_a,
        "split_depth_direction_A": sd_dir,
        "depth_or_parent_n_explains_C_vs_A": bool(
            (np.isfinite(pn_ca) and pn_ca >= AUROC_GATE)
            or (np.isfinite(sd_ca) and sd_ca >= AUROC_GATE)
        ),
    }


def determine_verdict(
    metric_rows: list[dict[str, Any]],
    depth_parent: dict[str, Any],
) -> tuple[int, str]:
    principled = [r for r in metric_rows if r["is_principled_geometry"]]
    taut = [r for r in metric_rows if r["is_tautological"]]

    best_principled_ca = max(
        (r["auroc_C_vs_A"] for r in principled if np.isfinite(r["auroc_C_vs_A"])),
        default=float("nan"),
    )
    best_principled_metric = next(
        (
            r["metric"]
            for r in principled
            if np.isfinite(r["auroc_C_vs_A"])
            and r["auroc_C_vs_A"] == best_principled_ca
        ),
        "none",
    )
    best_taut_ca = max(
        (r["auroc_C_vs_A"] for r in taut if np.isfinite(r["auroc_C_vs_A"])),
        default=float("nan"),
    )

    pn_ca = depth_parent.get("parent_n_auroc_C_vs_A", float("nan"))
    sd_ca = depth_parent.get("split_depth_auroc_C_vs_A", float("nan"))
    depth_explains = bool(
        (np.isfinite(pn_ca) and pn_ca >= AUROC_GATE)
        or (np.isfinite(sd_ca) and sd_ca >= AUROC_GATE)
    )

    if np.isfinite(best_principled_ca) and best_principled_ca >= AUROC_GATE:
        code = 1
        detail = (
            f"Principled geometry metric `{best_principled_metric}` reaches "
            f"AUROC={best_principled_ca:.3f} for C vs A (gate {AUROC_GATE}). "
            f"parent_n AUROC={pn_ca:.3f}, split_depth AUROC={sd_ca:.3f} "
            "(structural covariates, not required for separation)."
        )
    elif depth_explains:
        code = 3
        detail = (
            f"parent_n AUROC(C vs A)={pn_ca:.3f}, split_depth AUROC(C vs A)="
            f"{sd_ca:.3f} reach gate {AUROC_GATE}, but best principled geometry "
            f"metric `{best_principled_metric}` only achieves "
            f"{best_principled_ca:.3f}. Harmful peels track late recursion / "
            "small parent balls more than split-time geometry."
        )
    elif (
        np.isfinite(best_taut_ca)
        and best_taut_ca >= AUROC_GATE
        and (not np.isfinite(best_principled_ca) or best_principled_ca < AUROC_GATE)
    ):
        code = 2
        detail = (
            f"Tautological/deletion-adjacent metrics reach C vs A AUROC up to "
            f"{best_taut_ca:.3f}, but best principled geometry "
            f"`{best_principled_metric}` only {best_principled_ca:.3f}. "
            "Harmful peels align with del_ball filtering, not geometry alone."
        )
    else:
        code = 4
        detail = (
            f"No principled geometry metric clears AUROC {AUROC_GATE} for C vs A. "
            f"Best `{best_principled_metric}`={best_principled_ca:.3f}; "
            f"parent_n={pn_ca:.3f}, split_depth={sd_ca:.3f}."
        )

    return code, detail


def summarize(df: pd.DataFrame, runs: pd.DataFrame) -> dict[str, Any]:
    df = _add_composites(df)
    all_metrics = METRICS + [name for name, _ in COMPOSITE_SPECS]
    metric_rows = _metric_auroc_table(df, all_metrics)
    distributions = _class_distributions(df, all_metrics)
    depth_parent = _depth_parent_analysis(df)

    counts = df["split_class"].value_counts().to_dict() if len(df) else {}
    n = int(len(df))
    n_runs = int(len(runs)) if len(runs) else (
        int(df.groupby(["dataset", "seed"]).ngroups) if len(df) else 0
    )

    nontaut_ca = [
        r
        for r in metric_rows
        if r["is_principled_geometry"] and np.isfinite(r["auroc_C_vs_A"])
    ]
    best_principled = max(nontaut_ca, key=lambda r: r["auroc_C_vs_A"], default=None)

    verdict_code, verdict_detail = determine_verdict(metric_rows, depth_parent)

    composite_rows = [r for r in metric_rows if r["metric"].startswith("composite_")]

    return {
        "n_splits": n,
        "n_runs": n_runs,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_frac": {k: float(v / n) for k, v in counts.items()} if n else {},
        "auroc_gate": AUROC_GATE,
        "label_definitions": {
            "A": "small child contains global minority and is fully deleted by del_ball",
            "B": "singleton split whose small child is not a minority peel",
            "C": "both children survive del_ball",
            "D": "other",
        },
        "distributions_by_class": distributions,
        "metric_auroc_ranked": metric_rows,
        "composite_scores": composite_rows,
        "depth_parent_analysis": depth_parent,
        "best_principled_geometry_C_vs_A": best_principled,
        "main_question": {
            "question": (
                "Does a non-tautological split-quality statistic separate "
                "harmful minority peels (A) from useful retained splits (C) "
                f"with AUROC >= {AUROC_GATE}?"
            ),
            "answer": bool(
                best_principled
                and np.isfinite(best_principled["auroc_C_vs_A"])
                and best_principled["auroc_C_vs_A"] >= AUROC_GATE
            ),
            "best_metric": best_principled["metric"] if best_principled else None,
            "best_auroc": best_principled["auroc_C_vs_A"]
            if best_principled
            else float("nan"),
            "note": (
                "Excludes tautological/deletion-adjacent and structural "
                "(parent_n, split_depth) metrics."
            ),
        },
        "verdict_code": verdict_code,
        "verdict": VERDICT_LABELS[verdict_code],
        "verdict_detail": verdict_detail,
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if not np.isfinite(x):
            return "nan"
        return f"{x:.3f}"
    return str(x)


def write_report(
    path: Path,
    summary: dict[str, Any],
    runs: pd.DataFrame,
    failed: list[dict[str, str]],
    df: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Split utility diagnostic",
        "",
        "Read-only replay of frozen recursive KMeans(k=2) splits "
        f"(purity={PURITY}, min_samples_per_ball={MIN_SAMPLES_PER_BALL}, "
        f"del_ball num_data={MIN_BALL_COUNT}).",
        "",
        "Split geometry and del_ball fate only — **no Topo / CW / GBLSTSVM**.",
        "",
        "## Headline",
        "",
        f"- Splits: {summary['n_splits']} across {summary['n_runs']} runs",
        f"- Class counts: {summary['class_counts']}",
        f"- AUROC gate: {summary['auroc_gate']}",
        "",
        "## Main question",
        "",
        summary["main_question"]["question"],
        "",
        f"**Answer:** {'YES' if summary['main_question']['answer'] else 'NO'} "
        f"(best non-tautological: `{summary['main_question']['best_metric']}`, "
        f"AUROC(C vs A)={_fmt(summary['main_question']['best_auroc'])}).",
        "",
        "## Verdict",
        "",
        f"**{summary['verdict_code']}. {summary['verdict']}**",
        "",
        summary["verdict_detail"],
        "",
        "## Depth / parent size",
        "",
    ]
    dp = summary["depth_parent_analysis"]
    lines.extend(
        [
            f"- A vs C pairs: {dp['n_A_vs_C']}",
            f"- parent_n AUROC(C vs A): {_fmt(dp['parent_n_auroc_C_vs_A'])}",
            f"- split_depth AUROC(C vs A): {_fmt(dp['split_depth_auroc_C_vs_A'])}",
            f"- Depth/parent_n alone explains C vs A (>= gate): {dp['depth_or_parent_n_explains_C_vs_A']}",
            "",
            "## Composite scores (pre-specified, no threshold tuning)",
            "",
            "| composite | AUROC A | AUROC C vs A | tautological |",
            "|---|---:|---:|---|",
        ]
    )
    for r in summary["composite_scores"]:
        lines.append(
            f"| {r['metric']} | {_fmt(r['auroc_A'])} | {_fmt(r['auroc_C_vs_A'])} | "
            f"{r['is_tautological']} |"
        )

    lines.extend(
        [
            "",
            "## Metrics ranked by AUROC(C vs A)",
            "",
            "| metric | AUROC A | AUROC C vs A | tautological | dir C vs A |",
            "|---|---:|---:|---|---|",
        ]
    )
    for r in summary["metric_auroc_ranked"][:25]:
        lines.append(
            f"| {r['metric']} | {_fmt(r['auroc_A'])} | {_fmt(r['auroc_C_vs_A'])} | "
            f"{r['is_tautological']} | {r['auroc_C_vs_A_direction']} |"
        )

    lines.extend(["", "## Distributions by class (p10 / p50 / p90)", ""])
    for cls in ("A", "B", "C", "D"):
        pack = summary["distributions_by_class"][cls]
        lines.append(f"### Class {cls} (n={pack['n']})")
        if pack["n"] == 0:
            lines.append("")
            continue
        for metric in (
            "size_ratio",
            "radius_reduction",
            "center_radius_separation",
            "minority_margin_mean",
            "parent_n",
            "split_depth",
            "child_compactness_gain",
        ):
            q = pack["metrics"].get(metric, {})
            if q.get("n"):
                lines.append(
                    f"- {metric}: p10={_fmt(q['p10'])} p50={_fmt(q['p50'])} "
                    f"p90={_fmt(q['p90'])}"
                )
        lines.append("")

    lines.extend(
        [
            "## Per-run counts",
            "",
            "| dataset | seed | splits | A | B | C | D |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if len(runs) and "dataset" in runs.columns:
        run_iter = runs.sort_values(["dataset", "seed"]).iterrows()
        for _, r in run_iter:
            lines.append(
                f"| {r['dataset']} | {int(r['seed'])} | {int(r['n_splits'])} | "
                f"{int(r['n_A'])} | {int(r['n_B'])} | {int(r['n_C'])} | {int(r['n_D'])} |"
            )
    elif df is not None and len(df):
        for (ds, seed), g in df.groupby(["dataset", "seed"]):
            vc = g["split_class"].value_counts()
            lines.append(
                f"| {ds} | {int(seed)} | {len(g)} | "
                f"{int(vc.get('A', 0))} | {int(vc.get('B', 0))} | "
                f"{int(vc.get('C', 0))} | {int(vc.get('D', 0))} |"
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
        runs_path = OUT_DIR / "split_utility_runs.csv"
        runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
        summary = summarize(df, runs)
        proto = {}
        if OUT_SUMMARY.exists():
            proto = json.loads(OUT_SUMMARY.read_text(encoding="utf-8")).get("protocol", {})
        summary["protocol"] = proto
        failed = proto.get("failed", [])
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, summary, runs, failed, df=df)
        print(f"Verdict: {summary['verdict_code']}. {summary['verdict']}")
        print(summary["verdict_detail"])
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
                f"A={run['n_A']} B={run['n_B']} C={run['n_C']} D={run['n_D']}"
            )

    df = pd.DataFrame(all_rows)
    runs = pd.DataFrame(run_rows)
    df = _add_composites(df)
    df.to_csv(OUT_CSV, index=False)
    runs.to_csv(OUT_DIR / "split_utility_runs.csv", index=False)

    summary = summarize(df, runs)
    summary["protocol"] = {
        "purity": PURITY,
        "min_samples_per_ball": MIN_SAMPLES_PER_BALL,
        "min_ball_count": MIN_BALL_COUNT,
        "seeds": SEEDS,
        "datasets": [d["name"] for d in descriptors],
        "failed": failed,
        "weighted_child_radius": "(n1*r1 + n2*r2)/(n1+n2)",
        "radius_reduction": "parent_radius - weighted_child_radius",
        "center_radius_separation": "center_distance / (r1+r2+eps)",
        "child_compactness_gain": "1 - weighted_child_radius/parent_radius",
        "composite_specs": {k: v for k, v in COMPOSITE_SPECS},
        "tautological_metrics": sorted(TAUTOLOGICAL),
        "structural_metrics": sorted(STRUCTURAL),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, summary, runs, failed, df=df)

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print(f"\nVerdict: {summary['verdict_code']}. {summary['verdict']}")
    print(summary["verdict_detail"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
