"""
Minority isolation granulation diagnostic (read-only).

Replays frozen recursive KMeans(k=2) granulation (rho=0.95) and traces every
split, tracking global-minority samples through the tree, del_ball, and
final Topo n_opp / R_opp.

Does NOT modify granulation, del_ball, Topo, CW, or GBLSTSVM.

Run from GBLSTSVM root:
  python experiments/analyze_minority_isolation_diagnostic.py
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
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.granular_balls import BallInfo, GBList, GranularBall  # noqa: E402
from src.topology_membership import ball_topo_weights  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granulation"
TOPO_DIR = Path(__file__).resolve().parent / "output" / "topology"
SCREENING_CSV = TOPO_DIR / "topo_broad_screening_results.csv"
MECH_CSV = TOPO_DIR / "r_opp_mechanism_analysis.csv"

OUT_SPLIT_CSV = OUT_DIR / "minority_isolation_splits.csv"
OUT_SAMPLE_CSV = OUT_DIR / "minority_isolation_samples.csv"
OUT_LEAF_CSV = OUT_DIR / "minority_isolation_leaves.csv"
OUT_RUN_CSV = OUT_DIR / "minority_isolation_runs.csv"
OUT_SUMMARY_JSON = OUT_DIR / "minority_isolation_summary.json"
OUT_REPORT_MD = OUT_DIR / "minority_isolation_report.md"

MIN_SAMPLES_PER_BALL = 1
SMALL_BALL_MAX = 5
HD_EPS = 1e-12

# Pre-specified (not outcome-tuned) confirmation gates.
SURVIVAL_GAP_GATE = 0.02
ISOLATION_DEL_BALL_GATE = 0.50
CORR_GATE = 0.15


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


def _count_label(labels: np.ndarray, lab: float) -> int:
    return int(np.sum(labels == lab))


def _minority_hd_stats(
    features: np.ndarray,
    labels: np.ndarray,
    minority_label: float,
    radius: float,
) -> dict[str, float]:
    """hd1 / local opposite-class support for a designated minority label."""
    min_mask = labels == minority_label
    opp_mask = ~min_mask
    n_min = int(np.sum(min_mask))
    n_opp = int(np.sum(opp_mask))
    empty = {
        "minority_hd1_median": float("nan"),
        "minority_hd1_min": float("nan"),
        "minority_hd1_norm": float("nan"),
        "mean_n_opp_within_hd1": float("nan"),
        "mean_n_opp_within_2hd1": float("nan"),
        "frac_min_with_opp_support_ge2": float("nan"),
        "n_min": n_min,
        "n_opp_parent": n_opp,
    }
    if n_min == 0 or n_opp == 0:
        return empty

    min_pts = features[min_mask]
    opp_pts = features[opp_mask]
    dists = np.linalg.norm(min_pts[:, None, :] - opp_pts[None, :, :], axis=2)
    hd1_each = np.min(dists, axis=1)
    n_within_hd1 = np.sum(dists <= hd1_each[:, None] + 1e-12, axis=1)
    n_within_2hd1 = np.sum(dists <= (2.0 * hd1_each[:, None] + 1e-12), axis=1)
    med = float(np.median(hd1_each))
    scale = max(float(radius), HD_EPS)
    return {
        "minority_hd1_median": med,
        "minority_hd1_min": float(np.min(hd1_each)),
        "minority_hd1_norm": med / scale,
        "mean_n_opp_within_hd1": float(np.mean(n_within_hd1)),
        "mean_n_opp_within_2hd1": float(np.mean(n_within_2hd1)),
        "frac_min_with_opp_support_ge2": float(np.mean(n_within_2hd1 >= 2)),
        "n_min": n_min,
        "n_opp_parent": n_opp,
    }


def _child_row(
    child: GranularBall,
    *,
    global_min_label: float,
    parent_min_count: int,
) -> dict[str, Any]:
    labels = _labels(child)
    n_min = _count_label(labels, global_min_label)
    n = int(child.num)
    is_small = n <= SMALL_BALL_MAX
    is_singleton = n == 1
    would_leaf = (float(child.purity) >= PURITY) or (n <= MIN_SAMPLES_PER_BALL)
    would_del_if_leaf = would_leaf and n < MIN_BALL_COUNT
    return {
        "num": n,
        "purity": float(child.purity),
        "label": float(child.label),
        "radius": float(child.r),
        "n_global_min": n_min,
        "is_small": is_small,
        "is_singleton": is_singleton,
        "is_pure": float(child.purity) >= 1.0 - 1e-12,
        "is_min_label": float(child.label) == global_min_label,
        "would_leaf": would_leaf,
        "would_del_if_leaf": would_del_if_leaf,
        "took_all_parent_min": parent_min_count > 0 and n_min == parent_min_count,
        "indices": _indices(child).tolist(),
    }


def _region_flags(c1: dict[str, Any], c2: dict[str, Any], parent_n_min: int) -> dict[str, float]:
    """Classify a KMeans split relative to global-minority structure."""
    children = [c1, c2]
    isolating = False
    for ch, sib in ((c1, c2), (c2, c1)):
        if (
            parent_n_min > 0
            and ch["n_global_min"] > 0
            and ch["is_small"]
            and ch["is_pure"]
            and ch["took_all_parent_min"]
            and sib["n_global_min"] == 0
        ):
            isolating = True
            break

    both_have_min = c1["n_global_min"] > 0 and c2["n_global_min"] > 0
    neither_has_min = c1["n_global_min"] == 0 and c2["n_global_min"] == 0
    singleton_min = any(ch["is_singleton"] and ch["n_global_min"] > 0 for ch in children)
    small_min = any(ch["is_small"] and ch["n_global_min"] > 0 for ch in children)

    if isolating:
        region = "minority_isolating"
    elif both_have_min:
        region = "minority_preserving"
    elif neither_has_min:
        region = "majority_only"
    else:
        region = "minority_split_uneven"

    return {
        "region": region,
        "minority_isolated_small": float(isolating),
        "minority_preserving": float(both_have_min),
        "majority_only_split": float(neither_has_min),
        "singleton_minority_child": float(singleton_min),
        "small_minority_child": float(small_min),
    }


def _to_ball_info(ball: GranularBall) -> BallInfo:
    label = -1.0 if ball.label == -1 else 1.0
    return BallInfo(
        center=ball.center.copy(),
        radius=float(ball.r),
        label=label,
        purity=float(ball.purity),
        num_samples=int(ball.num),
    )


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    v = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(v) < 3:
        return float("nan")
    return float(v["x"].corr(v["y"]))


def _paired_mean_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return {"n": 0, "mean_diff": float("nan"), "frac_a_lt_b": float("nan")}
    diff = a - b
    return {
        "n": int(a.size),
        "mean_diff": float(np.mean(diff)),
        "frac_a_lt_b": float(np.mean(diff < -1e-15)),
        "frac_a_gt_b": float(np.mean(diff > 1e-15)),
    }


def trace_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    dataset: str,
    global_min_label: float,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    np.random.seed(seed)
    indexed = _build_indexed_matrix(X_train, y_train)
    y_all = indexed[:, -2]
    n_train = indexed.shape[0]
    n_min_train = int(np.sum(y_all == global_min_label))
    n_maj_train = n_train - n_min_train

    gb = GBList(indexed)
    balls = gb.granular_balls
    split_rows: list[dict] = []
    split_id = 0

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < PURITY and ball.num > MIN_SAMPLES_PER_BALL:
            parent_labels = _labels(ball)
            parent_feats = ball.data_no_label.copy()
            parent_n_min = _count_label(parent_labels, global_min_label)
            # Local minority = non-majority class in the parent.
            local_min_label = 1.0 if float(ball.label) == -1.0 else -1.0
            hd_global = _minority_hd_stats(
                parent_feats, parent_labels, global_min_label, float(ball.r)
            )
            hd_local = _minority_hd_stats(
                parent_feats, parent_labels, local_min_label, float(ball.r)
            )

            c1, c2 = ball.split_2balls()
            split_id += 1
            r1 = _child_row(c1, global_min_label=global_min_label, parent_min_count=parent_n_min)
            r2 = _child_row(c2, global_min_label=global_min_label, parent_min_count=parent_n_min)
            flags = _region_flags(r1, r2, parent_n_min)
            max_child_p = max(r1["purity"], r2["purity"])
            gain = max_child_p - float(ball.purity)

            split_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split_id": split_id,
                    "parent_num": int(ball.num),
                    "parent_purity": float(ball.purity),
                    "parent_label": float(ball.label),
                    "parent_radius": float(ball.r),
                    "parent_n_global_min": parent_n_min,
                    "parent_n_local_min": _count_label(parent_labels, local_min_label),
                    "parent_local_min_label": local_min_label,
                    "parent_local_min_is_global_min": float(
                        local_min_label == global_min_label
                    ),
                    "child1_num": r1["num"],
                    "child2_num": r2["num"],
                    "child1_n_global_min": r1["n_global_min"],
                    "child2_n_global_min": r2["n_global_min"],
                    "child1_purity": r1["purity"],
                    "child2_purity": r2["purity"],
                    "child1_label": r1["label"],
                    "child2_label": r2["label"],
                    "child1_is_small": r1["is_small"],
                    "child2_is_small": r2["is_small"],
                    "child1_is_singleton": r1["is_singleton"],
                    "child2_is_singleton": r2["is_singleton"],
                    "child1_would_del_if_leaf": r1["would_del_if_leaf"],
                    "child2_would_del_if_leaf": r2["would_del_if_leaf"],
                    "child1_would_leaf": r1["would_leaf"],
                    "child2_would_leaf": r2["would_leaf"],
                    "kmeans_purity_gain": gain,
                    "kmeans_max_child_purity": max_child_p,
                    **{f"global_{k}": v for k, v in hd_global.items() if k not in ("n_min",)},
                    **{f"local_{k}": v for k, v in hd_local.items() if k not in ("n_min",)},
                    **flags,
                    "child1_indices": ",".join(map(str, r1["indices"])),
                    "child2_indices": ",".join(map(str, r2["indices"])),
                }
            )
            balls[i] = c1
            balls.append(c2)
            ll += 1
        else:
            i += 1

    # Leaf mapping before del_ball.
    sample_leaf: dict[int, int] = {}
    leaf_rows: list[dict] = []
    for leaf_id, ball in enumerate(balls):
        idx = _indices(ball)
        labels = _labels(ball)
        n_min = _count_label(labels, global_min_label)
        n = int(ball.num)
        would_del = n < MIN_BALL_COUNT
        for s in idx:
            sample_leaf[int(s)] = leaf_id
        leaf_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "leaf_id": leaf_id,
                "num": n,
                "purity": float(ball.purity),
                "label": float(-1.0 if ball.label == -1 else 1.0),
                "radius": float(ball.r),
                "n_global_min": n_min,
                "is_small": n <= SMALL_BALL_MAX,
                "is_singleton": n == 1,
                "would_del_ball": would_del,
                "is_min_label": float((-1.0 if ball.label == -1 else 1.0) == global_min_label),
            }
        )

    deleted_leaf_ids = {r["leaf_id"] for r in leaf_rows if r["would_del_ball"]}

    # Annotate splits with whether child sample mass is later deleted.
    for row in split_rows:
        for which in ("1", "2"):
            idxs = [
                int(x)
                for x in row[f"child{which}_indices"].split(",")
                if x != ""
            ]
            if not idxs:
                row[f"child{which}_frac_later_del_ball"] = float("nan")
                row[f"child{which}_all_later_del_ball"] = float("nan")
                continue
            n_del = sum(1 for s in idxs if sample_leaf.get(s) in deleted_leaf_ids)
            row[f"child{which}_frac_later_del_ball"] = n_del / len(idxs)
            row[f"child{which}_all_later_del_ball"] = float(n_del == len(idxs))
        row["isolating_child_later_del_ball"] = float("nan")
        if row["minority_isolated_small"] == 1.0:
            later = []
            for which in ("1", "2"):
                if row[f"child{which}_n_global_min"] > 0 and row[f"child{which}_is_small"]:
                    later.append(row[f"child{which}_all_later_del_ball"])
            if later:
                row["isolating_child_later_del_ball"] = float(np.nanmean(later))

    # Sample survival.
    sample_rows: list[dict] = []
    survived_min = 0
    survived_maj = 0
    lost_min_small = 0
    min_in_small_before_del = 0
    for s in range(n_train):
        y_s = float(y_all[s])
        is_min = y_s == global_min_label
        leaf_id = sample_leaf[s]
        leaf = leaf_rows[leaf_id]
        deleted = leaf["would_del_ball"]
        if is_min:
            if leaf["is_small"]:
                min_in_small_before_del += 1
            if deleted:
                lost_min_small += int(leaf["is_small"])
            else:
                survived_min += 1
        else:
            if not deleted:
                survived_maj += 1
        sample_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "sample_index": s,
                "label": y_s,
                "is_global_min": is_min,
                "leaf_id": leaf_id,
                "leaf_num": leaf["num"],
                "leaf_purity": leaf["purity"],
                "leaf_is_small": leaf["is_small"],
                "leaf_is_singleton": leaf["is_singleton"],
                "deleted_by_del_ball": deleted,
            }
        )

    # Apply del_ball via existing API (no src edits).
    gb.del_ball(num_data=MIN_BALL_COUNT)
    kept_balls = [_to_ball_info(b) for b in gb.granular_balls]

    topo_stats = {
        "n_balls_before_del": len(leaf_rows),
        "n_balls_after_del": len(kept_balls),
        "n_deleted_balls": int(sum(r["would_del_ball"] for r in leaf_rows)),
        "mean_n_opp": float("nan"),
        "mean_R_opp": float("nan"),
        "frac_R_opp_eq_0": float("nan"),
        "mean_n_opp_min_balls": float("nan"),
        "mean_R_opp_min_balls": float("nan"),
        "frac_R_opp_eq_0_min_balls": float("nan"),
        "mean_n_opp_maj_balls": float("nan"),
        "mean_R_opp_maj_balls": float("nan"),
        "n_min_label_balls": 0,
        "n_maj_label_balls": 0,
    }
    if kept_balls:
        try:
            kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_all)
            _, _, topo_mem, _, _ = ball_topo_weights(kept_balls, kappa_pos, kappa_neg)
            labels_b = np.array([b.label for b in kept_balls], dtype=float)
            min_m = labels_b == global_min_label
            maj_m = ~min_m
            topo_stats.update(
                {
                    "mean_n_opp": float(np.mean(topo_mem.n_opp)),
                    "mean_R_opp": float(np.mean(topo_mem.r_opp)),
                    "frac_R_opp_eq_0": float(np.mean(topo_mem.r_opp == 0)),
                    "n_min_label_balls": int(np.sum(min_m)),
                    "n_maj_label_balls": int(np.sum(maj_m)),
                }
            )
            if np.any(min_m):
                topo_stats["mean_n_opp_min_balls"] = float(np.mean(topo_mem.n_opp[min_m]))
                topo_stats["mean_R_opp_min_balls"] = float(np.mean(topo_mem.r_opp[min_m]))
                topo_stats["frac_R_opp_eq_0_min_balls"] = float(
                    np.mean(topo_mem.r_opp[min_m] == 0)
                )
            if np.any(maj_m):
                topo_stats["mean_n_opp_maj_balls"] = float(np.mean(topo_mem.n_opp[maj_m]))
                topo_stats["mean_R_opp_maj_balls"] = float(np.mean(topo_mem.r_opp[maj_m]))
        except ValueError:
            # GBLSTSVM requires at least one ball per class after del_ball.
            topo_stats["topo_error"] = "missing_class_after_del_ball"

    def _mean_flag(key: str, subset: list[dict] | None = None) -> float:
        rows = split_rows if subset is None else subset
        if not rows:
            return float("nan")
        return float(np.mean([r[key] for r in rows]))

    isolating = [r for r in split_rows if r["minority_isolated_small"] == 1.0]
    preserving = [r for r in split_rows if r["minority_preserving"] == 1.0]
    majority_only = [r for r in split_rows if r["majority_only_split"] == 1.0]

    def _mean_key(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    run_row = {
        "dataset": dataset,
        "seed": seed,
        "run_id": f"{dataset}_s{seed}",
        "n_train": n_train,
        "n_min_train": n_min_train,
        "n_maj_train": n_maj_train,
        "global_min_label": global_min_label,
        "n_splits": len(split_rows),
        "n_minority_isolating_splits": len(isolating),
        "n_minority_preserving_splits": len(preserving),
        "n_majority_only_splits": len(majority_only),
        "frac_isolating_splits": _mean_flag("minority_isolated_small"),
        "frac_preserving_splits": _mean_flag("minority_preserving"),
        "frac_singleton_min_child": _mean_flag("singleton_minority_child"),
        "frac_small_min_child": _mean_flag("small_minority_child"),
        "mean_kmeans_purity_gain": _mean_key(split_rows, "kmeans_purity_gain"),
        "mean_kmeans_purity_gain_isolating": _mean_key(isolating, "kmeans_purity_gain"),
        "mean_kmeans_purity_gain_preserving": _mean_key(preserving, "kmeans_purity_gain"),
        "mean_global_hd1_norm": _mean_key(split_rows, "global_minority_hd1_norm"),
        "mean_global_hd1_norm_isolating": _mean_key(isolating, "global_minority_hd1_norm"),
        "mean_global_hd1_norm_preserving": _mean_key(preserving, "global_minority_hd1_norm"),
        "mean_local_opp_support_2hd1": _mean_key(
            split_rows, "global_mean_n_opp_within_2hd1"
        ),
        "mean_local_opp_support_isolating": _mean_key(
            isolating, "global_mean_n_opp_within_2hd1"
        ),
        "mean_local_opp_support_preserving": _mean_key(
            preserving, "global_mean_n_opp_within_2hd1"
        ),
        "frac_isolating_later_del_ball": _mean_key(
            isolating, "isolating_child_later_del_ball"
        ),
        "minority_survival": survived_min / n_min_train if n_min_train else float("nan"),
        "majority_survival": survived_maj / n_maj_train if n_maj_train else float("nan"),
        "survival_gap_min_minus_maj": (
            (survived_min / n_min_train) - (survived_maj / n_maj_train)
            if n_min_train and n_maj_train
            else float("nan")
        ),
        "frac_min_in_small_before_del": (
            min_in_small_before_del / n_min_train if n_min_train else float("nan")
        ),
        "n_min_lost_del_ball": n_min_train - survived_min,
        "n_maj_lost_del_ball": n_maj_train - survived_maj,
        **topo_stats,
    }
    # Drop bulky index fields from split export (kept until now for del_ball mapping).
    export_splits = []
    for row in split_rows:
        out = dict(row)
        out.pop("child1_indices", None)
        out.pop("child2_indices", None)
        export_splits.append(out)

    return export_splits, sample_rows, leaf_rows, run_row


def _build_report(summary: dict) -> str:
    lines = [
        "# Minority isolation granulation diagnostic",
        "",
        "Read-only replay of frozen recursive KMeans(k=2) splits "
        f"(rho={PURITY}, min_samples_per_ball={MIN_SAMPLES_PER_BALL}, "
        f"del_ball num_data={MIN_BALL_COUNT}).",
        "No granulation, Topo, CW, or GBLSTSVM code modified.",
        "",
        "Hypothesis: minority isolation → small-ball formation → del_ball/sample "
        "loss → sparse GB representation → poor classification.",
        "",
        f"Runs: **{summary['n_runs']}** | splits: **{summary['n_splits']}**",
        "",
        "## Mechanism tests (pre-specified)",
        "",
        "| Test | Result |",
        "|---|---|",
        f"| Minority vs majority survival (mean gap min−maj) | "
        f"{summary['survival']['mean_gap']:.4f} "
        f"(frac runs min<maj={summary['survival']['frac_runs_min_lt_maj']:.3f}) |",
        f"| Isolating splits whose minority child is later fully del_ball'd | "
        f"{summary['isolation']['frac_isolating_later_del_ball']:.3f} |",
        f"| Mean isolating-split rate | "
        f"{summary['isolation']['mean_frac_isolating_splits']:.3f} |",
        f"| Mean preserving-split rate | "
        f"{summary['isolation']['mean_frac_preserving_splits']:.3f} |",
        f"| hd1_norm isolating vs preserving (mean) | "
        f"{summary['isolation']['mean_hd1_norm_isolating']:.3f} vs "
        f"{summary['isolation']['mean_hd1_norm_preserving']:.3f} |",
        f"| local opp-support (2·hd1) isolating vs preserving | "
        f"{summary['isolation']['mean_opp_support_isolating']:.3f} vs "
        f"{summary['isolation']['mean_opp_support_preserving']:.3f} |",
        f"| KMeans purity gain isolating vs preserving | "
        f"{summary['isolation']['mean_purity_gain_isolating']:.3f} vs "
        f"{summary['isolation']['mean_purity_gain_preserving']:.3f} |",
        f"| Isolating splits that make a minority singleton | "
        f"{summary['isolation']['frac_isolating_singleton_min_child']:.3f} |",
        f"| r(isolating split rate, minority survival) | "
        f"{summary['isolation']['corr_isolating_vs_min_survival']:.3f} |",
        "",
        "## Correlation with Topo Delta Macro F1 (Pearson r)",
        "",
        "| Signal | r |",
        "|---|---:|",
    ]
    for name, rval in summary["outcome_correlations"].items():
        lines.append(f"| {name} | {rval:.3f} |")

    lines.extend(
        [
            "",
            "## Informative vs adversarial means (R_opp category)",
            "",
            "| Category | n | min survival | maj survival | frac isolating | "
            "frac min in small | mean n_opp min balls | Delta Macro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cat, row in summary["category_means"].items():
        lines.append(
            f"| {cat} | {row['n']} | {row['minority_survival']:.3f} | "
            f"{row['majority_survival']:.3f} | {row['frac_isolating_splits']:.3f} | "
            f"{row['frac_min_in_small_before_del']:.3f} | "
            f"{row['mean_n_opp_min_balls']:.3f} | {row['delta_macro_f1_topo_vs_ca']:.3f} |"
        )

    lines.extend(["", "## Verdict", "", summary["verdict"], ""])
    lines.extend(["## Proposed splitter change (not implemented)", "", summary["proposal"], ""])
    lines.extend(
        [
            "## Outputs",
            "",
            f"- `{OUT_RUN_CSV.name}`",
            f"- `{OUT_SPLIT_CSV.name}`",
            f"- `{OUT_LEAF_CSV.name}`",
            f"- `{OUT_SAMPLE_CSV.name}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SCREENING_CSV.exists():
        print(f"Missing {SCREENING_CSV}")
        return 1

    screening = pd.read_csv(SCREENING_CSV)
    topo_perf = screening[screening["method"] == "Topo"].set_index(["dataset", "seed"])
    descriptors = {d["name"]: d for d in discover_available_datasets()}

    all_splits: list[dict] = []
    all_samples: list[dict] = []
    all_leaves: list[dict] = []
    run_rows: list[dict] = []

    for (dataset, seed), perf_row in topo_perf.iterrows():
        desc = descriptors.get(dataset)
        if desc is None:
            continue
        split, err = try_load_descriptor(desc, seed=int(seed))
        if split is None:
            print(f"skip {dataset} seed={seed}: {err}")
            continue
        y_train = normalize_labels(split.y_train)
        n_pos = int(np.sum(y_train == 1.0))
        n_neg = int(np.sum(y_train == -1.0))
        global_min = 1.0 if n_pos <= n_neg else -1.0
        splits, samples, leaves, run_row = trace_run(
            split.X_train,
            y_train,
            seed=int(seed),
            dataset=split.name,
            global_min_label=global_min,
        )
        run_row["delta_macro_f1_topo_vs_ca"] = perf_row.get("delta_macro_f1_topo_vs_ca")
        run_row["delta_gmean_topo_vs_ca"] = perf_row.get("delta_gmean_topo_vs_ca")
        run_row["delta_minority_f1_topo_vs_ca"] = perf_row.get(
            "delta_minority_f1_topo_vs_ca"
        )
        all_splits.extend(splits)
        all_samples.extend(samples)
        all_leaves.extend(leaves)
        run_rows.append(run_row)
        print(
            f"{split.name} seed={seed}: splits={run_row['n_splits']} "
            f"isolating={run_row['n_minority_isolating_splits']} "
            f"min_surv={run_row['minority_survival']:.3f} "
            f"maj_surv={run_row['majority_survival']:.3f} "
            f"del_balls={run_row['n_deleted_balls']}"
        )

    split_df = pd.DataFrame(all_splits)
    sample_df = pd.DataFrame(all_samples)
    leaf_df = pd.DataFrame(all_leaves)
    run_df = pd.DataFrame(run_rows)

    if MECH_CSV.exists():
        mech = pd.read_csv(MECH_CSV)[["dataset", "seed", "r_opp_category"]]
        run_df = run_df.merge(mech, on=["dataset", "seed"], how="left")

    split_df.to_csv(OUT_SPLIT_CSV, index=False)
    sample_df.to_csv(OUT_SAMPLE_CSV, index=False)
    leaf_df.to_csv(OUT_LEAF_CSV, index=False)
    run_df.to_csv(OUT_RUN_CSV, index=False)

    surv = _paired_mean_diff(
        run_df["minority_survival"].to_numpy(dtype=float),
        run_df["majority_survival"].to_numpy(dtype=float),
    )
    survival = {
        "mean_min": float(run_df["minority_survival"].mean()),
        "mean_maj": float(run_df["majority_survival"].mean()),
        "mean_gap": float(run_df["survival_gap_min_minus_maj"].mean()),
        "frac_runs_min_lt_maj": surv["frac_a_lt_b"],
        "paired": surv,
    }
    iso_mask = split_df["minority_isolated_small"] == 1.0
    pres_mask = split_df["minority_preserving"] == 1.0
    isolation = {
        "mean_frac_isolating_splits": float(run_df["frac_isolating_splits"].mean()),
        "mean_frac_preserving_splits": float(run_df["frac_preserving_splits"].mean()),
        "n_isolating_splits": int(iso_mask.sum()) if len(split_df) else 0,
        "n_preserving_splits": int(pres_mask.sum()) if len(split_df) else 0,
        "frac_isolating_later_del_ball": float(
            run_df["frac_isolating_later_del_ball"].mean()
        ),
        "frac_isolating_singleton_min_child": float(
            split_df.loc[iso_mask, "singleton_minority_child"].mean()
        )
        if iso_mask.any()
        else float("nan"),
        "mean_purity_gain_isolating": float(
            split_df.loc[iso_mask, "kmeans_purity_gain"].mean()
        )
        if iso_mask.any()
        else float("nan"),
        "mean_purity_gain_preserving": float(
            split_df.loc[pres_mask, "kmeans_purity_gain"].mean()
        )
        if pres_mask.any()
        else float("nan"),
        "mean_hd1_norm_isolating": float(run_df["mean_global_hd1_norm_isolating"].mean()),
        "mean_hd1_norm_preserving": float(
            run_df["mean_global_hd1_norm_preserving"].mean()
        ),
        "mean_opp_support_isolating": float(
            run_df["mean_local_opp_support_isolating"].mean()
        ),
        "mean_opp_support_preserving": float(
            run_df["mean_local_opp_support_preserving"].mean()
        ),
        "mean_frac_min_in_small": float(run_df["frac_min_in_small_before_del"].mean()),
        "corr_isolating_vs_min_survival": _safe_corr(
            run_df["frac_isolating_splits"], run_df["minority_survival"]
        ),
    }

    signal_cols = {
        "survival_gap_min_minus_maj": "minority−majority survival",
        "frac_isolating_splits": "frac isolating splits",
        "frac_min_in_small_before_del": "frac minority in small balls",
        "minority_survival": "minority survival",
        "mean_global_hd1_norm": "mean minority hd1_norm",
        "mean_local_opp_support_2hd1": "mean local opp support (2·hd1)",
        "mean_n_opp": "mean n_opp (all balls)",
        "mean_n_opp_min_balls": "mean n_opp (minority-label balls)",
        "frac_R_opp_eq_0": "frac R_opp=0",
        "n_deleted_balls": "n deleted balls",
        "n_balls_after_del": "n balls after del_ball",
    }
    outcome_correlations = {
        label: _safe_corr(run_df[col], run_df["delta_macro_f1_topo_vs_ca"])
        for col, label in signal_cols.items()
    }

    category_means: dict[str, dict] = {}
    if "r_opp_category" in run_df.columns:
        for cat in sorted(run_df["r_opp_category"].dropna().unique()):
            sub = run_df[run_df["r_opp_category"] == cat]
            category_means[str(cat)] = {
                "n": int(len(sub)),
                "minority_survival": float(sub["minority_survival"].mean()),
                "majority_survival": float(sub["majority_survival"].mean()),
                "frac_isolating_splits": float(sub["frac_isolating_splits"].mean()),
                "frac_min_in_small_before_del": float(
                    sub["frac_min_in_small_before_del"].mean()
                ),
                "mean_n_opp_min_balls": float(sub["mean_n_opp_min_balls"].mean()),
                "delta_macro_f1_topo_vs_ca": float(
                    sub["delta_macro_f1_topo_vs_ca"].mean()
                ),
            }

    # Confirmation logic.
    gap_ok = (
        np.isfinite(survival["mean_gap"])
        and survival["mean_gap"] <= -SURVIVAL_GAP_GATE
        and surv["frac_a_lt_b"] >= 0.6
    )
    del_ok = (
        np.isfinite(isolation["frac_isolating_later_del_ball"])
        and isolation["frac_isolating_later_del_ball"] >= ISOLATION_DEL_BALL_GATE
    )
    iso_present = isolation["mean_frac_isolating_splits"] > 0.02
    r_iso = outcome_correlations["frac isolating splits"]
    sparse_ok = (
        np.isfinite(isolation["mean_hd1_norm_isolating"])
        and np.isfinite(isolation["mean_hd1_norm_preserving"])
        and isolation["mean_hd1_norm_isolating"]
        > isolation["mean_hd1_norm_preserving"]
        and isolation["mean_opp_support_isolating"]
        < isolation["mean_opp_support_preserving"]
    )
    # Classification hop requires the isolation rate itself, not a surrogate.
    classification_hop_ok = np.isfinite(r_iso) and abs(r_iso) >= CORR_GATE

    chain_ok = gap_ok and del_ok and iso_present and sparse_ok
    granulation_confirmed = chain_ok
    confirmed = granulation_confirmed and classification_hop_ok
    outcome_link = classification_hop_ok

    proposal_text = (
        "Smallest change, still KMeans(k=2), still rho=0.95, no downstream edits: "
        "after `split_2balls` assignment, repair only isolating children. "
        "If a child is small (n<=5) or singleton, carries the parent's local minority, "
        "and those points have heterogeneous support in the parent "
        "(n_opp_within_2*hd1 >= 2), reassign them to the sibling cluster. "
        "Leave unrepaired: (i) n_opp_within_2*hd1 == 1 outliers with large hd1_norm, "
        "(ii) mixed preserving splits. If repair would empty a child, skip that "
        "repair and keep the frozen KMeans partition. This keeps legitimate boundary "
        "minority inside a ball that survives del_ball, without changing KMeans(k=2), "
        "purity threshold, or GBLSTSVM."
    )

    if confirmed:
        verdict = (
            "CONFIRMED (full chain): minority isolation -> small/singleton balls -> "
            "del_ball sample loss -> sparse opposite-class support, and isolating-split "
            "rate associates with Topo Delta Macro F1."
        )
        proposal = proposal_text
    elif granulation_confirmed and not classification_hop_ok:
        verdict = (
            "PARTIAL: the granulation chain is reproducible "
            "(isolation -> singleton/small ball -> del_ball minority loss -> sparse "
            "heterogeneous support). The last hop to poor classification is NOT "
            "supported: isolating-split rate does not correlate with Topo Delta Macro F1 "
            "and is essentially the same in adversarial vs informative R_opp runs. "
            "Do not modify the splitter for classifier gain yet. The repair below is "
            "justified only as a future granulation experiment whose success metric "
            "should be minority survival / small-ball rate, not Topo F1."
        )
        proposal = proposal_text + (
            " Implementation is deferred until a granulation-only replay is approved; "
            "do not screen GBLSTSVM yet."
        )
    elif iso_present and not del_ok:
        verdict = (
            "NOT CONFIRMED as a del_ball mechanism: minority-isolating KMeans splits "
            "occur, but isolating children are not consistently removed by del_ball. "
            "Small-ball formation alone does not establish the hypothesized chain."
        )
        proposal = "No splitter change. Isolation without sample loss does not justify a KMeans repair."
    elif gap_ok and not iso_present:
        verdict = (
            "NOT CONFIRMED as a KMeans isolation mechanism: minority samples are lost "
            "relative to majority, but not primarily via isolating KMeans children. "
            "Look at other del_ball paths (two-sample pure leaves, etc.)."
        )
        proposal = "No KMeans-splitter change. The loss path is not the hypothesized isolation split."
    else:
        verdict = (
            "NOT CONFIRMED: minority isolation -> small-ball -> del_ball -> sparse GB "
            "is not a reproducible chain on the frozen screening runs."
        )
        proposal = "No granulation change. Keep KMeans(k=2) and rho as frozen."

    summary = {
        "n_runs": int(len(run_df)),
        "n_splits": int(len(split_df)),
        "n_leaves": int(len(leaf_df)),
        "protocol": {
            "purity": PURITY,
            "min_samples_per_ball": MIN_SAMPLES_PER_BALL,
            "min_ball_count": MIN_BALL_COUNT,
            "small_ball_max": SMALL_BALL_MAX,
            "seeds_source": "topo_broad_screening_results.csv (Topo rows)",
        },
        "gates": {
            "survival_gap_gate": SURVIVAL_GAP_GATE,
            "isolation_del_ball_gate": ISOLATION_DEL_BALL_GATE,
            "corr_gate": CORR_GATE,
            "gap_ok": gap_ok,
            "del_ok": del_ok,
            "iso_present": iso_present,
            "sparse_ok": sparse_ok,
            "classification_hop_ok": classification_hop_ok,
            "granulation_confirmed": granulation_confirmed,
            "outcome_link": outcome_link,
            "confirmed": confirmed,
        },
        "survival": survival,
        "isolation": isolation,
        "outcome_correlations": outcome_correlations,
        "category_means": category_means,
        "verdict": verdict,
        "proposal": proposal,
        "definitions": {
            "global_minority": "rarer class on the training split after normalize_labels",
            "minority_isolating": (
                "one child is small (n<=5), pure, and contains all parent global-minority "
                "samples; sibling has none"
            ),
            "minority_preserving": "both children contain at least one global-minority sample",
            "hd1_norm": "median minority-to-opposite min-distance / parent radius",
            "local_opp_support": "mean # opposite-class parent samples within 2*hd1 of a minority point",
            "survival": "fraction of class samples whose leaf has n >= min_ball_count",
        },
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_REPORT_MD.write_text(_build_report(summary), encoding="utf-8")

    print(f"\nWrote {OUT_RUN_CSV} ({len(run_df)} runs)")
    print(f"Wrote {OUT_SPLIT_CSV} ({len(split_df)} splits)")
    print(f"Wrote {OUT_LEAF_CSV} ({len(leaf_df)} leaves)")
    print(f"Wrote {OUT_SAMPLE_CSV} ({len(sample_df)} samples)")
    print(f"Wrote {OUT_SUMMARY_JSON}")
    print(f"Wrote {OUT_REPORT_MD}")
    print("\nVerdict:", verdict.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
