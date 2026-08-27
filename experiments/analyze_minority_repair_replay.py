"""
Read-only B0 vs hypothetical B1 minority-isolation repair replay.

B0: frozen KMeans(k=2), rho=0.95, del_ball(n<2).
B1: after each split, virtually reassign samples from a qualifying small/singleton
    child (parent-local minority, n_opp_within_2hd1 >= 2) to its sibling.

Does NOT modify src/, granulation, Topo, CW, or GBLSTSVM.

Run from GBLSTSVM root:
  python experiments/analyze_minority_repair_replay.py
"""

from __future__ import annotations

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

from experiments.analyze_cbcm_competition import _per_ball_knn_features  # noqa: E402
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from src.cbcm import resolve_k  # noqa: E402
from src.granular_balls import BallInfo, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granulation"
TOPO_DIR = Path(__file__).resolve().parent / "output" / "topology"
SCREENING_CSV = TOPO_DIR / "topo_broad_screening_results.csv"

OUT_REPAIR_CSV = OUT_DIR / "minority_repair_records.csv"
OUT_RUN_CSV = OUT_DIR / "minority_repair_runs.csv"
OUT_SUMMARY_JSON = OUT_DIR / "minority_repair_replay_summary.json"
OUT_REPORT_MD = OUT_DIR / "minority_repair_replay_report.md"

MIN_SAMPLES_PER_BALL = 1
SMALL_BALL_MAX = 5
HD_EPS = 1e-12

# Pre-specified repair classification (not outcome-tuned).
HD1_NORM_BENEFICIAL_MAX = 2.5
HD1_NORM_HARMFUL_MIN = 3.5
LARGE_SIBLING_MIN = 10
SUPPORTED_OPP_MEAN_MIN = 3.0
SEPARABILITY_HARMFUL_MAX = 0.5


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


def _local_minority_label(ball: GranularBall) -> float:
    return 1.0 if float(ball.label) == -1.0 else -1.0


def _point_parent_stats(
    x: np.ndarray,
    y_x: float,
    parent_feats: np.ndarray,
    parent_labels: np.ndarray,
    parent_radius: float,
) -> tuple[float, float, int]:
    opp_mask = parent_labels != y_x
    if not np.any(opp_mask):
        return float("nan"), float("nan"), 0
    opp_dists = np.linalg.norm(parent_feats[opp_mask] - x, axis=1)
    hd1 = float(np.min(opp_dists))
    n_2hd1 = int(np.sum(opp_dists <= (2.0 * hd1 + 1e-12)))
    scale = max(float(parent_radius), HD_EPS)
    return hd1, hd1 / scale, n_2hd1


def _child_minority_support(
    child: GranularBall,
    local_min_label: float,
    parent_feats: np.ndarray,
    parent_labels: np.ndarray,
    parent_radius: float,
) -> dict[str, Any]:
    child_labels = _labels(child)
    child_feats = child.data_no_label
    min_mask = child_labels == local_min_label
    if not np.any(min_mask):
        return {
            "n_local_min": 0,
            "qualifying_indices": [],
            "hd1_median": float("nan"),
            "hd1_norm_median": float("nan"),
            "n_opp_2hd1_min": 0,
            "n_opp_2hd1_mean": float("nan"),
            "all_min_ge2": False,
        }

    hd1s: list[float] = []
    hd1_norms: list[float] = []
    n2s: list[int] = []
    qualifying: list[int] = []
    for row_idx in np.flatnonzero(min_mask):
        x = child_feats[row_idx]
        y_x = float(child_labels[row_idx])
        hd1, hd1_norm, n2 = _point_parent_stats(
            x, y_x, parent_feats, parent_labels, parent_radius
        )
        hd1s.append(hd1)
        hd1_norms.append(hd1_norm)
        n2s.append(n2)
        if n2 >= 2:
            qualifying.append(int(child.data[row_idx, -1]))

    return {
        "n_local_min": int(np.sum(min_mask)),
        "qualifying_indices": qualifying,
        "hd1_median": float(np.median(hd1s)),
        "hd1_norm_median": float(np.median(hd1_norms)),
        "n_opp_2hd1_min": int(min(n2s)),
        "n_opp_2hd1_mean": float(np.mean(n2s)),
        "all_min_ge2": bool(np.all(np.array(n2s) >= 2)),
    }


def _ball_from_rows(rows: np.ndarray) -> GranularBall | None:
    if rows.size == 0:
        return None
    return GranularBall(rows)


def _merge_child_into_sibling(
    child: GranularBall,
    sibling: GranularBall,
    move_indices: list[int],
) -> tuple[GranularBall | None, GranularBall, bool]:
    """Return (new_child_or_None, new_sibling, child_becomes_empty)."""
    move_set = set(move_indices)
    child_idx = _indices(child)
    move_mask = np.isin(child_idx, list(move_set))
    keep_mask = ~move_mask
    move_rows = child.data[move_mask]
    if move_rows.size == 0:
        return child, sibling, False
    new_sibling_data = np.vstack([sibling.data, move_rows])
    new_sibling = GranularBall(new_sibling_data)
    if not np.any(keep_mask):
        return None, new_sibling, True
    new_child = GranularBall(child.data[keep_mask])
    return new_child, new_sibling, False


def _is_small_or_singleton(ball: GranularBall) -> bool:
    n = int(ball.num)
    return n == 1 or n <= SMALL_BALL_MAX


def _would_del_ball(ball: GranularBall) -> bool:
    return int(ball.num) < MIN_BALL_COUNT


def _sibling_is_heterogeneous(ball: GranularBall) -> bool:
    labels = _labels(ball)
    return _count_label(labels, 1.0) > 0 and _count_label(labels, -1.0) > 0


def _intra_ball_separability(ball: GranularBall) -> float:
    labels = _labels(ball)
    feats = ball.data_no_label
    pos = feats[labels == 1.0]
    neg = feats[labels == -1.0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    mu_pos = np.mean(pos, axis=0)
    mu_neg = np.mean(neg, axis=0)
    inter = float(np.linalg.norm(mu_pos - mu_neg))
    scale = max(float(ball.r), HD_EPS)
    return inter / scale


def _annotate_b0_loss(
    repairs: list[dict[str, Any]],
    *,
    deleted_samples_b0: set[int],
) -> None:
    for rec in repairs:
        moved = [int(x) for x in rec["moved_sample_indices"].split(",") if x]
        n_moved = len(moved)
        n_lost = sum(1 for s in moved if s in deleted_samples_b0)
        rec["b0_moved_samples_lost"] = n_lost
        rec["b0_frac_moved_lost"] = n_lost / n_moved if n_moved else float("nan")
        rec["b0_any_moved_lost"] = n_lost > 0
        rec["b0_all_moved_lost"] = n_lost == n_moved and n_moved > 0
        rec["b0_would_delete_child"] = bool(rec["b0_child_would_del_if_leaf"])
        rec["repair_class"] = _classify_repair(rec)


def _classify_repair(record: dict[str, Any]) -> str:
    """
    A = likely beneficial minority preservation
    B = ambiguous boundary reassignment
    C = likely harmful majority-ball contamination
    """
    b0_saves_samples = bool(record.get("b0_any_moved_lost", False))
    sib_was_pure = record["sibling_purity_before"] >= 1.0 - 1e-12
    sib_large = record["sibling_num_before"] >= LARGE_SIBLING_MIN
    sib_very_large = record["sibling_num_before"] >= 15
    hd1_norm = record["hd1_norm_median"]
    n2_min = record["n_opp_2hd1_min"]
    n2_mean = record["n_opp_2hd1_mean"]
    sep_after = record["sibling_separability_after"]

    boundary_signal = (
        (np.isfinite(hd1_norm) and hd1_norm <= HD1_NORM_BENEFICIAL_MAX)
        or (np.isfinite(n2_mean) and n2_mean >= SUPPORTED_OPP_MEAN_MIN)
    )
    contamination_guard = not (
        sib_very_large and sib_was_pure and np.isfinite(hd1_norm) and hd1_norm > 1.5
    )

    beneficial = (
        b0_saves_samples
        and n2_min >= 2
        and boundary_signal
        and contamination_guard
        and record["n_local_min_moved"] > 0
        and record["sibling_becomes_heterogeneous"]
        and (not np.isfinite(sep_after) or sep_after >= 0.8)
    )

    harmful = (
        (not b0_saves_samples and sib_was_pure and sib_large)
        or (
            np.isfinite(hd1_norm)
            and hd1_norm >= HD1_NORM_HARMFUL_MIN
            and (not np.isfinite(n2_mean) or n2_mean < SUPPORTED_OPP_MEAN_MIN)
        )
        or (
            record["sibling_becomes_heterogeneous"]
            and np.isfinite(sep_after)
            and sep_after < SEPARABILITY_HARMFUL_MAX
            and np.isfinite(hd1_norm)
            and hd1_norm > HD1_NORM_BENEFICIAL_MAX
        )
        or (
            sib_was_pure
            and sib_very_large
            and not b0_saves_samples
        )
    )

    if beneficial and not harmful:
        return "A"
    if harmful and not beneficial:
        return "C"
    return "B"


def _try_repair_child(
    *,
    parent: GranularBall,
    child: GranularBall,
    sibling: GranularBall,
    child_slot: str,
    split_id: int,
    dataset: str,
    seed: int,
    local_min_label: float,
    global_min_label: float,
) -> tuple[GranularBall, GranularBall, dict[str, Any] | None]:
    if not _is_small_or_singleton(child):
        return child, sibling, None

    parent_feats = parent.data_no_label
    parent_labels = _labels(parent)
    support = _child_minority_support(
        child, local_min_label, parent_feats, parent_labels, float(parent.r)
    )
    if support["n_local_min"] == 0 or not support["qualifying_indices"]:
        return child, sibling, None

    move_indices = support["qualifying_indices"]
    new_child, new_sibling, child_empty = _merge_child_into_sibling(
        child, sibling, move_indices
    )
    if child_empty:
        # Frozen rule: skip repair if it would empty a child cluster.
        return child, sibling, None

    max_child_p = max(float(child.purity), float(sibling.purity))
    gain = max_child_p - float(parent.purity)
    sibling_het_after = _sibling_is_heterogeneous(new_sibling)

    record: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "split_id": split_id,
        "child_slot": child_slot,
        "parent_num": int(parent.num),
        "parent_purity": float(parent.purity),
        "parent_label": float(parent.label),
        "parent_radius": float(parent.r),
        "parent_local_min_label": local_min_label,
        "parent_n_local_min": support["n_local_min"],
        "child_num_before": int(child.num),
        "child_label_before": float(child.label),
        "child_purity_before": float(child.purity),
        "sibling_num_before": int(sibling.num),
        "sibling_label_before": float(sibling.label),
        "sibling_purity_before": float(sibling.purity),
        "child_num_after": int(new_child.num) if new_child is not None else 0,
        "child_purity_after": float(new_child.purity) if new_child is not None else float("nan"),
        "sibling_num_after": int(new_sibling.num),
        "sibling_purity_after": float(new_sibling.purity),
        "sibling_label_after": float(new_sibling.label),
        "n_samples_moved": len(move_indices),
        "n_local_min_moved": len(move_indices),
        "minority_samples_affected": len(move_indices),
        "hd1_median": support["hd1_median"],
        "hd1_norm_median": support["hd1_norm_median"],
        "n_opp_2hd1_min": support["n_opp_2hd1_min"],
        "n_opp_2hd1_mean": support["n_opp_2hd1_mean"],
        "kmeans_purity_gain": gain,
        "b0_child_would_del_if_leaf": _would_del_ball(child),
        "sibling_becomes_heterogeneous": sibling_het_after,
        "repaired_child_becomes_empty": False,
        "sibling_separability_after": _intra_ball_separability(new_sibling),
        "child_is_singleton": int(child.num) == 1,
        "child_is_small": int(child.num) <= SMALL_BALL_MAX,
        "local_min_is_global_min": float(local_min_label == global_min_label),
        "moved_sample_indices": ",".join(map(str, move_indices)),
    }
    return new_child, new_sibling, record


def _granulate(
    indexed: np.ndarray,
    *,
    apply_b1_repair: bool,
    dataset: str,
    seed: int,
    global_min_label: float,
) -> tuple[list[GranularBall], list[dict[str, Any]]]:
    balls: list[GranularBall] = [GranularBall(indexed)]
    repairs: list[dict[str, Any]] = []
    split_id = 0
    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < PURITY and ball.num > MIN_SAMPLES_PER_BALL:
            local_min = _local_minority_label(ball)
            c1, c2 = ball.split_2balls()
            split_id += 1
            if apply_b1_repair:
                for slot, child, sibling in (
                    ("first", c1, c2),
                    ("second", c2, c1),
                ):
                    new_child, new_sibling, rec = _try_repair_child(
                        parent=ball,
                        child=child,
                        sibling=sibling,
                        child_slot=slot,
                        split_id=split_id,
                        dataset=dataset,
                        seed=seed,
                        local_min_label=local_min,
                        global_min_label=global_min_label,
                    )
                    if rec is not None:
                        repairs.append(rec)
                        if slot == "first":
                            c1, c2 = new_child, new_sibling
                        else:
                            c2, c1 = new_child, new_sibling
            balls[i] = c1
            balls.append(c2)
            ll += 1
        else:
            i += 1
    return balls, repairs


def _apply_del_ball(balls: list[GranularBall]) -> list[GranularBall]:
    return [b for b in balls if int(b.num) >= MIN_BALL_COUNT]


def _sample_leaf_map(balls: list[GranularBall]) -> dict[int, int]:
    out: dict[int, int] = {}
    for leaf_id, ball in enumerate(balls):
        for s in _indices(ball):
            out[int(s)] = leaf_id
    return out


def _to_ball_info(ball: GranularBall) -> BallInfo:
    label = -1.0 if ball.label == -1 else 1.0
    return BallInfo(
        center=ball.center.copy(),
        radius=float(ball.r),
        label=label,
        purity=float(ball.purity),
        num_samples=int(ball.num),
    )


def _minority_opp_support_in_ball(
    ball: GranularBall,
    global_min_label: float,
) -> float:
    labels = _labels(ball)
    feats = ball.data_no_label
    min_mask = labels == global_min_label
    if not np.any(min_mask) or not np.any(~min_mask):
        return float("nan")
    min_pts = feats[min_mask]
    opp_pts = feats[~min_mask]
    dists = np.linalg.norm(min_pts[:, None, :] - opp_pts[None, :, :], axis=2)
    hd1 = np.min(dists, axis=1)
    n2 = np.sum(dists <= (2.0 * hd1[:, None] + 1e-12), axis=1)
    return float(np.mean(n2))


def _summarize_final_balls(
    balls: list[GranularBall],
    *,
    y_all: np.ndarray,
    global_min_label: float,
    leaf_map: dict[int, int],
) -> dict[str, Any]:
    n_train = len(y_all)
    n_min = int(np.sum(y_all == global_min_label))
    n_maj = n_train - n_min
    global_maj = -global_min_label

    survived_min = 0
    survived_maj = 0
    min_in_small = 0
    min_in_singleton = 0
    min_total = 0
    for s in range(n_train):
        leaf_id = leaf_map.get(s)
        if leaf_id is None:
            continue
        ball = balls[leaf_id]
        y_s = float(y_all[s])
        if y_s == global_min_label:
            min_total += 1
            if int(ball.num) <= SMALL_BALL_MAX:
                min_in_small += 1
            if int(ball.num) == 1:
                min_in_singleton += 1
            survived_min += 1
        else:
            survived_maj += 1

    ball_infos = [_to_ball_info(b) for b in balls]
    labels_b = np.array([b.label for b in ball_infos], dtype=float)
    purities = np.array([b.purity for b in ball_infos], dtype=float)
    radii = np.array([b.radius for b in ball_infos], dtype=float)

    min_ball_support: list[float] = []
    het_separabilities: list[float] = []
    for ball in balls:
        sep = _intra_ball_separability(ball)
        if np.isfinite(sep):
            het_separabilities.append(sep)
        sup = _minority_opp_support_in_ball(ball, global_min_label)
        if np.isfinite(sup):
            min_ball_support.append(sup)

    centers = np.vstack([b.center for b in ball_infos]) if ball_infos else np.zeros((0, 1))
    labels_c = labels_b
    k = resolve_k(len(ball_infos)) if ball_infos else 0
    knn = (
        _per_ball_knn_features(centers, labels_c, k)
        if len(ball_infos) > 1
        else {
            "nearest_same_dist": np.array([]),
            "nearest_opp_dist": np.array([]),
            "opp_same_ratio": np.array([]),
        }
    )

    return {
        "n_balls": len(ball_infos),
        "n_min_label_balls": int(np.sum(labels_b == global_min_label)),
        "n_maj_label_balls": int(np.sum(labels_b == global_maj)),
        "minority_survival": survived_min / n_min if n_min else float("nan"),
        "majority_survival": survived_maj / n_maj if n_maj else float("nan"),
        "survival_gap_min_minus_maj": (
            (survived_min / n_min) - (survived_maj / n_maj)
            if n_min and n_maj
            else float("nan")
        ),
        "frac_min_in_small_balls": min_in_small / min_total if min_total else float("nan"),
        "frac_min_in_singleton_balls": min_in_singleton / min_total if min_total else float("nan"),
        "mean_purity": float(np.mean(purities)) if purities.size else float("nan"),
        "std_purity": float(np.std(purities)) if purities.size else float("nan"),
        "mean_radius": float(np.mean(radii)) if radii.size else float("nan"),
        "std_radius": float(np.std(radii)) if radii.size else float("nan"),
        "mean_min_opp_support_2hd1": float(np.mean(min_ball_support))
        if min_ball_support
        else float("nan"),
        "mean_nearest_opp_dist": float(np.nanmean(knn["nearest_opp_dist"]))
        if knn["nearest_opp_dist"].size
        else float("nan"),
        "mean_nearest_same_dist": float(np.nanmean(knn["nearest_same_dist"]))
        if knn["nearest_same_dist"].size
        else float("nan"),
        "mean_opp_same_ratio": float(np.nanmean(knn["opp_same_ratio"]))
        if knn["opp_same_ratio"].size
        else float("nan"),
        "mean_hetero_ball_separability": float(np.mean(het_separabilities))
        if het_separabilities
        else float("nan"),
        "n_heterogeneous_balls": len(het_separabilities),
    }


def replay_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    dataset: str,
    global_min_label: float,
) -> tuple[list[dict], dict[str, Any]]:
    np.random.seed(seed)
    indexed = _build_indexed_matrix(X_train, y_train)
    y_all = indexed[:, -2]

    balls_b0, _ = _granulate(
        indexed,
        apply_b1_repair=False,
        dataset=dataset,
        seed=seed,
        global_min_label=global_min_label,
    )
    np.random.seed(seed)
    indexed_b1 = _build_indexed_matrix(X_train, y_train)
    balls_b1_pre, repairs = _granulate(
        indexed_b1,
        apply_b1_repair=True,
        dataset=dataset,
        seed=seed,
        global_min_label=global_min_label,
    )

    pre_b0 = _sample_leaf_map(balls_b0)
    pre_b1 = _sample_leaf_map(balls_b1_pre)
    kept_b0 = _apply_del_ball(balls_b0)
    kept_b1 = _apply_del_ball(balls_b1_pre)
    post_b0 = _sample_leaf_map(kept_b0)
    post_b1 = _sample_leaf_map(kept_b1)
    deleted_samples_b0 = {
        s for s in range(len(y_all)) if s not in post_b0
    }
    _annotate_b0_loss(repairs, deleted_samples_b0=deleted_samples_b0)

    n_changed_representation = sum(
        1 for s in range(len(y_all)) if post_b0.get(s) != post_b1.get(s)
    )
    n_changed_pre_del = sum(
        1 for s in range(len(y_all)) if pre_b0.get(s) != pre_b1.get(s)
    )

    sum_b0 = _summarize_final_balls(
        kept_b0, y_all=y_all, global_min_label=global_min_label, leaf_map=post_b0
    )
    sum_b1 = _summarize_final_balls(
        kept_b1, y_all=y_all, global_min_label=global_min_label, leaf_map=post_b1
    )

    repair_counts = Counter(r["repair_class"] for r in repairs) if repairs else {}
    run_row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "run_id": f"{dataset}_s{seed}",
        "n_repairs_proposed": len(repairs),
        "n_repair_A": int(repair_counts.get("A", 0)),
        "n_repair_B": int(repair_counts.get("B", 0)),
        "n_repair_C": int(repair_counts.get("C", 0)),
        "frac_repair_A": float(repair_counts.get("A", 0) / len(repairs))
        if repairs
        else float("nan"),
        "frac_repair_C": float(repair_counts.get("C", 0) / len(repairs))
        if repairs
        else float("nan"),
        "frac_repair_saves_b0_lost": float(
            np.mean([r.get("b0_any_moved_lost", False) for r in repairs])
        )
        if repairs
        else float("nan"),
        "n_samples_representation_changed": n_changed_representation,
        "n_samples_changed_pre_del": n_changed_pre_del,
        "frac_samples_representation_changed": n_changed_representation / len(y_all),
    }
    for prefix, summary in (("b0_", sum_b0), ("b1_", sum_b1)):
        for k, v in summary.items():
            run_row[f"{prefix}{k}"] = v
    for k in sum_b0:
        b0v = sum_b0[k]
        b1v = sum_b1[k]
        if isinstance(b0v, (int, float)) and isinstance(b1v, (int, float)):
            if np.isfinite(b0v) and np.isfinite(b1v):
                run_row[f"delta_{k}"] = b1v - b0v
    return repairs, run_row


def _build_report(summary: dict) -> str:
    agg = summary["aggregate"]
    cls = summary["repair_classification"]
    lines = [
        "# Minority repair replay (B0 vs hypothetical B1)",
        "",
        "Read-only virtual replay. No src/, granulation, Topo, CW, or GBLSTSVM changes.",
        "",
        f"Runs: **{summary['n_runs']}** | repairs proposed: **{summary['n_repairs']}**",
        "",
        "## B0 vs B1 (mean across runs)",
        "",
        "| Metric | B0 | B1 | Delta (B1-B0) |",
        "|---|---:|---:|---:|",
    ]
    for name, row in agg["comparison_table"].items():
        lines.append(
            f"| {name} | {row['b0']:.4f} | {row['b1']:.4f} | {row['delta']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Repair classification",
            "",
            f"- A (likely beneficial): **{cls['A']}** ({cls['frac_A']:.1%})",
            f"- B (ambiguous): **{cls['B']}** ({cls['frac_B']:.1%})",
            f"- C (likely harmful): **{cls['C']}** ({cls['frac_C']:.1%})",
            "",
            "## Geometric defensibility",
            "",
            summary["verdict"],
            "",
            "## Outputs",
            "",
            f"- `{OUT_REPAIR_CSV.name}`",
            f"- `{OUT_RUN_CSV.name}`",
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

    all_repairs: list[dict] = []
    run_rows: list[dict] = []

    for (dataset, seed), _ in topo_perf.iterrows():
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
        repairs, run_row = replay_run(
            split.X_train,
            y_train,
            seed=int(seed),
            dataset=split.name,
            global_min_label=global_min,
        )
        all_repairs.extend(repairs)
        run_rows.append(run_row)
        print(
            f"{split.name} seed={seed}: repairs={len(repairs)} "
            f"A={run_row['n_repair_A']} B={run_row['n_repair_B']} C={run_row['n_repair_C']} "
            f"min_surv {run_row['b0_minority_survival']:.3f}->{run_row['b1_minority_survival']:.3f} "
            f"changed={run_row['n_samples_representation_changed']}"
        )

    repair_df = pd.DataFrame(all_repairs)
    run_df = pd.DataFrame(run_rows)
    repair_df.to_csv(OUT_REPAIR_CSV, index=False)
    run_df.to_csv(OUT_RUN_CSV, index=False)

    metric_keys = [
        ("minority_survival", "1. minority sample survival"),
        ("majority_survival", "2. majority sample survival"),
        ("n_min_label_balls", "3a. minority-label ball count"),
        ("n_maj_label_balls", "3b. majority-label ball count"),
        ("frac_min_in_singleton_balls", "4a. minority singleton-ball rate"),
        ("frac_min_in_small_balls", "4b. minority small-ball rate"),
        ("mean_purity", "5a. mean purity"),
        ("std_purity", "5b. purity std"),
        ("mean_radius", "6a. mean radius"),
        ("std_radius", "6b. radius std"),
        ("mean_min_opp_support_2hd1", "7. minority-to-opposite support"),
        ("mean_nearest_opp_dist", "8a. mean nearest opposite-ball dist"),
        ("mean_nearest_same_dist", "8b. mean nearest same-ball dist"),
        ("mean_opp_same_ratio", "8c. opp/same distance ratio"),
        ("mean_hetero_ball_separability", "9. heterogeneous-ball separability"),
        ("n_samples_representation_changed", "10. samples with changed representation"),
    ]
    comparison_table: dict[str, dict] = {}
    for key, label in metric_keys:
        if key == "n_samples_representation_changed":
            comparison_table[label] = {
                "b0": 0.0,
                "b1": float(run_df["n_samples_representation_changed"].mean()),
                "delta": float(run_df["n_samples_representation_changed"].mean()),
            }
            continue
        b0c = f"b0_{key}"
        b1c = f"b1_{key}"
        dc = f"delta_{key}"
        comparison_table[label] = {
            "b0": float(run_df[b0c].mean()),
            "b1": float(run_df[b1c].mean()),
            "delta": float(run_df[dc].mean()) if dc in run_df else float("nan"),
        }

    cls_counts = (
        Counter(repair_df["repair_class"].value_counts().to_dict())
        if len(repair_df)
        else Counter()
    )
    n_rep = len(repair_df)
    frac_a = cls_counts.get("A", 0) / n_rep if n_rep else 0.0
    frac_c = cls_counts.get("C", 0) / n_rep if n_rep else 0.0

    delta_min_surv = comparison_table["1. minority sample survival"]["delta"]
    delta_sep = comparison_table["9. heterogeneous-ball separability"]["delta"]
    delta_opp = comparison_table["7. minority-to-opposite support"]["delta"]
    frac_a = cls_counts.get("A", 0) / n_rep if n_rep else 0.0
    frac_c = cls_counts.get("C", 0) / n_rep if n_rep else 0.0
    frac_saves = float(run_df["frac_repair_saves_b0_lost"].mean()) if len(run_df) else 0.0

    geometrically_defensible = (
        delta_min_surv > 0.01
        and frac_a >= 0.35
        and frac_a > frac_c
        and frac_saves >= 0.15
        and delta_opp >= -1.0
        and (np.isnan(delta_sep) or delta_sep >= -0.05)
    )

    if geometrically_defensible:
        verdict = (
            "The repair is geometrically defensible: B1 raises minority survival, a "
            "material fraction of repairs are class A and save B0-del_ball samples, "
            "heterogeneous-ball separability holds, and minority-to-opposite support "
            "does not collapse."
        )
    elif delta_min_surv > 0.05 and frac_saves < 0.05:
        verdict = (
            "NOT geometrically defensible: B1 raises minority survival, but proposed "
            "repairs rarely target samples actually lost by del_ball in B0. The "
            "n_opp>=2 gate selects n=2-5 small balls that mostly survive del_ball "
            "already — survival gains are largely representation reshuffling, not "
            "recovery of the isolation->del_ball failure path."
        )
    elif delta_min_surv > 0.01 and frac_c >= frac_a:
        verdict = (
            "NOT geometrically defensible: B1 increases minority survival, but harmful "
            "(C) or ambiguous (B) repairs dominate beneficial (A). The change looks "
            "like majority-ball contamination rather than principled boundary "
            "preservation."
        )
    elif delta_min_surv > 0.01 and delta_opp < -2.0:
        verdict = (
            "MIXED / not clearly defensible: B1 improves minority survival, but "
            "minority-to-opposite support and cross-ball geometry weaken on average. "
            "This is closer to survival inflation than a geometrically principled "
            "granulation fix."
        )
    else:
        verdict = (
            "NOT supported as geometric improvement: B1 does not materially improve "
            "minority survival or repair quality on the frozen protocol."
        )

    summary = {
        "n_runs": int(len(run_df)),
        "n_repairs": n_rep,
        "protocol": {
            "b0": "KMeans(k=2), rho=0.95, del_ball(n<2)",
            "b1": (
                "virtual reassignment from small/singleton child with parent-local "
                "minority and n_opp_within_2hd1>=2; skip if child would empty"
            ),
            "repair_classes": {
                "A": "saves B0-del_ball samples + support>=2 + boundary signal + sibling het + sep>=0.8",
                "B": "mixed or conflicting signals",
                "C": "no del_ball benefit + majority contamination / weak geometry",
            },
        },
        "aggregate": {"comparison_table": comparison_table},
        "repair_classification": {
            "A": int(cls_counts.get("A", 0)),
            "B": int(cls_counts.get("B", 0)),
            "C": int(cls_counts.get("C", 0)),
            "frac_A": frac_a,
            "frac_B": cls_counts.get("B", 0) / n_rep if n_rep else 0.0,
            "frac_C": frac_c,
            "mean_frac_repair_saves_b0_lost": frac_saves,
        },
        "verdict": verdict,
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_REPORT_MD.write_text(_build_report(summary), encoding="utf-8")

    print(f"\nWrote {OUT_REPAIR_CSV} ({len(repair_df)} repairs)")
    print(f"Wrote {OUT_RUN_CSV} ({len(run_df)} runs)")
    print(f"Wrote {OUT_SUMMARY_JSON}")
    print(f"Wrote {OUT_REPORT_MD}")
    print("\nVerdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
