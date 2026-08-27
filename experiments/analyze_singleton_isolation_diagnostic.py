"""
Singleton isolation diagnostic (read-only).

Replays frozen recursive KMeans(k=2) granulation (rho=0.95, min_samples=1,
del_ball num_data=2). For every split that produces a singleton child, traces
that sample and its sibling inside the parent ball (plus global same-class
neighbours on the train set).

Labels A/B/C from local/global geometry only. Does NOT use Topo, CW, or
GBLSTSVM performance.

Does NOT modify granulation, del_ball, Topo, CW, or GBLSTSVM.

Run from GBLSTSVM root:
  python experiments/analyze_singleton_isolation_diagnostic.py
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

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granulation"
OUT_EVENT_CSV = OUT_DIR / "singleton_isolation_events.csv"
OUT_RUN_CSV = OUT_DIR / "singleton_isolation_runs.csv"
OUT_SUMMARY_JSON = OUT_DIR / "singleton_isolation_summary.json"
OUT_REPORT_MD = OUT_DIR / "singleton_isolation_report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
HD_EPS = 1e-12

# A/C overlap gates for the criterion verdict (pre-specified, not tuned).
AUROC_STRONG = 0.90
QUANTILE_SEPARATION_P90_P10 = True


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
    """Higher score should indicate the positive class (C / outlier)."""
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
    # Average ranks for ties.
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


def _hd_geometry(
    x: np.ndarray,
    y_x: float,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    indices: np.ndarray | None = None,
    self_index: int | None = None,
) -> dict[str, Any]:
    """Parent-local (or any set) distances from x. Excludes the singleton row."""
    diffs = features - x[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    same = labels == y_x
    opp = ~same
    if self_index is not None and indices is not None:
        keep = indices != self_index
        dists, same, opp = dists[keep], same[keep], opp[keep]
    else:
        self_mask = dists <= 1e-15
        if np.any(self_mask & same):
            self_idx = int(np.argmin(np.where(self_mask & same, dists, np.inf)))
            keep = np.ones(dists.shape[0], dtype=bool)
            keep[self_idx] = False
            dists, same, opp = dists[keep], same[keep], opp[keep]

    out: dict[str, Any] = {
        "hd1": float("nan"),
        "hd2": float("nan"),
        "hd2_defined": 0,
        "d_same": float("nan"),
        "n_same_excl_self": int(np.sum(same)),
        "n_opp": int(np.sum(opp)),
        "n_opp_1hd1": 0,
        "n_opp_2hd1": 0,
        "n_opp_3hd1": 0,
        "n_same_1hd1": 0,
        "n_same_2hd1": 0,
        "n_same_3hd1": 0,
        "n_same_hd2": 0,
        "s_size_hd2": 0,
        "edge_hd": "",
    }
    if not np.any(opp):
        out["edge_hd"] = "no_opposite"
        return out

    opp_d = dists[opp]
    hd1 = float(np.min(opp_d))
    out["hd1"] = hd1
    greater = opp_d[opp_d > hd1 + 1e-15]
    if greater.size:
        hd2 = float(np.min(greater))
        out["hd2"] = hd2
        out["hd2_defined"] = 1
    else:
        out["edge_hd"] = "no_second_heterogeneous"
        hd2 = float("nan")

    out["n_opp_1hd1"] = int(np.sum(opp_d <= hd1 + 1e-12))
    out["n_opp_2hd1"] = int(np.sum(opp_d <= (2.0 * hd1 + 1e-12)))
    out["n_opp_3hd1"] = int(np.sum(opp_d <= (3.0 * hd1 + 1e-12)))

    if np.any(same):
        same_d = dists[same]
        d_same = float(np.min(same_d))
        out["d_same"] = d_same
        out["n_same_1hd1"] = int(np.sum(same_d <= hd1 + 1e-12))
        out["n_same_2hd1"] = int(np.sum(same_d <= (2.0 * hd1 + 1e-12)))
        out["n_same_3hd1"] = int(np.sum(same_d <= (3.0 * hd1 + 1e-12)))
        if out["hd2_defined"]:
            n_hd2 = int(np.sum(same_d <= hd2 + 1e-12))
            out["n_same_hd2"] = n_hd2
            out["s_size_hd2"] = n_hd2 + 1  # include self, matching Mechanism 1
    elif out["hd2_defined"]:
        out["s_size_hd2"] = 1

    return out


def _nearest_same_index(
    x: np.ndarray,
    y_x: float,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    self_index: int,
) -> tuple[int | None, float]:
    same = (labels == y_x) & (indices != self_index)
    if not np.any(same):
        return None, float("nan")
    d = np.linalg.norm(features[same] - x[None, :], axis=1)
    j = int(np.argmin(d))
    return int(indices[same][j]), float(d[j])


def classify_event(row: dict[str, Any]) -> str:
    """
    A/B/C from geometry only (no Topo / no classifier metrics).

    A: minority-role point peeled from a same-class neighbour that KMeans
       assigned to the sibling, at a local scale (d_same <= 2*hd1).
    C: minority-or-majority point with no same-class neighbour globally
       inside 2*hd1 — isolated from its own class.
    B: everything else (lone-in-parent but globally near own class; majority
       peels; undefined hd1; conflicts).
    """
    if not np.isfinite(row["hd1"]):
        return "B"
    minority_role = bool(row["is_global_minority"] or row["is_parent_local_minority"])
    peel_local = (
        minority_role
        and int(row["nearest_same_in_sibling"]) == 1
        and np.isfinite(row["d_same_parent"])
        and float(row["d_same_parent"]) <= (2.0 * float(row["hd1"]) + 1e-12)
    )
    isolated_from_own = int(row["n_same_global_2hd1"]) == 0
    if peel_local and isolated_from_own:
        return "B"
    if peel_local:
        return "A"
    if isolated_from_own:
        return "C"
    return "B"


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
    n_train = indexed.shape[0]
    n_min = int(np.sum(y_all == global_min_label))
    n_maj = n_train - n_min

    gb = GBList(indexed)
    balls = gb.granular_balls
    events: list[dict[str, Any]] = []
    split_id = 0

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < PURITY and ball.num > MIN_SAMPLES_PER_BALL:
            parent_feats = ball.data_no_label.copy()
            parent_labels = _labels(ball).copy()
            parent_idx = _indices(ball).copy()
            parent_num = int(ball.num)
            parent_purity = float(ball.purity)
            parent_label = float(ball.label)
            parent_radius = float(ball.r)
            n_pos = int(np.sum(parent_labels == 1.0))
            n_neg = int(np.sum(parent_labels == -1.0))
            local_min_label = 1.0 if parent_label == -1.0 else -1.0

            c1, c2 = ball.split_2balls()
            split_id += 1
            children = (("first", c1, c2), ("second", c2, c1))
            max_child_p = max(float(c1.purity), float(c2.purity))
            purity_gain = max_child_p - parent_purity

            for child_slot, child, sibling in children:
                if int(child.num) != 1:
                    continue
                x = child.data_no_label[0].copy()
                y_x = float(child.data[0, -2])
                sample_index = int(child.data[0, -1])
                sib_idx = set(_indices(sibling).tolist())
                sib_labels = _labels(sibling)

                geo = _hd_geometry(
                    x,
                    y_x,
                    parent_feats,
                    parent_labels,
                    indices=parent_idx,
                    self_index=sample_index,
                )
                nn_parent, d_same_parent = _nearest_same_index(
                    x, y_x, parent_feats, parent_labels, parent_idx, sample_index
                )
                nn_global, d_same_global = _nearest_same_index(
                    x,
                    y_x,
                    feats_all,
                    y_all,
                    np.arange(n_train),
                    sample_index,
                )
                geo_g = _hd_geometry(
                    x,
                    y_x,
                    feats_all,
                    y_all,
                    indices=np.arange(n_train),
                    self_index=sample_index,
                )

                hd1 = geo["hd1"]
                scale = max(parent_radius, HD_EPS)
                hd1_norm = float(hd1 / scale) if np.isfinite(hd1) else float("nan")
                d_sib_center = float(np.linalg.norm(x - sibling.center))
                ratio_same_opp = (
                    float(d_same_parent / hd1)
                    if np.isfinite(d_same_parent) and np.isfinite(hd1) and hd1 > HD_EPS
                    else float("nan")
                )
                ratio_same_opp_global = (
                    float(d_same_global / hd1)
                    if np.isfinite(d_same_global) and np.isfinite(hd1) and hd1 > HD_EPS
                    else float("nan")
                )

                n_same_parent = int(np.sum(parent_labels == y_x))
                n_same_sib = int(np.sum(sib_labels == y_x))
                is_local_min = y_x == local_min_label
                is_global_min = y_x == global_min_label
                nearest_in_sib = int(
                    nn_parent is not None and int(nn_parent) in sib_idx
                )
                lone = n_same_parent == 1
                strong_iso = int(lone and is_local_min and n_same_sib == 0)
                near_boundary = int(
                    np.isfinite(hd1) and int(geo["n_opp_2hd1"]) >= 2
                )

                row: dict[str, Any] = {
                    "dataset": dataset,
                    "seed": seed,
                    "split_id": split_id,
                    "child_slot": child_slot,
                    "sample_index": sample_index,
                    "parent_num": parent_num,
                    "parent_n_pos": n_pos,
                    "parent_n_neg": n_neg,
                    "parent_n_singleton_class": n_same_parent,
                    "parent_n_other_class": parent_num - n_same_parent,
                    "parent_purity": parent_purity,
                    "parent_majority_label": parent_label,
                    "parent_radius": parent_radius,
                    "parent_local_min_label": local_min_label,
                    "kmeans_purity_gain": purity_gain,
                    "kmeans_max_child_purity": max_child_p,
                    "singleton_label": y_x,
                    "is_global_minority": int(is_global_min),
                    "is_parent_local_minority": int(is_local_min),
                    "sibling_num": int(sibling.num),
                    "sibling_label": float(sibling.label),
                    "sibling_purity": float(sibling.purity),
                    "sibling_n_singleton_class": n_same_sib,
                    "hd1": geo["hd1"],
                    "hd2": geo["hd2"],
                    "hd2_defined": geo["hd2_defined"],
                    "d_same_parent": d_same_parent,
                    "d_same_global": d_same_global,
                    "hd1_norm": hd1_norm,
                    "ratio_d_same_parent_over_hd1": ratio_same_opp,
                    "ratio_d_same_global_over_hd1": ratio_same_opp_global,
                    "n_opp_1hd1": geo["n_opp_1hd1"],
                    "n_opp_2hd1": geo["n_opp_2hd1"],
                    "n_opp_3hd1": geo["n_opp_3hd1"],
                    "n_same_parent_1hd1": geo["n_same_1hd1"],
                    "n_same_parent_2hd1": geo["n_same_2hd1"],
                    "n_same_parent_3hd1": geo["n_same_3hd1"],
                    "n_same_global_2hd1": geo_g["n_same_2hd1"],
                    "n_same_hd2": geo["n_same_hd2"],
                    "s_size_hd2": geo["s_size_hd2"],
                    "dist_to_sibling_center": d_sib_center,
                    "nearest_same_parent_index": -1 if nn_parent is None else nn_parent,
                    "nearest_same_in_sibling": nearest_in_sib,
                    "nearest_same_global_index": -1 if nn_global is None else nn_global,
                    "lone_in_parent": int(lone),
                    "strong_minority_isolation": strong_iso,
                    "near_class_boundary": near_boundary,
                    "only_one_opp_neighbour": int(geo["n_opp_1hd1"] == 1),
                    "ge2_same_class_neighbours_2hd1": int(geo_g["n_same_2hd1"] >= 2),
                    "edge_hd": geo["edge_hd"],
                    "would_del_if_leaf": int(int(child.num) < MIN_BALL_COUNT),
                }
                row["abc"] = classify_event(row)
                events.append(row)

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

    n_deleted = 0
    for row in events:
        leaf_n = sample_leaf_n.get(int(row["sample_index"]), 1)
        deleted = int(leaf_n < MIN_BALL_COUNT)
        row["leaf_num"] = leaf_n
        row["deleted_by_del_ball"] = deleted
        n_deleted += deleted

    gb.del_ball(num_data=MIN_BALL_COUNT)

    run = {
        "dataset": dataset,
        "seed": seed,
        "n_train": n_train,
        "n_global_min": n_min,
        "n_global_maj": n_maj,
        "global_min_label": global_min_label,
        "n_singleton_events": len(events),
        "n_deleted": n_deleted,
        "n_balls_after_del": len(gb.granular_balls),
    }
    return events, run


def _frac(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(np.mean(mask.astype(float)))


def summarize(events: pd.DataFrame, runs: pd.DataFrame) -> dict[str, Any]:
    n = int(len(events))
    abc_counts = events["abc"].value_counts().to_dict() if n else {}
    label_counts = (
        events["singleton_label"].map({1.0: "+1", -1.0: "-1"}).value_counts().to_dict()
        if n
        else {}
    )

    def sub(cls: str) -> pd.DataFrame:
        return events[events["abc"] == cls]

    a, b, c = sub("A"), sub("B"), sub("C")

    def pack_cls(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "n": int(len(df)),
            "frac": float(len(df) / n) if n else float("nan"),
            "frac_deleted": _frac(df["deleted_by_del_ball"]) if len(df) else float("nan"),
            "frac_global_min": _frac(df["is_global_minority"]) if len(df) else float("nan"),
            "frac_local_min": _frac(df["is_parent_local_minority"]) if len(df) else float("nan"),
            "frac_only_one_opp": _frac(df["only_one_opp_neighbour"]) if len(df) else float("nan"),
            "frac_n_opp_2hd1_eq1": _frac(df["n_opp_2hd1"] == 1) if len(df) else float("nan"),
            "frac_n_opp_2hd1_ge2": _frac(df["n_opp_2hd1"] >= 2) if len(df) else float("nan"),
            "frac_ge2_same": _frac(df["ge2_same_class_neighbours_2hd1"]) if len(df) else float("nan"),
            "frac_strong_iso": _frac(df["strong_minority_isolation"]) if len(df) else float("nan"),
            "frac_near_boundary": _frac(df["near_class_boundary"]) if len(df) else float("nan"),
            "frac_nearest_same_in_sibling": _frac(df["nearest_same_in_sibling"])
            if len(df)
            else float("nan"),
            "frac_lone_in_parent": _frac(df["lone_in_parent"]) if len(df) else float("nan"),
            "hd1_norm": _quantiles(df["hd1_norm"].to_numpy()) if len(df) else _quantiles(np.array([])),
            "ratio_same_parent": _quantiles(df["ratio_d_same_parent_over_hd1"].to_numpy())
            if len(df)
            else _quantiles(np.array([])),
            "ratio_same_global": _quantiles(df["ratio_d_same_global_over_hd1"].to_numpy())
            if len(df)
            else _quantiles(np.array([])),
            "kmeans_purity_gain": _quantiles(df["kmeans_purity_gain"].to_numpy())
            if len(df)
            else _quantiles(np.array([])),
            "parent_num": _quantiles(df["parent_num"].to_numpy()) if len(df) else _quantiles(np.array([])),
        }

    ac = events[events["abc"].isin(["A", "C"])]
    y_c = (ac["abc"] == "C").astype(int).to_numpy() if len(ac) else np.array([])

    def feat_auroc(col: str, invert: bool = False) -> float:
        if len(ac) == 0:
            return float("nan")
        s = ac[col].to_numpy(dtype=float)
        if invert:
            s = -s
        return _auroc(s, y_c)

    rule_rows = []

    def rule_table(name: str, pred_c: pd.Series, pred_a: pd.Series) -> dict[str, Any]:
        eval_df = events.copy()
        known = pred_c | pred_a
        eval_df = eval_df[known]
        if len(eval_df) == 0:
            return {"rule": name, "n_labeled": 0}
        pred = np.where(pred_c[known], "C", "A")
        true = eval_df["abc"].to_numpy()
        # Restrict to true A/C.
        keep = np.isin(true, ["A", "C"])
        pred, true = pred[keep], true[keep]
        if pred.size == 0:
            return {"rule": name, "n_labeled": int(known.sum()), "n_true_ac": 0}
        acc = float(np.mean(pred == true))
        # C as positive.
        tp = int(np.sum((pred == "C") & (true == "C")))
        fp = int(np.sum((pred == "C") & (true == "A")))
        fn = int(np.sum((pred == "A") & (true == "C")))
        tn = int(np.sum((pred == "A") & (true == "A")))
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        return {
            "rule": name,
            "n_labeled": int(known.sum()),
            "n_true_ac": int(pred.size),
            "accuracy_on_AC": acc,
            "C_precision": prec,
            "C_recall": rec,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    if n:
        r1_c = events["n_opp_2hd1"] == 1
        r1_a = events["n_opp_2hd1"] >= 2
        r2_ok = events["hd2_defined"] == 1
        r2_c = r2_ok & (events["s_size_hd2"] == 1)
        r2_a = r2_ok & (events["s_size_hd2"] > 1)
        r3_a = events["nearest_same_in_sibling"] == 1
        r3_c = events["nearest_same_in_sibling"] == 0
        r5_c = events["only_one_opp_neighbour"] == 1
        r5_a = events["only_one_opp_neighbour"] == 0
        rule_rows = [
            rule_table("n_opp_within_2hd1 == 1 → C else A", r1_c, r1_a),
            rule_table("hd2 |S|==1 → C, |S|>1 → A (defined hd2 only)", r2_c, r2_a),
            rule_table("nearest same-class not in sibling → C else A", r3_c, r3_a),
            rule_table("n_opp_within_hd1 == 1 → C else A", r5_c, r5_a),
        ]

    a_hd = pack_cls(a)["hd1_norm"]
    c_hd = pack_cls(c)["hd1_norm"]
    a_ratio = pack_cls(a)["ratio_same_global"]
    c_ratio = pack_cls(c)["ratio_same_global"]

    separated_hd1 = (
        np.isfinite(a_hd["p90"])
        and np.isfinite(c_hd["p10"])
        and a_hd["p90"] < c_hd["p10"]
    )
    separated_ratio = (
        np.isfinite(a_ratio["p90"])
        and np.isfinite(c_ratio["p10"])
        and a_ratio["p90"] < c_ratio["p10"]
    )

    aurocs = {
        "n_opp_2hd1_eq1": feat_auroc("n_opp_2hd1", invert=True) if n else float("nan"),
        "only_one_opp_neighbour": feat_auroc("only_one_opp_neighbour") if n else float("nan"),
        "hd1_norm": feat_auroc("hd1_norm") if n else float("nan"),
        "ratio_d_same_global_over_hd1": feat_auroc("ratio_d_same_global_over_hd1") if n else float("nan"),
        "lone_in_parent": feat_auroc("lone_in_parent") if n else float("nan"),
        "s_size_hd2": feat_auroc("s_size_hd2", invert=True) if n else float("nan"),
        "near_class_boundary": feat_auroc("near_class_boundary", invert=True) if n else float("nan"),
    }
    if n and len(ac):
        aurocs["n_opp_2hd1_eq1"] = _auroc((ac["n_opp_2hd1"] == 1).astype(float).to_numpy(), y_c)

    lone = events[events["lone_in_parent"] == 1] if n else events
    y_iso = (
        (lone["n_same_global_2hd1"] == 0).astype(int).to_numpy() if len(lone) else np.array([])
    )
    lone_parent_aurocs: dict[str, float] = {}
    if len(lone) and y_iso.size and y_iso.min() != y_iso.max():
        for col, invert in (
            ("hd1_norm", False),
            ("hd1", False),
            ("n_opp_2hd1", True),
            ("n_opp_3hd1", True),
            ("only_one_opp_neighbour", False),
            ("near_class_boundary", True),
            ("parent_purity", False),
            ("kmeans_purity_gain", False),
            ("parent_num", True),
            ("sibling_purity", False),
            ("dist_to_sibling_center", False),
            ("is_global_minority", False),
            ("is_parent_local_minority", False),
            ("strong_minority_isolation", False),
            ("hd2_defined", True),
        ):
            s = lone[col].to_numpy(dtype=float)
            if invert:
                s = -s
            lone_parent_aurocs[col] = _auroc(s, y_iso)
        lone_parent_aurocs["n_opp_2hd1_eq1"] = _auroc(
            (lone["n_opp_2hd1"] == 1).astype(float).to_numpy(), y_iso
        )

    iso = lone[lone["n_same_global_2hd1"] == 0] if len(lone) else lone
    near = lone[lone["n_same_global_2hd1"] >= 1] if len(lone) else lone
    max_parent_auroc = (
        max((v for v in lone_parent_aurocs.values() if np.isfinite(v)), default=float("nan"))
    )
    parent_local_separates = bool(
        np.isfinite(max_parent_auroc) and max_parent_auroc >= AUROC_STRONG
    )
    parent_local_ac_max = max(
        (
            v
            for k, v in aurocs.items()
            if k not in ("ratio_d_same_global_over_hd1", "lone_in_parent") and np.isfinite(v)
        ),
        default=float("nan"),
    )

    if n == 0:
        verdict = "No singleton events; cannot assess a local criterion."
        simple_exists = False
    elif parent_local_separates:
        simple_exists = True
        verdict = (
            "YES: among lone-in-parent singletons, a parent-local statistic "
            f"(max AUROC={max_parent_auroc:.3f}) predicts which points are "
            "globally isolated from their own class."
        )
    else:
        simple_exists = False
        verdict = (
            "NO parent-local criterion. Singleton children are always deleted by "
            "del_ball. Half of events are 2-sample parents, so n_opp at hd1 equals 1 "
            "and hd2 is usually undefined. hd1_norm, n_opp within {1,2,3}·hd1, sibling "
            "purity, and KMeans purity gain all sit near AUROC 0.5 for telling globally "
            "class-isolated outliers from lone-in-parent points that still have "
            "same-class neighbours outside the parent. The only clean separator is "
            "d_same_global / hd1 — looking outside the parent — which is the "
            "definition of C, not a new local rule. A repair that uses only the parent "
            "ball cannot distinguish legitimate isolated minority/boundary points from "
            "true outliers."
        )

    overall = {
        "n_events": n,
        "n_runs": int(len(runs)),
        "abc_counts": {k: int(v) for k, v in abc_counts.items()},
        "abc_frac": {k: float(v / n) for k, v in abc_counts.items()} if n else {},
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "frac_deleted": _frac(events["deleted_by_del_ball"]) if n else float("nan"),
        "frac_only_one_opp_neighbour": _frac(events["only_one_opp_neighbour"]) if n else float("nan"),
        "frac_ge2_same_class_neighbours": _frac(events["ge2_same_class_neighbours_2hd1"])
        if n
        else float("nan"),
        "frac_strong_minority_isolation": _frac(events["strong_minority_isolation"])
        if n
        else float("nan"),
        "frac_global_minority": _frac(events["is_global_minority"]) if n else float("nan"),
        "frac_near_boundary": _frac(events["near_class_boundary"]) if n else float("nan"),
        "frac_lone_in_parent": _frac(events["lone_in_parent"]) if n else float("nan"),
        "frac_nearest_same_in_sibling": _frac(events["nearest_same_in_sibling"]) if n else float("nan"),
        "frac_hd2_defined": _frac(events["hd2_defined"]) if n else float("nan"),
        "frac_parent_num_eq2": _frac(events["parent_num"] == 2) if n else float("nan"),
        "hd1_norm_all": _quantiles(events["hd1_norm"].to_numpy()) if n else _quantiles(np.array([])),
        "ratio_same_parent_all": _quantiles(events["ratio_d_same_parent_over_hd1"].to_numpy())
        if n
        else _quantiles(np.array([])),
        "ratio_same_global_all": _quantiles(events["ratio_d_same_global_over_hd1"].to_numpy())
        if n
        else _quantiles(np.array([])),
        "by_abc": {"A": pack_cls(a), "B": pack_cls(b), "C": pack_cls(c)},
        "auroc_C_vs_A": aurocs,
        "simple_rules": rule_rows,
        "range_separation": {
            "hd1_norm_p90A_lt_p10C": separated_hd1,
            "ratio_global_p90A_lt_p10C": separated_ratio,
        },
        "auroc_C_vs_A_note": (
            "lone_in_parent and ratio_d_same_global_over_hd1 are near-definitional "
            "for A vs C and must not be read as discovered gates."
        ),
        "parent_local_ac_auroc_max": parent_local_ac_max,
        "lone_global_isolation": {
            "n_lone": int(len(lone)) if n else 0,
            "n_peel": int((events["lone_in_parent"] == 0).sum()) if n else 0,
            "frac_lone_globally_isolated": _frac(lone["n_same_global_2hd1"] == 0)
            if len(lone)
            else float("nan"),
            "parent_local_auroc": lone_parent_aurocs,
            "max_parent_local_auroc": max_parent_auroc if lone_parent_aurocs else float("nan"),
            "isolated_hd1_norm": _quantiles(iso["hd1_norm"].to_numpy())
            if len(iso)
            else _quantiles(np.array([])),
            "supported_hd1_norm": _quantiles(near["hd1_norm"].to_numpy())
            if len(near)
            else _quantiles(np.array([])),
            "isolated_frac_n_opp_2hd1_eq1": _frac(iso["n_opp_2hd1"] == 1)
            if len(iso)
            else float("nan"),
            "supported_frac_n_opp_2hd1_eq1": _frac(near["n_opp_2hd1"] == 1)
            if len(near)
            else float("nan"),
        },
        "simple_parent_local_criterion_exists": simple_exists,
        "simple_local_criterion_exists": simple_exists,
        "verdict": verdict,
        "label_definitions": {
            "A": (
                "parent-local or global minority peeled from a same-class neighbour "
                "assigned to the sibling, with d_same_parent <= 2*hd1"
            ),
            "C": "no same-class neighbour anywhere in the train set within 2*hd1",
            "B": "neither A nor C, or A/C conflict",
        },
    }
    return overall


def _fmt_q(q: dict[str, float]) -> str:
    if not q or not np.isfinite(q.get("n", np.nan)) or q.get("n", 0) == 0:
        return "empty"
    return (
        f"n={q['n']} mean={q['mean']:.3f} p10={q['p10']:.3f} "
        f"p50={q['p50']:.3f} p90={q['p90']:.3f} [{q['min']:.3f}, {q['max']:.3f}]"
    )


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
) -> None:
    lines = [
        "# Singleton isolation diagnostic",
        "",
        "Read-only replay of frozen recursive KMeans(k=2) splits "
        f"(purity={PURITY}, min_samples_per_ball={MIN_SAMPLES_PER_BALL}, "
        f"del_ball num_data={MIN_BALL_COUNT}).",
        "",
        "A/B/C labels use parent-local and train-set geometry only. "
        "**Topo / CW / GBLSTSVM performance is not used.**",
        "",
        "## Label definitions",
        "",
        f"- **A** — {summary['label_definitions']['A']}",
        f"- **C** — {summary['label_definitions']['C']}",
        f"- **B** — {summary['label_definitions']['B']}",
        "",
        "## Headline",
        "",
        f"- Singleton events: {summary['n_events']} across {summary['n_runs']} runs",
        f"- A/B/C counts: {summary['abc_counts']}",
        f"- A/B/C fractions: { {k: round(v, 3) for k, v in summary['abc_frac'].items()} }",
        f"- Label (+1/−1) counts: {summary['label_counts']}",
        f"- Fraction eventually deleted by del_ball: {_fmt(summary['frac_deleted'])}",
        f"- Fraction with only 1 opposite neighbour (at hd1): {_fmt(summary['frac_only_one_opp_neighbour'])}",
        f"- Fraction with ≥2 same-class neighbours (global, 2·hd1): {_fmt(summary['frac_ge2_same_class_neighbours'])}",
        f"- Fraction with strong minority isolation: {_fmt(summary['frac_strong_minority_isolation'])}",
        f"- Fraction global minority: {_fmt(summary['frac_global_minority'])}",
        f"- Fraction near class boundary (n_opp 2·hd1 ≥ 2): {_fmt(summary['frac_near_boundary'])}",
        f"- Fraction lone-in-parent: {_fmt(summary['frac_lone_in_parent'])}",
        f"- Fraction nearest same-class in sibling: {_fmt(summary['frac_nearest_same_in_sibling'])}",
        f"- Fraction hd2 defined: {_fmt(summary['frac_hd2_defined'])}",
        f"- Fraction parent size == 2: {_fmt(summary.get('frac_parent_num_eq2', float('nan')))}",
        "",
        "## Distributions",
        "",
        f"- hd1_norm (all): {_fmt_q(summary['hd1_norm_all'])}",
        f"- d_same_parent / hd1 (all): {_fmt_q(summary['ratio_same_parent_all'])}",
        f"- d_same_global / hd1 (all): {_fmt_q(summary['ratio_same_global_all'])}",
        "",
    ]
    for cls in ("A", "B", "C"):
        pack = summary["by_abc"][cls]
        lines.append(f"### Class {cls} (n={pack['n']}, frac={_fmt(pack['frac'])})")
        lines.append(f"- hd1_norm: {_fmt_q(pack['hd1_norm'])}")
        lines.append(f"- d_same_parent/hd1: {_fmt_q(pack['ratio_same_parent'])}")
        lines.append(f"- d_same_global/hd1: {_fmt_q(pack['ratio_same_global'])}")
        lines.append(f"- KMeans purity gain: {_fmt_q(pack['kmeans_purity_gain'])}")
        lines.append(f"- parent size: {_fmt_q(pack['parent_num'])}")
        lines.append(
            f"- deleted={_fmt(pack['frac_deleted'])}, "
            f"global_min={_fmt(pack['frac_global_min'])}, "
            f"local_min={_fmt(pack['frac_local_min'])}, "
            f"only_1_opp={_fmt(pack['frac_only_one_opp'])}, "
            f"n_opp_2hd1==1={_fmt(pack['frac_n_opp_2hd1_eq1'])}, "
            f"≥2 same={_fmt(pack['frac_ge2_same'])}, "
            f"strong_iso={_fmt(pack['frac_strong_iso'])}, "
            f"boundary={_fmt(pack['frac_near_boundary'])}, "
            f"nearest_in_sibling={_fmt(pack['frac_nearest_same_in_sibling'])}, "
            f"lone={_fmt(pack['frac_lone_in_parent'])}"
        )
        lines.append("")

    lines.extend(
        [
            "## Can a simple local criterion separate A from C?",
            "",
            "A vs C AUROC for `lone_in_parent` and `d_same_global/hd1` is "
            "near-tautological (those quantities enter the label definitions). "
            "The non-circular test is: among **lone-in-parent** events, can "
            "**parent-local** features predict global isolation "
            "(`n_same_global within 2·hd1 == 0`)?",
            "",
            f"- hd1_norm p90(A) < p10(C): {summary['range_separation']['hd1_norm_p90A_lt_p10C']}",
            f"- d_same_global/hd1 p90(A) < p10(C): {summary['range_separation']['ratio_global_p90A_lt_p10C']}",
            f"- Max parent-local AUROC on A vs C (excluding definitional features): "
            f"{_fmt(summary.get('parent_local_ac_auroc_max', float('nan')))}",
            "",
            "| Feature (higher → more C-like) | AUROC C vs A |",
            "|---|---:|",
        ]
    )
    for name, val in summary["auroc_C_vs_A"].items():
        lines.append(f"| {name} | {_fmt(val)} |")

    lines.extend(["", "### Parameter-free rules (evaluated only on true A/C)", ""])
    lines.append("| Rule | n | acc | C prec | C rec | TP FP FN TN |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in summary["simple_rules"]:
        if r.get("n_true_ac", 0) == 0:
            lines.append(f"| {r['rule']} | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {r['rule']} | {r['n_true_ac']} | {_fmt(r['accuracy_on_AC'])} | "
            f"{_fmt(r['C_precision'])} | {_fmt(r['C_recall'])} | "
            f"{r['tp']} {r['fp']} {r['fn']} {r['tn']} |"
        )

    lone_iso = summary.get("lone_global_isolation", {})
    lines.extend(
        [
            "",
            "### Non-circular test: lone-in-parent → globally isolated?",
            "",
            f"- Lone-in-parent events: {lone_iso.get('n_lone', 0)}; peels: {lone_iso.get('n_peel', 0)}",
            f"- Fraction of lones with no same-class neighbour in 2·hd1 globally: "
            f"{_fmt(lone_iso.get('frac_lone_globally_isolated', float('nan')))}",
            f"- n_opp 2·hd1 == 1: isolated {_fmt(lone_iso.get('isolated_frac_n_opp_2hd1_eq1', float('nan')))} "
            f"vs still-supported {_fmt(lone_iso.get('supported_frac_n_opp_2hd1_eq1', float('nan')))}",
            f"- hd1_norm isolated: {_fmt_q(lone_iso.get('isolated_hd1_norm', {}))}",
            f"- hd1_norm supported: {_fmt_q(lone_iso.get('supported_hd1_norm', {}))}",
            f"- Max parent-local AUROC: {_fmt(lone_iso.get('max_parent_local_auroc', float('nan')))}",
            "",
            "| Parent-local feature | AUROC (lone → globally isolated) |",
            "|---|---:|",
        ]
    )
    for name, val in (lone_iso.get("parent_local_auroc") or {}).items():
        lines.append(f"| {name} | {_fmt(val)} |")

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            summary["verdict"],
            "",
            f"simple_parent_local_criterion_exists = {summary['simple_local_criterion_exists']}",
            "",
            "## Per-run event counts",
            "",
            "| dataset | seed | n_train | singletons | deleted | after del_ball |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in runs.sort_values(["dataset", "seed"]).iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['seed'])} | {int(r['n_train'])} | "
            f"{int(r['n_singleton_events'])} | {int(r['n_deleted'])} | "
            f"{int(r['n_balls_after_del'])} |"
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
            f"- `{OUT_EVENT_CSV.name}`",
            f"- `{OUT_RUN_CSV.name}`",
            f"- `{OUT_SUMMARY_JSON.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--summarize-only" in sys.argv:
        events_df = pd.read_csv(OUT_EVENT_CSV)
        runs_df = pd.read_csv(OUT_RUN_CSV)
        summary = summarize(events_df, runs_df)
        proto = {}
        if OUT_SUMMARY_JSON.exists():
            proto = json.loads(OUT_SUMMARY_JSON.read_text(encoding="utf-8")).get("protocol", {})
        summary["protocol"] = proto
        failed = proto.get("failed", [])
        OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT_MD, summary, runs_df, failed)
        print(summary["verdict"])
        print(f"Wrote {OUT_SUMMARY_JSON}")
        print(f"Wrote {OUT_REPORT_MD}")
        return 0

    descriptors = discover_available_datasets()
    all_events: list[dict[str, Any]] = []
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
            events, run = collect_run(
                split.X_train,
                split.y_train,
                seed=seed,
                dataset=split.name,
                global_min_label=float(split.minority_label),
            )
            abc = Counter(e["abc"] for e in events)
            run["n_A"] = int(abc.get("A", 0))
            run["n_B"] = int(abc.get("B", 0))
            run["n_C"] = int(abc.get("C", 0))
            run_rows.append(run)
            all_events.extend(events)
            print(
                f"{split.name} seed={seed}: events={len(events)} "
                f"A={run['n_A']} B={run['n_B']} C={run['n_C']} "
                f"deleted={run['n_deleted']}"
            )

    events_df = pd.DataFrame(all_events)
    runs_df = pd.DataFrame(run_rows)
    events_df.to_csv(OUT_EVENT_CSV, index=False)
    runs_df.to_csv(OUT_RUN_CSV, index=False)

    summary = summarize(events_df, runs_df)
    summary["protocol"] = {
        "purity": PURITY,
        "min_samples_per_ball": MIN_SAMPLES_PER_BALL,
        "min_ball_count": MIN_BALL_COUNT,
        "seeds": SEEDS,
        "datasets": [d["name"] for d in descriptors],
        "failed": failed,
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT_MD, summary, runs_df, failed)

    print(f"\nWrote {OUT_EVENT_CSV}")
    print(f"Wrote {OUT_RUN_CSV}")
    print(f"Wrote {OUT_SUMMARY_JSON}")
    print(f"Wrote {OUT_REPORT_MD}")
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
