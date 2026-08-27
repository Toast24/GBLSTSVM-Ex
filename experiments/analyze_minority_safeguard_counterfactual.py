"""
Read-only minority-safeguard counterfactual granulation replay.

Replays frozen recursive KMeans(k=2) (rho=0.95, min_samples=1, del_ball n<2)
under four hypothetical split gates. Does NOT change the actual granulation
implementation: each method is a virtual tree replay.

M0: accept every purity-eligible split (baseline).
M1: reject if smaller child n<2 AND it contains global-minority samples.
M2: reject if smaller child n<=3 AND minority_fraction_small_child >= 0.5.
M3: reject if parent_n<=4 AND minority_fraction_small_child >= 0.5.

Rejected splits keep the parent as a leaf; downstream children are not created.
Safeguards are defined from split geometry only (no Topo / CW / GBLSTSVM).

Run from GBLSTSVM root:
  python experiments/analyze_minority_safeguard_counterfactual.py
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
OUT_CSV = OUT_DIR / "minority_safeguard_counterfactual.csv"
OUT_SUMMARY = OUT_DIR / "minority_safeguard_counterfactual_summary.json"
OUT_REPORT = OUT_DIR / "minority_safeguard_counterfactual_report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
SMALL_BALL_MAX = 5
METHODS = ("M0", "M1", "M2", "M3")


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


def _pick_small_large(
    c1: GranularBall,
    c2: GranularBall,
    global_min_label: float,
) -> tuple[GranularBall, GranularBall]:
    n1, n2 = int(c1.num), int(c2.num)
    m1 = int(np.sum(_labels(c1) == global_min_label))
    m2 = int(np.sum(_labels(c2) == global_min_label))
    if n1 < n2:
        return c1, c2
    if n2 < n1:
        return c2, c1
    if m1 >= m2:
        return c1, c2
    return c2, c1


def _should_reject(
    method: str,
    *,
    parent_n: int,
    n_small: int,
    n_min_small: int,
) -> bool:
    min_frac = (n_min_small / n_small) if n_small > 0 else 0.0
    if method == "M0":
        return False
    if method == "M1":
        return n_small < 2 and n_min_small > 0
    if method == "M2":
        return n_small <= 3 and min_frac >= 0.5
    if method == "M3":
        return parent_n <= 4 and min_frac >= 0.5
    raise ValueError(f"Unknown method {method}")


def _split_flags(
    *,
    parent_n: int,
    n_small: int,
    n_large: int,
    n_min_small: int,
    min_frac_small: float,
) -> dict[str, Any]:
    singleton_creation = n_small == 1
    minority_peeling = n_min_small > 0 and n_small <= 3
    late_small_parent = parent_n <= 4
    useful_retained = n_small >= 2 and n_large >= 2 and not (
        n_min_small > 0 and n_small <= 3 and min_frac_small >= 0.5
    )
    if singleton_creation and n_min_small > 0:
        primary = "minority_peeling_singleton"
    elif singleton_creation:
        primary = "singleton_creation"
    elif minority_peeling and min_frac_small >= 0.5:
        primary = "minority_peeling"
    elif late_small_parent:
        primary = "late_small_parent"
    elif useful_retained:
        primary = "useful_retained"
    else:
        primary = "other"
    return {
        "is_singleton_creation": int(singleton_creation),
        "is_minority_peeling": int(minority_peeling),
        "is_late_small_parent": int(late_small_parent),
        "is_useful_retained": int(useful_retained),
        "primary_kind": primary,
        "minority_fraction_small_child": float(min_frac_small),
    }


def replay_method(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    dataset: str,
    global_min_label: float,
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    np.random.seed(seed)
    indexed = _build_indexed_matrix(X_train, y_train)
    y_all = indexed[:, -2]
    n_train = int(indexed.shape[0])
    n_min_train = int(np.sum(y_all == global_min_label))
    n_maj_train = n_train - n_min_train

    gb = GBList(indexed)
    balls = gb.granular_balls
    depths = [0]
    proposed: list[dict[str, Any]] = []
    split_id = 0

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < PURITY and ball.num > MIN_SAMPLES_PER_BALL:
            parent_n = int(ball.num)
            parent_depth = int(depths[i])
            parent_purity = float(ball.purity)
            c1, c2 = ball.split_2balls()
            split_id += 1
            small, large = _pick_small_large(c1, c2, global_min_label)
            n_small, n_large = int(small.num), int(large.num)
            n_min_small = int(np.sum(_labels(small) == global_min_label))
            min_frac = (n_min_small / n_small) if n_small else 0.0
            reject = _should_reject(
                method,
                parent_n=parent_n,
                n_small=n_small,
                n_min_small=n_min_small,
            )
            flags = _split_flags(
                parent_n=parent_n,
                n_small=n_small,
                n_large=n_large,
                n_min_small=n_min_small,
                min_frac_small=min_frac,
            )
            proposed.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "split_id": split_id,
                    "parent_n": parent_n,
                    "parent_depth": parent_depth,
                    "parent_purity": parent_purity,
                    "n_small": n_small,
                    "n_large": n_large,
                    "n_min_small": n_min_small,
                    "rejected": int(reject),
                    **flags,
                }
            )
            if reject:
                i += 1
            else:
                child_depth = parent_depth + 1
                balls[i] = c1
                depths[i] = child_depth
                balls.append(c2)
                depths.append(child_depth)
                ll += 1
        else:
            i += 1

    n_proposed = len(proposed)
    rejected = [r for r in proposed if r["rejected"] == 1]
    n_rejected = len(rejected)

    n_min_small_before = 0
    survived_min = 0
    survived_maj = 0
    n_removed = 0
    for ball in balls:
        n = int(ball.num)
        labs = _labels(ball)
        n_min = int(np.sum(labs == global_min_label))
        n_maj = n - n_min
        if n <= SMALL_BALL_MAX:
            n_min_small_before += n_min
        deleted = n < MIN_BALL_COUNT
        if deleted:
            n_removed += n
        else:
            survived_min += n_min
            survived_maj += n_maj

    gb.del_ball(num_data=MIN_BALL_COUNT)
    kept = gb.granular_balls
    n_balls = len(kept)
    purities = np.array([float(b.purity) for b in kept], dtype=float)
    radii = np.array([float(b.r) for b in kept], dtype=float)
    ball_labels = np.array(
        [-1.0 if b.label == -1 else 1.0 for b in kept], dtype=float
    )
    n_pos_balls = int(np.sum(ball_labels == 1.0))
    n_neg_balls = int(np.sum(ball_labels == -1.0))
    n_min_label_balls = int(np.sum(ball_labels == global_min_label))
    n_maj_label_balls = n_balls - n_min_label_balls

    def _mean_rej(key: str) -> float:
        if not rejected:
            return float("nan")
        return float(np.mean([r[key] for r in rejected]))

    def _frac_rej(key: str) -> float:
        if not rejected:
            return float("nan")
        return float(np.mean([r[key] for r in rejected]))

    parent_n_rej = np.array([r["parent_n"] for r in rejected], dtype=float)
    depth_rej = np.array([r["parent_depth"] for r in rejected], dtype=float)
    primary = Counter(r["primary_kind"] for r in rejected)

    run = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "n_train": n_train,
        "n_min_train": n_min_train,
        "n_maj_train": n_maj_train,
        "global_min_label": float(global_min_label),
        "n_balls_after_del": n_balls,
        "n_balls_before_del": len(balls),
        "n_min_survive_del_ball": survived_min,
        "n_maj_survive_del_ball": survived_maj,
        "minority_survival_rate": (
            survived_min / n_min_train if n_min_train else float("nan")
        ),
        "majority_survival_rate": (
            survived_maj / n_maj_train if n_maj_train else float("nan")
        ),
        "minority_small_ball_rate": (
            n_min_small_before / n_min_train if n_min_train else float("nan")
        ),
        "n_samples_removed_del_ball": n_removed,
        "mean_purity": float(np.mean(purities)) if purities.size else float("nan"),
        "mean_radius": float(np.mean(radii)) if radii.size else float("nan"),
        "n_pos_balls": n_pos_balls,
        "n_neg_balls": n_neg_balls,
        "n_min_label_balls": n_min_label_balls,
        "n_maj_label_balls": n_maj_label_balls,
        "n_proposed_splits": n_proposed,
        "n_rejected_splits": n_rejected,
        "frac_proposed_rejected": (
            n_rejected / n_proposed if n_proposed else 0.0
        ),
        "n_accepted_splits": n_proposed - n_rejected,
        "mean_parent_n_rejected": _mean_rej("parent_n"),
        "median_parent_n_rejected": (
            float(np.median(parent_n_rej)) if parent_n_rej.size else float("nan")
        ),
        "mean_depth_rejected": _mean_rej("parent_depth"),
        "median_depth_rejected": (
            float(np.median(depth_rej)) if depth_rej.size else float("nan")
        ),
        "n_rej_parent_n_le4": int(np.sum(parent_n_rej <= 4)) if parent_n_rej.size else 0,
        "n_rej_parent_n_gt4": int(np.sum(parent_n_rej > 4)) if parent_n_rej.size else 0,
        "frac_rej_singleton_creation": _frac_rej("is_singleton_creation"),
        "frac_rej_minority_peeling": _frac_rej("is_minority_peeling"),
        "frac_rej_late_small_parent": _frac_rej("is_late_small_parent"),
        "frac_rej_useful_retained": _frac_rej("is_useful_retained"),
        "n_rej_minority_peeling_singleton": int(primary.get("minority_peeling_singleton", 0)),
        "n_rej_singleton_creation": int(primary.get("singleton_creation", 0)),
        "n_rej_minority_peeling": int(primary.get("minority_peeling", 0)),
        "n_rej_late_small_parent": int(primary.get("late_small_parent", 0)),
        "n_rej_useful_retained": int(primary.get("useful_retained", 0)),
        "n_rej_other": int(primary.get("other", 0)),
    }
    return run, proposed


def _nanmean(values: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _quantiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float("nan") for k in ("n", "min", "p25", "p50", "p75", "max", "mean")}
    qs = np.quantile(v, [0.0, 0.25, 0.50, 0.75, 1.0])
    return {
        "n": int(v.size),
        "min": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "max": float(qs[4]),
        "mean": float(np.mean(v)),
    }


def _method_aggregate(df: pd.DataFrame, method: str) -> dict[str, Any]:
    sub = df[df["method"] == method]
    if sub.empty:
        return {"method": method, "n_runs": 0}
    return {
        "method": method,
        "n_runs": int(len(sub)),
        "mean_n_balls": _nanmean(sub["n_balls_after_del"].tolist()),
        "mean_n_min_survive": _nanmean(sub["n_min_survive_del_ball"].tolist()),
        "mean_n_maj_survive": _nanmean(sub["n_maj_survive_del_ball"].tolist()),
        "mean_minority_survival": _nanmean(sub["minority_survival_rate"].tolist()),
        "mean_majority_survival": _nanmean(sub["majority_survival_rate"].tolist()),
        "mean_minority_small_ball_rate": _nanmean(sub["minority_small_ball_rate"].tolist()),
        "mean_n_removed": _nanmean(sub["n_samples_removed_del_ball"].tolist()),
        "mean_purity": _nanmean(sub["mean_purity"].tolist()),
        "mean_radius": _nanmean(sub["mean_radius"].tolist()),
        "mean_n_min_label_balls": _nanmean(sub["n_min_label_balls"].tolist()),
        "mean_n_maj_label_balls": _nanmean(sub["n_maj_label_balls"].tolist()),
        "mean_n_proposed": _nanmean(sub["n_proposed_splits"].tolist()),
        "mean_n_rejected": _nanmean(sub["n_rejected_splits"].tolist()),
        "mean_frac_rejected": _nanmean(sub["frac_proposed_rejected"].tolist()),
        "mean_parent_n_rejected": _nanmean(sub["mean_parent_n_rejected"].tolist()),
        "mean_depth_rejected": _nanmean(sub["mean_depth_rejected"].tolist()),
        "mean_frac_rej_singleton": _nanmean(sub["frac_rej_singleton_creation"].tolist()),
        "mean_frac_rej_peeling": _nanmean(sub["frac_rej_minority_peeling"].tolist()),
        "mean_frac_rej_late_parent": _nanmean(sub["frac_rej_late_small_parent"].tolist()),
        "mean_frac_rej_useful": _nanmean(sub["frac_rej_useful_retained"].tolist()),
        "sum_rejected": int(sub["n_rejected_splits"].sum()),
        "sum_rej_peel_singleton": int(sub["n_rej_minority_peeling_singleton"].sum()),
        "sum_rej_singleton": int(sub["n_rej_singleton_creation"].sum()),
        "sum_rej_peeling": int(sub["n_rej_minority_peeling"].sum()),
        "sum_rej_late_parent": int(sub["n_rej_late_small_parent"].sum()),
        "sum_rej_useful": int(sub["n_rej_useful_retained"].sum()),
        "sum_rej_other": int(sub["n_rej_other"].sum()),
        "sum_rej_parent_le4": int(sub["n_rej_parent_n_le4"].sum()),
        "sum_rej_parent_gt4": int(sub["n_rej_parent_n_gt4"].sum()),
        "n_collapse_runs": int(
            (
                (sub["n_balls_after_del"] < 2)
                | (sub["n_min_label_balls"] == 0)
                | (sub["n_maj_label_balls"] == 0)
            ).sum()
        ),
        "frac_rejected_parent_gt4": (
            int(sub["n_rej_parent_n_gt4"].sum()) / int(sub["n_rejected_splits"].sum())
            if int(sub["n_rejected_splits"].sum())
            else 0.0
        ),
        "n_balls_quantiles": _quantiles(sub["n_balls_after_del"].to_numpy()),
        "min_surv_quantiles": _quantiles(sub["minority_survival_rate"].to_numpy()),
        "parent_n_rejected_quantiles": _quantiles(sub["median_parent_n_rejected"].to_numpy()),
        "depth_rejected_quantiles": _quantiles(sub["median_depth_rejected"].to_numpy()),
    }


def _tradeoff(df: pd.DataFrame) -> dict[str, Any]:
    m0 = df[df["method"] == "M0"].set_index(["dataset", "seed"])
    out: dict[str, Any] = {}
    for method in METHODS:
        empty = {
            "mean_delta_min_survival": 0.0,
            "mean_delta_maj_survival": 0.0,
            "mean_rel_ball_reduction": 0.0,
            "mean_delta_n_balls": 0.0,
            "mean_rel_proposed_reduction": 0.0,
            "mean_delta_small_ball_rate": 0.0,
            "mean_delta_purity": 0.0,
            "runs_min_surv_improved": 0,
            "runs_balls_cut_over_25pct": 0,
            "runs_collapse": 0,
            "n_paired_runs": 0,
            "frac_runs_min_surv_improved": float("nan"),
            "frac_rejected_parent_gt4": 0.0,
        }
        if method == "M0":
            out[method] = empty
            continue
        deltas_min: list[float] = []
        deltas_maj: list[float] = []
        rel_ball: list[float] = []
        d_balls: list[float] = []
        rel_prop: list[float] = []
        d_small: list[float] = []
        d_pur: list[float] = []
        n_improved = 0
        n_cut25 = 0
        n_collapse = 0
        for _, row in df[df["method"] == method].iterrows():
            key = (row["dataset"], int(row["seed"]))
            if key not in m0.index:
                continue
            base = m0.loc[key]
            dmin = float(row["minority_survival_rate"] - base["minority_survival_rate"])
            dmaj = float(row["majority_survival_rate"] - base["majority_survival_rate"])
            b0 = float(base["n_balls_after_del"])
            bm = float(row["n_balls_after_del"])
            rel = (b0 - bm) / b0 if b0 > 0 else float("nan")
            p0 = float(base["n_proposed_splits"])
            pm = float(row["n_proposed_splits"])
            relp = (p0 - pm) / p0 if p0 > 0 else float("nan")
            deltas_min.append(dmin)
            deltas_maj.append(dmaj)
            rel_ball.append(rel)
            d_balls.append(bm - b0)
            rel_prop.append(relp)
            d_small.append(
                float(row["minority_small_ball_rate"] - base["minority_small_ball_rate"])
            )
            d_pur.append(float(row["mean_purity"] - base["mean_purity"]))
            if dmin > 1e-12:
                n_improved += 1
            if np.isfinite(rel) and rel >= 0.25:
                n_cut25 += 1
            missing_class = int(row["n_min_label_balls"]) == 0 or int(row["n_maj_label_balls"]) == 0
            if int(row["n_balls_after_del"]) < 2 or missing_class:
                n_collapse += 1
        n_rej = int(df.loc[df["method"] == method, "n_rejected_splits"].sum())
        n_large = int(df.loc[df["method"] == method, "n_rej_parent_n_gt4"].sum())
        out[method] = {
            "mean_delta_min_survival": _nanmean(deltas_min),
            "mean_delta_maj_survival": _nanmean(deltas_maj),
            "mean_rel_ball_reduction": _nanmean(rel_ball),
            "mean_delta_n_balls": _nanmean(d_balls),
            "mean_rel_proposed_reduction": _nanmean(rel_prop),
            "mean_delta_small_ball_rate": _nanmean(d_small),
            "mean_delta_purity": _nanmean(d_pur),
            "runs_min_surv_improved": n_improved,
            "n_paired_runs": len(deltas_min),
            "runs_balls_cut_over_25pct": n_cut25,
            "runs_collapse": n_collapse,
            "frac_runs_min_surv_improved": (
                n_improved / len(deltas_min) if deltas_min else float("nan")
            ),
            "frac_rejected_parent_gt4": (n_large / n_rej) if n_rej else 0.0,
        }
    return out


def _rejected_character(agg: dict[str, Any]) -> str:
    total = int(agg.get("sum_rejected", 0))
    if total == 0:
        return "no rejections"
    parts = [
        ("minority peeling via singleton children", int(agg["sum_rej_peel_singleton"])),
        ("non-minority singleton creation", int(agg["sum_rej_singleton"])),
        ("non-singleton minority peeling (n_small<=3, mostly minority)", int(agg["sum_rej_peeling"])),
        ("late small-parent splits", int(agg["sum_rej_late_parent"])),
        ("useful retained-style splits (both children n>=2, not a small minority peel)", int(agg["sum_rej_useful"])),
        ("other", int(agg["sum_rej_other"])),
    ]
    parts_sorted = sorted(parts, key=lambda x: -x[1])
    top, n_top = parts_sorted[0]
    frac = n_top / total
    large_parent = int(agg["sum_rej_parent_gt4"])
    large_frac = large_parent / total
    extra = (
        f" {large_frac:.1%} of rejected proposals have parent_n>4."
        if large_parent
        else " Almost all rejected proposals have parent_n<=4."
    )
    return f"Predominantly {top} ({frac:.1%} of {total} rejected proposals).{extra}"


def _choose_best(trade: dict[str, Any], aggs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Best granulation-only tradeoff.

    Benefit: minority survival gain.
    Cost: (i) collapse (after del_ball, n_balls<2 or a class has 0 balls),
          (ii) stopping large parents (parent_n>4),
          (iii) mean relative reduction in proposed splits / surviving balls.

    Cutting balls is a cost, not a benefit. Methods that raise surviving ball
    count by blocking del_ball deletions are not penalized for that increase.
    Pre-specified ranking, not threshold-tuned.
    """
    candidates = []
    for method in ("M1", "M2", "M3"):
        t = trade[method]
        collapse = int(t["runs_collapse"])
        large_frac = float(t["frac_rejected_parent_gt4"])
        rel_cut = float(t["mean_rel_ball_reduction"])
        rel_prop = float(t["mean_rel_proposed_reduction"])
        # Granulation halt: collapse, or any rejection of a large mixed parent.
        excessive = collapse > 0 or large_frac > 0.0
        candidates.append(
            {
                "method": method,
                "excessive": excessive,
                "collapse": collapse,
                "large_frac": large_frac,
                "delta_min": t["mean_delta_min_survival"],
                "rel_cut": rel_cut,
                "rel_prop": rel_prop,
            }
        )
    # Prefer no collapse and no large-parent halt; then higher survival; then
    # smaller proposed-split reduction (keep recursive granulation).
    ranked = sorted(
        candidates,
        key=lambda c: (
            c["collapse"],
            c["excessive"],
            -c["delta_min"],
            c["rel_prop"] if np.isfinite(c["rel_prop"]) else 1.0,
        ),
    )
    best = ranked[0]
    reasons = []
    for c in candidates:
        reasons.append(
            f"{c['method']}: d_min_surv={c['delta_min']:+.4f}, "
            f"relative ball cut={c['rel_cut']:.3f}, "
            f"relative proposed-split cut={c['rel_prop']:.3f}, "
            f"collapse_runs={c['collapse']}, "
            f"frac_rej_parent_n>4={c['large_frac']:.3f}, "
            f"{'EXCESSIVE granulation halt' if c['excessive'] else 'no large-parent halt'}."
        )
    m0_surv = aggs["M0"]["mean_minority_survival"]
    m0_balls = aggs["M0"]["mean_n_balls"]
    bm = aggs[best["method"]]
    narrative = (
        f"{best['method']} is the best granulation-only tradeoff among M1-M3: "
        f"mean minority survival {bm['mean_minority_survival']:.3f} "
        f"(M0 {m0_surv:.3f}, d={best['delta_min']:+.4f}) with mean balls "
        f"{bm['mean_n_balls']:.1f} (M0 {m0_balls:.1f}). "
        "M1/M2 can reject a peel from a large mixed parent and freeze that parent, "
        "which collapses recursive granulation even when surviving ball count rises. "
        "M3 only fires on parent_n<=4. Ranking uses survival, collapse, and whether "
        "rejections hit large parents; Topo/GBLSTSVM are unused."
    )
    return {
        "best_method": best["method"],
        "narrative": narrative,
        "comparisons": reasons,
        "rule": (
            "Penalize collapse (n_balls<2 or missing class) and rejections with "
            "parent_n>4. Then rank by minority-survival gain and by keeping "
            "proposed-split volume. Ball-count increase from fewer del_ball deletions "
            "is not treated as a cost. Safeguard thresholds were not tuned."
        ),
    }


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    proto = summary["protocol"]
    aggs = summary["by_method"]
    trade = summary["tradeoff_vs_M0"]
    choice = summary["best_tradeoff"]
    lines = [
        "# Minority safeguard counterfactual (read-only)",
        "",
        "Hypothetical split gates on frozen recursive KMeans(k=2) "
        f"(rho={proto['purity']}, min_samples_per_ball={proto['min_samples_per_ball']}, "
        f"del_ball num_data={proto['min_ball_count']}).",
        "",
        "Actual granulation, Topo, CW, and GBLSTSVM are **not** modified. "
        "Safeguards do not use classifier or topology performance.",
        "",
        "Methods:",
        "",
        "- **M0** — baseline: accept every split that meets the purity rule.",
        "- **M1** — reject if the smaller child has n<2 and contains global-minority samples.",
        "- **M2** — reject if smaller child n≤3 and minority_fraction_small_child ≥ 0.5.",
        "- **M3** — reject if parent_n≤4 and minority_fraction_small_child ≥ 0.5.",
        "",
        "A rejected proposal leaves the parent unsplit (leaf), so later children of that "
        "branch are never created. KMeans(k=2) is still called to evaluate the hypothetical split.",
        "",
        f"Runs: **{summary['n_runs_total']}** "
        f"({summary['n_dataset_seed_pairs']} dataset×seed pairs × 4 methods).",
        "",
        "## Method means (across dataset×seed)",
        "",
        "| method | balls | min survive | maj survive | min surv rate | maj surv rate | min small-ball rate | removed | mean purity | mean r | min-label balls | maj-label balls | proposed | rejected | frac rej |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        a = aggs[m]
        lines.append(
            f"| {m} | {a['mean_n_balls']:.2f} | {a['mean_n_min_survive']:.2f} | "
            f"{a['mean_n_maj_survive']:.2f} | {a['mean_minority_survival']:.3f} | "
            f"{a['mean_majority_survival']:.3f} | {a['mean_minority_small_ball_rate']:.3f} | "
            f"{a['mean_n_removed']:.2f} | {a['mean_purity']:.3f} | {a['mean_radius']:.3f} | "
            f"{a['mean_n_min_label_balls']:.2f} | {a['mean_n_maj_label_balls']:.2f} | "
            f"{a['mean_n_proposed']:.1f} | {a['mean_n_rejected']:.1f} | "
            f"{a['mean_frac_rejected']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Tradeoff vs M0 (granulation only)",
            "",
            "| method | d min survival | d maj survival | d balls | relative ball cut | relative proposed cut | d min small-ball rate | d purity | min-surv up | >=25% ball cut | collapse | frac rej parent_n>4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for m in METHODS:
        t = trade[m]
        lines.append(
            f"| {m} | {t['mean_delta_min_survival']:+.4f} | "
            f"{t['mean_delta_maj_survival']:+.4f} | "
            f"{t['mean_delta_n_balls']:+.2f} | {t['mean_rel_ball_reduction']:.3f} | "
            f"{t.get('mean_rel_proposed_reduction', 0.0):.3f} | "
            f"{t['mean_delta_small_ball_rate']:+.4f} | {t['mean_delta_purity']:+.4f} | "
            f"{t.get('runs_min_surv_improved', 0)} | "
            f"{t.get('runs_balls_cut_over_25pct', 0)} | "
            f"{t.get('runs_collapse', 0)} | "
            f"{t.get('frac_rejected_parent_gt4', 0.0):.3f} |"
        )

    lines.extend(["", "## What rejected splits look like", ""])
    for m in ("M1", "M2", "M3"):
        a = aggs[m]
        lines.extend(
            [
                f"### {m}",
                "",
                f"- {summary['rejected_character'][m]}",
                f"- Mean parent size of rejected splits: {a['mean_parent_n_rejected']:.2f} "
                f"(run-median parent_n quantiles: "
                f"p50={a['parent_n_rejected_quantiles']['p50']:.1f}).",
                f"- Mean depth of rejected splits: {a['mean_depth_rejected']:.2f} "
                f"(run-median depth p50={a['depth_rejected_quantiles']['p50']:.1f}).",
                f"- Rejected with parent_n≤4 vs >4: {a['sum_rej_parent_le4']} / "
                f"{a['sum_rej_parent_gt4']}.",
                f"- Overlap flags among rejected (means over runs with ≥1 rejection): "
                f"singleton={a['mean_frac_rej_singleton']:.3f}, "
                f"minority peel n≤3={a['mean_frac_rej_peeling']:.3f}, "
                f"late small parent={a['mean_frac_rej_late_parent']:.3f}, "
                f"useful retained-style={a['mean_frac_rej_useful']:.3f}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Verdict",
            "",
            choice["narrative"],
            "",
            "Comparisons:",
            "",
        ]
    )
    for item in choice["comparisons"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"Ranking rule: {choice['rule']}",
            "",
            "Useful retained-style splits (both children n>=2, not a small minority peel) "
            "are not the rejected mass under M1-M3. The risk is freezing a large mixed "
            "parent when the first KMeans peel is a tiny minority child. M3 cannot do that.",
            "",
            "## Per dataset × seed × method",
            "",
            "| dataset | seed | method | balls | min surv n | maj surv n | min rate | maj rate | min small | removed | purity | radius | min balls | maj balls | proposed | rejected | frac rej |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    work = df.sort_values(["dataset", "seed", "method"])
    for _, r in work.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['seed'])} | {r['method']} | "
            f"{int(r['n_balls_after_del'])} | {int(r['n_min_survive_del_ball'])} | "
            f"{int(r['n_maj_survive_del_ball'])} | {r['minority_survival_rate']:.3f} | "
            f"{r['majority_survival_rate']:.3f} | {r['minority_small_ball_rate']:.3f} | "
            f"{int(r['n_samples_removed_del_ball'])} | {r['mean_purity']:.3f} | "
            f"{r['mean_radius']:.4f} | {int(r['n_min_label_balls'])} | "
            f"{int(r['n_maj_label_balls'])} | {int(r['n_proposed_splits'])} | "
            f"{int(r['n_rejected_splits'])} | {r['frac_proposed_rejected']:.3f} |"
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


def _write_outputs(
    df: pd.DataFrame,
    failed: list[dict[str, str]],
    rejected_parent_n: dict[str, list[int]],
    rejected_depth: dict[str, list[int]],
) -> dict[str, Any]:
    aggs = {m: _method_aggregate(df, m) for m in METHODS}
    trade = _tradeoff(df)
    rejected_character = {m: _rejected_character(aggs[m]) for m in METHODS}
    choice = _choose_best(trade, aggs)
    pooled_parent = {
        m: _quantiles(np.array(rejected_parent_n[m], dtype=float)) for m in METHODS
    }
    pooled_depth = {
        m: _quantiles(np.array(rejected_depth[m], dtype=float)) for m in METHODS
    }
    summary: dict[str, Any] = {
        "protocol": {
            "read_only": True,
            "purity": PURITY,
            "min_samples_per_ball": MIN_SAMPLES_PER_BALL,
            "min_ball_count": MIN_BALL_COUNT,
            "small_ball_max": SMALL_BALL_MAX,
            "seeds": SEEDS,
            "datasets": sorted(df["dataset"].unique().tolist()) if len(df) else [],
            "failed": failed,
            "kmeans": "GranularBall.split_2balls / sklearn KMeans(n_clusters=2), np.random.seed(seed)",
            "global_minority": "rarer class on the training split after normalize_labels",
            "minority_small_ball_rate": (
                "fraction of global-minority training samples whose leaf has n<=5 before del_ball"
            ),
            "survival": "fraction of class samples whose leaf has n >= min_ball_count",
            "methods": {
                "M0": "accept every purity-eligible split",
                "M1": "reject if n_small<2 and small child contains global minority",
                "M2": "reject if n_small<=3 and minority_fraction_small_child>=0.5",
                "M3": "reject if parent_n<=4 and minority_fraction_small_child>=0.5",
            },
            "unused": ["Topo", "CW", "GBLSTSVM", "classifier metrics"],
            "no_threshold_tuning": True,
        },
        "n_runs_total": int(len(df)),
        "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups) if len(df) else 0,
        "by_method": aggs,
        "tradeoff_vs_M0": trade,
        "rejected_character": rejected_character,
        "pooled_rejected_parent_n": pooled_parent,
        "pooled_rejected_depth": pooled_depth,
        "best_tradeoff": choice,
        "definitions": {
            "singleton_creation": "n_small==1",
            "minority_peeling": "small child contains global minority and n_small<=3",
            "late_small_parent": "parent_n<=4",
            "useful_retained": (
                "both children n>=2 and the small child is not a n<=3 majority-minority peel"
            ),
            "primary_kind": (
                "mutually exclusive: minority_peeling_singleton, singleton_creation, "
                "minority_peeling, late_small_parent, useful_retained, other"
            ),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary, failed=failed)
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _write_outputs(
            df,
            failed=[],
            rejected_parent_n={m: [] for m in METHODS},
            rejected_depth={m: [] for m in METHODS},
        )
        print(f"Wrote {OUT_SUMMARY}")
        print(f"Wrote {OUT_REPORT}")
        print("Best tradeoff:", summary["best_tradeoff"]["best_method"])
        print(summary["best_tradeoff"]["narrative"].encode("ascii", "replace").decode("ascii"))
        return 0

    descriptors = discover_available_datasets()
    run_rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    rejected_parent_n: dict[str, list[int]] = {m: [] for m in METHODS}
    rejected_depth: dict[str, list[int]] = {m: [] for m in METHODS}

    for desc in descriptors:
        for seed in SEEDS:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append(
                    {"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue
            y_train = normalize_labels(split.y_train)
            n_pos = int(np.sum(y_train == 1.0))
            n_neg = int(np.sum(y_train == -1.0))
            global_min = 1.0 if n_pos <= n_neg else -1.0
            for method in METHODS:
                run, proposed = replay_method(
                    split.X_train,
                    y_train,
                    seed=seed,
                    dataset=split.name,
                    global_min_label=global_min,
                    method=method,
                )
                run_rows.append(run)
                for p in proposed:
                    if p["rejected"]:
                        rejected_parent_n[method].append(int(p["parent_n"]))
                        rejected_depth[method].append(int(p["parent_depth"]))
                print(
                    f"{split.name} seed={seed} {method}: "
                    f"balls={run['n_balls_after_del']} "
                    f"min_surv={run['minority_survival_rate']:.3f} "
                    f"rej={run['n_rejected_splits']}/{run['n_proposed_splits']}"
                )

    df = pd.DataFrame(run_rows)
    df.to_csv(OUT_CSV, index=False)
    summary = _write_outputs(df, failed, rejected_parent_n, rejected_depth)
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print("Best tradeoff:", summary["best_tradeoff"]["best_method"])
    print(summary["best_tradeoff"]["narrative"].encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
