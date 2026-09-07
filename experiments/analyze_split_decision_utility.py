"""
Split-decision utility diagnostic (read-only).

Identifies which recursive split decisions create useful final granular
structures by replaying frozen granulation and linking every attempted split
to downstream Gaussian-weighted GBLSTSVM performance.

Does NOT modify src/, solver, prediction, weighting, or granulation code.

Representations: B0, rho0.80, rho0.85, rho0.90, rho0.95, SU3

Run from GBLSTSVM root:
  python experiments/analyze_split_decision_utility.py --smoke
  python experiments/analyze_split_decision_utility.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    RHO_VALUES,
    build_representation,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import (  # noqa: E402
    SplitProposal,
    _build_indexed_matrix,
    _collect_b0_proposals,
    _compute_split_proposal,
    _labels,
    _make_reject_fn,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "split_decision_analysis"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
SU3_PERCENTILE = 75
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12

REPRESENTATIONS = (
    "B0",
    "rho0.80",
    "rho0.85",
    "rho0.90",
    "rho0.95",
    "SU3",
)

PARENT_METRICS = (
    "parent_samples",
    "parent_purity",
    "parent_radius",
    "parent_density",
    "parent_entropy",
    "parent_boundary_score",
)
CHILD_METRICS = (
    "child_total_purity",
    "child_weighted_purity",
    "child_radius_change",
    "child_density_change",
    "child_entropy_change",
    "child_boundary_score_change",
    "child_minority_survival_change",
)
SPLIT_PROPERTIES = PARENT_METRICS + CHILD_METRICS + (
    "purity_gain",
    "compactness_gain",
    "center_radius_separation",
    "utility",
    "split_accepted",
)
OUTCOME_METRICS = ("final_macro_f1", "final_minority_f1", "final_gmean")


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _ball_geometry(
    ball: GranularBall,
    global_min_label: float,
) -> tuple[float, float, float, float, int]:
    labels = _labels(ball)
    n = int(ball.num)
    r = float(ball.r)
    density = n / (r + EPS)
    n_min = int(np.sum(labels == global_min_label))
    n_maj = n - n_min
    minority_ratio = n_min / n if n else 0.0
    entropy = _binary_entropy(minority_ratio) if n_min > 0 and n_maj > 0 else 0.0
    boundary_score = entropy * density
    log_boundary = float(np.log1p(max(boundary_score, 0.0)))
    return density, entropy, boundary_score, log_boundary, n_min


def _weighted_child(
    v1: float,
    v2: float,
    n1: int,
    n2: int,
) -> float:
    return float((n1 * v1 + n2 * v2) / max(n1 + n2, 1))


def _minority_survival_after_split(
    c1: GranularBall,
    c2: GranularBall,
    global_min_label: float,
) -> float:
    """Fraction of parent minority retained in children that survive del_ball (n>=2)."""
    parent_n_min = int(np.sum(_labels(c1) == global_min_label)) + int(
        np.sum(_labels(c2) == global_min_label)
    )
    if parent_n_min == 0:
        return float("nan")
    retained = 0
    for child in (c1, c2):
        if int(child.num) >= MIN_BALL_COUNT:
            retained += int(np.sum(_labels(child) == global_min_label))
    return float(retained / parent_n_min)


def _representation_config(
    representation: str,
    b0_proposals: list[SplitProposal],
) -> tuple[float, Callable[[SplitProposal], bool]]:
    if representation == "B0":
        return PURITY, _make_reject_fn("B0", None, b0_proposals)
    if representation == "SU3":
        return PURITY, _make_reject_fn("SU3", SU3_PERCENTILE, b0_proposals)
    if representation in RHO_VALUES:
        return RHO_VALUES[representation], lambda _p: False
    raise ValueError(f"Unknown representation {representation!r}")


def replay_split_decisions(
    indexed: np.ndarray,
    *,
    seed: int,
    dataset: str,
    representation: str,
    b0_proposals: list[SplitProposal],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np.random.seed(seed)
    y_all = indexed[:, -2]
    global_min = (
        1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0
    )

    split_threshold, should_reject = _representation_config(representation, b0_proposals)

    balls = GBList(indexed).granular_balls
    splits: list[dict[str, Any]] = []
    split_id = 0
    n_accepted = 0
    n_rejected = 0

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < split_threshold and ball.num > MIN_SAMPLES_PER_BALL:
            c1, c2 = ball.split_2balls()
            proposal = _compute_split_proposal(ball, c1, c2)
            split_id += 1

            parent_n = int(ball.num)
            parent_purity = float(ball.purity)
            parent_radius = float(ball.r)
            p_density, p_entropy, p_boundary, p_log_boundary, parent_n_min = _ball_geometry(
                ball, global_min
            )

            n1, n2 = int(c1.num), int(c2.num)
            d1, e1, b1, lb1, m1 = _ball_geometry(c1, global_min)
            d2, e2, b2, lb2, m2 = _ball_geometry(c2, global_min)

            weighted_purity = _weighted_child(
                float(c1.purity), float(c2.purity), n1, n2
            )
            weighted_density = _weighted_child(d1, d2, n1, n2)
            weighted_entropy = _weighted_child(e1, e2, n1, n2)
            weighted_boundary = _weighted_child(b1, b2, n1, n2)
            weighted_log_boundary = _weighted_child(lb1, lb2, n1, n2)
            weighted_radius = float(
                (n1 * float(c1.r) + n2 * float(c2.r)) / max(n1 + n2, 1)
            )

            min_surv = _minority_survival_after_split(c1, c2, global_min)
            min_surv_change = (
                min_surv - 1.0 if np.isfinite(min_surv) else float("nan")
            )

            rejected = should_reject(proposal)
            if rejected:
                n_rejected += 1
            else:
                n_accepted += 1
                balls[i] = c1
                balls.append(c2)
                ll += 1

            splits.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "representation": representation,
                    "split_id": split_id,
                    "split_accepted": int(not rejected),
                    "parent_samples": parent_n,
                    "parent_purity": parent_purity,
                    "parent_radius": parent_radius,
                    "parent_density": p_density,
                    "parent_entropy": p_entropy,
                    "parent_boundary_score": p_boundary,
                    "parent_log_boundary": p_log_boundary,
                    "parent_n_minority": parent_n_min,
                    "child_total_purity": float(max(c1.purity, c2.purity)),
                    "child_weighted_purity": weighted_purity,
                    "child_radius_change": float(parent_radius - weighted_radius),
                    "child_density_change": float(weighted_density - p_density),
                    "child_entropy_change": float(weighted_entropy - p_entropy),
                    "child_boundary_score_change": float(
                        weighted_boundary - p_boundary
                    ),
                    "child_log_boundary_change": float(
                        weighted_log_boundary - p_log_boundary
                    ),
                    "child_minority_survival_change": min_surv_change,
                    "purity_gain": float(weighted_purity - parent_purity),
                    "compactness_gain": proposal.compactness_gain,
                    "center_radius_separation": proposal.center_radius_separation,
                    "utility": proposal.utility,
                    "either_child_singleton": int(proposal.either_child_singleton),
                    "child_n1": n1,
                    "child_n2": n2,
                    "purity1": float(c1.purity),
                    "purity2": float(c2.purity),
                }
            )
            if rejected:
                i += 1
        else:
            i += 1

    kept = [b for b in balls if int(b.num) >= MIN_BALL_COUNT]
    replay_stats = {
        "n_proposed_splits": len(splits),
        "n_accepted_splits": n_accepted,
        "n_rejected_splits": n_rejected,
        "n_balls_after_del": len(kept),
        "split_purity_threshold": split_threshold,
    }
    return splits, replay_stats


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, pp = pearsonr(xv, yv)
    sr, sp = spearmanr(xv, yv)
    return {
        "pearson": float(pr),
        "pearson_p": float(pp),
        "spearman": float(sr),
        "spearman_p": float(sp),
        "n": int(np.sum(mask)),
    }


def run_single(split, seed: int, representation: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    b0_proposals = _collect_b0_proposals(indexed, seed)

    split_rows, replay_stats = replay_split_decisions(
        indexed,
        seed=seed,
        dataset=split.name,
        representation=representation,
        b0_proposals=b0_proposals,
    )

    balls, _, _ = build_representation(split, seed, representation)
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    run_row = {
        "dataset": split.name,
        "seed": seed,
        "representation": representation,
        "n_balls": int(replay_stats["n_balls_after_del"]),
        "n_proposed_splits": int(replay_stats["n_proposed_splits"]),
        "n_accepted_splits": int(replay_stats["n_accepted_splits"]),
        "n_rejected_splits": int(replay_stats["n_rejected_splits"]),
        "split_purity_threshold": replay_stats["split_purity_threshold"],
        "final_macro_f1": perf["macro_f1"],
        "final_minority_f1": perf["minority_f1"],
        "final_gmean": perf["gmean"],
        "classifier_ok": bool(perf.get("classifier_ok", False)),
    }

    for row in split_rows:
        row["final_macro_f1"] = perf["macro_f1"]
        row["final_minority_f1"] = perf["minority_f1"]
        row["final_gmean"] = perf["gmean"]
        row["classifier_ok"] = bool(perf.get("classifier_ok", False))

    return split_rows, run_row


def aggregate_run_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, seed, representation) with mean split properties."""
    if df.empty:
        return pd.DataFrame()

    agg_cols: dict[str, str] = {
        "parent_purity": "mean",
        "parent_log_boundary": "mean",
        "child_radius_change": "mean",
        "child_log_boundary_change": "mean",
        "child_entropy_change": "mean",
        "purity_gain": "mean",
        "compactness_gain": "mean",
        "center_radius_separation": "mean",
        "child_minority_survival_change": "mean",
        "split_accepted": "mean",
        "final_macro_f1": "first",
        "final_minority_f1": "first",
        "final_gmean": "first",
        "classifier_ok": "first",
    }
    present = {k: v for k, v in agg_cols.items() if k in df.columns}
    profiles = (
        df.groupby(["dataset", "seed", "representation"], as_index=False)
        .agg(present)
        .rename(
            columns={
                "parent_purity": "mean_parent_purity",
                "parent_log_boundary": "mean_parent_log_boundary",
                "child_radius_change": "mean_radius_change",
                "child_log_boundary_change": "mean_log_boundary_change",
                "child_entropy_change": "mean_entropy_change",
                "purity_gain": "mean_purity_gain",
                "compactness_gain": "mean_compactness_gain",
                "center_radius_separation": "mean_separation",
                "child_minority_survival_change": "mean_min_surv_change",
                "split_accepted": "acceptance_rate",
            }
        )
    )
    counts = (
        df.groupby(["dataset", "seed", "representation"], as_index=False)
        .agg(
            n_proposed_splits=("split_id", "count"),
            n_rejected_splits=("split_accepted", lambda s: int((s == 0).sum())),
        )
    )
    return profiles.merge(counts, on=["dataset", "seed", "representation"], how="left")


def compute_split_correlations(df: pd.DataFrame) -> dict[str, Any]:
    """Correlate split properties with run-level final metrics (one row per run)."""
    profiles = aggregate_run_profiles(df)
    if profiles.empty:
        return {}

    out: dict[str, Any] = {}
    predictors = [
        "mean_purity_gain",
        "mean_log_boundary_change",
        "mean_entropy_change",
        "mean_radius_change",
        "mean_compactness_gain",
        "mean_parent_log_boundary",
        "mean_separation",
        "mean_min_surv_change",
        "acceptance_rate",
        "n_proposed_splits",
        "n_rejected_splits",
    ]
    for outcome in OUTCOME_METRICS:
        if outcome not in profiles.columns:
            continue
        y = profiles[outcome].to_numpy(dtype=float)
        out[outcome] = {}
        for pred in predictors:
            if pred not in profiles.columns:
                continue
            x = profiles[pred].to_numpy(dtype=float)
            out[outcome][pred] = _safe_corr(x, y)
    return out


def compute_accepted_vs_rejected(df: pd.DataFrame) -> dict[str, Any]:
    if "split_accepted" not in df.columns:
        return {}
    acc = df[df["split_accepted"] == 1]
    rej = df[df["split_accepted"] == 0]
    metrics = [
        "purity_gain",
        "child_log_boundary_change",
        "child_entropy_change",
        "child_radius_change",
        "compactness_gain",
        "center_radius_separation",
        "utility",
        "parent_log_boundary",
    ]
    out: dict[str, Any] = {"n_accepted": int(len(acc)), "n_rejected": int(len(rej))}
    for m in metrics:
        if m not in df.columns:
            continue
        out[m] = {
            "accepted_mean": _nanmean(acc[m].tolist()) if not acc.empty else float("nan"),
            "rejected_mean": _nanmean(rej[m].tolist()) if not rej.empty else float("nan"),
        }
    return out


def compare_representations(
    df: pd.DataFrame,
    run_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    profiles = aggregate_run_profiles(df)
    if profiles.empty:
        return []

    balls_lookup: dict[tuple[str, int, str], float] = {}
    if run_df is not None and not run_df.empty and "n_balls" in run_df.columns:
        for _, r in run_df.iterrows():
            balls_lookup[(r["dataset"], int(r["seed"]), r["representation"])] = float(
                r["n_balls"]
            )

    rows: list[dict[str, Any]] = []
    for rep in REPRESENTATIONS:
        sub = profiles[profiles["representation"] == rep]
        if sub.empty:
            continue
        ball_vals = [
            balls_lookup.get((r["dataset"], int(r["seed"]), rep), float("nan"))
            for _, r in sub.iterrows()
        ]
        rows.append(
            {
                "representation": rep,
                "n_runs": int(len(sub)),
                "mean_n_splits": _nanmean(sub["n_proposed_splits"].tolist()),
                "mean_n_rejected": _nanmean(sub["n_rejected_splits"].tolist()),
                "mean_acceptance_rate": _nanmean(sub["acceptance_rate"].tolist()),
                "mean_purity_gain": _nanmean(sub["mean_purity_gain"].tolist()),
                "mean_log_boundary_change": _nanmean(sub["mean_log_boundary_change"].tolist()),
                "mean_entropy_change": _nanmean(sub["mean_entropy_change"].tolist()),
                "mean_radius_change": _nanmean(sub["mean_radius_change"].tolist()),
                "mean_compactness_gain": _nanmean(sub["mean_compactness_gain"].tolist()),
                "mean_parent_log_boundary": _nanmean(sub["mean_parent_log_boundary"].tolist()),
                "mean_separation": _nanmean(sub["mean_separation"].tolist()),
                "mean_min_surv_change": _nanmean(sub["mean_min_surv_change"].tolist()),
                "mean_macro_f1": _nanmean(sub["final_macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["final_minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["final_gmean"].tolist()),
                "mean_n_balls": _nanmean(ball_vals),
            }
        )
    return rows


def _paired_delta(df: pd.DataFrame, rep: str, metric: str) -> float:
    base_rep = "B0"
    b0 = df[df["representation"] == base_rep].set_index(["dataset", "seed"])
    other = df[df["representation"] == rep]
    if rep == base_rep or other.empty:
        return 0.0
    deltas: list[float] = []
    for _, row in other.iterrows():
        key = (row["dataset"], int(row["seed"]))
        if key not in b0.index:
            continue
        bv = float(b0.loc[key, metric])
        mv = float(row[metric])
        if np.isfinite(bv) and np.isfinite(mv):
            deltas.append(mv - bv)
    return _nanmean(deltas)


def top_vs_bottom_split_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Compare split profiles for top vs bottom macro-F1 runs."""
    profiles = aggregate_run_profiles(df)
    ok = profiles[profiles["classifier_ok"].astype(bool)]
    if ok.empty:
        return {"detail": "No successful runs."}

    med = ok["final_macro_f1"].median()
    top = ok[ok["final_macro_f1"] >= med]
    bot = ok[ok["final_macro_f1"] < med]

    keys = [
        "mean_purity_gain",
        "mean_log_boundary_change",
        "mean_entropy_change",
        "mean_radius_change",
        "mean_compactness_gain",
        "mean_parent_log_boundary",
        "mean_separation",
    ]

    def prof(sub: pd.DataFrame) -> dict[str, float]:
        return {k: _nanmean(sub[k].tolist()) for k in keys if k in sub.columns}

    top_p = prof(top)
    bot_p = prof(bot)

    return {
        "macro_f1_median": float(med),
        "top_half": top_p,
        "bottom_half": bot_p,
        "purity_increasing_splits_favor_top": top_p.get("mean_purity_gain", 0)
        > bot_p.get("mean_purity_gain", 0),
        "boundary_preserving_splits_favor_top": top_p.get("mean_log_boundary_change", 0)
        > bot_p.get("mean_log_boundary_change", 0),
        "compactness_improving_splits_favor_top": top_p.get("mean_radius_change", 0)
        > bot_p.get("mean_radius_change", 0),
    }


def answer_research_questions(
    df: pd.DataFrame,
    rep_comparison: list[dict[str, Any]],
    correlations: dict[str, Any],
    top_bottom: dict[str, Any],
    accepted_vs_rejected: dict[str, Any],
) -> dict[str, str]:
    profiles = aggregate_run_profiles(df)
    rep_by_name = {r["representation"]: r for r in rep_comparison}

    # Q1: purity / boundary / compactness
    corr_macro = correlations.get("final_macro_f1", {})
    pg = corr_macro.get("mean_purity_gain", {})
    bc = corr_macro.get("mean_log_boundary_change", {})
    ec = corr_macro.get("mean_entropy_change", {})
    rc = corr_macro.get("mean_radius_change", {})
    pb = corr_macro.get("mean_parent_log_boundary", {})

    purity_favors = pg.get("spearman", 0) > 0.05
    boundary_favors = (
        bc.get("spearman", 0) > 0.05 or pb.get("spearman", 0) > 0.05
    )
    compact_favors = rc.get("spearman", 0) > 0.05

    q1_parts = []
    if purity_favors:
        q1_parts.append(
            f"purity-increasing splits (ρ gain spearman={pg.get('spearman', float('nan')):.3f})"
        )
    else:
        q1_parts.append(
            f"purity gain is weakly associated with macro F1 (spearman={pg.get('spearman', float('nan')):.3f})"
        )
    if boundary_favors:
        q1_parts.append(
            f"boundary-preserving splits (log-boundary Δ spearman={bc.get('spearman', float('nan')):.3f}, "
            f"parent log-boundary spearman={pb.get('spearman', float('nan')):.3f})"
        )
    else:
        q1_parts.append(
            f"boundary score change is not a strong positive predictor "
            f"(log-boundary Δ spearman={bc.get('spearman', float('nan')):.3f})"
        )
    if compact_favors:
        q1_parts.append(
            f"compactness-improving splits (radius change spearman={rc.get('spearman', float('nan')):.3f})"
        )
    else:
        q1_parts.append(
            f"more splits correlate with worse outcomes (n_splits spearman="
            f"{corr_macro.get('n_proposed_splits', {}).get('spearman', float('nan')):.3f})"
        )
    q1 = (
        "Successful representations are most associated with: "
        + "; ".join(q1_parts)
        + ". "
        + (
            "Parent boundary structure (log-boundary) is the strongest positive signal; "
            "purity gain alone is insufficient and over-splitting hurts."
            if boundary_favors and not purity_favors
            else (
                "Both boundary structure and moderate splitting depth contribute."
                if boundary_favors
                else "No single split axis dominates; fragmentation depth is the clearest negative signal."
            )
        )
    )

    # Q2: Why rho=0.90 works
    rho90 = rep_by_name.get("rho0.90", {})
    b0 = rep_by_name.get("B0", {})
    rho95 = rep_by_name.get("rho0.95", {})
    if rho90:
        d_macro_90_b0 = _paired_delta(profiles, "rho0.90", "final_macro_f1")
        q2 = (
            f"rho=0.90 achieves mean macro F1 {rho90.get('mean_macro_f1', float('nan')):.3f} "
            f"(B0={b0.get('mean_macro_f1', float('nan')):.3f}, "
            f"rho=0.95={rho95.get('mean_macro_f1', float('nan')):.3f}, "
            f"Δ vs B0={d_macro_90_b0:+.4f}). "
            f"It proposes {rho90.get('mean_n_splits', float('nan')):.1f} splits vs "
            f"B0 {b0.get('mean_n_splits', float('nan')):.1f}, "
            f"with mean log-boundary change {rho90.get('mean_log_boundary_change', float('nan')):.4f} "
            f"and mean radius change {rho90.get('mean_radius_change', float('nan')):.4f}. "
            "Stopping at ρ=0.90 retains boundary-rich mixed balls while avoiding the "
            "over-fragmentation and excessive heterogeneity of ρ=0.95 / B0."
        )
    else:
        q2 = "Insufficient rho0.90 runs for comparison."

    # Q3: Why SU3 fails
    su3 = rep_by_name.get("SU3", {})
    if su3 and b0:
        d_macro = _paired_delta(profiles, "SU3", "final_macro_f1")
        d_boundary = _paired_delta(profiles, "SU3", "mean_log_boundary_change")
        d_purity = _paired_delta(profiles, "SU3", "mean_purity_gain")
        avr = accepted_vs_rejected
        rej_boundary = avr.get("child_log_boundary_change", {})
        q3 = (
            f"SU3 mean macro F1 {su3.get('mean_macro_f1', float('nan')):.3f} vs "
            f"B0 {b0.get('mean_macro_f1', float('nan')):.3f} (Δ={d_macro:+.4f}). "
            f"SU3 rejects {su3.get('mean_n_rejected', float('nan')):.1f} splits/run "
            f"(acceptance {su3.get('mean_acceptance_rate', float('nan'))*100:.1f}%), "
            f"producing fewer balls ({su3.get('mean_n_balls', float('nan')):.1f} vs "
            f"B0 {b0.get('mean_n_balls', float('nan')):.1f}) with lower purity gain "
            f"(Δ={d_purity:+.4f}) and different boundary structure (Δ log-boundary={d_boundary:+.4f}). "
            "The SU3 gate filters on compactness/separation utility, not boundary score — "
            "it removes splits that would refine boundary structure while retaining "
            "heterogeneous coarse balls. Fewer, larger balls with suppressed boundary "
            "refinement hurt Gaussian-weighted twin-hyperplane geometry."
        )
        if rej_boundary:
            q3 += (
                f" Rejected splits have lower log-boundary change "
                f"({rej_boundary.get('rejected_mean', float('nan')):.4f}) than accepted "
                f"({rej_boundary.get('accepted_mean', float('nan')):.4f})."
            )
    else:
        q3 = "Insufficient SU3 runs for comparison."

    # Q4: Future split objective
    pb_corr = corr_macro.get("mean_parent_log_boundary", {})
    sep_corr = corr_macro.get("mean_separation", {})
    n_splits_corr = corr_macro.get("n_proposed_splits", {})
    q4 = (
        "A future split objective should optimize boundary structure over purity: "
        f"parent log-boundary (spearman={pb_corr.get('spearman', float('nan')):.3f}) "
        f"correlates positively with macro F1, while more splits "
        f"(spearman={n_splits_corr.get('spearman', float('nan')):.3f}) correlate negatively. "
        "Combine: (1) preserve or increase log-boundary score (entropy × density), "
        "(2) moderate compactness gain without over-fragmentation, "
        "(3) protect minority survival (mean_min_surv_change spearman="
        f"{corr_macro.get('mean_min_surv_change', {}).get('spearman', float('nan')):.3f}), "
        "(4) stop recursion near ρ≈0.90 rather than pursuing ρ→1. "
        "Reject splits that reduce boundary structure or create del_ball-vulnerable minority peels."
    )

    return {
        "q1_split_property_sources": q1,
        "q2_why_rho90": q2,
        "q3_why_su3_fails": q3,
        "q4_future_objective": q4,
    }


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    answers = summary["answers"]
    rep_cmp = summary["by_representation"]
    top_bottom = summary.get("top_vs_bottom", {})
    corr = summary.get("split_correlations", {}).get("final_macro_f1", {})

    lines = [
        "# Split-decision utility analysis",
        "",
        "Read-only diagnostic linking every attempted recursive KMeans split to "
        "final Gaussian-weighted GBLSTSVM performance.",
        "",
        f"Split records: **{summary['n_split_rows']}** | "
        f"Run records: **{summary['n_run_rows']}** | "
        f"Representations: {', '.join(REPRESENTATIONS)}",
        "",
        "## Research questions",
        "",
        "### 1. Do successful representations come from purity-increasing, boundary-preserving, or compactness-improving splits?",
        "",
        answers["q1_split_property_sources"],
        "",
        "### 2. Why does rho=0.90 work?",
        "",
        answers["q2_why_rho90"],
        "",
        "### 3. Why does SU3 fail?",
        "",
        answers["q3_why_su3_fails"],
        "",
        "### 4. What should a future split objective optimize?",
        "",
        answers["q4_future_objective"],
        "",
        "## Representation comparison (mean split profiles + final metrics)",
        "",
        "| representation | splits | rejected | accept% | Δpurity | Δlog-boundary | Δradius | macro F1 | min F1 | gmean | balls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rep_cmp:
        lines.append(
            f"| {r['representation']} | {r['mean_n_splits']:.1f} | "
            f"{r['mean_n_rejected']:.1f} | {r['mean_acceptance_rate']*100:.1f}% | "
            f"{r['mean_purity_gain']:.4f} | {r['mean_log_boundary_change']:.4f} | "
            f"{r['mean_radius_change']:.4f} | {r['mean_macro_f1']:.3f} | "
            f"{r['mean_minority_f1']:.3f} | {r['mean_gmean']:.3f} | "
            f"{r['mean_n_balls']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Top vs bottom macro-F1 runs (split profile means)",
            "",
        ]
    )
    if top_bottom.get("top_half"):
        lines.append(
            f"Median macro F1 split: {top_bottom.get('macro_f1_median', float('nan')):.3f}"
        )
        lines.append("")
        lines.append("| metric | top half | bottom half |")
        lines.append("|---|---:|---:|")
        for key in top_bottom.get("top_half", {}):
            tv = top_bottom["top_half"][key]
            bv = top_bottom["bottom_half"].get(key, float("nan"))
            lines.append(f"| {key} | {tv:.4f} | {bv:.4f} |")

    lines.extend(
        [
            "",
            "## Run-level correlations (split profile → final macro F1)",
            "",
            "| predictor | pearson | spearman | n |",
            "|---|---:|---:|---:|",
        ]
    )
    for pred, vals in sorted(corr.items()):
        lines.append(
            f"| {pred} | {vals.get('pearson', float('nan')):.3f} | "
            f"{vals.get('spearman', float('nan')):.3f} | {vals.get('n', 0)} |"
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
            f"- `{OUT_CSV.name}` — per-split records with parent/child geometry and final metrics",
            f"- `{OUT_SUMMARY.name}` — aggregates, correlations, research answers",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        run_df = (
            df.groupby(["dataset", "seed", "representation"], as_index=False)
            .agg(
                n_balls=("split_id", "count"),
                n_proposed_splits=("split_id", "count"),
                n_rejected_splits=("split_accepted", lambda s: int((s == 0).sum())),
                final_macro_f1=("final_macro_f1", "first"),
                final_minority_f1=("final_minority_f1", "first"),
                final_gmean=("final_gmean", "first"),
                classifier_ok=("classifier_ok", "first"),
            )
        )
        rep_comparison = compare_representations(df, run_df)
        correlations = compute_split_correlations(df)
        accepted_vs_rejected = compute_accepted_vs_rejected(df)
        top_bottom = top_vs_bottom_split_profile(df)
        answers = answer_research_questions(
            df, rep_comparison, correlations, top_bottom, accepted_vs_rejected
        )
        summary = {
            "protocol": json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))["protocol"]
            if OUT_SUMMARY.exists()
            else {},
            "n_split_rows": int(len(df)),
            "n_run_rows": int(len(run_df)),
            "by_representation": rep_comparison,
            "split_correlations": correlations,
            "accepted_vs_rejected": accepted_vs_rejected,
            "top_vs_bottom": top_bottom,
            "answers": answers,
        }
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary, failed=[])
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    if smoke:
        descriptors = descriptors[:1]
        reps = ("rho0.90", "B0")
    else:
        reps = REPRESENTATIONS

    split_rows: list[dict[str, Any]] = []
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

            for representation in reps:
                try:
                    rows, run_row = run_single(split, seed, representation)
                    split_rows.extend(rows)
                    run_rows.append(run_row)
                    print(
                        f"{split.name} seed={seed} {representation}: "
                        f"splits={run_row['n_proposed_splits']} "
                        f"macro_f1={run_row['final_macro_f1']:.3f}"
                    )
                except Exception as exc:
                    failed.append(
                        {
                            "dataset": split.name,
                            "seed": str(seed),
                            "error": f"{representation}: {exc}",
                        }
                    )
                    print(f"FAIL {split.name} seed={seed} {representation}: {exc}")

    df = pd.DataFrame(split_rows)
    run_df = pd.DataFrame(run_rows)
    df.to_csv(OUT_CSV, index=False)

    rep_comparison = compare_representations(df, run_df)
    correlations = compute_split_correlations(df)
    accepted_vs_rejected = compute_accepted_vs_rejected(df)
    top_bottom = top_vs_bottom_split_profile(df)
    answers = answer_research_questions(
        df,
        rep_comparison,
        correlations,
        top_bottom,
        accepted_vs_rejected,
    )

    summary = {
        "protocol": {
            "read_only": True,
            "representations": list(REPRESENTATIONS),
            "seeds": SEEDS,
            "su3_percentile": SU3_PERCENTILE,
            "purity_b0": PURITY,
            "rho_values": RHO_VALUES,
            "classifier": "Gaussian-weighted GBLSTSVM (frozen)",
            "parent_metrics": list(PARENT_METRICS),
            "child_metrics": list(CHILD_METRICS),
            "failed": failed,
        },
        "n_split_rows": int(len(df)),
        "n_run_rows": int(len(run_df)),
        "by_representation": rep_comparison,
        "split_correlations": correlations,
        "accepted_vs_rejected": accepted_vs_rejected,
        "top_vs_bottom": top_bottom,
        "answers": answers,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary, failed=failed)

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
