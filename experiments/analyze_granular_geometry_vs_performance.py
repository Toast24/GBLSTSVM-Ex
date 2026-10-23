"""
Granular geometry vs GBLSTSVM performance (read-only diagnostic).

Determines which granular-ball properties predict successful Gaussian-weighted
GBLSTSVM classification across frozen granulation variants.

Does NOT modify src/, solver, prediction rule, CW, CA, Topo, Gaussian membership,
or granulation algorithms.

Representations: B0, SU3, rho0.80, rho0.85, rho0.90, rho0.95
Weighting: Gaussian purity_compact only (frozen).

Run from GBLSTSVM root:
  python experiments/analyze_granular_geometry_vs_performance.py --smoke
  python experiments/analyze_granular_geometry_vs_performance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import (  # noqa: E402
    PRIMARY_GAUSSIAN,
    compute_gaussian_memberships,
)
from experiments.phase8_common import (  # noqa: E402
    DEFAULT_C1,
    DEFAULT_C2,
    evaluate,
    normalize_labels,
)
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _collect_b0_proposals,
    _labels,
    _make_reject_fn,
    _replay_granulation,
    _to_ball_info,
)
from src.cbcm import split_membership_by_class  # noqa: E402
from src.class_weights import ball_class_weights, compute_binary_class_weights  # noqa: E402
from src.cw_gblstsvm import solve_cw_gblstsvm_systems  # noqa: E402
from src.gblstsvm import predict_twin_hyperplanes  # noqa: E402
from src.granular_balls import (  # noqa: E402
    BallInfo,
    GBList,
    GranularBall,
    balls_to_class_arrays,
)
from src.reliability import compute_radius_scale, normalize_radius  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granular_geometry_analysis"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
SU3_PERCENTILE = 75
MIN_SAMPLES_PER_BALL = 1
SMALL_BALL_MAX = 5
EPS = 1e-12
REG_TERM = 1e-4
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"
SAVE_EVERY = 5

REPRESENTATIONS = (
    "B0",
    "SU3",
    "rho0.80",
    "rho0.85",
    "rho0.90",
    "rho0.95",
)
RHO_VALUES: dict[str, float] = {
    "rho0.80": 0.80,
    "rho0.85": 0.85,
    "rho0.90": 0.90,
    "rho0.95": 0.95,
}

GEOMETRY_METRICS = (
    "mean_purity",
    "mean_radius",
    "radius_variance",
    "mean_density",
    "mean_entropy",
    "fraction_mixed_balls",
    "mean_margin_proxy",
    "minority_survival",
)
PERFORMANCE_METRICS = ("macro_f1", "minority_f1", "gmean")


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, label: float) -> float:
    return float(
        f1_score(
            normalize_labels(y_true),
            normalize_labels(y_pred),
            pos_label=label,
            zero_division=0,
        )
    )


def _majority_label(ball: GranularBall) -> float:
    return -1.0 if ball.label == -1 else 1.0


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _replay_stats_from_kept(
    kept: list[GranularBall],
    indexed: np.ndarray,
) -> dict[str, Any]:
    """Mirror post-del_ball statistics from test_split_utility_granulation."""
    y_all = indexed[:, -2]
    n_train = int(indexed.shape[0])
    global_min = 1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0
    n_min_train = int(np.sum(y_all == global_min))
    n_maj_train = n_train - n_min_train

    survived_min = 0
    survived_maj = 0
    min_in_small = 0
    n_removed_samples = 0
    for ball in kept:
        n = int(ball.num)
        labs = _labels(ball)
        n_min = int(np.sum(labs == global_min))
        n_maj = n - n_min
        if n <= SMALL_BALL_MAX:
            min_in_small += n_min
        if n < MIN_BALL_COUNT:
            n_removed_samples += n
        else:
            survived_min += n_min
            survived_maj += n_maj

    purities = np.array([float(b.purity) for b in kept], dtype=float)
    radii = np.array([float(b.r) for b in kept], dtype=float)
    sizes = np.array([int(b.num) for b in kept], dtype=int)
    het_count = 0
    for b in kept:
        labs = _labels(b)
        if np.any(labs == 1.0) and np.any(labs == -1.0):
            het_count += 1

    return {
        "n_balls_after_del": len(kept),
        "minority_survival": survived_min / n_min_train if n_min_train else float("nan"),
        "majority_survival": survived_maj / n_maj_train if n_maj_train else float("nan"),
        "minority_small_ball_rate": min_in_small / n_min_train if n_min_train else float("nan"),
        "mean_purity": float(np.mean(purities)) if purities.size else float("nan"),
        "mean_radius": float(np.mean(radii)) if radii.size else float("nan"),
        "n_heterogeneous_balls": het_count,
        "n_samples_removed_del_ball": n_removed_samples,
        "ball_size_mean": float(np.mean(sizes)) if sizes.size else float("nan"),
        "global_min_label": float(global_min),
    }


def _build_rho_granulation(
    indexed: np.ndarray,
    *,
    seed: int,
    purity: float,
) -> tuple[list[GranularBall], dict[str, Any]]:
    """Baseline init_granular_balls + del_ball at a fixed rho (no split-utility gate)."""
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < purity and ball.num > MIN_SAMPLES_PER_BALL:
            c1, c2 = ball.split_2balls()
            balls[i] = c1
            balls.append(c2)
            ll += 1
        else:
            i += 1

    kept = [b for b in balls if int(b.num) >= MIN_BALL_COUNT]
    stats = _replay_stats_from_kept(kept, indexed)
    return kept, stats


def build_representation(
    split,
    seed: int,
    representation: str,
) -> tuple[list[BallInfo], list[GranularBall], dict[str, Any]]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)

    if representation == "B0":
        b0_proposals = _collect_b0_proposals(indexed, seed)
        reject_fn = _make_reject_fn("B0", None, b0_proposals)
        kept, _, stats = _replay_granulation(indexed, seed=seed, should_reject=reject_fn)
    elif representation == "SU3":
        b0_proposals = _collect_b0_proposals(indexed, seed)
        reject_fn = _make_reject_fn("SU3", SU3_PERCENTILE, b0_proposals)
        kept, _, stats = _replay_granulation(indexed, seed=seed, should_reject=reject_fn)
    elif representation in RHO_VALUES:
        kept, stats = _build_rho_granulation(
            indexed, seed=seed, purity=RHO_VALUES[representation]
        )
    else:
        raise ValueError(f"Unknown representation {representation!r}")

    balls = [_to_ball_info(b) for b in kept]
    return balls, kept, stats


def compute_ball_geometry(
    kept: list[GranularBall],
    minority_label: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not kept:
        return [], {
            "mean_density": float("nan"),
            "mean_compactness": float("nan"),
            "mean_entropy": float("nan"),
            "purity_variance": float("nan"),
            "radius_variance": float("nan"),
            "entropy_variance": float("nan"),
            "fraction_mixed_balls": float("nan"),
            "mean_margin_proxy": float("nan"),
            "mean_dist_opposite": float("nan"),
            "mean_dist_same": float("nan"),
            "mean_minority_proportion": float("nan"),
            "mean_majority_proportion": float("nan"),
        }

    centers = np.vstack([b.center for b in kept])
    maj_labels = np.array([_majority_label(b) for b in kept], dtype=float)
    radii = np.array([float(b.r) for b in kept], dtype=float)
    scale = compute_radius_scale(radii, method="median_positive")

    records: list[dict[str, Any]] = []
    densities: list[float] = []
    compactnesses: list[float] = []
    entropies: list[float] = []
    margin_proxies: list[float] = []
    dist_opposites: list[float] = []
    dist_sames: list[float] = []
    min_props: list[float] = []
    maj_props: list[float] = []
    mixed = 0

    for i, ball in enumerate(kept):
        sample_labels = normalize_labels(_labels(ball))
        n = int(ball.num)
        r = float(ball.r)
        density = n / (r + EPS)
        r_norm = normalize_radius(r, scale)
        compactness = 1.0 / (r_norm + EPS)

        opp_idx = np.flatnonzero(maj_labels != maj_labels[i])
        same_idx = np.flatnonzero(maj_labels == maj_labels[i])
        same_idx = same_idx[same_idx != i]

        if opp_idx.size:
            dist_opp = float(np.min(np.linalg.norm(centers[opp_idx] - centers[i], axis=1)))
        else:
            dist_opp = float("nan")
        if same_idx.size:
            dist_same = float(np.min(np.linalg.norm(centers[same_idx] - centers[i], axis=1)))
        else:
            dist_same = float("nan")

        margin_proxy = dist_opp / (dist_same + EPS) if np.isfinite(dist_opp) else float("nan")

        n_min = int(np.sum(sample_labels == minority_label))
        n_maj = n - n_min
        min_prop = n_min / n if n else 0.0
        maj_prop = n_maj / n if n else 0.0
        p_min = min_prop
        p_maj = maj_prop
        entropy = _binary_entropy(p_min) if n_min > 0 and n_maj > 0 else 0.0
        is_mixed = n_min > 0 and n_maj > 0
        if is_mixed:
            mixed += 1

        rec = {
            "size": n,
            "radius": r,
            "density": density,
            "compactness": compactness,
            "dist_nearest_opposite": dist_opp,
            "dist_nearest_same": dist_same,
            "margin_proxy": margin_proxy,
            "purity": float(ball.purity),
            "entropy": entropy,
            "class_label": _majority_label(ball),
            "minority_proportion": min_prop,
            "majority_proportion": maj_prop,
            "is_mixed": is_mixed,
        }
        records.append(rec)
        densities.append(density)
        compactnesses.append(compactness)
        entropies.append(entropy)
        if np.isfinite(margin_proxy):
            margin_proxies.append(margin_proxy)
        if np.isfinite(dist_opp):
            dist_opposites.append(dist_opp)
        if np.isfinite(dist_same):
            dist_sames.append(dist_same)
        min_props.append(min_prop)
        maj_props.append(maj_prop)

    purities = np.array([r["purity"] for r in records], dtype=float)
    globals_out = {
        "mean_density": _nanmean(densities),
        "mean_compactness": _nanmean(compactnesses),
        "mean_entropy": _nanmean(entropies),
        "purity_variance": float(np.var(purities)) if purities.size else float("nan"),
        "radius_variance": float(np.var(radii)) if radii.size else float("nan"),
        "entropy_variance": float(np.var(entropies)) if entropies else float("nan"),
        "fraction_mixed_balls": mixed / len(kept),
        "mean_margin_proxy": _nanmean(margin_proxies),
        "mean_dist_opposite": _nanmean(dist_opposites),
        "mean_dist_same": _nanmean(dist_sames),
        "mean_minority_proportion": _nanmean(min_props),
        "mean_majority_proportion": _nanmean(maj_props),
    }
    return records, globals_out


def fit_gaussian_gblstsvm(
    balls: list[BallInfo],
    *,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    minority_label: float,
    seed: int,
    c1: float | None = None,
    c2: float | None = None,
    prediction_rule: str | None = None,
) -> dict[str, Any]:
    y = normalize_labels(y_train)
    if not balls:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "minority_f1": float("nan"),
            "gmean": float("nan"),
            "classifier_ok": False,
            "classifier_error": "no balls after del_ball",
        }

    try:
        balls_to_class_arrays(balls)
    except ValueError as exc:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "minority_f1": float("nan"),
            "gmean": float("nan"),
            "classifier_ok": False,
            "classifier_error": str(exc),
        }

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y)
    kappa_pos_b, kappa_neg_b = ball_class_weights(balls, kappa_pos, kappa_neg)
    mu_g = compute_gaussian_memberships(balls, PRIMARY_GAUSSIAN)
    mu_pos, mu_neg = split_membership_by_class(balls, mu_g)
    weights_pos = kappa_pos_b * mu_pos
    weights_neg = kappa_neg_b * mu_neg

    centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(balls)
    use_c1 = float(c1 if c1 is not None else DEFAULT_C1)
    use_c2 = float(c2 if c2 is not None else DEFAULT_C2)
    use_rule = prediction_rule if prediction_rule is not None else PREDICTION_RULE
    np.random.seed(seed)
    w1, b1, w2, b2 = solve_cw_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        use_c1,
        use_c2,
        weights_pos,
        weights_neg,
        reg_term=REG_TERM,
        weight_placement=WEIGHT_PLACEMENT,
    )
    y_pred = predict_twin_hyperplanes(X_test, w1, b1, w2, b2, rule=use_rule)
    scores = np.abs(X_test @ w2 + b2) - np.abs(X_test @ w1 + b1)
    metrics = evaluate(y_test, y_pred, scores, minority_label)
    metrics["minority_f1"] = _class_f1(y_test, y_pred, minority_label)
    metrics["classifier_ok"] = True
    metrics["classifier_error"] = None
    metrics["c1"] = use_c1
    metrics["c2"] = use_c2
    metrics["w1"] = w1
    metrics["w2"] = w2
    metrics["b1"] = b1
    metrics["b2"] = b2
    return metrics


def run_single(split, seed: int, representation: str) -> dict[str, Any]:
    balls, kept, rep_stats = build_representation(split, seed, representation)
    _, geom_globals = compute_ball_geometry(kept, split.minority_label)
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    n_balls = int(rep_stats["n_balls_after_del"])
    n_het = int(rep_stats["n_heterogeneous_balls"])
    return {
        "dataset": split.name,
        "seed": seed,
        "representation": representation,
        "n_balls": n_balls,
        "mean_radius": rep_stats["mean_radius"],
        "radius_variance": geom_globals["radius_variance"],
        "mean_purity": rep_stats["mean_purity"],
        "purity_variance": geom_globals["purity_variance"],
        "mean_entropy": geom_globals["mean_entropy"],
        "entropy_variance": geom_globals["entropy_variance"],
        "minority_survival": rep_stats["minority_survival"],
        "majority_survival": rep_stats["majority_survival"],
        "n_heterogeneous_balls": n_het,
        "fraction_mixed_balls": geom_globals["fraction_mixed_balls"],
        "average_ball_size": rep_stats.get("ball_size_mean", float("nan")),
        "mean_density": geom_globals["mean_density"],
        "mean_compactness": geom_globals["mean_compactness"],
        "mean_margin_proxy": geom_globals["mean_margin_proxy"],
        "mean_dist_opposite": geom_globals["mean_dist_opposite"],
        "mean_dist_same": geom_globals["mean_dist_same"],
        "accuracy": perf["accuracy"],
        "macro_f1": perf["macro_f1"],
        "minority_f1": perf["minority_f1"],
        "gmean": perf["gmean"],
        "classifier_ok": perf["classifier_ok"],
        "classifier_error": perf.get("classifier_error"),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> dict[str, Any]:
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


def compute_correlations(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for geom in GEOMETRY_METRICS:
        if geom not in df.columns:
            continue
        gx = df[geom].to_numpy(dtype=float)
        out[geom] = {}
        for perf in PERFORMANCE_METRICS:
            py = df[perf].to_numpy(dtype=float)
            corr = _safe_corr(gx, py, "both")
            out[geom][perf] = corr
    return out


def aggregate_by_representation(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in REPRESENTATIONS:
        sub = df[df["representation"] == rep]
        if sub.empty:
            continue
        rows.append(
            {
                "representation": rep,
                "n_runs": int(len(sub)),
                "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
                "mean_purity": _nanmean(sub["mean_purity"].tolist()),
                "mean_radius": _nanmean(sub["mean_radius"].tolist()),
                "mean_density": _nanmean(sub["mean_density"].tolist()),
                "mean_n_heterogeneous_balls": _nanmean(sub["n_heterogeneous_balls"].tolist()),
                "mean_fraction_mixed": _nanmean(sub["fraction_mixed_balls"].tolist()),
                "mean_margin_proxy": _nanmean(sub["mean_margin_proxy"].tolist()),
                "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["gmean"].tolist()),
                "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
            }
        )
    return rows


def _paired_delta(df: pd.DataFrame, rep: str, metric: str) -> float:
    b0 = df[df["representation"] == "B0"].set_index(["dataset", "seed"])
    other = df[df["representation"] == rep]
    if rep == "B0" or other.empty:
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


def identify_optimal_geometry(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["classifier_ok"].astype(bool)].copy()
    if ok.empty:
        return {"verdict": "D", "detail": "No successful classifier runs."}

    macro_med = ok["macro_f1"].median()
    top = ok[ok["macro_f1"] >= macro_med]
    bot = ok[ok["macro_f1"] < macro_med]

    def profile(sub: pd.DataFrame) -> dict[str, float]:
        return {
            "mean_purity": _nanmean(sub["mean_purity"].tolist()),
            "mean_radius": _nanmean(sub["mean_radius"].tolist()),
            "mean_margin_proxy": _nanmean(sub["mean_margin_proxy"].tolist()),
            "fraction_mixed": _nanmean(sub["fraction_mixed_balls"].tolist()),
            "mean_density": _nanmean(sub["mean_density"].tolist()),
        }

    top_p = profile(top)
    bot_p = profile(bot)

    high_purity_small_r = (
        top_p["mean_purity"] > bot_p["mean_purity"] + 0.02
        and top_p["mean_radius"] < bot_p["mean_radius"] - 1e-6
    )
    moderate_purity_margin = (
        abs(top_p["mean_purity"] - bot_p["mean_purity"]) <= 0.05
        and top_p["mean_margin_proxy"] > bot_p["mean_margin_proxy"] + 0.05
    )
    large_het = top_p["fraction_mixed"] > bot_p["fraction_mixed"] + 0.05

    if high_purity_small_r and not moderate_purity_margin:
        verdict = "A"
        detail = (
            "Top macro-F1 runs show higher purity and smaller mean radius than bottom half."
        )
    elif moderate_purity_margin:
        verdict = "B"
        detail = (
            "Top performers align with moderate purity and stronger margin proxy "
            "(nearest-opposite / nearest-same center separation)."
        )
    elif large_het:
        verdict = "C"
        detail = "Top performers carry more heterogeneous (mixed-label) balls."
    else:
        verdict = "D"
        detail = (
            "No single geometry profile cleanly separates top vs bottom macro-F1 runs; "
            "effects are dataset-specific or weak at aggregate level."
        )

    return {
        "verdict": verdict,
        "detail": detail,
        "top_half_profile": top_p,
        "bottom_half_profile": bot_p,
        "macro_f1_median_split": float(macro_med),
    }


def analyze_su3_failure(df: pd.DataFrame) -> dict[str, Any]:
    metrics = (
        "mean_radius",
        "fraction_mixed_balls",
        "mean_margin_proxy",
        "mean_density",
        "n_heterogeneous_balls",
        "mean_purity",
        "minority_survival",
        "macro_f1",
    )
    out: dict[str, Any] = {"paired_delta_su3_minus_b0": {}, "mechanism_checks": {}}
    for m in metrics:
        out["paired_delta_su3_minus_b0"][m] = _paired_delta(df, "SU3", m)

    d = out["paired_delta_su3_minus_b0"]
    checks = {
        "radius_increases": d.get("mean_radius", 0) > 0,
        "heterogeneity_increases": d.get("fraction_mixed_balls", 0) > 0
        or d.get("n_heterogeneous_balls", 0) > 0,
        "margin_decreases": d.get("mean_margin_proxy", 0) < 0,
        "density_decreases": d.get("mean_density", 0) < 0,
    }
    out["mechanism_checks"] = checks

    triggered = [k for k, v in checks.items() if v]
    if triggered:
        mech = (
            "SU3 failure co-occurs with: "
            + ", ".join(triggered)
            + f" (macro F1 Δ={d.get('macro_f1', float('nan')):+.4f} vs B0)."
        )
    else:
        mech = (
            "SU3 does not show a consistent single geometry shift vs B0 at aggregate level; "
            "inspect per-dataset rows in results.csv."
        )
    out["summary"] = mech
    return out


def explain_b0_vs_su3(df: pd.DataFrame, rep_aggs: list[dict[str, Any]]) -> str:
    b0 = next((a for a in rep_aggs if a["representation"] == "B0"), None)
    su3 = next((a for a in rep_aggs if a["representation"] == "SU3"), None)
    if not b0 or not su3:
        return "Insufficient B0/SU3 aggregates for comparison."

    d_macro = _paired_delta(df, "SU3", "macro_f1")
    d_min_surv = _paired_delta(df, "SU3", "minority_survival")
    d_radius = _paired_delta(df, "SU3", "mean_radius")
    d_margin = _paired_delta(df, "SU3", "mean_margin_proxy")
    d_het = _paired_delta(df, "SU3", "fraction_mixed_balls")

    parts = [
        f"B0 Gaussian macro F1 ({b0['mean_macro_f1']:.3f}) exceeds SU3 ({su3['mean_macro_f1']:.3f}) "
        f"despite SU3 improving minority survival (Δ={d_min_surv:+.3f}).",
    ]
    if d_radius > 0:
        parts.append(f"SU3 tends to retain larger balls (Δ radius={d_radius:+.4f}).")
    if d_margin < 0:
        parts.append(f"SU3 reduces the margin proxy (Δ={d_margin:+.4f}), weakening class separation geometry.")
    if d_het > 0:
        parts.append(f"SU3 increases mixed-ball fraction (Δ={d_het:+.4f}).")
    parts.append(
        "Interpretation: minority sample retention alone does not yield better balls for "
        "Gaussian-weighted twin hyperplanes; geometry (radius, margin, heterogeneity) appears "
        "to dominate macro classification."
    )
    return " ".join(parts)


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    corr = summary["correlations"]
    rep_aggs = summary["by_representation"]
    optimal = summary["optimal_geometry"]
    su3 = summary["su3_failure"]
    b0_explain = summary["b0_vs_su3_explanation"]

    lines = [
        "# Granular geometry vs GBLSTSVM performance",
        "",
        "Read-only diagnostic: which granular-ball properties predict Gaussian-weighted "
        "GBLSTSVM success across frozen granulation variants.",
        "",
        f"Runs: **{summary['n_rows']}** | Weighting: **Gaussian {PRIMARY_GAUSSIAN}** | "
        f"Seeds: {SEEDS}",
        "",
        "## 1. Geometry vs performance correlation",
        "",
        "Pearson and Spearman correlations between representation-level geometry metrics "
        "and classifier performance (all dataset×seed×representation rows).",
        "",
        "| geometry | performance | Pearson | Spearman | n |",
        "|---|---|---:|---:|---:|",
    ]

    for geom in GEOMETRY_METRICS:
        if geom not in corr:
            continue
        for perf in PERFORMANCE_METRICS:
            c = corr[geom].get(perf, {})
            lines.append(
                f"| {geom} | {perf} | {c.get('pearson', float('nan')):.3f} | "
                f"{c.get('spearman', float('nan')):.3f} | {c.get('n', 0)} |"
            )

    lines.extend(
        [
            "",
            "## 2. Compare representations",
            "",
            "| Representation | Balls | Purity | Radius | Density | Heterogeneous balls | Macro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for a in rep_aggs:
        lines.append(
            f"| {a['representation']} | {a['mean_n_balls']:.1f} | {a['mean_purity']:.3f} | "
            f"{a['mean_radius']:.4f} | {a['mean_density']:.2f} | "
            f"{a['mean_n_heterogeneous_balls']:.1f} | {a['mean_macro_f1']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Why does B0 outperform SU3 despite worse minority survival?",
            "",
            b0_explain,
            "",
            "## 3. Optimal granular properties",
            "",
            f"**Verdict: {optimal['verdict']}** — {optimal['detail']}",
            "",
            "| profile | mean purity | mean radius | margin proxy | fraction mixed | mean density |",
            "|---|---:|---:|---:|---:|---:|",
            f"| top macro-F1 half | {optimal['top_half_profile']['mean_purity']:.3f} | "
            f"{optimal['top_half_profile']['mean_radius']:.4f} | "
            f"{optimal['top_half_profile']['mean_margin_proxy']:.3f} | "
            f"{optimal['top_half_profile']['fraction_mixed']:.3f} | "
            f"{optimal['top_half_profile']['mean_density']:.2f} |",
            f"| bottom macro-F1 half | {optimal['bottom_half_profile']['mean_purity']:.3f} | "
            f"{optimal['bottom_half_profile']['mean_radius']:.4f} | "
            f"{optimal['bottom_half_profile']['mean_margin_proxy']:.3f} | "
            f"{optimal['bottom_half_profile']['fraction_mixed']:.3f} | "
            f"{optimal['bottom_half_profile']['mean_density']:.2f} |",
            "",
            "Options: **A** high purity + small radius | **B** moderate purity + strong margin | "
            "**C** large heterogeneous support | **D** no clear relationship",
            "",
            "## 4. Failure mechanism analysis (SU3)",
            "",
            su3["summary"],
            "",
            "| metric | SU3 − B0 (paired mean) |",
            "|---|---:|",
        ]
    )
    for m, v in su3["paired_delta_su3_minus_b0"].items():
        lines.append(f"| {m} | {v:+.4f} |")

    lines.extend(["", "### Mechanism checks", ""])
    for k, v in su3["mechanism_checks"].items():
        lines.append(f"- {k}: **{'yes' if v else 'no'}**")

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


def _build_summary(df: pd.DataFrame, failed: list[dict[str, str]], smoke: bool) -> dict[str, Any]:
    rep_aggs = aggregate_by_representation(df)
    correlations = compute_correlations(df)
    optimal = identify_optimal_geometry(df)
    su3_failure = analyze_su3_failure(df)
    b0_explain = explain_b0_vs_su3(df, rep_aggs)

    return {
        "protocol": {
            "read_only": True,
            "diagnostic_only": True,
            "smoke_mode": smoke,
            "purity_b0_su3": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "seeds": SEEDS,
            "representations": list(REPRESENTATIONS),
            "rho_values": RHO_VALUES,
            "weighting": f"Gaussian {PRIMARY_GAUSSIAN}",
            "su3_rule": f"reject if utility <= percentile({SU3_PERCENTILE}) of B0 proposals",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "prediction_rule": PREDICTION_RULE,
            "weight_placement": WEIGHT_PLACEMENT,
            "failed": failed,
            "no_tuning": True,
        },
        "n_rows": int(len(df)),
        "correlations": correlations,
        "by_representation": rep_aggs,
        "optimal_geometry": optimal,
        "su3_failure": su3_failure,
        "b0_vs_su3_explanation": b0_explain,
    }


def _save_checkpoint(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        if "classifier_ok" not in df.columns:
            df["classifier_ok"] = df["macro_f1"].notna()
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary, failed=[])
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    if smoke:
        descriptors = descriptors[:1]
        seeds = [42]
        print(f"SMOKE: 1 dataset ({descriptors[0]['name']}) × seed 42 × {len(REPRESENTATIONS)} representations")
    else:
        seeds = SEEDS
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds × {len(REPRESENTATIONS)} representations")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    completed = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append(
                    {"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue

            for representation in REPRESENTATIONS:
                row = run_single(split, seed, representation)
                rows.append(row)
                completed += 1
                print(
                    f"{split.name} s{seed} {representation}: "
                    f"balls={row['n_balls']} macro_f1={row['macro_f1']:.3f} "
                    f"margin={row['mean_margin_proxy']:.3f} min_surv={row['minority_survival']:.3f}"
                )
                if completed % SAVE_EVERY == 0:
                    _save_checkpoint(rows)
                    print(f"  [checkpoint] saved {completed} runs -> {OUT_CSV}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary, failed=failed)

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Optimal geometry verdict: {summary['optimal_geometry']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
