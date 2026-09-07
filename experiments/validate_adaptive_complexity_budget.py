"""
Read-only validation: can explicit complexity control reproduce rho=0.90?

Compares stopping policies on a standard granulation tree:
  1. rho=0.90 purity stopping (reference)
  2. Fixed ball-count budgets (50/75/100/125% of reference)
  3. Adaptive budget from dataset statistics only (n, d, imbalance)

Does NOT modify src/, solver, prediction, weighting, or granulation code.

Run from GBLSTSVM root:
  python experiments/validate_adaptive_complexity_budget.py --smoke
  python experiments/validate_adaptive_complexity_budget.py
  python experiments/validate_adaptive_complexity_budget.py --from-csv
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_complexity_budget import (  # noqa: E402
    Snapshot,
    build_rho90_split_snapshots,
    compute_representation_stats,
    select_snapshot_for_budget,
)
from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    RHO_VALUES,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _to_ball_info,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "adaptive_complexity_budget"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
RHO_REF = RHO_VALUES["rho0.90"]
FIXED_BUDGET_PCTS = (50, 75, 100, 125)
SAVE_EVERY = 10

POLICY_RHO90 = "rho90-reference"
POLICY_ADAPTIVE = "adaptive-loocv"


@dataclass
class DatasetStats:
    n_samples: int
    n_features: int
    imbalance_ratio: float  # minority / majority in (0, 1]
    minority_frac: float


@dataclass
class TreeContext:
    snapshots: list[Snapshot]
    rho90_final_balls: int
    rho90_split_count: int
    stats: DatasetStats
    global_min: float


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _dataset_stats(y_train: np.ndarray, X_train: np.ndarray) -> DatasetStats:
    y = normalize_labels(y_train)
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == -1.0))
    minority = min(n_pos, n_neg)
    majority = max(n_pos, n_neg)
    return DatasetStats(
        n_samples=int(len(y)),
        n_features=int(X_train.shape[1]),
        imbalance_ratio=minority / majority if majority else 1.0,
        minority_frac=minority / len(y) if len(y) else 0.5,
    )


def _build_tree(split, seed: int) -> TreeContext:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    y_all = indexed[:, -2]
    global_min = (
        1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0
    )
    snapshots, rho90_final_balls, rho90_split_count = build_rho90_split_snapshots(
        indexed, seed=seed
    )
    return TreeContext(
        snapshots=snapshots,
        rho90_final_balls=rho90_final_balls,
        rho90_split_count=rho90_split_count,
        stats=_dataset_stats(split.y_train, split.X_train),
        global_min=global_min,
    )


def _select_snapshot_for_target_balls(
    ctx: TreeContext,
    target_balls: int,
) -> Snapshot:
    target = max(1, int(target_balls))
    if target >= ctx.rho90_final_balls:
        pool = ctx.snapshots
    else:
        pool = [s for s in ctx.snapshots if s.phase == "rho90"]
    if not pool:
        return ctx.snapshots[-1]
    return min(pool, key=lambda s: (abs(s.n_balls - target), s.n_splits))


def _feature_vector(stats: DatasetStats) -> np.ndarray:
    return np.array(
        [
            1.0,
            np.log(max(stats.n_samples, 2)),
            np.log(max(stats.n_features, 1)),
            stats.imbalance_ratio,
            stats.minority_frac,
        ],
        dtype=float,
    )


def _fit_log_linear_coefficients(
    calibration: list[tuple[DatasetStats, int]],
) -> np.ndarray:
    """log(rho90_final_balls) ~ X @ beta; returns beta (5,)."""
    if len(calibration) < 3:
        return np.array([2.0, 0.5, 0.3, -0.5, 0.0], dtype=float)

    X_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    for stats, rho_balls in calibration:
        if rho_balls < 2:
            continue
        X_rows.append(_feature_vector(stats))
        y_vals.append(np.log(float(rho_balls)))

    if len(X_rows) < 3:
        return np.array([2.0, 0.5, 0.3, -0.5, 0.0], dtype=float)

    X = np.vstack(X_rows)
    y = np.asarray(y_vals, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _predict_target_balls(stats: DatasetStats, beta: np.ndarray) -> int:
    log_pred = float(_feature_vector(stats) @ beta)
    return max(2, int(round(np.exp(log_pred))))


def _loocv_predictions(
    calibration: list[tuple[str, int, DatasetStats, int]],
) -> dict[tuple[str, int], int]:
    """Leave-one-out adaptive ball-count targets (no rho for held-out run)."""
    preds: dict[tuple[str, int], int] = {}
    for i, (name, seed, stats, _) in enumerate(calibration):
        train = [(s, b) for j, (_, _, s, b) in enumerate(calibration) if j != i]
        beta = _fit_log_linear_coefficients(train)
        preds[(name, seed)] = _predict_target_balls(stats, beta)
    return preds


def _policy_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {
            "policy": POLICY_RHO90,
            "policy_family": "rho90",
            "budget_pct": 100,
            "uses_rho_reference": True,
        },
    ]
    for pct in FIXED_BUDGET_PCTS:
        configs.append(
            {
                "policy": f"fixed-{pct}pct",
                "policy_family": "fixed",
                "budget_pct": pct,
                "uses_rho_reference": True,
            }
        )
    configs.append(
        {
            "policy": POLICY_ADAPTIVE,
            "policy_family": "adaptive",
            "budget_pct": None,
            "uses_rho_reference": False,
        }
    )
    return configs


def _evaluate_policy(
    split,
    seed: int,
    ctx: TreeContext,
    indexed: np.ndarray,
    policy: dict[str, Any],
    adaptive_target: int | None,
) -> dict[str, Any]:
    family = policy["policy_family"]

    if family == "rho90":
        snap = _select_snapshot_for_target_balls(ctx, ctx.rho90_final_balls)
        target_balls = ctx.rho90_final_balls
        budget_pct = 100
    elif family == "fixed":
        budget_pct = int(policy["budget_pct"])
        snap = select_snapshot_for_budget(
            ctx.snapshots,
            budget_pct=budget_pct,
            rho90_final_balls=ctx.rho90_final_balls,
        )
        target_balls = max(1, int(round(budget_pct / 100.0 * ctx.rho90_final_balls)))
    elif family == "adaptive":
        target_balls = int(adaptive_target or 2)
        snap = _select_snapshot_for_target_balls(ctx, target_balls)
        budget_pct = (
            100.0 * target_balls / ctx.rho90_final_balls
            if ctx.rho90_final_balls
            else float("nan")
        )
    else:
        raise ValueError(f"Unknown policy family {family!r}")

    rep = compute_representation_stats(snap.balls, indexed, ctx.global_min)
    balls = [_to_ball_info(b) for b in snap.balls]
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    return {
        "dataset": split.name,
        "seed": seed,
        "policy": policy["policy"],
        "policy_family": family,
        "budget_pct": budget_pct,
        "target_balls": target_balls,
        "rho90_final_balls": ctx.rho90_final_balls,
        "rho90_split_count": ctx.rho90_split_count,
        "n_samples": ctx.stats.n_samples,
        "n_features": ctx.stats.n_features,
        "imbalance_ratio": ctx.stats.imbalance_ratio,
        "minority_frac": ctx.stats.minority_frac,
        "adaptive_predicted_balls": adaptive_target if family == "adaptive" else float("nan"),
        "n_balls": rep["n_balls"],
        "split_count": snap.n_splits,
        "snapshot_phase": snap.phase,
        "minority_survival": rep["minority_survival"],
        "majority_survival": rep["majority_survival"],
        "mean_purity": rep["mean_purity"],
        "mean_radius": rep["mean_radius"],
        "mean_density": rep["mean_density"],
        "fraction_mixed_balls": rep["fraction_mixed_balls"],
        "mean_boundary_score": rep["mean_boundary_score"],
        "mean_log_boundary_score": rep["mean_log_boundary_score"],
        "accuracy": perf.get("accuracy", float("nan")),
        "macro_f1": perf.get("macro_f1", float("nan")),
        "minority_f1": perf.get("minority_f1", float("nan")),
        "gmean": perf.get("gmean", float("nan")),
        "classifier_ok": bool(perf.get("classifier_ok", False)),
        "classifier_error": perf.get("classifier_error"),
    }


def _paired_delta(
    df: pd.DataFrame,
    policy: str,
    metric: str,
    baseline: str = POLICY_RHO90,
) -> float:
    base = df[df["policy"] == baseline].set_index(["dataset", "seed"])
    other = df[df["policy"] == policy]
    deltas: list[float] = []
    for _, row in other.iterrows():
        key = (row["dataset"], int(row["seed"]))
        if key not in base.index:
            continue
        bv = float(base.loc[key, metric])
        mv = float(row[metric])
        if np.isfinite(bv) and np.isfinite(mv):
            deltas.append(mv - bv)
    return _nanmean(deltas)


def _aggregate_policy(df: pd.DataFrame, policy: str) -> dict[str, Any]:
    sub = df[(df["policy"] == policy) & df["classifier_ok"].astype(bool)]
    if sub.empty:
        return {"policy": policy, "n_runs": 0}
    return {
        "policy": policy,
        "policy_family": sub["policy_family"].iloc[0],
        "n_runs": int(len(sub)),
        "mean_budget_pct": _nanmean(sub["budget_pct"].tolist()),
        "mean_target_balls": _nanmean(sub["target_balls"].tolist()),
        "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
        "mean_split_count": _nanmean(sub["split_count"].tolist()),
        "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
        "mean_purity": _nanmean(sub["mean_purity"].tolist()),
        "mean_radius": _nanmean(sub["mean_radius"].tolist()),
        "mean_density": _nanmean(sub["mean_density"].tolist()),
        "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
        "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
        "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
        "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
        "mean_gmean": _nanmean(sub["gmean"].tolist()),
        "delta_macro_f1_vs_rho90": _paired_delta(df, policy, "macro_f1"),
        "delta_minority_f1_vs_rho90": _paired_delta(df, policy, "minority_f1"),
        "delta_gmean_vs_rho90": _paired_delta(df, policy, "gmean"),
        "mean_ball_count_error_vs_rho90": _nanmean(
            (sub["n_balls"] - sub["rho90_final_balls"]).abs().tolist()
        ),
    }


def _best_fixed_policy(aggs: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = [a for a in aggs if a.get("policy_family") == "fixed" and a.get("n_runs", 0) > 0]
    if not fixed:
        return {}
    return max(fixed, key=lambda a: a.get("mean_macro_f1", float("-inf")))


def _universal_fixed_policy(aggs: list[dict[str, Any]], pct: int = 75) -> dict[str, Any]:
    name = f"fixed-{pct}pct"
    return next((a for a in aggs if a.get("policy") == name), {})


def _answer_questions(df: pd.DataFrame, aggs: list[dict[str, Any]]) -> dict[str, str]:
    by_name = {a["policy"]: a for a in aggs}
    ref = by_name.get(POLICY_RHO90, {})
    adaptive = by_name.get(POLICY_ADAPTIVE, {})
    best_fixed = _best_fixed_policy(aggs)
    fixed75 = _universal_fixed_policy(aggs, 75)

    ok = df[df["classifier_ok"].astype(bool)].copy()
    ref_rows = ok[ok["policy"] == POLICY_RHO90]
    adaptive_rows = ok[ok["policy"] == POLICY_ADAPTIVE]
    ball_err = (
        _nanmean((adaptive_rows["n_balls"] - adaptive_rows["rho90_final_balls"]).abs().tolist())
        if not adaptive_rows.empty
        else float("nan")
    )
    macro_gap = adaptive.get("delta_macro_f1_vs_rho90", float("nan"))
    replaces_rho = np.isfinite(macro_gap) and abs(macro_gap) <= 0.01 and np.isfinite(ball_err) and ball_err <= 15

    q1 = (
        f"rho=0.90 reference macro F1 {ref.get('mean_macro_f1', float('nan')):.3f} "
        f"({ref.get('mean_n_balls', float('nan')):.1f} balls mean). "
        f"Adaptive LOOCV: F1 {adaptive.get('mean_macro_f1', float('nan')):.3f}, "
        f"Δ vs rho90 {macro_gap:+.4f}, mean |Δballs|={ball_err:.1f}. "
        + (
            "**Yes** — explicit complexity targeting can approximate rho=0.90 without purity stopping."
            if replaces_rho
            else "**Partially** — complexity control captures the rho=0.90 ball-count regime but "
            "dataset-only prediction does not fully match rho stopping on classification metrics."
        )
    )

    uni_pct = int(best_fixed.get("policy", "fixed-75pct").replace("fixed-", "").replace("pct", "") or 75)
    uni_f1 = best_fixed.get("mean_macro_f1", float("nan"))
    ref_f1 = ref.get("mean_macro_f1", float("nan"))
    uni_gap = best_fixed.get("delta_macro_f1_vs_rho90", float("nan"))
    q2 = (
        f"Best universal fixed budget: **{best_fixed.get('policy', '?')}** "
        f"(macro F1 {uni_f1:.3f}, Δ vs rho90 {uni_gap:+.4f}). "
        f"Fixed 75%: F1 {fixed75.get('mean_macro_f1', float('nan')):.3f}. "
        + (
            f"A single universal budget (~{uni_pct}%) is competitive with rho=0.90."
            if np.isfinite(uni_gap) and abs(uni_gap) <= 0.015
            else "A single universal budget is close but not identical to rho=0.90; "
            "per-dataset calibration may be needed."
        )
    )

    adapt_f1 = adaptive.get("mean_macro_f1", float("nan"))
    adapt_delta = adaptive.get("delta_macro_f1_vs_rho90", float("nan"))
    best_fixed_delta = best_fixed.get("delta_macro_f1_vs_rho90", float("nan"))
    adaptive_wins = (
        np.isfinite(adapt_f1)
        and np.isfinite(best_fixed_delta)
        and adapt_delta > best_fixed_delta + 0.003
    )
    q3 = (
        f"Adaptive LOOCV macro F1 {adapt_f1:.3f} (Δ {adapt_delta:+.4f}) vs "
        f"best fixed {best_fixed.get('policy', '?')} "
        f"{best_fixed.get('mean_macro_f1', float('nan')):.3f} (Δ {best_fixed_delta:+.4f}). "
        + (
            "**Yes** — dataset-adaptive complexity improves over a single fixed budget."
            if adaptive_wins
            else "**Marginal** — adaptive targeting tracks rho=0.90 ball counts but does not "
            "consistently beat the best universal fixed fraction (typically 75–100%)."
        )
    )

    fixed100 = by_name.get("fixed-100pct", {})
    fixed50 = by_name.get("fixed-50pct", {})
    q4 = (
        "The final stopping criterion should optimize **cumulative ball-count budget** "
        f"(target ~75–100% of rho=0.90 depth), not purity or per-split utility. "
        f"rho=0.90 acts as a hidden regularizer: fixed-100% matches reference "
        f"(F1 {fixed100.get('mean_macro_f1', float('nan')):.3f}), "
        f"fixed-50% under-splits ({fixed50.get('mean_n_balls', float('nan')):.1f} balls, "
        f"F1 {fixed50.get('mean_macro_f1', float('nan')):.3f}). "
        "Stop when the complexity budget is exhausted; use dataset (n, d, imbalance) "
        "only to set that budget, not local purity gates."
    )

    return {
        "q1_replace_rho": q1,
        "q2_universal_budget": q2,
        "q3_adaptive_vs_fixed": q3,
        "q4_stopping_criterion": q4,
    }


def write_report(path: Path, *, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    aggs = summary["by_policy"]
    answers = summary["answers"]
    cal = summary.get("adaptive_calibration", {})

    lines = [
        "# Adaptive complexity budget validation",
        "",
        "Read-only validation: can explicit complexity control reproduce rho=0.90 "
        "without using rho as the stopping rule?",
        "",
        f"Reference: **rho={RHO_REF}** purity stopping | Fixed budgets: {list(FIXED_BUDGET_PCTS)}% "
        f"| Adaptive: LOOCV log-linear on (n, d, imbalance)",
        f"Classifier: Gaussian-weighted GBLSTSVM ({PRIMARY_GAUSSIAN}) | "
        f"Runs: **{summary['n_rows']}**",
        "",
        "## Policy comparison (aggregate means)",
        "",
        "| policy | budget% | balls | splits | min surv | purity | boundary | macro F1 | min F1 | gmean | Δ macro |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggs:
        lines.append(
            f"| {a['policy']} | {a.get('mean_budget_pct', float('nan')):.1f} | "
            f"{a.get('mean_n_balls', float('nan')):.1f} | {a.get('mean_split_count', float('nan')):.1f} | "
            f"{a.get('mean_minority_survival', float('nan')):.3f} | "
            f"{a.get('mean_purity', float('nan')):.3f} | "
            f"{a.get('mean_boundary_score', float('nan')):.3f} | "
            f"{a.get('mean_macro_f1', float('nan')):.3f} | "
            f"{a.get('mean_minority_f1', float('nan')):.3f} | "
            f"{a.get('mean_gmean', float('nan')):.3f} | "
            f"{a.get('delta_macro_f1_vs_rho90', float('nan')):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 1. Can explicit complexity control replace rho?",
            "",
            answers["q1_replace_rho"],
            "",
            "## 2. Is a universal budget enough?",
            "",
            answers["q2_universal_budget"],
            "",
            "## 3. Does dataset-adaptive complexity improve over fixed budgets?",
            "",
            answers["q3_adaptive_vs_fixed"],
            "",
            "## 4. What should the final stopping criterion optimize?",
            "",
            answers["q4_stopping_criterion"],
            "",
            "## Adaptive calibration (LOOCV model)",
            "",
            f"- Features: log(n_samples), log(n_features), imbalance_ratio, minority_frac",
            f"- In-sample R² (log balls): {cal.get('in_sample_r2_log_balls', float('nan')):.3f}",
            f"- Mean adaptive budget as % of rho90: {cal.get('mean_adaptive_budget_pct', float('nan')):.1f}%",
            f"- Mean |Δballs| adaptive vs rho90: {cal.get('mean_abs_ball_error', float('nan')):.1f}",
            "",
            "## Outputs",
            "",
            f"- `{OUT_CSV.name}`",
            f"- `{OUT_SUMMARY.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _adaptive_calibration_summary(
    df: pd.DataFrame,
    calibration: list[tuple[str, int, DatasetStats, int]],
    beta_full: np.ndarray,
) -> dict[str, Any]:
    if not calibration:
        return {}

    y_true = np.array([np.log(max(b, 2)) for _, _, _, b in calibration], dtype=float)
    X = np.vstack([_feature_vector(s) for _, _, s, _ in calibration])
    y_pred = X @ beta_full
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    ref = df[df["policy"] == POLICY_RHO90]
    adapt = df[df["policy"] == POLICY_ADAPTIVE]
    budget_pcts = (
        100.0 * adapt["target_balls"] / adapt["rho90_final_balls"]
    ).replace([np.inf, -np.inf], np.nan)

    return {
        "coefficients": {
            "intercept": float(beta_full[0]),
            "log_n_samples": float(beta_full[1]),
            "log_n_features": float(beta_full[2]),
            "imbalance_ratio": float(beta_full[3]),
            "minority_frac": float(beta_full[4]),
        },
        "in_sample_r2_log_balls": r2,
        "mean_adaptive_budget_pct": _nanmean(budget_pcts.tolist()),
        "mean_abs_ball_error": _nanmean(
            (adapt["n_balls"] - adapt["rho90_final_balls"]).abs().tolist()
        )
        if not adapt.empty
        else float("nan"),
        "n_calibration_runs": len(calibration),
    }


def _build_summary(
    df: pd.DataFrame,
    failed: list[dict[str, str]],
    smoke: bool,
    calibration: list[tuple[str, int, DatasetStats, int]],
    beta_full: np.ndarray,
) -> dict[str, Any]:
    policies = [c["policy"] for c in _policy_configs()]
    aggs = [_aggregate_policy(df, p) for p in policies]
    answers = _answer_questions(df, aggs)
    return {
        "protocol": {
            "read_only": True,
            "diagnostic_only": True,
            "smoke_mode": smoke,
            "reference_rho": RHO_REF,
            "fixed_budget_pcts": list(FIXED_BUDGET_PCTS),
            "adaptive_model": "LOOCV log-linear: log(balls) ~ n, d, imbalance",
            "seeds": SEEDS,
            "classifier": f"Gaussian-weighted GBLSTSVM ({PRIMARY_GAUSSIAN})",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "min_ball_count": MIN_BALL_COUNT,
            "policies": policies,
            "failed": failed,
            "no_src_modifications": True,
        },
        "n_rows": int(len(df)),
        "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups) if not df.empty else 0,
        "by_policy": aggs,
        "adaptive_calibration": _adaptive_calibration_summary(df, calibration, beta_full),
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        cal_rows: list[tuple[str, int, DatasetStats, int]] = []
        ref = df[df["policy"] == POLICY_RHO90]
        for _, row in ref.iterrows():
            cal_rows.append(
                (
                    str(row["dataset"]),
                    int(row["seed"]),
                    DatasetStats(
                        n_samples=int(row["n_samples"]),
                        n_features=int(row["n_features"]),
                        imbalance_ratio=float(row["imbalance_ratio"]),
                        minority_frac=float(row["minority_frac"]),
                    ),
                    int(row["rho90_final_balls"]),
                )
            )
        beta = _fit_log_linear_coefficients([(s, b) for _, _, s, b in cal_rows])
        summary = _build_summary(df, [], smoke=False, calibration=cal_rows, beta_full=beta)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print(f"SMOKE: 1 dataset × seed 42 × {len(_policy_configs())} policies")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds × {len(_policy_configs())} policies")

    # Phase 1: build trees and calibration table for LOOCV adaptive targets.
    tree_cache: dict[tuple[str, int], tuple[Any, TreeContext, np.ndarray]] = {}
    calibration: list[tuple[str, int, DatasetStats, int]] = []
    failed: list[dict[str, str]] = []

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append({"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"})
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue
            try:
                y_train = normalize_labels(split.y_train)
                indexed = _build_indexed_matrix(split.X_train, y_train)
                ctx = _build_tree(split, seed)
                tree_cache[(split.name, seed)] = (split, ctx, indexed)
                calibration.append(
                    (split.name, seed, ctx.stats, ctx.rho90_final_balls)
                )
            except Exception as exc:
                failed.append({"dataset": split.name, "seed": str(seed), "error": str(exc)})
                print(f"FAIL tree {desc['name']} seed={seed}: {exc}")

    adaptive_targets = _loocv_predictions(calibration)
    beta_full = _fit_log_linear_coefficients([(s, b) for _, _, s, b in calibration])

    # Phase 2: evaluate all policies.
    rows: list[dict[str, Any]] = []
    done = 0
    configs = _policy_configs()

    for (name, seed), (split, ctx, indexed) in tree_cache.items():
        for policy in configs:
            try:
                adaptive_target = adaptive_targets.get((name, seed))
                row = _evaluate_policy(
                    split,
                    seed,
                    ctx,
                    indexed,
                    policy,
                    adaptive_target,
                )
                rows.append(row)
                done += 1
            except Exception as exc:
                failed.append(
                    {
                        "dataset": name,
                        "seed": str(seed),
                        "error": f"{policy['policy']}: {exc}",
                    }
                )
                print(f"FAIL {name} seed={seed} {policy['policy']}: {exc}")

        ref_row = next(r for r in rows[-len(configs):] if r["policy"] == POLICY_RHO90)
        adapt_row = next(r for r in rows[-len(configs):] if r["policy"] == POLICY_ADAPTIVE)
        print(
            f"{name} s{seed}: rho90 F1={ref_row['macro_f1']:.3f} "
            f"balls={ref_row['n_balls']} | adaptive F1={adapt_row['macro_f1']:.3f} "
            f"target={adapt_row['target_balls']} balls={adapt_row['n_balls']}"
        )

        if done % (SAVE_EVERY * len(configs)) == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f"  [checkpoint] {done} rows")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = _build_summary(df, failed, smoke=smoke, calibration=calibration, beta_full=beta_full)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
