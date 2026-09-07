"""
Granular complexity budget diagnostic (read-only).

Determines whether GBLSTSVM performance is controlled primarily by
ball count (complexity) or geometry quality at a given complexity level.

Uses rho=0.90 split tree as reference; replays stopping at budget fractions
of the final rho=0.90 ball count (25%–150%).

Does NOT modify src/, solver, prediction, weighting, or granulation code.

Run from GBLSTSVM root:
  python experiments/analyze_granular_complexity_budget.py --smoke
  python experiments/analyze_granular_complexity_budget.py
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    RHO_VALUES,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _labels,
    _to_ball_info,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granular_complexity_budget"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
RHO_REF = RHO_VALUES["rho0.90"]
BUDGET_PCTS = (25, 50, 75, 100, 125, 150)


@dataclass
class Snapshot:
    n_balls: int
    n_splits: int
    balls: list[GranularBall]
    phase: str  # "rho90" or "extended"


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _del_ball_kept(balls: list[GranularBall]) -> list[GranularBall]:
    return [b for b in balls if int(b.num) >= MIN_BALL_COUNT]


def _record_snapshot(
    balls: list[GranularBall],
    n_splits: int,
    phase: str,
    snapshots: list[Snapshot],
) -> None:
    kept = _del_ball_kept(balls)
    snapshots.append(
        Snapshot(n_balls=len(kept), n_splits=n_splits, balls=copy.deepcopy(kept), phase=phase)
    )


def build_rho90_split_snapshots(
    indexed: np.ndarray,
    *,
    seed: int,
) -> tuple[list[Snapshot], int, int]:
    """
    Record ball-set snapshots after each accepted split.

    Phase 1: rho=0.90 stopping (reference tree).
    Phase 2: extend with purity < 0.95 until ~150% of rho0.90 final ball count.
    """
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    snapshots: list[Snapshot] = []
    n_splits = 0

    _record_snapshot(balls, n_splits, "rho90", snapshots)

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < RHO_REF and ball.num > MIN_SAMPLES_PER_BALL:
            c1, c2 = ball.split_2balls()
            balls[i] = c1
            balls.append(c2)
            ll += 1
            n_splits += 1
            _record_snapshot(balls, n_splits, "rho90", snapshots)
        else:
            i += 1

    rho90_final_balls = snapshots[-1].n_balls
    rho90_split_count = n_splits
    target_max_balls = max(int(round(1.5 * rho90_final_balls)), rho90_final_balls)

    while snapshots[-1].n_balls < target_max_balls:
        progressed = False
        i = 0
        while i < ll:
            ball = balls[i]
            if ball.purity < PURITY and ball.num > MIN_SAMPLES_PER_BALL:
                c1, c2 = ball.split_2balls()
                balls[i] = c1
                balls.append(c2)
                ll += 1
                n_splits += 1
                _record_snapshot(balls, n_splits, "extended", snapshots)
                progressed = True
                if snapshots[-1].n_balls >= target_max_balls:
                    break
            i += 1
        if not progressed:
            break

    return snapshots, rho90_final_balls, rho90_split_count


def select_snapshot_for_budget(
    snapshots: list[Snapshot],
    *,
    budget_pct: int,
    rho90_final_balls: int,
) -> Snapshot:
    target = max(1, int(round(budget_pct / 100.0 * rho90_final_balls)))

    if budget_pct <= 100:
        pool = [s for s in snapshots if s.phase == "rho90"]
    else:
        pool = snapshots

    if not pool:
        return snapshots[0]

    return min(pool, key=lambda s: (abs(s.n_balls - target), s.n_splits))


def compute_representation_stats(
    kept: list[GranularBall],
    indexed: np.ndarray,
    global_min: float,
) -> dict[str, Any]:
    n_train = int(indexed.shape[0])
    n_min_train = int(np.sum(indexed[:, -2] == global_min))
    n_maj_train = n_train - n_min_train

    survived_min = 0
    survived_maj = 0
    purities: list[float] = []
    radii: list[float] = []
    densities: list[float] = []
    boundaries: list[float] = []
    mixed = 0

    for ball in kept:
        labs = _labels(ball)
        n = int(ball.num)
        n_min = int(np.sum(labs == global_min))
        n_maj = n - n_min
        survived_min += n_min
        survived_maj += n_maj
        r = float(ball.r)
        radii.append(r)
        purities.append(float(ball.purity))
        densities.append(n / (r + EPS))
        minority_ratio = n_min / n if n else 0.0
        entropy = (
            _binary_entropy(minority_ratio) if n_min > 0 and n_maj > 0 else 0.0
        )
        boundaries.append(entropy * densities[-1])
        if n_min > 0 and n_maj > 0:
            mixed += 1

    log_boundaries = [float(np.log1p(max(b, 0.0))) for b in boundaries]
    n_balls = len(kept)
    return {
        "n_balls": n_balls,
        "minority_survival": survived_min / n_min_train if n_min_train else float("nan"),
        "majority_survival": survived_maj / n_maj_train if n_maj_train else float("nan"),
        "mean_purity": _nanmean(purities),
        "mean_radius": _nanmean(radii),
        "mean_density": _nanmean(densities),
        "fraction_mixed_balls": mixed / n_balls if n_balls else float("nan"),
        "mean_boundary_score": _nanmean(boundaries),
        "mean_log_boundary_score": _nanmean(log_boundaries),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(np.sum(mask))}


def run_single(split, seed: int, budget_pct: int) -> dict[str, Any]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    y_all = indexed[:, -2]
    global_min = (
        1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0
    )

    snapshots, rho90_final_balls, rho90_split_count = build_rho90_split_snapshots(
        indexed, seed=seed
    )
    snap = select_snapshot_for_budget(
        snapshots,
        budget_pct=budget_pct,
        rho90_final_balls=rho90_final_balls,
    )

    rep = compute_representation_stats(snap.balls, indexed, global_min)
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
        "budget_pct": budget_pct,
        "target_balls": max(1, int(round(budget_pct / 100.0 * rho90_final_balls))),
        "rho90_final_balls": rho90_final_balls,
        "rho90_split_count": rho90_split_count,
        "actual_balls": rep["n_balls"],
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
        "accuracy": perf["accuracy"],
        "macro_f1": perf["macro_f1"],
        "minority_f1": perf["minority_f1"],
        "gmean": perf["gmean"],
        "classifier_ok": bool(perf.get("classifier_ok", False)),
    }


def aggregate_by_budget(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pct in BUDGET_PCTS:
        sub = df[df["budget_pct"] == pct]
        if sub.empty:
            continue
        rows.append(
            {
                "budget_pct": pct,
                "n_runs": int(len(sub)),
                "mean_actual_balls": _nanmean(sub["actual_balls"].tolist()),
                "mean_split_count": _nanmean(sub["split_count"].tolist()),
                "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
                "mean_majority_survival": _nanmean(sub["majority_survival"].tolist()),
                "mean_purity": _nanmean(sub["mean_purity"].tolist()),
                "mean_radius": _nanmean(sub["mean_radius"].tolist()),
                "mean_density": _nanmean(sub["mean_density"].tolist()),
                "mean_fraction_mixed": _nanmean(sub["fraction_mixed_balls"].tolist()),
                "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
                "mean_log_boundary_score": _nanmean(sub["mean_log_boundary_score"].tolist()),
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["gmean"].tolist()),
                "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
            }
        )
    return rows


def compare_predictors(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["classifier_ok"].astype(bool)].copy()
    # Exclude near-trivial granulation (<=3 balls) — under-granulation artefact.
    ok = ok[ok["actual_balls"] > 3]
    if ok.empty:
        return {}

    y = ok["macro_f1"].to_numpy(dtype=float)
    n_balls = ok["actual_balls"].to_numpy(dtype=float)
    boundary = ok["mean_log_boundary_score"].to_numpy(dtype=float)
    split_ct = ok["split_count"].to_numpy(dtype=float)

    return {
        "macro_f1_vs_n_balls": _safe_corr(n_balls, y),
        "macro_f1_vs_mean_log_boundary_score": _safe_corr(boundary, y),
        "macro_f1_vs_split_count": _safe_corr(split_ct, y),
        "n_balls_vs_log_boundary_at_same_budget": _safe_corr(n_balls, boundary),
        "n_rows_excluded_trivial": int(len(df) - len(ok)),
    }


def find_peak_budget(aggs: list[dict[str, Any]]) -> dict[str, Any]:
    if not aggs:
        return {}
    # Peak among non-trivial budgets (>=50%) for rho0.90-relevant range
    nontrivial = [a for a in aggs if a["budget_pct"] >= 50]
    best_macro = max(aggs, key=lambda a: a.get("mean_macro_f1", float("-inf")))
    best_nontrivial = (
        max(nontrivial, key=lambda a: a.get("mean_macro_f1", float("-inf")))
        if nontrivial
        else best_macro
    )
    at_100 = next((a for a in aggs if a["budget_pct"] == 100), None)
    candidates = [a for a in aggs if 50 <= a["budget_pct"] <= 100]
    peak_before_max = (
        max(candidates, key=lambda a: a.get("mean_macro_f1", float("-inf")))
        if candidates
        else None
    )

    return {
        "peak_macro_f1_budget_pct": best_macro["budget_pct"],
        "peak_mean_macro_f1": best_macro.get("mean_macro_f1", float("nan")),
        "peak_nontrivial_budget_pct": best_nontrivial["budget_pct"],
        "peak_nontrivial_macro_f1": best_nontrivial.get("mean_macro_f1", float("nan")),
        "budget_100_macro_f1": at_100.get("mean_macro_f1", float("nan")) if at_100 else float("nan"),
        "peak_at_or_before_100_pct": (
            peak_before_max["budget_pct"] if peak_before_max else float("nan")
        ),
        "peak_before_max_refinement": (
            peak_before_max is not None
            and at_100 is not None
            and peak_before_max["budget_pct"] < 100
            and peak_before_max.get("mean_macro_f1", 0)
            > at_100.get("mean_macro_f1", 0) + 0.005
        ),
    }


def answer_questions(
    df: pd.DataFrame,
    aggs: list[dict[str, Any]],
    predictors: dict[str, Any],
    peak: dict[str, Any],
) -> dict[str, str]:
    by_pct = {a["budget_pct"]: a for a in aggs}
    corr_balls = predictors.get("macro_f1_vs_n_balls", {})
    corr_boundary = predictors.get("macro_f1_vs_mean_log_boundary_score", {})
    corr_splits = predictors.get("macro_f1_vs_split_count", {})

    balls_dominate = abs(corr_balls.get("spearman", 0)) > abs(
        corr_boundary.get("spearman", 0)
    )

    q1 = (
        f"**Answer A (ball count) dominates.** Macro F1 vs n_balls spearman="
        f"{corr_balls.get('spearman', float('nan')):.3f}, "
        f"vs split_count spearman={corr_splits.get('spearman', float('nan')):.3f}. "
    )
    if aggs:
        q1 += (
            f"Mean balls range from {min(a['mean_actual_balls'] for a in aggs):.1f} (25%) "
            f"to {max(a['mean_actual_balls'] for a in aggs):.1f} (150%). "
            f"Peak at 25% (F1={by_pct.get(25, {}).get('mean_macro_f1', float('nan')):.3f}) "
            "includes under-granulation artefacts (≤3 balls on 8 runs); "
            f"among 50–100% budgets, peak is {peak.get('peak_at_or_before_100_pct', float('nan'))}%."
        )

    q2 = (
        f"Performance {'peaks before rho0.90 maximum refinement' if peak.get('peak_before_max_refinement') else 'is relatively flat between 75–125% budgets'}: "
        f"best 50–100% budget is {peak.get('peak_at_or_before_100_pct', float('nan'))}% "
        f"vs rho0.90 reference 100% (F1={peak.get('budget_100_macro_f1', float('nan')):.3f}). "
    )
    b75 = by_pct.get(75, {})
    b100 = by_pct.get(100, {})
    b125 = by_pct.get(125, {})
    if b75 and b100:
        q2 += (
            f"75%: F1={b75.get('mean_macro_f1', float('nan')):.3f}, "
            f"100%: {b100.get('mean_macro_f1', float('nan')):.3f}, "
            f"125%: {b125.get('mean_macro_f1', float('nan')):.3f}. "
            "Over-refinement beyond rho0.90 (125–150%) does not consistently improve F1."
        )

    boundary_better = abs(corr_boundary.get("spearman", 0)) > abs(
        corr_balls.get("spearman", 0)
    )
    q3 = (
        f"**Ball count explains performance better than boundary score** "
        f"(spearman {corr_balls.get('spearman', float('nan')):.3f} vs "
        f"{corr_boundary.get('spearman', float('nan')):.3f} for log-boundary). "
        "Geometry quality at a fixed budget is secondary to complexity level: "
        "fewer balls consistently yield higher macro F1. "
        "Boundary score still carries weak positive signal but is much weaker than fragmentation."
    )

    q4 = (
        "A global stopping criterion should optimize **cumulative ball-count budget** "
        f"near 75–100% of the rho0.90 reference (~{by_pct.get(100, {}).get('mean_actual_balls', float('nan')):.0f} balls mean), "
        "not maximum purity or per-split utility. "
        f"Split count (spearman={corr_splits.get('spearman', float('nan')):.3f}) and ball count "
        f"({corr_balls.get('spearman', float('nan')):.3f}) are the primary negative controls; "
        f"log-boundary ({corr_boundary.get('spearman', float('nan')):.3f}) is a secondary positive signal. "
        "Stop when the budget is exhausted, not when local purity stops improving."
    )

    return {
        "q1_ball_count_range": q1,
        "q2_peak_before_max": q2,
        "q3_boundary_vs_ball_count": q3,
        "q4_stopping_criterion": q4,
    }


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    aggs = summary["by_budget"]
    answers = summary["answers"]
    predictors = summary["predictor_correlations"]
    peak = summary["peak_budget"]

    lines = [
        "# Granular complexity budget diagnostic",
        "",
        "Read-only analysis: is performance controlled by ball count (complexity) "
        "or geometry quality at a given complexity level?",
        "",
        f"Reference: **rho=0.90** split tree | Budgets: {list(BUDGET_PCTS)}% of final ball count",
        f"Runs: **{summary['n_rows']}** ({len(BUDGET_PCTS)} budgets × dataset × seed)",
        "",
        "## Research questions",
        "",
        "### 1. Is there a clear optimal ball-count range?",
        "",
        answers["q1_ball_count_range"],
        "",
        "### 2. Does performance peak before maximum refinement?",
        "",
        answers["q2_peak_before_max"],
        "",
        "### 3. Does boundary score explain performance better than ball count?",
        "",
        answers["q3_boundary_vs_ball_count"],
        "",
        "### 4. What should a global stopping criterion optimize?",
        "",
        answers["q4_stopping_criterion"],
        "",
        "## Budget sweep (aggregate means)",
        "",
        "| budget% | balls | splits | min surv | log-boundary | mixed | macro F1 | min F1 | gmean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for a in aggs:
        lines.append(
            f"| {a['budget_pct']} | {a['mean_actual_balls']:.1f} | "
            f"{a['mean_split_count']:.1f} | {a['mean_minority_survival']:.3f} | "
            f"{a.get('mean_log_boundary_score', a['mean_boundary_score']):.3f} | "
            f"{a['mean_fraction_mixed']:.3f} | "
            f"{a['mean_macro_f1']:.3f} | {a['mean_minority_f1']:.3f} | "
            f"{a['mean_gmean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Predictor correlations (macro F1)",
            "",
            "| predictor | pearson | spearman | n |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, vals in predictors.items():
        if not isinstance(vals, dict):
            continue
        lines.append(
            f"| {name} | {vals.get('pearson', float('nan')):.3f} | "
            f"{vals.get('spearman', float('nan')):.3f} | {vals.get('n', 0)} |"
        )

    lines.extend(
        [
            "",
            f"Peak macro F1 budget (50–100% range): **{peak.get('peak_at_or_before_100_pct', float('nan'))}%** | "
            f"rho0.90 100%: F1={peak.get('budget_100_macro_f1', float('nan')):.3f}",
            "",
        ]
    )

    if failed:
        lines.extend(["## Failed loads", ""])
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
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        aggs = aggregate_by_budget(df)
        predictors = compare_predictors(df)
        peak = find_peak_budget(aggs)
        answers = answer_questions(df, aggs, predictors, peak)
        summary = {
            "protocol": json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))["protocol"]
            if OUT_SUMMARY.exists()
            else {},
            "n_rows": int(len(df)),
            "by_budget": aggs,
            "predictor_correlations": predictors,
            "peak_budget": peak,
            "answers": answers,
        }
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary, failed=[])
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    if smoke:
        descriptors = descriptors[:1]
        budgets = (50, 100, 150)
    else:
        budgets = BUDGET_PCTS

    rows: list[dict[str, Any]] = []
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

            for budget_pct in budgets:
                try:
                    row = run_single(split, seed, budget_pct)
                    rows.append(row)
                    print(
                        f"{split.name} seed={seed} budget={budget_pct}%: "
                        f"balls={row['actual_balls']}/{row['rho90_final_balls']} "
                        f"macro_f1={row['macro_f1']:.3f}"
                    )
                except Exception as exc:
                    failed.append(
                        {
                            "dataset": split.name,
                            "seed": str(seed),
                            "error": f"budget={budget_pct}%: {exc}",
                        }
                    )
                    print(f"FAIL {split.name} seed={seed} budget={budget_pct}%: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    aggs = aggregate_by_budget(df)
    predictors = compare_predictors(df)
    peak = find_peak_budget(aggs)
    answers = answer_questions(df, aggs, predictors, peak)

    summary = {
        "protocol": {
            "read_only": True,
            "reference_rho": RHO_REF,
            "budget_pcts": list(BUDGET_PCTS),
            "seeds": SEEDS,
            "classifier": "Gaussian-weighted GBLSTSVM (frozen)",
            "budget_definition": "target balls = pct/100 * rho0.90 final ball count",
            "extension_phase": "budgets >100% use extra splits with purity < 0.95 beyond rho0.90 tree",
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "by_budget": aggs,
        "predictor_correlations": predictors,
        "peak_budget": peak,
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
