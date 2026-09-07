"""
Validate adaptive entropy v2 compact representation (read-only).

Multi-seed stability check: does ~4-split adaptive entropy v2 consistently
outperform full rho90 BFS despite far fewer balls?

Does NOT modify src/, solver, prediction, weighting, or baseline granulation.

Compares:
  baseline-rho90        — full BFS rho=0.90 granulation
  adaptive-entropy-v2   — online entropy v2, budget = rho90 split count

Run from GBLSTSVM root:
  python experiments/validate_entropy_compact_representation.py --smoke
  python experiments/validate_entropy_compact_representation.py
  python experiments/validate_entropy_compact_representation.py --from-csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_geometry_vs_performance import fit_gaussian_gblstsvm  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels  # noqa: E402
from experiments.test_adaptive_entropy_granulation_v2 import (  # noqa: E402
    build_adaptive_entropy_v2,
    build_baseline_rho90,
    compute_extended_stats,
    measure_rho90_split_count,
    _global_min_label,
)
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _to_ball_info,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "entropy_compact_validation"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = list(range(10))  # 0..9
CONFIG_RHO90 = "baseline-rho90"
CONFIG_V2 = "adaptive-entropy-v2"
WIN_THRESHOLD = 8  # of 10 seeds for "consistent"
SAVE_EVERY = 20
DELTA_EPS = 1e-6


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _nanstd(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v)) if v.size else float("nan")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(np.sum(mask))}


def run_single(
    split,
    seed: int,
    config: str,
    *,
    indexed: np.ndarray,
    global_min: float,
    rho90_split_count: int,
) -> dict[str, Any]:
    if config == CONFIG_RHO90:
        result = build_baseline_rho90(indexed, seed=seed, split_budget=None)
    else:
        result = build_adaptive_entropy_v2(
            indexed,
            seed=seed,
            split_budget=rho90_split_count,
            global_min=global_min,
            rho90_split_count=rho90_split_count,
        )

    rep = compute_extended_stats(result.kept, result.depths, indexed, global_min)
    balls = [_to_ball_info(b) for b in result.kept]
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    n_balls = rep["n_balls"]
    macro_f1 = float(perf["macro_f1"])
    f1_per_ball = macro_f1 / n_balls if n_balls > 0 else float("nan")

    return {
        "dataset": split.name,
        "seed": seed,
        "config": config,
        "rho90_split_count": rho90_split_count,
        "split_budget": result.split_budget,
        "split_count": result.n_successful_splits,
        "budget_met": int(result.budget_met),
        "n_balls": n_balls,
        "minority_survival": rep["minority_survival"],
        "majority_survival": rep["majority_survival"],
        "mean_purity": rep["mean_purity"],
        "mean_entropy": rep["mean_entropy"],
        "mean_boundary_score": rep["mean_boundary_score"],
        "mean_radius": rep["mean_radius"],
        "mean_depth": rep["mean_depth"],
        "max_depth": rep["max_depth"],
        "depth_std": rep["depth_std"],
        "fraction_mixed_balls": rep["fraction_mixed_balls"],
        "accuracy": perf["accuracy"],
        "macro_f1": macro_f1,
        "minority_f1": perf["minority_f1"],
        "gmean": perf["gmean"],
        "f1_per_ball": f1_per_ball,
        "classifier_ok": bool(perf.get("classifier_ok", False)),
    }


def _paired_runs(df: pd.DataFrame) -> pd.DataFrame:
    rho = df[df["config"] == CONFIG_RHO90].set_index(["dataset", "seed"])
    v2 = df[df["config"] == CONFIG_V2].set_index(["dataset", "seed"])
    keys = rho.index.intersection(v2.index)
    rows: list[dict[str, Any]] = []
    for key in keys:
        r = rho.loc[key]
        a = v2.loc[key]
        rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "delta_macro_f1": float(a["macro_f1"] - r["macro_f1"]),
                "delta_minority_f1": float(a["minority_f1"] - r["minority_f1"]),
                "delta_gmean": float(a["gmean"] - r["gmean"]),
                "delta_n_balls": float(a["n_balls"] - r["n_balls"]),
                "delta_splits": float(a["split_count"] - r["split_count"]),
                "delta_f1_per_ball": float(a["f1_per_ball"] - r["f1_per_ball"]),
                "rho90_macro_f1": float(r["macro_f1"]),
                "adaptive_macro_f1": float(a["macro_f1"]),
                "rho90_n_balls": float(r["n_balls"]),
                "adaptive_n_balls": float(a["n_balls"]),
                "rho90_splits": float(r["split_count"]),
                "adaptive_splits": float(a["split_count"]),
            }
        )
    return pd.DataFrame(rows)


def _classify_datasets(paired: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, grp in paired.groupby("dataset"):
        deltas = grp["delta_macro_f1"].to_numpy(dtype=float)
        n = len(deltas)
        n_wins = int(np.sum(deltas > DELTA_EPS))
        n_losses = int(np.sum(deltas < -DELTA_EPS))
        n_ties = n - n_wins - n_losses
        mean_delta = _nanmean(deltas)
        std_delta = _nanstd(deltas)

        if n_wins >= WIN_THRESHOLD and n_losses == 0:
            classification = "consistent_win"
        elif n_losses >= WIN_THRESHOLD and n_wins == 0:
            classification = "consistent_loss"
        elif n_wins >= WIN_THRESHOLD and n_losses <= 2:
            classification = "consistent_win"
        elif n_losses >= WIN_THRESHOLD and n_wins <= 2:
            classification = "consistent_loss"
        else:
            classification = "unstable"

        rows.append(
            {
                "dataset": dataset,
                "n_seeds": n,
                "n_wins": n_wins,
                "n_losses": n_losses,
                "n_ties": n_ties,
                "mean_delta_macro_f1": mean_delta,
                "std_delta_macro_f1": std_delta,
                "mean_adaptive_splits": _nanmean(grp["adaptive_splits"].tolist()),
                "mean_rho90_splits": _nanmean(grp["rho90_splits"].tolist()),
                "mean_adaptive_n_balls": _nanmean(grp["adaptive_n_balls"].tolist()),
                "mean_rho90_n_balls": _nanmean(grp["rho90_n_balls"].tolist()),
                "classification": classification,
            }
        )
    return sorted(rows, key=lambda r: r["mean_delta_macro_f1"], reverse=True)


def _aggregate_by_config(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in (CONFIG_RHO90, CONFIG_V2):
        sub = df[df["config"] == cfg]
        if sub.empty:
            continue
        rows.append(
            {
                "config": cfg,
                "n_runs": int(len(sub)),
                "mean_split_count": _nanmean(sub["split_count"].tolist()),
                "std_split_count": _nanstd(sub["split_count"].tolist()),
                "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
                "std_n_balls": _nanstd(sub["n_balls"].tolist()),
                "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
                "mean_majority_survival": _nanmean(sub["majority_survival"].tolist()),
                "mean_purity": _nanmean(sub["mean_purity"].tolist()),
                "mean_entropy": _nanmean(sub["mean_entropy"].tolist()),
                "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
                "mean_radius": _nanmean(sub["mean_radius"].tolist()),
                "mean_depth": _nanmean(sub["mean_depth"].tolist()),
                "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "std_macro_f1": _nanstd(sub["macro_f1"].tolist()),
                "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
                "mean_gmean": _nanmean(sub["gmean"].tolist()),
                "mean_f1_per_ball": _nanmean(sub["f1_per_ball"].tolist()),
            }
        )
    return rows


def _regularization_analysis(paired: pd.DataFrame) -> dict[str, Any]:
    ball_reduction = paired["rho90_n_balls"].to_numpy(dtype=float) - paired[
        "adaptive_n_balls"
    ].to_numpy(dtype=float)
    delta_f1 = paired["delta_macro_f1"].to_numpy(dtype=float)
    split_reduction = paired["rho90_splits"].to_numpy(dtype=float) - paired[
        "adaptive_splits"
    ].to_numpy(dtype=float)
    return {
        "delta_f1_vs_ball_reduction": _safe_corr(ball_reduction, delta_f1),
        "delta_f1_vs_split_reduction": _safe_corr(split_reduction, delta_f1),
        "mean_ball_reduction": _nanmean(ball_reduction),
        "mean_split_reduction": _nanmean(split_reduction),
        "mean_delta_f1_per_ball_adaptive": _nanmean(paired["delta_f1_per_ball"].tolist()),
    }


def _split_stability(paired: pd.DataFrame) -> dict[str, Any]:
    adapt_splits = paired["adaptive_splits"].to_numpy(dtype=float)
    return {
        "mean_adaptive_splits": _nanmean(adapt_splits),
        "std_adaptive_splits": _nanstd(adapt_splits),
        "median_adaptive_splits": float(np.median(adapt_splits)),
        "p10_adaptive_splits": float(np.percentile(adapt_splits, 10)),
        "p90_adaptive_splits": float(np.percentile(adapt_splits, 90)),
        "frac_adaptive_le_10_splits": float(np.mean(adapt_splits <= 10)),
    }


def answer_questions(
    df: pd.DataFrame,
    paired: pd.DataFrame,
    by_config: list[dict[str, Any]],
    by_dataset: list[dict[str, Any]],
    split_stab: dict[str, Any],
    reg: dict[str, Any],
) -> dict[str, str]:
    by_name = {a["config"]: a for a in by_config}
    rho = by_name.get(CONFIG_RHO90, {})
    v2 = by_name.get(CONFIG_V2, {})
    mean_delta = _nanmean(paired["delta_macro_f1"].tolist())
    std_delta = _nanstd(paired["delta_macro_f1"].tolist())

    n_win = sum(1 for d in by_dataset if d["classification"] == "consistent_win")
    n_loss = sum(1 for d in by_dataset if d["classification"] == "consistent_loss")
    n_unstable = sum(1 for d in by_dataset if d["classification"] == "unstable")

    q1 = (
        f"Across {len(paired)} runs ({paired['dataset'].nunique()} datasets × "
        f"{paired.groupby('dataset').size().iloc[0] if len(paired) else 0} seeds): "
        f"adaptive splits mean={split_stab.get('mean_adaptive_splits', float('nan')):.1f} "
        f"(std={split_stab.get('std_adaptive_splits', float('nan')):.1f}, "
        f"median={split_stab.get('median_adaptive_splits', float('nan')):.0f}, "
        f"P10–P90=[{split_stab.get('p10_adaptive_splits', float('nan')):.0f}, "
        f"{split_stab.get('p90_adaptive_splits', float('nan')):.0f}]). "
        f"Paired Δ macro F1 = {mean_delta:+.4f} ± {std_delta:.4f}. "
        f"Dataset outcomes: {n_win} consistent wins, {n_loss} consistent losses, "
        f"{n_unstable} unstable. "
    )
    if split_stab.get("frac_adaptive_le_10_splits", 0) > 0.9 and mean_delta > 0.01:
        q1 += (
            "**Mostly stable** — adaptive stays ≤10 splits in "
            f"{split_stab.get('frac_adaptive_le_10_splits', 0):.0%} of runs with positive mean ΔF1, "
            "but per-dataset variance exists."
        )
    elif mean_delta > 0 and n_win > n_loss:
        q1 += "**Directionally stable** — aggregate gain holds but not uniform across datasets."
    else:
        q1 += "**Unstable** — compact split count does not reliably transfer across seeds/datasets."

    f1pb_rho = rho.get("mean_f1_per_ball", float("nan"))
    f1pb_v2 = v2.get("mean_f1_per_ball", float("nan"))
    q2 = (
        f"Adaptive: {v2.get('mean_n_balls', float('nan')):.1f} balls, "
        f"F1/ball={f1pb_v2:.4f}. rho90: {rho.get('mean_n_balls', float('nan')):.1f} balls, "
        f"F1/ball={f1pb_rho:.4f} "
        f"(ratio {f1pb_v2 / f1pb_rho if f1pb_rho > 0 else float('nan'):.1f}×). "
        f"Mean purity adaptive={v2.get('mean_purity', float('nan')):.3f} "
        f"vs rho={rho.get('mean_purity', float('nan')):.3f}; "
        f"entropy {v2.get('mean_entropy', float('nan')):.3f} vs "
        f"{rho.get('mean_entropy', float('nan')):.3f}. "
    )
    if f1pb_v2 > f1pb_rho * 5 and mean_delta > 0:
        q2 += "**Yes** — far fewer balls carry comparable or better discriminative mass per ball."
    else:
        q2 += "**Unclear** — compactness is real but per-ball efficiency alone does not explain gains."

    top_winners = [d for d in by_dataset if d["classification"] == "consistent_win"][:5]
    top_losers = [d for d in by_dataset if d["classification"] == "consistent_loss"][:5]
    largest = (
        f"{by_dataset[0]['dataset']} ({by_dataset[0]['mean_delta_macro_f1']:+.3f})"
        if by_dataset
        else "n/a"
    )
    q3 = (
        f"Consistent wins ({n_win}): "
        f"{', '.join(d['dataset'] for d in top_winners) or 'none'}. "
        f"Consistent losses ({n_loss}): "
        f"{', '.join(d['dataset'] for d in top_losers) or 'none'}. "
        f"Largest mean ΔF1: {largest}. "
    )
    if n_win >= n_loss * 2:
        q3 += "**Entropy benefits a clear subset** — not universal but repeatable on winning datasets."
    else:
        q3 += "**Mixed** — benefits are dataset-specific, not a single benchmark artifact."

    ball_corr = reg.get("delta_f1_vs_ball_reduction", {})
    q4 = (
        f"Mean ball reduction: {reg.get('mean_ball_reduction', float('nan')):.1f}; "
        f"ΔF1 vs ball-reduction spearman={ball_corr.get('spearman', float('nan')):.3f}. "
        f"Mean Δ(f1_per_ball)={reg.get('mean_delta_f1_per_ball_adaptive', float('nan')):+.4f}. "
    )
    if ball_corr.get("spearman", 0) > 0.2:
        q4 += "**Likely yes** — larger ball-count reductions associate with larger F1 gains (regularization)."
    elif mean_delta > 0 and f1pb_v2 > f1pb_rho:
        q4 += (
            "**Partially** — fewer balls co-occur with gains but correlation is weak; "
            "may reflect implicit structural regularization rather than ball count alone."
        )
    else:
        q4 += "**No clear regularization story** — gains do not track ball-count reduction."

    q5 = (
        f"Aggregate: adaptive F1={v2.get('mean_macro_f1', float('nan')):.3f} "
        f"vs rho={rho.get('mean_macro_f1', float('nan')):.3f} "
        f"(Δ={mean_delta:+.4f}) at "
        f"{v2.get('mean_split_count', float('nan')):.1f}/{rho.get('mean_split_count', float('nan')):.0f} splits. "
        f"Unstable on {n_unstable}/{len(by_dataset)} datasets. "
    )
    if mean_delta > 0.01 and n_win > n_loss and split_stab.get("frac_adaptive_le_10_splits", 0) > 0.8:
        q5 += (
            "**Promising direction** — compact entropy scheduling is a viable research path, "
            "but requires per-dataset safeguards and fixing candidate exhaustion before production."
        )
    elif mean_delta > 0:
        q5 += (
            "**Conditional** — worth pursuing as a complexity-regularized variant, "
            "not yet a drop-in replacement for rho90."
        )
    else:
        q5 += "**Not yet viable** — aggregate advantage does not survive extended seed sweep."

    return {
        "q1_split_stability": q1,
        "q2_compact_useful": q2,
        "q3_which_datasets": q3,
        "q4_regularization": q4,
        "q5_viable_direction": q5,
    }


def write_report(path: Path, *, summary: dict[str, Any]) -> None:
    by_config = summary["by_config"]
    by_dataset = summary["by_dataset"]
    answers = summary["answers"]
    split_stab = summary["split_stability"]

    lines = [
        "# Entropy compact representation validation",
        "",
        "Multi-seed (0–9) read-only comparison: full rho90 BFS vs adaptive entropy v2.",
        "",
        f"Runs: **{summary['n_rows']}** ({summary['n_dataset_seed_pairs']} dataset×seed pairs)",
        "",
        "## Scheduler summary",
        "",
        "| config | splits | balls | F1/ball | macro F1 | min F1 | gmean | depth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in by_config:
        lines.append(
            f"| {a['config']} | {a['mean_split_count']:.1f} | {a['mean_n_balls']:.1f} | "
            f"{a['mean_f1_per_ball']:.4f} | {a['mean_macro_f1']:.3f} ± {a['std_macro_f1']:.3f} | "
            f"{a['mean_minority_f1']:.3f} | {a['mean_gmean']:.3f} | {a['mean_depth']:.1f} |"
        )

    lines.extend(
        [
            "",
            f"Adaptive split stability: mean={split_stab.get('mean_adaptive_splits', float('nan')):.1f}, "
            f"median={split_stab.get('median_adaptive_splits', float('nan')):.0f}, "
            f"≤10 splits in {split_stab.get('frac_adaptive_le_10_splits', 0):.0%} of runs.",
            "",
            "## Per-dataset classification (adaptive − rho90 macro F1)",
            "",
            "| dataset | class | wins | losses | mean ΔF1 | adapt splits | rho splits |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for d in by_dataset:
        lines.append(
            f"| {d['dataset']} | {d['classification']} | {d['n_wins']} | {d['n_losses']} | "
            f"{d['mean_delta_macro_f1']:+.4f} | {d['mean_adaptive_splits']:.1f} | "
            f"{d['mean_rho90_splits']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 1. Is the ~4-split result stable?",
            "",
            answers["q1_split_stability"],
            "",
            "## 2. Does entropy produce a compact but useful representation?",
            "",
            answers["q2_compact_useful"],
            "",
            "## 3. Which datasets benefit?",
            "",
            answers["q3_which_datasets"],
            "",
            "## 4. Is the improvement caused by regularization from fewer balls?",
            "",
            answers["q4_regularization"],
            "",
            "## 5. Is this a viable direction for a final method?",
            "",
            answers["q5_viable_direction"],
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
    paired = _paired_runs(df)
    by_config = _aggregate_by_config(df)
    by_dataset = _classify_datasets(paired)
    split_stab = _split_stability(paired)
    reg = _regularization_analysis(paired)
    answers = answer_questions(df, paired, by_config, by_dataset, split_stab, reg)
    return {
        "protocol": {
            "read_only": True,
            "configs": [CONFIG_RHO90, CONFIG_V2],
            "seeds": SEEDS,
            "n_datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "adaptive_budget": "rho90_split_count (100%)",
            "classification_rule": (
                f"consistent_win/loss if ≥{WIN_THRESHOLD}/10 seeds same sign; else unstable"
            ),
            "smoke_mode": smoke,
            "classifier": f"Gaussian-weighted GBLSTSVM ({PRIMARY_GAUSSIAN})",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "n_dataset_seed_pairs": int(paired.groupby(["dataset", "seed"]).ngroups)
        if not paired.empty
        else 0,
        "paired_mean_delta_macro_f1": _nanmean(paired["delta_macro_f1"].tolist()),
        "paired_std_delta_macro_f1": _nanstd(paired["delta_macro_f1"].tolist()),
        "by_config": by_config,
        "by_dataset": by_dataset,
        "dataset_classification_counts": {
            "consistent_win": sum(1 for d in by_dataset if d["classification"] == "consistent_win"),
            "consistent_loss": sum(1 for d in by_dataset if d["classification"] == "consistent_loss"),
            "unstable": sum(1 for d in by_dataset if d["classification"] == "unstable"),
        },
        "split_stability": split_stab,
        "regularization_analysis": reg,
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [0, 1] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print(f"SMOKE: 1 dataset × seeds {seeds}")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append(
                    {"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue
            try:
                y_train = normalize_labels(split.y_train)
                indexed = _build_indexed_matrix(split.X_train, y_train)
                global_min = _global_min_label(indexed)
                rho90_count = measure_rho90_split_count(indexed, seed=seed)

                for config in (CONFIG_RHO90, CONFIG_V2):
                    rows.append(
                        run_single(
                            split,
                            seed,
                            config,
                            indexed=indexed,
                            global_min=global_min,
                            rho90_split_count=rho90_count,
                        )
                    )
                    done += 1

                recent = rows[-2:]
                rho = next(r for r in recent if r["config"] == CONFIG_RHO90)
                v2 = next(r for r in recent if r["config"] == CONFIG_V2)
                delta = v2["macro_f1"] - rho["macro_f1"]
                print(
                    f"{split.name} s{seed}: rho F1={rho['macro_f1']:.3f} "
                    f"({rho['split_count']} splits, {rho['n_balls']} balls) | "
                    f"v2 F1={v2['macro_f1']:.3f} ({v2['split_count']} splits, "
                    f"{v2['n_balls']} balls) dF1={delta:+.3f}"
                )

                if done % (SAVE_EVERY * 2) == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    print(f"  [checkpoint] {done // 2} pairs")

            except Exception as exc:
                failed.append({"dataset": split.name, "seed": str(seed), "error": str(exc)})
                print(f"FAIL {desc['name']} seed={seed}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
