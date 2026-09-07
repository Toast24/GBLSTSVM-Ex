"""
Granular compression K-sensitivity diagnostic (read-only).

Explores whether rho90 ball-count reduction improves generalization, whether
entropy/boundary selection beats random at fixed K, and whether granular-ball
compression differs from ordinary sample-level reduction.

Does NOT modify src/, solver, prediction, weighting, or rho90 granulation.

Run from GBLSTSVM root:
  python experiments/test_granular_compression_k_sensitivity.py --smoke
  python experiments/test_granular_compression_k_sensitivity.py
  python experiments/test_granular_compression_k_sensitivity.py --from-csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_ball_utility_landscape import compute_ball_features  # noqa: E402
from experiments.analyze_granular_geometry_vs_performance import build_representation  # noqa: E402
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.granular_compression_k_common import (  # noqa: E402
    BALL_METHODS,
    DELTA_EPS,
    K_VALUES,
    RANDOM_REPS,
    SAMPLE_METHODS,
    WIN_THRESHOLD,
    build_rho90_from_train_indices,
    deterministic_rng_seed,
    evaluate_balls,
    representation_stats,
    select_stratified_random_indices,
    select_top_k_indices,
    select_training_sample_indices,
    samples_represented,
)
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels  # noqa: E402
from experiments.test_adaptive_entropy_granulation_v2 import _global_min_label  # noqa: E402
from experiments.test_split_utility_granulation import _build_indexed_matrix  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granular_compression_k_sensitivity"
RAW_CSV = OUT_DIR / "raw_results.csv"
SUMMARY_CSV = OUT_DIR / "summary_by_k_method.csv"
DATASET_CSV = OUT_DIR / "dataset_level_results.csv"
PAIRED_CSV = OUT_DIR / "paired_comparisons.csv"
FAILURE_CSV = OUT_DIR / "failure_analysis.csv"
README = OUT_DIR / "README.md"
REPRESENTATION = "rho0.90"
SEEDS = list(range(10))
SAVE_EVERY = 25

CRITERION_MAP = {
    "random_k": "random",
    "entropy_k": "entropy",
    "boundary_k": "boundary",
    "entropy_boundary_k": "entropy_boundary",
    "purity_k": "purity",
}

PLOT_METHODS = (
    "random_k",
    "entropy_k",
    "boundary_k",
    "entropy_boundary_k",
    "purity_k",
    "stratified_random_k",
    "sample_random_k",
    "sample_stratified_k",
    "full_rho90",
)


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _nanmedian(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _nanstd(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v)) if v.size else float("nan")


def _base_row(dataset: str, seed: int, method: str, k: int | None, repetition: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "K": k if k is not None else float("nan"),
        "repetition": repetition,
        "skipped": 0,
        "skip_reason": "",
    }


def run_dataset_seed(split, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    global_min = _global_min_label(indexed)
    minority_label = split.minority_label

    balls_rho90, kept_rho90, _ = build_representation(split, seed, REPRESENTATION)
    features = compute_ball_features(kept_rho90, minority_label)
    n_pool = len(balls_rho90)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    full_eval = evaluate_balls(
        balls_rho90,
        kept_rho90,
        split=split,
        seed=seed,
        indexed=indexed,
        global_min=global_min,
        minority_label=minority_label,
    )
    rows.append(
        {
            **_base_row(split.name, seed, "full_rho90", None, 0),
            "n_rho90_balls": n_pool,
            **full_eval,
        }
    )

    for k in K_VALUES:
        if n_pool < k:
            failures.append(
                {
                    "dataset": split.name,
                    "seed": seed,
                    "K": k,
                    "reason": f"rho90_pool={n_pool}<K={k}",
                }
            )
            continue

        entropy_idx = select_top_k_indices(
            balls_rho90,
            features,
            k,
            "entropy",
            minority_label,
            deterministic_rng_seed(split.name, seed, k, method="entropy_k"),
        )
        entropy_kept = [kept_rho90[i] for i in entropy_idx]
        n_target_samples = samples_represented(entropy_kept, minority_label)["n_samples_represented"]

        method_indices: dict[str, list[int]] = {
            "entropy_k": entropy_idx,
        }

        for rep in range(RANDOM_REPS):
            rng_seed = deterministic_rng_seed(
                split.name, seed, k, method="random_k", repetition=rep
            )
            idx = select_top_k_indices(
                balls_rho90, features, k, "random", minority_label, rng_seed
            )
            kept = [kept_rho90[i] for i in idx]
            ev = evaluate_balls(
                [balls_rho90[i] for i in idx],
                kept,
                split=split,
                seed=seed,
                indexed=indexed,
                global_min=global_min,
                minority_label=minority_label,
            )
            rows.append(
                {
                    **_base_row(split.name, seed, "random_k", k, rep),
                    "n_rho90_balls": n_pool,
                    "n_target_samples": n_target_samples,
                    **ev,
                }
            )

        for method in BALL_METHODS:
            if method in ("random_k", "entropy_k"):
                if method == "entropy_k":
                    idx = entropy_idx
                else:
                    continue
            elif method == "stratified_random_k":
                idx = select_stratified_random_indices(
                    balls_rho90,
                    k,
                    minority_label,
                    deterministic_rng_seed(split.name, seed, k, method=method),
                )
            else:
                idx = select_top_k_indices(
                    balls_rho90,
                    features,
                    k,
                    CRITERION_MAP[method],
                    minority_label,
                    deterministic_rng_seed(split.name, seed, k, method=method),
                )
            method_indices[method] = idx
            kept = [kept_rho90[i] for i in idx]
            ev = evaluate_balls(
                [balls_rho90[i] for i in idx],
                kept,
                split=split,
                seed=seed,
                indexed=indexed,
                global_min=global_min,
                minority_label=minority_label,
            )
            rows.append(
                {
                    **_base_row(split.name, seed, method, k, 0),
                    "n_rho90_balls": n_pool,
                    "n_target_samples": n_target_samples,
                    **ev,
                }
            )

        for method, stratified in (
            ("sample_random_k", False),
            ("sample_stratified_k", True),
        ):
            train_idx = select_training_sample_indices(
                y_train,
                n_target_samples,
                minority_label,
                deterministic_rng_seed(split.name, seed, k, method=method),
                stratified=stratified,
            )
            sub_balls, sub_kept = build_rho90_from_train_indices(
                indexed, train_idx, seed=seed
            )
            ev = evaluate_balls(
                sub_balls,
                sub_kept,
                split=split,
                seed=seed,
                indexed=indexed,
                global_min=global_min,
                minority_label=minority_label,
            )
            rows.append(
                {
                    **_base_row(split.name, seed, method, k, 0),
                    "n_rho90_balls": n_pool,
                    "n_target_samples": n_target_samples,
                    "n_train_samples_retained": int(len(train_idx)),
                    **ev,
                }
            )

    return rows, failures


def build_summary_by_k_method(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    methods = sorted(df["method"].unique())
    for method in methods:
        if method == "random_k":
            sub = df[df["method"] == "random_k"]
        else:
            sub = df[df["method"] == method]
        if sub.empty:
            continue
        for k, g in sub.groupby("K", dropna=False):
            kval = float(k) if pd.notna(k) else float("nan")
            rows.append(
                {
                    "method": method,
                    "K": kval,
                    "n_runs": int(len(g)),
                    "mean_macro_f1": _nanmean(g["macro_f1"].tolist()),
                    "median_macro_f1": _nanmedian(g["macro_f1"].tolist()),
                    "std_macro_f1": _nanstd(g["macro_f1"].tolist()),
                    "mean_accuracy": _nanmean(g["accuracy"].tolist()),
                    "mean_minority_f1": _nanmean(g["minority_f1"].tolist()),
                    "mean_gmean": _nanmean(g["gmean"].tolist()),
                    "frac_collapsed": float(g["collapsed"].mean()),
                    "mean_n_samples": _nanmean(g["n_samples_represented"].tolist()),
                    "mean_n_balls": _nanmean(g["n_balls"].tolist()),
                }
            )
    out = pd.DataFrame(rows)
    return out


def build_paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    full = df[df["method"] == "full_rho90"].set_index(["dataset", "seed"])
    rand_mean = (
        df[df["method"] == "random_k"]
        .groupby(["dataset", "seed", "K"])["macro_f1"]
        .mean()
    )

    comparisons = [
        ("entropy_k", "full_rho90"),
        ("entropy_k", "random_k"),
        ("entropy_k", "stratified_random_k"),
        ("entropy_k", "sample_random_k"),
        ("boundary_k", "random_k"),
        ("boundary_k", "full_rho90"),
        ("entropy_k", "boundary_k"),
    ]

    rows: list[dict[str, Any]] = []
    for focal, baseline in comparisons:
        for k in K_VALUES:
            keys: set[tuple[str, int]] = set()
            focal_series: dict[tuple[str, int], float] = {}
            base_series: dict[tuple[str, int], float] = {}

            if focal == "entropy_k" or focal in df["method"].unique():
                fsub = df[(df["method"] == focal) & (df["K"] == k)]
                for _, r in fsub.iterrows():
                    key = (r["dataset"], int(r["seed"]))
                    focal_series[key] = float(r["macro_f1"])
                    keys.add(key)

            if baseline == "full_rho90":
                for key in full.index:
                    if isinstance(key, tuple):
                        base_series[key] = float(full.loc[key, "macro_f1"])
                        keys.add(key)
            elif baseline == "random_k":
                for key, val in rand_mean.items():
                    if key[2] == k:
                        base_series[(key[0], int(key[1]))] = float(val)
                        keys.add((key[0], int(key[1])))
            else:
                bsub = df[(df["method"] == baseline) & (df["K"] == k)]
                for _, r in bsub.iterrows():
                    key = (r["dataset"], int(r["seed"]))
                    base_series[key] = float(r["macro_f1"])
                    keys.add(key)

            deltas: list[float] = []
            for key in keys:
                fv = focal_series.get(key)
                bv = base_series.get(key)
                if fv is None or bv is None:
                    continue
                if np.isfinite(fv) and np.isfinite(bv):
                    deltas.append(fv - bv)

            if not deltas:
                continue

            wins = sum(1 for d in deltas if d > DELTA_EPS)
            losses = sum(1 for d in deltas if d < -DELTA_EPS)
            ties = len(deltas) - wins - losses
            wilcoxon_p = float("nan")
            if len(deltas) >= 5:
                try:
                    stat, wilcoxon_p = wilcoxon(np.asarray(deltas), zero_method="wilcox")
                except Exception:
                    wilcoxon_p = float("nan")

            rows.append(
                {
                    "focal_method": focal,
                    "baseline_method": baseline,
                    "K": k,
                    "n_pairs": len(deltas),
                    "mean_delta_macro_f1": _nanmean(deltas),
                    "median_delta_macro_f1": _nanmedian(deltas),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "frac_wins": wins / len(deltas) if deltas else float("nan"),
                    "wilcoxon_p": wilcoxon_p,
                }
            )
    return pd.DataFrame(rows)


def build_dataset_level(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full = df[df["method"] == "full_rho90"].set_index(["dataset", "seed"])
    rand_mean = (
        df[df["method"] == "random_k"]
        .groupby(["dataset", "seed", "K"])["macro_f1"]
        .mean()
    )

    for method in df["method"].unique():
        if method == "full_rho90":
            continue
        for dataset, gds in df[df["method"] == method].groupby("dataset"):
            for k, gk in gds.groupby("K"):
                if method == "random_k":
                    deltas = []
                    for _, r in gk.iterrows():
                        key = (r["dataset"], int(r["seed"]))
                        if key not in full.index:
                            continue
                        deltas.append(float(r["macro_f1"] - full.loc[key, "macro_f1"]))
                else:
                    deltas = []
                    for _, r in gk.iterrows():
                        key = (r["dataset"], int(r["seed"]))
                        if key not in full.index:
                            continue
                        deltas.append(float(r["macro_f1"] - full.loc[key, "macro_f1"]))
                if not deltas:
                    continue
                mean_d = _nanmean(deltas)
                n_pos_seeds = sum(1 for d in deltas if d > DELTA_EPS)
                n_neg_seeds = sum(1 for d in deltas if d < -DELTA_EPS)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "K": float(k),
                        "mean_macro_f1": _nanmean(gk["macro_f1"].tolist()),
                        "mean_delta_vs_full_rho90": mean_d,
                        "n_positive_seeds_vs_full": n_pos_seeds,
                        "n_negative_seeds_vs_full": n_neg_seeds,
                        "consistent_positive": int(n_pos_seeds >= WIN_THRESHOLD),
                        "consistent_negative": int(n_neg_seeds >= WIN_THRESHOLD),
                    }
                )
    return pd.DataFrame(rows)


def build_k_regime_table(summary: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "random_k",
        "entropy_k",
        "boundary_k",
        "entropy_boundary_k",
        "purity_k",
        "stratified_random_k",
        "sample_random_k",
    ]
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        row: dict[str, Any] = {"K": k}
        for m in methods:
            sub = summary[(summary["method"] == m) & (summary["K"] == k)]
            row[m] = float(sub["mean_macro_f1"].iloc[0]) if not sub.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def find_optima(summary: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in summary["method"].unique():
        sub = summary[summary["method"] == method].copy()
        if sub.empty or sub["mean_macro_f1"].isna().all():
            continue
        best_mean = sub.loc[sub["mean_macro_f1"].idxmax()]
        best_median = sub.loc[sub["median_macro_f1"].idxmax()]
        out[method] = {
            "K_max_mean_macro_f1": float(best_mean["K"]),
            "max_mean_macro_f1": float(best_mean["mean_macro_f1"]),
            "K_max_median_macro_f1": float(best_median["K"]),
            "max_median_macro_f1": float(best_median["median_macro_f1"]),
        }
    return out


def answer_questions(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    dataset_level: pd.DataFrame,
    full_stats: dict[str, Any],
) -> dict[str, str]:
    full_f1 = full_stats.get("mean_macro_f1", float("nan"))
    small_k = summary[(summary["K"] <= 10) & (summary["method"] != "full_rho90")]
    comp_small = small_k.groupby("method")["mean_macro_f1"].mean().sort_values(ascending=False)

    ent_vs_rand = paired[
        (paired["focal_method"] == "entropy_k") & (paired["baseline_method"] == "random_k")
    ]
    ent_vs_sample = paired[
        (paired["focal_method"] == "entropy_k") & (paired["baseline_method"] == "sample_random_k")
    ]
    bnd_vs_rand = paired[
        (paired["focal_method"] == "boundary_k") & (paired["baseline_method"] == "random_k")
    ]
    ent_vs_full = paired[
        (paired["focal_method"] == "entropy_k") & (paired["baseline_method"] == "full_rho90")
    ]

    best_k_ent = ent_vs_full.sort_values("mean_delta_macro_f1", ascending=False).head(1)
    qA = (
        f"Full rho90 mean F1 **{full_f1:.3f}**. "
        f"At K≤10, mean F1 by method: "
        + ", ".join(f"{m}={v:.3f}" for m, v in comp_small.items())
        + ". "
    )
    if not best_k_ent.empty and best_k_ent.iloc[0]["mean_delta_macro_f1"] > 0.01:
        qA += (
            f"Compression to K={int(best_k_ent.iloc[0]['K'])} improves over full rho90 "
            f"(Δ={best_k_ent.iloc[0]['mean_delta_macro_f1']:+.3f} for entropy_k)."
        )
    else:
        qA += "No consistent improvement from compression alone across all K."

    qB = (
        f"Entropy vs random (paired across K): mean Δ "
        f"{_nanmean(ent_vs_rand['mean_delta_macro_f1'].tolist()):+.3f}; "
        f"boundary vs random: {_nanmean(bnd_vs_rand['mean_delta_macro_f1'].tolist()):+.3f}. "
    )
    if _nanmean(bnd_vs_rand["mean_delta_macro_f1"].tolist()) > 0.01:
        qB += "**Yes** — boundary/entropy ranking beats random at matched K on average."
    else:
        qB += "**Weak/no** — selection advantage over random is small or inconsistent."

    qC = (
        f"Entropy_k vs sample-random (matched train samples): mean Δ "
        f"{_nanmean(ent_vs_sample['mean_delta_macro_f1'].tolist()):+.3f}. "
    )
    if _nanmean(ent_vs_sample["mean_delta_macro_f1"].tolist()) > 0.01:
        qC += "**Yes** — granular-ball compression outperforms sample-level reduction."
    else:
        qC += "**No** — sample-level compression is competitive; granular structure adds little."

    qD = (
        "Decomposition: (1) **compression** — fewer balls/samples regularize vs full rho90; "
        "(2) **ball selection** — entropy/boundary ranking vs random; "
        "(3) **granular structure** — ball subset vs subsampled-then-granulated samples. "
    )
    if _nanmean(ent_vs_full["mean_delta_macro_f1"].tolist()) > 0.02:
        if abs(_nanmean(ent_vs_rand["mean_delta_macro_f1"].tolist())) < 0.005:
            qD += (
                "Primarily **representation compression** (fewer balls vs full rho90). "
                "Ball ranking is neutral at fixed K; sample-level controls are competitive."
            )
        else:
            qD += "Compression is necessary; ball ranking provides additional lift."
    elif _nanmean(ent_vs_rand["mean_delta_macro_f1"].tolist()) > 0.01:
        qD += "Combination of compression + ball selection; granular structure secondary."
    else:
        qD += "Primarily **representation compression**; selection and granular structure matter less."

    n_consistent = int(dataset_level["consistent_positive"].sum()) if not dataset_level.empty else 0
    q7 = (
        f"{n_consistent} dataset×method×K cells show ≥{WIN_THRESHOLD}/10 positive seeds vs full rho90. "
        "See dataset_level_results.csv — no cherry-picking."
    )

    return {
        "qA_compression_improves": qA,
        "qB_selection_vs_random": qB,
        "qC_granular_vs_sample": qC,
        "qD_mechanism": qD,
        "q_dataset_consistency": q7,
    }


def write_plots(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _plot_curve(y_col: str, fname: str, title: str, summary_col: str | None = None) -> None:
        scol = summary_col or f"mean_{y_col}"
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in PLOT_METHODS:
            if method == "full_rho90":
                sub = df[df["method"] == "full_rho90"]
                if sub.empty:
                    continue
                val_col = y_col if y_col in sub.columns else "collapsed"
                ax.axhline(sub[val_col].mean(), linestyle="--", label="full_rho90", alpha=0.7)
                continue
            sm = summary[(summary["method"] == method)].sort_values("K")
            if sm.empty or scol not in sm.columns:
                continue
            ax.plot(sm["K"], sm[scol], marker="o", label=method)
        ax.set_xlabel("K")
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)

    if not summary.empty:
        _plot_curve("macro_f1", "macro_f1_vs_k.png", "Mean macro F1 vs K (exploratory)")
        _plot_curve("minority_f1", "minority_f1_vs_k.png", "Mean minority F1 vs K")
        _plot_curve("collapsed", "collapse_rate_vs_k.png", "Collapse rate vs K", summary_col="frac_collapsed")

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("entropy_k", "boundary_k", "random_k", "sample_random_k"):
        sm = summary[summary["method"] == method]
        if sm.empty:
            continue
        ax.scatter(sm["mean_n_samples"], sm["mean_macro_f1"], label=method)
    ax.set_xlabel("Mean represented samples")
    ax.set_ylabel("Mean macro F1")
    ax.set_title("Macro F1 vs represented sample count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "macro_f1_vs_samples.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    k_focus = 8
    methods = ["random_k", "entropy_k", "boundary_k", "entropy_boundary_k", "purity_k", "stratified_random_k"]
    vals = []
    for m in methods:
        sm = summary[(summary["method"] == m) & (summary["K"] == k_focus)]
        vals.append(float(sm["mean_macro_f1"].iloc[0]) if not sm.empty else 0.0)
    ax.bar(methods, vals)
    ax.set_title(f"Method comparison at K={k_focus}")
    ax.set_ylabel("Mean macro F1")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / f"method_comparison_k{k_focus}.png", dpi=150)
    plt.close(fig)


def write_readme(
    summary: pd.DataFrame,
    regime: pd.DataFrame,
    optima: dict[str, Any],
    answers: dict[str, str],
    protocol: dict[str, Any],
) -> None:
    regime_md = ""
    if not regime.empty:
        regime_md = regime.to_csv(index=False)

    lines = [
        "# Granular compression K-sensitivity",
        "",
        "Read-only exploratory diagnostic. **Not** a tuned deployable model.",
        "",
        "## Protocol",
        "",
        "- Frozen rho90 ball pool per dataset×seed (train only).",
        "- Subsample **K** balls via documented ranking rules; classifier unchanged.",
        "- `entropy_boundary_k`: min-max normalize entropy and boundary_score within pool, "
        "score = norm(entropy) × norm(boundary). Boundary = entropy × density (existing).",
        "- Sample controls: retain `n_samples` from entropy_k selection, rebuild rho90 on subset.",
        "- KMeans prototype control **skipped** (no existing prototype utility).",
        f"- Seeds: {SEEDS}; datasets: 17; random_k: {RANDOM_REPS} reps with deterministic seeds.",
        "",
        "## K regime (mean macro F1)",
        "",
        "```csv",
        regime_md if regime_md else "(no data)",
        "```",
        "",
        "## Optima per method (exploratory)",
        "",
        json.dumps(optima, indent=2),
        "",
        "## Questions",
        "",
        "### A. Does compression improve generalization?",
        answers["qA_compression_improves"],
        "",
        "### B. Does boundary/entropy beat random at same K?",
        answers["qB_selection_vs_random"],
        "",
        "### C. Granular vs sample compression?",
        answers["qC_granular_vs_sample"],
        "",
        "### D. Mechanism",
        answers["qD_mechanism"],
        "",
        "### Dataset consistency",
        answers["q_dataset_consistency"],
        "",
        "## Outputs",
        "",
        "- raw_results.csv",
        "- summary_by_k_method.csv",
        "- dataset_level_results.csv",
        "- paired_comparisons.csv",
        "- failure_analysis.csv",
        "- plots/*.png",
        "",
    ]
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_and_write(df: pd.DataFrame, failures: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    summary = build_summary_by_k_method(df)
    paired = build_paired_comparisons(df)
    dataset_level = build_dataset_level(df)
    regime = build_k_regime_table(summary)
    optima = find_optima(summary)

    full_sub = df[df["method"] == "full_rho90"]
    full_stats = {
        "mean_macro_f1": _nanmean(full_sub["macro_f1"].tolist()),
        "frac_collapsed": float(full_sub["collapsed"].mean()) if not full_sub.empty else float("nan"),
        "mean_n_balls": _nanmean(full_sub["n_balls"].tolist()),
    }
    answers = answer_questions(summary, paired, dataset_level, full_stats)

    summary.to_csv(SUMMARY_CSV, index=False)
    dataset_level.to_csv(DATASET_CSV, index=False)
    paired.to_csv(PAIRED_CSV, index=False)
    pd.DataFrame(failures).to_csv(FAILURE_CSV, index=False)
    write_plots(df, summary)

    protocol = {
        "read_only": True,
        "representation": REPRESENTATION,
        "K_values": K_VALUES,
        "random_reps": RANDOM_REPS,
        "ball_methods": list(BALL_METHODS),
        "sample_methods": list(SAMPLE_METHODS),
        "sample_target": "n_samples from entropy_k selection",
        "entropy_boundary_score": "norm(entropy)*norm(boundary_score) within rho90 pool",
        "boundary_score": "entropy * density (ball_utility_landscape)",
        "kmeans_prototype": "skipped — no existing utility",
        "seeds": SEEDS,
        "smoke_mode": smoke,
        "classifier": f"Gaussian GBLSTSVM ({PRIMARY_GAUSSIAN})",
        "c1": DEFAULT_C1,
        "c2": DEFAULT_C2,
    }

    meta = {
        "protocol": protocol,
        "n_raw_rows": int(len(df)),
        "full_rho90_stats": full_stats,
        "k_regime_table": regime.to_dict(orient="records"),
        "optima_by_method": optima,
        "answers": answers,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_readme(summary, regime, optima, answers, protocol)
    return meta


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(RAW_CSV)
        failures = pd.read_csv(FAILURE_CSV).to_dict(orient="records") if FAILURE_CSV.exists() else []
        analyze_and_write(df, failures, smoke=False)
        print(f"Wrote analysis to {OUT_DIR}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print(f"SMOKE: 1 dataset × seed 42")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failures.append(
                    {"dataset": desc["name"], "seed": seed, "K": float("nan"), "reason": err or "load"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue
            try:
                run_rows, run_fail = run_dataset_seed(split, seed)
                rows.extend(run_rows)
                failures.extend(run_fail)
                done += 1
                ent8 = next(
                    (r for r in run_rows if r["method"] == "entropy_k" and r["K"] == 8),
                    None,
                )
                full = next(r for r in run_rows if r["method"] == "full_rho90")
                if ent8:
                    print(
                        f"{split.name} s{seed}: full F1={full['macro_f1']:.3f} "
                        f"({full['n_balls']} balls) | entropy_k8 F1={ent8['macro_f1']:.3f}"
                    )
                if done % SAVE_EVERY == 0:
                    pd.DataFrame(rows).to_csv(RAW_CSV, index=False)
                    print(f"  [checkpoint] {done} pairs, {len(rows)} rows")
            except Exception as exc:
                failures.append(
                    {"dataset": split.name, "seed": seed, "K": float("nan"), "reason": str(exc)}
                )
                print(f"FAIL {desc['name']} seed={seed}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(RAW_CSV, index=False)
    analyze_and_write(df, failures, smoke=smoke)
    print(f"\nWrote {RAW_CSV} ({len(df)} rows)")
    print(f"Wrote outputs under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
