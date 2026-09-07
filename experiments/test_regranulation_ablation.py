"""
Re-granulation ablation experiment (read-only diagnostic).

Isolates whether re-granulating a compressed sample set outperforms singleton-ball
representation when sample selection is held fixed.

Does NOT modify src/ or prior experiments.

Run from GBLSTSVM root:
  python experiments/test_regranulation_ablation.py --smoke
  python experiments/test_regranulation_ablation.py
  python experiments/test_regranulation_ablation.py --from-csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2  # noqa: E402
from experiments.regranulation_ablation_common import (  # noqa: E402
    DELTA_EPS,
    FULL_REFERENCE,
    K_VALUES,
    MATCHED_METHODS,
    REGRANULATE_RHOS,
    SECONDARY_BASELINE,
    SEEDS,
    build_rho_from_train_indices,
    compute_ball_geometry,
    run_full_rho90_reference,
    run_matched_k_block,
    selection_hash,
)
from experiments.sample_control_reconciliation_common import pack_indices_for_csv  # noqa: E402
from experiments.test_adaptive_entropy_granulation_v2 import _global_min_label  # noqa: E402
from experiments.test_split_utility_granulation import _build_indexed_matrix  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "regranulation_ablation"
SAVE_EVERY = 25

GEOMETRY_COLS = [
    "n_balls",
    "n_singletons",
    "n_non_singletons",
    "mean_ball_size",
    "median_ball_size",
    "max_ball_size",
    "mean_purity",
    "median_purity",
    "mean_radius",
    "median_radius",
    "mean_entropy",
    "median_entropy",
    "minority_balls",
    "majority_balls",
    "minority_sample_survival",
    "majority_sample_survival",
    "successful_splits",
    "failed_splits",
    "singleton_deletions",
    "collapse_flag",
    "max_depth",
    "mean_depth",
    "rho",
]

RAW_COLS = [
    "dataset",
    "seed",
    "method",
    "K",
    "requested_k",
    "actual_retained_samples",
    "n_target_samples",
    "selection_hash",
    "full_data_reference",
    "representation_type",
    "training_path",
    "macro_f1",
    "accuracy",
    "minority_f1",
    "gmean",
    "collapsed",
    "collapse_reason",
    "classifier_ok",
    "n_balls",
    "frac_collapsed",
]


def _nanmean(v):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def _nanmedian(v):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def _nanstd(v):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a)) if a.size else float("nan")


def _cohens_d(deltas: np.ndarray) -> float:
    if deltas.size < 2:
        return float("nan")
    return float(np.mean(deltas) / (np.std(deltas, ddof=1) + 1e-12))


def _row_from_result(split_name: str, seed: int, r: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": split_name,
        "seed": seed,
        "method": r.get("method", ""),
        "K": r.get("K", r.get("requested_k", float("nan"))),
        "requested_k": r.get("requested_k", r.get("K", float("nan"))),
        "actual_retained_samples": r.get(
            "actual_retained_samples", r.get("actual_samples", float("nan"))
        ),
        "n_target_samples": r.get("n_target_samples", float("nan")),
        "selection_hash": r.get("selection_hash", ""),
        "full_data_reference": bool(r.get("full_data_reference", False)),
        "representation_type": r.get("representation_type", ""),
        "training_path": r.get("training_path", ""),
        "macro_f1": r.get("macro_f1", float("nan")),
        "accuracy": r.get("accuracy", float("nan")),
        "minority_f1": r.get("minority_f1", float("nan")),
        "gmean": r.get("gmean", float("nan")),
        "collapsed": r.get("collapsed", 0),
        "collapse_reason": r.get("collapse_reason", ""),
        "classifier_ok": r.get("classifier_ok", False),
        "n_balls": r.get("n_balls", float("nan")),
        "rho": r.get("rho", float("nan")),
        "n_rho90_pool": r.get("n_rho90_pool", float("nan")),
        "selected_indices_csv": pack_indices_for_csv(r.get("selected_indices", np.array([]))),
    }


def _geometry_row(split_name: str, seed: int, r: dict[str, Any]) -> dict[str, Any]:
    row = {
        "dataset": split_name,
        "seed": seed,
        "method": r.get("method", ""),
        "K": r.get("K", r.get("requested_k", float("nan"))),
        "requested_k": r.get("requested_k", r.get("K", float("nan"))),
        "selection_hash": r.get("selection_hash", ""),
        "full_data_reference": bool(r.get("full_data_reference", False)),
    }
    for col in GEOMETRY_COLS:
        row[col] = r.get(col, float("nan"))
    row["classifier_collapsed"] = r.get("collapsed", 0)
    return row


def run_dataset_seed(split, seed: int, k_values: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict[str, Any]] = []
    geom_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    try:
        r_full = run_full_rho90_reference(split, seed)
        rows.append(_row_from_result(split.name, seed, r_full))
        geom_rows.append(_geometry_row(split.name, seed, r_full))
    except Exception as exc:
        failures.append({"dataset": split.name, "seed": seed, "K": float("nan"), "reason": f"full_rho90:{exc}"})

    for k in k_values:
        try:
            results, skip = run_matched_k_block(split, seed, k)
            if skip:
                failures.append(
                    {"dataset": split.name, "seed": seed, "K": k, "reason": skip["skip_reason"]}
                )
                continue
            for r in results:
                rows.append(_row_from_result(split.name, seed, r))
                geom_rows.append(_geometry_row(split.name, seed, r))
        except ValueError as exc:
            failures.append(
                {
                    "dataset": split.name,
                    "seed": seed,
                    "K": k,
                    "reason": f"fairness_violation:{exc}",
                }
            )
        except Exception as exc:
            failures.append(
                {"dataset": split.name, "seed": seed, "K": k, "reason": f"exception:{exc}"}
            )

    return rows, geom_rows, failures


def _paired_stats(
    df: pd.DataFrame,
    focal: str,
    baseline: str,
    *,
    k: float | None = None,
) -> dict[str, Any]:
    f = df[(df["method"] == focal) & (~df["full_data_reference"])]
    b = df[(df["method"] == baseline) & (~df["full_data_reference"])]
    if k is not None:
        f = f[f["K"] == k]
        b = b[b["K"] == k]
    keys = set(zip(f["dataset"], f["seed"], f["K"])) & set(zip(b["dataset"], b["seed"], b["K"]))
    deltas = []
    f_vals = []
    b_vals = []
    for ds, sd, kk in keys:
        fv = float(f[(f["dataset"] == ds) & (f["seed"] == sd) & (f["K"] == kk)]["macro_f1"].iloc[0])
        bv = float(b[(b["dataset"] == ds) & (b["seed"] == sd) & (b["K"] == kk)]["macro_f1"].iloc[0])
        if np.isfinite(fv) and np.isfinite(bv):
            deltas.append(fv - bv)
            f_vals.append(fv)
            b_vals.append(bv)
    if not deltas:
        return {
            "focal": focal,
            "baseline": baseline,
            "K": k if k is not None else "all",
            "n_pairs": 0,
            "mean_delta_macro_f1": float("nan"),
            "wilcoxon_p": float("nan"),
        }
    arr = np.asarray(deltas)
    wins = int(np.sum(arr > DELTA_EPS))
    ties = int(np.sum(np.abs(arr) <= DELTA_EPS))
    losses = int(np.sum(arr < -DELTA_EPS))
    try:
        wstat, wp = wilcoxon(arr) if not np.allclose(arr, 0) else (0.0, 1.0)
    except ValueError:
        wstat, wp = float("nan"), float("nan")
    return {
        "focal": focal,
        "baseline": baseline,
        "K": k if k is not None else "all",
        "n_pairs": len(deltas),
        "mean_macro_f1_focal": _nanmean(f_vals),
        "median_macro_f1_focal": _nanmedian(f_vals),
        "std_macro_f1_focal": _nanstd(f_vals),
        "mean_macro_f1_baseline": _nanmean(b_vals),
        "median_macro_f1_baseline": _nanmedian(b_vals),
        "mean_delta_macro_f1": float(np.mean(arr)),
        "median_delta_macro_f1": float(np.median(arr)),
        "std_delta_macro_f1": float(np.std(arr)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "pct_wins": 100.0 * wins / len(deltas),
        "wilcoxon_stat": float(wstat) if np.isfinite(wstat) else float("nan"),
        "wilcoxon_p": float(wp) if np.isfinite(wp) else float("nan"),
        "cohens_d": _cohens_d(arr),
        "frac_collapsed_focal": float(
            np.mean(
                [
                    f[(f["dataset"] == ds) & (f["seed"] == sd) & (f["K"] == kk)]["collapsed"].iloc[0]
                    for ds, sd, kk in keys
                ]
            )
        ),
        "mean_n_balls_focal": _nanmean(
            [
                f[(f["dataset"] == ds) & (f["seed"] == sd) & (f["K"] == kk)]["n_balls"].iloc[0]
                for ds, sd, kk in keys
            ]
        ),
    }


def build_paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in REGRANULATE_RHOS:
        rows.append(_paired_stats(df, method, "singleton", k=None))
        for k in K_VALUES:
            rows.append(_paired_stats(df, method, "singleton", k=k))
    for method in REGRANULATE_RHOS:
        rows.append(_paired_stats(df, method, SECONDARY_BASELINE, k=None))
        for k in K_VALUES:
            rows.append(_paired_stats(df, method, SECONDARY_BASELINE, k=k))
    return pd.DataFrame(rows)


def build_rho_k_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matched = df[~df["full_data_reference"]]
    methods = list(MATCHED_METHODS) + [SECONDARY_BASELINE]
    for method in methods:
        for k in K_VALUES:
            sub = matched[(matched["method"] == method) & (matched["K"] == k)]
            if sub.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "rho": float(sub["rho"].iloc[0]) if "rho" in sub.columns else float("nan"),
                    "K": k,
                    "n_runs": len(sub),
                    "mean_macro_f1": _nanmean(sub["macro_f1"]),
                    "median_macro_f1": _nanmedian(sub["macro_f1"]),
                    "std_macro_f1": _nanstd(sub["macro_f1"]),
                    "frac_collapsed": float(sub["collapsed"].mean()),
                    "mean_n_balls": _nanmean(sub["n_balls"]),
                    "median_n_balls": _nanmedian(sub["n_balls"]),
                    "mean_actual_retained_samples": _nanmean(sub["actual_retained_samples"]),
                }
            )
    return pd.DataFrame(rows)


def build_dataset_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matched = df[~df["full_data_reference"]]
    for (dataset, method, k), g in matched.groupby(["dataset", "method", "K"]):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "K": k,
                "n_seeds": len(g),
                "mean_macro_f1": _nanmean(g["macro_f1"]),
                "median_macro_f1": _nanmedian(g["macro_f1"]),
                "frac_collapsed": float(g["collapsed"].mean()),
                "mean_n_balls": _nanmean(g["n_balls"]),
            }
        )
    return pd.DataFrame(rows)


def build_collapse_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in list(MATCHED_METHODS) + [SECONDARY_BASELINE, FULL_REFERENCE]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        rows.append(
            {
                "method": method,
                "n_runs": len(sub),
                "frac_collapsed": float(sub["collapsed"].mean()),
                "mean_macro_f1_all": _nanmean(sub["macro_f1"]),
                "mean_macro_f1_non_collapsed": _nanmean(
                    sub.loc[sub["collapsed"] == 0, "macro_f1"]
                ),
                "mean_n_balls": _nanmean(sub["n_balls"]),
                "median_n_balls": _nanmedian(sub["n_balls"]),
            }
        )
    return pd.DataFrame(rows)


def build_correlation_analysis(df: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matched = df[(~df["full_data_reference"]) & (df["method"].isin(REGRANULATE_RHOS.keys()))]
    if not matched.empty:
        x = matched["n_balls"].astype(float)
        y = matched["macro_f1"].astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            pr, pp = pearsonr(x[mask], y[mask])
            sr, sp = spearmanr(x[mask], y[mask])
            rows.append(
                {
                    "comparison": "n_balls_vs_macro_f1",
                    "n": int(mask.sum()),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                }
            )

    p_singleton = paired[
        (paired["baseline"] == "singleton") & (paired["K"] == "all")
    ]
    for _, pr in p_singleton.iterrows():
        method = pr["focal"]
        sub = matched[matched["method"] == method]
        sing = df[(df["method"] == "singleton") & (~df["full_data_reference"])]
        keys = set(zip(sub["dataset"], sub["seed"], sub["K"])) & set(
            zip(sing["dataset"], sing["seed"], sing["K"])
        )
        deltas = []
        nballs = []
        for ds, sd, kk in keys:
            fv = float(sub[(sub["dataset"] == ds) & (sub["seed"] == sd) & (sub["K"] == kk)]["macro_f1"].iloc[0])
            bv = float(sing[(sing["dataset"] == ds) & (sing["seed"] == sd) & (sing["K"] == kk)]["macro_f1"].iloc[0])
            nb = float(sub[(sub["dataset"] == ds) & (sub["seed"] == sd) & (sub["K"] == kk)]["n_balls"].iloc[0])
            if np.isfinite(fv) and np.isfinite(bv):
                deltas.append(fv - bv)
                nballs.append(nb)
        if len(deltas) >= 5:
            pr_r, pr_p = pearsonr(nballs, deltas)
            rows.append(
                {
                    "comparison": f"n_balls_vs_delta_f1_singleton_{method}",
                    "n": len(deltas),
                    "pearson_r": float(pr_r),
                    "pearson_p": float(pr_p),
                    "spearman_r": float("nan"),
                    "spearman_p": float("nan"),
                }
            )

    for col in ("mean_purity", "mean_radius", "minority_sample_survival"):
        geom = pd.read_csv(OUT_DIR / "geometry_results.csv") if (OUT_DIR / "geometry_results.csv").exists() else pd.DataFrame()
        if geom.empty or col not in geom.columns:
            continue
        gsub = geom[geom["method"].isin(REGRANULATE_RHOS.keys())].merge(
            matched[["dataset", "seed", "K", "method", "macro_f1"]],
            on=["dataset", "seed", "K", "method"],
        )
        xv = gsub[col].astype(float)
        yv = gsub["macro_f1"].astype(float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        if mask.sum() >= 5:
            pr, pp = pearsonr(xv[mask], yv[mask])
            rows.append(
                {
                    "comparison": f"{col}_vs_macro_f1",
                    "n": int(mask.sum()),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float("nan"),
                    "spearman_p": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def determine_case(
    paired: pd.DataFrame,
    corr: pd.DataFrame,
    rho_summary: pd.DataFrame,
) -> tuple[str, str, dict[str, Any]]:
    p_vs_sing = paired[
        (paired["baseline"] == "singleton")
        & (paired["K"] == "all")
        & (paired["focal"].isin(REGRANULATE_RHOS.keys()))
        & (paired["n_pairs"] > 0)
    ]
    p_vs_korig = paired[
        (paired["baseline"] == SECONDARY_BASELINE)
        & (paired["K"] == "all")
        & (paired["focal"].isin(REGRANULATE_RHOS.keys()))
        & (paired["n_pairs"] > 0)
    ]

    mean_delta_sing = (
        _nanmean(p_vs_sing["mean_delta_macro_f1"].tolist()) if not p_vs_sing.empty else 0.0
    )
    mean_delta_korig = (
        _nanmean(p_vs_korig["mean_delta_macro_f1"].tolist()) if not p_vs_korig.empty else 0.0
    )
    sig_sing = bool(
        not p_vs_sing.empty
        and np.any(p_vs_sing["wilcoxon_p"].astype(float) < 0.05)
        and mean_delta_sing > DELTA_EPS
    )
    sig_korig = bool(
        not p_vs_korig.empty
        and np.any(p_vs_korig["wilcoxon_p"].astype(float) < 0.05)
        and mean_delta_korig > DELTA_EPS
    )
    approx_korig = abs(mean_delta_korig) < 0.01
    approx_sing = abs(mean_delta_sing) < 0.01

    ball_corr = corr[corr["comparison"] == "n_balls_vs_macro_f1"] if not corr.empty and "comparison" in corr.columns else pd.DataFrame()
    delta_corr = (
        corr[corr["comparison"].str.contains("delta_f1_singleton", na=False)]
        if not corr.empty and "comparison" in corr.columns
        else pd.DataFrame()
    )
    strong_ball = False
    if not ball_corr.empty and abs(float(ball_corr.iloc[0]["pearson_r"])) > 0.6:
        strong_ball = True
    if not delta_corr.empty:
        rs = [abs(float(r["pearson_r"])) for _, r in delta_corr.iterrows() if np.isfinite(r["pearson_r"])]
        if rs and max(rs) > 0.6:
            strong_ball = True

    best_rho_row = rho_summary[
        rho_summary["method"].isin(REGRANULATE_RHOS.keys())
    ].sort_values("mean_macro_f1", ascending=False)
    best_descriptive = (
        best_rho_row.iloc[0].to_dict() if not best_rho_row.empty else {}
    )

    meta = {
        "mean_delta_vs_singleton": mean_delta_sing,
        "mean_delta_vs_sample_compression": mean_delta_korig,
        "significant_vs_singleton": sig_sing,
        "significant_vs_sample_compression": sig_korig,
        "performance_tracks_ball_count": strong_ball,
        "post_hoc_descriptive_optimum": best_descriptive,
    }

    if strong_ball and not approx_sing:
        case = "D"
        text = (
            "> The apparent effect is better explained by representation complexity "
            "than by a specific granular-ball mechanism."
        )
    elif approx_sing:
        case = "C"
        text = (
            "> There is no robust evidence that the second granulation stage contributes "
            "materially once the retained sample set is fixed."
        )
    elif sig_sing and sig_korig and not approx_korig:
        case = "A"
        text = "> Evidence supports a granular-specific contribution from re-granulation."
    elif sig_sing or mean_delta_sing > 0.02:
        case = "B"
        text = (
            "> Re-granulation materially changes the compressed representation, but there is "
            "no evidence that granular structure provides an advantage beyond ordinary "
            "sample-level compression."
        )
    else:
        case = "C"
        text = (
            "> There is no robust evidence that the second granulation stage contributes "
            "materially once the retained sample set is fixed."
        )

    return case, text, meta


def write_plots(df: pd.DataFrame, paired: pd.DataFrame, rho_summary: pd.DataFrame) -> None:
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    matched = df[~df["full_data_reference"]]

    regran = rho_summary[rho_summary["method"].isin(REGRANULATE_RHOS.keys())].copy()
    if not regran.empty:
        regran["rho_val"] = regran["method"].map(REGRANULATE_RHOS)
        agg = regran.groupby("rho_val").agg(
            mean_f1=("mean_macro_f1", "mean"),
            collapse=("frac_collapsed", "mean"),
            mean_balls=("mean_n_balls", "mean"),
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(agg.index, agg["mean_f1"], marker="o")
        ax.set_xlabel("rho")
        ax.set_ylabel("Mean macro F1")
        ax.set_title("Macro F1 vs rho (pooled K)")
        fig.tight_layout()
        fig.savefig(plot_dir / "macro_f1_vs_rho.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(agg.index, agg["collapse"], marker="o", color="coral")
        ax.set_xlabel("rho")
        ax.set_ylabel("Collapse rate")
        ax.set_title("Collapse rate vs rho")
        fig.tight_layout()
        fig.savefig(plot_dir / "collapse_rate_vs_rho.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(agg.index, agg["mean_balls"], marker="o", color="green")
        ax.set_xlabel("rho")
        ax.set_ylabel("Mean n_balls")
        ax.set_title("Ball count vs rho")
        fig.tight_layout()
        fig.savefig(plot_dir / "ball_count_vs_rho.png", dpi=150)
        plt.close(fig)

    p_sing = paired[(paired["baseline"] == "singleton") & (paired["K"] == "all")]
    if not p_sing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [m.replace("regranulate_", "") for m in p_sing["focal"]]
        ax.bar(labels, p_sing["mean_delta_macro_f1"])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Delta F1 (regranulate - singleton)")
        ax.set_title("ΔF1 vs singleton")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(plot_dir / "delta_f1_vs_singleton.png", dpi=150)
        plt.close(fig)

    p_k = paired[(paired["baseline"] == SECONDARY_BASELINE) & (paired["K"] == "all")]
    if not p_k.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [m.replace("regranulate_", "") for m in p_k["focal"]]
        ax.bar(labels, p_k["mean_delta_macro_f1"])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"Delta F1 (regranulate - {SECONDARY_BASELINE})")
        ax.set_title("Delta F1 vs sample compression baseline")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(plot_dir / "delta_f1_vs_sample_compression.png", dpi=150)
        plt.close(fig)

    regran_runs = matched[matched["method"].isin(REGRANULATE_RHOS.keys())]
    if not regran_runs.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(regran_runs["n_balls"], regran_runs["macro_f1"], alpha=0.25, s=12)
        ax.set_xlabel("n_balls")
        ax.set_ylabel("Macro F1")
        ax.set_title("Ball count vs Macro F1 (re-granulated)")
        fig.tight_layout()
        fig.savefig(plot_dir / "ball_count_vs_macro_f1.png", dpi=150)
        plt.close(fig)

        geom = pd.read_csv(OUT_DIR / "geometry_results.csv") if (OUT_DIR / "geometry_results.csv").exists() else pd.DataFrame()
        if not geom.empty and "minority_sample_survival" in geom.columns:
            g = geom[geom["method"].isin(REGRANULATE_RHOS.keys())].merge(
                matched[["dataset", "seed", "K", "method", "macro_f1"]],
                on=["dataset", "seed", "K", "method"],
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(g["minority_sample_survival"], g["macro_f1"], alpha=0.25, s=12)
            ax.set_xlabel("Minority sample survival")
            ax.set_ylabel("Macro F1")
            ax.set_title("Minority survival vs Macro F1")
            fig.tight_layout()
            fig.savefig(plot_dir / "minority_survival_vs_macro_f1.png", dpi=150)
            plt.close(fig)


def print_final_report(
    df: pd.DataFrame,
    paired: pd.DataFrame,
    rho_summary: pd.DataFrame,
    case: str,
    conclusion: str,
    meta: dict[str, Any],
    hash_ok: bool,
) -> str:
    sing = df[(df["method"] == "singleton") & (~df["full_data_reference"])]
    korig = df[(df["method"] == SECONDARY_BASELINE) & (~df["full_data_reference"])]

    def _method_block(method: str, label: str) -> str:
        sub = df[(df["method"] == method) & (~df["full_data_reference"])]
        p = paired[(paired["focal"] == method) & (paired["baseline"] == "singleton") & (paired["K"] == "all")]
        delta = float(p.iloc[0]["mean_delta_macro_f1"]) if not p.empty else float("nan")
        return (
            f"{label}:\n"
            f"    Mean F1: {_nanmean(sub['macro_f1']):.4f}\n"
            f"    Collapse: {sub['collapsed'].mean():.3f}\n"
            f"    Mean balls: {_nanmean(sub['n_balls']):.1f}\n"
            f"    Delta vs singleton: {delta:+.4f}\n"
        )

    best = meta.get("post_hoc_descriptive_optimum", {})
    best_rho = best.get("method", "n/a")
    tracks = meta.get("performance_tracks_ball_count", False)
    tracks_label = "YES" if tracks else ("UNCLEAR" if abs(meta.get("mean_delta_vs_singleton", 0)) > 0.01 else "NO")
    beat_sing = meta.get("mean_delta_vs_singleton", 0) > 0.01
    beat_k = meta.get("mean_delta_vs_sample_compression", 0) > 0.01

    if case == "A":
        next_step = "Document granular-specific mechanism; do not add new heuristics without complexity control."
    elif case == "B":
        next_step = "Treat gains as sample-compression effect; stop granular-specific branch."
    elif case == "C":
        next_step = "Stop granular-structure branch; compression/weighting explains prior gap."
    else:
        next_step = "Pursue complexity-controlled compression rather than rho tuning."

    report = f"""
============================================================
RE-GRANULATION ABLATION
============================================================

Runs:
    17 datasets × 10 seeds × 6 K values × methods

Selected samples identical across branches:
    {'YES' if hash_ok else 'NO'}

Singleton:
    Mean F1: {_nanmean(sing['macro_f1']):.4f}
    Collapse: {sing['collapsed'].mean():.3f}
    Mean balls: {_nanmean(sing['n_balls']):.1f}

{_method_block('regranulate_rho80', 'Regranulate rho=0.80')}
{_method_block('regranulate_rho85', 'Regranulate rho=0.85')}
{_method_block('regranulate_rho90', 'Regranulate rho=0.90')}
{_method_block('regranulate_rho95', 'Regranulate rho=0.95')}
{_method_block('regranulate_rho98', 'Regranulate rho=0.98')}

Sample-level compression baseline:
    Mean F1: {_nanmean(korig['macro_f1']):.4f}

Best descriptive rho (post-hoc descriptive optimum):
    {best_rho} (mean F1={best.get('mean_macro_f1', float('nan')):.4f})

Does re-granulation beat singleton?
    {'YES' if beat_sing else 'NO'}

Does re-granulation beat sample compression?
    {'YES' if beat_k else 'NO'}

Does performance primarily track ball count?
    {tracks_label}

FINAL CASE:
    {case}

SCIENTIFIC CONCLUSION:
    {conclusion}

RECOMMENDED NEXT STEP:
    {next_step}
============================================================
"""
    print(report)
    return report


def _normalize_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[out["method"] == "singleton_ball_subset", "method"] = "singleton"
    return out


def analyze_and_write(
    df: pd.DataFrame,
    geom_df: pd.DataFrame,
    failures: list[dict],
    hash_ok: bool,
) -> None:
    from experiments.sample_control_reconciliation_common import CONTROL_SEMANTICS

    df = _normalize_methods(df)
    geom_df = _normalize_methods(geom_df) if "method" in geom_df.columns else geom_df

    paired = build_paired_comparisons(df)
    rho_summary = build_rho_k_summary(df)
    dataset_level = build_dataset_level(df)
    collapse = build_collapse_analysis(df)

    geom_df.to_csv(OUT_DIR / "geometry_results.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_comparisons.csv", index=False)
    dataset_level.to_csv(OUT_DIR / "dataset_level_results.csv", index=False)
    rho_summary.to_csv(OUT_DIR / "rho_k_summary.csv", index=False)
    collapse.to_csv(OUT_DIR / "collapse_analysis.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT_DIR / "failure_analysis.csv", index=False)

    corr = build_correlation_analysis(df, paired)
    corr.to_csv(OUT_DIR / "correlation_analysis.csv", index=False)

    case, conclusion, meta = determine_case(paired, corr, rho_summary)
    write_plots(df, paired, rho_summary)
    report = print_final_report(df, paired, rho_summary, case, conclusion, meta, hash_ok)

    summary = {
        "protocol": {
            "K_values": K_VALUES,
            "K_meaning": CONTROL_SEMANTICS["sample_stratified_k_original"]["k_meaning"],
            "sample_selection": "sample_stratified_k (deterministic_rng_seed method=sample_stratified_k)",
            "regranulate_rhos": REGRANULATE_RHOS,
            "classifier": f"Gaussian GBLSTSVM ({PRIMARY_GAUSSIAN})",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
        },
        "n_raw_rows": len(df),
        "hash_verification_passed": hash_ok,
        "case": case,
        "conclusion": conclusion,
        "meta": meta,
        "final_report": report,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = [
        "# Re-granulation ablation",
        "",
        "Controlled experiment: fixed stratified sample subset; compare singleton vs re-granulation at multiple rho.",
        "",
        "## Design",
        "",
        "- **Matched subset**: one `sample_stratified_k` selection per dataset×seed×K.",
        "- **Singleton**: existing reconciliation singleton-ball control (`y_sub` weights).",
        "- **Re-granulate**: rho ∈ {0.80, 0.85, 0.90, 0.95, 0.98} on identical indices (`full y_train` weights).",
        "- **Secondary baseline**: `sample_stratified_k_original` (rho90 rebuild on subset).",
        "- **Reference**: `full_rho90` on full train (`full_data_reference=True`).",
        "",
        "## Conclusion (CASE {})".format(case),
        "",
        conclusion,
        "",
        "### Post-hoc descriptive optimum",
        "",
        "Best mean F1 among re-granulate methods (NOT a deployable selected hyperparameter):",
        "",
        "```json",
        json.dumps(meta.get("post_hoc_descriptive_optimum", {}), indent=2),
        "```",
        "",
        report,
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--from-csv", action="store_true")
    parser.add_argument("--k-values", nargs="*", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k_values = args.k_values or K_VALUES

    if args.from_csv:
        df = _normalize_methods(pd.read_csv(OUT_DIR / "raw_results.csv"))
        geom_df = _normalize_methods(pd.read_csv(OUT_DIR / "geometry_results.csv"))
        failures = (
            pd.read_csv(OUT_DIR / "failure_analysis.csv").to_dict(orient="records")
            if (OUT_DIR / "failure_analysis.csv").exists()
            else []
        )
        hash_ok = not any("fairness_violation" in str(f.get("reason", "")) for f in failures)
        analyze_and_write(df, geom_df, failures, hash_ok)
        return 0

    descriptors = discover_available_datasets()
    seeds = [0] if args.smoke else SEEDS
    if args.smoke:
        descriptors = descriptors[:1]

    all_rows: list[dict] = []
    all_geom: list[dict] = []
    all_fail: list[dict] = []
    done = 0
    hash_ok = True

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                all_fail.append({"dataset": desc["name"], "seed": seed, "reason": err})
                continue
            rows, geom, fails = run_dataset_seed(split, seed, k_values)
            all_rows.extend(rows)
            all_geom.extend(geom)
            all_fail.extend(fails)
            if any("fairness_violation" in str(f.get("reason", "")) for f in fails):
                hash_ok = False
            done += 1
            if done % SAVE_EVERY == 0:
                pd.DataFrame(all_rows).to_csv(OUT_DIR / "raw_results.csv", index=False)
                pd.DataFrame(all_geom).to_csv(OUT_DIR / "geometry_results.csv", index=False)
                print(f"Checkpoint {done}")

    df = _normalize_methods(pd.DataFrame(all_rows))
    geom_df = _normalize_methods(pd.DataFrame(all_geom))
    df.to_csv(OUT_DIR / "raw_results.csv", index=False)
    geom_df.to_csv(OUT_DIR / "geometry_results.csv", index=False)
    analyze_and_write(df, geom_df, all_fail, hash_ok)
    print(f"Done -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
