"""
Sample-control reconciliation experiment (read-only diagnostic).

Reconciles the discrepancy between K-sensitivity sample_stratified_k and
validation sample_stratified_matched controls.

Run from GBLSTSVM root:
  python experiments/test_sample_control_reconciliation.py --smoke
  python experiments/test_sample_control_reconciliation.py
  python experiments/test_sample_control_reconciliation.py --from-csv
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
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2  # noqa: E402
from experiments.sample_control_reconciliation_common import (  # noqa: E402
    CONTROL_SEMANTICS,
    K_VALUES,
    SEEDS,
    class_counts,
    jaccard_similarity,
    pack_indices_for_csv,
    run_granular_compressed_entropy_k,
    run_raw_sample_subset_proxy,
    run_sample_stratified_k_original,
    run_sample_stratified_matched_original,
    run_singleton_ball_subset,
    selection_hash,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "sample_control_reconciliation"
DELTA_EPS = 1e-6
SAVE_EVERY = 25

METHODS = (
    "sample_stratified_k_original",
    "sample_stratified_matched_original",
    "singleton_ball_subset",
    "raw_sample_subset",
    "granular_compressed_entropy_k",
)


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


def run_dataset_seed(split, seed: int, k_values: list[int]) -> tuple[list[dict], list[dict]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for k in k_values:
        try:
            r_k = run_sample_stratified_k_original(split, seed, k)
            if r_k.get("skipped"):
                failures.append(
                    {"dataset": split.name, "seed": seed, "K": k, "method": "k_original", "reason": r_k["skip_reason"]}
                )
                continue

            train_idx = r_k["selected_indices"]
            r_m = run_sample_stratified_matched_original(split, seed, k)
            r_s = run_singleton_ball_subset(split, seed, k, train_idx)
            r_raw = run_raw_sample_subset_proxy(split, seed, k, train_idx)
            r_g = run_granular_compressed_entropy_k(split, seed, k)

            for r in (r_k, r_m, r_s, r_raw, r_g):
                if r.get("skipped"):
                    continue
                idx = r.get("selected_indices", np.array([]))
                row = {
                    "dataset": split.name,
                    "seed": seed,
                    "method": r["method"],
                    "K": k,
                    "requested_k": k,
                    "actual_samples": r.get("actual_samples", r.get("n_samples_represented", float("nan"))),
                    "minority_count": r.get("minority_count", r.get("minority_samples_represented", float("nan"))),
                    "majority_count": r.get("majority_count", r.get("majority_samples_represented", float("nan"))),
                    "class_ratio_minority": r.get("class_ratio_minority", float("nan")),
                    "n_balls": r.get("n_balls", float("nan")),
                    "n_singleton_balls": r.get("n_singleton_balls", 0),
                    "n_non_singleton_balls": r.get("n_non_singleton_balls", r.get("n_balls", 0)),
                    "n_target_samples": r.get("n_target_samples", float("nan")),
                    "selection_hash": r.get("selection_hash", ""),
                    "selected_indices_csv": pack_indices_for_csv(idx) if len(idx) else "",
                    "representation_type": r.get("representation_type", ""),
                    "training_path": r.get("training_path", ""),
                    "macro_f1": r.get("macro_f1", float("nan")),
                    "accuracy": r.get("accuracy", float("nan")),
                    "minority_recall": r.get("minority_recall", float("nan")),
                    "majority_recall": r.get("majority_recall", float("nan")),
                    "collapsed": r.get("collapsed", 0),
                    "collapse_reason": r.get("collapse_reason", ""),
                    "classifier_ok": r.get("classifier_ok", False),
                }
                rows.append(row)
        except Exception as exc:
            failures.append(
                {"dataset": split.name, "seed": seed, "K": k, "reason": f"exception:{exc}"}
            )

    return rows, failures


def build_control_definitions() -> pd.DataFrame:
    rows = []
    for control, meta in CONTROL_SEMANTICS.items():
        rows.append({"control": control, **meta})
    return pd.DataFrame(rows)


def build_discrepancy_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a_name = "sample_stratified_k_original"
    b_name = "sample_stratified_matched_original"
    for (dataset, seed, k), g in df.groupby(["dataset", "seed", "K"]):
        ga = g[g["method"] == a_name]
        gb = g[g["method"] == b_name]
        if ga.empty or gb.empty:
            continue
        ra, rb = ga.iloc[0], gb.iloc[0]
        idx_a = np.array([int(x) for x in ra["selected_indices_csv"].split(",") if x], dtype=int)
        idx_b = np.array([int(x) for x in rb["selected_indices_csv"].split(",") if x], dtype=int)
        jacc = jaccard_similarity(idx_a, idx_b) if len(idx_a) and len(idx_b) else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "K": k,
                "sample_count_A": ra["actual_samples"],
                "sample_count_B": rb["actual_samples"],
                "sample_count_delta": rb["actual_samples"] - ra["actual_samples"],
                "minority_A": ra["minority_count"],
                "minority_B": rb["minority_count"],
                "majority_A": ra["majority_count"],
                "majority_B": rb["majority_count"],
                "jaccard_similarity": jacc,
                "same_selected_samples": int(
                    len(idx_a) == len(idx_b) and set(idx_a) == set(idx_b)
                ),
                "hash_A": ra["selection_hash"],
                "hash_B": rb["selection_hash"],
                "f1_A": ra["macro_f1"],
                "f1_B": rb["macro_f1"],
                "f1_delta": rb["macro_f1"] - ra["macro_f1"],
                "collapse_A": ra["collapsed"],
                "collapse_B": rb["collapsed"],
                "repr_A": ra["representation_type"],
                "repr_B": rb["representation_type"],
                "training_path_A": ra["training_path"],
                "training_path_B": rb["training_path"],
                "semantically_matched": int(
                    ra["actual_samples"] == rb["actual_samples"]
                    and ra["minority_count"] == rb["minority_count"]
                    and ra["majority_count"] == rb["majority_count"]
                    and set(idx_a) == set(idx_b)
                    and ra["representation_type"] == rb["representation_type"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired(df: pd.DataFrame, focal: str, baseline: str, k: float | None = None) -> dict[str, Any]:
    f = df[df["method"] == focal]
    b = df[df["method"] == baseline]
    if k is not None:
        f = f[f["K"] == k]
        b = b[b["K"] == k]
    keys = set(zip(f["dataset"], f["seed"], f["K"])) & set(zip(b["dataset"], b["seed"], b["K"]))
    deltas = []
    for key in keys:
        ds, sd, kk = key
        fv = float(f[(f["dataset"] == ds) & (f["seed"] == sd) & (f["K"] == kk)]["macro_f1"].iloc[0])
        bv = float(b[(b["dataset"] == ds) & (b["seed"] == sd) & (b["K"] == kk)]["macro_f1"].iloc[0])
        if np.isfinite(fv) and np.isfinite(bv):
            deltas.append(fv - bv)
    if not deltas:
        return {"focal": focal, "baseline": baseline, "K": k, "n_pairs": 0}
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
        "mean_macro_f1_focal": _nanmean(
            [float(f[(f["dataset"] == ds) & (f["seed"] == sd) & (f["K"] == kk)]["macro_f1"].iloc[0]) for ds, sd, kk in keys]
        ),
        "mean_macro_f1_baseline": _nanmean(
            [float(b[(b["dataset"] == ds) & (b["seed"] == sd) & (b["K"] == kk)]["macro_f1"].iloc[0]) for ds, sd, kk in keys]
        ),
        "mean_delta_macro_f1": float(np.mean(arr)),
        "median_delta_macro_f1": float(np.median(arr)),
        "std_delta_macro_f1": float(np.std(arr)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "pct_wins": 100.0 * wins / len(deltas),
        "wilcoxon_stat": float(wstat) if np.isfinite(wstat) else float("nan"),
        "wilcoxon_p": float(wp) if np.isfinite(wp) else float("nan"),
        "frac_collapsed_focal": float(
            np.mean(
                [
                    f[(f["dataset"] == ds) & (f["seed"] == sd) & (f["K"] == kk)]["collapsed"].iloc[0]
                    for ds, sd, kk in keys
                ]
            )
        ),
        "frac_collapsed_baseline": float(
            np.mean(
                [
                    b[(b["dataset"] == ds) & (b["seed"] == sd) & (b["K"] == kk)]["collapsed"].iloc[0]
                    for ds, sd, kk in keys
                ]
            )
        ),
    }


def build_paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("sample_stratified_k_original", "sample_stratified_matched_original"),
        ("raw_sample_subset", "singleton_ball_subset"),
        ("granular_compressed_entropy_k", "raw_sample_subset"),
        ("granular_compressed_entropy_k", "singleton_ball_subset"),
        ("granular_compressed_entropy_k", "sample_stratified_k_original"),
    ]
    rows = []
    for focal, baseline in pairs:
        rows.append(_paired(df, focal, baseline, k=None))
        for k in K_VALUES:
            rows.append(_paired(df, focal, baseline, k=k))
    return pd.DataFrame(rows)


def build_dataset_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, k), g in df.groupby(["dataset", "method", "K"]):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "K": k,
                "n_seeds": len(g),
                "mean_macro_f1": _nanmean(g["macro_f1"]),
                "median_macro_f1": _nanmedian(g["macro_f1"]),
                "frac_collapsed": float(g["collapsed"].mean()),
                "mean_actual_samples": _nanmean(g["actual_samples"]),
            }
        )
    return pd.DataFrame(rows)


def determine_conclusion(paired: pd.DataFrame, discrepancy: pd.DataFrame) -> tuple[str, str]:
    disc = discrepancy.copy()
    same_frac = float(disc["same_selected_samples"].mean()) if not disc.empty else 0.0
    sem_match_frac = float(disc["semantically_matched"].mean()) if not disc.empty else 0.0
    repr_diff = (
        disc["repr_A"].iloc[0] != disc["repr_B"].iloc[0] if not disc.empty else True
    )

    p_k_vs_m = paired[
        (paired["focal"] == "sample_stratified_k_original")
        & (paired["baseline"] == "sample_stratified_matched_original")
        & (paired["K"] == "all")
    ]
    p_g_vs_raw = paired[
        (paired["focal"] == "granular_compressed_entropy_k")
        & (paired["baseline"] == "raw_sample_subset")
        & (paired["K"] == "all")
    ]
    p_g_vs_sing = paired[
        (paired["focal"] == "granular_compressed_entropy_k")
        & (paired["baseline"] == "singleton_ball_subset")
        & (paired["K"] == "all")
    ]

    def sig(row_df, need_positive=False):
        if row_df.empty:
            return False, float("nan")
        r = row_df.iloc[0]
        d = float(r["mean_delta_macro_f1"])
        p = float(r["wilcoxon_p"])
        ok = np.isfinite(p) and p < 0.05 and np.isfinite(d)
        if need_positive:
            ok = ok and d > 0
        return ok, d

    not_comparable = (
        same_frac < 0.05
        or sem_match_frac < 0.01
        or repr_diff
    )

    sig_g_raw, d_g_raw = sig(p_g_vs_raw, need_positive=True)
    sig_g_sing, d_g_sing = sig(p_g_vs_sing, need_positive=True)

    if not_comparable:
        case = "C"
        text = (
            "**CASE C — PREVIOUS RESULTS WERE NOT COMPARABLE**\n\n"
            "The K-sensitivity `sample_stratified_k` and validation `sample_stratified_matched` "
            "controls differ in sample selection RNG, post-selection representation "
            "(rho90-rebuild vs singleton balls), and classifier weighting (full y_train vs y_sub). "
            f"Identical sample sets: {same_frac*100:.1f}% of runs; semantically matched: {sem_match_frac*100:.1f}%."
        )
    elif sig_g_raw or sig_g_sing:
        case = "A"
        text = (
            "**CASE A — GRANULAR ADVANTAGE SUPPORTED (under matched-index semantic controls)**\n\n"
            "With identical selected samples, granular entropy-k compression improves over "
            "singleton-ball proxies at matched sample counts."
        )
    elif abs(d_g_raw) < 0.01 and abs(d_g_sing) < 0.01:
        case = "B"
        text = (
            "**CASE B — GRANULAR VS SAMPLE COMPRESSION INDISTINGUISHABLE**\n\n"
            "Properly aligned comparisons show similar performance; gains are primarily compression/regularization."
        )
    elif d_g_raw < 0 or d_g_sing < 0:
        case = "D"
        text = (
            "**CASE D — SAMPLE COMPRESSION WINS**\n\n"
            "No empirical advantage for granular-ball compression over sample-level proxies."
        )
    else:
        case = "B"
        text = "**CASE B** — Indistinguishable under reconciliation protocol."

    return case, text


def write_plots(df: pd.DataFrame, paired: pd.DataFrame, discrepancy: pd.DataFrame) -> None:
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for m in METHODS:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        ax.bar(m, _nanmean(sub["macro_f1"]), yerr=_nanstd(sub["macro_f1"]), capsize=3, alpha=0.8)
    ax.set_ylabel("Mean macro F1")
    ax.set_title("Macro F1 by control (all K pooled)")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "macro_f1_by_control.png", dpi=150)
    plt.close(fig)

    p = paired[
        (paired["K"] == "all")
        & (paired["focal"] == "sample_stratified_k_original")
        & (paired["baseline"] == "sample_stratified_matched_original")
    ]
    if not p.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(["delta F1"], [p.iloc[0]["mean_delta_macro_f1"]])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("K-original minus matched-original")
        fig.tight_layout()
        fig.savefig(plot_dir / "paired_delta_k_vs_matched.png", dpi=150)
        plt.close(fig)

    if not discrepancy.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(discrepancy["sample_count_A"], discrepancy["sample_count_B"], alpha=0.4, s=15)
        lim = max(discrepancy["sample_count_A"].max(), discrepancy["sample_count_B"].max())
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5)
        ax.set_xlabel("K-sensitivity sample count")
        ax.set_ylabel("Validation-matched sample count")
        ax.set_title("Retained sample counts")
        fig.tight_layout()
        fig.savefig(plot_dir / "sample_count_comparison.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(discrepancy["jaccard_similarity"].dropna(), bins=30)
        ax.set_xlabel("Jaccard similarity of selected indices")
        ax.set_title("Selection overlap (k_original vs matched_original)")
        fig.tight_layout()
        fig.savefig(plot_dir / "jaccard_similarity.png", dpi=150)
        plt.close(fig)

    for focal, baseline, fname in (
        ("granular_compressed_entropy_k", "singleton_ball_subset", "granular_vs_singleton.png"),
        ("granular_compressed_entropy_k", "raw_sample_subset", "granular_vs_raw_proxy.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in K_VALUES:
            fk = df[(df["method"] == focal) & (df["K"] == k)]["macro_f1"]
            bk = df[(df["method"] == baseline) & (df["K"] == k)]["macro_f1"]
            if len(fk) and len(bk):
                ax.scatter([k] * len(fk), fk, alpha=0.3, s=10, label=focal if k == K_VALUES[0] else "")
                ax.scatter([k + 0.1] * len(bk), bk, alpha=0.3, s=10, label=baseline if k == K_VALUES[0] else "")
        ax.set_xlabel("K")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"{focal} vs {baseline}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150)
        plt.close(fig)


def print_final_report(df: pd.DataFrame, paired: pd.DataFrame, discrepancy: pd.DataFrame, case: str, text: str) -> str:
    k_orig = df[df["method"] == "sample_stratified_k_original"]
    matched = df[df["method"] == "sample_stratified_matched_original"]
    raw_p = df[df["method"] == "raw_sample_subset"]
    sing = df[df["method"] == "singleton_ball_subset"]
    gran = df[df["method"] == "granular_compressed_entropy_k"]

    same_frac = float(discrepancy["same_selected_samples"].mean()) if not discrepancy.empty else float("nan")
    sem_frac = float(discrepancy["semantically_matched"].mean()) if not discrepancy.empty else float("nan")

    def _p(name_f, name_b):
        sub = paired[(paired["focal"] == name_f) & (paired["baseline"] == name_b) & (paired["K"] == "all")]
        if sub.empty:
            return float("nan"), float("nan"), float("nan")
        r = sub.iloc[0]
        return float(r["mean_delta_macro_f1"]), float(r["wilcoxon_p"]), float(r["pct_wins"])

    d_gr, p_gr, w_gr = _p("granular_compressed_entropy_k", "raw_sample_subset")
    d_gs, p_gs, w_gs = _p("granular_compressed_entropy_k", "singleton_ball_subset")

    report = f"""
============================================================
SAMPLE CONTROL RECONCILIATION
============================================================

Previous K-sensitivity sample control (sample_stratified_k_original):
    Mean Macro F1: {_nanmean(k_orig['macro_f1']):.4f}
    Mean retained samples: {_nanmean(k_orig['actual_samples']):.1f}
    Collapse rate: {k_orig['collapsed'].mean():.3f}

Previous validation matched control (sample_stratified_matched_original):
    Mean Macro F1: {_nanmean(matched['macro_f1']):.4f}
    Mean retained samples: {_nanmean(matched['actual_samples']):.1f}
    Collapse rate: {matched['collapsed'].mean():.3f}

Are they semantically equivalent?
    {'NO' if same_frac < 0.99 or sem_frac < 0.99 else 'YES'} (identical indices: {same_frac*100:.1f}%, full semantic match: {sem_frac*100:.1f}%)

Primary discrepancy:
    Different RNG seeds for sample selection; different post-selection representation
    (rho90-rebuild vs singleton balls); different y_train for class weights (full vs subset).

Raw sample subset (proxy — full-y singleton balls):
    Mean Macro F1: {_nanmean(raw_p['macro_f1']):.4f}
    Collapse rate: {raw_p['collapsed'].mean():.3f}

Singleton-ball subset (same indices as k_original):
    Mean Macro F1: {_nanmean(sing['macro_f1']):.4f}
    Collapse rate: {sing['collapsed'].mean():.3f}

Granular compressed (entropy_k):
    Mean Macro F1: {_nanmean(gran['macro_f1']):.4f}
    Collapse rate: {gran['collapsed'].mean():.3f}

Granular - raw proxy:
    Mean Delta: {d_gr:+.4f}
    Wilcoxon p: {p_gr:.4g}
    Win rate: {w_gr:.1f}%

Granular - singleton (same indices):
    Mean Delta: {d_gs:+.4f}
    Wilcoxon p: {p_gs:.4g}
    Win rate: {w_gs:.1f}%

FINAL SCIENTIFIC CONCLUSION:
    CASE {case}
{text}

RECOMMENDATION:
    {'Do not claim granular-specific advantage from validation experiment alone; reconcile controls before new variants.' if case == 'C' else 'Proceed based on reconciled evidence only.'}
============================================================
"""
    print(report)
    return report


def analyze_and_write(df: pd.DataFrame, failures: list[dict]) -> None:
    control_defs = build_control_definitions()
    discrepancy = build_discrepancy_analysis(df)
    paired = build_paired_comparisons(df)
    dataset_level = build_dataset_level(df)
    case, conclusion_text = determine_conclusion(paired, discrepancy)

    control_defs.to_csv(OUT_DIR / "control_definitions.csv", index=False)
    discrepancy.to_csv(OUT_DIR / "discrepancy_analysis.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_comparisons.csv", index=False)
    dataset_level.to_csv(OUT_DIR / "dataset_level_results.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT_DIR / "failure_analysis.csv", index=False)

    per_run = discrepancy.merge(
        df[df["method"].isin(METHODS)],
        on=["dataset", "seed", "K"],
        how="outer",
        suffixes=("_disc", ""),
    )
    per_run.to_csv(OUT_DIR / "per_run_comparison.csv", index=False)

    write_plots(df, paired, discrepancy)
    report = print_final_report(df, paired, discrepancy, case, conclusion_text)

    meta = {
        "control_semantics": CONTROL_SEMANTICS,
        "n_rows": len(df),
        "paired_all_k": paired[paired["K"] == "all"].to_dict(orient="records"),
        "discrepancy_summary": {
            "mean_jaccard": float(discrepancy["jaccard_similarity"].mean()) if not discrepancy.empty else float("nan"),
            "frac_identical_indices": float(discrepancy["same_selected_samples"].mean()) if not discrepancy.empty else float("nan"),
            "frac_semantically_matched": float(discrepancy["semantically_matched"].mean()) if not discrepancy.empty else float("nan"),
        },
        "case": case,
        "conclusion": conclusion_text,
        "final_report": report,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    readme = [
        "# Sample control reconciliation",
        "",
        "Diagnostic reconciling K-sensitivity vs validation sample-control discrepancy.",
        "",
        "## Why previous results disagreed",
        "",
        "| Aspect | sample_stratified_k (K-sensitivity) | sample_stratified_matched (validation) |",
        "|--------|--------------------------------------|----------------------------------------|",
        "| K meaning | # rho90 balls (entropy_k) | n_target only (from granular sample count) |",
        "| Sample selection RNG | `deterministic_rng_seed(..., method='sample_stratified_k')` | `sample_control_seed(..., method='sample_stratified_matched')` |",
        "| Post-selection | **Rebuild rho90** on subsample | **Singleton balls** (radius=1) |",
        "| Class weights y_train | **Full train** | **Subset only** |",
        "",
        "## raw_sample_subset limitation",
        "",
        "No non-ball raw-sample Gaussian GBLSTSVM path exists without modifying `src/`. ",
        "`raw_sample_subset` uses identical indices with singleton balls but **full-train y weights** as a proxy.",
        "",
        "## Conclusion",
        "",
        conclusion_text,
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
        df = pd.read_csv(OUT_DIR / "raw_results.csv")
        failures = pd.read_csv(OUT_DIR / "failure_analysis.csv").to_dict(orient="records")
        analyze_and_write(df, failures)
        return 0

    descriptors = discover_available_datasets()
    seeds = [0] if args.smoke else SEEDS
    if args.smoke:
        descriptors = descriptors[:1]

    all_rows: list[dict] = []
    all_fail: list[dict] = []
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                all_fail.append({"dataset": desc["name"], "seed": seed, "reason": err})
                continue
            rows, fails = run_dataset_seed(split, seed, k_values)
            all_rows.extend(rows)
            all_fail.extend(fails)
            done += 1
            if done % SAVE_EVERY == 0:
                pd.DataFrame(all_rows).to_csv(OUT_DIR / "raw_results.csv", index=False)
                print(f"Checkpoint {done}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "raw_results.csv", index=False)
    analyze_and_write(df, all_fail)
    print(f"Done -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
