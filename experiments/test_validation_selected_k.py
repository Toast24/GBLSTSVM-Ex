"""
Validation-based K-selection experiment (read-only).

Determines whether compression benefit from K-sensitivity can become a deployable
model-selection procedure using internal validation only.

Does NOT modify src/, rho90, solver, or prediction rule.

Run from GBLSTSVM root:
  python experiments/test_validation_selected_k.py --smoke
  python experiments/test_validation_selected_k.py
  python experiments/test_validation_selected_k.py --from-csv
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

from experiments.analyze_ball_utility_landscape import compute_ball_features  # noqa: E402
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.granular_compression_k_common import build_rho90_from_train_indices  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels  # noqa: E402
from experiments.test_adaptive_entropy_granulation_v2 import _global_min_label  # noqa: E402
from experiments.test_split_utility_granulation import _build_indexed_matrix  # noqa: E402
from experiments.validation_selected_k_common import (  # noqa: E402
    COMPRESSION_REPS,
    DELTA_EPS,
    FIXED_K_VALUES,
    K_CANDIDATES_DEFAULT,
    SEEDS_DEFAULT,
    KValidationSummary,
    Phase,
    assert_partition_disjoint_from_test,
    build_rho90_pool,
    compress_and_evaluate,
    compression_seed,
    eval_full_rho90,
    eval_full_sample,
    eval_sample_stratified_matched,
    make_train_val_partition,
    select_k_1se,
    select_k_max,
    summarize_k_validation,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "validation_selected_k"
RAW_CSV = OUT_DIR / "raw_results.csv"
VALIDATION_CSV = OUT_DIR / "validation_results.csv"
TEST_CSV = OUT_DIR / "test_results.csv"
SELECTED_K_CSV = OUT_DIR / "selected_k_by_run.csv"
STABILITY_CSV = OUT_DIR / "selection_stability.csv"
PAIRED_CSV = OUT_DIR / "paired_comparisons.csv"
DATASET_CSV = OUT_DIR / "dataset_level_results.csv"
FAILURE_CSV = OUT_DIR / "failure_analysis.csv"
README = OUT_DIR / "README.md"
SAVE_EVERY = 10

PRIMARY_METHODS = (
    "full_rho90",
    "fixed_K6",
    "fixed_K8",
    "fixed_K10",
    "validation_selected_max",
    "validation_selected_1se",
    "full_sample",
    "sample_stratified_matched",
)
OPTIONAL_SAMPLE_METHODS = ("sample_matched_K6", "sample_matched_K8", "sample_matched_K10")
SIGNIFICANCE_ALPHA = 0.05


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _nanmedian(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _base_meta(dataset: str, seed: int) -> dict[str, Any]:
    return {"dataset": dataset, "seed": seed}


def _row(
    dataset: str,
    seed: int,
    method: str,
    phase: str,
    *,
    repetition: int = 0,
    K: int | float | None = None,
    **extra,
) -> dict[str, Any]:
    return {
        **_base_meta(dataset, seed),
        "method": method,
        "phase": phase,
        "repetition": repetition,
        "K": float(K) if K is not None else float("nan"),
        **extra,
    }


def run_dataset_seed(
    split,
    seed: int,
    k_candidates: list[int],
    n_reps: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """
    Returns: validation_rows, test_rows, selected_k_rows, raw_rows, failures
    """
    dataset = split.name
    y_train = normalize_labels(split.y_train)
    X_train = split.X_train
    X_test = split.X_test
    y_test = split.y_test
    minority_label = split.minority_label

    # --- Anti-leakage: partition uses benchmark train only ---
    partition = make_train_val_partition(dataset, seed, X_train, y_train)
    assert_partition_disjoint_from_test(partition, len(X_train), len(X_test), context="init")

    sel_idx = partition.selection_indices
    val_idx = partition.validation_indices
    X_sel = X_train[sel_idx]
    y_sel = y_train[sel_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    indexed_full = _build_indexed_matrix(X_train, y_train)
    global_min = _global_min_label(indexed_full)

    validation_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    # Step B: rho90 on train_selection ONLY
    balls_sel, kept_sel, _, _ = build_rho90_pool(X_train, y_train, sel_idx, seed=seed)
    features_sel = compute_ball_features(kept_sel, minority_label)
    n_pool_sel = len(balls_sel)

    k_summaries: dict[int, KValidationSummary] = {}

    for k in k_candidates:
        rep_rows: list[dict[str, Any]] = []
        if n_pool_sel < k:
            skip = {
                **_row(dataset, seed, "random_compress", "selection", K=k),
                "skipped": 1,
                "skip_reason": f"rho90_pool={n_pool_sel}<K={k}",
            }
            validation_rows.append(skip)
            failures.append(
                {"dataset": dataset, "seed": seed, "K": k, "phase": "selection", "reason": skip["skip_reason"]}
            )
            k_summaries[k] = summarize_k_validation([skip])
            continue

        for rep in range(n_reps):
            rng_seed = compression_seed(dataset, seed, k, rep, phase="selection")
            ev = compress_and_evaluate(
                balls_sel,
                kept_sel,
                features_sel,
                k,
                minority_label=minority_label,
                rng_seed=rng_seed,
                y_train_fit=y_sel,
                X_eval=X_val,
                y_eval=y_val,
                seed=seed,
                indexed=indexed_full,
                global_min=global_min,
                phase="selection",
            )
            row = {
                **_row(dataset, seed, "random_compress", "selection", repetition=rep, K=k),
                "n_rho90_balls": n_pool_sel,
                "split_seed": partition.split_seed,
                "compression_seed": rng_seed,
                "skipped": 0,
                "skip_reason": "",
                **ev,
            }
            validation_rows.append(row)
            raw_rows.append(row)
            rep_rows.append(row)

        k_summaries[k] = summarize_k_validation(rep_rows)

    k_max, reason_max = select_k_max(k_summaries)
    k_1se, reason_1se, threshold_1se = select_k_1se(k_summaries)

    selected_k_row = {
        **_base_meta(dataset, seed),
        "K_selected_max": k_max if k_max is not None else float("nan"),
        "K_selected_1se": k_1se if k_1se is not None else float("nan"),
        "selection_reason_max": reason_max,
        "selection_reason_1se": reason_1se,
        "threshold_1se": threshold_1se,
        "n_rho90_balls_selection": n_pool_sel,
        "split_seed": partition.split_seed,
        **{
            f"val_mean_f1_K{k}": s.mean_macro_f1
            for k, s in k_summaries.items()
        },
    }
    if k_max is not None:
        selected_k_row["val_mean_f1_selected_max"] = k_summaries[k_max].mean_macro_f1
    if k_1se is not None:
        selected_k_row["val_mean_f1_selected_1se"] = k_summaries[k_1se].mean_macro_f1

    # --- Final evaluation on FULL benchmark train + untouched test ---
    # Rebuild rho90 on full training portion (benchmark train 70%)
    all_idx = np.arange(len(X_train), dtype=int)
    balls_full, kept_full = build_rho90_from_train_indices(indexed_full, all_idx, seed=seed)
    features_full = compute_ball_features(kept_full, minority_label)
    n_pool_full = len(balls_full)

    test_rows: list[dict[str, Any]] = []

    def _test_eval(method: str, ev: dict[str, Any], **meta) -> None:
        row = {
            **_row(dataset, seed, method, "final_test", **meta),
            "n_rho90_balls": n_pool_full,
            **ev,
        }
        test_rows.append(row)
        raw_rows.append(row)

    # full_rho90 baseline
    ev_full = eval_full_rho90(
        split, seed=seed, phase="final_test", X_eval=X_test, y_eval=y_test
    )
    _test_eval("full_rho90", ev_full, K=float("nan"), repetition=0)

    # full_sample baseline
    ev_sample = eval_full_sample(
        split, seed=seed, phase="final_test", X_eval=X_test, y_eval=y_test
    )
    _test_eval("full_sample", ev_sample, K=float("nan"), repetition=0)

    def _eval_granular_test(method: str, k: int | None, rep: int = 0) -> dict[str, Any]:
        if k is None or n_pool_full < k:
            return {
                "skipped": 1,
                "skip_reason": f"invalid_K={k}_pool={n_pool_full}",
                "macro_f1": float("nan"),
                "collapsed": 1,
            }
        rng_seed = compression_seed(dataset, seed, k, rep, phase="final_test")
        return compress_and_evaluate(
            balls_full,
            kept_full,
            features_full,
            k,
            minority_label=minority_label,
            rng_seed=rng_seed,
            y_train_fit=y_train,
            X_eval=X_test,
            y_eval=y_test,
            seed=seed,
            indexed=indexed_full,
            global_min=global_min,
            phase="final_test",
        )

    for fk, method in [(6, "fixed_K6"), (8, "fixed_K8"), (10, "fixed_K10")]:
        ev = _eval_granular_test(method, fk, rep=0)
        _test_eval(method, ev, K=fk, repetition=0)

    for sel_k, method in [(k_max, "validation_selected_max"), (k_1se, "validation_selected_1se")]:
        ev = _eval_granular_test(method, sel_k, rep=0)
        val_f1 = (
            k_summaries[sel_k].mean_macro_f1
            if sel_k is not None and sel_k in k_summaries
            else float("nan")
        )
        _test_eval(
            method,
            ev,
            K=sel_k if sel_k is not None else float("nan"),
            repetition=0,
            validation_macro_f1_at_selection=val_f1,
        )

    # Matched sample compression for selected max K
    n_target_max = float("nan")
    if k_max is not None and k_max in k_summaries:
        # use mean n_samples from full-train compression at selected K
        ev_probe = _eval_granular_test("probe", k_max, rep=0)
        n_target_max = int(ev_probe.get("n_samples_represented", 0))
        if n_target_max > 0:
            sample_reps: list[dict[str, Any]] = []
            for rep in range(n_reps):
                ev_m = eval_sample_stratified_matched(
                    split,
                    n_target_max,
                    seed=seed,
                    repetition=rep,
                    phase="final_test",
                    X_eval=X_test,
                    y_eval=y_test,
                    dataset=dataset,
                    benchmark_seed=seed,
                )
                sample_reps.append(ev_m)
                raw_rows.append(
                    {
                        **_row(
                            dataset,
                            seed,
                            "sample_stratified_matched",
                            "final_test",
                            repetition=rep,
                            K=k_max,
                        ),
                        "n_target_samples": n_target_max,
                        **ev_m,
                    }
                )
            agg = {
                k: _nanmean([r[k] for r in sample_reps if k in r])
                for k in (
                    "macro_f1",
                    "balanced_accuracy",
                    "accuracy",
                    "minority_f1",
                    "gmean",
                    "collapsed",
                    "n_balls",
                    "n_samples_represented",
                )
            }
            agg["collapse_reason"] = sample_reps[0].get("collapse_reason", "")
            agg["classifier_ok"] = all(r.get("classifier_ok") for r in sample_reps)
            _test_eval(
                "sample_stratified_matched",
                agg,
                K=k_max,
                repetition=-1,
                n_target_samples=n_target_max,
                n_reps_aggregated=n_reps,
            )

    # Optional fixed-K sample matched controls
    for fk, method in [(6, "sample_matched_K6"), (8, "sample_matched_K8"), (10, "sample_matched_K10")]:
        ev_g = _eval_granular_test(method, fk, rep=0)
        n_tgt = int(ev_g.get("n_samples_represented", 0))
        if n_tgt <= 0 or ev_g.get("skipped"):
            failures.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "reason": ev_g.get("skip_reason", "no_samples"),
                }
            )
            continue
        rep_metrics = []
        for rep in range(n_reps):
            ev_m = eval_sample_stratified_matched(
                split,
                n_tgt,
                seed=seed,
                repetition=rep,
                phase="final_test",
                X_eval=X_test,
                y_eval=y_test,
                dataset=dataset,
                benchmark_seed=seed,
            )
            rep_metrics.append(ev_m)
            raw_rows.append(
                {
                    **_row(dataset, seed, method, "final_test", repetition=rep, K=fk),
                    "n_target_samples": n_tgt,
                    **ev_m,
                }
            )
        agg = {
            k: _nanmean([r[k] for r in rep_metrics if k in r])
            for k in ("macro_f1", "balanced_accuracy", "minority_f1", "gmean", "collapsed")
        }
        _test_eval(method, agg, K=fk, repetition=-1, n_target_samples=n_tgt)

    return validation_rows, test_rows, [selected_k_row], raw_rows, failures


def _paired_delta(
    test_df: pd.DataFrame,
    focal: str,
    baseline: str,
) -> dict[str, Any]:
    f = test_df[test_df["method"] == focal].set_index(["dataset", "seed"])
    b = test_df[test_df["method"] == baseline].set_index(["dataset", "seed"])
    keys = sorted(set(f.index) & set(b.index))
    deltas = []
    for key in keys:
        fv = float(f.loc[key, "macro_f1"])
        bv = float(b.loc[key, "macro_f1"])
        if np.isfinite(fv) and np.isfinite(bv):
            deltas.append(fv - bv)
    if not deltas:
        return {
            "focal": focal,
            "baseline": baseline,
            "n_pairs": 0,
            "mean_delta_macro_f1": float("nan"),
            "median_delta_macro_f1": float("nan"),
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "pct_wins": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
        }
    arr = np.asarray(deltas, dtype=float)
    wins = int(np.sum(arr > DELTA_EPS))
    ties = int(np.sum(np.abs(arr) <= DELTA_EPS))
    losses = int(np.sum(arr < -DELTA_EPS))
    try:
        if np.allclose(arr, 0):
            wstat, wp = 0.0, 1.0
        else:
            wstat, wp = wilcoxon(arr)
    except ValueError:
        wstat, wp = float("nan"), float("nan")
    return {
        "focal": focal,
        "baseline": baseline,
        "n_pairs": len(deltas),
        "mean_delta_macro_f1": float(np.mean(arr)),
        "median_delta_macro_f1": float(np.median(arr)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "pct_wins": 100.0 * wins / len(deltas),
        "wilcoxon_stat": float(wstat) if np.isfinite(wstat) else float("nan"),
        "wilcoxon_p": float(wp) if np.isfinite(wp) else float("nan"),
    }


def build_paired_comparisons(test_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("validation_selected_max", "full_rho90"),
        ("validation_selected_1se", "full_rho90"),
        ("fixed_K6", "full_rho90"),
        ("fixed_K8", "full_rho90"),
        ("fixed_K10", "full_rho90"),
        ("validation_selected_max", "sample_stratified_matched"),
        ("validation_selected_1se", "sample_stratified_matched"),
    ]
    return pd.DataFrame([_paired_delta(test_df, f, b) for f, b in pairs])


def build_dataset_level(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in test_df.groupby(["dataset", "method"]):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_seeds": len(g),
                "mean_macro_f1": _nanmean(g["macro_f1"].tolist()),
                "median_macro_f1": _nanmedian(g["macro_f1"].tolist()),
                "mean_balanced_accuracy": _nanmean(g["balanced_accuracy"].tolist()),
                "frac_collapsed": float(g["collapsed"].mean()),
                "mean_delta_vs_full_rho90": float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    full = out[out["method"] == "full_rho90"].set_index("dataset")["mean_macro_f1"]
    for i, row in out.iterrows():
        if row["method"] == "full_rho90":
            continue
        ds = row["dataset"]
        if ds in full.index and np.isfinite(full.loc[ds]):
            out.at[i, "mean_delta_vs_full_rho90"] = row["mean_macro_f1"] - float(full.loc[ds])
    return out


def build_selection_stability(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule_col in ("K_selected_max", "K_selected_1se"):
        vals = selected_df[rule_col].dropna()
        if vals.empty:
            continue
        freq = vals.value_counts().sort_index()
        for k, count in freq.items():
            rows.append(
                {
                    "rule": rule_col,
                    "K": float(k),
                    "count": int(count),
                    "fraction": float(count / len(selected_df)),
                }
            )
        rows.append(
            {
                "rule": rule_col,
                "K": float("nan"),
                "count": int(len(selected_df)),
                "fraction": float("nan"),
                "median_K": float(np.median(vals)),
                "mean_K": float(np.mean(vals)),
                "iqr_K": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            }
        )
    # per-dataset mode consistency
    for dataset, g in selected_df.groupby("dataset"):
        for rule_col, label in (
            ("K_selected_max", "max"),
            ("K_selected_1se", "1se"),
        ):
            ks = g[rule_col].dropna()
            if ks.empty:
                continue
            mode_k = float(ks.mode().iloc[0])
            frac = float(np.mean(ks == mode_k))
            rows.append(
                {
                    "rule": f"dataset_mode_{label}",
                    "dataset": dataset,
                    "K": mode_k,
                    "count": int((ks == mode_k).sum()),
                    "fraction": frac,
                }
            )
    return pd.DataFrame(rows)


def interpret_results(paired: pd.DataFrame, stability: pd.DataFrame, selected: pd.DataFrame) -> str:
    def _sig(row_name: str, baseline: str) -> tuple[float, float, bool]:
        sub = paired[(paired["focal"] == row_name) & (paired["baseline"] == baseline)]
        if sub.empty:
            return float("nan"), float("nan"), False
        r = sub.iloc[0]
        p = float(r["wilcoxon_p"])
        d = float(r["mean_delta_macro_f1"])
        sig = np.isfinite(p) and p < SIGNIFICANCE_ALPHA and np.isfinite(d)
        return d, p, sig

    d_max_rho, _, sig_max_rho = _sig("validation_selected_max", "full_rho90")
    d_max_samp, _, sig_max_samp = _sig("validation_selected_max", "sample_stratified_matched")
    d_1se_rho, _, _ = _sig("validation_selected_1se", "full_rho90")

    fixed_similar = all(
        abs(_sig(f"fixed_K{k}", "full_rho90")[0] - d_max_rho) < 0.02
        for k in (6, 8, 10)
        if np.isfinite(d_max_rho)
    )

    mode_fracs = stability[stability["rule"].str.startswith("dataset_mode_", na=False)]
    unstable = False
    if not mode_fracs.empty:
        # Unstable if typical dataset-level mode agreement is below 40%
        unstable = bool(mode_fracs["fraction"].median() < 0.4)

    sig_granular_over_sample = (
        np.isfinite(d_max_samp) and d_max_samp > 0 and sig_max_samp
    )

    if not np.isfinite(d_max_rho) or d_max_rho <= 0:
        case = "D"
        text = (
            "**Case D** — Validation-selected compression does not reliably improve over full rho90. "
            "The compression hypothesis is not sufficiently robust as the primary extension."
        )
    elif sig_max_rho and sig_granular_over_sample:
        case = "A"
        text = (
            "**Case A** — Validation-selected granular compression significantly improves over full rho90 "
            "and over matched sample compression. Evidence supports a granular-specific contribution."
        )
    elif sig_max_rho and not sig_granular_over_sample:
        case = "B"
        text = (
            "**Case B** — Compression/regularization appears useful vs full rho90, but matched sample "
            "compression is statistically comparable. No established granular-ball-specific advantage."
        )
    elif fixed_similar and sig_max_rho:
        case = "C"
        text = (
            "**Case C** — Fixed mid-K performs similarly to validation-selected K. "
            "Validation-based selection adds little; characterize as a broad mid-K compression regime."
        )
    elif unstable:
        case = "E"
        text = (
            "**Case E** — K selection is highly unstable across seeds/datasets. "
            "A universal deployable K-selection rule is not supported by the current evidence."
        )
    else:
        case = "B"
        text = (
            "**Case B** — Compression may help vs full rho90, but granular structure is not clearly "
            "superior to matched sample compression."
        )

    return (
        f"Automated interpretation: {case}\n\n{text}\n\n"
        f"validation_selected_max vs full_rho90: Δ={d_max_rho:+.4f} (sig={sig_max_rho})\n"
        f"validation_selected_max vs sample_matched: Δ={d_max_samp:+.4f} (sig={sig_max_samp})\n"
        f"validation_selected_1se vs full_rho90: Δ={d_1se_rho:+.4f}"
    )


def write_plots(test_df: pd.DataFrame, selected_df: pd.DataFrame, validation_df: pd.DataFrame) -> None:
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in (("K_selected_max", "max-F1"), ("K_selected_1se", "1-SE")):
        vals = selected_df[col].dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=range(3, 36, 2), alpha=0.6, label=label)
    ax.set_xlabel("Selected K")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of validation-selected K (170 runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "selected_k_distribution.png", dpi=150)
    plt.close(fig)

    sel = test_df[test_df["method"] == "validation_selected_max"]
    if not sel.empty and "validation_macro_f1_at_selection" in sel.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            sel["validation_macro_f1_at_selection"],
            sel["macro_f1"],
            alpha=0.5,
            s=20,
        )
        ax.set_xlabel("Validation macro F1 (at selected K)")
        ax.set_ylabel("Test macro F1")
        ax.set_title("Validation vs test F1 (selected max)")
        fig.tight_layout()
        fig.savefig(plot_dir / "validation_vs_test_f1.png", dpi=150)
        plt.close(fig)

    methods = ["full_rho90", "fixed_K6", "fixed_K8", "fixed_K10", "validation_selected_max", "validation_selected_1se"]
    means = []
    for m in methods:
        sub = test_df[test_df["method"] == m]
        means.append(_nanmean(sub["macro_f1"].tolist()) if not sub.empty else float("nan"))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(methods, means)
    ax.set_ylabel("Mean test macro F1")
    ax.set_title("Selected K vs fixed-K baselines")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "selected_k_vs_fixed_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ("validation_selected_max", "sample_stratified_matched", "full_rho90"):
        sub = test_df[test_df["method"] == m]
        if sub.empty:
            continue
        ax.scatter(sub["n_samples_represented"], sub["macro_f1"], label=m, alpha=0.6)
    ax.set_xlabel("Represented / retained samples")
    ax.set_ylabel("Test macro F1")
    ax.set_title("Granular vs sample compression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "granular_vs_sample_compression.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sub = test_df[test_df["method"].isin(PRIMARY_METHODS)]
    if not sub.empty:
        for m in PRIMARY_METHODS:
            g = sub[sub["method"] == m]
            if g.empty:
                continue
            ax.scatter(g["collapsed"], g["macro_f1"], label=m, alpha=0.5, s=18)
    ax.set_xlabel("Collapsed (0/1)")
    ax.set_ylabel("Test macro F1")
    ax.set_title("Collapse vs macro F1")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "collapse_vs_macro_f1.png", dpi=150)
    plt.close(fig)


def write_readme(
    paired: pd.DataFrame,
    stability: pd.DataFrame,
    interpretation: str,
    protocol: dict[str, Any],
) -> None:
    lines = [
        "# Validation-based K-selection",
        "",
        "Deployability experiment: can internal validation turn rho90 compression into a model-selection procedure?",
        "",
        "## Protocol",
        "",
        "```",
        "benchmark train (70%) → stratified 80/20 → train_selection | validation_internal",
        "train_selection → rho90 pool → random-K compression (5 reps) → val macro F1",
        "select K (max-F1 or 1-SE) on validation only",
        "full benchmark train → rebuild rho90 → compress at K → train → evaluate on untouched test (15%)",
        "```",
        "",
        "## Leakage prevention",
        "",
        "- Test data never enters granulation, compression, or K selection.",
        "- Benchmark val (15%) is unused; internal validation is carved from benchmark train.",
        "- `validation` phase rows use internal val features only; `final_test` uses benchmark test only.",
        "",
        "## Selection rules",
        "",
        "- **K_selected_max**: argmax mean validation macro F1 across 5 random compressions; "
        "tie-break: lower collapse rate, then smaller K.",
        "- **K_selected_1se**: threshold = best_mean − SE(best), SE = std(reps)/√n_successful; "
        "smallest K with mean ≥ threshold. Reps are dependent compressions of the same pool — SE is approximate.",
        "",
        "## Controls",
        "",
        "- `full_rho90`: uncompressed rho90 on full train.",
        "- `fixed_K6/8/10`: random ball compression at fixed K.",
        "- `full_sample`: one ball per training sample (no granulation).",
        "- `sample_stratified_matched`: stratified sample subset matched to granular sample count.",
        "",
        "## Paired comparisons (test macro F1)",
        "",
        paired.to_csv(index=False) if False else "",
    ]
    lines = [l for l in lines if l != ""]
    lines.append("```csv")
    lines.append(paired.to_csv(index=False).strip())
    lines.append("```")
    lines.append("")
    lines.append("## Selection stability")
    lines.append("")
    lines.append("```csv")
    lines.append(stability.head(30).to_csv(index=False).strip())
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(interpretation)
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for f in (
        "raw_results.csv",
        "validation_results.csv",
        "test_results.csv",
        "selected_k_by_run.csv",
        "selection_stability.csv",
        "paired_comparisons.csv",
        "dataset_level_results.csv",
        "failure_analysis.csv",
        "summary.json",
        "plots/*.png",
    ):
        lines.append(f"- {f}")
    lines.append("")
    lines.append(f"Classifier: Gaussian GBLSTSVM ({PRIMARY_GAUSSIAN}); C1={DEFAULT_C1}; C2={DEFAULT_C2}")
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_and_write(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    failures: list[dict],
) -> None:
    paired = build_paired_comparisons(test_df)
    dataset_level = build_dataset_level(test_df)
    stability = build_selection_stability(selected_df)
    interpretation = interpret_results(paired, stability, selected_df)

    paired.to_csv(PAIRED_CSV, index=False)
    dataset_level.to_csv(DATASET_CSV, index=False)
    stability.to_csv(STABILITY_CSV, index=False)
    pd.DataFrame(failures).to_csv(FAILURE_CSV, index=False)
    write_plots(test_df, selected_df, validation_df)

    meta = {
        "n_validation_rows": int(len(validation_df)),
        "n_test_rows": int(len(test_df)),
        "n_runs": int(len(selected_df)),
        "paired_comparisons": paired.to_dict(orient="records"),
        "interpretation": interpretation,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_readme(paired, stability, interpretation, {})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation-based K-selection experiment")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--from-csv", action="store_true")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--k-values", nargs="*", type=int, default=None)
    p.add_argument("--repetitions", type=int, default=COMPRESSION_REPS)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global OUT_DIR, RAW_CSV, VALIDATION_CSV, TEST_CSV, SELECTED_K_CSV
    global PAIRED_CSV, DATASET_CSV, STABILITY_CSV, FAILURE_CSV, README
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CSV = OUT_DIR / "raw_results.csv"
    VALIDATION_CSV = OUT_DIR / "validation_results.csv"
    TEST_CSV = OUT_DIR / "test_results.csv"
    SELECTED_K_CSV = OUT_DIR / "selected_k_by_run.csv"
    PAIRED_CSV = OUT_DIR / "paired_comparisons.csv"
    DATASET_CSV = OUT_DIR / "dataset_level_results.csv"
    STABILITY_CSV = OUT_DIR / "selection_stability.csv"
    FAILURE_CSV = OUT_DIR / "failure_analysis.csv"
    README = OUT_DIR / "README.md"

    k_candidates = args.k_values if args.k_values else K_CANDIDATES_DEFAULT
    n_reps = args.repetitions

    if args.from_csv:
        validation_df = pd.read_csv(VALIDATION_CSV)
        test_df = pd.read_csv(TEST_CSV)
        selected_df = pd.read_csv(SELECTED_K_CSV)
        raw_df = pd.read_csv(RAW_CSV)
        failures = pd.read_csv(FAILURE_CSV).to_dict(orient="records") if FAILURE_CSV.exists() else []
        analyze_and_write(validation_df, test_df, selected_df, raw_df, failures)
        print(f"Wrote analysis to {OUT_DIR}")
        return 0

    descriptors = discover_available_datasets()
    if args.datasets:
        names = set(args.datasets)
        descriptors = [d for d in descriptors if d["name"] in names]
    seeds = args.seeds if args.seeds else SEEDS_DEFAULT
    if args.smoke:
        descriptors = descriptors[:1]
        seeds = [0]

    all_val: list[dict] = []
    all_test: list[dict] = []
    all_sel: list[dict] = []
    all_raw: list[dict] = []
    all_fail: list[dict] = []
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                all_fail.append({"dataset": desc["name"], "seed": seed, "reason": err})
                continue
            try:
                v, t, s, r, f = run_dataset_seed(split, seed, k_candidates, n_reps)
                all_val.extend(v)
                all_test.extend(t)
                all_sel.extend(s)
                all_raw.extend(r)
                all_fail.extend(f)
            except Exception as exc:
                all_fail.append(
                    {"dataset": split.name, "seed": seed, "reason": f"exception:{exc}"}
                )
            done += 1
            if done % SAVE_EVERY == 0:
                pd.DataFrame(all_val).to_csv(VALIDATION_CSV, index=False)
                pd.DataFrame(all_test).to_csv(TEST_CSV, index=False)
                pd.DataFrame(all_sel).to_csv(SELECTED_K_CSV, index=False)
                pd.DataFrame(all_raw).to_csv(RAW_CSV, index=False)
                print(f"Checkpoint {done} runs")

    validation_df = pd.DataFrame(all_val)
    test_df = pd.DataFrame(all_test)
    selected_df = pd.DataFrame(all_sel)
    raw_df = pd.DataFrame(all_raw)

    validation_df.to_csv(VALIDATION_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    selected_df.to_csv(SELECTED_K_CSV, index=False)
    raw_df.to_csv(RAW_CSV, index=False)

    analyze_and_write(validation_df, test_df, selected_df, raw_df, all_fail)
    print(f"Done: {len(selected_df)} dataset x seed runs -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
