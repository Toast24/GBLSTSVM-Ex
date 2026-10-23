"""
Principled K-selection decision experiment (read-only).

Final controlled comparison of compression, validation-selected K, fixed K,
and simple pre-registered rules across 17 datasets × 10 seeds.

Does NOT modify src/ or prior experiments.

Run from GBLSTSVM root:
  python experiments/test_principled_complexity_rule.py --smoke
  python experiments/test_principled_complexity_rule.py
  python experiments/test_principled_complexity_rule.py --from-csv
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2  # noqa: E402
from experiments.principled_complexity_rule_common import (  # noqa: E402
    ALL_METHODS,
    K_CANDIDATES,
    METHOD_FIXED_K10,
    METHOD_FULL_RHO90,
    METHOD_PREREG_K10,
    METHOD_VAL_1SE,
    METHOD_VAL_MAX,
    PRIMARY_COMPARISONS,
    RUN_COLUMNS,
    SEEDS,
    build_dataset_summary,
    build_k_selection_frequency,
    build_paired_comparisons,
    build_summary_by_method,
    build_validation_test_agreement,
    run_dataset_seed,
    spearman_k_vs_f1,
)
from experiments.validation_selected_k_common import COMPRESSION_REPS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "principled_complexity_rule"
SAVE_EVERY = 10
SIGNIFICANCE_ALPHA = 0.05


def analyze_and_write(
    all_runs: pd.DataFrame,
    oracle_df: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary_by_method(all_runs)
    paired = build_paired_comparisons(all_runs)
    k_freq = build_k_selection_frequency(all_runs)
    dataset_summary, dataset_wtl = build_dataset_summary(all_runs)
    agreement = build_validation_test_agreement(all_runs, oracle_df)
    spearman = spearman_k_vs_f1(all_runs)

    all_runs.to_csv(OUT_DIR / "all_runs.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_by_method.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_comparisons.csv", index=False)
    k_freq.to_csv(OUT_DIR / "k_selection_frequency.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    if not dataset_wtl.empty:
        dataset_wtl.to_csv(OUT_DIR / "dataset_wtl.csv", index=False)
    agreement.to_csv(OUT_DIR / "validation_test_agreement.csv", index=False)
    oracle_df.to_csv(OUT_DIR / "oracle_k_diagnostic.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT_DIR / "failure_analysis.csv", index=False)

    _write_plots(all_runs, k_freq, agreement, paired)
    interpretation = _interpret(all_runs, paired, agreement, k_freq, spearman)
    _write_readme(summary, paired, k_freq, agreement, interpretation, spearman)

    meta = {
        "n_dataset_seed_pairs": int(all_runs[["dataset", "seed"]].drop_duplicates().shape[0]),
        "n_run_rows": int(len(all_runs)),
        "methods": list(ALL_METHODS),
        "K_candidates": K_CANDIDATES,
        "spearman": spearman,
        "interpretation": interpretation,
        "paired_comparisons": paired.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _interpret(
    all_runs: pd.DataFrame,
    paired: pd.DataFrame,
    agreement: pd.DataFrame,
    k_freq: pd.DataFrame,
    spearman: dict[str, float],
) -> str:
    def _row(focal: str, baseline: str) -> dict[str, Any]:
        sub = paired[(paired["focal"] == focal) & (paired["baseline"] == baseline)]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    r_full = _row(METHOD_VAL_MAX, METHOD_FULL_RHO90)
    r_k6 = _row(METHOD_VAL_MAX, "FIXED_K6")
    r_k10 = _row(METHOD_VAL_MAX, "FIXED_K10")
    r_size = _row(METHOD_VAL_MAX, "SIZE_BASED_RULE")
    r_frac = _row(METHOD_VAL_MAX, "FRACTION_OF_RHO90_RULE")

    d_full = float(r_full.get("mean_delta_macro_f1", float("nan")))
    p_full = float(r_full.get("wilcoxon_p", float("nan")))
    sig_full = np.isfinite(p_full) and p_full < SIGNIFICANCE_ALPHA and d_full > 0

    max_agree_1se = float(agreement["max_agrees_1se"].mean()) if not agreement.empty else float("nan")
    max_agree_oracle = float(agreement["max_agrees_oracle"].mean()) if not agreement.empty else float("nan")

    fixed10 = all_runs[all_runs["method"] == METHOD_FIXED_K10]
    prereg = all_runs[all_runs["method"] == METHOD_PREREG_K10]
    prereg_match = bool(
        not fixed10.empty
        and not prereg.empty
        and np.allclose(fixed10["macro_f1"].values, prereg["macro_f1"].values, equal_nan=True)
    )

    lines = [
        "## Scientific verdict",
        "",
        f"- **Compression vs full rho90**: mean delta={d_full:+.4f}, Wilcoxon p={p_full:.4g}, "
        f"{'significant improvement' if sig_full else 'no significant improvement'}.",
        f"- **Validation MAX vs FIXED_K6**: mean delta={float(r_k6.get('mean_delta_macro_f1', float('nan'))):+.4f}, "
        f"W/T/L={r_k6.get('wins', '—')}/{r_k6.get('ties', '—')}/{r_k6.get('losses', '—')}.",
        f"- **Validation MAX vs FIXED_K10**: mean delta={float(r_k10.get('mean_delta_macro_f1', float('nan'))):+.4f}.",
        f"- **MAX vs 1SE agreement**: {max_agree_1se:.1%} of runs select the same K.",
        f"- **Simple rules vs MAX**: SIZE delta={float(r_size.get('mean_delta_macro_f1', float('nan'))):+.4f}; "
        f"FRACTION delta={float(r_frac.get('mean_delta_macro_f1', float('nan'))):+.4f}.",
        f"- **Validation vs oracle-K agreement**: {max_agree_oracle:.1%}.",
        f"- **PRE_REGISTERED_K10 == FIXED_K10**: {prereg_match}.",
        f"- **Spearman(K, macro F1)**: {spearman.get('K_selected_vs_macro_f1_rho', float('nan')):.3f}; "
        f"**Spearman(n_balls, macro F1)**: {spearman.get('n_balls_final_vs_macro_f1_rho', float('nan')):.3f}.",
    ]
    return "\n".join(lines)


def _write_plots(
    all_runs: pd.DataFrame,
    k_freq: pd.DataFrame,
    agreement: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary_by_method(all_runs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["method"], summary["mean_macro_f1"])
    ax.set_ylabel("Mean test macro F1")
    ax.set_title("Principled K-selection: method comparison")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "method_comparison.png", dpi=150)
    plt.close(fig)

    if not k_freq.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in k_freq["method"].unique():
            sub = k_freq[k_freq["method"] == method]
            ax.bar(sub["K"].astype(str) + f"\n{method[:8]}", sub["fraction"], alpha=0.7, label=method)
        ax.set_ylabel("Selection frequency")
        ax.set_title("Validation-selected K distribution")
        fig.tight_layout()
        fig.savefig(plot_dir / "k_selection_frequency.png", dpi=150)
        plt.close(fig)

    sub = all_runs[all_runs["method"] == METHOD_VAL_MAX]
    if not sub.empty and "validation_best_f1" in sub.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub["validation_best_f1"], sub["macro_f1"], alpha=0.5, s=20)
        ax.set_xlabel("Validation macro F1 (selected K)")
        ax.set_ylabel("Test macro F1")
        ax.set_title("Validation vs test (VALIDATION_SELECTED_MAX)")
        fig.tight_layout()
        fig.savefig(plot_dir / "validation_vs_test.png", dpi=150)
        plt.close(fig)


def _write_readme(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    k_freq: pd.DataFrame,
    agreement: pd.DataFrame,
    interpretation: str,
    spearman: dict[str, float],
) -> None:
    lines = [
        "# Principled K-selection decision experiment",
        "",
        "Pre-registered comparison of granular-ball compression complexity rules.",
        "",
        "## Protocol",
        "",
        "- 17 datasets × seeds 0–9 (170 pairs).",
        "- Benchmark split: 70% train / 15% val / 15% test (test never used for K selection).",
        "- rho=0.90 recursive KMeans(k=2) granulation; random ball compression.",
        f"- Candidate K: {K_CANDIDATES}.",
        "- Validation selection: 80/20 split of benchmark train; "
        f"{COMPRESSION_REPS} compression reps per K.",
        "",
        "## Methods",
        "",
        *[f"- `{m}`" for m in ALL_METHODS],
        "",
        "## Summary by method",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Primary paired comparisons",
        "",
        "```csv",
        paired.to_csv(index=False).strip(),
        "```",
        "",
        "## K selection frequency",
        "",
        "```csv",
        k_freq.to_csv(index=False).strip() if not k_freq.empty else "(no data)",
        "```",
        "",
        interpretation,
        "",
        f"Classifier: Gaussian GBLSTSVM ({PRIMARY_GAUSSIAN}); C1={DEFAULT_C1}; C2={DEFAULT_C2}",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Principled K-selection decision experiment")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--from-csv", action="store_true")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global OUT_DIR
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        all_runs = pd.read_csv(OUT_DIR / "all_runs.csv")
        oracle_df = pd.read_csv(OUT_DIR / "oracle_k_diagnostic.csv")
        failures = (
            pd.read_csv(OUT_DIR / "failure_analysis.csv").to_dict(orient="records")
            if (OUT_DIR / "failure_analysis.csv").exists()
            else []
        )
        meta = analyze_and_write(all_runs, oracle_df, failures)
        print(f"Wrote analysis to {OUT_DIR}")
        print(meta["interpretation"])
        return 0

    descriptors = discover_available_datasets()
    if args.datasets:
        names = set(args.datasets)
        descriptors = [d for d in descriptors if d["name"] in names]
    seeds = args.seeds if args.seeds else SEEDS
    if args.smoke:
        descriptors = descriptors[:2]
        seeds = [0]

    all_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failures.append({"dataset": desc["name"], "seed": seed, "reason": err})
                continue
            try:
                runs, oracle, fail = run_dataset_seed(split, seed)
                all_rows.extend(runs)
                oracle_rows.extend(oracle)
                failures.extend(fail)
                done += 1
                full = next(r for r in runs if r["method"] == METHOD_FULL_RHO90)
                vmax = next(r for r in runs if r["method"] == METHOD_VAL_MAX)
                print(
                    f"{split.name} s{seed}: FULL={full['macro_f1']:.3f} "
                    f"MAX_K={vmax.get('K_selected', float('nan'))} "
                    f"MAX_F1={vmax['macro_f1']:.3f}"
                )
            except Exception as exc:
                failures.append(
                    {"dataset": desc["name"], "seed": seed, "reason": f"exception:{exc}"}
                )
            if done % SAVE_EVERY == 0 and done > 0:
                pd.DataFrame(all_rows).to_csv(OUT_DIR / "all_runs.csv", index=False)
                print(f"Checkpoint {done} dataset×seed pairs")

    all_runs = pd.DataFrame(all_rows)
    oracle_df = pd.DataFrame(oracle_rows)
    for col in RUN_COLUMNS:
        if col not in all_runs.columns:
            all_runs[col] = float("nan")

    meta = analyze_and_write(all_runs, oracle_df, failures)
    print(f"\nDone: {done} dataset×seed pairs -> {OUT_DIR}")
    print(meta["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
