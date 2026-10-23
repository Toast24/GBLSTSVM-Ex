"""
A4 — Estimator-information ladder benchmark.

Run from GBLSTSVM root:
  python experiments/test_estimator_information_ladder.py --smoke
  python experiments/test_estimator_information_ladder.py
  python experiments/test_estimator_information_ladder.py --from-csv
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
from experiments.estimator_information_ladder_common import (  # noqa: E402
    PROTOCOL,
    SEEDS,
    evaluate_method,
    paired_comparison,
    run_moment_identity_audit,
    verify_scatter_splits,
)
from experiments.test_split_utility_granulation import _build_indexed_matrix  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "estimator_information_ladder"
PLOT_DIR = OUT_DIR / "plots"

PRIMARY_METHODS = [
    "RAW_LSTSVM",
    "L0_CENTER_UNIT",
    "L1_POPULATION",
    "L2_POPULATION_SCATTER",
    "L3_MOMENT_EXACT",
]
ALL_METHODS = PRIMARY_METHODS + ["MATCHED_K_SAMPLES"]

PAIRS = [
    ("L1_POPULATION", "L0_CENTER_UNIT"),
    ("L2_POPULATION_SCATTER", "L1_POPULATION"),
    ("L2_POPULATION_SCATTER", "L0_CENTER_UNIT"),
    ("L3_MOMENT_EXACT", "RAW_LSTSVM"),
    ("L1_POPULATION", "MATCHED_K_SAMPLES"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--from-csv", action="store_true")
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--skip-identity-gate", action="store_true")
    return p.parse_args()


def run_identity_gate(descriptors: list[dict], seeds: list[int]) -> tuple[pd.DataFrame, bool]:
    rows = []
    scatter_rows = []
    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                rows.append(
                    {
                        "dataset": desc["name"],
                        "seed": seed,
                        "identity_status": "LOAD_FAIL",
                        "IDENTITY_CHECK": "FAIL",
                        "error": err,
                    }
                )
                continue
            row = run_moment_identity_audit(split, seed)
            rows.append(row)
            y = normalize_labels(split.y_train)
            indexed = _build_indexed_matrix(split.X_train, y)
            ok, err = verify_scatter_splits(indexed, seed)
            scatter_rows.append(
                {
                    "dataset": split.name,
                    "seed": seed,
                    "scatter_pass": int(ok),
                    "max_abs_error": err,
                }
            )
    id_df = pd.DataFrame(rows)
    sc_df = pd.DataFrame(scatter_rows)
    sc_df.to_csv(OUT_DIR / "scatter_identity_audit.csv", index=False)
    id_df.to_csv(OUT_DIR / "identity_audit.csv", index=False)
    all_pass = bool((id_df["IDENTITY_CHECK"] == "PASS").all()) if len(id_df) else False
    return id_df, all_pass


def run_benchmark(descriptors: list[dict], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                for method in ALL_METHODS:
                    rows.append(
                        {
                            "dataset": desc["name"],
                            "seed": seed,
                            "method": method,
                            "status": "load_failure",
                            "macro_f1": float("nan"),
                            "collapsed": 1,
                            "collapse_reason": err,
                        }
                    )
                continue
            for method in ALL_METHODS:
                rows.append(evaluate_method(split, seed, method))
    return pd.DataFrame(rows)


def analyze(all_runs: pd.DataFrame) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(OUT_DIR / "all_runs.csv", index=False)

    summary = (
        all_runs.groupby("method")
        .agg(
            n_runs=("macro_f1", "count"),
            mean_macro_f1=("macro_f1", "mean"),
            median_macro_f1=("macro_f1", "median"),
            std_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            collapse_rate=("collapsed", "mean"),
            mean_n_balls=("n_balls", "mean"),
            mean_compression=("compression_ratio", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "summary_by_method.csv", index=False)

    ds_rows = []
    for (dataset, method), sub in all_runs.groupby(["dataset", "method"]):
        ds_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_seeds": len(sub),
                "mean_macro_f1": float(sub["macro_f1"].mean()),
                "mean_accuracy": float(sub["accuracy"].mean()),
                "mean_balanced_accuracy": float(sub["balanced_accuracy"].mean()),
                "collapse_rate": float(sub["collapsed"].mean()),
                "mean_n_balls": float(sub["n_balls"].mean()),
                "mean_compression_ratio": float(sub["compression_ratio"].mean()),
            }
        )
    pd.DataFrame(ds_rows).to_csv(OUT_DIR / "dataset_summary.csv", index=False)

    paired = pd.DataFrame([paired_comparison(all_runs, a, b) for a, b in PAIRS])
    paired.to_csv(OUT_DIR / "paired_comparisons.csv", index=False)

    storage = (
        all_runs.groupby("dataset")
        .agg(
            n_train=("n_train", "first"),
            n_features=("n_features", "first"),
            mean_k=("n_balls", "mean"),
            mean_n_d=("storage_n_d", "first"),
            mean_k_d2=("storage_k_d2", "mean"),
            ratio_kd2_over_nd=("storage_ratio_kd2_over_nd", "mean"),
        )
        .reset_index()
    )
    storage["kd2_exceeds_nd"] = storage["mean_k_d2"] > storage["mean_n_d"]
    storage.to_csv(OUT_DIR / "storage_analysis.csv", index=False)

    fa = []
    for method in ALL_METHODS:
        sub = all_runs[all_runs["method"] == method]
        fa.append(
            {
                "method": method,
                "total_runs": len(sub),
                "finite_runs": int((sub["status"] == "finite").sum()),
                "classifier_failures": int((~sub["classifier_ok"]).sum()),
                "non_finite_macro_f1": int((~np.isfinite(sub["macro_f1"])).sum()),
                "collapsed_runs": int(sub["collapsed"].astype(int).sum()),
                "collapse_rate": float(sub["collapsed"].mean()),
            }
        )
    pd.DataFrame(fa).to_csv(OUT_DIR / "failure_analysis.csv", index=False)

    hp = (
        all_runs.groupby("method")
        .agg(
            mean_c1=("c1", "mean"),
            std_c1=("c1", "std"),
            mean_c2=("c2", "mean"),
            std_c2=("c2", "std"),
            mean_val_f1=("validation_macro_f1", "mean"),
        )
        .reset_index()
    )
    hp.to_csv(OUT_DIR / "hyperparameter_summary.csv", index=False)

    _make_plots(all_runs, paired)
    verdict = _verdict(all_runs, paired)
    _write_readme(verdict, paired, summary)
    _print_report(all_runs, paired, summary, verdict)
    meta = {"verdict": verdict, "paired": paired.to_dict(orient="records")}
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _make_plots(all_runs: pd.DataFrame, paired: pd.DataFrame) -> None:
    per_ds = (
        all_runs.groupby(["dataset", "method"])["macro_f1"].mean().reset_index()
    )
    pivot = per_ds.pivot(index="dataset", columns="method", values="macro_f1")
    cols = [c for c in PRIMARY_METHODS if c in pivot.columns]
    ax = pivot[cols].mean().sort_values().plot(kind="barh", figsize=(8, 4), title="Mean macro F1 by arm (dataset means)")
    ax.set_xlabel("macro F1")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "macro_f1_by_arm.png", dpi=120)
    plt.close()

    delta_cols = [
        ("L1_POPULATION", "L0_CENTER_UNIT", "L1-L0"),
        ("L2_POPULATION_SCATTER", "L1_POPULATION", "L2-L1"),
        ("L2_POPULATION_SCATTER", "L0_CENTER_UNIT", "L2-L0"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, means = [], []
    for a, b, name in delta_cols:
        row = paired[paired["comparison"] == f"{a}_minus_{b}"]
        if row.empty:
            continue
        labels.append(name)
        means.append(float(row["mean_delta"].iloc[0]))
    ax.barh(labels, means)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Dataset-level mean Δ macro F1")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "delta_macro_f1_pairs.png", dpi=120)
    plt.close()

    comp = all_runs.groupby("method")["compression_ratio"].mean()
    comp.plot(kind="bar", figsize=(7, 3), title="Mean compression ratio (k/n_train)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "compression_ratio.png", dpi=120)
    plt.close()

    id_df = pd.read_csv(OUT_DIR / "identity_audit.csv")
    ax = id_df["parameter_error_abs"].clip(upper=1e-6).plot(
        kind="hist", bins=30, figsize=(6, 3), title="L3 vs RAW param |Δ| (partition path)"
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "l3_raw_parameter_error.png", dpi=120)
    plt.close()


def _verdict(all_runs: pd.DataFrame, paired: pd.DataFrame) -> str:
    id_df = pd.read_csv(OUT_DIR / "identity_audit.csv")
    if not (id_df["IDENTITY_CHECK"] == "PASS").all():
        return "D — IMPLEMENTATION / IDENTITY FAILURE"

    def _delta(a: str, b: str) -> float:
        row = paired[paired["comparison"] == f"{a}_minus_{b}"]
        return float(row["mean_delta"].iloc[0]) if len(row) else float("nan")

    l1_l0 = _delta("L1_POPULATION", "L0_CENTER_UNIT")
    l2_l1 = _delta("L2_POPULATION_SCATTER", "L1_POPULATION")
    l3_raw = _delta("L3_MOMENT_EXACT", "RAW_LSTSVM")
    l1_matched = _delta("L1_POPULATION", "MATCHED_K_SAMPLES")

    l3_raw_ok = abs(l3_raw) < 0.01  # performance on baseline path may differ from RAW

    strong = (
        l1_l0 > 0.02
        and l1_matched > 0
        and l2_l1 > 0.005
    )
    partial = (l1_l0 > 0.01) or (l2_l1 > 0.01) or (l1_matched > 0)
    if strong:
        return "A — STRONG ESTIMATOR-MISMATCH EVIDENCE"
    if partial:
        return "B — PARTIAL ESTIMATOR-MISMATCH EVIDENCE"
    return "C — NO STRONG SUPPORT"


def _write_readme(verdict: str, paired: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# A4 — Estimator-information ladder",
        "",
        "## Prediction rule",
        PROTOCOL.prediction,
        "",
        "RULE_REPO is locked for continuity with Phase 7/8 and prior audits. "
        "The published normalized rule is materially different (A3) and is **not** the primary rule here.",
        "",
        "## Arms",
        "- **RAW_LSTSVM**: per-sample L3 moments (radius shift off)",
        "- **L0_CENTER_UNIT**: unit-weight center outer products",
        "- **L1_POPULATION**: n_j-weighted centers",
        "- **L2/L3**: n_j center + scatter block; L3 equals exact sum of sample outer products per ball",
        "- **MATCHED_K_SAMPLES**: stratified k-sample RAW baseline (k = baseline rho90 ball count)",
        "",
        "## Radius",
        PROTOCOL.radius_shift,
        "",
        "## Granulation paths",
        f"- Baseline performance: {PROTOCOL.baseline_path}",
        f"- Identity audit: {PROTOCOL.partition_path}",
        "",
        "## Protocol",
        PROTOCOL.split,
        PROTOCOL.granulation,
        PROTOCOL.tuning,
        "",
        "## Equations",
        "Per-class H and g assembled as in `src/ladder_gblstsvm.py` with radius removed from rhs. "
        "L3: sum_j ( n_j u_j u_j^T + blockdiag(S_j,0) ), g = sum_j n_j u_j.",
        "",
        "## Statistics",
        "Primary unit: dataset (mean over 10 seeds). Wilcoxon + bootstrap CI on 17 datasets.",
        "",
        "## Verdict",
        verdict,
        "",
        "## Paired comparisons",
        paired.to_csv(None, index=False) if False else "",
    ]
    lines = [ln for ln in lines if ln != "" or lines.index(ln) < 30]
    text = "\n".join(
        [
            "# A4 — Estimator-information ladder",
            "",
            "## Prediction rule",
            PROTOCOL.prediction,
            "",
            "RULE_REPO locked; paper rule is sensitivity-only per A3.",
            "",
            "## Verdict",
            verdict,
            "",
            "See CSV outputs for full tables.",
        ]
    )
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def _print_report(
    all_runs: pd.DataFrame,
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    verdict: str,
) -> None:
    id_df = pd.read_csv(OUT_DIR / "identity_audit.csv")
    print("\n=== A4 ESTIMATOR-INFORMATION LADDER ===\n")
    print("A. Implementation: experiment-only modules; radius OFF; RULE_REPO=reference")
    print("\nB. Identity audit")
    print(f"   IDENTITY_CHECK all pass: {(id_df['IDENTITY_CHECK']=='PASS').all()}")
    print(f"   max H abs err: {id_df['H_error_abs'].max():.3e}")
    print(f"   max param err: {id_df['parameter_error_abs'].max():.3e}")
    print(f"   max pred disagreement: {id_df['prediction_disagreement'].max():.3e}")

    print("\nC. Main performance (macro F1)")
    for _, r in summary.iterrows():
        print(
            f"   {r['method']:28s} mean={r['mean_macro_f1']:.4f} "
            f"collapse={r['collapse_rate']:.3f} k={r['mean_n_balls']:.1f}"
        )

    print("\nD. Pairwise (dataset-level delta)")
    for _, r in paired.iterrows():
        print(
            f"   {r['comparison']:40s} delta={r.get('mean_delta', float('nan')):+.4f} "
            f"p={r.get('wilcoxon_p', float('nan')):.4f}"
        )

    print("\nE. Key answers")
    print("   1. Population: see L1-L0 delta")
    print("   2. Scatter: see L2-L1 delta")
    print("   3. L3=RAW on partition path: identity_audit.csv")
    print("   4. L1 vs matched k: paired row")
    print("   5. Compression: L0-L2 share same k on baseline path")
    print("\nF. Limitations: k*d^2 may exceed n*d; baseline path drops singleton balls")
    print(f"\nG. Final verdict: {verdict}\n")


def main() -> int:
    global OUT_DIR, PLOT_DIR
    args = parse_args()
    OUT_DIR = args.output_dir
    PLOT_DIR = OUT_DIR / "plots"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        all_runs = pd.read_csv(OUT_DIR / "all_runs.csv")
        analyze(all_runs)
        return 0

    descriptors = discover_available_datasets()
    seeds = SEEDS
    if args.smoke:
        descriptors = descriptors[:2]
        seeds = [0]

    id_df, gate_ok = run_identity_gate(descriptors, seeds)
    print(f"Identity gate: {'PASS' if gate_ok else 'FAIL'} ({len(id_df)} checks)")
    if not gate_ok and not args.skip_identity_gate:
        print("STOP: identity check failed. Fix implementation before full benchmark.")
        return 1

    all_runs = run_benchmark(descriptors, seeds)
    analyze(all_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
