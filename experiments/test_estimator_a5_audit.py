"""
A5 — Estimator audit: k-invariance, RULE_REF rescoring, collapse analysis.

Run from GBLSTSVM root:
  python experiments/test_estimator_a5_audit.py --smoke
  python experiments/test_estimator_a5_audit.py
  python experiments/test_estimator_a5_audit.py --from-csv
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
from experiments.estimator_a5_audit_common import (  # noqa: E402
    K_VALUES,
    RESCORE_METHODS,
    RULE_REF,
    RULE_REPO,
    SEEDS,
    dataset_level_stats,
    paired_macro_f1_deltas,
    run_k_invariance_cell,
    run_rule_rescore_cell,
)
from experiments.estimator_information_ladder_common import IDENTITY_TOL as A4_ID_TOL  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "estimator_a5_audit"
PLOT_DIR = OUT_DIR / "plots"
A4_RUNS_CSV = (
    Path(__file__).resolve().parent / "output" / "estimator_information_ladder" / "all_runs.csv"
)

PAIR_COMPARISONS = [
    ("L2_POPULATION_SCATTER", "L1_POPULATION"),
    ("L2_POPULATION_SCATTER", "L0_CENTER_UNIT"),
    ("L1_POPULATION", "L0_CENTER_UNIT"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A5 estimator audit")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--from-csv", action="store_true")
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--skip-k", action="store_true", help="Skip Part A (k sweep)")
    p.add_argument("--skip-rescore", action="store_true", help="Skip Part B/C rescoring")
    return p.parse_args()


def run_k_sweep(descriptors: list[dict], seeds: list[int], k_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                for k in k_values:
                    rows.append(
                        {
                            "dataset": desc["name"],
                            "seed": seed,
                            "requested_k": k,
                            "identity_pass": 0,
                            "load_error": err,
                        }
                    )
                continue
            for k in k_values:
                rows.append(run_k_invariance_cell(split, seed, k))
    return pd.DataFrame(rows)


def run_rescore(descriptors: list[dict], seeds: list[int], a4_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = a4_runs.set_index(["dataset", "seed", "method"])
    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                for method in RESCORE_METHODS:
                    for rule in (RULE_REPO, RULE_REF):
                        rows.append(
                            {
                                "dataset": desc["name"],
                                "seed": seed,
                                "method": method,
                                "prediction_rule": rule,
                                "load_error": err,
                                "collapsed": 1,
                            }
                        )
                continue
            for method in RESCORE_METHODS:
                key = (split.name, seed, method)
                if key not in lookup.index:
                    c1, c2 = 100.0, 1e-2
                else:
                    rec = lookup.loc[key]
                    c1, c2 = float(rec["c1"]), float(rec["c2"])
                rows.extend(run_rule_rescore_cell(split, seed, method, c1, c2))
    return pd.DataFrame(rows)


def summarize_k_invariance(k_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if k_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    ok = k_df[k_df["H_relative_error"].notna()] if "H_relative_error" in k_df.columns else k_df
    summary = (
        ok.groupby("requested_k")
        .agg(
            n_runs=("identity_pass", "count"),
            identity_pass_rate=("identity_pass", "mean"),
            max_H_rel=("H_relative_error", "max"),
            max_g_rel=("g_relative_error", "max"),
            max_param_rel=("parameter_relative_error", "max"),
            max_pred_dis=("prediction_disagreement_vs_raw", "max"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .reset_index()
    )
    identity = (
        ok.groupby(["dataset", "seed"])
        .agg(
            n_k=("requested_k", "count"),
            all_identity_pass=("identity_pass", "min"),
            max_param_rel=("parameter_relative_error", "max"),
            max_pred_dis=("prediction_disagreement_vs_raw", "max"),
            partition_complete_min=("partition_complete", "min"),
        )
        .reset_index()
    )
    identity["k_invariance_pass"] = (
        (identity["all_identity_pass"] == 1)
        & (identity["partition_complete_min"] == 1)
        & (identity["max_param_rel"] <= A4_ID_TOL)
        & (identity["max_pred_dis"] <= 1e-12)
    ).astype(int)
    return summary, identity


def summarize_rule_rescore(rescore_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule in (RULE_REPO, RULE_REF):
        for a, b in PAIR_COMPARISONS:
            deltas = paired_macro_f1_deltas(rescore_df, a, b, rule)
            stats = dataset_level_stats(deltas)
            rows.append(
                {
                    "prediction_rule": rule,
                    "comparison": f"{a}_minus_{b}",
                    "analysis": "confirmatory_robustness_check",
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def collapse_table(rescore_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, rule), sub in rescore_df.groupby(["method", "prediction_rule"]):
        rows.append(
            {
                "method": method,
                "prediction_rule": rule,
                "total_runs": len(sub),
                "collapsed_runs": int(sub["collapsed"].astype(int).sum()),
                "collapse_rate": float(sub["collapsed"].mean()),
                "classifier_failures": int(sub["classifier_failure"].sum()),
                "finite_macro_f1_runs": int(sub["finite_macro_f1"].sum()),
            }
        )
    return pd.DataFrame(rows)


def dataset_pairwise_rule(rescore_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for a, b in PAIR_COMPARISONS:
        for rule in (RULE_REPO, RULE_REF):
            sub = rescore_df[rescore_df["prediction_rule"] == rule]
            wide = sub.pivot_table(
                index=["dataset", "seed"], columns="method", values="macro_f1"
            )
            if a not in wide.columns or b not in wide.columns:
                continue
            delta = (wide[a] - wide[b]).groupby(level=0).mean()
            for ds, val in delta.items():
                rows.append(
                    {
                        "dataset": ds,
                        "comparison": f"{a}_minus_{b}",
                        "prediction_rule": rule,
                        "delta_macro_f1": float(val),
                    }
                )
    return pd.DataFrame(rows)


def paired_collapse_changes(rescore_df: pd.DataFrame) -> pd.DataFrame:
    """Paired collapse: same run under RULE_REPO vs RULE_REF."""
    rows: list[dict[str, Any]] = []
    for method in RESCORE_METHODS:
        sub = rescore_df[rescore_df["method"] == method]
        wide = sub.pivot_table(
            index=["dataset", "seed"], columns="prediction_rule", values="collapsed"
        )
        if RULE_REPO not in wide.columns or RULE_REF not in wide.columns:
            continue
        repo = wide[RULE_REPO].astype(int)
        ref = wide[RULE_REF].astype(int)
        rows.append(
            {
                "method": method,
                "n_pairs": int(len(wide)),
                "repo_collapsed": int(repo.sum()),
                "ref_collapsed": int(ref.sum()),
                "both_collapsed": int(((repo == 1) & (ref == 1)).sum()),
                "repo_only": int(((repo == 1) & (ref == 0)).sum()),
                "ref_only": int(((repo == 0) & (ref == 1)).sum()),
                "neither": int(((repo == 0) & (ref == 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _make_plots(k_df: pd.DataFrame, rescore_df: pd.DataFrame, rule_summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ok = k_df[k_df.get("load_error", pd.Series(dtype=float)).isna()] if "load_error" in k_df else k_df

    if not ok.empty and "requested_k" in ok.columns:
        by_k = ok.groupby("requested_k")["macro_f1"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        by_k.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("L3 macro F1 vs requested k (run means)")
        ax.set_xlabel("requested_k")
        ax.set_ylabel("macro F1")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "l3_macro_f1_vs_k.png", dpi=120)
        plt.close()

        by_k_p = ok.groupby("requested_k")["parameter_relative_error"].median()
        fig, ax = plt.subplots(figsize=(7, 4))
        by_k_p.plot(kind="bar", ax=ax, color="coral")
        ax.set_title("L3 parameter rel error vs k (median over runs)")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "l3_parameter_error_vs_k.png", dpi=120)
        plt.close()

    if not rule_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        comps = [f"{a}_minus_{b}" for a, b in PAIR_COMPARISONS]
        x = np.arange(len(comps))
        w = 0.35
        for i, (rule, color) in enumerate([(RULE_REPO, "tab:blue"), (RULE_REF, "tab:orange")]):
            sub = rule_summary[rule_summary["prediction_rule"] == rule].set_index("comparison")
            vals = [float(sub.loc[c, "mean_delta"]) if c in sub.index else 0.0 for c in comps]
            ax.bar(x + (i - 0.5) * w, vals, width=w, label=rule, color=color)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["L2−L1", "L2−L0", "L1−L0"])
        ax.set_title("Dataset-level mean Δ macro F1 (A4 arms)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "rule_ref_vs_repo_deltas.png", dpi=120)
        plt.close()

    if not rescore_df.empty:
        coll = collapse_table(rescore_df)
        pivot = coll.pivot(index="method", columns="prediction_rule", values="collapse_rate")
        ax = pivot.plot(kind="bar", figsize=(9, 4), rot=15)
        ax.set_title("Collapse rate by arm and prediction rule")
        ax.set_ylabel("collapse rate")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "collapse_rate_by_arm_and_rule.png", dpi=120)
        plt.close()


def gate_verdicts(
    k_identity: pd.DataFrame,
    rule_summary: pd.DataFrame,
    collapse_df: pd.DataFrame,
) -> dict[str, str]:
    gate_a = "FAIL"
    if not k_identity.empty and (k_identity["k_invariance_pass"] == 1).all():
        gate_a = "PASS"

    def _mean_delta(rule: str, comp: str) -> float:
        row = rule_summary[
            (rule_summary["prediction_rule"] == rule)
            & (rule_summary["comparison"] == comp)
        ]
        return float(row["mean_delta"].iloc[0]) if len(row) else float("nan")

    l2_l1_repo = _mean_delta(RULE_REPO, "L2_POPULATION_SCATTER_minus_L1_POPULATION")
    l2_l0_repo = _mean_delta(RULE_REPO, "L2_POPULATION_SCATTER_minus_L0_CENTER_UNIT")
    l2_l1_ref = _mean_delta(RULE_REF, "L2_POPULATION_SCATTER_minus_L1_POPULATION")
    l2_l0_ref = _mean_delta(RULE_REF, "L2_POPULATION_SCATTER_minus_L0_CENTER_UNIT")

    gate_b = "FAIL"
    if l2_l1_ref > 0 and l2_l0_ref > 0:
        if l2_l1_repo > 0 and l2_l0_repo > 0:
            if l2_l1_ref >= 0.5 * l2_l1_repo and l2_l0_ref >= 0.5 * l2_l0_repo:
                gate_b = "STRONG PASS"
            else:
                gate_b = "PARTIAL PASS"
        else:
            gate_b = "PARTIAL PASS"
    elif l2_l1_ref > 0 or l2_l0_ref > 0:
        gate_b = "PARTIAL PASS"

    l2_repo = collapse_df[
        (collapse_df["method"] == "L2_POPULATION_SCATTER")
        & (collapse_df["prediction_rule"] == RULE_REPO)
    ]
    l0_repo = collapse_df[
        (collapse_df["method"] == "L0_CENTER_UNIT")
        & (collapse_df["prediction_rule"] == RULE_REPO)
    ]
    l1_repo = collapse_df[
        (collapse_df["method"] == "L1_POPULATION")
        & (collapse_df["prediction_rule"] == RULE_REPO)
    ]
    gate_c = "NEUTRAL"
    if not l2_repo.empty and not l0_repo.empty and not l1_repo.empty:
        r2 = float(l2_repo["collapse_rate"].iloc[0])
        r0 = float(l0_repo["collapse_rate"].iloc[0])
        r1 = float(l1_repo["collapse_rate"].iloc[0])
        if r2 < r0 - 0.02 and r2 < r1 - 0.02:
            gate_c = "POSITIVE"
        elif r2 > r0 + 0.02 or r2 > r1 + 0.02:
            gate_c = "NEGATIVE"

    overall = "FAIL"
    if gate_a == "PASS":
        if gate_b in ("STRONG PASS", "PARTIAL PASS") and gate_c in ("POSITIVE", "NEUTRAL"):
            overall = "ROBUST" if gate_b == "STRONG PASS" and gate_c == "POSITIVE" else "PARTIAL"
        elif gate_b == "FAIL":
            overall = "PARTIAL" if gate_c != "NEGATIVE" else "FAIL"

    return {
        "GATE_A_k_invariance": gate_a,
        "GATE_B_rule_robustness": gate_b,
        "GATE_C_collapse": gate_c,
        "OVERALL": overall,
    }


def write_readme(verdicts: dict[str, str]) -> None:
    text = f"""# A5 — Estimator audit (confirmatory / robustness)

## Purpose

1. **k-invariance (Part A):** Exact moment-preserving L3 on a **complete partition** should match RAW regardless of ball count k. This is a property of linear LSTSVM sufficient statistics, **not** evidence that granular balls help.
2. **RULE_REF rescoring (Part B):** Sensitivity check on A4 fitted hyperparameters (c1/c2 from `estimator_information_ladder/all_runs.csv`); weights recomputed, **no C-grid retuning**.
3. **Collapse (Part C):** A4 collapse definition: `n_balls<=3` OR classifier failure OR non-finite macro-F1.

## Framing

- The k-sweep establishes **partition invariance** for exact L3.
- The empirical phenomenon of interest remains the gap **L0 → L1 → L2** (lossy representations).
- L3=RAW is a **mathematical control**, not the proposed contribution.
- Do not interpret L2 as proving granular balls are necessary; directional scatter claims await later anisotropy experiments.

## Gates

- GATE A: `{verdicts['GATE_A_k_invariance']}`
- GATE B: `{verdicts['GATE_B_rule_robustness']}`
- GATE C: `{verdicts['GATE_C_collapse']}`
- OVERALL: `{verdicts['OVERALL']}`

## Outputs

See CSV files and `plots/` in this directory.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def analyze(
    k_df: pd.DataFrame,
    rescore_df: pd.DataFrame,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k_df.to_csv(OUT_DIR / "k_invariance_runs.csv", index=False)
    k_summary, k_identity = summarize_k_invariance(k_df)
    k_summary.to_csv(OUT_DIR / "k_invariance_summary.csv", index=False)
    k_identity.to_csv(OUT_DIR / "k_invariance_identity.csv", index=False)

    rescore_df.to_csv(OUT_DIR / "rule_rescore_runs.csv", index=False)
    rule_summary = summarize_rule_rescore(rescore_df)
    rule_summary.to_csv(OUT_DIR / "rule_rescore_summary.csv", index=False)
    collapse_df = collapse_table(rescore_df)
    collapse_df.to_csv(OUT_DIR / "collapse_analysis.csv", index=False)
    ds_pw = dataset_pairwise_rule(rescore_df)
    ds_pw.to_csv(OUT_DIR / "dataset_pairwise_rule_comparisons.csv", index=False)
    paired_collapse_changes(rescore_df).to_csv(
        OUT_DIR / "collapse_rule_paired.csv", index=False
    )

    verdicts = gate_verdicts(k_identity, rule_summary, collapse_df)
    _make_plots(k_df, rescore_df, rule_summary)
    write_readme(verdicts)

    meta = {
        "verdicts": verdicts,
        "k_summary": k_summary.to_dict(orient="records"),
        "rule_summary": rule_summary.to_dict(orient="records"),
        "collapse": collapse_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _print_verdict(verdicts, k_identity, rule_summary, collapse_df)
    return meta


def _print_verdict(
    verdicts: dict[str, str],
    k_identity: pd.DataFrame,
    rule_summary: pd.DataFrame,
    collapse_df: pd.DataFrame,
) -> None:
    print("\n=== A5 ESTIMATOR AUDIT — MACHINE READABLE VERDICT ===")
    for k, v in verdicts.items():
        print(f"{k}: {v}")
    if not k_identity.empty:
        print(
            f"k_invariance_pass_rate: "
            f"{k_identity['k_invariance_pass'].mean():.4f} "
            f"({int(k_identity['k_invariance_pass'].sum())}/{len(k_identity)} dataset-seeds)"
        )
    print("\nRule summary (dataset-level delta macro F1):")
    for _, r in rule_summary.iterrows():
        print(
            f"  {r['prediction_rule']:16s} {r['comparison']:45s} "
            f"mean={r.get('mean_delta', float('nan')):+.4f} "
            f"p={r.get('wilcoxon_p', float('nan')):.4f}"
        )
    print("\nCollapse rates (RULE_REPO):")
    sub = collapse_df[collapse_df["prediction_rule"] == RULE_REPO]
    for _, r in sub.iterrows():
        print(f"  {r['method']:28s} rate={r['collapse_rate']:.3f} ({r['collapsed_runs']}/{r['total_runs']})")
    print(f"\nOutputs: {OUT_DIR.resolve()}")
    print(f"Plots:   {(PLOT_DIR).resolve()}\n")


def main() -> int:
    global OUT_DIR, PLOT_DIR
    args = parse_args()
    OUT_DIR = args.output_dir
    PLOT_DIR = OUT_DIR / "plots"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        k_path = OUT_DIR / "k_invariance_runs.csv"
        r_path = OUT_DIR / "rule_rescore_runs.csv"
        if not k_path.exists() or not r_path.exists():
            print("Missing CSV for --from-csv", file=sys.stderr)
            return 1
        analyze(pd.read_csv(k_path), pd.read_csv(r_path))
        return 0

    descriptors = discover_available_datasets()
    seeds = SEEDS
    k_values = K_VALUES
    if args.smoke:
        descriptors = descriptors[:2]
        seeds = [0]
        k_values = [1, 2, 4]

    k_df = pd.DataFrame()
    if not args.skip_k:
        print(f"Part A: k-invariance ({len(descriptors)} datasets × {len(seeds)} seeds × {len(k_values)} k)")
        k_df = run_k_sweep(descriptors, seeds, k_values)

    rescore_df = pd.DataFrame()
    if not args.skip_rescore:
        if not A4_RUNS_CSV.exists():
            print(f"Missing A4 runs: {A4_RUNS_CSV}", file=sys.stderr)
            return 1
        a4 = pd.read_csv(A4_RUNS_CSV)
        print(f"Part B/C: rule rescore from stored c1/c2 ({len(descriptors)}×{len(seeds)}×{len(RESCORE_METHODS)})")
        rescore_df = run_rescore(descriptors, seeds, a4)

    analyze(k_df, rescore_df)
    if not k_df.empty:
        id_df = summarize_k_invariance(k_df)[1]
        if not id_df.empty and not (id_df["k_invariance_pass"] == 1).all():
            print("WARNING: GATE A may FAIL — review k_invariance_identity.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
