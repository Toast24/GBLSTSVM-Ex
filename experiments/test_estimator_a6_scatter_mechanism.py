"""
A6 — Scatter mechanism: directional vs isotropic vs ridge-matched control.

  python experiments/test_estimator_a6_scatter_mechanism.py --smoke
  python experiments/test_estimator_a6_scatter_mechanism.py
  python experiments/test_estimator_a6_scatter_mechanism.py --from-csv
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
from experiments.estimator_a6_scatter_mechanism_common import (  # noqa: E402
    ARMS,
    RULES,
    SEEDS,
    dataset_level_stats,
    evaluate_cell,
    fit_and_predict,
    paired_delta,
    scatter_geometry,
    tune_c1_c2_for_arm,
    tune_ridge_multiplier,
    trace_matched_lambda,
)
from experiments.estimator_information_ladder_common import (  # noqa: E402
    balls_from_samples,
    build_baseline_rho90_balls,
    solve_twin_from_balls,
    split_balls,
)
from experiments.phase8_common import normalize_labels  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "estimator_a6_scatter_mechanism"
PLOT_DIR = OUT_DIR / "plots"

TUNING_MODES_LOCAL = ["MODE_A_SHARED_C", "MODE_B_PER_ARM_C"]

PAIR_TESTS = [
    ("FULL_SCATTER", "DIAG_SCATTER", "directionality"),
    ("FULL_SCATTER", "ISOTROPIC_SCATTER", "directionality"),
    ("DIAG_SCATTER", "ISOTROPIC_SCATTER", "directionality"),
    ("ISOTROPIC_SCATTER", "RIDGE_TRACE_MATCH", "ridge_control"),
    ("FULL_SCATTER", "RIDGE_TRACE_MATCH", "ridge_control"),
    ("FULL_SCATTER", "L1_POPULATION", "information_ladder"),
    ("DIAG_SCATTER", "L1_POPULATION", "information_ladder"),
    ("ISOTROPIC_SCATTER", "L1_POPULATION", "information_ladder"),
]

DECISION_ROWS = [
    ("FULL_SCATTER", "DIAG_SCATTER"),
    ("FULL_SCATTER", "ISOTROPIC_SCATTER"),
    ("ISOTROPIC_SCATTER", "RIDGE_TRACE_MATCH"),
    ("FULL_SCATTER", "L1_POPULATION"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--from-csv", action="store_true")
    p.add_argument(
        "--dump-params",
        action="store_true",
        help="Serialize w1,w2,b1,b2 per dataset/seed/arm/tuning_mode (+ RAW) to params/",
    )
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def run_benchmark(descriptors: list[dict], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ridge_rows: list[dict[str, Any]] = []
    geom_rows: list[dict[str, Any]] = []
    total = len(descriptors) * len(seeds)
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                continue
            balls = build_baseline_rho90_balls(split, seed)
            geom = scatter_geometry(balls)
            geom_rows.append({"dataset": split.name, "seed": seed, **geom})
            dim = int(split.X_train.shape[1])
            pos = [b for b in balls if b.label == 1.0]
            neg = [b for b in balls if b.label != 1.0]
            ridge_rows.append(
                {
                    "dataset": split.name,
                    "seed": seed,
                    "lambda_pos_trace_match": trace_matched_lambda(pos, dim),
                    "lambda_neg_trace_match": trace_matched_lambda(neg, dim),
                    "trace_pos": sum(np.trace(b.scatter) for b in pos if b.scatter is not None),
                    "trace_neg": sum(np.trace(b.scatter) for b in neg if b.scatter is not None),
                    "formula": "lambda_c = sum_j tr(S_j) / d added to H_c[:d,:d] after L1 moments",
                }
            )
            shared_c1, shared_c2, _ = tune_c1_c2_for_arm(
                split, balls, "L1_POPULATION", seed
            )
            for mode in TUNING_MODES_LOCAL:
                for arm in ARMS:
                    rows.extend(
                        evaluate_cell(split, seed, arm, mode, shared_c1, shared_c2)
                    )
            done += 1
            if done % 5 == 0 or done == total:
                pd.DataFrame(rows).to_csv(OUT_DIR / "runs.csv", index=False)
                print(f"  progress {done}/{total} dataset-seeds", flush=True)
    pd.DataFrame(ridge_rows).to_csv(OUT_DIR / "ridge_matching.csv", index=False)
    pd.DataFrame(geom_rows).to_csv(OUT_DIR / "scatter_geometry.csv", index=False)
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adj = [1.0] * m
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, p_values[idx] * (m - rank))
    return adj


def build_pairwise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ds_rows = []
    for rule in RULES:
        for mode in TUNING_MODES_LOCAL:
            for a, b, category in PAIR_TESTS:
                sub = df[
                    (df["prediction_rule"] == rule)
                    & (df["tuning_mode"] == mode)
                    & (df["arm"].isin([a, b]))
                ]
                wide = sub.pivot_table(
                    index=["dataset", "seed"], columns="arm", values="macro_f1"
                )
                if a in wide.columns and b in wide.columns:
                    for ds, val in (wide[a] - wide[b]).groupby(level=0).mean().items():
                        ds_rows.append(
                            {
                                "dataset": ds,
                                "prediction_rule": rule,
                                "tuning_mode": mode,
                                "comparison": f"{a}_minus_{b}",
                                "category": category,
                                "delta_macro_f1": float(val),
                            }
                        )
                deltas = paired_delta(df, a, b, rule, mode)
                stats = dataset_level_stats(deltas)
                rows.append(
                    {
                        "prediction_rule": rule,
                        "tuning_mode": mode,
                        "comparison": f"{a}_minus_{b}",
                        "category": category,
                        **stats,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty and "wilcoxon_p" in out.columns:
        raw = out["wilcoxon_p"].fillna(1.0).tolist()
        out["wilcoxon_p_holm"] = holm_adjust(raw)
    return out, pd.DataFrame(ds_rows)


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, arm, mode, rule), sub in df.groupby(
        ["dataset", "arm", "tuning_mode", "prediction_rule"]
    ):
        rows.append(
            {
                "dataset": ds,
                "arm": arm,
                "tuning_mode": mode,
                "prediction_rule": rule,
                "mean_macro_f1": float(sub["macro_f1"].mean()),
                "n_seeds": len(sub),
            }
        )
    return pd.DataFrame(rows)


def tuning_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for rule in RULES:
            da = df[
                (df["arm"] == arm)
                & (df["prediction_rule"] == rule)
                & (df["tuning_mode"] == "MODE_A_SHARED_C")
            ]
            db = df[
                (df["arm"] == arm)
                & (df["prediction_rule"] == rule)
                & (df["tuning_mode"] == "MODE_B_PER_ARM_C")
            ]
            wide_a = da.groupby("dataset")["macro_f1"].mean()
            wide_b = db.groupby("dataset")["macro_f1"].mean()
            common = wide_a.index.intersection(wide_b.index)
            deltas = (wide_b.loc[common] - wide_a.loc[common]).to_numpy()
            rows.append(
                {
                    "arm": arm,
                    "prediction_rule": rule,
                    "comparison": "MODE_B_minus_MODE_A",
                    **dataset_level_stats(deltas),
                }
            )
    return pd.DataFrame(rows)


def rule_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for mode in TUNING_MODES_LOCAL:
            repo = df[
                (df["arm"] == arm)
                & (df["tuning_mode"] == mode)
                & (df["prediction_rule"] == "RULE_REPO")
            ]
            ref = df[
                (df["arm"] == arm)
                & (df["tuning_mode"] == mode)
                & (df["prediction_rule"] == "RULE_REFERENCE")
            ]
            wr = repo.groupby("dataset")["macro_f1"].mean()
            wf = ref.groupby("dataset")["macro_f1"].mean()
            common = wr.index.intersection(wf.index)
            deltas = (wr.loc[common] - wf.loc[common]).to_numpy()
            rows.append(
                {
                    "arm": arm,
                    "tuning_mode": mode,
                    "comparison": "RULE_REPO_minus_RULE_REF",
                    **dataset_level_stats(deltas),
                }
            )
    return pd.DataFrame(rows)


def classify_result(pairwise: pd.DataFrame) -> str:
    def mean_delta(rule: str, mode: str, comp: str) -> float:
        row = pairwise[
            (pairwise["prediction_rule"] == rule)
            & (pairwise["tuning_mode"] == mode)
            & (pairwise["comparison"] == comp)
        ]
        return float(row["mean_delta"].iloc[0]) if len(row) else float("nan")

    mode = "MODE_B_PER_ARM_C"
    rule_repo = "RULE_REPO"
    rule_ref = "RULE_REFERENCE"

    full_diag = mean_delta(rule_repo, mode, "FULL_SCATTER_minus_DIAG_SCATTER")
    full_iso = mean_delta(rule_repo, mode, "FULL_SCATTER_minus_ISOTROPIC_SCATTER")
    iso_ridge = mean_delta(rule_repo, mode, "ISOTROPIC_SCATTER_minus_RIDGE_TRACE_MATCH")
    full_l1 = mean_delta(rule_repo, mode, "FULL_SCATTER_minus_L1_POPULATION")

    full_diag_ref = mean_delta(rule_ref, mode, "FULL_SCATTER_minus_DIAG_SCATTER")
    full_iso_ref = mean_delta(rule_ref, mode, "FULL_SCATTER_minus_ISOTROPIC_SCATTER")

    rule_sensitive = (full_diag > 0.005 and full_diag_ref <= 0) or (
        full_iso > 0.005 and full_iso_ref <= 0
    )

    directional = full_diag > 0.002 and full_iso > 0.002
    reg_expl = (
        abs(full_iso) < 0.003
        and abs(full_diag) < 0.003
        and abs(iso_ridge) < 0.003
    )
    if directional and not rule_sensitive:
        if full_l1 > 0.01 and abs(iso_ridge) < 0.01:
            return "MODE_B_RULE_REPO_DIRECTIONAL_SCATTER_SUPPORTED"
    if reg_expl and full_l1 > 0.01:
        return "MODE_B_RULE_REPO_REGULARIZATION_EXPLANATION_SUPPORTED"
    if rule_sensitive:
        return "MODE_B_RULE_REPO_MIXED_RULE_SENSITIVE"
    if full_l1 <= 0.005:
        return "MODE_B_RULE_REPO_NO_CLEAR_SCATTER_ADVANTAGE"
    return "MODE_B_RULE_REPO_MIXED_RULE_SENSITIVE"


def _make_plots(df: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "MODE_A_SHARED_C"
    rule = "RULE_REPO"

    def plot_pair(a: str, b: str, fname: str, title: str) -> None:
        comp = f"{a}_minus_{b}"
        row = pairwise[
            (pairwise["comparison"] == comp)
            & (pairwise["tuning_mode"] == mode)
            & (pairwise["prediction_rule"] == rule)
        ]
        if row.empty:
            return
        deltas = paired_delta(df, a, b, rule, mode)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(deltas, bins=12, color="steelblue", edgecolor="white")
        ax.axvline(float(row["mean_delta"].iloc[0]), color="crimson", ls="--", label="mean")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / fname, dpi=120)
        plt.close()

    plot_pair("FULL_SCATTER", "DIAG_SCATTER", "full_vs_diag_delta.png", "FULL - DIAG (dataset deltas)")
    plot_pair(
        "FULL_SCATTER",
        "ISOTROPIC_SCATTER",
        "full_vs_isotropic_delta.png",
        "FULL - ISOTROPIC",
    )
    plot_pair(
        "ISOTROPIC_SCATTER",
        "RIDGE_TRACE_MATCH",
        "isotropic_vs_ridge_delta.png",
        "ISOTROPIC - RIDGE_TRACE",
    )

    ds = dataset_summary(df)
    sub = ds[
        (ds["tuning_mode"] == mode)
        & (ds["prediction_rule"] == rule)
        & (ds["arm"].isin(["L1_POPULATION", "FULL_SCATTER", "ISOTROPIC_SCATTER"]))
    ]
    pivot = sub.pivot(index="dataset", columns="arm", values="mean_macro_f1")
    if not pivot.empty:
        delta = pivot.get("FULL_SCATTER", 0) - pivot.get("L1_POPULATION", 0)
        fig, ax = plt.subplots(figsize=(10, 5))
        delta.sort_values().plot(kind="barh", ax=ax)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("FULL - L1 macro F1 by dataset")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "effect_by_dataset.png", dpi=120)
        plt.close()

    geom = pd.read_csv(OUT_DIR / "scatter_geometry.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(geom["trace_full"], geom["fro_full_minus_isotropic"], alpha=0.5, s=12)
    ax.set_xlabel("trace(S) total")
    ax.set_ylabel("||S_full - S_iso||_F per ball (summed)")
    ax.set_title("Scatter geometry across runs")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scatter_trace_distribution.png", dpi=120)
    plt.close()


def print_decision_table(df: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    print("\n=== A6 DECISION TABLE (mean dataset-level delta macro F1) ===")
    for mode in TUNING_MODES_LOCAL:
        for rule in RULES:
            print(f"\n  [{mode} | {rule}]")
            for a, b in DECISION_ROWS:
                comp = f"{a}_minus_{b}"
                row = pairwise[
                    (pairwise["comparison"] == comp)
                    & (pairwise["tuning_mode"] == mode)
                    & (pairwise["prediction_rule"] == rule)
                ]
                if row.empty:
                    print(f"    {a} - {b}: n/a")
                    continue
                md = float(row["mean_delta"].iloc[0])
                p = float(row.get("wilcoxon_p", pd.Series([float("nan")])).iloc[0])
                print(f"    {a} - {b}: {md:+.4f}  (Wilcoxon p={p:.4f})")


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "runs.csv", index=False)
    pairwise, ds_pairwise = build_pairwise(df)
    pairwise.to_csv(OUT_DIR / "pairwise_comparisons.csv", index=False)
    ds_pairwise.to_csv(OUT_DIR / "dataset_pairwise_deltas.csv", index=False)
    dataset_summary(df).to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    rule_comparison(df).to_csv(OUT_DIR / "rule_comparisons.csv", index=False)
    tuning_comparison(df).to_csv(OUT_DIR / "tuning_comparisons.csv", index=False)

    verdict = classify_result(pairwise)
    _make_plots(df, pairwise)
    write_readme(verdict, pairwise)
    print_decision_table(df, pairwise)
    print(f"\nA6_AUTO_LABEL_MODE_B_RULE_REPO: {verdict}\n")

    meta = {
        "classification": verdict,
        "pairwise": pairwise.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Outputs: {OUT_DIR.resolve()}")
    return meta


def write_readme(verdict: str, pairwise: pd.DataFrame) -> None:
    text = f"""# A6 — Scatter mechanism experiment

## Guardrails

1. L3=RAW is partition-invariant (A5); mathematical control only.
2. A5 showed L2−L1 is **prediction-rule sensitive** (RULE_REPO vs RULE_REF).
3. A6 tests whether scatter gains are **directional** vs **isotropic magnitude / ridge**.
4. FULL beating ISOTROPIC alone is **not** sufficient; compare to **RIDGE_TRACE_MATCH**.
5. Consistent FULL > DIAG and FULL > ISOTROPIC supports directional second-order information.
6. If ISOTROPIC ≈ trace-matched ridge, do **not** claim covariance structure is necessary.

## Ridge placement

Per class c, L1 moments accumulate as in A4. Ridge/scatter add to `H_c[:d,:d]`:
- FULL/DIAG/ISOTROPIC: sum_j S_j^(variant) per ball.
- RIDGE_TRACE_MATCH: add `lambda_c I_d` with `lambda_c = (sum_j tr(S_j))/d`.

## Tuning

- **MODE_A_SHARED_C**: c1/c2 from L1 validation grid (A4-comparable); RIDGE_VAL_TUNED tunes lambda multiplier only.
- **MODE_B_PER_ARM_C**: independent c1/c2 per arm; RIDGE_VAL_TUNED also tunes lambda on validation.

## Classification

`{verdict}`

Primary inferential unit: **dataset** (mean over seeds). Exploratory sequence A2–A6; report full comparison matrix.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def run_dump_params(descriptors: list[dict], seeds: list[int], params_dir: Path) -> None:
    """Fit once per (dataset, seed, arm, tuning_mode) and save hyperplane parameters."""
    params_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed, onehot=True)
            if split is None:
                continue
            balls = build_baseline_rho90_balls(split, seed)
            shared_c1, shared_c2, _ = tune_c1_c2_for_arm(
                split, balls, "L1_POPULATION", seed
            )
            y = normalize_labels(split.y_train)
            for mode in TUNING_MODES_LOCAL:
                for arm in list(ARMS) + ["RAW_LSTSVM"]:
                    ridge_lp = ridge_ln = 0.0
                    if arm == "RAW_LSTSVM":
                        if mode == "MODE_A_SHARED_C":
                            c1, c2 = shared_c1, shared_c2
                        else:
                            c1, c2, _ = tune_c1_c2_for_arm(
                                split, balls, "L1_POPULATION", seed
                            )
                        pos, neg = balls_from_samples(split.X_train, y)
                        try:
                            w1, b1, w2, b2 = solve_twin_from_balls(
                                pos, neg, c1, c2, "L3"
                            )
                            ok = True
                        except (ValueError, np.linalg.LinAlgError):
                            w1 = w2 = np.zeros(split.X_train.shape[1])
                            b1 = b2 = 0.0
                            ok = False
                    else:
                        if mode == "MODE_A_SHARED_C":
                            c1, c2 = shared_c1, shared_c2
                            if arm == "RIDGE_VAL_TUNED":
                                ridge_lp, ridge_ln, _ = tune_ridge_multiplier(
                                    split, balls, seed, c1, c2
                                )
                        else:
                            c1, c2, _ = tune_c1_c2_for_arm(split, balls, arm, seed)
                            if arm == "RIDGE_VAL_TUNED":
                                ridge_lp, ridge_ln, _ = tune_ridge_multiplier(
                                    split, balls, seed, c1, c2
                                )
                        w1, b1, w2, b2, ok, _, _, _ = fit_and_predict(
                            split, balls, arm, c1, c2, ridge_lp, ridge_ln
                        )
                    tag = f"{split.name}_s{seed}_{arm}_{mode}.npz"
                    path = params_dir / tag
                    np.savez(
                        path,
                        w1=w1,
                        b1=b1,
                        w2=w2,
                        b2=b2,
                        c1=c1,
                        c2=c2,
                        classifier_ok=ok,
                    )
                    manifest.append(
                        {
                            "dataset": split.name,
                            "seed": seed,
                            "arm": arm,
                            "tuning_mode": mode,
                            "path": str(path.relative_to(params_dir)),
                            "c1": c1,
                            "c2": c2,
                            "classifier_ok": ok,
                        }
                    )
    pd.DataFrame(manifest).to_csv(params_dir / "manifest.csv", index=False)


def main() -> int:
    global OUT_DIR, PLOT_DIR
    args = parse_args()
    OUT_DIR = args.output_dir
    PLOT_DIR = OUT_DIR / "plots"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        analyze(pd.read_csv(OUT_DIR / "runs.csv"))
        return 0

    descriptors = discover_available_datasets()
    seeds = SEEDS
    if args.smoke:
        descriptors = descriptors[:2]
        seeds = [0]

    if args.dump_params:
        params_dir = OUT_DIR / "params"
        print(f"Dumping parameters to {params_dir} ...")
        run_dump_params(descriptors, seeds, params_dir)
        return 0

    print(f"Running A6 ({len(descriptors)} datasets x {len(seeds)} seeds)...")
    df = run_benchmark(descriptors, seeds)
    analyze(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
