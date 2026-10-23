"""
Experiment A3 — prediction-rule audit (RULE_REPO vs RULE_REFERENCE).

Run from GBLSTSVM root:
  python experiments/audit_prediction_rules.py
  python experiments/audit_prediction_rules.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.prediction_rule_audit_common import (  # noqa: E402
    CONVENTION,
    SEEDS,
    classify_disagreement_examples,
    dataset_level_stats,
    disagreement_stats,
    fit_rho90_hyperplanes,
    metrics_for_rule,
    predict_with_rule,
    run_deterministic_disagreement_check,
    spearman_safe,
)
from src.prediction_diagnostics import compute_prediction_diagnostics, summarize_disagreements  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "audit_prediction_rules"

EXPECTED_DATASETS = [
    "australian",
    "breast_cancer",
    "breast_cancer_wisconsin_diagnostic",
    "breast_cancer_wisconsin_original",
    "chronic_kidney_disease",
    "congressional_voting_records",
    "credit_approval",
    "diabetes",
    "ecoli2",
    "german_credit",
    "heart",
    "ionosphere",
    "messidor_features",
    "spambase",
    "tic_tac_toe",
    "vehicle0",
    "wholesale_customers",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A3 prediction-rule audit")
    p.add_argument("--smoke", action="store_true", help="2 datasets, seed 0 only")
    p.add_argument(
        "--from-csv",
        action="store_true",
        help="Recompute summaries from existing run_level_results.csv",
    )
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def run_benchmark(descriptors: list[dict], seeds: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    run_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    all_repo: list[float] = []
    all_ref: list[float] = []
    all_dis: list[float] = []
    all_norm_ratio: list[float] = []
    all_min_abs_f: list[float] = []

    diag_examples: list[dict[str, Any]] = []

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed=seed)
            if split is None:
                failures.append(
                    {
                        "dataset": desc["name"],
                        "seed": seed,
                        "stage": "load",
                        "reason": err,
                    }
                )
                for rule in ("RULE_REPO", "RULE_REFERENCE"):
                    run_rows.append(
                        {
                            "dataset": desc["name"],
                            "seed": seed,
                            "prediction_rule": rule,
                            "status": "exception",
                            "classifier_ok": False,
                            "accuracy": float("nan"),
                            "macro_f1": float("nan"),
                            "balanced_accuracy": float("nan"),
                            "collapsed": 1,
                            "collapse_reason": err,
                        }
                    )
                continue

            fit = fit_rho90_hyperplanes(split, seed)
            if fit.status == "exception":
                failures.append(
                    {
                        "dataset": split.name,
                        "seed": seed,
                        "stage": "fit",
                        "reason": fit.classifier_error,
                    }
                )

            m_repo = metrics_for_rule(split, fit, "RULE_REPO")
            m_ref = metrics_for_rule(split, fit, "RULE_REFERENCE")

            dis = {"disagree_rate": float("nan")}
            if (
                fit.classifier_ok
                and fit.w1 is not None
                and "y_pred" in m_repo
                and "y_pred" in m_ref
            ):
                y_repo = m_repo.pop("y_pred")
                y_ref = m_ref.pop("y_pred")
                dis = disagreement_stats(y_repo, y_ref, split.y_test)
                if len(diag_examples) < 3 and dis["n_disagree"] > 0:
                    ex = classify_disagreement_examples(
                        split.X_test,
                        fit.w1,
                        fit.b1,
                        fit.w2,
                        fit.b2,
                        y_repo,
                        y_ref,
                        max_examples=3,
                    )
                    for e in ex:
                        e["dataset"] = split.name
                        e["seed"] = seed
                        diag_examples.append(e)

                abs_f1 = np.abs(split.X_test @ fit.w1 + fit.b1)
                abs_f2 = np.abs(split.X_test @ fit.w2 + fit.b2)
                min_margin = float(np.min(np.minimum(abs_f1, abs_f2)))
                all_dis.append(dis["disagree_rate"])
                all_norm_ratio.append(m_repo["norm_ratio"])
                all_min_abs_f.append(min_margin)
                all_repo.extend(y_repo.tolist())
                all_ref.extend(y_ref.tolist())

            imb = float(split.info.get("imbalance_ratio", float("nan")))
            n_test = int(split.X_test.shape[0])

            pair_rows.append(
                {
                    "dataset": split.name,
                    "seed": seed,
                    "n_test": n_test,
                    "n_balls": fit.n_balls,
                    "imbalance_ratio": imb,
                    "classifier_ok": fit.classifier_ok,
                    "fit_status": fit.status,
                    "classifier_error": fit.classifier_error,
                    "norm_w1": m_repo.get("norm_w1", float("nan")),
                    "norm_w2": m_repo.get("norm_w2", float("nan")),
                    "norm_ratio": m_repo.get("norm_ratio", float("nan")),
                    "disagree_rate": dis.get("disagree_rate", float("nan")),
                    "macro_f1_repo": m_repo.get("macro_f1", float("nan")),
                    "macro_f1_reference": m_ref.get("macro_f1", float("nan")),
                    "delta_macro_f1_ref_minus_repo": (
                        float(m_ref["macro_f1"] - m_repo["macro_f1"])
                        if np.isfinite(m_ref.get("macro_f1", float("nan")))
                        and np.isfinite(m_repo.get("macro_f1", float("nan")))
                        else float("nan")
                    ),
                    "collapsed_repo": m_repo.get("collapsed", 1),
                    "collapsed_reference": m_ref.get("collapsed", 1),
                    **{k: v for k, v in dis.items() if k not in ("cm_repo_rows_ref_cols",)},
                }
            )

            for rule_name, m in (("RULE_REPO", m_repo), ("RULE_REFERENCE", m_ref)):
                cm = m.get("confusion_matrix")
                row = {
                    "dataset": split.name,
                    "seed": seed,
                    "prediction_rule": rule_name,
                    "n_test": n_test,
                    "n_balls": fit.n_balls,
                    "imbalance_ratio": imb,
                    "norm_w1": m.get("norm_w1", float("nan")),
                    "norm_w2": m.get("norm_w2", float("nan")),
                    "norm_ratio": m.get("norm_ratio", float("nan")),
                    "disagree_rate_vs_other_rule": dis.get("disagree_rate", float("nan")),
                    "accuracy": m.get("accuracy", float("nan")),
                    "macro_f1": m.get("macro_f1", float("nan")),
                    "balanced_accuracy": m.get("balanced_accuracy", float("nan")),
                    "f1_pos": m.get("f1_pos", float("nan")),
                    "f1_neg": m.get("f1_neg", float("nan")),
                    "status": m.get("status", "nan"),
                    "classifier_ok": m.get("classifier_ok", False),
                    "classifier_error": m.get("classifier_error"),
                    "collapsed": m.get("collapsed", 1),
                    "collapse_reason": m.get("collapse_reason"),
                    "prediction_collapse": m.get("prediction_collapse", float("nan")),
                }
                if cm is not None:
                    row["cm_tn"] = cm[0][0]
                    row["cm_fp"] = cm[0][1]
                    row["cm_fn"] = cm[1][0]
                    row["cm_tp"] = cm[1][1]
                run_rows.append(row)

    # Global pooled disagreement (benchmark test predictions)
    pooled: dict[str, Any] = {}
    if all_repo:
        y_r = np.array(all_repo, dtype=float)
        y_p = np.array(all_ref, dtype=float)
        pooled = {
            "n_predictions_pooled": int(y_r.size),
            "n_agree_pooled": int(np.sum(y_r == y_p)),
            "n_disagree_pooled": int(np.sum(y_r != y_p)),
            "disagree_rate_pooled": float(np.mean(y_r != y_p)),
        }

    return run_rows, pair_rows, failures, pooled, diag_examples, {
        "all_dis": all_dis,
        "all_norm_ratio": all_norm_ratio,
        "all_min_abs_f": all_min_abs_f,
    }


def build_summaries(
    run_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    pooled: dict[str, Any],
    det: dict[str, Any],
    diag_examples: list[dict],
    geom_arrays: dict[str, list[float]],
) -> None:
    # Dataset comparison
    ds_rows = []
    for dataset, sub in pair_df.groupby("dataset"):
        deltas = sub["delta_macro_f1_ref_minus_repo"].to_numpy(dtype=float)
        d_acc = (
            sub["macro_f1_reference"].to_numpy(dtype=float)
            - sub["macro_f1_repo"].to_numpy(dtype=float)
        )
        ds_rows.append(
            {
                "dataset": dataset,
                "n_seeds": len(sub),
                "mean_disagree_rate": float(np.nanmean(sub["disagree_rate"])),
                "mean_norm_ratio": float(np.nanmean(sub["norm_ratio"])),
                "mean_log_norm_ratio": float(
                    np.nanmean(np.log(np.maximum(sub["norm_ratio"], 1e-300)))
                ),
                "mean_delta_macro_f1": float(np.nanmean(deltas)),
                "mean_delta_accuracy_proxy": float(np.nanmean(d_acc)),
                "frac_seeds_collapsed_repo": float(np.mean(sub["collapsed_repo"])),
                "frac_seeds_collapsed_reference": float(np.mean(sub["collapsed_reference"])),
                "imbalance_ratio": float(sub["imbalance_ratio"].iloc[0]),
            }
        )
    dataset_comparison = pd.DataFrame(ds_rows)

    def _dataset_deltas(col: str) -> pd.Series:
        wide = run_df.pivot_table(
            index=["dataset", "seed"], columns="prediction_rule", values=col
        )
        delta = wide["RULE_REFERENCE"] - wide["RULE_REPO"]
        return delta.groupby(level=0).mean()

    summary_rows = []
    for metric_name, col in (
        ("delta_macro_f1_reference_minus_repo", "macro_f1"),
        ("delta_accuracy_reference_minus_repo", "accuracy"),
        ("delta_balanced_accuracy_reference_minus_repo", "balanced_accuracy"),
    ):
        ds_delta = _dataset_deltas(col)
        stats_row = dataset_level_stats(ds_delta.to_numpy())
        summary_rows.append({"metric": metric_name, **stats_row})
    dataset_summary = pd.DataFrame(summary_rows)
    delta_stats = {k: v for k, v in summary_rows[0].items() if k != "metric"}

    # Disagreement summary
    dis_summary = {
        "deterministic_synthetic": det,
        "benchmark_test_pooled": pooled,
        "per_class_synthetic": {},
    }
    # Per-class on synthetic
    from experiments.prediction_rule_audit_common import fit_synthetic_clean_gblstsvm

    model, X, y = fit_synthetic_clean_gblstsvm()
    diag = compute_prediction_diagnostics(X, model.w1_, model.b1_, model.w2_, model.b2_)
    y_r = predict_with_rule(X, model.w1_, model.b1_, model.w2_, model.b2_, "RULE_REPO")
    y_p = predict_with_rule(X, model.w1_, model.b1_, model.w2_, model.b2_, "RULE_REFERENCE")
    dis_summary["per_class_synthetic"] = disagreement_stats(y_r, y_p, y)

    dis_df = pd.DataFrame(
        [
            {"scope": "synthetic_train", **{k: v for k, v in det.items() if k != "context"}},
            {"scope": "benchmark_test_pooled", **pooled},
        ]
    )

    # Norm geometry correlations at dataset level
    dc = dataset_comparison
    r_nr, p_nr = spearman_safe(
        dc["mean_disagree_rate"].to_numpy(), dc["mean_norm_ratio"].to_numpy()
    )
    r_log, p_log = spearman_safe(
        dc["mean_disagree_rate"].to_numpy(), dc["mean_log_norm_ratio"].to_numpy()
    )
    r_imb, p_imb = spearman_safe(
        dc["mean_disagree_rate"].to_numpy(), dc["imbalance_ratio"].to_numpy()
    )
    r_dlt, p_dlt = spearman_safe(
        dc["mean_disagree_rate"].to_numpy(), dc["mean_delta_macro_f1"].to_numpy()
    )
    geom_df = pd.DataFrame(
        [
            {
                "correlation_pair": "mean_disagree_rate vs mean_norm_ratio",
                "spearman_r": r_nr,
                "spearman_p": p_nr,
                "level": "dataset",
            },
            {
                "correlation_pair": "mean_disagree_rate vs mean_log_norm_ratio",
                "spearman_r": r_log,
                "spearman_p": p_log,
                "level": "dataset",
            },
            {
                "correlation_pair": "mean_disagree_rate vs imbalance_ratio",
                "spearman_r": r_imb,
                "spearman_p": p_imb,
                "level": "dataset",
            },
            {
                "correlation_pair": "mean_disagree_rate vs mean_delta_macro_f1",
                "spearman_r": r_dlt,
                "spearman_p": p_dlt,
                "level": "dataset",
            },
        ]
    )

    # Seed-level descriptive
    seed_desc = {
        "mean_disagree_rate": float(np.nanmean(pair_df["disagree_rate"])),
        "median_disagree_rate": float(np.nanmedian(pair_df["disagree_rate"])),
        "std_disagree_rate": float(np.nanstd(pair_df["disagree_rate"], ddof=1)),
    }

    # Failure analysis
    fa_rows = []
    for rule in ("RULE_REPO", "RULE_REFERENCE"):
        sub = run_df[run_df["prediction_rule"] == rule]
        n = len(sub)
        fa_rows.append(
            {
                "prediction_rule": rule,
                "total_runs": n,
                "finite_runs": int((sub["status"] == "finite").sum()),
                "nan_runs": int((sub["status"] == "nan").sum()),
                "exception_runs": int((sub["status"] == "exception").sum()),
                "collapsed_runs": int(sub["collapsed"].astype(int).sum()),
                "collapse_rate": float(sub["collapsed"].astype(int).mean()),
                "classifier_failures": int((~sub["classifier_ok"].astype(bool)).sum()),
            }
        )

    # Finite-only and failure-aware macro F1 means at dataset level
    comp_rows = []
    for dataset, sub in pair_df.groupby("dataset"):
        fin = sub[np.isfinite(sub["macro_f1_repo"]) & np.isfinite(sub["macro_f1_reference"])]
        comp_rows.append(
            {
                "dataset": dataset,
                "view": "finite_only",
                "mean_macro_f1_repo": float(fin["macro_f1_repo"].mean()) if len(fin) else float("nan"),
                "mean_macro_f1_reference": float(fin["macro_f1_reference"].mean())
                if len(fin)
                else float("nan"),
                "mean_delta": float(fin["delta_macro_f1_ref_minus_repo"].mean())
                if len(fin)
                else float("nan"),
            }
        )
        sub_fa = sub.copy()
        sub_fa.loc[sub_fa["collapsed_repo"] == 1, "macro_f1_repo"] = 0.0
        sub_fa.loc[sub_fa["collapsed_reference"] == 1, "macro_f1_reference"] = 0.0
        comp_rows.append(
            {
                "dataset": dataset,
                "view": "failure_aware_zero",
                "mean_macro_f1_repo": float(sub_fa["macro_f1_repo"].mean()),
                "mean_macro_f1_reference": float(sub_fa["macro_f1_reference"].mean()),
                "mean_delta": float(
                    sub_fa["macro_f1_reference"].mean() - sub_fa["macro_f1_repo"].mean()
                ),
            }
        )
    comp_extra = pd.DataFrame(comp_rows)

    dis_summary["seed_level_descriptive"] = seed_desc
    dis_summary["disagreement_example_diagnostics"] = diag_examples
    dis_summary["synthetic_summarize"] = summarize_disagreements(diag)

    run_df.to_csv(OUT_DIR / "run_level_results.csv", index=False)
    pair_df.to_csv(OUT_DIR / "pair_level_results.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    dataset_comparison.to_csv(OUT_DIR / "dataset_comparison.csv", index=False)
    dis_df.to_csv(OUT_DIR / "disagreement_summary.csv", index=False)
    geom_df.to_csv(OUT_DIR / "norm_geometry.csv", index=False)
    pd.DataFrame(fa_rows).to_csv(OUT_DIR / "failure_analysis.csv", index=False)
    comp_extra.to_csv(OUT_DIR / "dataset_comparison_views.csv", index=False)
    (OUT_DIR / "disagreement_detail.json").write_text(
        json.dumps(dis_summary, indent=2, default=str), encoding="utf-8"
    )

    _write_readme(delta_stats, det, pooled, seed_desc, fa_rows)
    verdict = _print_verdict(delta_stats, det, pooled, dataset_comparison, fa_rows, pair_df)
    return verdict


def _write_readme(
    delta_stats: dict,
    det: dict,
    pooled: dict,
    seed_desc: dict,
    fa_rows: list[dict],
) -> None:
    c = CONVENTION
    lines = [
        "# A3 — Prediction rule audit",
        "",
        "## Rules",
        f"- **RULE_REPO**: {c.rule_repo}",
        f"- **RULE_REFERENCE (paper)**: {c.rule_reference}",
        "",
        "## Mathematical definitions",
        f"- {c.f_definition}",
        f"- {c.w_definition}",
        f"- {c.norm_definition}",
        f"- {c.hyperplane_1}",
        f"- {c.hyperplane_2}",
        "",
        "## Class / sign convention",
        f"- {c.label_encoding}",
        f"- {c.repo_zero_handling}",
        f"- Ambiguity: {c.ambiguity}",
        "",
        "## Protocol",
        "- 17 discoverable datasets × seeds 0–9 (170 pairs).",
        "- Split: 70% train / 15% val / 15% test (stratified, `phase8_common.stratified_three_way_split`).",
        "- Granulation: rho=0.90 (`validation_selected_k_common.REPRESENTATION`).",
        "- Fit: single twin-hyperplane solve per pair (CW + Gaussian weights, fixed c1/c2) — same as `fit_gaussian_with_predictions` / FULL_RHO90 path; **not refitted** between rules.",
        "- Predictions: both rules applied to identical (w1,b1,w2,b2).",
        "",
        "## Runs",
        "- 170 dataset×seed fits × 2 rules = 340 run-level metric rows.",
        "",
        "## Failure / collapse",
        "- Collapse: n_balls<=3, classifier failure, non-finite macro F1, or single predicted class (`validation_selected_k` policy).",
        "- Status `finite` / `nan` / `exception` preserved; metrics not silently zeroed in primary tables.",
        "- Secondary `dataset_comparison_views.csv`: finite-only vs failure-aware (zero on collapse).",
        "",
        "## Primary statistics",
        "- Unit of inference: **dataset** (mean delta macro F1 across 10 seeds).",
        "- Wilcoxon signed-rank on dataset means; bootstrap 95% CI over datasets.",
        "",
        "## Deviations from prior scripts",
        "- Does not run compression / K-selection; uses full rho90 ball set only.",
        "- Does not retune c1/c2 or scale features (matches principled-complexity FULL_RHO90 loader path).",
        "",
        "## Key synthetic diagnostic",
        f"- Reproduced disagree_rate: {det.get('disagree_rate', float('nan')):.4f} (target ~0.29)",
        "",
        "## Pooled benchmark test disagreement",
        f"- {pooled.get('disagree_rate_pooled', float('nan'))}",
        "",
        "## Dataset-level delta macro F1 (reference − repo)",
        f"- mean={delta_stats.get('mean_delta', float('nan'))}, "
        f"median={delta_stats.get('median_delta', float('nan'))}, "
        f"W/T/L={delta_stats.get('wins')}/{delta_stats.get('ties')}/{delta_stats.get('losses')}, "
        f"Wilcoxon p={delta_stats.get('wilcoxon_p', float('nan'))}",
        "",
        "## Failure rates",
    ]
    for row in fa_rows:
        lines.append(
            f"- {row['prediction_rule']}: collapse_rate={row['collapse_rate']:.3f}, "
            f"finite={row['finite_runs']}/{row['total_runs']}"
        )
    lines.append("")
    lines.append("## Seed-level (descriptive only)")
    lines.append(json.dumps(seed_desc, indent=2))
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _print_verdict(
    delta_stats: dict,
    det: dict,
    pooled: dict,
    dataset_comparison: pd.DataFrame,
    fa_rows: list[dict],
    pair_df: pd.DataFrame,
) -> str:
    mean_delta = float(delta_stats.get("mean_delta", float("nan")))
    repro = abs(float(det.get("disagree_rate", 0)) - 0.29) < 0.02
    pooled_rate = float(pooled.get("disagree_rate_pooled", float("nan")))

    repo_fa = next(r for r in fa_rows if r["prediction_rule"] == "RULE_REPO")
    ref_fa = next(r for r in fa_rows if r["prediction_rule"] == "RULE_REFERENCE")

    substantial_disagree = pooled_rate > 0.05 or float(det.get("disagree_rate", 0)) > 0.05
    wilcoxon_p = float(delta_stats.get("wilcoxon_p", 1))
    substantial_perf = abs(mean_delta) > 0.02 and wilcoxon_p < 0.05
    collapse_diff = abs(repo_fa["collapse_rate"] - ref_fa["collapse_rate"]) > 0.02

    if not CONVENTION.ambiguity.startswith("None"):
        verdict = "D. IMPLEMENTATION/CONVENTION AMBIGUITY — DO NOT INTERPRET"
        return verdict

    if substantial_disagree and (substantial_perf or collapse_diff):
        verdict = "A. PREDICTION RULE IS MATERIAL"
    elif substantial_disagree:
        verdict = "B. PREDICTION RULE DIFFERENCE IS REAL BUT NOT DECISIVE"
    else:
        verdict = "C. PREDICTION RULE IS NOT MATERIAL"

    print("\n" + "=" * 60)
    print("A3 PREDICTION-RULE AUDIT — RESEARCH VERDICT")
    print("=" * 60)
    print(verdict)
    print()
    print("Answers:")
    print(f"1. ~29% synthetic disagreement reproduced? {'YES' if repro else 'NO'} "
          f"(rate={det.get('disagree_rate'):.4f}, n={det.get('n_samples')})")
    print(f"2. Reference rule improves macro F1 (dataset mean delta)? "
          f"{'YES' if mean_delta > 0 else 'NO'} (delta={mean_delta:+.4f})")
    print(f"3. Reduces collapse? {'YES' if ref_fa['collapse_rate'] < repo_fa['collapse_rate'] else 'NO'} "
          f"(repo={repo_fa['collapse_rate']:.3f}, ref={ref_fa['collapse_rate']:.3f})")
    n_pos = int((dataset_comparison["mean_delta_macro_f1"] > 0).sum())
    n_neg = int((dataset_comparison["mean_delta_macro_f1"] < 0).sum())
    print(f"4. Consistent across datasets? mixed ({n_pos} positive mean delta, {n_neg} negative)")
    r, _ = spearman_safe(
        dataset_comparison["mean_disagree_rate"].to_numpy(),
        dataset_comparison["mean_norm_ratio"].to_numpy(),
    )
    print(f"5. Correlated with norm imbalance? Spearman r={r:.3f} (see norm_geometry.csv)")
    print("6. Subsequent ladder: keep RULE_REPO (repository reference) unless paper replication required.")
    print("7. Prior results using reference rule unchanged; paper-rule counterfactual differs where reported.")
    print("8. Valid regardless of rule: granulation/solver audits on same hyperplanes; collapse from ball count.")
    print(f"\nPooled test disagreement rate: {pooled_rate:.4f}")
    print(f"Seed-level mean disagree rate: {pair_df['disagree_rate'].mean():.4f}")
    return verdict


def main() -> int:
    global OUT_DIR
    args = parse_args()
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        run_df = pd.read_csv(OUT_DIR / "run_level_results.csv")
        pair_path = OUT_DIR / "pair_level_results.csv"
        if not pair_path.exists():
            raise SystemExit("Missing pair_level_results.csv; rerun full audit first.")
        pair_df = pd.read_csv(pair_path)
        det = run_deterministic_disagreement_check()
        dis_path = OUT_DIR / "disagreement_detail.json"
        pooled = {}
        if dis_path.exists():
            pooled = json.loads(dis_path.read_text(encoding="utf-8")).get("benchmark_test_pooled", {})
        diag_examples = json.loads(dis_path.read_text(encoding="utf-8")).get(
            "disagreement_example_diagnostics", []
        )
        build_summaries(run_df, pair_df, pooled, det, diag_examples, {})
        print(f"Regenerated summaries in {OUT_DIR}")
        return 0

    det = run_deterministic_disagreement_check()
    print("=== Part 1: deterministic synthetic diagnostic ===")
    for k, v in det.items():
        print(f"  {k}: {v}")

    descriptors = discover_available_datasets()
    names = {d["name"] for d in descriptors}
    missing = [n for n in EXPECTED_DATASETS if n not in names]
    if missing:
        print(f"WARNING: missing datasets: {missing}")

    seeds = SEEDS
    if args.smoke:
        descriptors = descriptors[:2]
        seeds = [0]

    run_rows, pair_rows, failures, pooled, diag_examples, geom_arrays = run_benchmark(
        descriptors, seeds
    )
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "load_fit_failures.csv", index=False)

    run_df = pd.DataFrame(run_rows)
    pair_df = pd.DataFrame(pair_rows)
    verdict = build_summaries(run_df, pair_df, pooled, det, diag_examples, geom_arrays)
    print(f"\nWrote outputs to {OUT_DIR}")
    print(f"Recommended next experiment: proceed to estimator-information ladder (A4) with RULE_REPO locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
