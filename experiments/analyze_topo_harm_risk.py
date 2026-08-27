"""
Extend Topo diagnostics: ball-level purity_compact μ_G and run-level harm_risk.

Read-only — recomputes frozen granulation / membership for diagnostics only.
Does not modify src model, granulation, Topo, or weighting implementations.

harm_risk = fraction of balls where:
  compA > 0.5 AND μ_G < run median μ_G AND (n_opp <= 2 OR R_opp == 0)

Run from GBLSTSVM root:
  python experiments/analyze_topo_harm_risk.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402
from src.cbcm import ball_cbcm_weights  # noqa: E402
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.reliability import compute_radius_scale, normalize_radius  # noqa: E402
from src.topology_membership import ball_topo_weights  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "topology"
SCREENING_CSV = OUT_DIR / "topo_broad_screening_results.csv"
MECH_CSV = OUT_DIR / "r_opp_mechanism_analysis.csv"
OUT_RUN_CSV = OUT_DIR / "topo_harm_risk_runs.csv"
OUT_BALL_CSV = OUT_DIR / "topo_harm_risk_per_ball.csv"
OUT_SUMMARY_JSON = OUT_DIR / "topo_harm_risk_summary.json"
OUT_REPORT_MD = OUT_DIR / "topo_harm_risk_report.md"

SEEDS = [42, 0, 1]
COMPA_THRESHOLD = 0.5
N_OPP_THRESHOLD = 2
RADIUS_EPS = 1e-12


def _ball_weight_vector(
    weights_pos: np.ndarray,
    weights_neg: np.ndarray,
    balls,
) -> np.ndarray:
    pos_i = 0
    neg_i = 0
    out = np.empty(len(balls), dtype=float)
    for i, b in enumerate(balls):
        if b.label == 1:
            out[i] = weights_pos[pos_i]
            pos_i += 1
        else:
            out[i] = weights_neg[neg_i]
            neg_i += 1
    return out


def compute_purity_compact_mu_g(balls) -> np.ndarray:
    """μ_G = purity * exp(-(r / median_positive_radius)^2)."""
    if not balls:
        return np.array([], dtype=float)
    radii = np.array([b.radius for b in balls], dtype=float)
    purities = np.array([b.purity for b in balls], dtype=float)
    scale = compute_radius_scale(radii, method="median_positive")
    r_tilde = np.array([normalize_radius(r, scale) for r in radii], dtype=float)
    return np.clip(purities * np.exp(-np.square(r_tilde)), 0.0, 1.0)


def _analyze_run(descriptor: dict, seed: int, perf: dict) -> tuple[dict, list[dict]]:
    split, err = try_load_descriptor(descriptor, seed=seed)
    if split is None:
        raise RuntimeError(err)

    y_train = normalize_labels(split.y_train)
    np.random.seed(seed)
    balls = generate_granular_balls(
        split.X_train,
        y_train,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    ca_pos, ca_neg, ca_mem, _, _ = ball_cbcm_weights(
        balls, kappa_pos, kappa_neg, variant="competition"
    )
    topo_pos, topo_neg, topo_mem, _, _ = ball_topo_weights(
        balls, kappa_pos, kappa_neg
    )

    ca_w = _ball_weight_vector(ca_pos, ca_neg, balls)
    topo_w = _ball_weight_vector(topo_pos, topo_neg, balls)
    topo_minus_ca = topo_w - ca_w

    comp_a = topo_mem.ambiguity
    r_opp = topo_mem.r_opp
    n_opp = topo_mem.n_opp
    mu_g = compute_purity_compact_mu_g(balls)

    n_balls = len(balls)
    mu_g_median = float(np.median(mu_g)) if mu_g.size else np.nan
    high_comp = comp_a > COMPA_THRESHOLD

    low_mu_g = mu_g < mu_g_median
    sparse_opp = n_opp <= N_OPP_THRESHOLD
    r_opp_zero = r_opp == 0

    harm_mask = high_comp & low_mu_g & (sparse_opp | r_opp_zero)
    n_high_comp = int(np.sum(high_comp))

    harm_risk = float(np.mean(harm_mask)) if n_balls else np.nan
    frac_n_opp_risk = (
        float(np.mean(high_comp & sparse_opp)) if n_balls else np.nan
    )
    frac_r_opp_zero_hc = (
        float(np.mean(high_comp & r_opp_zero)) if n_balls else np.nan
    )
    frac_low_mu_g_hc = (
        float(np.mean(high_comp & low_mu_g)) if n_balls else np.nan
    )

    ball_rows: list[dict] = []
    for i, b in enumerate(balls):
        ball_rows.append(
            {
                "dataset": split.name,
                "seed": seed,
                "ball_id": i,
                "ball_class": float(b.label),
                "purity": float(b.purity),
                "radius": float(b.radius),
                "mu_G": float(mu_g[i]),
                "compA": float(comp_a[i]),
                "n_opp": int(n_opp[i]),
                "R_opp": float(r_opp[i]),
                "ca_weight": float(ca_w[i]),
                "topo_weight": float(topo_w[i]),
                "Topo_minus_CA": float(topo_minus_ca[i]),
                "high_compA": bool(high_comp[i]),
                "low_muG_vs_run_median": bool(low_mu_g[i]),
                "sparse_n_opp": bool(sparse_opp[i]),
                "R_opp_eq_0": bool(r_opp_zero[i]),
                "harm_ball": bool(harm_mask[i]),
            }
        )

    run_row = {
        "dataset": split.name,
        "seed": seed,
        "run_id": f"{split.name}_s{seed}",
        "n_balls": n_balls,
        "n_high_compA": n_high_comp,
        "mu_G_mean": float(np.mean(mu_g)) if mu_g.size else np.nan,
        "mu_G_median": mu_g_median,
        "mu_G_std": float(np.std(mu_g)) if mu_g.size else np.nan,
        "harm_risk": harm_risk,
        "frac_n_opp_risk": frac_n_opp_risk,
        "frac_r_opp_zero_hc": frac_r_opp_zero_hc,
        "frac_low_muG_hc": frac_low_mu_g_hc,
        "mean_n_opp": float(np.mean(n_opp)) if n_opp.size else np.nan,
        "median_n_opp": float(np.median(n_opp)) if n_opp.size else np.nan,
        "frac_R_opp_eq_0": float(np.mean(r_opp_zero)) if r_opp.size else np.nan,
        "R_opp_mean": float(np.mean(r_opp)) if r_opp.size else np.nan,
        "mean_abs_topo_ca": float(np.mean(np.abs(topo_minus_ca))),
        "delta_macro_f1_topo_vs_ca": perf.get("delta_macro_f1_topo_vs_ca"),
        "delta_gmean_topo_vs_ca": perf.get("delta_gmean_topo_vs_ca"),
        "delta_minority_f1_topo_vs_ca": perf.get("delta_minority_f1_topo_vs_ca"),
    }
    return run_row, ball_rows


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    v = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(v) < 3:
        return float("nan")
    return float(v["x"].corr(v["y"]))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(
        ((a.size - 1) * np.var(a) + (b.size - 1) * np.var(b))
        / (a.size + b.size - 2)
    )
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Higher score should predict label=1 (adversarial)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(np.sum(pos))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = float(np.sum(ranks[pos]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _build_report(df: pd.DataFrame, summary: dict) -> str:
    lines = [
        "# Topo harm_risk diagnostic report",
        "",
        "Read-only extension of Gaussian-membership and Topo diagnostics.",
        "Fixed rules only — no tuning, no CV, no model changes.",
        "",
        "## harm_risk definition",
        "",
        "Per run:",
        "",
        "```",
        "μ_G = purity * exp(-(r / median_positive_radius)^2)",
        "harm_risk = frac balls with compA>0.5 AND μ_G<run_median AND (n_opp<=2 OR R_opp==0)",
        "```",
        "",
        f"Runs analyzed: **{summary['n_runs']}**",
        "",
        "## Correlations with Topo outcomes (Pearson r)",
        "",
        "| Signal | Δ Macro F1 | Δ G-mean | Δ minority F1 | mean |Topo−CA| |",
        "|---|---:|---:|---:|---:|",
    ]
    for sig, corrs in summary["outcome_correlations"].items():
        lines.append(
            f"| {sig} | {corrs['delta_macro_f1_topo_vs_ca']:.3f} | "
            f"{corrs['delta_gmean_topo_vs_ca']:.3f} | "
            f"{corrs['delta_minority_f1_topo_vs_ca']:.3f} | "
            f"{corrs['mean_abs_topo_ca']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Group means by R_opp category",
            "",
            "| Category | n | harm_risk | frac_n_opp | frac_R0 hc | frac_low_μG hc | Δ Macro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cat, row in summary["category_means"].items():
        lines.append(
            f"| {cat} | {row['n']} | {row['harm_risk']:.3f} | "
            f"{row['frac_n_opp_risk']:.3f} | {row['frac_r_opp_zero_hc']:.3f} | "
            f"{row['frac_low_muG_hc']:.3f} | {row['delta_macro_f1_topo_vs_ca']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Adversarial vs informative separation (excludes redundant)",
            "",
            "| Signal | mean adversarial | mean informative | Cohen's d | AUROC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sig, row in summary["adv_vs_inf_separation"].items():
        lines.append(
            f"| {sig} | {row['mean_adversarial']:.3f} | "
            f"{row['mean_informative']:.3f} | {row['cohens_d']:.3f} | "
            f"{row['auroc']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Predictive separation ranking",
            "",
            f"Best AUROC (adversarial vs informative): **{summary['best_auroc_signal']}** "
            f"(AUROC={summary['best_auroc_value']:.3f})",
            "",
            "AUROC order (higher = better separation):",
        ]
    )
    for sig, auc in summary["auroc_ranking"]:
        lines.append(f"- {sig}: {auc:.3f}")

    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    lines.extend(["", "## Outputs", "", f"- `{OUT_RUN_CSV.name}`", f"- `{OUT_BALL_CSV.name}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    if not SCREENING_CSV.exists():
        raise FileNotFoundError(f"Missing screening results: {SCREENING_CSV}")

    screening = pd.read_csv(SCREENING_CSV)
    topo_perf = screening[screening["method"] == "Topo"].set_index(["dataset", "seed"])

    descriptors = {d["name"]: d for d in discover_available_datasets()}
    run_rows: list[dict] = []
    ball_rows: list[dict] = []

    for (dataset, seed), perf_row in topo_perf.iterrows():
        desc = descriptors.get(dataset)
        if desc is None:
            continue
        perf = perf_row.to_dict()
        run_row, balls = _analyze_run(desc, int(seed), perf)
        run_rows.append(run_row)
        ball_rows.extend(balls)

    df = pd.DataFrame(run_rows)
    ball_df = pd.DataFrame(ball_rows)

    if MECH_CSV.exists():
        mech = pd.read_csv(MECH_CSV)[["dataset", "seed", "r_opp_category"]]
        df = df.merge(mech, on=["dataset", "seed"], how="left")
    else:
        df["r_opp_category"] = np.nan

    df.to_csv(OUT_RUN_CSV, index=False)
    ball_df.to_csv(OUT_BALL_CSV, index=False)

    signal_cols = {
        "frac_n_opp_risk": "n_opp alone (compA>0.5, n_opp<=2)",
        "frac_r_opp_zero_hc": "R_opp=0 alone (compA>0.5)",
        "frac_low_muG_hc": "μ_G alone (compA>0.5, μ_G<median)",
        "harm_risk": "harm_risk (joint rule)",
        "mean_n_opp": "mean n_opp (run)",
        "frac_R_opp_eq_0": "frac R_opp=0 (all balls)",
    }

    outcome_cols = [
        "delta_macro_f1_topo_vs_ca",
        "delta_gmean_topo_vs_ca",
        "delta_minority_f1_topo_vs_ca",
        "mean_abs_topo_ca",
        "mean_n_opp",
        "frac_R_opp_eq_0",
    ]

    outcome_correlations: dict[str, dict[str, float]] = {}
    for col, label in signal_cols.items():
        outcome_correlations[label] = {
            out: _safe_corr(df[col], df[out]) for out in outcome_cols
        }

    category_means: dict[str, dict] = {}
    for cat in sorted(df["r_opp_category"].dropna().unique()):
        sub = df[df["r_opp_category"] == cat]
        category_means[cat] = {
            "n": int(len(sub)),
            "harm_risk": float(sub["harm_risk"].mean()),
            "frac_n_opp_risk": float(sub["frac_n_opp_risk"].mean()),
            "frac_r_opp_zero_hc": float(sub["frac_r_opp_zero_hc"].mean()),
            "frac_low_muG_hc": float(sub["frac_low_muG_hc"].mean()),
            "delta_macro_f1_topo_vs_ca": float(sub["delta_macro_f1_topo_vs_ca"].mean()),
            "mean_abs_topo_ca": float(sub["mean_abs_topo_ca"].mean()),
            "frac_R_opp_eq_0": float(sub["frac_R_opp_eq_0"].mean()),
            "mean_n_opp": float(sub["mean_n_opp"].mean()),
        }

    adv_inf = df[df["r_opp_category"].isin(["adversarial_r_opp", "informative_r_opp"])]
    adv_mask = adv_inf["r_opp_category"] == "adversarial_r_opp"
    adv_vs_inf: dict[str, dict] = {}
    compare_cols = [
        "frac_n_opp_risk",
        "frac_r_opp_zero_hc",
        "frac_low_muG_hc",
        "harm_risk",
    ]
    for col in compare_cols:
        adv_vals = adv_inf.loc[adv_mask, col].values
        inf_vals = adv_inf.loc[~adv_mask, col].values
        labels = adv_mask.values.astype(int)
        scores = adv_inf[col].values
        adv_vs_inf[col] = {
            "mean_adversarial": float(np.nanmean(adv_vals)),
            "mean_informative": float(np.nanmean(inf_vals)),
            "cohens_d": _cohens_d(adv_vals, inf_vals),
            "auroc": _auroc(scores, labels),
        }

    auroc_ranking = sorted(
        [(signal_cols[c], adv_vs_inf[c]["auroc"]) for c in compare_cols],
        key=lambda x: (x[1] if np.isfinite(x[1]) else -1),
        reverse=True,
    )
    best_sig, best_auc = auroc_ranking[0]

    harm_auc = adv_vs_inf["harm_risk"]["auroc"]
    nopp_auc = adv_vs_inf["frac_n_opp_risk"]["auroc"]
    r0_auc = adv_vs_inf["frac_r_opp_zero_hc"]["auroc"]
    mug_auc = adv_vs_inf["frac_low_muG_hc"]["auroc"]

    if np.isfinite(harm_auc) and harm_auc >= max(
        x for x in [nopp_auc, r0_auc, mug_auc] if np.isfinite(x)
    ):
        interp = (
            "harm_risk separates adversarial from informative runs at least as well as "
            "any single-component signal (AUROC comparison). The joint rule captures "
            "high-competition balls that are both low-μ_G and opposite-support fragile."
        )
    elif np.isfinite(r0_auc) and r0_auc >= harm_auc:
        interp = (
            "R_opp=0 on high-compA balls alone matches or exceeds harm_risk for "
            "adversarial/informative separation; μ_G adds little beyond zero-reciprocity."
        )
    elif np.isfinite(nopp_auc) and nopp_auc >= harm_auc:
        interp = (
            "Sparse n_opp on high-compA balls alone matches or exceeds harm_risk; "
            "opposite-class graph sparsity is the dominant separator."
        )
    else:
        interp = (
            "Single signals and harm_risk show modest separation; no metric cleanly "
            "partitions adversarial from informative runs across all 51 screenings."
        )

    summary = {
        "n_runs": int(len(df)),
        "harm_risk_definition": {
            "mu_G": "purity * exp(-(r/median_positive_radius)^2)",
            "compA_threshold": COMPA_THRESHOLD,
            "n_opp_threshold": N_OPP_THRESHOLD,
            "harm_risk": "frac(compA>0.5 & mu_G<run_median & (n_opp<=2 | R_opp==0))",
        },
        "outcome_correlations": outcome_correlations,
        "category_means": category_means,
        "adv_vs_inf_separation": {
            signal_cols[c]: adv_vs_inf[c] for c in compare_cols
        },
        "auroc_ranking": auroc_ranking,
        "best_auroc_signal": best_sig,
        "best_auroc_value": float(best_auc),
        "interpretation": interp,
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_REPORT_MD.write_text(_build_report(df, summary), encoding="utf-8")

    print(f"Wrote {OUT_RUN_CSV} ({len(df)} runs)")
    print(f"Wrote {OUT_BALL_CSV} ({len(ball_df)} balls)")
    print(f"Wrote {OUT_SUMMARY_JSON}")
    print(f"Wrote {OUT_REPORT_MD}")
    print("AUROC ranking:", [(s, round(a, 3)) for s, a in auroc_ranking])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
