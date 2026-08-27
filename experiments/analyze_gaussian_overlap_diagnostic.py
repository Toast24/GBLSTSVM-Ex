"""
Gaussian-aware split diagnostic (read-only).

For each final granular ball, estimates class-conditional Gaussian overlap
from in-ball samples and compares against purity, heterogeneous distance,
KMeans split replay, radius, and Topo n_opp / R_opp.

Does NOT modify granulation, Topo, or model code.

Run from GBLSTSVM root:
  python experiments/analyze_gaussian_overlap_diagnostic.py
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
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.granular_balls import GranularBall, generate_granular_balls_with_membership  # noqa: E402
from src.topology_membership import ball_topo_weights  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granulation"
TOPO_DIR = Path(__file__).resolve().parent / "output" / "topology"
SCREENING_CSV = TOPO_DIR / "topo_broad_screening_results.csv"
MECH_CSV = TOPO_DIR / "r_opp_mechanism_analysis.csv"
OUT_BALL_CSV = OUT_DIR / "gaussian_overlap_per_ball.csv"
OUT_RUN_CSV = OUT_DIR / "gaussian_overlap_runs.csv"
OUT_SUMMARY_JSON = OUT_DIR / "gaussian_overlap_summary.json"
OUT_REPORT_MD = OUT_DIR / "gaussian_overlap_report.md"

VAR_REG = 1e-6
HD_EPS = 1e-12


def bhattacharyya_coefficient_diag(
    x_pos: np.ndarray,
    x_neg: np.ndarray,
    reg: float = VAR_REG,
) -> float:
    """
    Bhattacharyya coefficient BC = exp(-B) for diagonal-covariance Gaussians
    fit to in-ball class samples (features only).
    """
    x_pos = np.asarray(x_pos, dtype=float)
    x_neg = np.asarray(x_neg, dtype=float)
    if x_pos.size == 0 or x_neg.size == 0:
        return float("nan")

    mu1 = np.mean(x_pos, axis=0)
    mu2 = np.mean(x_neg, axis=0)
    if x_pos.shape[0] > 1:
        v1 = np.var(x_pos, axis=0, ddof=1)
    else:
        v1 = np.full(x_pos.shape[1], reg)
    if x_neg.shape[0] > 1:
        v2 = np.var(x_neg, axis=0, ddof=1)
    else:
        v2 = np.full(x_neg.shape[1], reg)
    v1 = np.maximum(v1, reg)
    v2 = np.maximum(v2, reg)

    term1 = 0.25 * np.sum(np.log((v1 + v2) / (2.0 * np.sqrt(v1 * v2))))
    term2 = 0.25 * np.sum((mu1 - mu2) ** 2 / (v1 + v2))
    B = float(term1 + term2)
    return float(np.exp(-B))


def minority_hd1_stats(
    features: np.ndarray,
    labels: np.ndarray,
    majority_label: float,
    radius: float,
) -> tuple[float, float, float]:
    """Median/min hd1 for minority points; hd1 normalized by ball radius."""
    labels = np.asarray(labels, dtype=float)
    maj_mask = labels == majority_label
    min_mask = ~maj_mask
    if not np.any(min_mask) or not np.any(maj_mask):
        return float("nan"), float("nan"), float("nan")

    min_pts = features[min_mask]
    maj_pts = features[maj_mask]
    dists = np.linalg.norm(min_pts[:, None, :] - maj_pts[None, :, :], axis=2)
    hd1_each = np.min(dists, axis=1)
    med = float(np.median(hd1_each))
    mn = float(np.min(hd1_each))
    scale = max(float(radius), HD_EPS)
    return med, mn, med / scale


def replay_kmeans_split(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Replay current GranularBall.split_2balls() KMeans(k=2) on in-ball samples."""
    n = features.shape[0]
    if n <= 1:
        return {
            "kmeans_max_child_purity": float("nan"),
            "kmeans_purity_gain": float("nan"),
            "kmeans_both_clusters_mixed": float("nan"),
            "kmeans_minority_isolated": float("nan"),
        }

    data = np.column_stack([features, labels])
    ball = GranularBall(data)
    c1, c2 = ball.split_2balls()
    p1, p2 = float(c1.purity), float(c2.purity)
    max_child = max(p1, p2)
    parent_purity = float(ball.purity)
    gain = max_child - parent_purity

    both_mixed = float(p1 < 1.0 and p2 < 1.0)
    sizes = sorted([c1.num, c2.num])
    smaller_purity = min(p1, p2)
    minority_isolated = float(sizes[0] == 1 and smaller_purity == 1.0 and gain > 0)

    return {
        "kmeans_max_child_purity": max_child,
        "kmeans_purity_gain": gain,
        "kmeans_both_clusters_mixed": both_mixed,
        "kmeans_minority_isolated": minority_isolated,
    }


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    v = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(v) < 3:
        return float("nan")
    return float(v["x"].corr(v["y"]))


def _partial_corr(df: pd.DataFrame, target: str, predictor: str, controls: list[str]) -> float:
    cols = [target, predictor] + controls
    sub = df[cols].dropna()
    if len(sub) < len(cols) + 2:
        return float("nan")

    def resid(a: str, b_list: list[str]) -> np.ndarray:
        y = sub[a].values
        if not b_list:
            return y - np.mean(y)
        X = sub[b_list].values
        beta, _, _, _ = np.linalg.lstsq(
            np.column_stack([np.ones(len(y)), X]), y, rcond=None
        )
        return y - (beta[0] + X @ beta[1:])

    ry = resid(target, controls)
    rp = resid(predictor, controls)
    if np.std(ry) < 1e-15 or np.std(rp) < 1e-15:
        return float("nan")
    return float(np.corrcoef(ry, rp)[0, 1])


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
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


def analyze_run(descriptor: dict, seed: int, perf: dict) -> tuple[list[dict], dict]:
    split, err = try_load_descriptor(descriptor, seed=seed)
    if split is None:
        raise RuntimeError(err)

    X_train = split.X_train
    y_train = normalize_labels(split.y_train)
    np.random.seed(seed)

    balls, memberships = generate_granular_balls_with_membership(
        X_train,
        y_train,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    _, _, topo_mem, _, _ = ball_topo_weights(balls, kappa_pos, kappa_neg)

    ball_rows: list[dict] = []
    for i, (ball, idx) in enumerate(zip(balls, memberships)):
        feats = X_train[idx]
        labels = y_train[idx]
        maj_label = ball.label
        n_pos = int(np.sum(labels == 1.0))
        n_neg = int(np.sum(labels == -1.0))
        heterogeneous = n_pos > 0 and n_neg > 0

        x_pos = feats[labels == 1.0]
        x_neg = feats[labels == -1.0]

        bc = (
            bhattacharyya_coefficient_diag(x_pos, x_neg)
            if heterogeneous
            else float("nan")
        )
        med_hd1, min_hd1, hd1_norm = (
            minority_hd1_stats(feats, labels, maj_label, ball.radius)
            if heterogeneous
            else (float("nan"), float("nan"), float("nan"))
        )
        km = (
            replay_kmeans_split(feats, labels)
            if heterogeneous
            else {
                "kmeans_max_child_purity": float("nan"),
                "kmeans_purity_gain": float("nan"),
                "kmeans_both_clusters_mixed": float("nan"),
                "kmeans_minority_isolated": float("nan"),
            }
        )

        ball_rows.append(
            {
                "dataset": split.name,
                "seed": seed,
                "ball_id": i,
                "ball_label": float(ball.label),
                "num_samples": int(ball.num_samples),
                "purity": float(ball.purity),
                "radius": float(ball.radius),
                "heterogeneous": heterogeneous,
                "n_class_pos": n_pos,
                "n_class_neg": n_neg,
                "BC_overlap": bc,
                "minority_hd1_median": med_hd1,
                "minority_hd1_min": min_hd1,
                "minority_hd1_norm": hd1_norm,
                **km,
                "n_opp": int(topo_mem.n_opp[i]),
                "R_opp": float(topo_mem.r_opp[i]),
                "compA": float(topo_mem.ambiguity[i]),
            }
        )

    het = [r for r in ball_rows if r["heterogeneous"]]
    n_balls = len(ball_rows)
    n_het = len(het)

    def _mean(key: str) -> float:
        vals = [r[key] for r in het if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    bc_vals = [r["BC_overlap"] for r in het if np.isfinite(r["BC_overlap"])]

    run_row = {
        "dataset": split.name,
        "seed": seed,
        "run_id": f"{split.name}_s{seed}",
        "n_balls": n_balls,
        "n_heterogeneous_balls": n_het,
        "frac_heterogeneous": n_het / n_balls if n_balls else float("nan"),
        "mean_BC_overlap_het": _mean("BC_overlap"),
        "median_BC_overlap_het": float(np.median(bc_vals)) if bc_vals else float("nan"),
        "mean_minority_hd1_norm_het": _mean("minority_hd1_norm"),
        "mean_minority_hd1_median_het": _mean("minority_hd1_median"),
        "mean_purity_het": _mean("purity"),
        "mean_radius_het": _mean("radius"),
        "mean_kmeans_purity_gain_het": _mean("kmeans_purity_gain"),
        "frac_kmeans_minority_isolated_het": float(
            np.mean([r["kmeans_minority_isolated"] for r in het])
        )
        if het
        else float("nan"),
        "frac_kmeans_both_mixed_het": float(
            np.mean([r["kmeans_both_clusters_mixed"] for r in het])
        )
        if het
        else float("nan"),
        "mean_n_opp": float(np.mean(topo_mem.n_opp)),
        "mean_R_opp": float(np.mean(topo_mem.r_opp)),
        "frac_R_opp_eq_0": float(np.mean(topo_mem.r_opp == 0)),
        "delta_macro_f1_topo_vs_ca": perf.get("delta_macro_f1_topo_vs_ca"),
        "delta_gmean_topo_vs_ca": perf.get("delta_gmean_topo_vs_ca"),
        "delta_minority_f1_topo_vs_ca": perf.get("delta_minority_f1_topo_vs_ca"),
    }
    return ball_rows, run_row


def _build_report(summary: dict) -> str:
    lines = [
        "# Gaussian overlap split diagnostic",
        "",
        "Read-only diagnostic on frozen granular balls (51 Topo screening runs).",
        "No granulation or model code modified.",
        "",
        "## Proposed statistic",
        "",
        "Per heterogeneous ball (both classes present), fit diagonal Gaussians",
        "N(mu_c, diag(sigma_c^2)) to in-ball samples per class (variance floor 1e-6).",
        "",
        "**BC_overlap** = Bhattacharyya coefficient",
        "",
        "```",
        "BC = exp(-B),  B = 1/4 sum_j log((s1j^2+s2j^2)/(2*s1j*s2j))",
        "              + 1/4 sum_j (mu1j-mu2j)^2/(s1j^2+s2j^2)",
        "```",
        "",
        "Structural interpretation (no outcome-tuned thresholds):",
        "- High BC_overlap: class Gaussians overlap - genuine mixed structure.",
        "- Low BC_overlap + high minority_hd1_norm: compact majority + isolated minority.",
        "",
        "Companion signals on the same balls:",
        "- minority_hd1_norm = median minority-point hd1 / ball radius",
        "- kmeans_purity_gain = replay of current KMeans(k=2) split",
        "- purity, radius, n_opp, R_opp from frozen Topo membership",
        "",
        f"Runs: **{summary['n_runs']}** | heterogeneous balls: **{summary['n_heterogeneous_balls']}**",
        "",
        "## Informative vs adversarial means (R_opp category)",
        "",
        "| Category | n | BC_overlap | hd1_norm | purity(het) | kmeans_gain | frac isolated | Delta Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cat, row in summary["category_means"].items():
        lines.append(
            f"| {cat} | {row['n']} | {row['mean_BC_overlap_het']:.3f} | "
            f"{row['mean_minority_hd1_norm_het']:.3f} | {row['mean_purity_het']:.3f} | "
            f"{row['mean_kmeans_purity_gain_het']:.3f} | "
            f"{row['frac_kmeans_minority_isolated_het']:.3f} | "
            f"{row['delta_macro_f1_topo_vs_ca']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Correlation with Topo performance (Pearson r)",
            "",
            "| Signal | Delta Macro F1 | Delta G-mean | Delta minority F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for sig, corrs in summary["outcome_correlations"].items():
        lines.append(
            f"| {sig} | {corrs['delta_macro_f1_topo_vs_ca']:.3f} | "
            f"{corrs['delta_gmean_topo_vs_ca']:.3f} | "
            f"{corrs['delta_minority_f1_topo_vs_ca']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Incremental information beyond purity + hd1",
            "",
            f"Partial corr(BC_overlap, Delta Macro F1 | purity, hd1_norm): "
            f"**{summary['partial_corr_BC_given_purity_hd1']:.3f}**",
            f"Partial corr(hd1_norm, Delta Macro F1 | purity, BC_overlap): "
            f"**{summary['partial_corr_hd1_given_purity_BC']:.3f}**",
            "",
            "| Predictor set | |r| with Delta Macro F1 |",
            "|---|---:|",
        ]
    )
    for name, rval in summary["macro_f1_correlation_comparison"].items():
        lines.append(f"| {name} | {rval:.3f} |")

    lines.extend(
        [
            "",
            "## Adversarial vs informative AUROC (higher = better separation)",
            "",
            "| Signal | AUROC |",
            "|---|---:|",
        ]
    )
    for sig, auc in summary["adv_inf_auroc"].items():
        lines.append(f"| {sig} | {auc:.3f} |")

    lines.extend(["", "## Verdict", "", summary["verdict"]])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{OUT_RUN_CSV.name}`",
            f"- `{OUT_BALL_CSV.name}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    screening = pd.read_csv(SCREENING_CSV)
    topo_perf = screening[screening["method"] == "Topo"].set_index(["dataset", "seed"])
    descriptors = {d["name"]: d for d in discover_available_datasets()}

    all_balls: list[dict] = []
    run_rows: list[dict] = []

    for (dataset, seed), perf_row in topo_perf.iterrows():
        desc = descriptors.get(dataset)
        if desc is None:
            continue
        balls, run_row = analyze_run(desc, int(seed), perf_row.to_dict())
        all_balls.extend(balls)
        run_rows.append(run_row)

    ball_df = pd.DataFrame(all_balls)
    run_df = pd.DataFrame(run_rows)

    if MECH_CSV.exists():
        mech = pd.read_csv(MECH_CSV)[["dataset", "seed", "r_opp_category"]]
        run_df = run_df.merge(mech, on=["dataset", "seed"], how="left")

    ball_df.to_csv(OUT_BALL_CSV, index=False)
    run_df.to_csv(OUT_RUN_CSV, index=False)

    outcomes = [
        "delta_macro_f1_topo_vs_ca",
        "delta_gmean_topo_vs_ca",
        "delta_minority_f1_topo_vs_ca",
    ]
    signal_cols = {
        "mean_BC_overlap_het": "BC_overlap (het mean)",
        "mean_minority_hd1_norm_het": "minority_hd1_norm (het mean)",
        "mean_purity_het": "purity (het mean)",
        "mean_radius_het": "radius (het mean)",
        "mean_kmeans_purity_gain_het": "kmeans_purity_gain (het mean)",
        "frac_kmeans_minority_isolated_het": "frac kmeans minority isolated",
        "mean_n_opp": "mean n_opp",
        "frac_R_opp_eq_0": "frac R_opp=0",
        "mean_R_opp": "mean R_opp",
    }

    outcome_correlations: dict[str, dict[str, float]] = {}
    for col, label in signal_cols.items():
        outcome_correlations[label] = {
            out: _safe_corr(run_df[col], run_df[out]) for out in outcomes
        }

    partial_bc = _partial_corr(
        run_df,
        "delta_macro_f1_topo_vs_ca",
        "mean_BC_overlap_het",
        ["mean_purity_het", "mean_minority_hd1_norm_het"],
    )
    partial_hd1 = _partial_corr(
        run_df,
        "delta_macro_f1_topo_vs_ca",
        "mean_minority_hd1_norm_het",
        ["mean_purity_het", "mean_BC_overlap_het"],
    )

    r_purity = abs(_safe_corr(run_df["mean_purity_het"], run_df["delta_macro_f1_topo_vs_ca"]))
    r_hd1 = abs(
        _safe_corr(run_df["mean_minority_hd1_norm_het"], run_df["delta_macro_f1_topo_vs_ca"])
    )
    r_bc = abs(_safe_corr(run_df["mean_BC_overlap_het"], run_df["delta_macro_f1_topo_vs_ca"]))
    r_both = abs(
        _safe_corr(
            run_df["mean_purity_het"] + run_df["mean_minority_hd1_norm_het"],
            run_df["delta_macro_f1_topo_vs_ca"],
        )
    )

    macro_f1_correlation_comparison = {
        "purity alone": r_purity,
        "hd1_norm alone": r_hd1,
        "BC_overlap alone": r_bc,
        "purity + hd1_norm (sum)": r_both,
        "purity + hd1 + BC (sum)": abs(
            _safe_corr(
                run_df["mean_purity_het"]
                + run_df["mean_minority_hd1_norm_het"]
                + run_df["mean_BC_overlap_het"],
                run_df["delta_macro_f1_topo_vs_ca"],
            )
        ),
    }

    category_means: dict[str, dict] = {}
    for cat in sorted(run_df["r_opp_category"].dropna().unique()):
        sub = run_df[run_df["r_opp_category"] == cat]
        category_means[cat] = {
            "n": int(len(sub)),
            "mean_BC_overlap_het": float(sub["mean_BC_overlap_het"].mean()),
            "mean_minority_hd1_norm_het": float(sub["mean_minority_hd1_norm_het"].mean()),
            "mean_purity_het": float(sub["mean_purity_het"].mean()),
            "mean_kmeans_purity_gain_het": float(sub["mean_kmeans_purity_gain_het"].mean()),
            "frac_kmeans_minority_isolated_het": float(
                sub["frac_kmeans_minority_isolated_het"].mean()
            ),
            "delta_macro_f1_topo_vs_ca": float(sub["delta_macro_f1_topo_vs_ca"].mean()),
        }

    adv_inf = run_df[run_df["r_opp_category"].isin(["adversarial_r_opp", "informative_r_opp"])]
    adv_mask = adv_inf["r_opp_category"] == "adversarial_r_opp"
    adv_inf_auroc: dict[str, float] = {}
    for col, label in [
        ("mean_BC_overlap_het", "BC_overlap"),
        ("mean_minority_hd1_norm_het", "minority_hd1_norm"),
        ("mean_purity_het", "purity"),
        ("frac_R_opp_eq_0", "frac R_opp=0"),
        ("mean_n_opp", "mean n_opp"),
    ]:
        adv_inf_auroc[label] = _auroc(adv_inf[col].values, adv_mask.values.astype(int))

    bc_adds = (
        macro_f1_correlation_comparison["purity + hd1 + BC (sum)"]
        > macro_f1_correlation_comparison["purity + hd1_norm (sum)"] + 0.02
    )
    partial_significant = np.isfinite(partial_bc) and abs(partial_bc) > 0.15

    if bc_adds and partial_significant:
        verdict = (
            "BC_overlap adds measurable incremental association with Topo outcomes beyond "
            "purity and heterogeneous distance. A Gaussian-aware split experiment is "
            "justified as a read-only granulation replay (not yet implemented)."
        )
    elif r_bc > max(r_purity, r_hd1) + 0.05:
        verdict = (
            "BC_overlap correlates more strongly with Topo Macro F1 than purity or hd1 alone, "
            "but incremental partial correlation is weak. A cautious Gaussian-aware split "
            "replay experiment may be informative on heterogeneous balls only."
        )
    else:
        verdict = (
            "BC_overlap does not materially outperform purity + minority_hd1_norm for "
            "predicting Topo success/failure or separating adversarial from informative runs. "
            "A Gaussian-aware split experiment is not strongly justified by these diagnostics."
        )

    summary = {
        "n_runs": int(len(run_df)),
        "n_heterogeneous_balls": int(ball_df["heterogeneous"].sum()),
        "statistic_definition": {
            "BC_overlap": "Bhattacharyya coefficient of diagonal Gaussians per class in ball",
            "minority_hd1_norm": "median minority hd1 / ball radius within ball",
            "kmeans_purity_gain": "max child purity - parent purity under frozen KMeans(k=2)",
        },
        "category_means": category_means,
        "outcome_correlations": outcome_correlations,
        "partial_corr_BC_given_purity_hd1": partial_bc,
        "partial_corr_hd1_given_purity_BC": partial_hd1,
        "macro_f1_correlation_comparison": macro_f1_correlation_comparison,
        "adv_inf_auroc": adv_inf_auroc,
        "verdict": verdict,
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_REPORT_MD.write_text(_build_report(summary), encoding="utf-8")

    print(f"Wrote {OUT_RUN_CSV} ({len(run_df)} runs)")
    print(f"Wrote {OUT_BALL_CSV} ({len(ball_df)} balls)")
    print(f"Wrote {OUT_SUMMARY_JSON}")
    print(f"Wrote {OUT_REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
