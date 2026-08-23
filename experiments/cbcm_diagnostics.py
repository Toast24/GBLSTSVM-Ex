"""
CBCM mandatory diagnostic gate (run BEFORE classifier training).

Produces cbcm_diagnostics_report.md + histogram plots per dataset.
Checks explicit stop conditions and reports pass/fail per dataset.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase8_common import (  # noqa: E402
    DATASETS,
    DEFAULT_MIN_BALL_COUNT,
    DEFAULT_PURITY,
    load_dataset_split,
)
from src.cbcm import (  # noqa: E402
    bonferroni_correct,
    compute_cbcm_membership,
    compute_naive_density_weight,
    kappa_per_ball_array,
    label_permutation_null_ks,
    percentile_rank,
    resolve_k,
    safe_correlation,
)
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.gblstsvm import GBLSTSVM  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.bmru import ball_bmru_weights  # noqa: E402
from src.reliability import ball_reliability_weights  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "cbcm"
REPORT_PATH = OUT_DIR / "cbcm_diagnostics_report.md"
JSON_PATH = OUT_DIR / "cbcm_diagnostics_report.json"

SEED = 42
PURITY = DEFAULT_PURITY
MIN_BALL_COUNT = DEFAULT_MIN_BALL_COUNT
N_PERM = 200

# Stop-condition thresholds (fixed, not tuned).
KS_ALPHA = 0.05
CW_REDUNDANCY_R = 0.9
AB_REDUNDANCY_R = 0.8
NOISE_PERCENTILE_THRESHOLD = 70.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _summary_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return {}
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
    }


def _within_class_variance(mu_b: np.ndarray, labels: np.ndarray, y: float) -> float:
    mask = labels == y
    vals = mu_b[mask]
    if vals.size <= 1:
        return 0.0
    return float(np.var(vals))


def _synthetic_noise_check(
    X_train: np.ndarray,
    y_train: np.ndarray,
    balls,
    membership_result,
    kappa_pos: float,
    kappa_neg: float,
    minority_label: float,
    rng: np.random.Generator,
) -> dict:
    """Inject isolated minority noise and compare muC vs naive density weighting."""
    n_balls = len(balls)
    m = max(3, int(np.ceil(0.02 * n_balls)))
    labels_ball = np.array([b.label for b in balls], dtype=float)
    minority_mask = labels_ball == minority_label
    minority_idx = np.flatnonzero(minority_mask)

    centers = np.vstack([b.center for b in balls])
    noise_centers: list[np.ndarray] = []

    if minority_idx.size > 0:
        # Farthest minority balls by nearest same-class distance.
        min_same_dist = np.full(minority_idx.size, np.inf)
        for ii, gi in enumerate(minority_idx):
            others = minority_idx[minority_idx != gi]
            if others.size == 0:
                min_same_dist[ii] = np.inf
            else:
                d = np.linalg.norm(centers[gi] - centers[others], axis=1)
                min_same_dist[ii] = float(np.min(d))
        order = np.argsort(-min_same_dist)
        chosen = minority_idx[order[: min(m, minority_idx.size)]]
        noise_centers = [centers[i].copy() for i in chosen]
        method = "farthest_minority_balls"
    else:
        method = "synthetic_majority_region"

    if len(noise_centers) < m:
        majority_mask = labels_ball != minority_label
        maj_idx = np.flatnonzero(majority_mask)
        if maj_idx.size == 0:
            return {"error": "no majority balls for synthetic placement"}
        pick = rng.choice(maj_idx, size=m - len(noise_centers), replace=True)
        jitter_scale = 0.01 * np.std(centers, axis=0, keepdims=True) + 1e-6
        for j in pick:
            noise_centers.append(
                centers[j] + rng.normal(0.0, 1.0, size=centers.shape[1]) * jitter_scale.ravel()
            )

    # Augment training data with noise points labeled minority.
    X_aug = np.vstack([X_train, np.vstack(noise_centers[:m])])
    y_aug = np.concatenate(
        [y_train, np.full(m, minority_label, dtype=float)]
    )

    balls_aug = generate_granular_balls(
        X_aug,
        y_aug,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )
    centers_aug = np.vstack([b.center for b in balls_aug])
    labels_aug = np.array([b.label for b in balls_aug], dtype=float)
    mem_aug = compute_cbcm_membership(centers_aug, labels_aug)

    kappa_aug = kappa_per_ball_array(balls_aug, kappa_pos, kappa_neg)
    naive_w = compute_naive_density_weight(centers_aug, labels_aug, kappa_aug)

    # Identify noise-related balls: minority balls nearest injected noise centers.
    noise_arr = np.vstack(noise_centers[:m])
    minority_ball_idx = [
        i for i, b in enumerate(balls_aug) if b.label == minority_label
    ]
    if not minority_ball_idx:
        return {"error": "no minority balls after augmentation"}

    dist_to_noise = np.min(
        np.linalg.norm(
            centers_aug[minority_ball_idx][:, None, :]
            - noise_arr[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    noise_ball_local = int(np.argmin(dist_to_noise))
    noise_ball_global = minority_ball_idx[noise_ball_local]

    mu_c_noise = float(mem_aug.mu_c[noise_ball_global])
    naive_noise = float(naive_w[noise_ball_global])
    mu_c_pop = mem_aug.mu_c
    naive_pop = naive_w

    return {
        "m_noise_points": m,
        "method": method,
        "noise_ball_mu_c": mu_c_noise,
        "noise_ball_mu_c_percentile": percentile_rank(mu_c_noise, mu_c_pop),
        "noise_ball_naive_density_weight": naive_noise,
        "noise_ball_naive_percentile": percentile_rank(naive_noise, naive_pop),
        "overall_mu_c_summary": _summary_stats(mu_c_pop),
        "overall_naive_summary": _summary_stats(naive_pop),
    }


def _plot_histograms(
    dataset: str,
    mu_a: np.ndarray,
    mu_b: np.ndarray,
    mu_c: np.ndarray,
    out_dir: Path,
) -> list[str]:
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, data, title in zip(
        axes,
        [mu_a, mu_b, mu_c],
        ["muA (competition)", "muB (connectivity)", "muC (combined)"],
    ):
        ax.hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("membership")
        ax.set_ylabel("count")
    fig.suptitle(f"CBCM membership distributions — {dataset}")
    fig.tight_layout()
    path = out_dir / f"{dataset}_membership_histograms.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    return paths


def _check_stop_conditions(ds_result: dict) -> dict:
    checks = {}
    ks_p_adj = ds_result["null_model_a"]["p_value_bonferroni"]
    checks["compA_indistinguishable_from_null"] = {
        "triggered": ks_p_adj > KS_ALPHA,
        "p_value_bonferroni": ks_p_adj,
        "threshold": KS_ALPHA,
    }
    pearson_b_cw = abs(ds_result["redundancy_b_vs_cw"]["pearson_r"])
    checks["muB_redundant_with_CW"] = {
        "triggered": pearson_b_cw >= CW_REDUNDANCY_R
        or ds_result["redundancy_b_vs_cw"]["within_class_variance_minority"] < 1e-12,
        "pearson_r": ds_result["redundancy_b_vs_cw"]["pearson_r"],
        "within_class_var_pos": ds_result["redundancy_b_vs_cw"]["within_class_variance_pos"],
        "within_class_var_neg": ds_result["redundancy_b_vs_cw"]["within_class_variance_neg"],
        "threshold_r": CW_REDUNDANCY_R,
    }
    pearson_ab = abs(ds_result["redundancy_a_vs_b"]["pearson_r"])
    checks["muA_muB_highly_correlated"] = {
        "triggered": pearson_ab >= AB_REDUNDANCY_R,
        "pearson_r": ds_result["redundancy_a_vs_b"]["pearson_r"],
        "threshold_r": AB_REDUNDANCY_R,
    }
    noise_pct = ds_result["synthetic_noise"].get("noise_ball_mu_c_percentile", float("nan"))
    checks["synthetic_noise_rewarded"] = {
        "triggered": noise_pct > NOISE_PERCENTILE_THRESHOLD,
        "mu_c_percentile": noise_pct,
        "threshold_percentile": NOISE_PERCENTILE_THRESHOLD,
    }
    checks["dataset_gate_pass"] = not any(
        c["triggered"] for c in checks.values() if isinstance(c, dict) and "triggered" in c
    )
    return checks


def run_dataset_diagnostics(dataset_name: str) -> dict:
    split = load_dataset_split(dataset_name, seed=SEED)
    ds = split.name
    X_train = split.X_train
    y_train = split.y_train

    balls = generate_granular_balls(
        X_train,
        y_train,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )
    centers = np.vstack([b.center for b in balls])
    labels = np.array([b.label for b in balls], dtype=float)
    n_balls = len(balls)
    k = resolve_k(n_balls)
    logger.info("Dataset %s: N=%d balls, k=%d", ds, n_balls, k)

    membership = compute_cbcm_membership(centers, labels, k=k)
    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    kappa_ball = kappa_per_ball_array(balls, kappa_pos, kappa_neg)

    null_a = label_permutation_null_ks(
        centers, labels, n_permutations=N_PERM, k=k, random_state=SEED
    )

    minority_label = split.minority_label
    minority_mask = labels == minority_label

    redundancy_b_cw = {
        "pearson_r": safe_correlation(membership.mu_b, kappa_ball, "pearson"),
        "spearman_r": safe_correlation(membership.mu_b, kappa_ball, "spearman"),
        "within_class_variance_pos": _within_class_variance(membership.mu_b, labels, 1.0),
        "within_class_variance_neg": _within_class_variance(membership.mu_b, labels, -1.0),
        "within_class_variance_minority": _within_class_variance(
            membership.mu_b, labels, minority_label
        ),
    }

    redundancy_ab = {
        "pearson_r_all": safe_correlation(membership.mu_a, membership.mu_b, "pearson"),
        "spearman_r_all": safe_correlation(membership.mu_a, membership.mu_b, "spearman"),
        "pearson_r": safe_correlation(membership.mu_a, membership.mu_b, "pearson"),
        "spearman_r": safe_correlation(membership.mu_a, membership.mu_b, "spearman"),
    }
    if np.any(minority_mask):
        redundancy_ab["pearson_r_minority_only"] = safe_correlation(
            membership.mu_a[minority_mask],
            membership.mu_b[minority_mask],
            "pearson",
        )
        redundancy_ab["spearman_r_minority_only"] = safe_correlation(
            membership.mu_a[minority_mask],
            membership.mu_b[minority_mask],
            "spearman",
        )

    # R and BMRU context correlations.
    rel_pos, rel_neg, _, reliabilities = ball_reliability_weights(balls)
    rel_all = np.zeros(n_balls)
    pi = ni = 0
    for i, b in enumerate(balls):
        if b.label == 1:
            rel_all[i] = rel_pos[pi]
            pi += 1
        else:
            rel_all[i] = rel_neg[ni]
            ni += 1

    bmru_context = {"available": False}
    try:
        base = GBLSTSVM(
            c1=100.0,
            c2=1e-2,
            purity_threshold=PURITY,
            min_ball_count=MIN_BALL_COUNT,
            random_state=SEED,
        ).fit(X_train, y_train)
        _, _, u_all, mu_bmru, _ = ball_bmru_weights(
            balls, base.w1_, base.b1_, base.w2_, base.b2_
        )
        bmru_context = {
            "available": True,
            "muA_vs_R_pearson": safe_correlation(membership.mu_a, rel_all, "pearson"),
            "muA_vs_R_spearman": safe_correlation(membership.mu_a, rel_all, "spearman"),
            "muA_vs_BMRU_mu_pearson": safe_correlation(membership.mu_a, mu_bmru, "pearson"),
            "muA_vs_BMRU_mu_spearman": safe_correlation(membership.mu_a, mu_bmru, "spearman"),
            "muA_vs_BMRU_u_pearson": safe_correlation(membership.mu_a, u_all, "pearson"),
        }
    except Exception as exc:  # noqa: BLE001
        bmru_context = {"available": False, "error": str(exc)}

    rng = np.random.default_rng(SEED)
    synthetic = _synthetic_noise_check(
        X_train,
        y_train,
        balls,
        membership,
        kappa_pos,
        kappa_neg,
        minority_label,
        rng,
    )

    plot_paths = _plot_histograms(
        ds, membership.mu_a, membership.mu_b, membership.mu_c, OUT_DIR
    )

    result = {
        "dataset": ds,
        "n_balls": n_balls,
        "k": k,
        "membership_summary": {
            "compA": _summary_stats(membership.comp_a),
            "muA": _summary_stats(membership.mu_a),
            "muB": _summary_stats(membership.mu_b),
            "muC": _summary_stats(membership.mu_c),
        },
        "null_model_a": null_a,
        "redundancy_b_vs_cw": redundancy_b_cw,
        "redundancy_a_vs_b": redundancy_ab,
        "redundancy_a_vs_r_bmru": bmru_context,
        "synthetic_noise": synthetic,
        "plot_paths": plot_paths,
        "connectivity_meta": membership.connectivity_meta,
    }
    result["stop_conditions"] = None  # filled after Bonferroni correction in main()
    return result


def _write_markdown_report(all_results: list[dict], global_pass: bool) -> None:
    lines = [
        "# CBCM Diagnostic Gate Report",
        "",
        f"**Global gate:** {'PASS' if global_pass else 'FAIL — do not proceed to training without review'}",
        "",
        "## Configuration",
        f"- k formula: max(5, ceil(0.05 * N))",
        f"- Permutation null reps: {N_PERM}",
        f"- Bonferroni alpha: {KS_ALPHA}",
        f"- Granulation: purity={PURITY}, min_ball_count={MIN_BALL_COUNT}, seed={SEED}",
        "",
    ]

    for res in all_results:
        ds = res["dataset"]
        gate = res["stop_conditions"]["dataset_gate_pass"]
        lines.extend(
            [
                f"## {ds}",
                "",
                f"**Gate:** {'PASS' if gate else 'FAIL'}",
                f"- N balls: {res['n_balls']}, k={res['k']}",
                "",
                "### Membership summaries",
            ]
        )
        for key in ("compA", "muA", "muB", "muC"):
            s = res["membership_summary"][key]
            lines.append(
                f"- **{key}**: mean={s.get('mean', 'n/a'):.4f}, "
                f"median={s.get('median', 'n/a'):.4f}, "
                f"std={s.get('std', 'n/a'):.4f}"
            )
        lines.append("")
        null = res["null_model_a"]
        lines.extend(
            [
                "### Null-model check (Component A)",
                f"- KS statistic: {null['ks_statistic']:.4f}",
                f"- p-value (raw): {null['p_value']:.6f}",
                f"- p-value (Bonferroni): {null.get('p_value_bonferroni', 'n/a')}",
                f"- Minority class KS p: {null['per_class']['minority']['p_value']:.6f}",
                f"- Majority class KS p: {null['per_class']['majority']['p_value']:.6f}",
                "",
                "### Redundancy: muB vs CW",
                f"- Pearson r: {res['redundancy_b_vs_cw']['pearson_r']:.4f}",
                f"- Spearman r: {res['redundancy_b_vs_cw']['spearman_r']:.4f}",
                f"- Within-class var (+1): {res['redundancy_b_vs_cw']['within_class_variance_pos']:.6f}",
                f"- Within-class var (-1): {res['redundancy_b_vs_cw']['within_class_variance_neg']:.6f}",
                "",
                "### Redundancy: muA vs muB",
                f"- Pearson r (all): {res['redundancy_a_vs_b']['pearson_r_all']:.4f}",
                f"- Pearson r (minority only): {res['redundancy_a_vs_b'].get('pearson_r_minority_only', float('nan')):.4f}",
                "",
            ]
        )
        rb = res["redundancy_a_vs_r_bmru"]
        if rb.get("available"):
            lines.extend(
                [
                    "### Context: muA vs R / BMRU",
                    f"- muA vs R Pearson: {rb['muA_vs_R_pearson']:.4f}",
                    f"- muA vs BMRU mu Pearson: {rb['muA_vs_BMRU_mu_pearson']:.4f}",
                    "",
                ]
            )
        syn = res["synthetic_noise"]
        lines.extend(
            [
                "### Synthetic noise sanity",
                f"- Method: {syn.get('method', 'n/a')}",
                f"- Noise ball muC percentile: {syn.get('noise_ball_mu_c_percentile', float('nan')):.1f}",
                f"- Noise ball naive-density percentile: {syn.get('noise_ball_naive_percentile', float('nan')):.1f}",
                "",
                "### Stop conditions",
            ]
        )
        for name, check in res["stop_conditions"].items():
            if name == "dataset_gate_pass":
                continue
            triggered = check.get("triggered", False)
            lines.append(f"- **{name}**: {'TRIGGERED' if triggered else 'ok'}")
        lines.append("")
        for p in res.get("plot_paths", []):
            rel = Path(p).name
            lines.append(f"![{ds} histograms]({rel})")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    raw_p_values: list[float] = []
    for ds_file in DATASETS:
        res = run_dataset_diagnostics(ds_file)
        raw_p_values.append(res["null_model_a"]["p_value"])
        all_results.append(res)

    corrected = bonferroni_correct(raw_p_values)
    for res, p_adj in zip(all_results, corrected):
        res["null_model_a"]["p_value_bonferroni"] = p_adj
        res["stop_conditions"] = _check_stop_conditions(res)

    global_pass = all(r["stop_conditions"]["dataset_gate_pass"] for r in all_results)

    payload = {
        "global_gate_pass": global_pass,
        "datasets": {r["dataset"]: r for r in all_results},
        "config": {
            "seed": SEED,
            "purity": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "n_permutations": N_PERM,
            "stop_thresholds": {
                "ks_alpha_bonferroni": KS_ALPHA,
                "cw_redundancy_r": CW_REDUNDANCY_R,
                "ab_redundancy_r": AB_REDUNDANCY_R,
                "noise_percentile": NOISE_PERCENTILE_THRESHOLD,
            },
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_markdown_report(all_results, global_pass)

    print(f"\nReport written to {REPORT_PATH}")
    print(f"JSON written to {JSON_PATH}")
    print(f"\nGlobal diagnostic gate: {'PASS' if global_pass else 'FAIL'}")
    for res in all_results:
        status = "PASS" if res["stop_conditions"]["dataset_gate_pass"] else "FAIL"
        print(f"  {res['dataset']}: {status} (k={res['k']}, N={res['n_balls']})")

    return 0 if global_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
