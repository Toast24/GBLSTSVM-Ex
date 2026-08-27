"""
Gaussian ball-membership screening (read-only experiment).

Compares frozen CA membership against literature-grounded Gaussian memberships
built from existing ball statistics (center spread / radius / purity / count).

Models (same granulation, solver, splits as frozen protocol):
  B0  CA-GBLSTSVM           w = CW * muA
  G1  Gaussian + CW         w = CW * mu_G   (primary: purity x compactness)
  G2  Gaussian + Topo        w = CW * mu_G * muTopo

Does NOT modify src/ implementations.

Run from GBLSTSVM root:
  python experiments/gaussian_membership_screen.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.phase8_common import (  # noqa: E402
    DATASETS,
    DEFAULT_C1,
    DEFAULT_C2,
    evaluate,
    load_dataset_split,
    normalize_labels,
)
from src.cbcm import (  # noqa: E402
    compute_competition_membership,
    safe_correlation,
    split_membership_by_class,
)
from src.cbcm_gblstsvm import CAGBLSTSVM  # noqa: E402
from src.class_weights import ball_class_weights, compute_binary_class_weights  # noqa: E402
from src.cw_gblstsvm import solve_cw_gblstsvm_systems  # noqa: E402
from src.gblstsvm import predict_twin_hyperplanes  # noqa: E402
from src.granular_balls import BallInfo, balls_to_class_arrays, generate_granular_balls  # noqa: E402
from src.reliability import compute_radius_scale, normalize_radius  # noqa: E402
from src.topology_membership import compute_topology_membership  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "gaussian_membership"
RESULTS_CSV = OUT_DIR / "gaussian_membership_screen_results.csv"
FORMULATIONS_CSV = OUT_DIR / "gaussian_membership_formulations.csv"
SUMMARY_JSON = OUT_DIR / "gaussian_membership_screen_summary.json"
REPORT_PATH = OUT_DIR / "gaussian_membership_screen_report.md"

SEEDS = [42, 0, 1]
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"
REG_TERM = 1e-4
RADIUS_EPS = 1e-12

# Primary Gaussian used for G1/G2 classifiers (fixed, not dataset-tuned).
PRIMARY_GAUSSIAN = "purity_compact"

# Literature-grounded variants using only BallInfo statistics.
GAUSSIAN_FORMULATIONS: dict[str, str] = {
    "compact": "exp(-r_tilde^2); FSVM-style spread on mean ball radius",
    "purity_compact": "purity * exp(-r_tilde^2); fuzzy granular-ball quality",
    "density": "n/max(n) * exp(-r_tilde^2); GB density with compactness (IGBkNN spirit)",
    "inverse_spread": "exp(-r_tilde); exponential decay in normalized radius",
}


@dataclass
class WeightedGBLSTSVM:
    w1_: np.ndarray
    b1_: float
    w2_: np.ndarray
    b2_: float
    balls_: list[BallInfo]
    membership_: np.ndarray
    weights_pos_: np.ndarray
    weights_neg_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_twin_hyperplanes(
            X, self.w1_, self.b1_, self.w2_, self.b2_, rule=PREDICTION_RULE
        )

    def twin_scores(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        f1 = X @ self.w1_ + self.b1_
        f2 = X @ self.w2_ + self.b2_
        return np.abs(f2) - np.abs(f1)


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "max": float(np.max(v)),
    }


def compute_gaussian_memberships(
    balls: list[BallInfo],
    formulation: str,
) -> np.ndarray:
    """
    Per-ball Gaussian membership from existing ball statistics only.

    r_tilde = r / median(r>0)  (same scale as reliability weighting, frozen).
    """
    if not balls:
        return np.array([], dtype=float)

    radii = np.array([b.radius for b in balls], dtype=float)
    purities = np.array([b.purity for b in balls], dtype=float)
    counts = np.array([b.num_samples for b in balls], dtype=float)
    scale = compute_radius_scale(radii, method="median_positive")
    r_tilde = np.array([normalize_radius(r, scale) for r in radii], dtype=float)

    if formulation == "compact":
        mu = np.exp(-np.square(r_tilde))
    elif formulation == "purity_compact":
        mu = purities * np.exp(-np.square(r_tilde))
    elif formulation == "density":
        safe_r = np.maximum(radii, RADIUS_EPS)
        density = counts / safe_r
        density_norm = density / max(float(np.max(density)), RADIUS_EPS)
        mu = density_norm * np.exp(-np.square(r_tilde))
    elif formulation == "inverse_spread":
        mu = np.exp(-r_tilde)
    else:
        raise ValueError(f"Unknown Gaussian formulation: {formulation!r}")

    return np.clip(mu, 0.0, 1.0)


def fit_weighted_model_from_balls(
    balls: list[BallInfo],
    y_train: np.ndarray,
    mu_ball: np.ndarray,
) -> WeightedGBLSTSVM:
    y = normalize_labels(y_train)
    mu = np.asarray(mu_ball, dtype=float).ravel()
    if mu.shape[0] != len(balls):
        raise ValueError("membership length must match number of granular balls.")

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y)
    kappa_pos_b, kappa_neg_b = ball_class_weights(balls, kappa_pos, kappa_neg)
    mu_pos, mu_neg = split_membership_by_class(balls, mu)
    weights_pos = kappa_pos_b * mu_pos
    weights_neg = kappa_neg_b * mu_neg

    centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(balls)
    w1, b1, w2, b2 = solve_cw_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        DEFAULT_C1,
        DEFAULT_C2,
        weights_pos,
        weights_neg,
        reg_term=REG_TERM,
        weight_placement=WEIGHT_PLACEMENT,
    )
    return WeightedGBLSTSVM(
        w1_=w1,
        b1_=b1,
        w2_=w2,
        b2_=b2,
        balls_=balls,
        membership_=mu,
        weights_pos_=weights_pos,
        weights_neg_=weights_neg,
    )


def _class_f1(y_true, y_pred, label: float) -> float:
    return float(
        f1_score(
            normalize_labels(y_true),
            normalize_labels(y_pred),
            pos_label=label,
            zero_division=0,
        )
    )


def _eval_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    minority_label: float,
) -> dict[str, float | None]:
    m = evaluate(y_true, y_pred, scores, minority_label)
    m["minority_f1"] = _class_f1(y_true, y_pred, minority_label)
    return m


def _prediction_change_rate(y_pred_base: np.ndarray, y_pred_other: np.ndarray) -> float:
    a = normalize_labels(y_pred_base)
    b = normalize_labels(y_pred_other)
    a[a == 0] = -1.0
    b[b == 0] = -1.0
    return float(np.mean(a != b))


def run_single(split, seed: int) -> tuple[list[dict], list[dict], dict]:
    y_train = normalize_labels(split.y_train)
    minority_label = split.minority_label

    np.random.seed(seed)
    balls = generate_granular_balls(
        split.X_train,
        y_train,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )
    centers = np.vstack([b.center for b in balls])
    labels = np.array([b.label for b in balls], dtype=float)

    comp, mu_a, k = compute_competition_membership(centers, labels)
    topo = compute_topology_membership(centers, labels, k=k)

    gaussian_maps = {
        name: compute_gaussian_memberships(balls, name)
        for name in GAUSSIAN_FORMULATIONS
    }
    mu_g = gaussian_maps[PRIMARY_GAUSSIAN]
    mu_g_topo = np.clip(mu_g * topo.mu_topo, 0.0, 1.0)

    formulation_rows: list[dict[str, Any]] = []
    for name, mu_vals in gaussian_maps.items():
        dist = _distribution_summary(mu_vals)
        formulation_rows.append(
            {
                "dataset": split.name,
                "seed": seed,
                "formulation": name,
                "n_balls": len(balls),
                **{f"mu_{k}": v for k, v in dist.items()},
                "corr_with_muA": safe_correlation(mu_vals, mu_a, "pearson"),
                "corr_with_muTopo": safe_correlation(mu_vals, topo.mu_topo, "pearson"),
                "corr_with_comp": safe_correlation(mu_vals, comp, "pearson"),
            }
        )

    # B0: frozen CA implementation
    ca_model = CAGBLSTSVM(
        c1=DEFAULT_C1,
        c2=DEFAULT_C2,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
        prediction_rule=PREDICTION_RULE,
        random_state=seed,
        weight_placement=WEIGHT_PLACEMENT,
    ).fit(split.X_train, y_train)

    g1_model = fit_weighted_model_from_balls(balls, y_train, mu_g)
    g2_model = fit_weighted_model_from_balls(balls, y_train, mu_g_topo)

    models = {
        "B0_CA": ca_model,
        "G1_Gaussian_CW": g1_model,
        "G2_Gaussian_Topo": g2_model,
    }

    result_rows: list[dict[str, Any]] = []
    preds: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        y_pred = model.predict(split.X_test)
        scores = np.abs(split.X_test @ model.w2_ + model.b2_) - np.abs(
            split.X_test @ model.w1_ + model.b1_
        )
        preds[model_name] = y_pred
        metrics = _eval_predictions(
            split.y_test, y_pred, scores, minority_label
        )
        result_rows.append(
            {
                "dataset": split.name,
                "seed": seed,
                "model": model_name,
                "gaussian_formulation": PRIMARY_GAUSSIAN
                if model_name != "B0_CA"
                else "n/a",
                "macro_f1": metrics["macro_f1"],
                "gmean": metrics["gmean"],
                "minority_f1": metrics["minority_f1"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "auroc": metrics.get("auroc"),
            }
        )

    base_pred = preds["B0_CA"]
    diag = {
        "dataset": split.name,
        "seed": seed,
        "n_balls": len(balls),
        "primary_gaussian": PRIMARY_GAUSSIAN,
        "muA_distribution": _distribution_summary(mu_a),
        "muG_distribution": _distribution_summary(mu_g),
        "muTopo_distribution": _distribution_summary(topo.mu_topo),
        "corr_muG_muA": safe_correlation(mu_g, mu_a, "pearson"),
        "corr_muG_muTopo": safe_correlation(mu_g, topo.mu_topo, "pearson"),
        "corr_muA_muTopo": safe_correlation(mu_a, topo.mu_topo, "pearson"),
        "corr_muGTopo_muA": safe_correlation(mu_g_topo, mu_a, "pearson"),
        "pred_change_G1_vs_B0": _prediction_change_rate(base_pred, preds["G1_Gaussian_CW"]),
        "pred_change_G2_vs_B0": _prediction_change_rate(base_pred, preds["G2_Gaussian_Topo"]),
        "pred_change_G2_vs_G1": _prediction_change_rate(
            preds["G1_Gaussian_CW"], preds["G2_Gaussian_Topo"]
        ),
        "formulation_correlations": {
            name: {
                "corr_muA": safe_correlation(vals, mu_a, "pearson"),
                "corr_muTopo": safe_correlation(vals, topo.mu_topo, "pearson"),
            }
            for name, vals in gaussian_maps.items()
        },
    }

    for row in result_rows:
        if row["model"] == "G1_Gaussian_CW":
            row["pred_change_vs_B0"] = diag["pred_change_G1_vs_B0"]
        elif row["model"] == "G2_Gaussian_Topo":
            row["pred_change_vs_B0"] = diag["pred_change_G2_vs_B0"]
            row["pred_change_vs_G1"] = diag["pred_change_G2_vs_G1"]
        else:
            row["pred_change_vs_B0"] = 0.0

    return result_rows, formulation_rows, diag


def aggregate_failures(results_df: pd.DataFrame) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        b0 = sub[sub["model"] == "B0_CA"]
        for model in ("G1_Gaussian_CW", "G2_Gaussian_Topo"):
            g = sub[sub["model"] == model]
            merged = b0.merge(
                g,
                on=["dataset", "seed"],
                suffixes=("_b0", "_g"),
            )
            for _, row in merged.iterrows():
                for metric in ("gmean", "minority_f1", "macro_f1"):
                    delta = float(row[f"{metric}_g"]) - float(row[f"{metric}_b0"])
                    if delta < -0.01:
                        failures.append(
                            {
                                "dataset": dataset,
                                "seed": int(row["seed"]),
                                "model": model,
                                "metric": metric,
                                "b0": float(row[f"{metric}_b0"]),
                                "value": float(row[f"{metric}_g"]),
                                "delta_vs_b0": delta,
                            }
                        )
    return failures


def write_report(
    path: Path,
    results_df: pd.DataFrame,
    formulations_df: pd.DataFrame,
    diagnostics: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    lines = [
        "# Gaussian Membership Screening Report",
        "",
        "Read-only experiment; frozen granulation / CA / Topo / solver unchanged.",
        "",
        f"- Primary Gaussian for G1/G2: **{PRIMARY_GAUSSIAN}** "
        f"({GAUSSIAN_FORMULATIONS[PRIMARY_GAUSSIAN]})",
        f"- Datasets: {', '.join(d.replace('.csv', '') for d in DATASETS)}",
        f"- Seeds: {SEEDS}",
        "",
        "## Model definitions",
        "",
        "| Model | Weights |",
        "|---|---|",
        "| B0 | CW × μA (frozen CA-GBLSTSVM) |",
        f"| G1 | CW × μ_G ({PRIMARY_GAUSSIAN}) |",
        f"| G2 | CW × μ_G × μTopo |",
        "",
        "## Classifier metrics (test set)",
        "",
    ]

    lines.append("See CSV for full table. Mean test metrics by dataset:")
    lines.append("")
    for dataset in sorted(results_df["dataset"].unique()):
        lines.append(f"### {dataset}")
        dsub = results_df[results_df["dataset"] == dataset]
        for model in ("B0_CA", "G1_Gaussian_CW", "G2_Gaussian_Topo"):
            msub = dsub[dsub["model"] == model]
            lines.append(
                f"- **{model}**: "
                f"macro_f1={msub['macro_f1'].mean():.3f}, "
                f"gmean={msub['gmean'].mean():.3f}, "
                f"minority_f1={msub['minority_f1'].mean():.3f}, "
                f"acc={msub['accuracy'].mean():.3f}"
            )
        lines.append("")

    lines.extend(["## Membership vs CA / Topo (primary Gaussian)", ""])
    for d in diagnostics:
        lines.append(
            f"- **{d['dataset']} s{d['seed']}** (n_balls={d['n_balls']}): "
            f"corr(μ_G, μ_A)={d['corr_muG_muA']:.3f}, "
            f"corr(μ_G, μ_Topo)={d['corr_muG_muTopo']:.3f}, "
            f"corr(μ_A, μ_Topo)={d['corr_muA_muTopo']:.3f}; "
            f"pred change G1 vs B0={d['pred_change_G1_vs_B0']:.1%}, "
            f"G2 vs B0={d['pred_change_G2_vs_B0']:.1%}"
        )

    lines.extend(["", "## Formulation correlations with μA", ""])
    for form in GAUSSIAN_FORMULATIONS:
        fsub = formulations_df[formulations_df["formulation"] == form]
        lines.append(
            f"- **{form}**: mean corr(μ, μ_A)="
            f"{fsub['corr_with_muA'].mean():.3f}, "
            f"mean corr(μ, μ_Topo)={fsub['corr_with_muTopo'].mean():.3f}"
        )

    lines.extend(["", "## Failures vs B0 (delta < -0.01)", ""])
    if not failures:
        lines.append("- None under the chosen threshold.")
    else:
        for f in failures[:20]:
            lines.append(
                f"- {f['dataset']} s{f['seed']} {f['model']} {f['metric']}: "
                f"{f['value']:.3f} vs B0 {f['b0']:.3f} (delta={f['delta_vs_b0']:+.3f})"
            )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- **Independent signal**: low |corr(μ_G, μ_A)| and non-trivial pred-change vs B0.",
            "- **Reproduces CA**: high |corr(μ_G, μ_A)| and near-zero pred-change with similar metrics.",
            "- G2 adds Topo beyond Gaussian only if corr(μ_G, μ_Topo) is moderate and G2 ≠ G1.",
            "",
            f"Outputs: `{RESULTS_CSV.name}`, `{FORMULATIONS_CSV.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    all_formulations: list[dict] = []
    all_diagnostics: list[dict] = []

    for dataset_file in DATASETS:
        dataset_name = dataset_file.replace(".csv", "")
        for seed in SEEDS:
            split = load_dataset_split(dataset_name, seed=seed)
            results, formulations, diag = run_single(split, seed)
            all_results.extend(results)
            all_formulations.extend(formulations)
            all_diagnostics.append(diag)
            b0 = next(r for r in results if r["model"] == "B0_CA")
            g1 = next(r for r in results if r["model"] == "G1_Gaussian_CW")
            print(
                f"{dataset_name} s{seed}: "
                f"B0 gmean={b0['gmean']:.3f} | "
                f"G1={g1['gmean']:.3f} corr(G,A)={diag['corr_muG_muA']:.3f} "
                f"d_pred={diag['pred_change_G1_vs_B0']:.1%}"
            )

    results_df = pd.DataFrame(all_results)
    formulations_df = pd.DataFrame(all_formulations)
    failures = aggregate_failures(results_df)

    results_df.to_csv(RESULTS_CSV, index=False)
    formulations_df.to_csv(FORMULATIONS_CSV, index=False)

    payload = {
        "protocol": {
            "purity": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "seeds": SEEDS,
            "datasets": [d.replace(".csv", "") for d in DATASETS],
            "primary_gaussian": PRIMARY_GAUSSIAN,
            "formulations": GAUSSIAN_FORMULATIONS,
        },
        "diagnostics": all_diagnostics,
        "failures_vs_b0": failures,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(
        REPORT_PATH, results_df, formulations_df, all_diagnostics, failures
    )

    print(f"\nWrote {RESULTS_CSV}")
    print(f"Wrote {FORMULATIONS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
