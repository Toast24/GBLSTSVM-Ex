"""
Broad Topo screening: GBLSTSVM, CW, CA, Topo across all available datasets.

Frozen 70/15/15 split, seeds {42, 0, 1}, no hyperparameter tuning.

Run from GBLSTSVM root:
  python experiments/topo_broad_screening.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_topology import (  # noqa: E402
    MIN_BALL_COUNT,
    PURITY,
    _synthetic_noise_topo,
)
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import (  # noqa: E402
    DEFAULT_C1,
    DEFAULT_C2,
    evaluate,
    normalize_labels,
    twin_scores,
)
from src.cbcm import kappa_per_ball_array, safe_correlation  # noqa: E402
from src.cbcm_gblstsvm import CAGBLSTSVM  # noqa: E402
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.cw_gblstsvm import CWGBLSTSVM  # noqa: E402
from src.gblstsvm import GBLSTSVM  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.topology_gblstsvm import TopoGBLSTSVM  # noqa: E402
from src.topology_membership import compute_topology_membership  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "topology"
RESULTS_CSV = OUT_DIR / "topo_broad_screening_results.csv"
RESULTS_JSON = OUT_DIR / "topo_broad_screening_results.json"
AGGREGATE_CSV = OUT_DIR / "topo_broad_screening_aggregate.csv"
SUMMARY_JSON = OUT_DIR / "topo_broad_screening_summary.json"
FAILURES_JSON = OUT_DIR / "topo_broad_screening_failures.json"

SEEDS = [42, 0, 1]
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"

MODELS = {
    "GBLSTSVM": GBLSTSVM,
    "CW": CWGBLSTSVM,
    "CA": CAGBLSTSVM,
    "Topo": TopoGBLSTSVM,
}

METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "minority_recall",
    "gmean",
    "auroc",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _class_f1(y_true, y_pred, label: float) -> float:
    return float(
        f1_score(
            normalize_labels(y_true),
            normalize_labels(y_pred),
            pos_label=label,
            zero_division=0,
        )
    )


def _fit_model(name: str, X_train, y_train, seed: int):
    params = {
        "c1": DEFAULT_C1,
        "c2": DEFAULT_C2,
        "purity_threshold": PURITY,
        "min_ball_count": MIN_BALL_COUNT,
        "prediction_rule": PREDICTION_RULE,
        "random_state": seed,
    }
    if name != "GBLSTSVM":
        params["weight_placement"] = WEIGHT_PLACEMENT
    return MODELS[name](**params).fit(X_train, normalize_labels(y_train))


def _eval_model(model, X, y, minority_label: float) -> dict:
    y_pred = model.predict(X)
    scores = twin_scores(model, X)
    m = evaluate(y, y_pred, scores, minority_label)
    m["minority_f1"] = _class_f1(y, y_pred, minority_label)
    return m


def _ball_weights(model) -> np.ndarray:
    return np.concatenate([model.weights_pos_, model.weights_neg_])


def _mechanism_diagnostics(
    split,
    seed: int,
    models: dict,
    metrics: dict,
) -> dict:
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
    membership = compute_topology_membership(centers, labels)

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    kappa_ball = kappa_per_ball_array(balls, kappa_pos, kappa_neg)

    cw_model = models["CW"]
    ca_model = models["CA"]
    topo_model = models["Topo"]

    cw_w = _ball_weights(cw_model)
    ca_w = _ball_weights(ca_model)
    topo_w = _ball_weights(topo_model)

    minority_mask = labels == minority_label
    majority_mask = ~minority_mask

    rng = np.random.default_rng(seed)
    synthetic = _synthetic_noise_topo(
        split.X_train, y_train, balls, minority_label, rng
    )

    y_pred_cw = cw_model.predict(split.X_test)
    y_pred_topo = topo_model.predict(split.X_test)

    cw_metrics = metrics["CW"]
    topo_metrics = metrics["Topo"]

    return {
        "n_balls": len(balls),
        "class_imbalance_ratio": split.info.get("imbalance_ratio"),
        "compA_mean": float(np.mean(membership.comp)),
        "compA_minority_mean": float(np.mean(membership.comp[minority_mask]))
        if minority_mask.any()
        else None,
        "R_opp_mean": float(np.mean(membership.r_opp)),
        "R_opp_minority_mean": float(np.mean(membership.r_opp[minority_mask]))
        if minority_mask.any()
        else None,
        "muTopo_mean": float(np.mean(membership.mu_topo)),
        "muTopo_minority_mean": float(np.mean(membership.mu_topo[minority_mask]))
        if minority_mask.any()
        else None,
        "corr_topo_weights_vs_cw": safe_correlation(topo_w, cw_w, "pearson"),
        "corr_topo_weights_vs_ca": safe_correlation(topo_w, ca_w, "pearson"),
        "corr_muTopo_vs_cw_kappa": safe_correlation(membership.mu_topo, kappa_ball, "pearson"),
        "corr_A_vs_CW": safe_correlation(membership.ambiguity, kappa_ball, "pearson"),
        "minority_majority_weight_ratio_topo": (
            float(np.mean(topo_w[minority_mask]) / np.mean(topo_w[majority_mask]))
            if minority_mask.any() and majority_mask.any() and np.mean(topo_w[majority_mask]) > 0
            else None
        ),
        "noise_ball_muTopo_percentile": synthetic.get("noise_ball_muTopo_percentile"),
        "noise_ball_A_percentile": synthetic.get("noise_ball_A_percentile"),
        "prediction_change_rate_vs_cw": float(np.mean(y_pred_topo != y_pred_cw)),
        "delta_gmean_topo_vs_cw": float(topo_metrics["gmean"] - cw_metrics["gmean"]),
        "delta_minority_f1_topo_vs_cw": float(
            topo_metrics["minority_f1"] - cw_metrics["minority_f1"]
        ),
        "delta_gmean_topo_vs_ca": float(topo_metrics["gmean"] - metrics["CA"]["gmean"]),
        "delta_minority_f1_topo_vs_ca": float(
            topo_metrics["minority_f1"] - metrics["CA"]["minority_f1"]
        ),
        "mechanism_degenerate": float(np.std(membership.r_opp)) == 0.0,
    }


def _run_dataset_seed(descriptor: dict, seed: int) -> tuple[list[dict], dict | None]:
    split, err = try_load_descriptor(descriptor, seed=seed)
    if split is None:
        return [], {"dataset": descriptor["name"], "seed": seed, "error": err}

    rows: list[dict] = []
    fitted: dict = {}
    metrics_by_model: dict = {}

    for model_name in MODELS:
        try:
            model = _fit_model(model_name, split.X_train, split.y_train, seed)
            m = _eval_model(model, split.X_test, split.y_test, split.minority_label)
            fitted[model_name] = model
            metrics_by_model[model_name] = m
            rows.append(
                {
                    "dataset": split.name,
                    "seed": seed,
                    "method": model_name,
                    "source": descriptor["source"],
                    "n_features": split.info.get("n_features"),
                    "n_samples": split.info.get("n_samples"),
                    "class_imbalance_ratio": split.info.get("imbalance_ratio"),
                    **{k: m[k] for k in METRIC_KEYS},
                    "minority_f1": m["minority_f1"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            return [], {
                "dataset": descriptor["name"],
                "seed": seed,
                "error": f"{model_name}: {type(exc).__name__}: {exc}",
            }

    mech = _mechanism_diagnostics(split, seed, fitted, metrics_by_model)
    for row in rows:
        if row["method"] == "Topo":
            row.update(mech)
        row["delta_macro_f1_topo_vs_cw"] = (
            metrics_by_model["Topo"]["macro_f1"] - metrics_by_model["CW"]["macro_f1"]
        )
        row["delta_macro_f1_topo_vs_ca"] = (
            metrics_by_model["Topo"]["macro_f1"] - metrics_by_model["CA"]["macro_f1"]
        )
        row["delta_macro_f1_topo_vs_gblstsvm"] = (
            metrics_by_model["Topo"]["macro_f1"] - metrics_by_model["GBLSTSVM"]["macro_f1"]
        )
        row["delta_gmean_topo_vs_cw"] = (
            metrics_by_model["Topo"]["gmean"] - metrics_by_model["CW"]["gmean"]
        )
        row["delta_gmean_topo_vs_ca"] = (
            metrics_by_model["Topo"]["gmean"] - metrics_by_model["CA"]["gmean"]
        )

    return rows, None


def _aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    agg_rows = []
    for dataset, g in df.groupby("dataset"):
        pivot_f1 = g.pivot_table(index="seed", columns="method", values="macro_f1", aggfunc="first")
        pivot_gm = g.pivot_table(index="seed", columns="method", values="gmean", aggfunc="first")
        mean_f1 = pivot_f1.mean(numeric_only=True)
        mean_gm = pivot_gm.mean(numeric_only=True)
        agg_rows.append(
            {
                "dataset": dataset,
                "GBLSTSVM_macro_f1": mean_f1.get("GBLSTSVM"),
                "CW_macro_f1": mean_f1.get("CW"),
                "CA_macro_f1": mean_f1.get("CA"),
                "Topo_macro_f1": mean_f1.get("Topo"),
                "Topo-CW_macro_f1": mean_f1.get("Topo", 0) - mean_f1.get("CW", 0)
                if "Topo" in mean_f1 and "CW" in mean_f1
                else None,
                "Topo-CA_macro_f1": mean_f1.get("Topo", 0) - mean_f1.get("CA", 0)
                if "Topo" in mean_f1 and "CA" in mean_f1
                else None,
                "GBLSTSVM_gmean": mean_gm.get("GBLSTSVM"),
                "CW_gmean": mean_gm.get("CW"),
                "CA_gmean": mean_gm.get("CA"),
                "Topo_gmean": mean_gm.get("Topo"),
                "Topo-CW_gmean": mean_gm.get("Topo", 0) - mean_gm.get("CW", 0)
                if "Topo" in mean_gm and "CW" in mean_gm
                else None,
                "Topo-CA_gmean": mean_gm.get("Topo", 0) - mean_gm.get("CA", 0)
                if "Topo" in mean_gm and "CA" in mean_gm
                else None,
            }
        )
    return pd.DataFrame(agg_rows)


def _comparison_stats(agg: pd.DataFrame, delta_col: str) -> dict:
    deltas = agg[delta_col].dropna().to_numpy(dtype=float)
    if deltas.size == 0:
        return {"wins": 0, "ties": 0, "losses": 0, "mean": None, "median": None, "std": None}
    wins = int(np.sum(deltas > 1e-9))
    ties = int(np.sum(np.abs(deltas) <= 1e-9))
    losses = int(np.sum(deltas < -1e-9))
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean": float(np.mean(deltas)),
        "median": float(np.median(deltas)),
        "std": float(np.std(deltas)),
    }


def run_screening() -> dict:
    descriptors = discover_available_datasets()
    all_rows: list[dict] = []
    failures: list[dict] = []
    completed: set[str] = set()

    for desc in descriptors:
        dataset_failures = []
        dataset_rows = []
        for seed in SEEDS:
            rows, failure = _run_dataset_seed(desc, seed)
            if failure:
                dataset_failures.append(failure)
            else:
                dataset_rows.extend(rows)

        if dataset_rows:
            all_rows.extend(dataset_rows)
            completed.add(desc["name"])
        else:
            failures.append(
                {
                    "dataset": desc["name"],
                    "source": desc["source"],
                    "errors": dataset_failures,
                }
            )
            logger.warning("Failed: %s", desc["name"])

    df = pd.DataFrame(all_rows)
    agg = _aggregate_table(df) if not df.empty else pd.DataFrame()

    summary = {
        "datasets_detected": len(descriptors),
        "successfully_completed": len(completed),
        "failed": len(failures),
        "seeds": SEEDS,
        "completed_datasets": sorted(completed),
        "topo_vs_cw_macro_f1": _comparison_stats(agg, "Topo-CW_macro_f1"),
        "topo_vs_ca_macro_f1": _comparison_stats(agg, "Topo-CA_macro_f1"),
        "topo_vs_cw_gmean": _comparison_stats(agg, "Topo-CW_gmean"),
        "topo_vs_ca_gmean": _comparison_stats(agg, "Topo-CA_gmean"),
    }

    if not df.empty:
        topo_rows = df[df["method"] == "Topo"].copy()
        mech_findings = {
            "mean_prediction_change_rate_vs_cw": float(
                topo_rows["prediction_change_rate_vs_cw"].mean()
            ),
            "datasets_topo_beats_cw_gmean": int(
                (agg["Topo-CW_gmean"] > 0).sum()
            ),
            "datasets_topo_beats_ca_gmean": int(
                (agg["Topo-CA_gmean"] > 0).sum()
            ),
            "datasets_high_competition_minority": int(
                (topo_rows["compA_minority_mean"] > topo_rows["compA_mean"]).sum()
            ),
            "mean_corr_topo_weights_vs_ca": float(
                topo_rows["corr_topo_weights_vs_ca"].mean()
            ),
            "degenerate_mechanism_count": int(topo_rows["mechanism_degenerate"].sum()),
        }
        summary["mechanism_findings"] = mech_findings

    return {
        "summary": summary,
        "results": all_rows,
        "aggregate": agg.to_dict(orient="records"),
        "failures": failures,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run_screening()

    pd.DataFrame(payload["results"]).to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(payload["aggregate"]).to_csv(AGGREGATE_CSV, index=False)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    SUMMARY_JSON.write_text(
        json.dumps(payload["summary"], indent=2, default=str), encoding="utf-8"
    )
    FAILURES_JSON.write_text(
        json.dumps(payload["failures"], indent=2, default=str), encoding="utf-8"
    )

    s = payload["summary"]
    print(f"\nDatasets detected: {s['datasets_detected']}")
    print(f"Successfully completed: {s['successfully_completed']}")
    print(f"Failed: {s['failed']}")
    print(f"\nResults: {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
