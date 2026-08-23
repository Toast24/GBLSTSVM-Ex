"""
SupportAware-Topo screening: GBLSTSVM, CW, CA, original Topo, SupportAware-Topo.

Same protocol as topo_broad_screening.py (frozen 70/15/15, seeds {42,0,1},
frozen C1/C2/purity). Original Topo is not modified.

Run from GBLSTSVM root:
  python experiments/topo_support_aware_screening.py
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

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
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
from src.cbcm import ball_cbcm_weights  # noqa: E402
from src.cbcm_gblstsvm import CAGBLSTSVM  # noqa: E402
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.cw_gblstsvm import CWGBLSTSVM  # noqa: E402
from src.gblstsvm import GBLSTSVM  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.topology_gblstsvm import SupportAwareTopoGBLSTSVM, TopoGBLSTSVM  # noqa: E402
from src.topology_membership import ball_support_aware_topo_weights, ball_topo_weights  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "topology"
RESULTS_CSV = OUT_DIR / "topo_support_aware_screening_results.csv"
AGGREGATE_CSV = OUT_DIR / "topo_support_aware_screening_aggregate.csv"
SUMMARY_JSON = OUT_DIR / "topo_support_aware_screening_summary.json"
FAILURES_JSON = OUT_DIR / "topo_support_aware_screening_failures.json"
DIAG_BALL_CSV = OUT_DIR / "topo_support_aware_ball_diagnostics.csv"
DIAG_RUN_CSV = OUT_DIR / "topo_support_aware_run_diagnostics.csv"
REPORT_MD = OUT_DIR / "topo_support_aware_report.md"

SEEDS = [42, 0, 1]
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"

MODELS = {
    "GBLSTSVM": GBLSTSVM,
    "CW": CWGBLSTSVM,
    "CA": CAGBLSTSVM,
    "Topo": TopoGBLSTSVM,
    "SupportAware-Topo": SupportAwareTopoGBLSTSVM,
}

METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "minority_recall",
    "gmean",
    "auroc",
)

FOCUS_RUNS = {
    ("ecoli2", 42),
    ("ionosphere", 0),
    ("vehicle0", 1),
    ("chronic_kidney_disease", 42),
    ("credit_approval", 42),
    ("spambase", 42),
}

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


def _ball_weight_vector(weights_pos, weights_neg, balls) -> np.ndarray:
    pos_i = neg_i = 0
    out = np.empty(len(balls), dtype=float)
    for i, b in enumerate(balls):
        if b.label == 1:
            out[i] = weights_pos[pos_i]
            pos_i += 1
        else:
            out[i] = weights_neg[neg_i]
            neg_i += 1
    return out


def _weight_diagnostics(split, seed: int) -> tuple[list[dict], dict]:
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
    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    ca_pos, ca_neg, _, _, _ = ball_cbcm_weights(
        balls, kappa_pos, kappa_neg, variant="competition"
    )
    topo_pos, topo_neg, topo_mem, _, _ = ball_topo_weights(balls, kappa_pos, kappa_neg)
    sa_pos, sa_neg, sa_mem, _, _ = ball_support_aware_topo_weights(
        balls, kappa_pos, kappa_neg
    )

    ca_w = _ball_weight_vector(ca_pos, ca_neg, balls)
    topo_w = _ball_weight_vector(topo_pos, topo_neg, balls)
    sa_w = _ball_weight_vector(sa_pos, sa_neg, balls)

    run_id = f"{split.name}_s{seed}"
    ball_rows: list[dict] = []
    keep_balls = (split.name, seed) in FOCUS_RUNS
    for i, b in enumerate(balls):
        if not keep_balls:
            break
        ball_rows.append(
            {
                "run_id": run_id,
                "dataset": split.name,
                "seed": seed,
                "ball_id": i,
                "is_minority": bool(b.label == minority_label),
                "n_opp": int(sa_mem.n_opp[i]),
                "R_opp": float(sa_mem.r_opp[i]),
                "R_opp_adjusted": float(sa_mem.r_opp_adjusted[i]),
                "support_confidence": float(sa_mem.support_confidence[i]),
                "compA": float(sa_mem.ambiguity[i]),
                "ca_weight": float(ca_w[i]),
                "topo_weight": float(topo_w[i]),
                "support_aware_weight": float(sa_w[i]),
                "Topo-CA": float(topo_w[i] - ca_w[i]),
                "SupportAwareTopo-CA": float(sa_w[i] - ca_w[i]),
            }
        )

    minority = np.array([b.label == minority_label for b in balls])
    zero_r = sa_mem.r_opp == 0.0
    run_row = {
        "run_id": run_id,
        "dataset": split.name,
        "seed": seed,
        "n_balls": len(balls),
        "n_opp_mean": float(np.mean(sa_mem.n_opp)),
        "n_opp_median": float(np.median(sa_mem.n_opp)),
        "R_opp_mean": float(np.mean(sa_mem.r_opp)),
        "R_opp_adjusted_mean": float(np.mean(sa_mem.r_opp_adjusted)),
        "frac_R_opp_eq_0": float(np.mean(zero_r)),
        "frac_R_opp_eq_0_and_n_opp_le_1": float(
            np.mean(zero_r & (sa_mem.n_opp <= 1))
        ),
        "mean_ca_weight": float(np.mean(ca_w)),
        "mean_topo_weight": float(np.mean(topo_w)),
        "mean_support_aware_weight": float(np.mean(sa_w)),
        "mean_Topo-CA": float(np.mean(topo_w - ca_w)),
        "mean_SupportAwareTopo-CA": float(np.mean(sa_w - ca_w)),
        "mean_abs_Topo-CA": float(np.mean(np.abs(topo_w - ca_w))),
        "mean_abs_SupportAwareTopo-CA": float(np.mean(np.abs(sa_w - ca_w))),
        "minority_mean_Topo-CA": float(np.mean((topo_w - ca_w)[minority]))
        if minority.any()
        else None,
        "minority_mean_SupportAwareTopo-CA": float(np.mean((sa_w - ca_w)[minority]))
        if minority.any()
        else None,
        "high_comp_minority_mean_Topo-CA": (
            float(np.mean((topo_w - ca_w)[minority & (sa_mem.ambiguity > 0.5)]))
            if np.any(minority & (sa_mem.ambiguity > 0.5))
            else None
        ),
        "high_comp_minority_mean_SupportAwareTopo-CA": (
            float(np.mean((sa_w - ca_w)[minority & (sa_mem.ambiguity > 0.5)]))
            if np.any(minority & (sa_mem.ambiguity > 0.5))
            else None
        ),
        "focus_run": keep_balls,
    }
    return ball_rows, run_row


def _run_dataset_seed(descriptor: dict, seed: int):
    split, err = try_load_descriptor(descriptor, seed=seed)
    if split is None:
        return [], None, None, {"dataset": descriptor["name"], "seed": seed, "error": err}

    rows: list[dict] = []
    metrics_by_model: dict = {}

    for model_name in MODELS:
        try:
            model = _fit_model(model_name, split.X_train, split.y_train, seed)
            m = _eval_model(model, split.X_test, split.y_test, split.minority_label)
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
            return [], None, None, {
                "dataset": descriptor["name"],
                "seed": seed,
                "error": f"{model_name}: {type(exc).__name__}: {exc}",
            }

    ca = metrics_by_model["CA"]
    topo = metrics_by_model["Topo"]
    sa = metrics_by_model["SupportAware-Topo"]
    for row in rows:
        row["delta_macro_f1_vs_ca"] = (
            metrics_by_model[row["method"]]["macro_f1"] - ca["macro_f1"]
        )
        row["delta_gmean_vs_ca"] = (
            metrics_by_model[row["method"]]["gmean"] - ca["gmean"]
        )
        row["delta_minority_f1_vs_ca"] = (
            metrics_by_model[row["method"]]["minority_f1"] - ca["minority_f1"]
        )

    ball_rows, run_diag = _weight_diagnostics(split, seed)
    run_diag.update(
        {
            "CA_macro_f1": ca["macro_f1"],
            "Topo_macro_f1": topo["macro_f1"],
            "SA_macro_f1": sa["macro_f1"],
            "CA_gmean": ca["gmean"],
            "Topo_gmean": topo["gmean"],
            "SA_gmean": sa["gmean"],
            "CA_minority_f1": ca["minority_f1"],
            "Topo_minority_f1": topo["minority_f1"],
            "SA_minority_f1": sa["minority_f1"],
            "delta_macro_f1_Topo_vs_CA": topo["macro_f1"] - ca["macro_f1"],
            "delta_macro_f1_SA_vs_CA": sa["macro_f1"] - ca["macro_f1"],
            "delta_gmean_Topo_vs_CA": topo["gmean"] - ca["gmean"],
            "delta_gmean_SA_vs_CA": sa["gmean"] - ca["gmean"],
            "delta_minority_f1_Topo_vs_CA": topo["minority_f1"] - ca["minority_f1"],
            "delta_minority_f1_SA_vs_CA": sa["minority_f1"] - ca["minority_f1"],
        }
    )
    return rows, ball_rows, run_diag, None


def _pivot_mean(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="dataset", columns="method", values=value, aggfunc="mean")


def _wtl(deltas: np.ndarray) -> dict:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {"wins": 0, "ties": 0, "losses": 0, "mean": None, "median": None}
    return {
        "wins": int(np.sum(deltas > 1e-9)),
        "ties": int(np.sum(np.abs(deltas) <= 1e-9)),
        "losses": int(np.sum(deltas < -1e-9)),
        "mean": float(np.mean(deltas)),
        "median": float(np.median(deltas)),
    }


def _method_vs_ca(df: pd.DataFrame, method: str) -> dict:
    sub = df[df["method"] == method]
    return {
        "macro_f1": _wtl(sub["delta_macro_f1_vs_ca"].to_numpy()),
        "gmean": _wtl(sub["delta_gmean_vs_ca"].to_numpy()),
        "minority_f1": _wtl(sub["delta_minority_f1_vs_ca"].to_numpy()),
    }


def _write_report(summary: dict, agg: pd.DataFrame, run_diag: pd.DataFrame) -> None:
    formula = summary["formula"]
    lines = [
        "# SupportAware-Topo screening",
        "",
        "## Transformation",
        "",
        "Original Topo (unchanged):",
        "",
        r"$$\mu^{\mathrm{Topo}} = A \cdot R_{\mathrm{opp}}$$",
        "",
        "SupportAware-Topo (unit-information Laplace / credibility shrinkage toward 1):",
        "",
        r"$$c(n_{\mathrm{opp}}) = \frac{n_{\mathrm{opp}}}{n_{\mathrm{opp}}+\lambda},\quad \lambda=1$$",
        "",
        r"$$R_{\mathrm{opp}}^{S} = c\cdot R_{\mathrm{opp}} + (1-c)\cdot 1$$",
        "",
        r"$$\mu^{\mathrm{Topo\text{-}S}} = A \cdot R_{\mathrm{opp}}^{S},\qquad w = \mathrm{CW}(y)\, \mu^{\mathrm{Topo\text{-}S}}$$",
        "",
        f"`λ = {formula['lambda']}`, prior = {formula['prior']} (not tuned per dataset).",
        "",
        "Low `n_opp` reduces the influence of `R_opp`. High `n_opp` recovers original Topo.",
        "`R_opp = 0` with `n_opp = 0` yields `R_opp^S = 1`, so CA is not fully suppressed.",
        "",
        "## Overall vs CA (run-level, all dataset×seed)",
        "",
    ]
    for method in ("Topo", "SupportAware-Topo"):
        s = summary["vs_ca"][method]
        lines.append(f"### {method}")
        for metric in ("macro_f1", "gmean", "minority_f1"):
            d = s[metric]
            lines.append(
                f"- {metric}: W/T/L = {d['wins']}/{d['ties']}/{d['losses']}; "
                f"mean Δ = {d['mean']:.4f}; median Δ = {d['median']:.4f}"
            )
        lines.append("")

    lines.extend(["## Dataset means", "", agg.to_csv(index=False), ""])

    focus = run_diag[run_diag["focus_run"]].sort_values(["dataset", "seed"])
    if not focus.empty:
        cols = [
            "run_id",
            "n_balls",
            "n_opp_mean",
            "R_opp_mean",
            "R_opp_adjusted_mean",
            "frac_R_opp_eq_0",
            "mean_abs_Topo-CA",
            "mean_abs_SupportAwareTopo-CA",
            "high_comp_minority_mean_Topo-CA",
            "high_comp_minority_mean_SupportAwareTopo-CA",
            "delta_macro_f1_Topo_vs_CA",
            "delta_macro_f1_SA_vs_CA",
            "delta_gmean_Topo_vs_CA",
            "delta_gmean_SA_vs_CA",
            "delta_minority_f1_Topo_vs_CA",
            "delta_minority_f1_SA_vs_CA",
        ]
        lines.extend(
            [
                "## Focus runs",
                "",
                focus[cols].to_csv(index=False),
                "",
            ]
        )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run_screening() -> dict:
    descriptors = discover_available_datasets()
    all_rows: list[dict] = []
    all_balls: list[dict] = []
    all_runs: list[dict] = []
    failures: list[dict] = []
    completed: set[str] = set()

    for desc in descriptors:
        dataset_failures = []
        dataset_rows = []
        for seed in SEEDS:
            rows, ball_rows, run_diag, failure = _run_dataset_seed(desc, seed)
            if failure:
                dataset_failures.append(failure)
            else:
                dataset_rows.extend(rows)
                all_balls.extend(ball_rows or [])
                all_runs.append(run_diag)

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
    run_diag = pd.DataFrame(all_runs)
    ball_df = pd.DataFrame(all_balls)

    if df.empty:
        agg = pd.DataFrame()
    else:
        f1 = _pivot_mean(df, "macro_f1")
        gm = _pivot_mean(df, "gmean")
        mf = _pivot_mean(df, "minority_f1")
        agg = pd.DataFrame({"dataset": f1.index.astype(str)})
        for method in MODELS:
            if method in f1.columns:
                agg[f"{method}_macro_f1"] = f1[method].to_numpy()
                agg[f"{method}_gmean"] = gm[method].to_numpy()
                agg[f"{method}_minority_f1"] = mf[method].to_numpy()
        if "Topo" in f1.columns and "CA" in f1.columns:
            agg["Topo-CA_macro_f1"] = (f1["Topo"] - f1["CA"]).to_numpy()
            agg["SA-CA_macro_f1"] = (f1["SupportAware-Topo"] - f1["CA"]).to_numpy()
            agg["Topo-CA_gmean"] = (gm["Topo"] - gm["CA"]).to_numpy()
            agg["SA-CA_gmean"] = (gm["SupportAware-Topo"] - gm["CA"]).to_numpy()
            agg["Topo-CA_minority_f1"] = (mf["Topo"] - mf["CA"]).to_numpy()
            agg["SA-CA_minority_f1"] = (mf["SupportAware-Topo"] - mf["CA"]).to_numpy()

    summary = {
        "formula": {
            "name": "Laplace/credibility shrinkage of R_opp toward 1",
            "R_opp_S": "c(n_opp)*R_opp + (1-c(n_opp))*1",
            "c": "n_opp / (n_opp + lambda)",
            "lambda": 1.0,
            "prior": 1.0,
            "mu": "A * R_opp_S",
        },
        "datasets_detected": len(descriptors),
        "successfully_completed": len(completed),
        "failed": len(failures),
        "seeds": SEEDS,
        "completed_datasets": sorted(completed),
        "vs_ca": {
            "Topo": _method_vs_ca(df, "Topo") if not df.empty else {},
            "SupportAware-Topo": _method_vs_ca(df, "SupportAware-Topo")
            if not df.empty
            else {},
        },
    }
    return {
        "summary": summary,
        "results": df,
        "aggregate": agg,
        "run_diagnostics": run_diag,
        "ball_diagnostics": ball_df,
        "failures": failures,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run_screening()
    payload["results"].to_csv(RESULTS_CSV, index=False)
    payload["aggregate"].to_csv(AGGREGATE_CSV, index=False)
    payload["run_diagnostics"].to_csv(DIAG_RUN_CSV, index=False)
    payload["ball_diagnostics"].to_csv(DIAG_BALL_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(payload["summary"], indent=2, default=str), encoding="utf-8"
    )
    FAILURES_JSON.write_text(
        json.dumps(payload["failures"], indent=2, default=str), encoding="utf-8"
    )
    _write_report(payload["summary"], payload["aggregate"], payload["run_diagnostics"])

    s = payload["summary"]
    print(f"\nDatasets detected: {s['datasets_detected']}")
    print(f"Successfully completed: {s['successfully_completed']}")
    print(f"Failed: {s['failed']}")
    print(f"Report: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
