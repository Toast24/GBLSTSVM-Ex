"""
BMRU experiment (Stages 4–6): train/val/test on Phase 7 benchmark datasets.

Runs only:
  - BMRU-GBLSTSVM
  - CW+BMRU-GBLSTSVM
  - CW+R+BMRU-GBLSTSVM

Does not rerun frozen GBLSTSVM/CW/R/CR baselines.
Uses the same fixed protocol as validate_multidataset.py (Phase 7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.phase8_common import (
    DATASETS,
    METRIC_KEYS,
    DatasetSplit,
    delta_metrics,
    fit_scaler,
    load_dataset_split,
    predict_and_evaluate,
    save_csv,
    save_json,
    twin_scores,
)
from src.bmru import DEFAULT_ALPHA, DEFAULT_EPSILON, DEFAULT_MAX_ITERATIONS, bmru_diagnostics_summary, iteration_record_to_dict
from src.bmru_gblstsvm import BMRUGBLSTSVM, CRBMRUGBLSTSVM, CWBMRUGBLSTSVM
from src.reliability import DEFAULT_EPSILON as REL_EPSILON, DEFAULT_GAMMA, DEFAULT_LAMBDA_R

OUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 42
C1 = 100.0
C2 = 1e-2
PURITY = 0.95
MIN_BALL_COUNT = 2
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"
MARGIN_MODE = "reference"
MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS
ALPHA = DEFAULT_ALPHA
BMRU_EPSILON = DEFAULT_EPSILON

BMRU_MODELS = [
    "BMRU-GBLSTSVM",
    "CW+BMRU-GBLSTSVM",
    "CW+R+BMRU-GBLSTSVM",
]

COMPARISON_PAIRS = [
    ("BMRU-GBLSTSVM", "R-GBLSTSVM", "bmru_vs_r_note"),
    ("CW+BMRU-GBLSTSVM", "CW-GBLSTSVM", "cw_bmru_vs_cw_note"),
    ("CW+R+BMRU-GBLSTSVM", "CR-GBLSTSVM", "cr_bmru_vs_cr_note"),
]


def fit_bmru_model(name: str, X_train, y_train, margin_mode: str = MARGIN_MODE):
    base = {
        "c1": C1,
        "c2": C2,
        "purity_threshold": PURITY,
        "min_ball_count": MIN_BALL_COUNT,
        "prediction_rule": PREDICTION_RULE,
        "random_state": SEED,
        "alpha": ALPHA,
        "epsilon": BMRU_EPSILON,
        "margin_mode": margin_mode,
        "max_iterations": MAX_ITERATIONS,
        "weight_placement": WEIGHT_PLACEMENT,
    }
    if name == "BMRU-GBLSTSVM":
        return BMRUGBLSTSVM(**base).fit(X_train, y_train)
    if name == "CW+BMRU-GBLSTSVM":
        return CWBMRUGBLSTSVM(**base).fit(X_train, y_train)
    if name == "CW+R+BMRU-GBLSTSVM":
        return CRBMRUGBLSTSVM(
            **base,
            rel_epsilon=REL_EPSILON,
            gamma=DEFAULT_GAMMA,
            lambda_r=DEFAULT_LAMBDA_R,
            reliability_variant="full",
        ).fit(X_train, y_train)
    raise ValueError(name)


def model_diagnostics(name: str, model) -> dict:
    rel = getattr(model, "reliabilities_", None)
    cw = None
    if hasattr(model, "kappa_pos_balls_") and model.kappa_pos_balls_ is not None:
        cw = np.concatenate([model.kappa_pos_balls_, model.kappa_neg_balls_])
    diag = bmru_diagnostics_summary(
        model.balls_,
        model.u_all_,
        model.mu_all_,
        reliabilities=rel,
        class_weights=cw,
    )
    if model.iteration_history_:
        last = model.iteration_history_[-1]
        diag["iteration_count"] = len(model.iteration_history_)
        diag["final_max_abs_weight_change"] = last.max_abs_weight_change
        diag["final_n_u_lt_1"] = last.n_u_lt_1
        diag["final_n_u_ge_1"] = last.n_u_ge_1
    diag["model"] = name
    return diag


def aggregate_rows(all_results: list[dict]) -> list[dict]:
    """Mean ± std per dataset/model/metric across folds (here single split per dataset)."""
    rows = []
    for rec in all_results:
        for model_name, model_data in rec["models"].items():
            for split_name in ("val", "test"):
                metrics = model_data.get(split_name, {})
                row = {
                    "dataset": rec["dataset"],
                    "model": model_name,
                    "split": split_name,
                    "seed": SEED,
                }
                for k in METRIC_KEYS:
                    v = metrics.get(k)
                    row[k] = v
                    row[f"{k}_mean"] = v
                    row[f"{k}_std"] = 0.0
                rows.append(row)
    return rows


def run_dataset(split: DatasetSplit) -> dict:
    X_train, X_val, X_test = split.X_train, split.X_val, split.X_test
    y_train, y_val, y_test = split.y_train, split.y_val, split.y_test
    minority = split.minority_label

    results = {
        "dataset": split.name,
        "protocol": {
            "seed": SEED,
            "c1": C1,
            "c2": C2,
            "purity_threshold": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "prediction_rule": PREDICTION_RULE,
            "weight_placement": WEIGHT_PLACEMENT,
            "margin_mode": MARGIN_MODE,
            "alpha": ALPHA,
            "bmru_epsilon": BMRU_EPSILON,
            "max_iterations": MAX_ITERATIONS,
            "scaling": "none",
            "reliability_epsilon": REL_EPSILON,
            "reliability_gamma": DEFAULT_GAMMA,
            "reliability_lambda_r": DEFAULT_LAMBDA_R,
            "tuning": "none",
        },
        "models": {},
        "diagnostics": {},
        "iteration_history": {},
    }

    for name in BMRU_MODELS:
        try:
            model = fit_bmru_model(name, X_train, y_train)
        except Exception as exc:  # noqa: BLE001
            results["models"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        val_m, val_pred, val_scores = predict_and_evaluate(
            model, name, X_val, y_val, minority
        )
        test_m, test_pred, test_scores = predict_and_evaluate(
            model, name, X_test, y_test, minority
        )

        results["models"][name] = {
            "val": val_m,
            "test": test_m,
            "n_iterations": len(model.iteration_history_ or []),
        }
        results["diagnostics"][name] = model_diagnostics(name, model)
        results["iteration_history"][name] = [
            iteration_record_to_dict(h) for h in (model.iteration_history_ or [])
        ]

        # Raw per-split rows for later paired statistics
        for split_name, metrics, preds, scores in (
            ("val", val_m, val_pred, val_scores),
            ("test", test_m, test_pred, test_scores),
        ):
            results.setdefault("raw_rows", []).append(
                {
                    "dataset": split.name,
                    "model": name,
                    "split": split_name,
                    "seed": SEED,
                    **{k: metrics.get(k) for k in METRIC_KEYS},
                    "predictions": preds.tolist(),
                    "scores": scores.tolist() if scores is not None else None,
                }
            )

    return results


def load_phase7_baseline_test_metrics() -> dict:
    """Load existing Phase 7 test metrics for delta comparisons."""
    path = OUT_DIR / "multidataset_train_val_test.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    baseline_map = {
        "R-GBLSTSVM": "BMRU-GBLSTSVM",
        "CW-GBLSTSVM": "CW+BMRU-GBLSTSVM",
        "CR-GBLSTSVM": "CW+R+BMRU-GBLSTSVM",
    }
    for ds_name, ds_data in data.get("datasets", {}).items():
        out[ds_name] = {}
        models = ds_data.get("results", {})
        for base_name, bmru_name in baseline_map.items():
            base_test = models.get(base_name, {}).get("test", {})
            if base_test:
                out[ds_name][bmru_name] = {"baseline": base_name, "baseline_test": base_test}
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    raw_rows = []
    diag_rows = []

    for ds_file in DATASETS:
        split = load_dataset_split(ds_file, SEED)
        rec = run_dataset(split)
        all_results.append(rec)
        raw_rows.extend(rec.get("raw_rows", []))
        for model_name, diag in rec.get("diagnostics", {}).items():
            diag_rows.append({"dataset": split.name, **diag})

    baseline_ref = load_phase7_baseline_test_metrics()
    comparisons = []
    for rec in all_results:
        ds = rec["dataset"]
        for bmru_name, base_info in baseline_ref.get(ds, {}).items():
            bmru_test = rec["models"].get(bmru_name, {}).get("test")
            base_test = base_info.get("baseline_test")
            if bmru_test and base_test:
                comparisons.append(
                    {
                        "dataset": ds,
                        "bmru_model": bmru_name,
                        "baseline_model": base_info["baseline"],
                        "delta_test": delta_metrics(bmru_test, base_test),
                    }
                )

    payload = {
        "experiment": "bmru_stages_4_6",
        "models": BMRU_MODELS,
        "datasets": {r["dataset"]: r for r in all_results},
        "comparisons_vs_phase7_baselines": comparisons,
        "aggregate_rows": aggregate_rows(all_results),
    }

    save_json(OUT_DIR / "validate_bmru.json", payload)
    save_csv(OUT_DIR / "validate_bmru.csv", aggregate_rows(all_results))
    save_csv(OUT_DIR / "validate_bmru_raw_rows.csv", raw_rows)
    save_csv(OUT_DIR / "validate_bmru_diagnostics.csv", diag_rows)

    print(f"Wrote {OUT_DIR / 'validate_bmru.json'}")
    print(f"Wrote {OUT_DIR / 'validate_bmru.csv'}")
    print(f"Wrote {OUT_DIR / 'validate_bmru_diagnostics.csv'}")


if __name__ == "__main__":
    main()
