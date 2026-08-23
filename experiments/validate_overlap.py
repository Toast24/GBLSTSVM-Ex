"""
OBRU experiment (Stages 7–9): train/val/test on Phase 7 benchmark datasets.

Runs only:
  - OBRU-GBLSTSVM
  - CW+OBRU-GBLSTSVM
  - CW+R+OBRU-GBLSTSVM

Uses the same fixed protocol as validate_multidataset.py (Phase 7).
Does not rerun frozen GBLSTSVM/CW/R/CR baselines.
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
    load_dataset_split,
    predict_and_evaluate,
    save_csv,
    save_json,
)
from src.overlap import DEFAULT_ALPHA, DEFAULT_EPSILON, DEFAULT_TOP_K
from src.overlap_gblstsvm import CROBRUGBLSTSVM, CWOBRUGBLSTSVM, OBRUGBLSTSVM
from src.reliability import DEFAULT_EPSILON as REL_EPSILON, DEFAULT_GAMMA, DEFAULT_LAMBDA_R

OUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 42
C1 = 100.0
C2 = 1e-2
PURITY = 0.95
MIN_BALL_COUNT = 2
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"
ALPHA = DEFAULT_ALPHA
OBRU_EPSILON = DEFAULT_EPSILON
TOP_K = DEFAULT_TOP_K

OBRU_MODELS = [
    "OBRU-GBLSTSVM",
    "CW+OBRU-GBLSTSVM",
    "CW+R+OBRU-GBLSTSVM",
]


def fit_obru_model(name: str, X_train, y_train):
    base = {
        "c1": C1,
        "c2": C2,
        "purity_threshold": PURITY,
        "min_ball_count": MIN_BALL_COUNT,
        "prediction_rule": PREDICTION_RULE,
        "random_state": SEED,
        "alpha": ALPHA,
        "epsilon": OBRU_EPSILON,
        "top_k": TOP_K,
        "weight_placement": WEIGHT_PLACEMENT,
    }
    if name == "OBRU-GBLSTSVM":
        return OBRUGBLSTSVM(**base).fit(X_train, y_train)
    if name == "CW+OBRU-GBLSTSVM":
        return CWOBRUGBLSTSVM(**base).fit(X_train, y_train)
    if name == "CW+R+OBRU-GBLSTSVM":
        return CROBRUGBLSTSVM(
            **base,
            rel_epsilon=REL_EPSILON,
            gamma=DEFAULT_GAMMA,
            lambda_r=DEFAULT_LAMBDA_R,
            reliability_variant="full",
        ).fit(X_train, y_train)
    raise ValueError(name)


def model_diagnostics(name: str, model) -> dict:
    O = model.overlap_all_
    mu = model.mu_all_
    return {
        "model": name,
        "n_balls": len(model.balls_),
        "mean_overlap": float(np.mean(O)) if O is not None and O.size else float("nan"),
        "mean_mu": float(np.mean(mu)) if mu is not None and mu.size else float("nan"),
        "min_mu": float(np.min(mu)) if mu is not None and mu.size else float("nan"),
        "max_mu": float(np.max(mu)) if mu is not None and mu.size else float("nan"),
        "frac_overlap_gt_0": float(np.mean(O > 0.0))
        if O is not None and O.size
        else float("nan"),
    }


def aggregate_rows(all_results: list[dict]) -> list[dict]:
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


def load_phase7_baseline_test_metrics() -> dict:
    path = OUT_DIR / "multidataset_train_val_test.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    baseline_map = {
        "CW-GBLSTSVM": "CW+OBRU-GBLSTSVM",
        "CR-GBLSTSVM": "CW+R+OBRU-GBLSTSVM",
    }
    for ds_name, ds_data in data.get("datasets", {}).items():
        out[ds_name] = {}
        models = ds_data.get("results", {})
        for base_name, obru_name in baseline_map.items():
            base_test = models.get(base_name, {}).get("test", {})
            if base_test:
                out[ds_name][obru_name] = {
                    "baseline": base_name,
                    "baseline_test": base_test,
                }
    return out


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
            "alpha": ALPHA,
            "obru_epsilon": OBRU_EPSILON,
            "top_k": TOP_K,
            "scaling": "none",
            "reliability_epsilon": REL_EPSILON,
            "reliability_gamma": DEFAULT_GAMMA,
            "reliability_lambda_r": DEFAULT_LAMBDA_R,
            "tuning": "none",
        },
        "models": {},
        "diagnostics": {},
    }

    for name in OBRU_MODELS:
        try:
            model = fit_obru_model(name, X_train, y_train)
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
        }
        results["diagnostics"][name] = model_diagnostics(name, model)

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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    raw_rows = []

    for ds_file in DATASETS:
        split = load_dataset_split(ds_file, SEED)
        rec = run_dataset(split)
        all_results.append(rec)
        raw_rows.extend(rec.get("raw_rows", []))

    baseline_ref = load_phase7_baseline_test_metrics()
    comparisons = []
    for rec in all_results:
        ds = rec["dataset"]
        for obru_name, base_info in baseline_ref.get(ds, {}).items():
            obru_test = rec["models"].get(obru_name, {}).get("test")
            base_test = base_info.get("baseline_test")
            if obru_test and base_test:
                comparisons.append(
                    {
                        "dataset": ds,
                        "obru_model": obru_name,
                        "baseline_model": base_info["baseline"],
                        "delta_test": delta_metrics(obru_test, base_test),
                    }
                )

    payload = {
        "experiment": "obru_stages_7_9",
        "models": OBRU_MODELS,
        "datasets": {r["dataset"]: r for r in all_results},
        "comparisons_vs_phase7_baselines": comparisons,
        "aggregate_rows": aggregate_rows(all_results),
    }

    save_json(OUT_DIR / "validate_overlap.json", payload)
    save_csv(OUT_DIR / "validate_overlap.csv", aggregate_rows(all_results))
    save_csv(OUT_DIR / "overlap_eval_raw.csv", raw_rows)

    print(f"Wrote {OUT_DIR / 'validate_overlap.json'}")
    print(f"Wrote {OUT_DIR / 'validate_overlap.csv'}")
    print(f"Wrote {OUT_DIR / 'overlap_eval_raw.csv'}")


if __name__ == "__main__":
    main()
