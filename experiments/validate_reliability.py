"""
Sanity-check script for R-GBLSTSVM on synthetic data.

Not a tuned research experiment — records reliability/ball statistics and
basic metrics for balanced and moderately imbalanced conditions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

from src.gblstsvm import GBLSTSVM
from src.r_gblstsvm import RGBLSTSVM
from src.reliability import DEFAULT_EPSILON, DEFAULT_GAMMA, DEFAULT_LAMBDA_R

SEED = 42
N_FEATURES = 6
C1 = 100.0
C2 = 1e-2
PURITY = 0.95
MIN_BALL_COUNT = 2


def make_dataset(n_minority: int, n_majority: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_minority + n_majority
    X = rng.standard_normal((n, N_FEATURES))
    X[:n_minority, 0] += 1.5
    X[n_minority:, 0] -= 1.5
    y = np.array([1.0] * n_minority + [-1.0] * n_majority)
    return X, y


def minority_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true == 1
    if mask.sum() == 0:
        return float("nan")
    return float(recall_score(y_true[mask], y_pred[mask], pos_label=1.0))


def distribution_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q50": float(np.quantile(values, 0.5)),
    }


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    t0 = time.perf_counter()
    preds = model.predict(X)
    runtime = time.perf_counter() - t0
    y_bin = np.where(y > 0, 1.0, -1.0)
    return {
        "accuracy": float(accuracy_score(y_bin, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_bin, preds)),
        "macro_f1": float(f1_score(y_bin, preds, average="macro")),
        "minority_recall": minority_recall(y_bin, preds),
        "predict_runtime_sec": runtime,
    }


def run_scenario(name: str, n_minority: int, n_majority: int) -> dict:
    X, y = make_dataset(n_minority, n_majority, SEED)
    params = {
        "c1": C1,
        "c2": C2,
        "purity_threshold": PURITY,
        "min_ball_count": MIN_BALL_COUNT,
        "prediction_rule": "reference",
        "random_state": SEED,
        "epsilon": DEFAULT_EPSILON,
        "gamma": DEFAULT_GAMMA,
        "lambda_r": DEFAULT_LAMBDA_R,
    }

    fit_t0 = time.perf_counter()
    r_model = RGBLSTSVM(**params).fit(X, y)
    r_fit_time = time.perf_counter() - fit_t0

    base = GBLSTSVM(
        c1=C1,
        c2=C2,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
        prediction_rule="reference",
        random_state=SEED,
    ).fit(X, y)

    purities = np.array([b.purity for b in r_model.balls_])
    radii = np.array([b.radius for b in r_model.balls_])

    record = {
        "scenario": name,
        "n_minority": n_minority,
        "n_majority": n_majority,
        "n_balls": len(r_model.balls_),
        "radius_scale": r_model.radius_scale_,
        "purity_distribution": distribution_summary(purities),
        "radius_distribution": distribution_summary(radii),
        "reliability_distribution": distribution_summary(r_model.reliabilities_),
        "fit_runtime_sec": r_fit_time,
        "metrics": {
            "r_gblstsvm": evaluate_model(r_model, X, y),
            "baseline_gblstsvm": evaluate_model(base, X, y),
        },
        "reliability_variants": {},
    }

    for variant in ("purity_only", "radius_impurity_only", "full"):
        m = RGBLSTSVM(**params, reliability_variant=variant).fit(X, y)
        record["reliability_variants"][variant] = {
            "reliability_mean": float(np.mean(m.reliabilities_)),
            "metrics": evaluate_model(m, X, y),
        }

    return record


def main() -> None:
    results = [
        run_scenario("balanced_100_100", 100, 100),
        run_scenario("imbalanced_50_150", 50, 150),
    ]

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validate_reliability.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}\n")
    for row in results:
        print(f"=== {row['scenario']} ===")
        print(f"  balls: {row['n_balls']}, radius_scale: {row['radius_scale']:.4f}")
        rd = row["reliability_distribution"]
        print(
            f"  reliability: min={rd['min']:.3f}, mean={rd['mean']:.3f}, max={rd['max']:.3f}"
        )
        m = row["metrics"]["r_gblstsvm"]
        b = row["metrics"]["baseline_gblstsvm"]
        print(
            f"  R: acc={m['accuracy']:.3f}, bal_acc={m['balanced_accuracy']:.3f}, "
            f"macro_f1={m['macro_f1']:.3f}, min_recall={m['minority_recall']:.3f}"
        )
        print(
            f"  baseline: acc={b['accuracy']:.3f}, bal_acc={b['balanced_accuracy']:.3f}, "
            f"macro_f1={b['macro_f1']:.3f}, min_recall={b['minority_recall']:.3f}"
        )


if __name__ == "__main__":
    main()
