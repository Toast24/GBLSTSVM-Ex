"""
Sanity-check script for CR-GBLSTSVM on synthetic data.

Compares GBLSTSVM, CW-GBLSTSVM, R-GBLSTSVM, and CR-GBLSTSVM on the same
train/test split and random seed. Reports weight diagnostics, reliability
saturation, and Delta_reliability / Delta_class metrics.
"""

from __future__ import annotations

import csv
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
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.cr_gblstsvm import CRGBLSTSVM
from src.cw_gblstsvm import CWGBLSTSVM
from src.gblstsvm import GBLSTSVM
from src.r_gblstsvm import RGBLSTSVM
from src.reliability import DEFAULT_EPSILON, DEFAULT_GAMMA, DEFAULT_LAMBDA_R

SEED = 42
N_FEATURES = 6
TEST_SIZE = 0.3
C1 = 100.0
C2 = 1e-2
PURITY = 0.95
MIN_BALL_COUNT = 2

SCENARIOS = [
    ("balanced", 100, 100),
    ("imbalanced_1_3", 50, 150),
    ("imbalanced_1_5", 33, 167),
]


def make_dataset(
    n_minority: int, n_majority: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
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


def minority_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_pred == 1
    if mask.sum() == 0:
        return float("nan")
    return float(precision_score(y_true, y_pred, pos_label=1.0, zero_division=0))


def distribution_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    cv = float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float("nan")
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "cv": cv,
    }


def saturation_stats(reliabilities: np.ndarray) -> dict:
    if reliabilities.size == 0:
        return {}
    return {
        "frac_ge_0_99": float(np.mean(reliabilities >= 0.99)),
        "frac_ge_0_95": float(np.mean(reliabilities >= 0.95)),
        "frac_lt_0_95": float(np.mean(reliabilities < 0.95)),
        "frac_lt_0_90": float(np.mean(reliabilities < 0.90)),
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    runtime = time.perf_counter() - t0
    y_bin = np.where(y_test > 0, 1.0, -1.0)
    return {
        "accuracy": float(accuracy_score(y_bin, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_bin, preds)),
        "macro_f1": float(f1_score(y_bin, preds, average="macro")),
        "minority_recall": minority_recall(y_bin, preds),
        "minority_precision": minority_precision(y_bin, preds),
        "predict_runtime_sec": runtime,
    }


def ball_summary(balls) -> dict:
    pos = [b for b in balls if b.label == 1]
    neg = [b for b in balls if b.label != 1]
    return {
        "n_balls_pos": len(pos),
        "n_balls_neg": len(neg),
    }


def write_weight_diagnostics(
    scenario: str,
    model: CRGBLSTSVM,
    out_rows: list[dict],
) -> None:
    pos_idx = 0
    neg_idx = 0
    for i, ball in enumerate(model.balls_):
        if ball.label == 1:
            kappa = model.kappa_pos_balls_[pos_idx]
            final_w = model.weights_pos_[pos_idx]
            pos_idx += 1
        else:
            kappa = model.kappa_neg_balls_[neg_idx]
            final_w = model.weights_neg_[neg_idx]
            neg_idx += 1

        out_rows.append(
            {
                "scenario": scenario,
                "ball_index": i,
                "class": int(ball.label),
                "n_samples": int(ball.num_samples),
                "purity": float(ball.purity),
                "radius": float(ball.radius),
                "reliability": float(model.reliabilities_[i]),
                "kappa": float(kappa),
                "final_weight": float(final_w),
            }
        )


def compute_correlations(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = ["kappa", "reliability", "final_weight", "purity", "radius", "n_samples"]
    data = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    corr = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            if np.std(data[k1]) == 0 or np.std(data[k2]) == 0:
                corr[f"{k1}_vs_{k2}"] = float("nan")
            else:
                corr[f"{k1}_vs_{k2}"] = float(np.corrcoef(data[k1], data[k2])[0, 1])
    return corr


def cw_vs_cr_weight_comparison(model: CRGBLSTSVM) -> dict:
    """Compare CW (kappa only) vs CR (kappa * a_i) weight distributions."""
    result = {}
    for label, kappa_arr, final_arr in (
        ("pos", model.kappa_pos_balls_, model.weights_pos_),
        ("neg", model.kappa_neg_balls_, model.weights_neg_),
    ):
        if kappa_arr.size == 0:
            continue
        result[label] = {
            "cw_kappa": distribution_summary(kappa_arr),
            "cr_final": distribution_summary(final_arr),
        }
    return result


def run_scenario(name: str, n_minority: int, n_majority: int) -> dict:
    X, y = make_dataset(n_minority, n_majority, SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    base_params = {
        "c1": C1,
        "c2": C2,
        "purity_threshold": PURITY,
        "min_ball_count": MIN_BALL_COUNT,
        "prediction_rule": "reference",
        "random_state": SEED,
    }
    r_params = {
        **base_params,
        "epsilon": DEFAULT_EPSILON,
        "gamma": DEFAULT_GAMMA,
        "lambda_r": DEFAULT_LAMBDA_R,
    }

    models = {}
    fit_times = {}
    for label, cls, params in [
        ("GBLSTSVM", GBLSTSVM, base_params),
        ("CW-GBLSTSVM", CWGBLSTSVM, base_params),
        ("R-GBLSTSVM", RGBLSTSVM, r_params),
        ("CR-GBLSTSVM", CRGBLSTSVM, r_params),
    ]:
        t0 = time.perf_counter()
        models[label] = cls(**params).fit(X_train, y_train)
        fit_times[label] = time.perf_counter() - t0

    cr = models["CR-GBLSTSVM"]
    cw = models["CW-GBLSTSVM"]
    r_model = models["R-GBLSTSVM"]

    metrics = {
        label: evaluate_model(m, X_test, y_test) for label, m in models.items()
    }
    for label in metrics:
        metrics[label]["fit_runtime_sec"] = fit_times[label]

    cr_m = metrics["CR-GBLSTSVM"]
    cw_m = metrics["CW-GBLSTSVM"]
    r_m = metrics["R-GBLSTSVM"]

    delta_reliability = {
        "balanced_accuracy": cr_m["balanced_accuracy"] - cw_m["balanced_accuracy"],
        "macro_f1": cr_m["macro_f1"] - cw_m["macro_f1"],
        "minority_recall": cr_m["minority_recall"] - cw_m["minority_recall"],
    }
    delta_class = {
        "balanced_accuracy": cr_m["balanced_accuracy"] - r_m["balanced_accuracy"],
        "macro_f1": cr_m["macro_f1"] - r_m["macro_f1"],
        "minority_recall": cr_m["minority_recall"] - r_m["minority_recall"],
    }

    rel = cr.reliabilities_
    rel_pos = rel[[i for i, b in enumerate(cr.balls_) if b.label == 1]]
    rel_neg = rel[[i for i, b in enumerate(cr.balls_) if b.label != 1]]

    final_all = np.concatenate([cr.weights_pos_, cr.weights_neg_])

    record = {
        "scenario": name,
        "n_minority": n_minority,
        "n_majority": n_majority,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "class_weights": {"kappa_pos": cr.kappa_pos_, "kappa_neg": cr.kappa_neg_},
        "balls": ball_summary(cr.balls_),
        "radius_scale": cr.radius_scale_,
        "reliability_distribution": distribution_summary(rel),
        "reliability_saturation": saturation_stats(rel),
        "reliability_pos": {
            **distribution_summary(rel_pos),
            **saturation_stats(rel_pos),
        },
        "reliability_neg": {
            **distribution_summary(rel_neg),
            **saturation_stats(rel_neg),
        },
        "final_weight_distribution": distribution_summary(final_all),
        "cw_vs_cr_weights": cw_vs_cr_weight_comparison(cr),
        "metrics": metrics,
        "delta_reliability_cr_minus_cw": delta_reliability,
        "delta_class_cr_minus_r": delta_class,
    }
    return record, cr


def main() -> None:
    results = []
    diag_rows: list[dict] = []

    for name, n_min, n_maj in SCENARIOS:
        record, cr_model = run_scenario(name, n_min, n_maj)
        write_weight_diagnostics(name, cr_model, diag_rows)
        record["correlations"] = compute_correlations(
            [r for r in diag_rows if r["scenario"] == name]
        )
        results.append(record)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "validate_cr_gblstsvm.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = out_dir / "cr_weight_diagnostics.csv"
    if diag_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diag_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}\n")

    header = f"{'Model':<14} {'Acc':>6} {'BalAcc':>7} {'MacroF1':>8} {'MinRec':>7}"
    for row in results:
        print(f"=== {row['scenario']} ===")
        print(
            f"  kappa: pos={row['class_weights']['kappa_pos']:.4f}, "
            f"neg={row['class_weights']['kappa_neg']:.4f}"
        )
        print(
            f"  balls: pos={row['balls']['n_balls_pos']}, "
            f"neg={row['balls']['n_balls_neg']}"
        )
        rd = row["reliability_distribution"]
        print(
            f"  reliability: min={rd['min']:.3f}, mean={rd['mean']:.3f}, max={rd['max']:.3f}"
        )
        sat = row["reliability_saturation"]
        print(
            f"  saturation: >=0.99={sat['frac_ge_0_99']:.2%}, "
            f">=0.95={sat['frac_ge_0_95']:.2%}, "
            f"<0.95={sat['frac_lt_0_95']:.2%}"
        )
        print(header)
        for model_name in ("GBLSTSVM", "CW-GBLSTSVM", "R-GBLSTSVM", "CR-GBLSTSVM"):
            m = row["metrics"][model_name]
            print(
                f"  {model_name:<12} {m['accuracy']:6.3f} "
                f"{m['balanced_accuracy']:7.3f} {m['macro_f1']:8.3f} "
                f"{m['minority_recall']:7.3f}"
            )
        dr = row["delta_reliability_cr_minus_cw"]
        dc = row["delta_class_cr_minus_r"]
        print(
            f"  Delta_reliability (CR-CW): bal_acc={dr['balanced_accuracy']:+.4f}, "
            f"macro_f1={dr['macro_f1']:+.4f}, min_recall={dr['minority_recall']:+.4f}"
        )
        print(
            f"  Delta_class (CR-R): bal_acc={dc['balanced_accuracy']:+.4f}, "
            f"macro_f1={dc['macro_f1']:+.4f}, min_recall={dc['minority_recall']:+.4f}"
        )
        print()


if __name__ == "__main__":
    main()
