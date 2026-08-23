"""
Sanity-check script for CW-GBLSTSVM on synthetic imbalanced data.

Not a tuned research experiment — records class weights, ball statistics,
and basic metrics at several imbalance ratios.
"""

from __future__ import annotations

import json
import sys
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

from src.cw_gblstsvm import CWGBLSTSVM
from src.gblstsvm import GBLSTSVM

SEED = 42
N_SAMPLES = 200
N_FEATURES = 6
MINORITY_FRACS = [0.5, 1 / 4, 1 / 6, 1 / 11]  # ~1:1, 1:3, 1:5, 1:10
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


def evaluate_variant(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    rule: str,
) -> dict:
    model.prediction_rule = rule
    preds = model.predict(X)
    y_bin = np.where(y > 0, 1.0, -1.0)
    return {
        "variant": name,
        "prediction_rule": rule,
        "accuracy": float(accuracy_score(y_bin, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_bin, preds)),
        "macro_f1": float(f1_score(y_bin, preds, average="macro")),
        "minority_recall": minority_recall(y_bin, preds),
    }


def ball_summary(balls) -> dict:
    pos = [b for b in balls if b.label == 1]
    neg = [b for b in balls if b.label != 1]
    return {
        "n_balls_pos": len(pos),
        "n_balls_neg": len(neg),
        "purities_pos": [float(b.purity) for b in pos],
        "purities_neg": [float(b.purity) for b in neg],
        "ball_sizes_pos": [int(b.num_samples) for b in pos],
        "ball_sizes_neg": [int(b.num_samples) for b in neg],
    }


def run_ratio(frac_minority: float) -> dict:
    n_minority = max(5, int(round(N_SAMPLES * frac_minority)))
    n_majority = N_SAMPLES - n_minority
    ratio_label = f"1:{n_majority // max(n_minority, 1)}"

    X, y = make_dataset(n_minority, n_majority, seed=SEED)

    cw = CWGBLSTSVM(
        c1=C1,
        c2=C2,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
        prediction_rule="reference",
        random_state=SEED,
    )
    cw.fit(X, y)

    base = GBLSTSVM(
        c1=C1,
        c2=C2,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
        prediction_rule="reference",
        random_state=SEED,
    )
    base.fit(X, y)

    record = {
        "approx_ratio": ratio_label,
        "n_minority": n_minority,
        "n_majority": n_majority,
        "class_counts": {"pos": cw.n_pos_, "neg": cw.n_neg_},
        "class_weights": {"kappa_pos": cw.kappa_pos_, "kappa_neg": cw.kappa_neg_},
        "balls": ball_summary(cw.balls_),
        "metrics": {
            "cw_reference": evaluate_variant("CW-GBLSTSVM", cw, X, y, "reference"),
            "cw_paper": evaluate_variant("CW-GBLSTSVM", cw, X, y, "paper"),
            "baseline_reference": evaluate_variant("GBLSTSVM", base, X, y, "reference"),
        },
    }
    return record


def main() -> None:
    results = []
    for frac in MINORITY_FRACS:
        results.append(run_ratio(frac))

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validate_class_weighting.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}\n")
    for row in results:
        print(f"=== ratio ~{row['approx_ratio']} ===")
        print(f"  counts: pos={row['class_counts']['pos']}, neg={row['class_counts']['neg']}")
        print(
            f"  kappa: pos={row['class_weights']['kappa_pos']:.4f}, "
            f"neg={row['class_weights']['kappa_neg']:.4f}"
        )
        print(
            f"  balls: pos={row['balls']['n_balls_pos']}, "
            f"neg={row['balls']['n_balls_neg']}"
        )
        m = row["metrics"]["cw_reference"]
        print(
            f"  CW (reference): acc={m['accuracy']:.3f}, "
            f"bal_acc={m['balanced_accuracy']:.3f}, "
            f"macro_f1={m['macro_f1']:.3f}, "
            f"min_recall={m['minority_recall']:.3f}"
        )


if __name__ == "__main__":
    main()
