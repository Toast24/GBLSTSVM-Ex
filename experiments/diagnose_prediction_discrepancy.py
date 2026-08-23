"""
Diagnostic experiment: paper vs reference prediction rules on synthetic data.

Run from GBLSTSVM root:
  python experiments/diagnose_prediction_discrepancy.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.gblstsvm import GBLSTSVM
from src.prediction_diagnostics import (
    compute_prediction_diagnostics,
    diagnostics_to_records,
    summarize_disagreements,
)

OUTPUT_CSV = ROOT / "experiments" / "output" / "prediction_discrepancy_diagnostics.csv"


def build_synthetic_dataset(seed: int = 0, n: int = 100, d: int = 4) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    X = np.random.randn(n, d)
    y = np.sign(X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n))
    y[y == 0] = 1
    return X, y


def main() -> None:
    purity = 0.985
    min_ball_count = 2
    c1 = 1000.0
    c2 = 1e-5
    seed = 0

    X, y = build_synthetic_dataset(seed=seed)

    model = GBLSTSVM(
        c1=c1,
        c2=c2,
        purity_threshold=purity,
        min_ball_count=min_ball_count,
        prediction_rule="paper",
        random_state=seed,
    )
    model.fit(X, y)

    diag = compute_prediction_diagnostics(
        X, model.w1_, model.b1_, model.w2_, model.b2_
    )
    summary = summarize_disagreements(diag)
    records = diagnostics_to_records(diag)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "f1",
                "f2",
                "norm_w1",
                "norm_w2",
                "d1",
                "d2",
                "reference_prediction",
                "paper_prediction",
                "disagreement",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print("=== Prediction rule diagnostic (synthetic) ===")
    print(f"seed={seed}, n={X.shape[0]}, d={X.shape[1]}")
    print(f"purity={purity}, min_ball_count={min_ball_count}, c1={c1}, c2={c2}")
    print(f"CSV written to: {OUTPUT_CSV}")
    print()
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if summary["n_disagree"] > 0:
        print()
        print("Interpretation:")
        print(
            "  Reference: sign(|f2| - |f1|). Paper: +1 when |f1|/||w1|| <= |f2|/||w2||."
        )
        print(
            "  Disagreements occur when unnormalized |f1| vs |f2| ordering differs from"
        )
        print(
            "  normalized ordering, or when |f1|==|f2| (reference yields sign(0)=0)."
        )
        print(f"  ||w1||/||w2|| = {summary['norm_ratio_w1_over_w2']:.6f}")


if __name__ == "__main__":
    main()
