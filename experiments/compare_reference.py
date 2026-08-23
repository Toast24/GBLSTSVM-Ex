"""
Deterministic synthetic experiment: reference vs clean GBLSTSVM.

Run from repository root:
  python experiments/compare_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.gblstsvm import GBLSTSVM, solve_gblstsvm_systems
from src.granular_balls import generate_granular_balls, balls_to_class_arrays
from src.reference_pipeline import reference_gen_balls, reference_predict_from_matrix
from src.reference_pipeline import balls_to_training_matrix


def main() -> None:
    np.random.seed(0)
    n, d = 100, 4
    X = np.random.randn(n, d)
    y = np.sign(X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n))
    y[y == 0] = 1

    purity = 0.985
    min_ball_count = 2
    c1 = 1000.0
    c2 = 1e-5

    print("=== GBLSTSVM reference vs clean reproduction ===")
    print(f"samples={n}, features={d}, purity={purity}, min_ball_count={min_ball_count}")
    print(f"c1={c1}, c2={c2}")

    np.random.seed(0)
    clean_balls = generate_granular_balls(
        X, y, purity_threshold=purity, min_ball_count=min_ball_count
    )
    np.random.seed(0)
    ref_balls = reference_gen_balls(X, y, purity, min_ball_count)

    print(f"\nBall count: clean={len(clean_balls)}, reference={len(ref_balls)}")

    cp, rp, cn, rn = balls_to_class_arrays(clean_balls)
    w1, b1, w2, b2 = solve_gblstsvm_systems(cp, rp, cn, rn, c1, c2)

    train_matrix = balls_to_training_matrix(ref_balls)
    ref_pred = reference_predict_from_matrix(train_matrix, X, c1, c2)

    model = GBLSTSVM(
        c1=c1,
        c2=c2,
        purity_threshold=purity,
        min_ball_count=min_ball_count,
        prediction_rule="reference",
        random_state=0,
    )
    model.fit(X, y)
    clean_pred_ref_rule = model.predict(X)

    model_paper = GBLSTSVM(
        c1=c1,
        c2=c2,
        purity_threshold=purity,
        min_ball_count=min_ball_count,
        prediction_rule="paper",
        random_state=0,
    )
    model_paper.fit(X, y)
    clean_pred_paper_rule = model_paper.predict(X)

    agree_ref = np.mean(clean_pred_ref_rule == ref_pred)
    agree_paper_vs_ref = np.mean(clean_pred_paper_rule == ref_pred)

    print(f"\nHyperplanes (clean): w1={w1}, b1={b1:.6f}")
    print(f"Hyperplanes (clean): w2={w2}, b2={b2:.6f}")
    print(f"\nPrediction agreement (reference rule): {agree_ref:.4f}")
    print(f"Prediction agreement (paper vs reference rule): {agree_paper_vs_ref:.4f}")

    max_center_diff = 0.0
    max_radius_diff = 0.0
    if len(clean_balls) != len(ref_balls):
        print("WARNING: ball counts differ; KMeans seeding may not match.")
    clean_sorted = sorted(
        clean_balls,
        key=lambda b: (b.label, b.center[0], b.radius),
    )
    ref_sorted = sorted(
        ref_balls,
        key=lambda b: (b.label, b.center[0], b.radius),
    )
    for cb, rb in zip(clean_sorted, ref_sorted):
        max_center_diff = max(max_center_diff, np.max(np.abs(cb.center - rb.center)))
        max_radius_diff = max(max_radius_diff, abs(cb.radius - rb.radius))

    print(f"Max |center diff| across balls: {max_center_diff:.2e}")
    print(f"Max |radius diff| across balls: {max_radius_diff:.2e}")


if __name__ == "__main__":
    main()
