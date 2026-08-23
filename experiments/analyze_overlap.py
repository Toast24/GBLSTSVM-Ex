"""
Overlap (OBRU) diagnostic analysis on Phase 7 benchmark datasets.

Reports overlap statistics, class-conditional mu, and correlations with
purity, radius, CW, R, and CR weights. Does not modify frozen implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.phase8_common import DATASETS, load_dataset_split, save_csv, save_json
from src.class_weights import ball_class_weights, compute_binary_class_weights
from src.cr_gblstsvm import ball_cr_weights
from src.overlap import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_TOP_K,
    ball_obru_weights,
    minority_label_from_training,
    overlap_diagnostics_summary,
)
from src.reliability import (
    DEFAULT_EPSILON as REL_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_R,
    ball_reliability_weights,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
SEED = 42
C1 = 100.0
C2 = 1e-2
PURITY = 0.95
MIN_BALL_COUNT = 2


def fit_balls(X_train, y_train):
    from src.granular_balls import generate_granular_balls

    np.random.seed(SEED)
    return generate_granular_balls(
        X_train,
        y_train,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
    )


def per_ball_records(
    dataset: str,
    balls,
    minority_label: float,
    O_all: np.ndarray,
    mu_all: np.ndarray,
    kappa_pos: float,
    kappa_neg: float,
    reliabilities: np.ndarray | None,
    cr_weights_all: np.ndarray | None,
) -> list[dict]:
    k_pos_balls, k_neg_balls = ball_class_weights(balls, kappa_pos, kappa_neg)
    pos_idx = neg_idx = 0
    rows = []
    for i, ball in enumerate(balls):
        if ball.label == 1:
            cw = float(k_pos_balls[pos_idx])
            cr_w = float(cr_weights_all[i]) if cr_weights_all is not None else float("nan")
            pos_idx += 1
        else:
            cw = float(k_neg_balls[neg_idx])
            cr_w = float(cr_weights_all[i]) if cr_weights_all is not None else float("nan")
            neg_idx += 1
        rel = float(reliabilities[i]) if reliabilities is not None else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "ball_id": i,
                "label": float(ball.label),
                "is_minority": int(ball.label == minority_label),
                "radius": float(ball.radius),
                "purity": float(ball.purity),
                "num_samples": int(ball.num_samples),
                "overlap": float(O_all[i]),
                "mu": float(mu_all[i]),
                "cw_weight": cw,
                "r_weight": rel,
                "cr_weight": cr_w,
            }
        )
    return rows


def combined_cr_weights_per_ball(balls, kappa_pos, kappa_neg) -> np.ndarray:
    w_pos, w_neg, _, _, _, _ = ball_cr_weights(
        balls,
        kappa_pos,
        kappa_neg,
        epsilon=REL_EPSILON,
        gamma=DEFAULT_GAMMA,
        lambda_r=DEFAULT_LAMBDA_R,
        variant="full",
    )
    out = np.zeros(len(balls), dtype=float)
    pos_idx = neg_idx = 0
    for i, ball in enumerate(balls):
        if ball.label == 1:
            out[i] = w_pos[pos_idx]
            pos_idx += 1
        else:
            out[i] = w_neg[neg_idx]
            neg_idx += 1
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    corr_rows: list[dict] = []
    per_ball_rows: list[dict] = []

    for ds_file in DATASETS:
        split = load_dataset_split(ds_file, SEED)
        balls = fit_balls(split.X_train, split.y_train)
        minority = minority_label_from_training(split.y_train)
        kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(split.y_train)

        _, _, O_all, mu_all = ball_obru_weights(
            balls,
            minority,
            alpha=DEFAULT_ALPHA,
            epsilon=DEFAULT_EPSILON,
            top_k=DEFAULT_TOP_K,
        )

        k_pos_balls, k_neg_balls = ball_class_weights(balls, kappa_pos, kappa_neg)
        cw_all = np.zeros(len(balls))
        rel_pos, rel_neg, _, reliabilities = ball_reliability_weights(
            balls,
            epsilon=REL_EPSILON,
            gamma=DEFAULT_GAMMA,
            lambda_r=DEFAULT_LAMBDA_R,
            variant="full",
        )
        cr_all = combined_cr_weights_per_ball(balls, kappa_pos, kappa_neg)

        pos_idx = neg_idx = 0
        for i, ball in enumerate(balls):
            if ball.label == 1:
                cw_all[i] = k_pos_balls[pos_idx]
                pos_idx += 1
            else:
                cw_all[i] = k_neg_balls[neg_idx]
                neg_idx += 1

        diag = overlap_diagnostics_summary(
            balls,
            O_all,
            mu_all,
            minority,
            cw_weights=cw_all,
            rel_weights=reliabilities,
            cr_weights=cr_all,
        )

        summary = diag["summary"]
        summary_row = {"dataset": split.name, **summary}
        for key in ("radius", "overlap"):
            for stat in ("mean", "median", "min", "max"):
                summary_row[f"{key}_{stat}"] = summary[key].get(stat, float("nan"))
        summary_rows.append(summary_row)

        corr_row = {"dataset": split.name, **diag["correlations"]}
        corr_rows.append(corr_row)

        per_ball_rows.extend(
            per_ball_records(
                split.name,
                balls,
                minority,
                O_all,
                mu_all,
                kappa_pos,
                kappa_neg,
                reliabilities,
                cr_all,
            )
        )

    save_csv(OUT_DIR / "overlap_per_ball.csv", per_ball_rows)
    save_csv(OUT_DIR / "overlap_summary.csv", summary_rows)
    save_csv(OUT_DIR / "overlap_correlations.csv", corr_rows)
    save_json(
        OUT_DIR / "overlap_analysis.json",
        {
            "protocol": {
                "seed": SEED,
                "c1": C1,
                "c2": C2,
                "purity_threshold": PURITY,
                "min_ball_count": MIN_BALL_COUNT,
                "alpha": DEFAULT_ALPHA,
                "epsilon": DEFAULT_EPSILON,
                "top_k": DEFAULT_TOP_K,
            },
            "summary": summary_rows,
            "correlations": corr_rows,
        },
    )

    print(f"Wrote {OUT_DIR / 'overlap_per_ball.csv'}")
    print(f"Wrote {OUT_DIR / 'overlap_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'overlap_correlations.csv'}")


if __name__ == "__main__":
    run()
