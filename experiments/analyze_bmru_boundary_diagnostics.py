"""
BMRU boundary-straddling falsification diagnostic.

Tests whether high u corresponds to empirical boundary-near / straddling balls.
Does not modify BMRU formulation or frozen baseline implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.phase8_common import DATASETS, load_dataset_split, save_csv, save_json
from experiments.validate_bmru import (
    ALPHA,
    BMRU_EPSILON,
    C1,
    C2,
    MAX_ITERATIONS,
    MIN_BALL_COUNT,
    PURITY,
    SEED,
    fit_bmru_model,
)
from src.bmru import (
    ball_bmru_weights,
    boundary_correlations,
    boundary_u_detector_confusion,
    per_ball_boundary_records,
    summarize_boundary_groups,
)
from src.cw_gblstsvm import solve_cw_gblstsvm_systems
from src.gblstsvm import solve_gblstsvm_systems
from src.granular_balls import (
    balls_to_class_arrays,
    generate_granular_balls_with_membership,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
MARGIN_MODES = ["reference", "paper"]
PRIMARY_MODEL = "BMRU-GBLSTSVM"
ITERATION_SNAPSHOTS = (1, 5)


def _dataset_summary_row(
    per_ball_rows: list[dict],
    *,
    dataset: str,
    model: str,
    margin_mode: str,
    iteration: int | None = None,
) -> dict:
    n = len(per_ball_rows)
    u_vals = np.array([r["u"] for r in per_ball_rows], dtype=float)
    corr = boundary_correlations(per_ball_rows)
    row = {
        "dataset": dataset,
        "model": model,
        "margin_mode": margin_mode,
        "iteration": iteration if iteration is not None else MAX_ITERATIONS,
        "n_balls": n,
        "mean_u": float(np.mean(u_vals)) if n else float("nan"),
        "frac_u_ge_1": float(np.mean(u_vals >= 1.0)) if n else float("nan"),
        "fraction_boundary_straddling": float(
            np.mean([r["boundary_straddles"] for r in per_ball_rows])
        )
        if n
        else float("nan"),
        "fraction_boundary_near": float(
            np.mean([r["boundary_near"] for r in per_ball_rows])
        )
        if n
        else float("nan"),
        "mean_max_radius_ratio": float(
            np.mean([r["max_radius_ratio"] for r in per_ball_rows])
        )
        if n
        else float("nan"),
        **corr,
    }
    return row


def _iteration_boundary_snapshots(
    X_train: np.ndarray,
    y_train: np.ndarray,
    margin_mode: str,
    target_iterations: tuple[int, ...],
) -> list[tuple[int, np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray]]:
    """Return (iteration, w1, b1, w2, b2, u_all, mu_all) at BMRU weight evaluation steps."""
    balls, memberships = generate_granular_balls_with_membership(
        X_train,
        y_train,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )
    centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(balls)

    w1, b1, w2, b2 = solve_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        C1,
        C2,
        reg_term=1e-4,
    )

    snapshots: list[
        tuple[int, np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray]
    ] = []
    weight_placement = "opposite_only"

    for it in range(1, MAX_ITERATIONS + 1):
        pos, neg, u_all, mu_all, _ = ball_bmru_weights(
            balls,
            w1,
            b1,
            w2,
            b2,
            alpha=ALPHA,
            epsilon=BMRU_EPSILON,
            margin_mode=margin_mode,
        )

        if it in target_iterations:
            snapshots.append(
                (it, w1.copy(), b1, w2.copy(), b2, u_all.copy(), mu_all.copy())
            )

        w1, b1, w2, b2 = solve_cw_gblstsvm_systems(
            centers_pos,
            radii_pos,
            centers_neg,
            radii_neg,
            C1,
            C2,
            pos,
            neg,
            reg_term=1e-4,
            weight_placement=weight_placement,
        )

    return snapshots


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_ball_rows: list[dict] = []
    summary_rows: list[dict] = []
    confusion_rows: list[dict] = []
    group_rows: list[dict] = []
    payload: dict = {"experiment": "bmru_boundary_diagnostic", "datasets": {}}

    for ds_file in DATASETS:
        split = load_dataset_split(ds_file, SEED)
        ds_payload: dict = {}

        for margin_mode in MARGIN_MODES:
            model = fit_bmru_model(PRIMARY_MODEL, split.X_train, split.y_train, margin_mode)

            balls, memberships = generate_granular_balls_with_membership(
                split.X_train,
                split.y_train,
                purity_threshold=PURITY,
                min_samples_per_ball=1,
                min_ball_count=MIN_BALL_COUNT,
            )

            meta = {
                "dataset": split.name,
                "model": PRIMARY_MODEL,
                "margin_mode": margin_mode,
                "iteration": MAX_ITERATIONS,
                "seed": SEED,
            }

            ball_rows = per_ball_boundary_records(
                balls,
                memberships,
                split.X_train,
                model.w1_,
                model.b1_,
                model.w2_,
                model.b2_,
                model.u_all_,
                model.mu_all_,
                margin_mode=margin_mode,
                epsilon=BMRU_EPSILON,
                metadata=meta,
            )
            per_ball_rows.extend(ball_rows)

            summary_rows.append(
                _dataset_summary_row(
                    ball_rows,
                    dataset=split.name,
                    model=PRIMARY_MODEL,
                    margin_mode=margin_mode,
                )
            )

            for group in summarize_boundary_groups(ball_rows):
                group_rows.append({**meta, **group})

            for empirical_field, empirical_label in (
                ("boundary_straddles", "boundary_straddles"),
                ("boundary_near", "boundary_near"),
            ):
                confusion_rows.append(
                    {
                        **meta,
                        **boundary_u_detector_confusion(
                            ball_rows,
                            empirical_field=empirical_field,
                            empirical_label=empirical_label,
                        ),
                    }
                )

            margin_payload: dict = {
                "summary": summary_rows[-1],
                "groups": [g for g in group_rows if g["margin_mode"] == margin_mode],
                "confusion": [
                    c for c in confusion_rows if c["margin_mode"] == margin_mode
                ],
            }

            if split.name == "diabetes" and margin_mode == "reference":
                iter_payload = []
                for it, w1, b1, w2, b2, u_all, mu_all in _iteration_boundary_snapshots(
                    split.X_train,
                    split.y_train,
                    margin_mode,
                    ITERATION_SNAPSHOTS,
                ):
                    iter_meta = {**meta, "iteration": it}
                    iter_ball_rows = per_ball_boundary_records(
                        balls,
                        memberships,
                        split.X_train,
                        w1,
                        b1,
                        w2,
                        b2,
                        u_all,
                        mu_all,
                        margin_mode=margin_mode,
                        epsilon=BMRU_EPSILON,
                        metadata=iter_meta,
                    )
                    per_ball_rows.extend(iter_ball_rows)
                    summary_rows.append(
                        _dataset_summary_row(
                            iter_ball_rows,
                            dataset=split.name,
                            model=PRIMARY_MODEL,
                            margin_mode=margin_mode,
                            iteration=it,
                        )
                    )
                    iter_payload.append(
                        {
                            "iteration": it,
                            "summary": summary_rows[-1],
                            "mean_u": summary_rows[-1]["mean_u"],
                            "fraction_boundary_near": summary_rows[-1][
                                "fraction_boundary_near"
                            ],
                            "fraction_boundary_straddling": summary_rows[-1][
                                "fraction_boundary_straddling"
                            ],
                            "corr_u_min_abs_sample_margin": summary_rows[-1][
                                "corr_u_min_abs_sample_margin"
                            ],
                        }
                    )
                margin_payload["iteration_snapshots"] = iter_payload

            ds_payload[margin_mode] = margin_payload

        payload["datasets"][split.name] = ds_payload

    save_csv(OUT_DIR / "bmru_boundary_per_ball.csv", per_ball_rows)
    save_csv(OUT_DIR / "bmru_boundary_summary.csv", summary_rows)
    save_csv(OUT_DIR / "bmru_boundary_confusion.csv", confusion_rows)
    save_json(
        OUT_DIR / "bmru_boundary_diagnostics.json",
        {
            "experiment": payload["experiment"],
            "summary": summary_rows,
            "groups": group_rows,
            "confusion": confusion_rows,
            "datasets": payload["datasets"],
        },
    )

    return payload


def main() -> None:
    run()
    print(f"Wrote {OUT_DIR / 'bmru_boundary_per_ball.csv'}")
    print(f"Wrote {OUT_DIR / 'bmru_boundary_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'bmru_boundary_confusion.csv'}")


if __name__ == "__main__":
    main()
