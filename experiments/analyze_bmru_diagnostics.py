"""
BMRU diagnostic / ablation pass.

Runs reference vs paper margin modes with class-wise weight mass,
iteration convergence, prediction degeneration, and R/BMRU overlap stats.

Does not modify BMRU formulation or frozen baseline implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.phase8_common import (
    DATASETS,
    METRIC_KEYS,
    load_dataset_split,
    predict_and_evaluate,
    save_csv,
    save_json,
)
from experiments.validate_bmru import (
    ALPHA,
    BMRU_EPSILON,
    BMRU_MODELS,
    MAX_ITERATIONS,
    SEED,
    fit_bmru_model,
)
from src.bmru import (
    bmru_diagnostics_summary,
    class_weight_mass_diagnostics,
    combined_weights_per_ball,
    iteration_record_to_dict,
    margin_ablation_weight_stats,
    prediction_distribution,
    safe_correlation,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
MARGIN_MODES = ["reference", "paper"]
EVAL_SPLITS = ("val", "test")


def per_ball_records(
    model,
    dataset: str,
    model_name: str,
    margin_mode: str,
) -> list[dict]:
    rows = []
    reliabilities = getattr(model, "reliabilities_", None)
    combined = combined_weights_per_ball(model)
    pos_idx = neg_idx = 0

    for i, ball in enumerate(model.balls_):
        if hasattr(model, "kappa_pos_") and model.kappa_pos_ is not None:
            cw = float(model.kappa_pos_ if ball.label == 1 else model.kappa_neg_)
        else:
            cw = 1.0
        rel = float(reliabilities[i]) if reliabilities is not None else float("nan")
        if ball.label == 1:
            final_w = float(model.weights_pos_[pos_idx])
            pos_idx += 1
        else:
            final_w = float(model.weights_neg_[neg_idx])
            neg_idx += 1
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "margin_mode": margin_mode,
                "ball_id": i,
                "label": float(ball.label),
                "radius": float(ball.radius),
                "purity": float(ball.purity),
                "num_samples": int(ball.num_samples),
                "margin_s": float(model.margins_all_[i]),
                "u": float(model.u_all_[i]),
                "mu": float(model.mu_all_[i]),
                "class_weight": cw,
                "reliability_weight": rel,
                "final_training_weight": final_w,
                "combined_weight": float(combined[i]),
            }
        )
    return rows


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    margin_ablation_rows: list[dict] = []
    class_weight_rows: list[dict] = []
    iteration_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    per_ball_rows: list[dict] = []
    raw_rows: list[dict] = []
    payload: dict = {"experiment": "bmru_diagnostic_ablation", "datasets": {}}

    for ds_file in DATASETS:
        split = load_dataset_split(ds_file, SEED)
        ds_payload: dict = {}

        for margin_mode in MARGIN_MODES:
            margin_payload: dict = {}
            for name in BMRU_MODELS:
                model = fit_bmru_model(name, split.X_train, split.y_train, margin_mode)

                combined = combined_weights_per_ball(model)
                rel = getattr(model, "reliabilities_", None)
                cw = None
                if hasattr(model, "kappa_pos_balls_") and model.kappa_pos_balls_ is not None:
                    cw = np.concatenate([model.kappa_pos_balls_, model.kappa_neg_balls_])

                weight_stats = margin_ablation_weight_stats(model.u_all_, model.mu_all_)
                mechanism = bmru_diagnostics_summary(
                    model.balls_,
                    model.u_all_,
                    model.mu_all_,
                    reliabilities=rel,
                    class_weights=cw,
                )

                radii = np.array([b.radius for b in model.balls_], dtype=float)
                purities = np.array([b.purity for b in model.balls_], dtype=float)
                mu = model.mu_all_
                overlap = {
                    "corr_reliability_bmru_weight": safe_correlation(rel, mu)
                    if rel is not None
                    else float("nan"),
                    "corr_radius_bmru_weight": safe_correlation(radii, mu),
                    "corr_purity_bmru_weight": safe_correlation(purities, mu),
                    "corr_class_weight_bmru_weight": safe_correlation(cw, mu)
                    if cw is not None
                    else float("nan"),
                }

                split_metrics: dict = {}
                split_pred: dict = {}
                for eval_split in EVAL_SPLITS:
                    X = split.X_val if eval_split == "val" else split.X_test
                    y = split.y_val if eval_split == "val" else split.y_test
                    metrics, y_pred, _ = predict_and_evaluate(
                        model, name, X, y, split.minority_label
                    )
                    pred_dist = prediction_distribution(y_pred)
                    split_metrics[eval_split] = metrics
                    split_pred[eval_split] = pred_dist

                    raw_rows.append(
                        {
                            "dataset": split.name,
                            "model": name,
                            "margin_mode": margin_mode,
                            "split": eval_split,
                            "seed": SEED,
                            **{k: metrics.get(k) for k in METRIC_KEYS},
                            **pred_dist,
                        }
                    )

                    margin_ablation_rows.append(
                        {
                            "dataset": split.name,
                            "model": name,
                            "margin_mode": margin_mode,
                            "split": eval_split,
                            "seed": SEED,
                            **weight_stats,
                            **{k: metrics.get(k) for k in METRIC_KEYS},
                            **pred_dist,
                        }
                    )

                for cls_row in class_weight_mass_diagnostics(
                    model.balls_, model.mu_all_, combined
                ):
                    class_weight_rows.append(
                        {
                            "dataset": split.name,
                            "model": name,
                            "margin_mode": margin_mode,
                            **cls_row,
                        }
                    )

                for rec in model.iteration_history_ or []:
                    iteration_rows.append(
                        {
                            "dataset": split.name,
                            "model": name,
                            "margin_mode": margin_mode,
                            **iteration_record_to_dict(rec),
                        }
                    )

                mechanism_row = {
                    "dataset": split.name,
                    "model": name,
                    "margin_mode": margin_mode,
                    **mechanism,
                    **overlap,
                    "n_iterations": len(model.iteration_history_ or []),
                    "test_macro_f1": split_metrics["test"].get("macro_f1"),
                    "test_gmean": split_metrics["test"].get("gmean"),
                    "test_balanced_accuracy": split_metrics["test"].get(
                        "balanced_accuracy"
                    ),
                    "test_minority_recall": split_metrics["test"].get("minority_recall"),
                    "test_auroc": split_metrics["test"].get("auroc"),
                    "test_n_positive_predictions": split_pred["test"][
                        "n_positive_predictions"
                    ],
                    "test_n_negative_predictions": split_pred["test"][
                        "n_negative_predictions"
                    ],
                    "test_positive_prediction_fraction": split_pred["test"][
                        "positive_prediction_fraction"
                    ],
                    "test_prediction_collapse": split_pred["test"]["prediction_collapse"],
                }
                mechanism_rows.append(mechanism_row)
                per_ball_rows.extend(
                    per_ball_records(model, split.name, name, margin_mode)
                )

                margin_payload[name] = {
                    "weight_stats": weight_stats,
                    "overlap": overlap,
                    "splits": split_metrics,
                    "prediction": split_pred,
                    "mechanism": mechanism_row,
                    "iteration_history": [
                        iteration_record_to_dict(r)
                        for r in (model.iteration_history_ or [])
                    ],
                }

            ds_payload[margin_mode] = margin_payload
        payload["datasets"][split.name] = ds_payload

    save_json(OUT_DIR / "bmru_margin_ablation.json", payload)
    save_csv(OUT_DIR / "bmru_margin_ablation.csv", margin_ablation_rows)
    save_csv(OUT_DIR / "bmru_class_weight_diagnostics.csv", class_weight_rows)
    save_csv(OUT_DIR / "bmru_iteration_diagnostics.csv", iteration_rows)
    save_csv(OUT_DIR / "bmru_mechanism_diagnostics.csv", mechanism_rows)
    save_json(OUT_DIR / "bmru_mechanism_diagnostics.json", {"rows": mechanism_rows})
    save_csv(OUT_DIR / "bmru_per_ball_raw.csv", per_ball_rows)
    save_csv(OUT_DIR / "bmru_eval_raw.csv", raw_rows)

    # Backward-compatible aliases
    save_csv(OUT_DIR / "analyze_bmru_diagnostics.csv", mechanism_rows)
    save_json(OUT_DIR / "analyze_bmru_diagnostics.json", payload)
    save_csv(OUT_DIR / "analyze_bmru_per_ball.csv", per_ball_rows)

    return payload


def main() -> None:
    run()
    print(f"Wrote {OUT_DIR / 'bmru_margin_ablation.csv'}")
    print(f"Wrote {OUT_DIR / 'bmru_class_weight_diagnostics.csv'}")
    print(f"Wrote {OUT_DIR / 'bmru_iteration_diagnostics.csv'}")
    print(f"Wrote {OUT_DIR / 'bmru_mechanism_diagnostics.csv'}")


if __name__ == "__main__":
    main()
