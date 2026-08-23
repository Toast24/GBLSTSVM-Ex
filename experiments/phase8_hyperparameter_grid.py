"""Phase 8 Experiment 3: Hyperparameter grid (validation selection, top-5 test)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase8_common import (
    CLEAN_MODELS,
    DATASETS,
    DEFAULT_SEED,
    ExperimentConfig,
    ensure_out_dir,
    fit_model,
    fit_scaler,
    git_status_snapshot,
    load_dataset_split,
    predict_and_evaluate,
    save_csv,
    save_json,
)

C1_GRID = [0.01, 0.1, 1, 10, 100, 1000]
C2_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
TOP_K = 5


def main() -> None:
    out = ensure_out_dir()
    rows: list[dict] = []
    payload = {
        "experiment": "hyperparameter_grid",
        "c1_grid": C1_GRID,
        "c2_grid": C2_GRID,
        "top_k_test": TOP_K,
        "seed": DEFAULT_SEED,
        "git": git_status_snapshot(),
        "datasets": {},
    }

    for ds_name in DATASETS:
        split = load_dataset_split(ds_name, DEFAULT_SEED)
        ds_entry = {"info": split.info, "models": {}}

        for model_name in CLEAN_MODELS:
            val_results: list[dict] = []
            for c1 in C1_GRID:
                for c2 in C2_GRID:
                    config = ExperimentConfig(c1=c1, c2=c2, seed=DEFAULT_SEED)
                    X_train, X_val, X_test, _ = fit_scaler(
                        config.scaling, split.X_train, split.X_val, split.X_test
                    )
                    try:
                        model = fit_model(model_name, X_train, split.y_train, config)
                        val_m, _, _ = predict_and_evaluate(
                            model, model_name, X_val, split.y_val, split.minority_label
                        )
                    except Exception as exc:  # noqa: BLE001
                        val_m = {
                            "macro_f1": -1.0,
                            "balanced_accuracy": -1.0,
                            "minority_recall": -1.0,
                            "error": str(exc),
                        }
                    rec = {
                        "c1": c1,
                        "c2": c2,
                        "val_macro_f1": val_m.get("macro_f1"),
                        "val_balanced_accuracy": val_m.get("balanced_accuracy"),
                        "val_minority_recall": val_m.get("minority_recall"),
                        "val_accuracy": val_m.get("accuracy"),
                        "val_gmean": val_m.get("gmean"),
                        "val_auroc": val_m.get("auroc"),
                    }
                    val_results.append(rec)
                    rows.append({
                        "dataset": split.name,
                        "model": model_name,
                        "phase": "validation_grid",
                        "c1": c1,
                        "c2": c2,
                        **rec,
                    })

            val_results.sort(key=lambda r: (r["val_macro_f1"] or -1), reverse=True)
            top_configs = val_results[:TOP_K]
            test_results = []
            for rank, cfg in enumerate(top_configs, start=1):
                c1, c2 = cfg["c1"], cfg["c2"]
                config = ExperimentConfig(c1=c1, c2=c2, seed=DEFAULT_SEED)
                X_train, X_val, X_test, _ = fit_scaler(
                    config.scaling, split.X_train, split.X_val, split.X_test
                )
                try:
                    model = fit_model(model_name, X_train, split.y_train, config)
                    test_m, _, _ = predict_and_evaluate(
                        model, model_name, X_test, split.y_test, split.minority_label
                    )
                except Exception as exc:  # noqa: BLE001
                    test_m = {"error": str(exc)}
                test_rec = {
                    "rank": rank,
                    "c1": c1,
                    "c2": c2,
                    "val_macro_f1": cfg["val_macro_f1"],
                    "val_balanced_accuracy": cfg["val_balanced_accuracy"],
                    "val_minority_recall": cfg["val_minority_recall"],
                    "test_macro_f1": test_m.get("macro_f1"),
                    "test_balanced_accuracy": test_m.get("balanced_accuracy"),
                    "test_minority_recall": test_m.get("minority_recall"),
                    "test_accuracy": test_m.get("accuracy"),
                    "test_gmean": test_m.get("gmean"),
                    "test_auroc": test_m.get("auroc"),
                }
                test_results.append(test_rec)
                rows.append({
                    "dataset": split.name,
                    "model": model_name,
                    "phase": "test_top_k",
                    "selected_rank": rank,
                    **test_rec,
                })

            ds_entry["models"][model_name] = {
                "validation_grid_count": len(val_results),
                "top_val_configs": top_configs,
                "test_evaluations": test_results,
                "best_val_config": top_configs[0] if top_configs else None,
            }

        payload["datasets"][split.name] = ds_entry

    save_json(out / "hyperparameter_grid.json", payload)
    save_csv(out / "hyperparameter_grid.csv", rows)
    print(f"Wrote {out / 'hyperparameter_grid.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
