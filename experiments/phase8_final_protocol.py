"""Phase 8 Experiment 4: Combined best-protocol (validation selection, single test eval)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.phase8_common import (
    ALL_MODELS,
    CLEAN_MODELS,
    DATASETS,
    DEFAULT_MIN_BALL_COUNT,
    DEFAULT_SEED,
    ExperimentConfig,
    delta_metrics,
    ensure_out_dir,
    fit_model,
    fit_original,
    fit_scaler,
    git_status_snapshot,
    load_dataset_split,
    predict_and_evaluate,
    run_clean_family,
    save_csv,
    save_json,
)

PURITY_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98]
SCALING_OPTIONS = ["none", "standard", "minmax"]


def load_top_c1_c2(out_dir: Path, dataset: str, k: int = 5) -> list[tuple[float, float]]:
    hp_path = out_dir / "hyperparameter_grid.json"
    if not hp_path.exists():
        return [(100.0, 1e-2)]
    data = json.loads(hp_path.read_text(encoding="utf-8"))
    model_data = data["datasets"].get(dataset, {}).get("models", {}).get("Clean GBLSTSVM", {})
    top = model_data.get("top_val_configs", [])
    pairs = []
    for rec in top[:k]:
        pairs.append((float(rec["c1"]), float(rec["c2"])))
    if not pairs:
        pairs = [(100.0, 1e-2)]
    return pairs


def select_protocol(split_name: str, out_dir: Path) -> dict:
    split = load_dataset_split(f"{split_name}.csv", DEFAULT_SEED)
    c1_c2_pairs = load_top_c1_c2(out_dir, split.name)
    candidates = []
    for scaling in SCALING_OPTIONS:
        for purity in PURITY_THRESHOLDS:
            for c1, c2 in c1_c2_pairs:
                config = ExperimentConfig(
                    scaling=scaling,
                    purity_threshold=purity,
                    c1=c1,
                    c2=c2,
                    seed=DEFAULT_SEED,
                )
                X_train, X_val, X_test, _ = fit_scaler(
                    config.scaling, split.X_train, split.X_val, split.X_test
                )
                try:
                    model = fit_model("Clean GBLSTSVM", X_train, split.y_train, config)
                    val_m, _, _ = predict_and_evaluate(
                        model, "Clean GBLSTSVM", X_val, split.y_val, split.minority_label
                    )
                    macro_f1 = val_m["macro_f1"]
                except Exception:  # noqa: BLE001
                    macro_f1 = -1.0
                    val_m = {}
                candidates.append({
                    "scaling": scaling,
                    "purity_threshold": purity,
                    "c1": c1,
                    "c2": c2,
                    "val_macro_f1_clean": macro_f1,
                    "val_metrics": val_m,
                })
    candidates.sort(key=lambda c: c["val_macro_f1_clean"], reverse=True)
    best = candidates[0]
    return {
        "dataset": split.name,
        "split": split,
        "selected_protocol": best,
        "candidate_count": len(candidates),
        "top_5_candidates": candidates[:5],
    }


def run_final_protocol(selection: dict) -> dict:
    split = selection["split"]
    sel = selection["selected_protocol"]
    config = ExperimentConfig(
        scaling=sel["scaling"],
        purity_threshold=sel["purity_threshold"],
        c1=sel["c1"],
        c2=sel["c2"],
        min_ball_count=DEFAULT_MIN_BALL_COUNT,
        seed=DEFAULT_SEED,
    )

    X_train_full = np.vstack([split.X_train, split.X_val])
    y_train_full = np.concatenate([split.y_train, split.y_val])

    X_tr, _, X_test, scaler = fit_scaler(
        config.scaling, X_train_full, split.X_val, split.X_test
    )
    if scaler is not None:
        config_fitted_scaling = config.scaling
    else:
        config_fitted_scaling = "none"

    results: dict = {
        "selected_protocol": sel,
        "config": {
            "scaling": config_fitted_scaling,
            "purity_threshold": config.purity_threshold,
            "c1": config.c1,
            "c2": config.c2,
            "min_ball_count": config.min_ball_count,
            "seed": config.seed,
            "train_n": int(len(y_train_full)),
            "test_n": int(len(split.y_test)),
            "note": "Trained on train+validation; evaluated once on untouched test.",
        },
        "models": {},
        "deltas_test": {},
    }

    orig = fit_original(X_tr, y_train_full, config)
    if orig.supported:
        test_m, test_pred, test_scores = predict_and_evaluate(
            orig, "Original/reference GBLSTSVM", X_test, split.y_test, split.minority_label
        )
        results["models"]["Original/reference GBLSTSVM"] = {
            "test": test_m,
            "ball_stats": {
                "n_balls_total": len(orig.balls_) if orig.balls_ else 0,
            },
        }
    else:
        results["models"]["Original/reference GBLSTSVM"] = {
            "skipped": True,
            "reason": orig.skip_reason,
        }

    fitted = {}
    for name in CLEAN_MODELS:
        model = fit_model(name, X_tr, y_train_full, config)
        fitted[name] = model
        test_m, test_pred, _ = predict_and_evaluate(
            model, name, X_test, split.y_test, split.minority_label
        )
        from experiments.phase8_common import model_ball_and_reliability_stats

        results["models"][name] = {
            "test": test_m,
            "ball_stats": model_ball_and_reliability_stats(name, model),
            "test_predictions": test_pred.tolist(),
        }

    clean_test = results["models"]["Clean GBLSTSVM"]["test"]
    cw_test = results["models"]["CW-GBLSTSVM"]["test"]
    r_test = results["models"]["R-GBLSTSVM"]["test"]
    cr_test = results["models"]["CR-GBLSTSVM"]["test"]
    results["deltas_test"] = {
        "CW_minus_clean": delta_metrics(cw_test, clean_test),
        "R_minus_clean": delta_metrics(r_test, clean_test),
        "CR_minus_clean": delta_metrics(cr_test, clean_test),
        "CR_minus_CW": delta_metrics(cr_test, cw_test),
        "CR_minus_R": delta_metrics(cr_test, r_test),
    }
    results["fitted"] = fitted
    results["X_test_scaled"] = X_test
    results["split_ref"] = split
    return results


def main() -> None:
    out = ensure_out_dir()
    rows: list[dict] = []
    payload = {
        "experiment": "final_protocol",
        "seed": DEFAULT_SEED,
        "git": git_status_snapshot(),
        "selection_criterion": "validation Macro F1 on Clean GBLSTSVM",
        "datasets": {},
    }

    for ds_name in DATASETS:
        ds_stem = ds_name.replace(".csv", "")
        selection = select_protocol(ds_stem, out)
        final = run_final_protocol(selection)
        split = selection["split"]
        sel = final["selected_protocol"]

        ds_payload = {
            "info": split.info,
            "label_mapping": split.label_mapping,
            "selection": {
                "selected_protocol": sel,
                "candidate_count": selection["candidate_count"],
                "top_5_candidates": selection["top_5_candidates"],
            },
            "config": final["config"],
            "models": {
                k: {kk: vv for kk, vv in v.items() if kk != "test_predictions"}
                for k, v in final["models"].items()
            },
            "deltas_test": final["deltas_test"],
        }
        payload["datasets"][split.name] = ds_payload

        for name in ALL_MODELS:
            m = final["models"].get(name, {})
            if "test" not in m:
                continue
            t = m["test"]
            rows.append({
                "dataset": split.name,
                "model": name,
                "scaling": sel["scaling"],
                "purity_threshold": sel["purity_threshold"],
                "c1": sel["c1"],
                "c2": sel["c2"],
                "val_macro_f1_clean": sel["val_macro_f1_clean"],
                "test_accuracy": t.get("accuracy"),
                "test_balanced_accuracy": t.get("balanced_accuracy"),
                "test_macro_f1": t.get("macro_f1"),
                "test_minority_recall": t.get("minority_recall"),
                "test_gmean": t.get("gmean"),
                "test_auroc": t.get("auroc"),
                "prediction_collapse": t.get("prediction_collapse"),
                **m.get("ball_stats", {}),
            })

        # stash fitted models for mechanism diagnostics
        stash_path = out / f"final_protocol_fitted_{split.name}.json"
        stash = {
            "dataset": split.name,
            "config": final["config"],
            "deltas_test": final["deltas_test"],
        }
        save_json(stash_path, stash)

    save_json(out / "final_protocol.json", payload)
    save_csv(out / "final_protocol.csv", rows)
    print(f"Wrote {out / 'final_protocol.json'}")


if __name__ == "__main__":
    main()
