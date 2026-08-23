"""Phase 8 Experiment 6: Mechanism diagnostics on final protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.phase8_common import (
    CLEAN_MODELS,
    DATASETS,
    DEFAULT_MIN_BALL_COUNT,
    ExperimentConfig,
    delta_metrics,
    ensure_out_dir,
    fit_model,
    fit_scaler,
    git_status_snapshot,
    load_dataset_split,
    mechanism_diagnostics,
    model_ball_and_reliability_stats,
    predict_and_evaluate,
    save_csv,
    save_json,
)

DEFAULT_SEED = 42


def load_selected_protocol(out_dir: Path, dataset: str) -> dict:
    fp = out_dir / "final_protocol.json"
    data = json.loads(fp.read_text(encoding="utf-8"))
    sel = data["datasets"][dataset]["selection"]["selected_protocol"]
    return {
        "scaling": sel["scaling"],
        "purity_threshold": sel["purity_threshold"],
        "c1": sel["c1"],
        "c2": sel["c2"],
    }


def main() -> None:
    out = ensure_out_dir()
    fp_path = out / "final_protocol.json"
    if not fp_path.exists():
        raise FileNotFoundError("Run phase8_final_protocol.py first.")

    fp_data = json.loads(fp_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    payload = {
        "experiment": "mechanism_diagnostics",
        "git": git_status_snapshot(),
        "datasets": {},
    }

    for ds_name in DATASETS:
        ds_stem = ds_name.replace(".csv", "")
        protocol = load_selected_protocol(out, ds_stem)
        split = load_dataset_split(ds_name, DEFAULT_SEED)
        config = ExperimentConfig(
            scaling=protocol["scaling"],
            purity_threshold=protocol["purity_threshold"],
            c1=protocol["c1"],
            c2=protocol["c2"],
            min_ball_count=DEFAULT_MIN_BALL_COUNT,
            seed=DEFAULT_SEED,
        )

        X_train_full = np.vstack([split.X_train, split.X_val])
        y_train_full = np.concatenate([split.y_train, split.y_val])
        X_tr, X_val, X_test, _ = fit_scaler(
            config.scaling, X_train_full, split.X_val, split.X_test
        )

        fitted = {}
        for name in CLEAN_MODELS:
            fitted[name] = fit_model(name, X_tr, y_train_full, config)

        cw = fitted["CW-GBLSTSVM"]
        cr = fitted["CR-GBLSTSVM"]
        clean = fitted["Clean GBLSTSVM"]
        r_model = fitted["R-GBLSTSVM"]

        mech = mechanism_diagnostics(
            cw, cr, split.y_val, split.y_test, X_val, X_test, split.minority_label
        )

        clean_test_m, clean_pred, _ = predict_and_evaluate(
            clean, "Clean GBLSTSVM", X_test, split.y_test, split.minority_label
        )
        cw_test_m, cw_pred, _ = predict_and_evaluate(
            cw, "CW-GBLSTSVM", X_test, split.y_test, split.minority_label
        )
        r_test_m, r_pred, _ = predict_and_evaluate(
            r_model, "R-GBLSTSVM", X_test, split.y_test, split.minority_label
        )
        cr_test_m, cr_pred, _ = predict_and_evaluate(
            cr, "CR-GBLSTSVM", X_test, split.y_test, split.minority_label
        )

        ds_entry = {
            "protocol": protocol,
            "reliability": {
                k: mech[k]
                for k in mech
                if k.startswith("reliability") or k.startswith("frac_reliability")
            },
            "weight_differences": {
                "frac_cr_weights_differ_from_cw": mech["frac_cr_weights_differ_from_cw"],
                "mean_abs_cr_minus_cw_weight": mech["mean_abs_cr_minus_cw_weight"],
                "max_abs_cr_minus_cw_weight": mech["max_abs_cr_minus_cw_weight"],
            },
            "hyperplane_differences_cr_vs_cw": {
                "w1_norm_diff": mech["w1_norm_diff"],
                "w2_norm_diff": mech["w2_norm_diff"],
                "b1_abs_diff": mech["b1_abs_diff"],
                "b2_abs_diff": mech["b2_abs_diff"],
            },
            "hyperplane_differences_r_vs_clean": {
                "w1_norm_diff": float(np.linalg.norm(r_model.w1_ - clean.w1_)),
                "w2_norm_diff": float(np.linalg.norm(r_model.w2_ - clean.w2_)),
                "b1_abs_diff": float(abs(r_model.b1_ - clean.b1_)),
                "b2_abs_diff": float(abs(r_model.b2_ - clean.b2_)),
            },
            "prediction_differences": {
                "val_frac_pred_cr_neq_cw": mech["val_frac_pred_cr_neq_cw"],
                "test_frac_pred_cr_neq_cw": mech["test_frac_pred_cr_neq_cw"],
                "test_frac_pred_r_neq_clean": float(np.mean(r_pred != clean_pred)),
                "test_frac_pred_cr_neq_clean": float(np.mean(cr_pred != clean_pred)),
            },
            "model_ball_stats": {
                name: model_ball_and_reliability_stats(name, fitted[name])
                for name in CLEAN_MODELS
            },
            "test_deltas": {
                "CW_minus_clean": delta_metrics(cw_test_m, clean_test_m),
                "R_minus_clean": delta_metrics(r_test_m, clean_test_m),
                "CR_minus_clean": delta_metrics(cr_test_m, clean_test_m),
                "CR_minus_CW": delta_metrics(cr_test_m, cw_test_m),
                "CR_minus_R": delta_metrics(cr_test_m, r_test_m),
            },
            "final_protocol_match": fp_data["datasets"][ds_stem].get("deltas_test"),
        }
        payload["datasets"][ds_stem] = ds_entry

        row = {
            "dataset": ds_stem,
            "protocol_scaling": protocol["scaling"],
            "protocol_purity": protocol["purity_threshold"],
            "protocol_c1": protocol["c1"],
            "protocol_c2": protocol["c2"],
            **ds_entry["reliability"],
            **ds_entry["weight_differences"],
            **ds_entry["hyperplane_differences_cr_vs_cw"],
            **ds_entry["prediction_differences"],
            "test_macro_f1_clean": clean_test_m["macro_f1"],
            "test_macro_f1_cw": cw_test_m["macro_f1"],
            "test_macro_f1_r": r_test_m["macro_f1"],
            "test_macro_f1_cr": cr_test_m["macro_f1"],
        }
        rows.append(row)

    save_json(out / "mechanism_diagnostics.json", payload)
    save_csv(out / "mechanism_diagnostics.csv", rows)
    print(f"Wrote {out / 'mechanism_diagnostics.json'}")


if __name__ == "__main__":
    main()
