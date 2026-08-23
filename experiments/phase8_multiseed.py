"""Phase 8 Experiment 5: Multi-seed stability for final selected protocol."""

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

SEEDS = [0, 1, 2, 3, 4, 42]


def load_selected_protocol(out_dir: Path, dataset: str) -> dict:
    fp = out_dir / "final_protocol.json"
    if not fp.exists():
        raise FileNotFoundError("Run phase8_final_protocol.py first.")
    data = json.loads(fp.read_text(encoding="utf-8"))
    ds = data["datasets"][dataset]
    sel = ds["selection"]["selected_protocol"]
    return {
        "scaling": sel["scaling"],
        "purity_threshold": sel["purity_threshold"],
        "c1": sel["c1"],
        "c2": sel["c2"],
    }


def aggregate_metrics(records: list[dict], prefix: str) -> dict:
    keys = ["balanced_accuracy", "macro_f1", "minority_recall", "gmean", "auroc"]
    out = {}
    for k in keys:
        vals = [r.get(f"{prefix}_{k}") for r in records if r.get(f"{prefix}_{k}") is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out[f"{prefix}_{k}_mean"] = float(np.mean(arr))
        out[f"{prefix}_{k}_std"] = float(np.std(arr))
        out[f"{prefix}_{k}_median"] = float(np.median(arr))
        out[f"{prefix}_{k}_min"] = float(np.min(arr))
        out[f"{prefix}_{k}_max"] = float(np.max(arr))
    return out


def main() -> None:
    out = ensure_out_dir()
    rows: list[dict] = []
    payload = {
        "experiment": "multiseed",
        "seeds": SEEDS,
        "git": git_status_snapshot(),
        "datasets": {},
    }

    for ds_name in DATASETS:
        ds_stem = ds_name.replace(".csv", "")
        protocol = load_selected_protocol(out, ds_stem)
        seed_records = []
        delta_cr_cw_records = []
        delta_cw_clean_records = []

        for seed in SEEDS:
            split = load_dataset_split(ds_name, seed)
            config = ExperimentConfig(
                scaling=protocol["scaling"],
                purity_threshold=protocol["purity_threshold"],
                c1=protocol["c1"],
                c2=protocol["c2"],
                min_ball_count=DEFAULT_MIN_BALL_COUNT,
                seed=seed,
            )
            X_train, X_val, X_test, _ = fit_scaler(
                config.scaling, split.X_train, split.X_val, split.X_test
            )
            seed_row = {"dataset": split.name, "seed": seed, "protocol": protocol}
            fitted = {}
            for name in CLEAN_MODELS:
                model = fit_model(name, X_train, split.y_train, config)
                fitted[name] = model
                val_m, _, _ = predict_and_evaluate(
                    model, name, X_val, split.y_val, split.minority_label
                )
                test_m, _, _ = predict_and_evaluate(
                    model, name, X_test, split.y_test, split.minority_label
                )
                bs = model_ball_and_reliability_stats(name, model)
                seed_row.update({
                    f"{name}_val_macro_f1": val_m["macro_f1"],
                    f"{name}_test_macro_f1": test_m["macro_f1"],
                    f"{name}_test_balanced_accuracy": test_m["balanced_accuracy"],
                    f"{name}_test_minority_recall": test_m["minority_recall"],
                    f"{name}_test_gmean": test_m["gmean"],
                    f"{name}_test_auroc": test_m.get("auroc"),
                    f"{name}_n_balls": bs.get("n_balls_total"),
                })
                if name == "CR-GBLSTSVM":
                    seed_row["frac_cr_weights_differ_from_cw"] = bs.get(
                        "frac_cr_weights_differ_from_cw"
                    )

            clean_test = {
                "macro_f1": seed_row["Clean GBLSTSVM_test_macro_f1"],
                "balanced_accuracy": seed_row["Clean GBLSTSVM_test_balanced_accuracy"],
                "minority_recall": seed_row["Clean GBLSTSVM_test_minority_recall"],
                "gmean": seed_row["Clean GBLSTSVM_test_gmean"],
                "auroc": seed_row["Clean GBLSTSVM_test_auroc"],
            }
            cw_test = {
                "macro_f1": seed_row["CW-GBLSTSVM_test_macro_f1"],
                "balanced_accuracy": seed_row["CW-GBLSTSVM_test_balanced_accuracy"],
                "minority_recall": seed_row["CW-GBLSTSVM_test_minority_recall"],
                "gmean": seed_row["CW-GBLSTSVM_test_gmean"],
                "auroc": seed_row["CW-GBLSTSVM_test_auroc"],
            }
            cr_test = {
                "macro_f1": seed_row["CR-GBLSTSVM_test_macro_f1"],
                "balanced_accuracy": seed_row["CR-GBLSTSVM_test_balanced_accuracy"],
                "minority_recall": seed_row["CR-GBLSTSVM_test_minority_recall"],
                "gmean": seed_row["CR-GBLSTSVM_test_gmean"],
                "auroc": seed_row["CR-GBLSTSVM_test_auroc"],
            }
            d_cw_clean = delta_metrics(cw_test, clean_test)
            d_cr_cw = delta_metrics(cr_test, cw_test)
            seed_row["delta_CW_minus_clean_macro_f1"] = d_cw_clean.get("macro_f1")
            seed_row["delta_CR_minus_CW_macro_f1"] = d_cr_cw.get("macro_f1")
            delta_cw_clean_records.append(d_cw_clean)
            delta_cr_cw_records.append(d_cr_cw)

            mech = mechanism_diagnostics(
                fitted["CW-GBLSTSVM"],
                fitted["CR-GBLSTSVM"],
                split.y_val,
                split.y_test,
                X_val,
                X_test,
                split.minority_label,
            )
            seed_row["w1_norm_diff"] = mech["w1_norm_diff"]
            seed_row["test_frac_pred_cr_neq_cw"] = mech["test_frac_pred_cr_neq_cw"]

            rows.append(seed_row)
            seed_records.append(seed_row)

        cr_cw_f1 = [r["delta_CR_minus_CW_macro_f1"] for r in seed_records]
        cw_clean_f1 = [r["delta_CW_minus_clean_macro_f1"] for r in seed_records]
        summary = {
            "protocol": protocol,
            "per_seed": seed_records,
            "aggregate": aggregate_metrics(seed_records, "Clean GBLSTSVM_test"),
            "delta_CR_minus_CW_macro_f1_mean": float(np.mean(cr_cw_f1)),
            "delta_CR_minus_CW_macro_f1_std": float(np.std(cr_cw_f1)),
            "delta_CW_minus_clean_macro_f1_mean": float(np.mean(cw_clean_f1)),
            "delta_CW_minus_clean_macro_f1_std": float(np.std(cw_clean_f1)),
        }
        payload["datasets"][ds_stem] = summary

    save_json(out / "multiseed.json", payload)
    save_csv(out / "multiseed.csv", rows)
    print(f"Wrote {out / 'multiseed.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
