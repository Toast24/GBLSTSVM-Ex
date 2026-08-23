"""Phase 8 Experiment 1: Granulation sensitivity."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase8_common import (
    CLEAN_MODELS,
    DATASETS,
    DEFAULT_MIN_BALL_COUNT,
    DEFAULT_SEED,
    ExperimentConfig,
    delta_metrics,
    ensure_out_dir,
    git_status_snapshot,
    load_dataset_split,
    run_clean_family,
    save_csv,
    save_json,
)

PURITY_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98]


def main() -> None:
    out = ensure_out_dir()
    rows: list[dict] = []
    payload = {
        "experiment": "granulation_sensitivity",
        "purity_thresholds": PURITY_THRESHOLDS,
        "min_ball_count": DEFAULT_MIN_BALL_COUNT,
        "seed": DEFAULT_SEED,
        "git": git_status_snapshot(),
        "datasets": {},
    }

    for ds_name in DATASETS:
        split = load_dataset_split(ds_name, DEFAULT_SEED)
        ds_results = []
        for purity in PURITY_THRESHOLDS:
            config = ExperimentConfig(
                purity_threshold=purity,
                min_ball_count=DEFAULT_MIN_BALL_COUNT,
                seed=DEFAULT_SEED,
            )
            run = run_clean_family(split, config)
            if "val" not in run["models"].get("Clean GBLSTSVM", {}):
                entry = {
                    "purity_threshold": purity,
                    "error": run["models"].get("Clean GBLSTSVM", {}).get("error"),
                    "models": run["models"],
                }
                ds_results.append(entry)
                continue
            clean_val = run["models"]["Clean GBLSTSVM"]["val"]
            clean_test = run["models"]["Clean GBLSTSVM"]["test"]
            entry = {
                "purity_threshold": purity,
                "models": {},
                "deltas_test": {},
            }
            for name in CLEAN_MODELS:
                m = run["models"].get(name, {})
                if "val" not in m:
                    rows.append({
                        "dataset": split.name,
                        "purity_threshold": purity,
                        "model": name,
                        "error": m.get("error"),
                    })
                    continue
                entry["models"][name] = {
                    "ball_stats": m["ball_stats"],
                    "val": m["val"],
                    "test": m["test"],
                }
                row = {
                    "dataset": split.name,
                    "purity_threshold": purity,
                    "model": name,
                    **{f"val_{k}": m["val"].get(k) for k in (
                        "accuracy", "balanced_accuracy", "macro_f1",
                        "minority_recall", "gmean", "auroc",
                    )},
                    **{f"test_{k}": m["test"].get(k) for k in (
                        "accuracy", "balanced_accuracy", "macro_f1",
                        "minority_recall", "gmean", "auroc",
                    )},
                    **m["ball_stats"],
                }
                rows.append(row)

            cw_test = run["models"].get("CW-GBLSTSVM", {}).get("test", {})
            r_test = run["models"].get("R-GBLSTSVM", {}).get("test", {})
            cr_test = run["models"].get("CR-GBLSTSVM", {}).get("test", {})
            if cw_test and clean_test:
                entry["deltas_test"] = {
                    "CW_minus_clean": delta_metrics(cw_test, clean_test),
                    "R_minus_clean": delta_metrics(r_test, clean_test),
                    "CR_minus_clean": delta_metrics(cr_test, clean_test),
                    "CR_minus_CW": delta_metrics(cr_test, cw_test),
                    "CR_minus_R": delta_metrics(cr_test, r_test),
                }
            entry["best_val_macro_f1_clean"] = clean_val["macro_f1"]
            ds_results.append(entry)

        payload["datasets"][split.name] = {
            "info": split.info,
            "label_mapping": split.label_mapping,
            "results": ds_results,
        }

    save_json(out / "granulation_sensitivity.json", payload)
    save_csv(out / "granulation_sensitivity.csv", rows)
    print(f"Wrote {out / 'granulation_sensitivity.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
