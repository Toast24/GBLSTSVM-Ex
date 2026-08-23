"""Phase 8 Experiment 2: Feature scaling ablation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase8_common import (
    CLEAN_MODELS,
    DATASETS,
    DEFAULT_PURITY,
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

SCALING_OPTIONS = ["none", "standard", "minmax"]


def main() -> None:
    out = ensure_out_dir()
    rows: list[dict] = []
    payload = {
        "experiment": "scaling_ablation",
        "scaling_options": SCALING_OPTIONS,
        "purity_threshold": DEFAULT_PURITY,
        "seed": DEFAULT_SEED,
        "git": git_status_snapshot(),
        "datasets": {},
    }

    for ds_name in DATASETS:
        split = load_dataset_split(ds_name, DEFAULT_SEED)
        ds_results = []
        for scaling in SCALING_OPTIONS:
            config = ExperimentConfig(scaling=scaling, seed=DEFAULT_SEED)
            run = run_clean_family(split, config)
            clean_val = run["models"]["Clean GBLSTSVM"]["val"]
            clean_test = run["models"]["Clean GBLSTSVM"]["test"]
            entry = {"scaling": scaling, "models": {}, "deltas_test": {}}
            for name in CLEAN_MODELS:
                m = run["models"][name]
                entry["models"][name] = {
                    "ball_stats": m["ball_stats"],
                    "val": m["val"],
                    "test": m["test"],
                }
                row = {
                    "dataset": split.name,
                    "scaling": scaling,
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

            cw_test = run["models"]["CW-GBLSTSVM"]["test"]
            r_test = run["models"]["R-GBLSTSVM"]["test"]
            cr_test = run["models"]["CR-GBLSTSVM"]["test"]
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
            "results": ds_results,
        }

    save_json(out / "scaling_ablation.json", payload)
    save_csv(out / "scaling_ablation.csv", rows)
    print(f"Wrote {out / 'scaling_ablation.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
