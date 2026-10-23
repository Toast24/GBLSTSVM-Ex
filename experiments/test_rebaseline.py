"""
R1 factorial re-baseline: rho90 arm, 17 datasets × 10 seeds × 8 flag cells.

Run from GBLSTSVM root:
  python experiments/test_rebaseline.py
  python experiments/test_rebaseline.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.rebaseline_common import (  # noqa: E402
    RebaselineFlags,
    SEEDS,
    all_flag_cells,
    run_rho90_cell,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "rebaseline"

EXTERNAL_DATASETS = (
    "spambase",
    "tic_tac_toe",
    "ecoli2",
    "breast_cancer",
    "ionosphere",
    "heart",
)
PUBLISHED_LINEAR_ACC = {
    "spambase": 90.88,
    "tic_tac_toe": 99.65,
    "ecoli2": 89.11,
    "breast_cancer": 74.42,
}


def _aggregate_cell(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["degenerate"] == 0]
    return {
        "n_runs": len(df),
        "n_non_degenerate": len(ok),
        "mean_macro_f1": float(ok["macro_f1"].mean()) if len(ok) else float("nan"),
        "mean_accuracy": float(ok["accuracy"].mean()) if len(ok) else float("nan"),
        "degenerate_rate": float(df["degenerate"].mean()),
        "tiny_rate": float(df["tiny"].mean()),
        "mean_w_norm_ratio": float(ok["w_norm_ratio"].mean()) if len(ok) else float("nan"),
        "disagree_frac": float(ok["disagree_frac"].mean()) if len(ok) else float("nan"),
    }


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_runs.csv", index=False)

    summary_rows = []
    for scale in (False, True):
        for rule in ("raw", "norm"):
            for tune_c in (False, True):
                sub = df[
                    (df["scale"] == scale)
                    & (df["predict_rule"] == rule)
                    & (df["tune_c"] == tune_c)
                ]
                agg = _aggregate_cell(sub)
                summary_rows.append(
                    {
                        "scale": scale,
                        "predict_rule": rule,
                        "tune_c": tune_c,
                        **agg,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "factorial_summary.csv", index=False)

    ref = df[
        (df["scale"] == True)
        & (df["predict_rule"] == "norm")
        & (df["tune_c"] == True)
    ]
    per_ds = (
        ref[ref["degenerate"] == 0]
        .groupby("dataset")["macro_f1"]
        .mean()
        .reset_index()
        .rename(columns={"macro_f1": "mean_macro_f1"})
    )
    per_ds.to_csv(OUT_DIR / "per_dataset_norm_tune.csv", index=False)

    ext = ref[ref["dataset"].isin(EXTERNAL_DATASETS)]
    ext_rows = []
    for ds in EXTERNAL_DATASETS:
        sub = ext[(ext["dataset"] == ds) & (ext["degenerate"] == 0)]
        acc = float(sub["accuracy"].mean()) if len(sub) else float("nan")
        pub = PUBLISHED_LINEAR_ACC.get(ds)
        within = None
        if pub is not None and np.isfinite(acc):
            within = abs(acc * 100.0 - pub) <= 3.0
        ext_rows.append(
            {
                "dataset": ds,
                "mean_accuracy_pct": acc * 100.0 if np.isfinite(acc) else float("nan"),
                "published_linear_acc": pub,
                "within_3pp": within,
            }
        )
    external = pd.DataFrame(ext_rows)
    external.to_csv(OUT_DIR / "external_validity.csv", index=False)

    meta = {
        "n_runs": len(df),
        "summary": summary_rows,
        "per_dataset_norm_tune": per_ds.to_dict(orient="records"),
        "external_validity": ext_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R1 factorial re-baseline")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--from-csv", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        df = pd.read_csv(OUT_DIR / "all_runs.csv")
        meta = analyze(df)
        print(json.dumps(meta["summary"], indent=2))
        return 0

    descriptors = discover_available_datasets()
    seeds = SEEDS
    if args.smoke:
        descriptors = descriptors[:2]
        seeds = [0]

    rows: list[dict[str, Any]] = []
    for flags in all_flag_cells():
        for desc in descriptors:
            for seed in seeds:
                split, err = try_load_descriptor(desc, seed)
                if split is None:
                    continue
                rows.append(run_rho90_cell(split, seed, flags))

    df = pd.DataFrame(rows)
    meta = analyze(df)
    print(f"Wrote {len(df)} runs -> {OUT_DIR}")
    print(meta["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
