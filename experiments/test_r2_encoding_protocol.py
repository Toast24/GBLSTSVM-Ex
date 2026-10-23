"""
R2: encoding audit, ONEHOT external validity, protocol sensitivity.

Run from GBLSTSVM root:
  python experiments/test_r2_encoding_protocol.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    load_uci_dataset,
    try_load_descriptor,
)
from experiments.phase8_common import (
    DatasetSplit,
    inspect_dataset,
    map_to_binary,
    minority_class,
    normalize_labels,
)
from experiments.rebaseline_common import SEEDS, run_rho90_protocol

OUT_DIR = Path(__file__).resolve().parent / "output" / "rebaseline"

ENCODING_DATASETS = (
    "tic_tac_toe",
    "breast_cancer",
    "congressional_voting_records",
    "ecoli2",
)
EXTERNAL_DATASETS = ENCODING_DATASETS
PROTOCOL_DATASETS = ("spambase", "ecoli2")

PUBLISHED_LINEAR_ACC = {
    "spambase": 90.88,
    "tic_tac_toe": 99.65,
    "ecoli2": 89.11,
    "breast_cancer": 74.42,
}


def _mean_acc(rows: list[dict[str, Any]]) -> float:
    ok = [r["accuracy"] for r in rows if r.get("classifier_ok") and np.isfinite(r["accuracy"])]
    return float(np.mean(ok) * 100.0) if ok else float("nan")


def run_external_cell(onehot: bool) -> dict[str, float]:
    flags_rows: dict[str, list[dict[str, Any]]] = {d: [] for d in EXTERNAL_DATASETS}
    desc_map = {d["name"]: d for d in discover_available_datasets()}
    for ds in EXTERNAL_DATASETS:
        desc = desc_map.get(ds)
        if desc is None:
            continue
        for seed in SEEDS:
            split, err = try_load_descriptor(desc, seed, onehot=onehot)
            if split is None:
                continue
            flags_rows[ds].append(
                run_rho90_protocol(
                    split,
                    seed,
                    scale=True,
                    predict_rule="norm",
                    tune_c=True,
                    tune_cv=False,
                )
            )
    return {ds: _mean_acc(flags_rows[ds]) for ds in EXTERNAL_DATASETS}


def load_70_30_split(name: str, seed: int, *, onehot: bool = False) -> DatasetSplit:
    desc_map = {d["name"]: d for d in discover_available_datasets()}
    desc = desc_map[name]
    if desc["source"] == "data_dir":
        X, y_raw, info = inspect_dataset(desc["path"])
        y, label_mapping = map_to_binary(y_raw, info)
    else:
        X, y_raw, info = load_uci_dataset(desc["path"], onehot=onehot)
        y, label_mapping = map_to_binary(y_raw, info)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    y_tr_n = normalize_labels(y_tr)
    min_lab = minority_class(y_tr_n)
    return DatasetSplit(
        name=name,
        info=info,
        X_train=X_tr,
        X_val=X_tr[:0],
        X_test=X_te,
        y_train=y_tr,
        y_val=y_tr[:0],
        y_test=y_te,
        minority_label=min_lab,
        label_mapping=label_mapping,
    )


def run_protocol_sensitivity() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds in PROTOCOL_DATASETS:
        per_seed: list[float] = []
        for seed in SEEDS:
            split = load_70_30_split(ds, seed)
            r = run_rho90_protocol(
                split,
                seed,
                scale=True,
                predict_rule="norm",
                tune_c=True,
                tune_cv=True,
            )
            if r.get("classifier_ok") and np.isfinite(r["accuracy"]):
                per_seed.append(float(r["accuracy"]) * 100.0)
        acc = float(np.mean(per_seed)) if per_seed else float("nan")
        pub = PUBLISHED_LINEAR_ACC.get(ds)
        rows.append(
            {
                "dataset": ds,
                "mean_accuracy_pct": acc,
                "published_linear_acc": pub,
                "gap_to_published_pp": acc - pub if pub is not None and np.isfinite(acc) else float("nan"),
            }
        )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ordinal = run_external_cell(onehot=False)
    onehot = run_external_cell(onehot=True)

    ext_rows = []
    for ds in EXTERNAL_DATASETS:
        pub = PUBLISHED_LINEAR_ACC.get(ds)
        oa = ordinal.get(ds, float("nan"))
        ha = onehot.get(ds, float("nan"))
        gap = ha - pub if pub is not None and np.isfinite(ha) else float("nan")
        ext_rows.append(
            {
                "dataset": ds,
                "ordinal_acc_pct": oa,
                "onehot_acc_pct": ha,
                "published_linear_acc": pub,
                "gap_to_published_pp": gap,
            }
        )
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(OUT_DIR / "r2_external_onehot.csv", index=False)

    protocol_rows = run_protocol_sensitivity()
    protocol_df = pd.DataFrame(protocol_rows)
    protocol_df.to_csv(OUT_DIR / "r2_protocol_sensitivity.csv", index=False)

    meta = {"external": ext_rows, "protocol": protocol_rows}
    (OUT_DIR / "r2_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
