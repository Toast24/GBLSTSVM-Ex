"""
Download UCI datasets used in the GBFTSVM benchmark.

Creates:
    datasets/uci/
        <dataset_name>/
            data.csv
            metadata.json
            variables.csv

Also creates:
    datasets/uci/download_log.json

Install:
    pip install -U ucimlrepo pandas
"""

from pathlib import Path
import json
import traceback

import pandas as pd
from ucimlrepo import fetch_ucirepo


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------

OUT_DIR = Path("datasets") / "uci"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Dataset registry
#
# IDs are official UCI dataset IDs.
# ---------------------------------------------------------

DATASETS = {
    "australian": 143,
    "breast_cancer_wisconsin_original": 15,
    "breast_cancer_wisconsin_diagnostic": 17,
    "chronic_kidney_disease": 336,
    "congressional_voting_records": 105,
    "credit_approval": 27,
    "german_credit": 144,
    "heart": 145,
    "ionosphere": 52,
    "spambase": 94,
    "tic_tac_toe": 101,
    "wholesale_customers": 292,

    # Paper names that map cleanly enough to UCI datasets
    # but should be verified against the exact paper preprocessing.
    "messidor_features": 329,
}


# ---------------------------------------------------------
# Ambiguous / unavailable names from the paper
# ---------------------------------------------------------

MANUAL_REVIEW = [
    "breast-cancer",
    "breast-cancer-wisc",
    "conn-bench-sonar-mines-rocks",
    "diabetes",
    "diabetes-upload",
    "electrical",
    "fourclass",
    "liver-disorders",
    "twonorm",
]


def save_dataset(name, dataset):
    """Save UCI dataset data + metadata."""

    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Combined data
    # -------------------------

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Give target columns an explicit prefix so they cannot
    # accidentally get confused with features.
    y.columns = [f"target_{c}" for c in y.columns]

    df = pd.concat([X, y], axis=1)
    df.to_csv(out / "data.csv", index=False)

    # -------------------------
    # Variables
    # -------------------------

    if dataset.variables is not None:
        dataset.variables.to_csv(out / "variables.csv", index=False)

    # -------------------------
    # Metadata
    # -------------------------

    metadata = {}

    try:
        metadata = dict(dataset.metadata)
    except Exception:
        metadata = {
            "name": name,
            "error": "Could not convert metadata to dict"
        }

    # Make metadata JSON-safe.
    def json_safe(obj):
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    metadata = {
        str(k): json_safe(v)
        for k, v in metadata.items()
    }

    metadata["download_name"] = name
    metadata["num_rows_saved"] = len(df)
    metadata["num_features_saved"] = len(X)
    metadata["num_targets_saved"] = len(y)

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return len(X), len(y)


def main():

    print("=" * 70)
    print("UCI GBFTSVM BENCHMARK DATASET DOWNLOADER")
    print("=" * 70)
    print(f"Output: {OUT_DIR.resolve()}")
    print()

    results = {}

    for name, uci_id in DATASETS.items():

        print(f"[DOWNLOAD] {name} (UCI ID {uci_id})")

        try:
            dataset = fetch_ucirepo(id=uci_id)

            n_features, n_targets = save_dataset(
                name,
                dataset
            )

            results[name] = {
                "status": "success",
                "uci_id": uci_id,
                "features": n_features,
                "targets": n_targets,
            }

            print(
                f"  OK: {n_features} feature rows, "
                f"{n_targets} target columns"
            )

        except Exception as e:

            results[name] = {
                "status": "failed",
                "uci_id": uci_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

            print(f"  FAILED: {e}")

        print()

    # -----------------------------------------------------
    # Save download log
    # -----------------------------------------------------

    log_path = OUT_DIR / "download_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # -----------------------------------------------------
    # Print manual-review datasets
    # -----------------------------------------------------

    print("=" * 70)
    print("DATASETS REQUIRING MANUAL IDENTIFICATION")
    print("=" * 70)

    for name in MANUAL_REVIEW:
        print(f"  - {name}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    successful = sum(
        r["status"] == "success"
        for r in results.values()
    )

    failed = sum(
        r["status"] == "failed"
        for r in results.values()
    )

    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Manual:     {len(MANUAL_REVIEW)}")
    print()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()