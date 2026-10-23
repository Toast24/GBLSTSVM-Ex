"""L3_k2 noise sweep and AUDIT D2 section."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import discover_available_datasets  # noqa: E402
from experiments.ladder_common import (  # noqa: E402
    K2_EXACTNESS_SEEDS,
    NOISE_LEVELS,
    SEEDS,
    run_l3_k2_cell,
    storage_table_rows,
)
from experiments.phase8_common import save_csv  # noqa: E402
from tests.test_ladder_k2 import run_exactness_gate  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "ladder_k2"
OUT_CSV = OUT_DIR / "all_runs.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"


def format_audit_section(
    ok: bool,
    max_dev1: float,
    max_dev2: float,
    sweep_rows: list[dict],
    storage: list[dict],
) -> str:
    import pandas as pd

    df = pd.DataFrame(sweep_rows)
    per_ds = (
        df.groupby(["dataset", "noise_pct"], as_index=False)[["macro_f1", "accuracy"]]
        .mean()
        .rename(columns={"macro_f1": "mean_macro_f1", "accuracy": "mean_accuracy"})
    )
    grid = (
        per_ds.groupby("noise_pct")[["mean_macro_f1", "mean_accuracy"]]
        .mean()
        .reset_index()
    )

    lines = [
        "## D2 k=2 exactness",
        "",
        "### 1. Parameter exactness (LINEAR, ONEHOT=True)",
        "",
        f"Assert L3_k2 == L3 == raw LSTSVM on 17 datasets × 3 seeds (`tests/test_ladder_k2.py`).",
        "",
        "| check | result | max abs dev [w1;b1] | max abs dev [w2;b2] | tolerance |",
        "|---|---|---:|---:|---:|",
        f"| L3_k2 / L3 / raw | **{'PASS' if ok else 'FAIL'}** | {max_dev1:.3e} | {max_dev2:.3e} | 1e-8 |",
        "",
        "L3_k2: one ball per class with (n_j, c_j, S_j) over all same-class train points; "
        "L3 normal equations (`src/ladder_gblstsvm.py:127–148`). No granulation.",
        "",
        "### 2. L3_k2 noise sweep (17 datasets × 10 seeds × 5 noise levels)",
        "",
        "Per-dataset means over seeds first; table = mean across datasets.",
        "",
        "| noise% | mean macro F1 | mean accuracy |",
        "|---:|---:|---:|",
    ]
    for _, row in grid.sort_values("noise_pct").iterrows():
        lines.append(
            f"| {int(row['noise_pct'])} | {row['mean_macro_f1']:.4f} | {row['mean_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### 3. Storage complexity (seed=0, rho90 k from MIN_BALL_COUNT=2 granulation)",
            "",
            "| dataset | d | n_train | k_rho90 | raw n×d | rho90 k×(d+1)² | L3_k2 2×(d+1)² | ratio_rho90/raw | ratio_k2/raw |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(storage, key=lambda r: r["dataset"]):
        lines.append(
            f"| {row['dataset']} | {row['d']} | {row['n_train']} | {row['k_rho90']} | "
            f"{row['raw_n_d']} | {row['rho90_k_d1_sq']} | {row['L3_k2_2_d1_sq']} | "
            f"{row['ratio_rho90_to_raw']:.4f} | {row['ratio_k2_to_raw']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, max_dev1, max_dev2 = run_exactness_gate()
    if not ok:
        print("ESCALATE: exactness gate failed")
        sys.exit(1)

    descriptors = discover_available_datasets()
    rows: list[dict] = []
    total = len(descriptors) * len(SEEDS) * len(NOISE_LEVELS)
    done = 0
    for desc in descriptors:
        for seed in SEEDS:
            for noise in NOISE_LEVELS:
                cell = run_l3_k2_cell(desc, seed, noise, tune_c=True)
                if cell is not None:
                    rows.append(cell)
                done += 1
                if done % 50 == 0:
                    print(f"progress {done}/{total}", flush=True)
    save_csv(OUT_CSV, rows)
    storage = storage_table_rows(seed=0)
    section = format_audit_section(ok, max_dev1, max_dev2, rows, storage)
    OUT_SUMMARY.write_text(
        json.dumps({"exactness_ok": ok, "max_dev1": max_dev1, "max_dev2": max_dev2}, indent=2),
        encoding="utf-8",
    )
    audit_path = ROOT.parent / "AUDIT.md"
    text = audit_path.read_text(encoding="utf-8")
    if "## D2 k=2 exactness" in text:
        text = text.split("## D2 k=2 exactness")[0].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    audit_path.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")
    print(f"appended ## D2 k=2 exactness to {audit_path}")


if __name__ == "__main__":
    main()
