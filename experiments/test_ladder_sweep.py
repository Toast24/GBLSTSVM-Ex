"""Ladder noise sweep: 17 datasets × 10 seeds × 4 arms × 5 noise levels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset_registry import discover_available_datasets  # noqa: E402
from experiments.ladder_common import (  # noqa: E402
    ARMS,
    NOISE_LEVELS,
    REG_BASE,
    SEEDS,
    aggregate_sweep_rows,
    run_ladder_cell,
)
from experiments.phase8_common import save_csv  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "ladder"
OUT_CSV = OUT_DIR / "all_runs.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"


def format_audit_section(summary: dict) -> str:
    lines = [
        "## D Ladder",
        "",
        f"Protocol: ONEHOT=True, rho={0.90}, scale=StandardScaler(train), "
        f"predict=raw, ridge λ = {REG_BASE} × trace(design)/dim per subproblem, "
        f"(c1,c2) retuned per arm×noise cell on internal val, "
        f"label noise on train only post-split pre-granulation.",
        "",
        "### (a) Mean macro F1 by arm × noise (dataset means, bootstrap 95% CI)",
        "",
        "| arm | noise% | mean F1 | CI lo | CI hi | n datasets |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary["grid"].sort_values(["noise_pct", "arm"]).iterrows():
        lines.append(
            f"| {row['arm']} | {int(row['noise_pct'])} | "
            f"{row['macro_f1_mean']:.4f} | {row['ci_lo']:.4f} | {row['ci_hi']:.4f} | "
            f"{int(row['n_datasets'])} |"
        )
    lines.extend(["", "### (b) L3 − L0 delta by noise", ""])
    lines.extend(["| noise% | mean Δ | CI lo | CI hi |", "|---:|---:|---:|---:|"])
    ddf = summary["deltas"]
    for noise in NOISE_LEVELS:
        sub = ddf[(ddf["delta"] == "L3-L0") & (ddf["noise_pct"] == noise)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"| {int(noise)} | {r['mean']:.4f} | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} |"
        )
    lines.extend(["", "### (c) L3 − L1 and L1 − L0 deltas", ""])
    lines.extend(["| delta | noise% | mean | CI lo | CI hi |", "|---|---:|---:|---:|---:|"])
    for delta in ("L3-L1", "L1-L0"):
        for noise in NOISE_LEVELS:
            sub = ddf[(ddf["delta"] == delta) & (ddf["noise_pct"] == noise)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            lines.append(
                f"| {delta} | {int(noise)} | {r['mean']:.4f} | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} |"
            )
    lines.extend(["", "### (d) Per-dataset L3 − L0 at 0% and 20% noise", ""])
    lines.extend(["| dataset | noise% | L3 − L0 |", "|---|---:|---:|"])
    for _, row in summary["per_dataset_delta"].sort_values(["noise_pct", "dataset"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['noise_pct'])} | {row['L3_minus_L0']:.4f} |"
        )
    lines.extend(["", "### (e) Ball geometry at 0% noise (per dataset)", ""])
    lines.extend(
        [
            "| dataset | mean n_balls | mean n_j | mean trace(S_j)/n_j |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in summary["geometry"].sort_values("dataset").iterrows():
        lines.append(
            f"| {row['dataset']} | {row['n_balls']:.1f} | {row['mean_n_j']:.2f} | "
            f"{row['mean_trace_S_over_n']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    descriptors = discover_available_datasets()
    rows: list[dict] = []
    total = len(descriptors) * len(SEEDS) * len(ARMS) * len(NOISE_LEVELS)
    done = 0
    for desc in descriptors:
        for seed in SEEDS:
            for arm in ARMS:
                for noise in NOISE_LEVELS:
                    cell = run_ladder_cell(desc, seed, arm, noise, tune_c=True)
                    if cell is not None:
                        rows.append(cell)
                    done += 1
                    if done % 100 == 0:
                        print(f"progress {done}/{total}", flush=True)
    save_csv(OUT_CSV, rows)
    summary = aggregate_sweep_rows(rows)
    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "n_runs": len(rows),
                "grid": summary["grid"].to_dict(orient="records"),
                "deltas": summary["deltas"].to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    audit_path = ROOT.parent / "AUDIT.md"
    section = format_audit_section(summary)
    text = audit_path.read_text(encoding="utf-8")
    if "## D Ladder" in text:
        text = text.split("## D Ladder")[0].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    audit_path.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")
    print(f"appended ## D Ladder to {audit_path}")


if __name__ == "__main__":
    main()
