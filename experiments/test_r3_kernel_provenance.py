"""R3: kernel provenance, linear R1 re-run (ONEHOT=True), ONEHOT ablation."""

from __future__ import annotations

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
from experiments.ladder_common import (  # noqa: E402
    build_rho90_balls,
    fit_eval_ladder,
    tune_c1_c2_ladder,
)
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, save_csv
from experiments.rebaseline_common import (  # noqa: E402
    RebaselineFlags,
    SEEDS,
    all_flag_cells,
    disagree_fraction,
    failure_flags,
    scale_benchmark_split,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "rebaseline_r3_linear"
OUT_CSV = OUT_DIR / "all_runs.csv"
OUT_SUMMARY = OUT_DIR / "factorial_summary.csv"


def run_rho90_cell_linear(
    split,
    seed: int,
    flags: RebaselineFlags,
) -> dict[str, Any]:
    scaled = scale_benchmark_split(split, flags.scale)
    try:
        balls = build_rho90_balls(scaled, seed)
    except ValueError:
        balls = []
    n_balls = len(balls)
    c1, c2 = DEFAULT_C1, DEFAULT_C2
    if flags.tune_c and balls:
        c1, c2 = tune_c1_c2_ladder(
            scaled, balls, "L0", seed, prediction_rule=flags.predict_rule
        )
    if balls:
        perf = fit_eval_ladder(
            scaled, balls, "L0", c1, c2, prediction_rule=flags.predict_rule
        )
    else:
        perf = {
            "macro_f1": float("nan"),
            "accuracy": float("nan"),
            "classifier_ok": False,
            "w1": None,
            "w2": None,
            "b1": None,
            "b2": None,
        }
    macro_f1 = float(perf.get("macro_f1", float("nan")))
    accuracy = float(perf.get("accuracy", float("nan")))
    classifier_ok = bool(perf.get("classifier_ok", False))
    degenerate, tiny = failure_flags(n_balls, classifier_ok, macro_f1)
    w1, w2, b1, b2 = perf.get("w1"), perf.get("w2"), perf.get("b1"), perf.get("b2")
    w_ratio = float("nan")
    disagree = float("nan")
    if w1 is not None and w2 is not None:
        n1 = float(np.linalg.norm(w1))
        n2 = float(np.linalg.norm(w2))
        w_ratio = n1 / n2 if n2 > 0 else float("nan")
        if b1 is not None and b2 is not None:
            disagree = disagree_fraction(scaled.X_test, w1, b1, w2, b2)
    return {
        "dataset": split.name,
        "seed": int(seed),
        "scale": flags.scale,
        "predict_rule": flags.predict_rule,
        "tune_c": flags.tune_c,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "degenerate": degenerate,
        "tiny": tiny,
        "n_balls": n_balls,
        "c1": float(c1),
        "c2": float(c2),
        "w_norm_ratio": w_ratio,
        "disagree_frac": disagree,
    }


def run_linear_r1_factorial() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    descriptors = discover_available_datasets()
    total = len(descriptors) * len(SEEDS) * len(all_flag_cells())
    done = 0
    for flags in all_flag_cells():
        for desc in descriptors:
            for seed in SEEDS:
                split, _ = try_load_descriptor(desc, seed=seed, onehot=True)
                if split is None:
                    continue
                rows.append(run_rho90_cell_linear(split, seed, flags))
                done += 1
                if done % 100 == 0:
                    print(f"R1 linear progress {done}/{total}", flush=True)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    summary_rows = []
    for scale in (False, True):
        for rule in ("raw", "norm"):
            for tune_c in (False, True):
                sub = df[
                    (df["scale"] == scale)
                    & (df["predict_rule"] == rule)
                    & (df["tune_c"] == tune_c)
                ]
                ok = sub[sub["degenerate"] == 0]
                summary_rows.append(
                    {
                        "scale": scale,
                        "predict_rule": rule,
                        "tune_c": tune_c,
                        "n_runs": len(sub),
                        "mean_macro_f1": float(ok["macro_f1"].mean()) if len(ok) else float("nan"),
                        "mean_accuracy": float(ok["accuracy"].mean()) if len(ok) else float("nan"),
                        "degenerate_rate": float(sub["degenerate"].mean()),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    return df, summary


def run_onehot_ablation() -> pd.DataFrame:
    flags = RebaselineFlags(scale=True, predict_rule="raw", tune_c=True)
    rows: list[dict[str, Any]] = []
    for desc in discover_available_datasets():
        for onehot in (False, True):
            for seed in SEEDS:
                split, _ = try_load_descriptor(desc, seed=seed, onehot=onehot)
                if split is None:
                    continue
                cell = run_rho90_cell_linear(split, seed, flags)
                rows.append(
                    {
                        "dataset": desc["name"],
                        "onehot": onehot,
                        "seed": seed,
                        "macro_f1": cell["macro_f1"],
                        "n_features": split.X_train.shape[1],
                    }
                )
    df = pd.DataFrame(rows)
    off = df[~df["onehot"]].groupby("dataset")["macro_f1"].mean()
    on = df[df["onehot"]].groupby("dataset")["macro_f1"].mean()
    nf_off = df[~df["onehot"]].groupby("dataset")["n_features"].first()
    nf_on = df[df["onehot"]].groupby("dataset")["n_features"].first()
    table = pd.DataFrame(
        {
            "off": off,
            "on": on,
            "delta": on - off,
            "n_feat_off": nf_off,
            "n_feat_on": nf_on,
        }
    ).reset_index()
    table = table.sort_values("delta")
    save_csv(OUT_DIR / "onehot_ablation.csv", table.to_dict(orient="records"))
    return table


def format_audit_section(summary: pd.DataFrame, ablation: pd.DataFrame) -> str:
    canonical = summary[
        (summary["scale"] == True)
        & (summary["predict_rule"] == "raw")
        & (summary["tune_c"] == True)
    ].iloc[0]
    lines = [
        "## R3 Kernel provenance",
        "",
        "### 1. `fit_gaussian_gblstsvm` body (`analyze_granular_geometry_vs_performance.py:359–430`)",
        "",
        "```python",
        "def fit_gaussian_gblstsvm(balls, *, y_train, X_test, y_test, minority_label, seed,",
        "                          c1=None, c2=None, prediction_rule=None) -> dict:",
        "    ...",
        "    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y)",
        "    kappa_pos_b, kappa_neg_b = ball_class_weights(balls, kappa_pos, kappa_neg)",
        "    mu_g = compute_gaussian_memberships(balls, PRIMARY_GAUSSIAN)  # :396",
        "    mu_pos, mu_neg = split_membership_by_class(balls, mu_g)",
        "    weights_pos = kappa_pos_b * mu_pos                              # :398-399",
        "    weights_neg = kappa_neg_b * mu_neg",
        "    centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(balls)",
        "    w1, b1, w2, b2 = solve_cw_gblstsvm_systems(                   # :406-417",
        "        centers_pos, radii_pos, centers_neg, radii_neg,",
        "        use_c1, use_c2, weights_pos, weights_neg, reg_term=REG_TERM, ...)",
        "    y_pred = predict_twin_hyperplanes(X_test, w1, b1, w2, b2, ...)  # :418",
        "    scores = np.abs(X_test @ w2 + b2) - np.abs(X_test @ w1 + b1)  # :419",
        "```",
        "",
        "**Verdict:** No RBF/Gaussian kernel map is applied to ball centres. "
        "\"Gaussian\" is a **naming artifact for ball-membership weighting** in an otherwise "
        "**linear** twin-hyperplane solve.",
        "",
        "Deciding lines:",
        "- `:396–399` — `compute_gaussian_memberships` yields per-ball scalars μ_j multiplied into "
        "class weights; centres enter unchanged.",
        "- `:406–417` — `solve_cw_gblstsvm_systems` builds weighted normal equations on raw centre "
        "matrix `[c_j; 1]` (see `cw_gblstsvm.py:73–76`).",
        "- `:418–419` — prediction is `X @ w + b` in original feature space (linear).",
        "",
        "### 2. Sigma / tuning in R1",
        "",
        "Membership uses `PRIMARY_GAUSSIAN = \"purity_compact\"` "
        "(`gaussian_membership_screen.py:69`): μ_j = purity_j × exp(−r̃_j²) with "
        "r̃_j = r_j / median(r>0) (`gaussian_membership_screen.py:134–140`). "
        "**No σ parameter or σ grid exists; σ was not tuned in R1.** Only (c1, c2) are grid-searched "
        "when `tune_c=True` (`rebaseline_common.py:26`, `:95–112`).",
        "",
        "### 3. R1 factorial re-run: LINEAR L0 ladder, ONEHOT=True",
        "",
        "17 datasets × 10 seeds × 8 cells; rho90 granulation; ladder arm L0 + `predict_twin_hyperplanes`. "
        f"Output: `{OUT_CSV.relative_to(ROOT.parent)}`.",
        "",
        "| scale | predict | tune_c | mean macro F1 | mean accuracy | degenerate rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scale']} | {row['predict_rule']} | {row['tune_c']} | "
            f"{row['mean_macro_f1']:.4f} | {row['mean_accuracy']:.4f} | {row['degenerate_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Canonical cell (scale=True, raw, tune_c=True): mean macro F1 = "
            f"{canonical['mean_macro_f1']:.4f}, mean accuracy = {canonical['mean_accuracy']:.4f}.",
            "",
            "### 4. True ONEHOT ablation (LINEAR L0, scale=True, raw, tune_c=True, rho90)",
            "",
            "| dataset | off | on | delta | n_feat_off | n_feat_on |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in ablation.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['off']:.4f} | {row['on']:.4f} | {row['delta']:.4f} | "
            f"{int(row['n_feat_off'])} | {int(row['n_feat_on'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("Running linear R1 factorial (ONEHOT=True)...", flush=True)
    _, summary = run_linear_r1_factorial()
    print("Running ONEHOT ablation...", flush=True)
    ablation = run_onehot_ablation()
    section = format_audit_section(summary, ablation)
    audit_path = ROOT.parent / "AUDIT.md"
    text = audit_path.read_text(encoding="utf-8")
    if "## R3 Kernel provenance" in text:
        text = text.split("## R3 Kernel provenance")[0].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    audit_path.write_text(text, encoding="utf-8")
    print(f"appended ## R3 Kernel provenance to {audit_path}")


if __name__ == "__main__":
    main()
