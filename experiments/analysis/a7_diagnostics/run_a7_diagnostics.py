"""
A7 follow-up diagnostics on A6 scatter-mechanism outputs.

  python experiments/analysis/a7_diagnostics/run_a7_diagnostics.py --from-csv
  python experiments/analysis/a7_diagnostics/run_a7_diagnostics.py --dump-params
  python experiments/analysis/a7_diagnostics/run_a7_diagnostics.py --from-csv --angles
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

GBLSTSVM = Path(__file__).resolve().parents[3]
if str(GBLSTSVM) not in sys.path:
    sys.path.insert(0, str(GBLSTSVM))

A6_OUT = GBLSTSVM / "experiments" / "output" / "estimator_a6_scatter_mechanism"
A7_OUT = Path(__file__).resolve().parent
PARAMS_DIR = A6_OUT / "params"

from experiments.rebaseline_common import C_GRID  # noqa: E402

BOOTSTRAP_SEED = 42
N_BOOT = 10_000

ARMS = [
    "L1_POPULATION",
    "FULL_SCATTER",
    "DIAG_SCATTER",
    "ISOTROPIC_SCATTER",
    "RIDGE_TRACE_MATCH",
    "RIDGE_VAL_TUNED",
]

TUNING_MODES = ["MODE_A_SHARED_C", "MODE_B_PER_ARM_C"]
RULES = ["RULE_REPO", "RULE_REFERENCE"]

PRIMARY_CONTRASTS: list[tuple[str, str]] = [
    ("FULL_SCATTER", "DIAG_SCATTER"),
    ("FULL_SCATTER", "ISOTROPIC_SCATTER"),
    ("ISOTROPIC_SCATTER", "RIDGE_TRACE_MATCH"),
    ("FULL_SCATTER", "L1_POPULATION"),
    ("ISOTROPIC_SCATTER", "L1_POPULATION"),
    ("DIAG_SCATTER", "L1_POPULATION"),
    ("DIAG_SCATTER", "ISOTROPIC_SCATTER"),
    ("FULL_SCATTER", "RIDGE_TRACE_MATCH"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from-csv", action="store_true", help="Recompute diagnostics from A6 CSVs")
    p.add_argument(
        "--dump-params",
        action="store_true",
        help="Invoke A6 --dump-params then run angle diagnostics",
    )
    p.add_argument("--angles", action="store_true", help="Run Task 3 (requires params/)")
    p.add_argument("--a6-dir", type=Path, default=A6_OUT)
    p.add_argument("--out-dir", type=Path, default=A7_OUT)
    return p.parse_args()


def _grid_rank(val: float, grid: list[float]) -> tuple[int, str]:
    for i, g in enumerate(grid):
        if np.isclose(val, g, rtol=0, atol=1e-12):
            if i == 0:
                return i, "minimum"
            if i == len(grid) - 1:
                return i, "maximum"
            return i, "interior"
    return -1, "off_grid"


def task_inventory(a6: Path, out: Path) -> None:
    runs = pd.read_csv(a6 / "runs.csv")
    ridge = pd.read_csv(a6 / "ridge_matching.csv")
    geom = pd.read_csv(a6 / "scatter_geometry.csv")
    dpd = pd.read_csv(a6 / "dataset_pairwise_deltas.csv")

    lines = [
        "# Task 0 — Artifact inventory",
        "",
        "## runs.csv",
        f"- rows: {len(runs)}",
        "- columns and dtypes:",
        "",
        "```",
        runs.dtypes.to_string(),
        "```",
        "",
        "## ridge_matching.csv",
        "```",
        ridge.dtypes.to_string(),
        "```",
        "",
        "## scatter_geometry.csv",
        "```",
        geom.dtypes.to_string(),
        "```",
        "",
        "## dataset_pairwise_deltas.csv",
        "```",
        dpd.dtypes.to_string(),
        "```",
        "",
        "## Distinct factor levels",
        f"- `arm`: {sorted(runs['arm'].unique().tolist())}",
        f"- `tuning_mode`: {sorted(runs['tuning_mode'].unique().tolist())}",
        f"- `prediction_rule`: {sorted(runs['prediction_rule'].unique().tolist())}",
        "",
        "## Fitted parameters on disk",
        f"- `runs.csv` columns w1/w2/b1/b2: **none** (checked columns: {list(runs.columns)})",
        f"- `.npz`/`.pkl`/`.joblib` under `experiments/output/`: **none found** (glob at inventory time)",
        f"- A6 script `--dump-params` writes to `{a6 / 'params'}/` when invoked",
        "",
        "## A4 (`estimator_information_ladder`) RAW parameters",
        "- `all_runs.csv` has no w1/w2/b1/b2 columns; no serialized models in that output tree.",
        "",
        "## c-grid (MODE B)",
        f"- Source: `experiments/rebaseline_common.py` → `C_GRID = {C_GRID}`",
        "- Per-run selected c1/c2 are in `runs.csv` (`c1`, `c2` columns).",
        "",
        "## Hyperparameter columns present",
        "- **Yes**: `c1`, `c2`, `lambda`, `ridge_lambda_pos`, `ridge_lambda_neg`, `validation_macro_f1`.",
        "",
    ]
    (out / "inventory.md").write_text("\n".join(lines), encoding="utf-8")


def _collapsed_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (~df["classifier_ok"].astype(bool))
        | (df["finite_macro_f1"].astype(int) == 0)
        | (~np.isfinite(df["macro_f1"]))
    )


def task_iso_railing(a6: Path, out: Path) -> None:
    runs = pd.read_csv(a6 / "runs.csv")
    ridge = pd.read_csv(a6 / "ridge_matching.csv")
    geom = pd.read_csv(a6 / "scatter_geometry.csv")
    grid = list(C_GRID)
    c_min, c_max = min(grid), max(grid)

    ridge["ridge_trace_total"] = ridge["lambda_pos_trace_match"] + ridge["lambda_neg_trace_match"]
    geom = geom.rename(columns={"trace_isotropic": "geom_trace_isotropic"})

    rows: list[dict[str, Any]] = []
    for _, r in runs.iterrows():
        c1, c2 = float(r["c1"]), float(r["c2"])
        r1, p1 = _grid_rank(c1, grid)
        r2, p2 = _grid_rank(c2, grid)
        rg = ridge[(ridge["dataset"] == r["dataset"]) & (ridge["seed"] == r["seed"])].iloc[0]
        gm = geom[(geom["dataset"] == r["dataset"]) & (geom["seed"] == r["seed"])].iloc[0]
        trace_iso = float(r.get("trace_contribution", gm["geom_trace_isotropic"]))
        if r["arm"] in ("ISOTROPIC_SCATTER", "RIDGE_TRACE_MATCH"):
            eff_ridge = trace_iso
        else:
            eff_ridge = float("nan")
        rows.append(
            {
                "dataset": r["dataset"],
                "seed": int(r["seed"]),
                "tuning_mode": r["tuning_mode"],
                "prediction_rule": r["prediction_rule"],
                "arm": r["arm"],
                "c1": c1,
                "c2": c2,
                "c1_rank": r1,
                "c2_rank": r2,
                "c1_position": p1,
                "c2_position": p2,
                "c1_at_grid_minimum": int(p1 == "minimum"),
                "c2_at_grid_minimum": int(p2 == "minimum"),
                "ridge_trace_total": float(rg["ridge_trace_total"]),
                "trace_contribution_run": float(r.get("trace_contribution", np.nan)),
                "geom_trace_isotropic": float(gm["geom_trace_isotropic"]),
                "ridge_over_c1": eff_ridge / c1 if c1 > 0 and np.isfinite(eff_ridge) else np.nan,
                "ridge_over_c2": eff_ridge / c2 if c2 > 0 and np.isfinite(eff_ridge) else np.nan,
                "ridge_over_c_min": eff_ridge / c_min if np.isfinite(eff_ridge) else np.nan,
                "ridge_over_c_max": eff_ridge / c_max if np.isfinite(eff_ridge) else np.nan,
                "macro_f1": float(r["macro_f1"]),
            }
        )
    iso_df = pd.DataFrame(rows)
    iso_df.to_csv(out / "iso_railing.csv", index=False)

    summary_lines = ["# iso_railing_summary.md", ""]
    for mode in TUNING_MODES:
        summary_lines.append(f"## Railing rate ({mode})")
        sub = iso_df[iso_df["tuning_mode"] == mode]
        for arm in ARMS:
            a = sub[sub["arm"] == arm]
            summary_lines.append(
                f"- **{arm}**: c1@min={a['c1_at_grid_minimum'].mean():.3f}, "
                f"c2@min={a['c2_at_grid_minimum'].mean():.3f} (n={len(a)//2} runs per rule)"
            )
        summary_lines.append("")

    iso_only = iso_df[iso_df["arm"] == "ISOTROPIC_SCATTER"]
    dose_rows = []
    for ds in iso_only["dataset"].unique():
        for mode in TUNING_MODES:
            for rule in RULES:
                sub = iso_only[
                    (iso_only["dataset"] == ds)
                    & (iso_only["tuning_mode"] == mode)
                    & (iso_only["prediction_rule"] == rule)
                ]
                l1 = iso_df[
                    (iso_df["dataset"] == ds)
                    & (iso_df["arm"] == "L1_POPULATION")
                    & (iso_df["tuning_mode"] == mode)
                    & (iso_df["prediction_rule"] == rule)
                ]
                if sub.empty or l1.empty:
                    continue
                delta = float(sub["macro_f1"].mean() - l1["macro_f1"].mean())
                ridge_mag = float(sub["geom_trace_isotropic"].iloc[0])
                c_sel = float(sub["c1"].iloc[0])
                dose_rows.append(
                    {
                        "dataset": ds,
                        "tuning_mode": mode,
                        "prediction_rule": rule,
                        "iso_minus_l1": delta,
                        "log_ridge_over_c1": np.log(ridge_mag / c_sel) if c_sel > 0 else np.nan,
                    }
                )
    dose = pd.DataFrame(dose_rows)
    for mode in TUNING_MODES:
        for rule in RULES:
            d = dose[(dose["tuning_mode"] == mode) & (dose["prediction_rule"] == rule)]
            d = d[np.isfinite(d["log_ridge_over_c1"]) & np.isfinite(d["iso_minus_l1"])]
            if len(d) >= 3:
                rho, p = stats.spearmanr(d["log_ridge_over_c1"], d["iso_minus_l1"])
                summary_lines.append(
                    f"- Dose-response Spearman ({mode}, {rule}): rho={rho:.4f}, p={p:.4g} (n={len(d)} datasets)"
                )
    for mode in TUNING_MODES:
        for rule in RULES:
            d = dose[(dose["tuning_mode"] == mode) & (dose["prediction_rule"] == rule)]
            summary_lines.append(
                f"- ISOTROPIC−L1 mean delta ({mode}, {rule}): {d['iso_minus_l1'].mean():+.4f}"
            )
    summary_lines.append("")
    summary_lines.append(
        f"Effective ridge column: `geom_trace_isotropic` from scatter_geometry "
        f"(equals sum_j tr(S_j) for isotropic arm); per-class lambdas in ridge_matching."
    )
    (out / "iso_railing_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def _dataset_deltas(
    runs: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    rule: str,
    mode: str,
    policy: str,
) -> pd.Series:
    sub = runs[(runs["prediction_rule"] == rule) & (runs["tuning_mode"] == mode)].copy()
    collapsed = _collapsed_mask(sub)
    sub["macro_used"] = sub["macro_f1"]
    if policy == "exclude_collapsed":
        sub = sub[~collapsed]
    elif policy == "zero_collapsed":
        sub.loc[collapsed, "macro_used"] = 0.0
    wide = sub.pivot_table(
        index=["dataset", "seed"], columns="arm", values="macro_used", aggfunc="first"
    )
    if arm_a not in wide.columns or arm_b not in wide.columns:
        return pd.Series(dtype=float)
    return (wide[arm_a] - wide[arm_b]).groupby(level=0).mean()


def _bca_ci(data: np.ndarray, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return float("nan"), float("nan")
    res = stats.bootstrap(
        (data,),
        np.mean,
        n_resamples=N_BOOT,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def task_contrasts(a6: Path, out: Path) -> None:
    runs = pd.read_csv(a6 / "runs.csv")
    collapsed = _collapsed_mask(runs)
    collapse_rates = (
        runs.assign(collapsed=collapsed.astype(int))
        .groupby(["arm", "prediction_rule"])["collapsed"]
        .mean()
        .reset_index(name="collapse_rate")
    )

    rows: list[dict[str, Any]] = []
    holm_records: list[dict[str, Any]] = []
    for policy in ("exclude_collapsed", "zero_collapsed"):
        for rule in RULES:
            for mode in TUNING_MODES:
                for arm_a, arm_b in PRIMARY_CONTRASTS:
                    deltas = _dataset_deltas(runs, arm_a, arm_b, rule, mode, policy)
                    d = deltas.to_numpy()
                    d = d[np.isfinite(d)]
                    if d.size == 0:
                        continue
                    w = int(np.sum(d > 0))
                    t = int(np.sum(d == 0))
                    l = int(np.sum(d < 0))
                    try:
                        _, p = stats.wilcoxon(d, zero_method="wilcox")
                    except ValueError:
                        p = float("nan")
                    lo, hi = _bca_ci(d)
                    order = np.argsort(np.abs(d))[::-1]
                    top = deltas.iloc[order[:2]] if len(deltas) else pd.Series()
                    rows.append(
                        {
                            "contrast": f"{arm_a}_minus_{arm_b}",
                            "tuning_mode": mode,
                            "prediction_rule": rule,
                            "collapse_policy": policy,
                            "n_datasets": int(d.size),
                            "mean_delta": float(np.mean(d)),
                            "median_delta": float(np.median(d)),
                            "wins": w,
                            "ties": t,
                            "losses": l,
                            "wilcoxon_p": float(p),
                            "bca_ci_low": lo,
                            "bca_ci_high": hi,
                            "bootstrap_seed": BOOTSTRAP_SEED,
                            "bootstrap_n": N_BOOT,
                            "top1_dataset": top.index[0] if len(top) > 0 else "",
                            "top1_delta": float(top.iloc[0]) if len(top) > 0 else float("nan"),
                            "top2_dataset": top.index[1] if len(top) > 1 else "",
                            "top2_delta": float(top.iloc[1]) if len(top) > 1 else float("nan"),
                        }
                    )
                    if policy == "exclude_collapsed":
                        holm_records.append(
                            {
                                "contrast": f"{arm_a}_minus_{arm_b}",
                                "tuning_mode": mode,
                                "prediction_rule": rule,
                                "wilcoxon_p": float(p),
                            }
                        )

    cdf = pd.DataFrame(rows)
    cdf.to_csv(out / "contrast_distributions.csv", index=False)

    holm_df = pd.DataFrame(holm_records)
    holm_df = holm_df.sort_values("wilcoxon_p")
    m = len(holm_df)
    adj = []
    for i, p in enumerate(holm_df["wilcoxon_p"].fillna(1.0)):
        adj.append(min(1.0, p * (m - i)))
    holm_df["holm_p"] = adj
    holm_df["survives_0.05"] = holm_df["holm_p"] < 0.05

    md = [
        "# contrast_distributions.md",
        "",
        f"Bootstrap: BCa, n={N_BOOT}, seed={BOOTSTRAP_SEED}.",
        "",
        "## Collapse definition used",
        "- `collapsed` := (not classifier_ok) OR (finite_macro_f1==0) OR (macro_f1 non-finite).",
        "- `runs.csv` has no `collapsed` column; derived at analysis time.",
        "",
        "## Collapse rate by arm and rule",
        collapse_rates.to_string(index=False),
        "",
        "## Multiplicity (Holm-Bonferroni)",
        f"- Family: {m} tests = {len(PRIMARY_CONTRASTS)} contrasts × "
        f"{len(TUNING_MODES)} tuning modes × {len(RULES)} rules, "
        f"policy=exclude_collapsed, two-sided Wilcoxon on dataset-level deltas.",
        "",
        holm_df.to_string(index=False),
        "",
        "## FULL − ISOTROPIC under MODE B (per-dataset vectors)",
        "",
    ]
    for rule in RULES:
        d_repo = _dataset_deltas(
            runs, "FULL_SCATTER", "ISOTROPIC_SCATTER", rule, "MODE_B_PER_ARM_C", "exclude_collapsed"
        )
        md.append(f"### {rule}")
        md.append("```\n" + d_repo.sort_index().to_string() + "\n```")
        md.append("")
    (out / "contrast_distributions.md").write_text("\n".join(md), encoding="utf-8")


def _angle_deg(w_a: np.ndarray, w_b: np.ndarray) -> float:
    na, nb = np.linalg.norm(w_a), np.linalg.norm(w_b)
    if na == 0 or nb == 0:
        return float("nan")
    cos = abs(float(np.dot(w_a, w_b))) / (na * nb)
    cos = min(1.0, cos)
    return float(np.degrees(np.arccos(cos)))


def task_angles(params_dir: Path, out: Path) -> bool:
    manifest = params_dir / "manifest.csv"
    if not manifest.exists():
        (out / "angle_to_raw.md").write_text(
            "# angle_to_raw\n\nBlocked: no `params/manifest.csv`. Run `--dump-params` first.\n",
            encoding="utf-8",
        )
        return False

    man = pd.read_csv(manifest)
    rows: list[dict[str, Any]] = []
    for _, rec in man.iterrows():
        path = params_dir / rec["path"]
        if not path.exists():
            continue
        data = np.load(path)
        w1, w2 = data["w1"], data["w2"]
        b1, b2 = float(data["b1"]), float(data["b2"])
        n1, n2 = float(np.linalg.norm(w1)), float(np.linalg.norm(w2))
        r_ratio = n1 / n2 if n2 > 0 else float("nan")
        rows.append(
            {
                "dataset": rec["dataset"],
                "seed": int(rec["seed"]),
                "arm": rec["arm"],
                "tuning_mode": rec["tuning_mode"],
                "norm_w1": n1,
                "norm_w2": n2,
                "norm_ratio_w1_over_w2": r_ratio,
                "abs_log_norm_ratio": abs(np.log(r_ratio)) if r_ratio > 0 else float("nan"),
                "classifier_ok": bool(rec.get("classifier_ok", True)),
            }
        )
    base = pd.DataFrame(rows)
    raw = base[base["arm"] == "RAW_LSTSVM"]
    angle_rows: list[dict[str, Any]] = []
    for _, rec in man.iterrows():
        if rec["arm"] == "RAW_LSTSVM":
            continue
        raw_rec = man[
            (man["dataset"] == rec["dataset"])
            & (man["seed"] == rec["seed"])
            & (man["tuning_mode"] == rec["tuning_mode"])
            & (man["arm"] == "RAW_LSTSVM")
        ]
        if raw_rec.empty:
            continue
        d_arm = np.load(params_dir / rec["path"])
        d_raw = np.load(params_dir / raw_rec.iloc[0]["path"])
        angle_rows.append(
            {
                "dataset": rec["dataset"],
                "seed": int(rec["seed"]),
                "arm": rec["arm"],
                "tuning_mode": rec["tuning_mode"],
                "theta1_deg": _angle_deg(d_arm["w1"], d_raw["w1"]),
                "theta2_deg": _angle_deg(d_arm["w2"], d_raw["w2"]),
                "norm_w1_ratio": float(np.linalg.norm(d_arm["w1"]) / (np.linalg.norm(d_raw["w1"]) + 1e-12)),
                "norm_w2_ratio": float(np.linalg.norm(d_arm["w2"]) / (np.linalg.norm(d_raw["w2"]) + 1e-12)),
            }
        )
    adf = pd.DataFrame(angle_rows)
    adf.to_csv(out / "angle_to_raw.csv", index=False)

    md = ["# angle_to_raw.md", "", "## Theta summary (degrees, vs RAW)", ""]
    if adf.empty:
        md.append("No angle rows computed.")
    else:
        for arm in ARMS:
            sub = adf[adf["arm"] == arm]
            md.append(
                f"- **{arm}**: theta1 mean={sub['theta1_deg'].mean():.2f}, "
                f"median={sub['theta1_deg'].median():.2f}; "
                f"theta2 mean={sub['theta2_deg'].mean():.2f}, median={sub['theta2_deg'].median():.2f}"
            )
        md.append("")
        md.append("## Norm ratios vs RAW")
        for arm in ARMS:
            sub = adf[adf["arm"] == arm]
            md.append(
                f"- **{arm}**: ||w1||/||w1_RAW|| mean={sub['norm_w1_ratio'].mean():.3f}, "
                f"||w2||/||w2_RAW|| mean={sub['norm_w2_ratio'].mean():.3f}"
            )
        md.append("")
        md.append("## |log(w1/w2)| from params manifest norms")
        for arm in ARMS + ["RAW_LSTSVM"]:
            sub = base[base["arm"] == arm]
            md.append(f"- **{arm}**: mean={sub['abs_log_norm_ratio'].mean():.4f}")
    md.append("")
    md.append("## Pairwise theta contrasts (dataset mean, MODE B)")
    for mode in TUNING_MODES:
        for arm_a, arm_b in [
            ("FULL_SCATTER", "L1_POPULATION"),
            ("FULL_SCATTER", "DIAG_SCATTER"),
            ("FULL_SCATTER", "ISOTROPIC_SCATTER"),
        ]:
            sub = adf[adf["tuning_mode"] == mode]
            da = sub[sub["arm"] == arm_a].groupby("dataset")["theta1_deg"].mean()
            db = sub[sub["arm"] == arm_b].groupby("dataset")["theta1_deg"].mean()
            common = da.index.intersection(db.index)
            if len(common) < 2:
                continue
            d = (da.loc[common] - db.loc[common]).to_numpy()
            try:
                _, p = stats.wilcoxon(d, zero_method="wilcox")
            except ValueError:
                p = float("nan")
            lo, hi = _bca_ci(d)
            md.append(
                f"- {mode} theta1 {arm_a}-{arm_b}: mean={np.mean(d):.2f}, "
                f"median={np.median(d):.2f}, p={p:.4g}, BCa=[{lo:.2f},{hi:.2f}]"
            )
    (out / "angle_to_raw.md").write_text("\n".join(md), encoding="utf-8")
    return True


def write_report(out: Path, angles_ok: bool) -> None:
    text = f"""# A7 diagnostics REPORT

## Produced
- `inventory.md` (Task 0)
- `iso_railing.csv`, `iso_railing_summary.md` (Task 1)
- `contrast_distributions.csv`, `contrast_distributions.md` (Task 2)
- `angle_to_raw.csv`, `angle_to_raw.md` (Task 3: {"completed" if angles_ok else "blocked without params"})

## Task 4 (A6 script)
- Auto-label now uses **MODE_B_PER_ARM_C** + **RULE_REPO** only; printed as `A6_AUTO_LABEL_MODE_B_RULE_REPO: ...`

## Entry point
`python experiments/analysis/a7_diagnostics/run_a7_diagnostics.py --from-csv [--angles]`

## Schema notes
- Arms are suffixed (`L1_POPULATION`, `FULL_SCATTER`, ...), not bare `L1`/`FULL`.
- No `collapsed` column in `runs.csv`; collapse derived from `classifier_ok` and `finite_macro_f1`.
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    a6 = args.a6_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.dump_params:
        cmd = [
            sys.executable,
            str(GBLSTSVM / "experiments" / "test_estimator_a6_scatter_mechanism.py"),
            "--dump-params",
            "--output-dir",
            str(a6),
        ]
        subprocess.run(cmd, cwd=str(GBLSTSVM), check=True)
        args.angles = True

    if not (a6 / "runs.csv").exists():
        print(f"Missing {a6 / 'runs.csv'}", file=sys.stderr)
        return 1

    task_inventory(a6, out)
    task_iso_railing(a6, out)
    task_contrasts(a6, out)

    angles_ok = False
    if args.angles or (PARAMS_DIR / "manifest.csv").exists():
        angles_ok = task_angles(PARAMS_DIR if args.a6_dir == A6_OUT else a6 / "params", out)

    write_report(out, angles_ok)
    print(f"A7 diagnostics written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
