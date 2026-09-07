"""
Budget-controlled granulation experiment (read-only extension).

Tests whether explicit ball-count budget B can replace rho purity stopping
for GBLSTSVM granulation. Implements a new granulation wrapper only; does not
modify src/, solver, prediction, CW, CA, Topo, Gaussian membership, or
baseline granulation.

Compares:
  - Baseline: rho=0.90 purity stopping
  - Budget-controlled: stop when kept ball count reaches target B
  - Budget fractions: 0.5/0.75/1.0/1.25 × rho90 reference ball count
  - Dataset-scaled: B = k × sqrt(n)

Run from GBLSTSVM root:
  python experiments/test_complexity_budget_gblstsvm.py --smoke
  python experiments/test_complexity_budget_gblstsvm.py
  python experiments/test_complexity_budget_gblstsvm.py --from-csv
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_complexity_budget import compute_representation_stats  # noqa: E402
from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    RHO_VALUES,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2, normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _to_ball_info,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "complexity_budget_gblstsvm"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
RHO_REF = RHO_VALUES["rho0.90"]
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
SAVE_EVERY = 10

BUDGET_FRACS = (0.5, 0.75, 1.0, 1.25)
SQRT_K_VALUES = (2, 4, 6, 8)

BASELINE_CONFIG = "rho90-baseline"


@dataclass(frozen=True)
class GranulationConfig:
    name: str
    family: str  # "rho" | "budget_frac" | "sqrt"
    target_balls: int | None = None
    budget_frac: float | None = None
    sqrt_k: float | None = None
    rho: float | None = None


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _del_ball_kept(balls: list[GranularBall]) -> list[GranularBall]:
    return [b for b in balls if int(b.num) >= MIN_BALL_COUNT]


def build_budget_granulation(
    indexed: np.ndarray,
    *,
    seed: int,
    target_balls: int,
) -> tuple[list[GranularBall], dict[str, Any]]:
    """
    Budget-controlled granulation wrapper.

    Replaces rho purity stopping with a fixed ball-count budget B.
    Uses the same KMeans(k=2) split primitive and scan order as baseline
    granulation. Only accepts splits that keep both children above del_ball
    threshold; records snapshots after each accepted split and returns the
    snapshot whose kept ball count is closest to B.
    """
    target = max(MIN_BALL_COUNT, int(target_balls))
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    ll = len(balls)
    i = 0
    n_splits = 0
    snapshots: list[tuple[int, int, list[GranularBall]]] = []

    def _record() -> None:
        kept = _del_ball_kept(balls)
        snapshots.append((len(kept), n_splits, copy.deepcopy(kept)))

    _record()

    while i < ll:
        kept = _del_ball_kept(balls)
        if len(kept) >= target:
            break
        ball = balls[i]
        if ball.num > MIN_BALL_COUNT:
            c1, c2 = ball.split_2balls()
            if int(c1.num) >= MIN_BALL_COUNT and int(c2.num) >= MIN_BALL_COUNT:
                balls[i] = c1
                balls.append(c2)
                ll += 1
                n_splits += 1
                _record()
            else:
                i += 1
        else:
            i += 1

    if not snapshots:
        kept = _del_ball_kept(balls)
        return kept, {
            "target_balls": target,
            "n_balls": len(kept),
            "n_splits": n_splits,
            "stopped_by_budget": len(kept) >= target,
        }

    best_nb, best_ns, best_balls = min(
        snapshots,
        key=lambda t: (abs(t[0] - target), t[1]),
    )
    return best_balls, {
        "target_balls": target,
        "n_balls": best_nb,
        "n_splits": best_ns,
        "stopped_by_budget": best_nb >= target,
    }


def build_rho_granulation(
    indexed: np.ndarray,
    *,
    seed: int,
    purity: float = RHO_REF,
) -> tuple[list[GranularBall], dict[str, Any]]:
    """Baseline rho purity stopping (unchanged algorithm, local wrapper)."""
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    ll = len(balls)
    i = 0
    n_splits = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < purity and ball.num > MIN_SAMPLES_PER_BALL:
            c1, c2 = ball.split_2balls()
            balls[i] = c1
            balls.append(c2)
            ll += 1
            n_splits += 1
        else:
            i += 1

    kept = _del_ball_kept(balls)
    return kept, {
        "target_balls": None,
        "n_balls": len(kept),
        "n_splits": n_splits,
        "rho": purity,
        "stopped_by_budget": False,
    }


def _global_min_label(indexed: np.ndarray) -> float:
    y_all = indexed[:, -2]
    return 1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0


def _make_configs(
    rho90_balls: int,
    n_samples: int,
) -> list[GranulationConfig]:
    configs: list[GranulationConfig] = [
        GranulationConfig(
            name=BASELINE_CONFIG,
            family="rho",
            rho=RHO_REF,
        ),
    ]
    for frac in BUDGET_FRACS:
        pct = int(round(frac * 100))
        configs.append(
            GranulationConfig(
                name=f"budget-{pct}pct-rho90",
                family="budget_frac",
                budget_frac=frac,
                target_balls=max(MIN_BALL_COUNT, int(round(frac * rho90_balls))),
            )
        )
    for k in SQRT_K_VALUES:
        configs.append(
            GranulationConfig(
                name=f"sqrt-k{k}",
                family="sqrt",
                sqrt_k=float(k),
                target_balls=max(MIN_BALL_COUNT, int(round(k * np.sqrt(n_samples)))),
            )
        )
    return configs


def run_single(
    split,
    seed: int,
    config: GranulationConfig,
    *,
    indexed: np.ndarray,
    global_min: float,
    rho90_balls: int,
) -> dict[str, Any]:
    if config.family == "rho":
        kept, gran_stats = build_rho_granulation(indexed, seed=seed, purity=RHO_REF)
        target_balls = int(gran_stats["n_balls"])
    else:
        target_balls = int(config.target_balls or MIN_BALL_COUNT)
        kept, gran_stats = build_budget_granulation(
            indexed, seed=seed, target_balls=target_balls
        )

    rep = compute_representation_stats(kept, indexed, global_min)
    balls = [_to_ball_info(b) for b in kept]
    perf = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    n_train = int(indexed.shape[0])
    return {
        "dataset": split.name,
        "seed": seed,
        "config": config.name,
        "family": config.family,
        "budget_frac": config.budget_frac if config.budget_frac is not None else float("nan"),
        "sqrt_k": config.sqrt_k if config.sqrt_k is not None else float("nan"),
        "target_balls": target_balls,
        "rho90_reference_balls": rho90_balls,
        "budget_as_pct_rho90": (
            100.0 * target_balls / rho90_balls if rho90_balls > 0 else float("nan")
        ),
        "n_train": n_train,
        "n_balls": rep["n_balls"],
        "n_splits": gran_stats["n_splits"],
        "stopped_by_budget": bool(gran_stats.get("stopped_by_budget", False)),
        "minority_survival": rep["minority_survival"],
        "mean_purity": rep["mean_purity"],
        "mean_radius": rep["mean_radius"],
        "mean_boundary_score": rep["mean_boundary_score"],
        "mean_log_boundary_score": rep["mean_log_boundary_score"],
        "accuracy": perf.get("accuracy", float("nan")),
        "macro_f1": perf.get("macro_f1", float("nan")),
        "minority_f1": perf.get("minority_f1", float("nan")),
        "gmean": perf.get("gmean", float("nan")),
        "classifier_ok": bool(perf.get("classifier_ok", False)),
        "classifier_error": perf.get("classifier_error"),
    }


def _paired_delta(
    df: pd.DataFrame,
    config: str,
    metric: str,
    baseline: str = BASELINE_CONFIG,
) -> float:
    base = df[
        (df["config"] == baseline) & df["classifier_ok"].astype(bool)
    ].set_index(["dataset", "seed"])
    other = df[(df["config"] == config) & df["classifier_ok"].astype(bool)]
    deltas: list[float] = []
    for _, row in other.iterrows():
        key = (row["dataset"], int(row["seed"]))
        if key not in base.index:
            continue
        bv = float(base.loc[key, metric])
        mv = float(row[metric])
        if np.isfinite(bv) and np.isfinite(mv):
            deltas.append(mv - bv)
    return _nanmean(deltas)


def _aggregate_config(df: pd.DataFrame, config: str) -> dict[str, Any]:
    sub = df[(df["config"] == config) & df["classifier_ok"].astype(bool)]
    if sub.empty:
        return {"config": config, "n_runs": 0}
    return {
        "config": config,
        "family": sub["family"].iloc[0],
        "n_runs": int(len(sub)),
        "mean_target_balls": _nanmean(sub["target_balls"].tolist()),
        "mean_n_balls": _nanmean(sub["n_balls"].tolist()),
        "mean_budget_pct_rho90": _nanmean(sub["budget_as_pct_rho90"].tolist()),
        "mean_purity": _nanmean(sub["mean_purity"].tolist()),
        "mean_radius": _nanmean(sub["mean_radius"].tolist()),
        "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
        "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
        "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
        "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
        "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
        "mean_gmean": _nanmean(sub["gmean"].tolist()),
        "delta_macro_f1_vs_rho90": _paired_delta(df, config, "macro_f1"),
        "delta_minority_f1_vs_rho90": _paired_delta(df, config, "minority_f1"),
        "delta_gmean_vs_rho90": _paired_delta(df, config, "gmean"),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(np.sum(mask))}


def _ball_count_correlations(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["classifier_ok"].astype(bool)].copy()
    ok = ok[ok["config"] != BASELINE_CONFIG]
    if ok.empty:
        return {}

    base = df[df["config"] == BASELINE_CONFIG].set_index(["dataset", "seed"])
    deltas: list[float] = []
    balls: list[float] = []
    for _, row in ok.iterrows():
        key = (row["dataset"], int(row["seed"]))
        if key not in base.index:
            continue
        bv = float(base.loc[key, "macro_f1"])
        mv = float(row["macro_f1"])
        if np.isfinite(bv) and np.isfinite(mv):
            deltas.append(mv - bv)
            balls.append(float(row["n_balls"]))

    x = np.asarray(balls, dtype=float)
    y = np.asarray(deltas, dtype=float)
    return {
        "delta_macro_f1_vs_n_balls": _safe_corr(x, y),
        "macro_f1_vs_n_balls_all_configs": _safe_corr(
            ok["n_balls"].to_numpy(dtype=float),
            ok["macro_f1"].to_numpy(dtype=float),
        ),
    }


def _best_config(aggs: list[dict[str, Any]], family: str | None = None) -> dict[str, Any]:
    pool = [a for a in aggs if a.get("n_runs", 0) > 0 and a["config"] != BASELINE_CONFIG]
    if family:
        pool = [a for a in pool if a.get("family") == family]
    if not pool:
        return {}
    return max(pool, key=lambda a: a.get("mean_macro_f1", float("-inf")))


def _answer_questions(
    df: pd.DataFrame,
    aggs: list[dict[str, Any]],
    correlations: dict[str, Any],
) -> dict[str, str]:
    by_name = {a["config"]: a for a in aggs}
    ref = by_name.get(BASELINE_CONFIG, {})
    best_overall = _best_config(aggs)
    best_frac = _best_config(aggs, "budget_frac")
    best_sqrt = _best_config(aggs, "sqrt")
    budget100 = by_name.get("budget-100pct-rho90", {})

    ball_err = float("nan")
    if "budget-100pct-rho90" in by_name:
        base_balls = df[df["config"] == BASELINE_CONFIG].set_index(["dataset", "seed"])["n_balls"]
        b100 = df[df["config"] == "budget-100pct-rho90"]
        errs: list[float] = []
        for _, row in b100.iterrows():
            key = (row["dataset"], int(row["seed"]))
            if key in base_balls.index:
                errs.append(abs(float(row["n_balls"]) - float(base_balls.loc[key])))
        ball_err = _nanmean(errs)

    macro_gap_100 = budget100.get("delta_macro_f1_vs_rho90", float("nan"))
    replaces = (
        np.isfinite(macro_gap_100)
        and abs(macro_gap_100) <= 0.01
        and np.isfinite(ball_err)
        and ball_err <= 5
    )

    q1 = (
        f"rho=0.90 baseline: {ref.get('mean_n_balls', float('nan')):.1f} balls, "
        f"macro F1 {ref.get('mean_macro_f1', float('nan')):.3f}. "
        f"Budget-100%rho90 wrapper: {budget100.get('mean_n_balls', float('nan')):.1f} balls, "
        f"F1 {budget100.get('mean_macro_f1', float('nan')):.3f} (Δ {macro_gap_100:+.4f}). "
        + (
            "**Yes** — explicit ball-count stopping at the rho90-calibrated budget "
            "reproduces rho=0.90 without a purity threshold."
            if replaces
            else "**Partially** — budget stopping reaches similar ball counts but "
            "split selection without purity differs from rho stopping on some datasets."
        )
    )

    q2 = (
        f"Best budget-frac: **{best_frac.get('config', '?')}** "
        f"(F1 {best_frac.get('mean_macro_f1', float('nan')):.3f}, "
        f"Δ {best_frac.get('delta_macro_f1_vs_rho90', float('nan')):+.4f}). "
        f"Best sqrt-scaled: **{best_sqrt.get('config', '?')}** "
        f"(F1 {best_sqrt.get('mean_macro_f1', float('nan')):.3f}, "
        f"Δ {best_sqrt.get('delta_macro_f1_vs_rho90', float('nan')):+.4f}). "
        f"Overall best non-baseline: **{best_overall.get('config', '?')}**."
    )

    best_delta = best_overall.get("delta_macro_f1_vs_rho90", float("nan"))
    improves = np.isfinite(best_delta) and best_delta > 0.005
    q3 = (
        f"Best config **{best_overall.get('config', '?')}** macro F1 "
        f"{best_overall.get('mean_macro_f1', float('nan')):.3f} vs rho90 "
        f"{ref.get('mean_macro_f1', float('nan')):.3f} (Δ {best_delta:+.4f}). "
        + (
            "**Yes** — some budget settings improve over rho=0.90 on average."
            if improves
            else "**No** — rho=0.90 outperforms all budget-only wrappers on macro F1; "
            "matching ball count without purity-based split selection is insufficient."
        )
    )

    corr = correlations.get("delta_macro_f1_vs_n_balls", {})
    spearman = corr.get("spearman", float("nan"))
    q4 = (
        f"Paired Δ macro F1 (vs rho90) vs n_balls: spearman={spearman:.3f} "
        f"(n={corr.get('n', 0)}). "
        + (
            "**Yes** — ball count explains performance variation across budget configs."
            if np.isfinite(spearman) and abs(spearman) > 0.2
            else "**Weakly** — gains are not fully explained by ball count alone; "
            "which balls are split (purity vs budget order) also matters."
        )
    )

    return {
        "q1_replace_rho": q1,
        "q2_best_scaling": q2,
        "q3_improve_over_rho90": q3,
        "q4_ball_count_explains": q4,
    }


def write_report(path: Path, *, summary: dict[str, Any]) -> None:
    aggs = summary["by_config"]
    answers = summary["answers"]
    correlations = summary.get("ball_count_correlations", {})

    lines = [
        "# Complexity budget GBLSTSVM experiment",
        "",
        "New granulation wrapper: stop at target ball budget **B** instead of rho purity.",
        f"Baseline: rho={RHO_REF} | Budget fracs: {list(BUDGET_FRACS)} × rho90 balls | "
        f"Sqrt budgets: k×sqrt(n), k∈{list(SQRT_K_VALUES)}",
        f"Classifier: Gaussian-weighted GBLSTSVM ({PRIMARY_GAUSSIAN}) | "
        f"Runs: **{summary['n_rows']}**",
        "",
        "## Configuration means",
        "",
        "| config | family | balls | purity | boundary | min surv | macro F1 | min F1 | gmean | Δ macro |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggs:
        lines.append(
            f"| {a['config']} | {a.get('family', '')} | "
            f"{a.get('mean_n_balls', float('nan')):.1f} | "
            f"{a.get('mean_purity', float('nan')):.3f} | "
            f"{a.get('mean_boundary_score', float('nan')):.3f} | "
            f"{a.get('mean_minority_survival', float('nan')):.3f} | "
            f"{a.get('mean_macro_f1', float('nan')):.3f} | "
            f"{a.get('mean_minority_f1', float('nan')):.3f} | "
            f"{a.get('mean_gmean', float('nan')):.3f} | "
            f"{a.get('delta_macro_f1_vs_rho90', float('nan')):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 1. Can explicit complexity control replace rho?",
            "",
            answers["q1_replace_rho"],
            "",
            "## 2. Which budget scaling works best?",
            "",
            answers["q2_best_scaling"],
            "",
            "## 3. Does performance improve over rho90?",
            "",
            answers["q3_improve_over_rho90"],
            "",
            "## 4. Does ball count explain gains?",
            "",
            answers["q4_ball_count_explains"],
            "",
            "## Ball-count correlations",
            "",
        ]
    )
    for name, vals in correlations.items():
        if isinstance(vals, dict):
            lines.append(
                f"- {name}: pearson={vals.get('pearson', float('nan')):.3f}, "
                f"spearman={vals.get('spearman', float('nan')):.3f} (n={vals.get('n', 0)})"
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{OUT_CSV.name}`",
            f"- `{OUT_SUMMARY.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _all_config_names() -> list[str]:
    names = [BASELINE_CONFIG]
    for frac in BUDGET_FRACS:
        names.append(f"budget-{int(round(frac * 100))}pct-rho90")
    for k in SQRT_K_VALUES:
        names.append(f"sqrt-k{k}")
    return names


def _build_summary(
    df: pd.DataFrame,
    failed: list[dict[str, str]],
    smoke: bool,
) -> dict[str, Any]:
    names = _all_config_names()
    aggs = [_aggregate_config(df, n) for n in names]
    correlations = _ball_count_correlations(df)
    answers = _answer_questions(df, aggs, correlations)
    return {
        "protocol": {
            "read_only_extension": True,
            "smoke_mode": smoke,
            "baseline": BASELINE_CONFIG,
            "reference_rho": RHO_REF,
            "budget_fracs": list(BUDGET_FRACS),
            "sqrt_k_values": list(SQRT_K_VALUES),
            "granulation_wrapper": "build_budget_granulation (ball-count stop, no rho)",
            "seeds": SEEDS,
            "classifier": f"Gaussian-weighted GBLSTSVM ({PRIMARY_GAUSSIAN})",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "min_ball_count": MIN_BALL_COUNT,
            "configs": names,
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups) if not df.empty else 0,
        "by_config": aggs,
        "ball_count_correlations": correlations,
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print(f"SMOKE: 1 dataset × seed 42")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append({"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"})
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue

            try:
                y_train = normalize_labels(split.y_train)
                indexed = _build_indexed_matrix(split.X_train, y_train)
                global_min = _global_min_label(indexed)

                rho_kept, rho_stats = build_rho_granulation(indexed, seed=seed, purity=RHO_REF)
                rho90_balls = int(rho_stats["n_balls"])
                configs = _make_configs(rho90_balls, int(indexed.shape[0]))

                for config in configs:
                    row = run_single(
                        split,
                        seed,
                        config,
                        indexed=indexed,
                        global_min=global_min,
                        rho90_balls=rho90_balls,
                    )
                    rows.append(row)
                    done += 1

                ref = next(r for r in rows[-len(configs):] if r["config"] == BASELINE_CONFIG)
                best_budget = max(
                    rows[-len(configs):],
                    key=lambda r: r["macro_f1"] if r["config"] != BASELINE_CONFIG else -1,
                )
                print(
                    f"{split.name} s{seed}: rho90 F1={ref['macro_f1']:.3f} "
                    f"balls={ref['n_balls']} | best {best_budget['config']} "
                    f"F1={best_budget['macro_f1']:.3f} balls={best_budget['n_balls']}"
                )

                if done % (SAVE_EVERY * len(configs)) == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    print(f"  [checkpoint] {done} rows")

            except Exception as exc:
                failed.append({"dataset": split.name, "seed": str(seed), "error": str(exc)})
                print(f"FAIL {desc['name']} seed={seed}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
