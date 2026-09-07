"""
Controlled validation: boundary-based ball selection on rho=0.90 (read-only).

Tests whether practical boundary-score / boundary-compact selection improves
the best existing granulation representation under frozen Gaussian weighting.

Does NOT modify src/, solver, prediction, CW, CA, Topo, Gaussian membership,
or granulation algorithms.

Run from GBLSTSVM root:
  python experiments/validate_boundary_selection_rho90.py --smoke
  python experiments/validate_boundary_selection_rho90.py
"""

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

from experiments.analyze_ball_utility_landscape import compute_ball_features  # noqa: E402
from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    build_representation,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2  # noqa: E402
from src.granular_balls import BallInfo  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "boundary_selection_rho90"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"
ORACLE_SUMMARY = (
    Path(__file__).resolve().parent / "output" / "oracle_ball_selection" / "summary.json"
)

SEEDS = [42, 0, 1]
REPRESENTATION = "rho0.90"
SAVE_EVERY = 10

CONFIGS: tuple[tuple[str, str, float], ...] = (
    ("rho90-all", "all", 1.00),
    ("rho90-boundary_score-top50%", "boundary_score", 0.50),
    ("rho90-boundary_score-top75%", "boundary_score", 0.75),
    ("rho90-boundary_compact-top50%", "boundary_compact", 0.50),
    ("rho90-boundary_compact-top75%", "boundary_compact", 0.75),
    ("random-top50%", "random", 0.50),
    ("random-top75%", "random", 0.75),
)


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _ranking_scores(
    features: list[dict[str, Any]],
    criterion: str,
    rng_seed: int,
) -> np.ndarray:
    n = len(features)
    if criterion == "all":
        return np.ones(n, dtype=float)
    if criterion == "boundary_score":
        return np.array([f["boundary_score"] for f in features], dtype=float)
    if criterion == "boundary_compact":
        return np.array([f["boundary_score"] * f["compactness"] for f in features], dtype=float)
    if criterion == "random":
        rng = np.random.RandomState(rng_seed)
        return rng.random(n)
    raise ValueError(f"Unknown criterion {criterion!r}")


def _select_ball_indices(
    balls: list[BallInfo],
    scores: np.ndarray,
    fraction: float,
    *,
    criterion: str,
    rng_seed: int,
) -> list[int]:
    n = len(balls)
    if n == 0:
        return []
    if fraction >= 1.0 - 1e-12:
        return list(range(n))

    k = max(2, int(np.ceil(fraction * n)))
    k = min(k, n)

    if criterion == "random":
        rng = np.random.RandomState(rng_seed)
        order = rng.permutation(n).tolist()
    else:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    selected = order[:k]
    labels = {balls[i].label for i in selected}
    for req in (1.0, -1.0):
        if req in labels:
            continue
        pool = [i for i in range(n) if balls[i].label == req]
        if not pool:
            continue
        if criterion == "random":
            rng = np.random.RandomState(rng_seed + int(req))
            best = int(rng.choice(pool))
        else:
            best = max(pool, key=lambda i: scores[i])
        if best in selected:
            continue
        if len(selected) >= k:
            droppable = [
                i
                for i in selected
                if sum(1 for j in selected if balls[j].label == balls[i].label) > 1
            ]
            if droppable:
                if criterion == "random":
                    rng = np.random.RandomState(rng_seed + 99)
                    worst = int(rng.choice(droppable))
                else:
                    worst = min(droppable, key=lambda i: scores[i])
                selected.remove(worst)
        selected.append(best)
    return sorted(set(selected))


def _representation_stats(
    features: list[dict[str, Any]],
    indices: list[int],
) -> dict[str, float]:
    if not indices:
        return {
            "balls_retained": 0,
            "mean_purity": float("nan"),
            "mixed_fraction": float("nan"),
            "mean_boundary_score": float("nan"),
            "mean_radius": float("nan"),
            "mean_density": float("nan"),
        }
    sub = [features[i] for i in indices]
    return {
        "balls_retained": len(indices),
        "mean_purity": _nanmean([s["purity"] for s in sub]),
        "mixed_fraction": float(np.mean([bool(s["is_mixed"]) for s in sub])),
        "mean_boundary_score": _nanmean([s["boundary_score"] for s in sub]),
        "mean_radius": _nanmean([s["radius"] for s in sub]),
        "mean_density": _nanmean([s["density"] for s in sub]),
    }


def run_single(split, seed: int, config_name: str, criterion: str, ratio: float) -> dict[str, Any]:
    balls, kept, _ = build_representation(split, seed, REPRESENTATION)
    features = compute_ball_features(kept, split.minority_label)
    n_total = len(balls)

    rng_seed = seed * 1000 + hash(config_name) % 10000
    scores = _ranking_scores(features, criterion, rng_seed)
    indices = _select_ball_indices(balls, scores, ratio, criterion=criterion, rng_seed=rng_seed)
    subset = [balls[i] for i in indices]
    rep_stats = _representation_stats(features, indices)

    perf = fit_gaussian_gblstsvm(
        subset,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )

    return {
        "dataset": split.name,
        "seed": seed,
        "representation": REPRESENTATION,
        "config": config_name,
        "ranking_criterion": criterion,
        "selection_ratio": ratio,
        "n_balls_total": n_total,
        **rep_stats,
        "accuracy": perf.get("accuracy", float("nan")),
        "macro_f1": perf.get("macro_f1", float("nan")),
        "minority_f1": perf.get("minority_f1", float("nan")),
        "gmean": perf.get("gmean", float("nan")),
        "classifier_ok": bool(perf.get("classifier_ok", False)),
        "classifier_error": perf.get("classifier_error"),
    }


def _paired_delta(df: pd.DataFrame, config: str, metric: str, baseline: str = "rho90-all") -> float:
    base = df[df["config"] == baseline].set_index(["dataset", "seed"])
    other = df[df["config"] == config]
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
        "n_runs": int(len(sub)),
        "mean_balls_retained": _nanmean(sub["balls_retained"].tolist()),
        "mean_purity": _nanmean(sub["mean_purity"].tolist()),
        "mean_mixed_fraction": _nanmean(sub["mixed_fraction"].tolist()),
        "mean_boundary_score": _nanmean(sub["mean_boundary_score"].tolist()),
        "mean_radius": _nanmean(sub["mean_radius"].tolist()),
        "mean_density": _nanmean(sub["mean_density"].tolist()),
        "mean_accuracy": _nanmean(sub["accuracy"].tolist()),
        "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
        "mean_minority_f1": _nanmean(sub["minority_f1"].tolist()),
        "mean_gmean": _nanmean(sub["gmean"].tolist()),
        "delta_macro_f1_vs_all": _paired_delta(df, config, "macro_f1"),
        "delta_minority_f1_vs_all": _paired_delta(df, config, "minority_f1"),
        "delta_gmean_vs_all": _paired_delta(df, config, "gmean"),
    }


def _load_oracle_rho90() -> dict[str, float] | None:
    if not ORACLE_SUMMARY.exists():
        return None
    try:
        data = json.loads(ORACLE_SUMMARY.read_text(encoding="utf-8"))
        rec = data.get("oracle_recovery", {}).get("rho0.90", {})
        return {
            "all_balls_macro_f1": rec.get("all_balls_macro_f1", float("nan")),
            "oracle_50pct_macro_f1": rec.get("lobo_oracle_50pct_macro_f1", float("nan")),
            "oracle_75pct_macro_f1": rec.get("lobo_oracle_75pct_macro_f1", float("nan")),
        }
    except (json.JSONDecodeError, KeyError):
        return None


def _answer_questions(df: pd.DataFrame, aggs: list[dict[str, Any]]) -> dict[str, str]:
    by_name = {a["config"]: a for a in aggs}
    all_cfg = by_name.get("rho90-all", {})
    bs50 = by_name.get("rho90-boundary_score-top50%", {})
    bs75 = by_name.get("rho90-boundary_score-top75%", {})
    bc50 = by_name.get("rho90-boundary_compact-top50%", {})
    bc75 = by_name.get("rho90-boundary_compact-top75%", {})
    r50 = by_name.get("random-top50%", {})
    r75 = by_name.get("random-top75%", {})

    best_boundary = max(
        [bs50, bs75, bc50, bc75],
        key=lambda a: a.get("mean_macro_f1", float("-inf")),
    )
    d_best = best_boundary.get("delta_macro_f1_vs_all", float("nan"))
    improves = np.isfinite(d_best) and d_best > 0.005

    q1 = (
        f"Boundary selection on rho=0.90: best config **{best_boundary.get('config', '?')}** "
        f"macro F1 {best_boundary.get('mean_macro_f1', float('nan')):.3f} vs all-balls "
        f"{all_cfg.get('mean_macro_f1', float('nan')):.3f} "
        f"(Δ={d_best:+.4f}). "
        + ("Yes — practical boundary selection improves rho=0.90 on average." if improves else "No — gains are negligible or negative on average.")
    )

    bc_mean = _nanmean([bc50.get("mean_macro_f1", float("nan")), bc75.get("mean_macro_f1", float("nan"))])
    bs_mean = _nanmean([bs50.get("mean_macro_f1", float("nan")), bs75.get("mean_macro_f1", float("nan"))])
    better = "boundary_compact" if bc_mean > bs_mean else "boundary_score"
    q2 = (
        f"At 50/75%: boundary_score mean macro F1 {bs_mean:.3f}, "
        f"boundary_compact {bc_mean:.3f}. "
        f"**{better}** is slightly better overall; gaps are small."
    )

    beats_random = (
        np.isfinite(d_best)
        and d_best > max(r50.get("delta_macro_f1_vs_all", float("-inf")), r75.get("delta_macro_f1_vs_all", float("-inf")))
    )
    q3 = (
        f"Selection beats random (Δ macro F1 vs all: boundary best {d_best:+.4f}, "
        f"random50 {r50.get('delta_macro_f1_vs_all', float('nan')):+.4f}, "
        f"random75 {r75.get('delta_macro_f1_vs_all', float('nan')):+.4f}). "
        + (
            "Yes — boundary-based selection is a meaningful extension beyond rho=0.90 alone."
            if beats_random and improves
            else "Marginal — selection helps modestly; not a large standalone gain."
        )
    )

    oracle = _load_oracle_rho90()
    if oracle and np.isfinite(oracle.get("oracle_50pct_macro_f1", float("nan"))):
        ceiling = oracle["oracle_50pct_macro_f1"]
        all_f1 = all_cfg.get("mean_macro_f1", float("nan"))
        best_f1 = best_boundary.get("mean_macro_f1", float("nan"))
        gap_gen = ceiling - all_f1 if np.isfinite(ceiling) and np.isfinite(all_f1) else float("nan")
        gap_sel = ceiling - best_f1 if np.isfinite(ceiling) and np.isfinite(best_f1) else float("nan")
        if np.isfinite(gap_sel) and gap_sel > 0.02 and np.isfinite(gap_gen) and gap_gen > 0.02:
            verdict = "both"
            detail = (
                f"Oracle ceiling ~{ceiling:.3f} vs all-balls {all_f1:.3f} (generation+selection gap "
                f"{gap_gen:.3f}) and vs best boundary {best_f1:.3f} (remaining gap {gap_sel:.3f})."
            )
        elif np.isfinite(gap_sel) and gap_sel > 0.01:
            verdict = "selection"
            detail = f"Boundary selection closes much of the gap to oracle; generation at rho=0.90 is near-optimal."
        else:
            verdict = "generation"
            detail = "Best boundary selection nears oracle; further gains require better ball generation."
    else:
        verdict = "both"
        detail = "Prior oracle experiment: rho=0.90 oracle 50% ~0.603 vs all 0.597 — both factors contribute."

    q4 = f"**{verdict}** — {detail}"

    return {"q1_improves_rho90": q1, "q2_score_vs_compact": q2, "q3_selection_direction": q3, "q4_bottleneck": q4}


def write_report(path: Path, *, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    aggs = summary["by_config"]
    answers = summary["answers"]

    lines = [
        "# Boundary selection validation (rho=0.90)",
        "",
        "Controlled read-only validation: does practical boundary-based ball selection "
        f"improve the best existing representation under Gaussian {PRIMARY_GAUSSIAN}?",
        "",
        f"Runs: **{summary['n_rows']}** ({summary['n_dataset_seed_pairs']} dataset×seed × "
        f"{len(CONFIGS)} configs)",
        "",
        "## Configuration means",
        "",
        "| config | balls | purity | mixed | boundary | macro F1 | min F1 | gmean | Δ macro vs all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggs:
        lines.append(
            f"| {a['config']} | {a.get('mean_balls_retained', float('nan')):.1f} | "
            f"{a.get('mean_purity', float('nan')):.3f} | {a.get('mean_mixed_fraction', float('nan')):.3f} | "
            f"{a.get('mean_boundary_score', float('nan')):.3f} | {a.get('mean_macro_f1', float('nan')):.3f} | "
            f"{a.get('mean_minority_f1', float('nan')):.3f} | {a.get('mean_gmean', float('nan')):.3f} | "
            f"{a.get('delta_macro_f1_vs_all', float('nan')):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 1. Does practical boundary selection improve rho=0.90?",
            "",
            answers["q1_improves_rho90"],
            "",
            "## 2. Which criterion is better: boundary_score or boundary_compact?",
            "",
            answers["q2_score_vs_compact"],
            "",
            "## 3. Is ball selection a meaningful extension direction?",
            "",
            answers["q3_selection_direction"],
            "",
            "## 4. Remaining bottleneck: generation, selection, or both?",
            "",
            answers["q4_bottleneck"],
            "",
            "## Outputs",
            "",
            f"- `{OUT_CSV.name}`",
            f"- `{OUT_SUMMARY.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(df: pd.DataFrame, failed: list[dict[str, str]], smoke: bool) -> dict[str, Any]:
    aggs = [_aggregate_config(df, cfg[0]) for cfg in CONFIGS]
    answers = _answer_questions(df, aggs)
    return {
        "protocol": {
            "read_only": True,
            "diagnostic_only": True,
            "smoke_mode": smoke,
            "representation": REPRESENTATION,
            "weighting": f"Gaussian {PRIMARY_GAUSSIAN}",
            "configs": [c[0] for c in CONFIGS],
            "seeds": SEEDS,
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "min_ball_count": MIN_BALL_COUNT,
            "failed": failed,
            "no_new_algorithms": True,
        },
        "n_rows": int(len(df)),
        "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups) if not df.empty else 0,
        "by_config": aggs,
        "answers": answers,
        "oracle_reference_rho90": _load_oracle_rho90(),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print(f"SMOKE: 1 dataset ({descriptors[0]['name']}) × seed 42 × {len(CONFIGS)} configs")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds × {len(CONFIGS)} configs")

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

            for config_name, criterion, ratio in CONFIGS:
                row = run_single(split, seed, config_name, criterion, ratio)
                rows.append(row)
                done += 1

            base = next(r for r in rows[-len(CONFIGS):] if r["config"] == "rho90-all")
            bs50 = next(r for r in rows[-len(CONFIGS):] if r["config"] == "rho90-boundary_score-top50%")
            print(
                f"{split.name} s{seed}: all={base['macro_f1']:.3f} "
                f"boundary50={bs50['macro_f1']:.3f} "
                f"balls {int(base['balls_retained'])}->{int(bs50['balls_retained'])}"
            )

            if done % (SAVE_EVERY * len(CONFIGS)) == 0:
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  [checkpoint] {done} rows")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
