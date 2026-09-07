"""
Oracle ball selection diagnostic (read-only).

Tests whether GBLSTSVM performance is limited by bad ball generation or bad
ball selection by ranking and subsampling balls under frozen Gaussian weighting.

Does NOT modify src/, solver, prediction, CW, CA, Topo, Gaussian membership,
or granulation algorithms.

Run from GBLSTSVM root:
  python experiments/oracle_ball_selection_analysis.py --smoke
  python experiments/oracle_ball_selection_analysis.py
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

from experiments.analyze_ball_utility_landscape import (  # noqa: E402
    _lobo_valid,
    compute_ball_features,
)
from experiments.analyze_granular_geometry_vs_performance import (  # noqa: E402
    build_representation,
    fit_gaussian_gblstsvm,
)
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import PRIMARY_GAUSSIAN  # noqa: E402
from experiments.phase8_common import DEFAULT_C1, DEFAULT_C2  # noqa: E402
from src.granular_balls import BallInfo, balls_to_class_arrays  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "oracle_ball_selection"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"
LOBO_CACHE_PATH = (
    Path(__file__).resolve().parent / "output" / "ball_utility_landscape" / "ball_records.csv"
)

SEEDS = [42, 0, 1]
SU3_PERCENTILE = 75
REG_TERM = 1e-4
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"
SAVE_EVERY = 10

REPRESENTATIONS = (
    "B0",
    "rho0.80",
    "rho0.85",
    "rho0.90",
    "rho0.95",
    "SU3",
)

SELECTION_RATIOS = (0.25, 0.50, 0.75, 1.00)

RANKING_CRITERIA = (
    "purity",
    "boundary_score",
    "density",
    "entropy",
    "boundary_compact",
    "random",
    "lobo_oracle",
)


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _load_lobo_cache() -> pd.DataFrame | None:
    if not LOBO_CACHE_PATH.exists():
        return None
    df = pd.read_csv(LOBO_CACHE_PATH)
    needed = {"dataset", "seed", "representation", "ball_id", "delta_macro_f1", "lobo_ok"}
    if not needed.issubset(df.columns):
        return None
    return df


def _lobo_usefulness_from_cache(
    cache: pd.DataFrame,
    dataset: str,
    seed: int,
    representation: str,
    n_balls: int,
) -> list[float] | None:
    sub = cache[
        (cache["dataset"] == dataset)
        & (cache["seed"] == seed)
        & (cache["representation"] == representation)
    ]
    if sub.empty:
        return None
    by_id = sub.set_index("ball_id")
    scores: list[float] = []
    for i in range(n_balls):
        if i not in by_id.index:
            return None
        row = by_id.loc[i]
        if not bool(row.get("lobo_ok", False)) or not np.isfinite(row["delta_macro_f1"]):
            scores.append(float("-inf"))
        else:
            scores.append(float(-row["delta_macro_f1"]))
    return scores


def _compute_lobo_usefulness(
    balls: list[BallInfo],
    split,
    seed: int,
) -> list[float]:
    baseline = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )
    base_macro = baseline["macro_f1"]
    baseline_ok = bool(baseline.get("classifier_ok", False))
    scores: list[float] = []
    for i in range(len(balls)):
        if not baseline_ok or not _lobo_valid(balls, i):
            scores.append(float("-inf"))
            continue
        remaining = [b for j, b in enumerate(balls) if j != i]
        lobo = fit_gaussian_gblstsvm(
            remaining,
            y_train=split.y_train,
            X_test=split.X_test,
            y_test=split.y_test,
            minority_label=split.minority_label,
            seed=seed,
        )
        if lobo.get("classifier_ok", False) and np.isfinite(lobo["macro_f1"]):
            scores.append(float(base_macro - lobo["macro_f1"]))
        else:
            scores.append(float("-inf"))
    return scores


def _ranking_scores(
    features: list[dict[str, Any]],
    criterion: str,
    *,
    lobo_scores: list[float] | None,
    rng_seed: int,
) -> np.ndarray:
    n = len(features)
    if criterion == "purity":
        return np.array([f["purity"] for f in features], dtype=float)
    if criterion == "boundary_score":
        return np.array([f["boundary_score"] for f in features], dtype=float)
    if criterion == "density":
        return np.array([f["density"] for f in features], dtype=float)
    if criterion == "entropy":
        return np.array([f["entropy"] for f in features], dtype=float)
    if criterion == "boundary_compact":
        return np.array(
            [f["boundary_score"] * f["compactness"] for f in features], dtype=float
        )
    if criterion == "random":
        rng = np.random.RandomState(rng_seed)
        return rng.random(n)
    if criterion == "lobo_oracle":
        if lobo_scores is None:
            return np.full(n, float("-inf"))
        return np.asarray(lobo_scores, dtype=float)
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


def _metrics_row(
    *,
    split,
    seed: int,
    representation: str,
    criterion: str,
    selection_ratio: float,
    n_balls_total: int,
    n_balls_selected: int,
    baseline_macro: float,
    perf: dict[str, Any],
) -> dict[str, Any]:
    macro = perf.get("macro_f1", float("nan"))
    return {
        "dataset": split.name,
        "seed": seed,
        "representation": representation,
        "ranking_criterion": criterion,
        "selection_ratio": selection_ratio,
        "n_balls_total": n_balls_total,
        "n_balls_selected": n_balls_selected,
        "baseline_macro_f1": baseline_macro,
        "accuracy": perf.get("accuracy", float("nan")),
        "macro_f1": macro,
        "minority_f1": perf.get("minority_f1", float("nan")),
        "gmean": perf.get("gmean", float("nan")),
        "delta_macro_f1_vs_all_balls": (
            float(macro - baseline_macro) if np.isfinite(macro) and np.isfinite(baseline_macro) else float("nan")
        ),
        "classifier_ok": bool(perf.get("classifier_ok", False)),
        "classifier_error": perf.get("classifier_error"),
    }


def run_configuration(
    split,
    seed: int,
    representation: str,
    lobo_cache: pd.DataFrame | None,
    *,
    compute_lobo: bool,
) -> list[dict[str, Any]]:
    balls, kept, _ = build_representation(split, seed, representation)
    features = compute_ball_features(kept, split.minority_label)
    n_balls = len(balls)

    baseline = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )
    baseline_macro = float(baseline.get("macro_f1", float("nan")))

    lobo_scores: list[float] | None = None
    if lobo_cache is not None:
        lobo_scores = _lobo_usefulness_from_cache(
            lobo_cache, split.name, seed, representation, n_balls
        )
    if lobo_scores is None and compute_lobo:
        lobo_scores = _compute_lobo_usefulness(balls, split, seed)

    rows: list[dict[str, Any]] = []
    for criterion in RANKING_CRITERIA:
        if criterion == "lobo_oracle" and lobo_scores is None:
            continue
        scores = _ranking_scores(
            features,
            criterion,
            lobo_scores=lobo_scores,
            rng_seed=seed * 1000 + hash(representation) % 1000,
        )
        for ratio in SELECTION_RATIOS:
            indices = _select_ball_indices(
                balls,
                scores,
                ratio,
                criterion=criterion,
                rng_seed=seed * 1000 + int(ratio * 100) + hash(criterion) % 100,
            )
            subset = [balls[i] for i in indices]
            perf = fit_gaussian_gblstsvm(
                subset,
                y_train=split.y_train,
                X_test=split.X_test,
                y_test=split.y_test,
                minority_label=split.minority_label,
                seed=seed,
            )
            rows.append(
                _metrics_row(
                    split=split,
                    seed=seed,
                    representation=representation,
                    criterion=criterion,
                    selection_ratio=ratio,
                    n_balls_total=n_balls,
                    n_balls_selected=len(indices),
                    baseline_macro=baseline_macro,
                    perf=perf,
                )
            )
    return rows


def _mean_macro(
    df: pd.DataFrame,
    representation: str,
    criterion: str,
    ratio: float,
) -> float:
    sub = df[
        (df["representation"] == representation)
        & (df["ranking_criterion"] == criterion)
        & (df["selection_ratio"] == ratio)
        & df["classifier_ok"].astype(bool)
    ]
    return _nanmean(sub["macro_f1"].tolist())


def analyze_oracle_recovery(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rep in REPRESENTATIONS:
        base = _mean_macro(df, rep, "purity", 1.00)
        oracle_50 = _mean_macro(df, rep, "lobo_oracle", 0.50)
        oracle_75 = _mean_macro(df, rep, "lobo_oracle", 0.75)
        out[rep] = {
            "all_balls_macro_f1": base,
            "lobo_oracle_50pct_macro_f1": oracle_50,
            "lobo_oracle_75pct_macro_f1": oracle_75,
            "recoverable_gain_50pct": oracle_50 - base if np.isfinite(oracle_50) and np.isfinite(base) else float("nan"),
            "recoverable_gain_75pct": oracle_75 - base if np.isfinite(oracle_75) and np.isfinite(base) else float("nan"),
        }
    return out


def analyze_criterion_vs_oracle(df: pd.DataFrame, ratio: float = 0.50) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ok = df[df["classifier_ok"].astype(bool)]
    for crit in RANKING_CRITERIA:
        if crit == "lobo_oracle":
            continue
        sub = ok[(ok["ranking_criterion"] == crit) & (ok["selection_ratio"] == ratio)]
        oracle = ok[(ok["ranking_criterion"] == "lobo_oracle") & (ok["selection_ratio"] == ratio)]
        if sub.empty or oracle.empty:
            continue
        merged = sub.merge(
            oracle[["dataset", "seed", "representation", "macro_f1"]],
            on=["dataset", "seed", "representation"],
            suffixes=("", "_oracle"),
        )
        gaps = merged["macro_f1_oracle"] - merged["macro_f1"]
        rows.append(
            {
                "ranking_criterion": crit,
                "selection_ratio": ratio,
                "mean_macro_f1": _nanmean(sub["macro_f1"].tolist()),
                "mean_oracle_macro_f1": _nanmean(oracle["macro_f1"].tolist()),
                "mean_gap_to_oracle": _nanmean(gaps.tolist()),
                "n_pairs": int(len(merged)),
            }
        )
    rows.sort(key=lambda r: r["mean_gap_to_oracle"])
    return rows


def analyze_boundary_vs_all(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["classifier_ok"].astype(bool)]
    all_balls = ok[(ok["ranking_criterion"] == "purity") & (ok["selection_ratio"] == 1.00)]
    boundary_50 = ok[(ok["ranking_criterion"] == "boundary_score") & (ok["selection_ratio"] == 0.50)]
    boundary_75 = ok[(ok["ranking_criterion"] == "boundary_score") & (ok["selection_ratio"] == 0.75)]
    merged_50 = all_balls.merge(
        boundary_50[["dataset", "seed", "representation", "macro_f1"]],
        on=["dataset", "seed", "representation"],
        suffixes=("_all", "_boundary50"),
    )
    merged_75 = all_balls.merge(
        boundary_75[["dataset", "seed", "representation", "macro_f1"]],
        on=["dataset", "seed", "representation"],
        suffixes=("_all", "_boundary75"),
    )
    d50 = merged_50["macro_f1_boundary50"] - merged_50["macro_f1_all"]
    d75 = merged_75["macro_f1_boundary75"] - merged_75["macro_f1_all"]
    return {
        "mean_delta_boundary_50pct_vs_all": _nanmean(d50.tolist()),
        "mean_delta_boundary_75pct_vs_all": _nanmean(d75.tolist()),
        "fraction_improved_at_50pct": float(np.mean(d50 > 1e-6)) if len(d50) else float("nan"),
        "fraction_improved_at_75pct": float(np.mean(d75 > 1e-6)) if len(d75) else float("nan"),
    }


def diagnose_bottleneck(df: pd.DataFrame, recovery: dict[str, Any]) -> dict[str, Any]:
    ok = df[df["classifier_ok"].astype(bool)]
    all_rep_means = {
        rep: _mean_macro(ok, rep, "purity", 1.00) for rep in REPRESENTATIONS
    }
    best_gen = max(all_rep_means, key=lambda r: all_rep_means[r] if np.isfinite(all_rep_means[r]) else -1)
    best_gen_f1 = all_rep_means[best_gen]

    mean_oracle_gain_50 = _nanmean(
        [recovery[r]["recoverable_gain_50pct"] for r in REPRESENTATIONS if r in recovery]
    )
    mean_boundary_gain_50 = analyze_boundary_vs_all(df)["mean_delta_boundary_50pct_vs_all"]

    selection_matters = np.isfinite(mean_oracle_gain_50) and mean_oracle_gain_50 > 0.01
    generation_matters = best_gen != "SU3" or (
        np.isfinite(all_rep_means.get("SU3", float("nan")))
        and np.isfinite(best_gen_f1)
        and best_gen_f1 - all_rep_means["SU3"] > 0.02
    )

    if selection_matters and generation_matters:
        verdict = "both"
        detail = (
            "Oracle subsampling recovers macro F1 over all-ball training, and best-generation "
            f"representation ({best_gen}) still outperforms SU3 at 100% — both selection and "
            "generation limit performance."
        )
    elif selection_matters:
        verdict = "ball_selection"
        detail = (
            "LOBO-oracle subsampling improves macro F1 while generation choice (rho) has smaller "
            "spread at 100% balls — the primary bottleneck is which balls are kept, not only how "
            "many are generated."
        )
    elif generation_matters:
        verdict = "ball_generation"
        detail = (
            "Oracle selection yields limited recovery; representation choice dominates at 100% "
            "balls — the primary bottleneck is ball generation quality."
        )
    else:
        verdict = "unclear"
        detail = "Effects are small or dataset-specific; inspect per-representation recovery table."

    return {
        "verdict": verdict,
        "detail": detail,
        "best_generation_representation": best_gen,
        "best_generation_macro_f1": best_gen_f1,
        "mean_oracle_recoverable_gain_50pct": mean_oracle_gain_50,
        "mean_boundary_selection_gain_50pct": mean_boundary_gain_50,
        "all_representation_all_ball_macro_f1": all_rep_means,
    }


def write_report(path: Path, *, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    recovery = summary["oracle_recovery"]
    crit50 = summary["criterion_vs_oracle_50pct"]
    boundary = summary["boundary_vs_all"]
    bottleneck = summary["bottleneck"]

    lines = [
        "# Oracle ball selection diagnostic",
        "",
        "Read-only: can subsampling balls by ranking criteria recover performance?",
        "",
        f"Rows: **{summary['n_rows']}** | Weighting: **Gaussian {PRIMARY_GAUSSIAN}**",
        "",
        "## 1. How much performance is recoverable with oracle selection?",
        "",
        "| representation | all balls | LOBO oracle 50% | LOBO oracle 75% | gain @50% | gain @75% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rep in REPRESENTATIONS:
        r = recovery.get(rep, {})
        lines.append(
            f"| {rep} | {r.get('all_balls_macro_f1', float('nan')):.3f} | "
            f"{r.get('lobo_oracle_50pct_macro_f1', float('nan')):.3f} | "
            f"{r.get('lobo_oracle_75pct_macro_f1', float('nan')):.3f} | "
            f"{r.get('recoverable_gain_50pct', float('nan')):+.4f} | "
            f"{r.get('recoverable_gain_75pct', float('nan')):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Which ranking criterion approaches oracle? (50% selection)",
            "",
            "| criterion | mean macro F1 | gap to oracle |",
            "|---|---:|---:|",
        ]
    )
    for row in crit50:
        lines.append(
            f"| {row['ranking_criterion']} | {row['mean_macro_f1']:.3f} | "
            f"{row['mean_gap_to_oracle']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Does selecting boundary balls improve over all-ball training?",
            "",
            f"- Mean Δ macro F1 (boundary 50% − all balls): **{boundary['mean_delta_boundary_50pct_vs_all']:+.4f}**",
            f"- Mean Δ macro F1 (boundary 75% − all balls): **{boundary['mean_delta_boundary_75pct_vs_all']:+.4f}**",
            f"- Fraction improved @50%: {boundary['fraction_improved_at_50pct']:.1%}",
            f"- Fraction improved @75%: {boundary['fraction_improved_at_75pct']:.1%}",
            "",
            "## 4. Bottleneck diagnosis",
            "",
            f"**Verdict: {bottleneck['verdict']}**",
            "",
            bottleneck["detail"],
            "",
            f"- Best generation @100%: **{bottleneck['best_generation_representation']}** "
            f"(macro F1 {bottleneck['best_generation_macro_f1']:.3f})",
            f"- Mean oracle recoverable gain @50%: {bottleneck['mean_oracle_recoverable_gain_50pct']:+.4f}",
            f"- Mean boundary selection gain @50%: {bottleneck['mean_boundary_selection_gain_50pct']:+.4f}",
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
    recovery = analyze_oracle_recovery(df)
    crit50 = analyze_criterion_vs_oracle(df, 0.50)
    crit75 = analyze_criterion_vs_oracle(df, 0.75)
    boundary = analyze_boundary_vs_all(df)
    bottleneck = diagnose_bottleneck(df, recovery)
    return {
        "protocol": {
            "read_only": True,
            "diagnostic_only": True,
            "smoke_mode": smoke,
            "seeds": SEEDS,
            "representations": list(REPRESENTATIONS),
            "selection_ratios": list(SELECTION_RATIOS),
            "ranking_criteria": list(RANKING_CRITERIA),
            "weighting": f"Gaussian {PRIMARY_GAUSSIAN}",
            "lobo_cache": str(LOBO_CACHE_PATH),
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "failed": failed,
            "no_new_algorithms": True,
        },
        "n_rows": int(len(df)),
        "oracle_recovery": recovery,
        "criterion_vs_oracle_50pct": crit50,
        "criterion_vs_oracle_75pct": crit75,
        "boundary_vs_all": boundary,
        "bottleneck": bottleneck,
    }


def _append_checkpoint(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        combined = pd.concat([pd.read_csv(OUT_CSV), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(OUT_CSV, index=False)
    rows.clear()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv
    compute_lobo = "--compute-lobo" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    lobo_cache = _load_lobo_cache()
    if lobo_cache is not None:
        print(f"LOBO cache: {LOBO_CACHE_PATH} ({len(lobo_cache)} rows)")
    elif compute_lobo:
        print("No LOBO cache — will compute LOBO per configuration (--compute-lobo)")
    else:
        print("No LOBO cache — lobo_oracle criterion will be skipped (use --compute-lobo)")

    descriptors = discover_available_datasets()
    if smoke:
        descriptors = descriptors[:1]
        seeds = [42]
        print(f"SMOKE: 1 dataset × seed 42 × {len(REPRESENTATIONS)} representations")
    else:
        seeds = SEEDS
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds × {len(REPRESENTATIONS)} representations")

    if OUT_CSV.exists() and "--resume" not in sys.argv and not smoke:
        OUT_CSV.unlink()

    done: set[tuple[str, int, str]] = set()
    if OUT_CSV.exists() and "--resume" in sys.argv:
        prev = pd.read_csv(OUT_CSV)
        for _, r in prev.iterrows():
            done.add((str(r["dataset"]), int(r["seed"]), str(r["representation"])))
        print(f"RESUME: skipping {len(done)} configurations")

    pending: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    completed = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append({"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"})
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue

            for representation in REPRESENTATIONS:
                key = (split.name, seed, representation)
                if key in done:
                    continue

                rows = run_configuration(
                    split,
                    seed,
                    representation,
                    lobo_cache,
                    compute_lobo=compute_lobo or smoke,
                )
                pending.extend(rows)
                completed += 1

                base = next(r for r in rows if r["selection_ratio"] == 1.0 and r["ranking_criterion"] == "purity")
                b50 = next(
                    (r for r in rows if r["ranking_criterion"] == "boundary_score" and r["selection_ratio"] == 0.50),
                    None,
                )
                o50 = next(
                    (r for r in rows if r["ranking_criterion"] == "lobo_oracle" and r["selection_ratio"] == 0.50),
                    None,
                )
                msg = (
                    f"{split.name} s{seed} {representation}: "
                    f"balls={base['n_balls_total']} all={base['macro_f1']:.3f}"
                )
                if b50:
                    msg += f" boundary50={b50['macro_f1']:.3f}"
                if o50:
                    msg += f" oracle50={o50['macro_f1']:.3f}"
                print(msg)

                if completed % SAVE_EVERY == 0:
                    _append_checkpoint(pending)

    _append_checkpoint(pending)

    df = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame(pending)
    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Bottleneck: {summary['bottleneck']['verdict']} — {summary['bottleneck']['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
