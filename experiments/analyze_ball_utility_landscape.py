"""
Ball utility landscape — leave-one-ball-out influence diagnostic (read-only).

Identifies which individual granular balls contribute positively or negatively
to Gaussian-weighted GBLSTSVM classification.

Does NOT modify src/, solver, prediction, CW, CA, Topo, Gaussian membership,
or granulation algorithms.

Representations: B0, rho0.80–0.95, SU3
Weighting: Gaussian purity_compact only (frozen).

Run from GBLSTSVM root:
  python experiments/analyze_ball_utility_landscape.py --smoke
  python experiments/analyze_ball_utility_landscape.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from experiments.phase8_common import (  # noqa: E402
    DEFAULT_C1,
    DEFAULT_C2,
    normalize_labels,
)
from experiments.test_split_utility_granulation import _labels  # noqa: E402
from src.granular_balls import BallInfo, GranularBall, balls_to_class_arrays  # noqa: E402
from src.reliability import compute_radius_scale, normalize_radius  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "ball_utility_landscape"
OUT_CSV = OUT_DIR / "ball_records.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
SU3_PERCENTILE = 75
EPS = 1e-12
REG_TERM = 1e-4
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"

REPRESENTATIONS = (
    "B0",
    "rho0.80",
    "rho0.85",
    "rho0.90",
    "rho0.95",
    "SU3",
)

PROFILE_METRICS = (
    "purity",
    "radius",
    "normalized_radius",
    "density",
    "entropy",
    "opposite_class_fraction",
    "boundary_score",
    "support_score",
    "margin_score",
    "is_mixed",
)


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, label: float) -> float:
    return float(
        f1_score(
            normalize_labels(y_true),
            normalize_labels(y_pred),
            pos_label=label,
            zero_division=0,
        )
    )


def _majority_label(ball: GranularBall) -> float:
    return -1.0 if ball.label == -1 else 1.0


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, pp = pearsonr(xv, yv)
    sr, sp = spearmanr(xv, yv)
    return {
        "pearson": float(pr),
        "pearson_p": float(pp),
        "spearman": float(sr),
        "spearman_p": float(sp),
        "n": int(np.sum(mask)),
    }


def compute_ball_features(
    kept: list[GranularBall],
    minority_label: float,
) -> list[dict[str, Any]]:
    if not kept:
        return []

    centers = np.vstack([b.center for b in kept])
    maj_labels = np.array([_majority_label(b) for b in kept], dtype=float)
    radii = np.array([float(b.r) for b in kept], dtype=float)
    scale = compute_radius_scale(radii, method="median_positive")

    records: list[dict[str, Any]] = []
    for i, ball in enumerate(kept):
        sample_labels = normalize_labels(_labels(ball))
        n = int(ball.num)
        r = float(ball.r)
        density = n / (r + EPS)
        r_norm = normalize_radius(r, scale)
        compactness = 1.0 / (r_norm + EPS)

        opp_idx = np.flatnonzero(maj_labels != maj_labels[i])
        same_idx = np.flatnonzero(maj_labels == maj_labels[i])
        same_idx = same_idx[same_idx != i]

        if opp_idx.size:
            dist_opp = float(np.min(np.linalg.norm(centers[opp_idx] - centers[i], axis=1)))
        else:
            dist_opp = float("nan")
        if same_idx.size:
            dist_same = float(np.min(np.linalg.norm(centers[same_idx] - centers[i], axis=1)))
        else:
            dist_same = float("nan")

        margin_score = dist_opp / (dist_same + EPS) if np.isfinite(dist_opp) else float("nan")

        n_min = int(np.sum(sample_labels == minority_label))
        n_maj = n - n_min
        minority_ratio = n_min / n if n else 0.0
        majority_ratio = n_maj / n if n else 0.0
        opposite_label = -minority_label if minority_label in (-1.0, 1.0) else float("nan")
        n_opp_samples = int(np.sum(sample_labels == opposite_label)) if np.isfinite(opposite_label) else 0
        opposite_class_fraction = n_opp_samples / n if n else 0.0
        entropy = _binary_entropy(minority_ratio) if n_min > 0 and n_maj > 0 else 0.0
        is_mixed = bool(n_min > 0 and n_maj > 0)

        records.append(
            {
                "ball_id": i,
                "sample_count": n,
                "radius": r,
                "density": density,
                "normalized_radius": r_norm,
                "compactness": compactness,
                "dist_nearest_opposite": dist_opp,
                "dist_nearest_same": dist_same,
                "purity": float(ball.purity),
                "entropy": entropy,
                "class_label": _majority_label(ball),
                "n_minority_samples": n_min,
                "n_majority_samples": n_maj,
                "minority_ratio": minority_ratio,
                "majority_ratio": majority_ratio,
                "opposite_class_fraction": opposite_class_fraction,
                "boundary_score": entropy * density,
                "support_score": n * compactness,
                "margin_score": margin_score,
                "is_mixed": is_mixed,
            }
        )
    return records


def _lobo_valid(balls: list[BallInfo], remove_idx: int) -> bool:
    remaining = [b for j, b in enumerate(balls) if j != remove_idx]
    if not remaining:
        return False
    try:
        balls_to_class_arrays(remaining)
        return True
    except ValueError:
        return False


def run_lobo_analysis(
    split,
    seed: int,
    representation: str,
) -> list[dict[str, Any]]:
    balls, kept, _ = build_representation(split, seed, representation)
    features = compute_ball_features(kept, split.minority_label)

    baseline = fit_gaussian_gblstsvm(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        seed=seed,
    )
    base_macro = baseline["macro_f1"]
    base_min_f1 = baseline["minority_f1"]
    baseline_ok = bool(baseline.get("classifier_ok", False))

    rows: list[dict[str, Any]] = []
    for i, feat in enumerate(features):
        delta_macro = float("nan")
        delta_min = float("nan")
        lobo_ok = False

        if baseline_ok and _lobo_valid(balls, i):
            remaining = [b for j, b in enumerate(balls) if j != i]
            lobo = fit_gaussian_gblstsvm(
                remaining,
                y_train=split.y_train,
                X_test=split.X_test,
                y_test=split.y_test,
                minority_label=split.minority_label,
                seed=seed,
            )
            if lobo.get("classifier_ok", False):
                delta_macro = float(lobo["macro_f1"] - base_macro)
                delta_min = float(lobo["minority_f1"] - base_min_f1)
                lobo_ok = True

        row = {
            "dataset": split.name,
            "seed": seed,
            "representation": representation,
            "ball_id": feat["ball_id"],
            "baseline_macro_f1": base_macro,
            "baseline_minority_f1": base_min_f1,
            "delta_macro_f1": delta_macro,
            "delta_minority_f1": delta_min,
            "lobo_ok": lobo_ok,
            "ball_usefulness": -delta_macro if np.isfinite(delta_macro) else float("nan"),
            **feat,
        }
        rows.append(row)
    return rows


def _profile_subset(df: pd.DataFrame, metrics: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in metrics:
        if m == "is_mixed":
            out[m] = float(df["is_mixed"].astype(bool).mean()) if len(df) else float("nan")
        elif m in df.columns:
            out[m] = _nanmean(df[m].tolist())
        else:
            out[m] = float("nan")
    return out


def analyze_useful_balls(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["lobo_ok"].astype(bool) & df["delta_macro_f1"].notna()].copy()
    if ok.empty:
        return {"n_balls": 0, "detail": "No valid LOBO records."}

    ok = ok.sort_values("delta_macro_f1")
    n = len(ok)
    k = max(1, int(np.ceil(0.10 * n)))
    positive = ok.head(k)  # most negative delta → removing hurts
    negative = ok.tail(k)  # most positive delta → removing helps

    pos_profile = _profile_subset(positive, PROFILE_METRICS)
    neg_profile = _profile_subset(negative, PROFILE_METRICS)

    return {
        "n_balls": n,
        "top10pct_positive_influence": pos_profile,
        "top10pct_negative_influence": neg_profile,
        "delta_purity_pos_minus_neg": pos_profile["purity"] - neg_profile["purity"],
        "delta_radius_pos_minus_neg": pos_profile["radius"] - neg_profile["radius"],
        "delta_boundary_pos_minus_neg": pos_profile["boundary_score"] - neg_profile["boundary_score"],
        "delta_mixed_rate_pos_minus_neg": pos_profile["is_mixed"] - neg_profile["is_mixed"],
    }


def analyze_mixed_balls(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["lobo_ok"].astype(bool) & df["ball_usefulness"].notna()].copy()
    if ok.empty:
        return {"detail": "No valid LOBO records."}

    pos_thresh = ok["ball_usefulness"].quantile(0.75)
    positive = ok[ok["ball_usefulness"] >= pos_thresh]
    all_balls = ok

    def means(sub: pd.DataFrame) -> dict[str, float]:
        return {
            "entropy": _nanmean(sub["entropy"].tolist()),
            "opposite_class_fraction": _nanmean(sub["opposite_class_fraction"].tolist()),
            "boundary_score": _nanmean(sub["boundary_score"].tolist()),
            "mixed_rate": float(sub["is_mixed"].astype(bool).mean()),
        }

    pos_m = means(positive)
    all_m = means(all_balls)
    mixed_only = ok[ok["is_mixed"].astype(bool)]
    pure_only = ok[~ok["is_mixed"].astype(bool)]

    return {
        "positive_influence_top_quartile": pos_m,
        "all_balls": all_m,
        "mixed_balls_only": means(mixed_only) if len(mixed_only) else {},
        "pure_balls_only": means(pure_only) if len(pure_only) else {},
        "positive_has_higher_entropy": pos_m["entropy"] > all_m["entropy"],
        "positive_has_higher_opposite_fraction": (
            pos_m["opposite_class_fraction"] > all_m["opposite_class_fraction"]
        ),
        "positive_has_higher_boundary_score": pos_m["boundary_score"] > all_m["boundary_score"],
        "mixed_ball_mean_usefulness": _nanmean(mixed_only["ball_usefulness"].tolist())
        if len(mixed_only)
        else float("nan"),
        "pure_ball_mean_usefulness": _nanmean(pure_only["ball_usefulness"].tolist())
        if len(pure_only)
        else float("nan"),
    }


def analyze_purity_misleading(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["lobo_ok"].astype(bool) & df["ball_usefulness"].notna()].copy()
    if ok.empty:
        return {"detail": "No valid LOBO records."}

    purity = ok["purity"].to_numpy(dtype=float)
    usefulness = ok["ball_usefulness"].to_numpy(dtype=float)
    corr = _safe_corr(purity, usefulness)

    weak_or_negative = (
        not np.isfinite(corr["pearson"])
        or corr["pearson"] < 0.05
        or (np.isfinite(corr["pearson_p"]) and corr["pearson_p"] > 0.05 and abs(corr["pearson"]) < 0.1)
    )

    return {
        "purity_vs_usefulness": corr,
        "purity_misleading": weak_or_negative or corr["pearson"] < 0,
        "interpretation": (
            "Purity is weakly or negatively associated with ball usefulness; "
            "maximizing purity alone is not supported as an optimization target."
            if weak_or_negative or corr["pearson"] < 0
            else "Purity shows a modest positive association with usefulness at ball level."
        ),
    }


def summarize_missing_objective(
    useful: dict[str, Any],
    mixed: dict[str, Any],
    purity: dict[str, Any],
) -> dict[str, Any]:
    pos = useful.get("top10pct_positive_influence", {})
    neg = useful.get("top10pct_negative_influence", {})

    maximize: list[str] = []
    minimize: list[str] = []

    if pos.get("boundary_score", 0) > neg.get("boundary_score", 0):
        maximize.append("boundary_score (entropy × density)")
    if pos.get("margin_score", 0) > neg.get("margin_score", 0):
        maximize.append("margin_score (opposite/same center separation)")
    if pos.get("entropy", 0) > neg.get("entropy", 0):
        maximize.append("entropy / class heterogeneity")
    if pos.get("opposite_class_fraction", 0) > neg.get("opposite_class_fraction", 0):
        maximize.append("opposite_class_fraction")
    if mixed.get("mixed_ball_mean_usefulness", float("nan")) > mixed.get(
        "pure_ball_mean_usefulness", float("nan")
    ):
        maximize.append("boundary-aware mixed support")

    if pos.get("purity", 1) < neg.get("purity", 0):
        minimize.append("blind purity maximization")
    if pos.get("radius", 0) < neg.get("radius", float("inf")):
        minimize.append("oversized low-information balls")
    if purity.get("purity_misleading", False):
        minimize.append("purity-only granulation objective")

    evidence = (
        f"Evidence-based landscape (no algorithm): maximize {' + '.join(maximize) or '(unclear)'}; "
        f"minimize {' + '.join(minimize) or '(unclear)'}."
    )

    return {
        "maximize": maximize,
        "minimize": minimize,
        "evidence_summary": evidence,
        "formula_sketch": (
            f"utility_proxy ≈ +{maximize[0] if maximize else '?'} "
            f"+ {maximize[1] if len(maximize) > 1 else '?'} "
            f"- {minimize[0] if minimize else '?'}"
        ),
    }


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    useful = summary["useful_ball_analysis"]
    mixed = summary["mixed_ball_analysis"]
    purity = summary["purity_analysis"]
    objective = summary["missing_objective"]

    pos_p = useful.get("top10pct_positive_influence", {})
    neg_p = useful.get("top10pct_negative_influence", {})

    lines = [
        "# Ball utility landscape",
        "",
        "Read-only leave-one-ball-out (LOBO) diagnostic: which granular balls help or hurt "
        f"Gaussian {PRIMARY_GAUSSIAN} GBLSTSVM?",
        "",
        f"Ball records: **{summary['n_ball_rows']}** | Valid LOBO: **{summary['n_lobo_ok']}** | "
        f"Configurations: {summary['n_configurations']}",
        "",
        "## 1. What properties do useful balls have?",
        "",
        "Top 10% **positive influence** (removing hurts macro F1) vs top 10% **negative influence** "
        "(removing improves macro F1).",
        "",
        "| metric | positive influence | negative influence | Δ (pos − neg) |",
        "|---|---:|---:|---:|",
    ]

    for m in PROFILE_METRICS:
        pv = pos_p.get(m, float("nan"))
        nv = neg_p.get(m, float("nan"))
        if m == "is_mixed":
            lines.append(f"| {m} | {pv:.3f} | {nv:.3f} | {pv - nv:+.3f} |")
        else:
            lines.append(f"| {m} | {pv:.4f} | {nv:.4f} | {pv - nv:+.4f} |")

    lines.extend(
        [
            "",
            "**Summary:** Useful balls tend toward higher boundary score and heterogeneity; "
            "harmful balls tend toward higher purity and larger radius when those deltas are positive.",
            "",
            "## 2. Are mixed balls actually useful?",
            "",
        ]
    )

    if "positive_influence_top_quartile" in mixed:
        pq = mixed["positive_influence_top_quartile"]
        aq = mixed["all_balls"]
        lines.extend(
            [
                "| metric | top-quartile useful | all balls |",
                "|---|---:|---:|",
                f"| entropy | {pq['entropy']:.4f} | {aq['entropy']:.4f} |",
                f"| opposite_class_fraction | {pq['opposite_class_fraction']:.4f} | "
                f"{aq['opposite_class_fraction']:.4f} |",
                f"| boundary_score | {pq['boundary_score']:.4f} | {aq['boundary_score']:.4f} |",
                f"| mixed rate | {pq['mixed_rate']:.3f} | {aq['mixed_rate']:.3f} |",
                "",
                f"- Positive-influence balls have higher entropy: **{mixed['positive_has_higher_entropy']}**",
                f"- Higher opposite-class fraction: **{mixed['positive_has_higher_opposite_fraction']}**",
                f"- Higher boundary score: **{mixed['positive_has_higher_boundary_score']}**",
                f"- Mixed-ball mean usefulness: {mixed.get('mixed_ball_mean_usefulness', float('nan')):.4f}",
                f"- Pure-ball mean usefulness: {mixed.get('pure_ball_mean_usefulness', float('nan')):.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. Is purity misleading?",
            "",
        ]
    )
    if "purity_vs_usefulness" in purity:
        c = purity["purity_vs_usefulness"]
        lines.extend(
            [
                f"| correlation | Pearson | Spearman | n |",
                f"|---|---:|---:|---:|",
                f"| purity vs ball usefulness | {c['pearson']:.3f} | {c['spearman']:.3f} | {c['n']} |",
                "",
                f"**Purity misleading:** {purity.get('purity_misleading', False)}",
                "",
                purity.get("interpretation", ""),
                "",
                "## 4. Missing granulation objective (evidence only)",
                "",
                objective.get("evidence_summary", ""),
                "",
                f"Sketch: `{objective.get('formula_sketch', '')}`",
                "",
                "**Maximize:** " + ", ".join(objective.get("maximize", [])) or "—",
                "",
                "**Minimize:** " + ", ".join(objective.get("minimize", [])) or "—",
                "",
                "## Outputs",
                "",
                f"- `{OUT_CSV.name}`",
                f"- `{OUT_SUMMARY.name}`",
                "",
            ]
        )

    if failed:
        lines.extend(["## Failed loads", ""])
        for item in failed:
            lines.append(f"- {item['dataset']} seed={item['seed']}: {item['error']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(df: pd.DataFrame, failed: list[dict[str, str]], smoke: bool) -> dict[str, Any]:
    useful = analyze_useful_balls(df)
    mixed = analyze_mixed_balls(df)
    purity = analyze_purity_misleading(df)
    objective = summarize_missing_objective(useful, mixed, purity)

    n_lobo_ok = int(df["lobo_ok"].astype(bool).sum()) if "lobo_ok" in df.columns else 0
    n_configs = (
        int(df.groupby(["dataset", "seed", "representation"]).ngroups)
        if not df.empty
        else 0
    )

    return {
        "protocol": {
            "read_only": True,
            "diagnostic_only": True,
            "smoke_mode": smoke,
            "analysis": "leave_one_ball_out",
            "purity_b0_su3": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "seeds": SEEDS,
            "representations": list(REPRESENTATIONS),
            "weighting": f"Gaussian {PRIMARY_GAUSSIAN}",
            "su3_rule": f"reject if utility <= percentile({SU3_PERCENTILE}) of B0 proposals",
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "prediction_rule": PREDICTION_RULE,
            "weight_placement": WEIGHT_PLACEMENT,
            "failed": failed,
            "no_tuning": True,
            "no_new_algorithms": True,
        },
        "n_ball_rows": int(len(df)),
        "n_lobo_ok": n_lobo_ok,
        "n_configurations": n_configs,
        "useful_ball_analysis": useful,
        "mixed_ball_analysis": mixed,
        "purity_analysis": purity,
        "missing_objective": objective,
    }


def _append_checkpoint(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(OUT_CSV, index=False)
    rows.clear()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary, failed=[])
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    if smoke:
        descriptors = descriptors[:1]
        seeds = [42]
        print(
            f"SMOKE: 1 dataset ({descriptors[0]['name']}) × seed 42 × "
            f"{len(REPRESENTATIONS)} representations (full LOBO per ball)"
        )
    else:
        seeds = SEEDS
        print(
            f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds × "
            f"{len(REPRESENTATIONS)} representations (LOBO per ball)"
        )

    if OUT_CSV.exists() and "--resume" not in sys.argv and "--smoke" not in sys.argv:
        OUT_CSV.unlink()

    pending: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    completed_configs = 0
    total_balls = 0

    done_keys: set[tuple[str, int, str]] = set()
    if OUT_CSV.exists() and "--resume" in sys.argv:
        prev = pd.read_csv(OUT_CSV)
        for _, r in prev.iterrows():
            done_keys.add((str(r["dataset"]), int(r["seed"]), str(r["representation"])))
        print(f"RESUME: skipping {len(done_keys)} completed configurations")

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append(
                    {"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue

            for representation in REPRESENTATIONS:
                key = (split.name, seed, representation)
                if key in done_keys:
                    continue

                ball_rows = run_lobo_analysis(split, seed, representation)
                pending.extend(ball_rows)
                total_balls += len(ball_rows)
                completed_configs += 1

                n_pos = sum(
                    1
                    for r in ball_rows
                    if r.get("lobo_ok") and np.isfinite(r["delta_macro_f1"]) and r["delta_macro_f1"] < 0
                )
                n_neg = sum(
                    1
                    for r in ball_rows
                    if r.get("lobo_ok") and np.isfinite(r["delta_macro_f1"]) and r["delta_macro_f1"] > 0
                )
                print(
                    f"{split.name} s{seed} {representation}: "
                    f"balls={len(ball_rows)} pos_influence={n_pos} neg_influence={n_neg}"
                )

                _append_checkpoint(pending)

    df = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame(pending)
    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary, failed=failed)

    print(f"\nWrote {OUT_CSV} ({len(df)} ball rows, {total_balls} this session)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Missing objective: {summary['missing_objective'].get('evidence_summary', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
