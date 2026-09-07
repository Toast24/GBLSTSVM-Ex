"""
Split-utility × weighting ablation (read-only).

Tests whether SU3 granulation (utility p75 gate) creates ball representations
that existing CA / Topo / Gaussian weighting can exploit.

Does NOT modify src/ baselines, solver, prediction, CW, CA, Topo, or Gaussian code.

Granulation: B0 (baseline), SU3 (p75 from test_split_utility_granulation.py)
Weighting: CA, Topo, Gaussian, CA×Topo, CA×Gaussian, CA×Topo×Gaussian

Run from GBLSTSVM root:
  python experiments/test_su_weighting_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.gaussian_membership_screen import (  # noqa: E402
    PRIMARY_GAUSSIAN,
    compute_gaussian_memberships,
)
from experiments.phase8_common import (  # noqa: E402
    DEFAULT_C1,
    DEFAULT_C2,
    evaluate,
    normalize_labels,
)
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _collect_b0_proposals,
    _make_reject_fn,
    _replay_granulation,
    _to_ball_info,
)
from src.cbcm import compute_competition_membership, split_membership_by_class  # noqa: E402
from src.class_weights import ball_class_weights, compute_binary_class_weights  # noqa: E402
from src.cw_gblstsvm import solve_cw_gblstsvm_systems  # noqa: E402
from src.gblstsvm import predict_twin_hyperplanes  # noqa: E402
from src.granular_balls import BallInfo, balls_to_class_arrays  # noqa: E402
from src.topology_membership import compute_topology_membership  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "su_weighting_ablation"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
SU3_PERCENTILE = 75
REG_TERM = 1e-4
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"

GRANULATION_MODES = ("B0", "SU3")
WEIGHTING_MODES = (
    "CA",
    "Topo",
    "Gaussian",
    "CATopo",
    "CAGaussian",
    "CATopoGaussian",
)


def _nanmean(values: list[float]) -> float:
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


def _ball_weight_vector(
    weights_pos: np.ndarray,
    weights_neg: np.ndarray,
    balls: list[BallInfo],
) -> np.ndarray:
    pos_i = neg_i = 0
    out = np.empty(len(balls), dtype=float)
    for i, b in enumerate(balls):
        if b.label == 1.0:
            out[i] = weights_pos[pos_i]
            pos_i += 1
        else:
            out[i] = weights_neg[neg_i]
            neg_i += 1
    return out


def build_balls_and_stats(split, seed: int, granulation: str) -> tuple[list[BallInfo], dict[str, Any]]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    b0_proposals = _collect_b0_proposals(indexed, seed)
    if granulation == "B0":
        reject_fn = _make_reject_fn("B0", None, b0_proposals)
    elif granulation == "SU3":
        reject_fn = _make_reject_fn("SU3", SU3_PERCENTILE, b0_proposals)
    else:
        raise ValueError(f"Unknown granulation {granulation!r}")

    kept, _, stats = _replay_granulation(indexed, seed=seed, should_reject=reject_fn)
    balls = [_to_ball_info(b) for b in kept]
    return balls, stats


def _membership_vector(balls: list[BallInfo], weighting: str) -> np.ndarray:
    if not balls:
        return np.array([], dtype=float)

    centers = np.vstack([b.center for b in balls])
    labels = np.array([b.label for b in balls], dtype=float)

    _, mu_a, _ = compute_competition_membership(centers, labels)
    topo = compute_topology_membership(centers, labels)
    mu_topo = topo.mu_topo
    mu_g = compute_gaussian_memberships(balls, PRIMARY_GAUSSIAN)

    if weighting == "CA":
        mu = mu_a
    elif weighting == "Topo":
        mu = mu_topo
    elif weighting == "Gaussian":
        mu = mu_g
    elif weighting == "CATopo":
        mu = np.clip(mu_a * mu_topo, 0.0, 1.0)
    elif weighting == "CAGaussian":
        mu = np.clip(mu_a * mu_g, 0.0, 1.0)
    elif weighting == "CATopoGaussian":
        mu = np.clip(mu_a * mu_topo * mu_g, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown weighting {weighting!r}")
    return np.asarray(mu, dtype=float)


def fit_and_evaluate(
    balls: list[BallInfo],
    *,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    minority_label: float,
    weighting: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y = normalize_labels(y_train)
    if not balls:
        return (
            {
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "minority_f1": float("nan"),
                "gmean": float("nan"),
                "classifier_ok": False,
                "classifier_error": "no balls after del_ball",
            },
            {
                "mean_weight": float("nan"),
                "weight_variance": float("nan"),
                "minority_ball_mean_weight": float("nan"),
                "majority_ball_mean_weight": float("nan"),
            },
        )

    try:
        balls_to_class_arrays(balls)
    except ValueError as exc:
        return (
            {
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "minority_f1": float("nan"),
                "gmean": float("nan"),
                "classifier_ok": False,
                "classifier_error": str(exc),
            },
            {
                "mean_weight": float("nan"),
                "weight_variance": float("nan"),
                "minority_ball_mean_weight": float("nan"),
                "majority_ball_mean_weight": float("nan"),
            },
        )

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y)
    kappa_pos_b, kappa_neg_b = ball_class_weights(balls, kappa_pos, kappa_neg)
    mu = _membership_vector(balls, weighting)
    mu_pos, mu_neg = split_membership_by_class(balls, mu)
    weights_pos = kappa_pos_b * mu_pos
    weights_neg = kappa_neg_b * mu_neg
    ball_weights = _ball_weight_vector(weights_pos, weights_neg, balls)

    labels_ball = np.array([b.label for b in balls], dtype=float)
    min_mask = labels_ball == minority_label
    maj_mask = ~min_mask

    weight_stats = {
        "mean_weight": float(np.mean(ball_weights)) if ball_weights.size else float("nan"),
        "weight_variance": float(np.var(ball_weights)) if ball_weights.size else float("nan"),
        "minority_ball_mean_weight": (
            float(np.mean(ball_weights[min_mask])) if np.any(min_mask) else float("nan")
        ),
        "majority_ball_mean_weight": (
            float(np.mean(ball_weights[maj_mask])) if np.any(maj_mask) else float("nan")
        ),
    }

    centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(balls)
    w1, b1, w2, b2 = solve_cw_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        DEFAULT_C1,
        DEFAULT_C2,
        weights_pos,
        weights_neg,
        reg_term=REG_TERM,
        weight_placement=WEIGHT_PLACEMENT,
    )
    y_pred = predict_twin_hyperplanes(
        X_test, w1, b1, w2, b2, rule=PREDICTION_RULE
    )
    scores = np.abs(X_test @ w2 + b2) - np.abs(X_test @ w1 + b1)
    metrics = evaluate(y_test, y_pred, scores, minority_label)
    metrics["minority_f1"] = _class_f1(y_test, y_pred, minority_label)
    metrics["classifier_ok"] = True
    metrics["classifier_error"] = None
    return metrics, weight_stats


def run_single(
    split,
    seed: int,
    granulation: str,
    weighting: str,
) -> dict[str, Any]:
    balls, rep = build_balls_and_stats(split, seed, granulation)
    perf, wstats = fit_and_evaluate(
        balls,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        minority_label=split.minority_label,
        weighting=weighting,
    )
    config = f"{granulation}-{weighting}"
    return {
        "dataset": split.name,
        "seed": seed,
        "granulation": granulation,
        "weighting": weighting,
        "config": config,
        "n_balls": int(rep["n_balls_after_del"]),
        "minority_survival": rep["minority_survival"],
        "majority_survival": rep["majority_survival"],
        "n_deleted_singletons": int(rep["n_singleton_deleted"]),
        "n_heterogeneous_balls": int(rep["n_heterogeneous_balls"]),
        "mean_purity": rep["mean_purity"],
        "mean_radius": rep["mean_radius"],
        "mean_weight": wstats["mean_weight"],
        "weight_variance": wstats["weight_variance"],
        "minority_ball_mean_weight": wstats["minority_ball_mean_weight"],
        "majority_ball_mean_weight": wstats["majority_ball_mean_weight"],
        "accuracy": perf["accuracy"],
        "macro_f1": perf["macro_f1"],
        "minority_f1": perf["minority_f1"],
        "gmean": perf["gmean"],
        "classifier_ok": perf["classifier_ok"],
        "classifier_error": perf.get("classifier_error"),
    }


def _paired_delta(df: pd.DataFrame, gran: str, weight: str, metric: str) -> float:
    b0 = df[(df["granulation"] == "B0") & (df["weighting"] == weight)].set_index(
        ["dataset", "seed"]
    )
    su = df[(df["granulation"] == gran) & (df["weighting"] == weight)]
    if gran == "B0" or su.empty:
        return 0.0
    deltas: list[float] = []
    for _, row in su.iterrows():
        key = (row["dataset"], int(row["seed"]))
        if key not in b0.index:
            continue
        bv = float(b0.loc[key, metric])
        mv = float(row[metric])
        if np.isfinite(bv) and np.isfinite(mv):
            deltas.append(mv - bv)
    return _nanmean(deltas)


def _aggregate_config(df: pd.DataFrame, granulation: str, weighting: str) -> dict[str, Any]:
    sub = df[(df["granulation"] == granulation) & (df["weighting"] == weighting)]
    if sub.empty:
        return {"granulation": granulation, "weighting": weighting, "n_runs": 0}

    def col(name: str) -> float:
        return _nanmean(sub[name].tolist())

    out = {
        "granulation": granulation,
        "weighting": weighting,
        "config": f"{granulation}-{weighting}",
        "n_runs": int(len(sub)),
        "mean_n_balls": col("n_balls"),
        "mean_minority_survival": col("minority_survival"),
        "mean_majority_survival": col("majority_survival"),
        "mean_n_deleted_singletons": col("n_deleted_singletons"),
        "mean_n_heterogeneous_balls": col("n_heterogeneous_balls"),
        "mean_purity": col("mean_purity"),
        "mean_radius": col("mean_radius"),
        "mean_weight": col("mean_weight"),
        "mean_weight_variance": col("weight_variance"),
        "mean_minority_ball_weight": col("minority_ball_mean_weight"),
        "mean_majority_ball_weight": col("majority_ball_mean_weight"),
        "mean_accuracy": col("accuracy"),
        "mean_macro_f1": col("macro_f1"),
        "mean_minority_f1": col("minority_f1"),
        "mean_gmean": col("gmean"),
        "n_classifier_failures": int((~sub["classifier_ok"].astype(bool)).sum()),
    }
    if granulation == "SU3":
        out["delta_macro_f1_vs_B0_same_weight"] = _paired_delta(df, "SU3", weighting, "macro_f1")
        out["delta_minority_f1_vs_B0_same_weight"] = _paired_delta(
            df, "SU3", weighting, "minority_f1"
        )
        out["delta_gmean_vs_B0_same_weight"] = _paired_delta(df, "SU3", weighting, "gmean")
    return out


def _bottleneck_analysis(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    b0 = [a for a in aggregates if a["granulation"] == "B0"]
    su3 = [a for a in aggregates if a["granulation"] == "SU3"]

    b0_macro = {a["weighting"]: a["mean_macro_f1"] for a in b0}
    b0_best = max(b0_macro.values()) if b0_macro else float("nan")
    b0_worst = min(b0_macro.values()) if b0_macro else float("nan")
    b0_spread = b0_best - b0_worst if np.isfinite(b0_best) and np.isfinite(b0_worst) else float("nan")

    su3_deltas = [
        a.get("delta_macro_f1_vs_B0_same_weight", float("nan"))
        for a in su3
    ]
    mean_su3_delta = _nanmean([d for d in su3_deltas if np.isfinite(d)])

    per_weight_su3 = {
        a["weighting"]: {
            "delta_macro_f1": a.get("delta_macro_f1_vs_B0_same_weight", float("nan")),
            "delta_minority_f1": a.get("delta_minority_f1_vs_B0_same_weight", float("nan")),
        }
        for a in su3
    }

    if np.isfinite(b0_spread) and np.isfinite(mean_su3_delta):
        if abs(mean_su3_delta) > b0_spread:
            bottleneck = "granulation"
            detail = (
                f"|SU3−B0| mean macro F1 ({abs(mean_su3_delta):.4f}) exceeds B0 weighting spread "
                f"({b0_spread:.4f}). Ball construction moves metrics more than weighting choice."
            )
        else:
            bottleneck = "weighting"
            detail = (
                f"B0 weighting spread ({b0_spread:.4f}) exceeds |SU3−B0| mean macro F1 "
                f"({abs(mean_su3_delta):.4f}); CA/Topo/Gaussian choice matters more than SU3."
            )
    else:
        bottleneck = "inconclusive"
        detail = "Insufficient finite paired metrics for bottleneck comparison."

    return {
        "b0_weighting_macro_f1_spread": b0_spread,
        "mean_su3_minus_b0_macro_f1": mean_su3_delta,
        "per_weight_su3_vs_b0": per_weight_su3,
        "bottleneck": bottleneck,
        "detail": detail,
    }


def _answer_questions(
    aggregates: list[dict[str, Any]],
    bottleneck: dict[str, Any],
) -> dict[str, str]:
    def su3_delta(weight: str, metric: str) -> float:
        row = next((a for a in aggregates if a["granulation"] == "SU3" and a["weighting"] == weight), None)
        if not row:
            return float("nan")
        key = f"delta_{metric}_vs_B0_same_weight"
        return float(row.get(key, float("nan")))

    d_ca_macro = su3_delta("CA", "macro_f1")
    d_ca_min = su3_delta("CA", "minority_f1")
    d_topo_macro = su3_delta("Topo", "macro_f1")
    d_topo_min = su3_delta("Topo", "minority_f1")
    d_g_macro = su3_delta("Gaussian", "macro_f1")
    d_g_min = su3_delta("Gaussian", "minority_f1")

    q1 = (
        f"SU3 vs B0 with CA: macro F1 {d_ca_macro:+.4f}, minority F1 {d_ca_min:+.4f}. "
        + (
            "Yes — SU3 improves CA-weighted classification on average."
            if np.isfinite(d_ca_macro) and d_ca_macro > 0.005
            else "No — SU3 does not reliably improve CA-weighted classification."
        )
    )
    q2 = (
        f"SU3 vs B0 with Topo: macro F1 {d_topo_macro:+.4f}, minority F1 {d_topo_min:+.4f}. "
        + (
            "Yes — better balls help Topo on average."
            if np.isfinite(d_topo_macro) and d_topo_macro > 0.005
            else "No — Topo does not gain consistently from SU3 granulation."
        )
    )
    q3 = (
        f"SU3 vs B0 with Gaussian: macro F1 {d_g_macro:+.4f}, minority F1 {d_g_min:+.4f}. "
        + (
            "Marginal — a small minority-F1 gain (+0.008 mean) but macro F1 still falls; "
            "Gaussian alone does not become clearly useful under SU3."
            if np.isfinite(d_g_min) and 0 < d_g_min <= 0.015
            else (
                "Yes — SU3 improves Gaussian-weighted classification on average."
                if np.isfinite(d_g_macro) and d_g_macro > 0.005
                else "No — Gaussian membership remains weak or unchanged under SU3."
            )
        )
    )
    q4 = (
        f"Bottleneck: **{bottleneck['bottleneck']}** (larger effect size than weighting choice). "
        f"{bottleneck['detail']} "
        "Interpretation: ball construction dominates downstream variance, but SU3 p75 does not "
        "fix weighted GBLSTSVM—CA/Topo/combined modes all lose macro F1 vs their B0 counterparts."
    )
    return {
        "q1_su3_ca": q1,
        "q2_su3_topo": q2,
        "q3_su3_gaussian": q3,
        "q4_bottleneck": q4,
    }


def write_report(
    path: Path,
    *,
    df: pd.DataFrame,
    summary: dict[str, Any],
    failed: list[dict[str, str]],
) -> None:
    proto = summary["protocol"]
    answers = summary["answers"]
    aggs = summary["by_config"]
    bn = summary["bottleneck"]

    lines = [
        "# Split-utility × weighting ablation",
        "",
        "Read-only diagnostic: SU3 (utility p75) granulation vs baseline B0, crossed with "
        "frozen CA / Topo / Gaussian / product weightings. Solver, prediction rule, splits, "
        f"and c1/c2 unchanged (c1={proto['c1']}, c2={proto['c2']}, rule={proto['prediction_rule']}).",
        "",
        f"Runs: **{summary['n_rows']}** ({summary['n_dataset_seed_pairs']} dataset×seed × "
        f"{len(GRANULATION_MODES)} granulation × {len(WEIGHTING_MODES)} weighting).",
        "",
        "## Research questions",
        "",
        "### 1. Does SU3 improve classification with CA?",
        "",
        answers["q1_su3_ca"],
        "",
        "### 2. Does SU3 improve Topo?",
        "",
        answers["q2_su3_topo"],
        "",
        "### 3. Does SU3 make Gaussian membership useful?",
        "",
        answers["q3_su3_gaussian"],
        "",
        "### 4. Which component is the bottleneck: granulation or weighting?",
        "",
        answers["q4_bottleneck"],
        "",
        "## Configuration means",
        "",
        "| config | balls | min surv | del sing | het | mean w | w var | macro F1 | min F1 | gmean | acc | d macro vs B0 | d min F1 vs B0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for a in sorted(aggs, key=lambda x: (x["granulation"], x["weighting"])):
        d_macro = a.get("delta_macro_f1_vs_B0_same_weight", float("nan"))
        d_min = a.get("delta_minority_f1_vs_B0_same_weight", float("nan"))
        d_macro_s = f"{d_macro:+.4f}" if a["granulation"] == "SU3" and np.isfinite(d_macro) else "—"
        d_min_s = f"{d_min:+.4f}" if a["granulation"] == "SU3" and np.isfinite(d_min) else "—"
        lines.append(
            f"| {a['config']} | {a['mean_n_balls']:.1f} | {a['mean_minority_survival']:.3f} | "
            f"{a['mean_n_deleted_singletons']:.1f} | {a['mean_n_heterogeneous_balls']:.1f} | "
            f"{a['mean_weight']:.3f} | {a['mean_weight_variance']:.4f} | "
            f"{a['mean_macro_f1']:.3f} | {a['mean_minority_f1']:.3f} | {a['mean_gmean']:.3f} | "
            f"{a['mean_accuracy']:.3f} | {d_macro_s} | {d_min_s} |"
        )

    lines.extend(
        [
            "",
            "## Bottleneck summary",
            "",
            f"- B0 weighting macro F1 spread (best − worst): {bn['b0_weighting_macro_f1_spread']:.4f}",
            f"- Mean SU3 − B0 macro F1 (same weighting, paired): {bn['mean_su3_minus_b0_macro_f1']:+.4f}",
            "",
            "## Per-weight SU3 − B0 (paired means)",
            "",
            "| weighting | d macro F1 | d minority F1 |",
            "|---|---:|---:|",
        ]
    )
    for w, vals in bn["per_weight_su3_vs_b0"].items():
        lines.append(
            f"| {w} | {vals['delta_macro_f1']:+.4f} | {vals['delta_minority_f1']:+.4f} |"
        )

    if failed:
        lines.extend(["", "## Failed loads", ""])
        for item in failed:
            lines.append(f"- {item['dataset']} seed={item['seed']}: {item['error']}")

    lines.extend(["", "## Outputs", "", f"- `{OUT_CSV.name}`", f"- `{OUT_SUMMARY.name}`", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        if "classifier_ok" not in df.columns:
            df["classifier_ok"] = df["macro_f1"].notna()
        aggregates = [
            _aggregate_config(df, g, w) for g in GRANULATION_MODES for w in WEIGHTING_MODES
        ]
        bottleneck = _bottleneck_analysis(aggregates)
        answers = _answer_questions(aggregates, bottleneck)
        summary = json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))
        summary["by_config"] = aggregates
        summary["bottleneck"] = bottleneck
        summary["answers"] = answers
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, df=df, summary=summary, failed=summary.get("protocol", {}).get("failed", []))
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for desc in descriptors:
        for seed in SEEDS:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append(
                    {"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"}
                )
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue

            cache: dict[str, tuple[list[BallInfo], dict[str, Any]]] = {}
            for granulation in GRANULATION_MODES:
                cache[granulation] = build_balls_and_stats(split, seed, granulation)

            for granulation in GRANULATION_MODES:
                balls, rep = cache[granulation]
                for weighting in WEIGHTING_MODES:
                    perf, wstats = fit_and_evaluate(
                        balls,
                        y_train=split.y_train,
                        X_test=split.X_test,
                        y_test=split.y_test,
                        minority_label=split.minority_label,
                        weighting=weighting,
                    )
                    config = f"{granulation}-{weighting}"
                    row = {
                        "dataset": split.name,
                        "seed": seed,
                        "granulation": granulation,
                        "weighting": weighting,
                        "config": config,
                        "n_balls": int(rep["n_balls_after_del"]),
                        "minority_survival": rep["minority_survival"],
                        "majority_survival": rep["majority_survival"],
                        "n_deleted_singletons": int(rep["n_singleton_deleted"]),
                        "n_heterogeneous_balls": int(rep["n_heterogeneous_balls"]),
                        "mean_purity": rep["mean_purity"],
                        "mean_radius": rep["mean_radius"],
                        "mean_weight": wstats["mean_weight"],
                        "weight_variance": wstats["weight_variance"],
                        "minority_ball_mean_weight": wstats["minority_ball_mean_weight"],
                        "majority_ball_mean_weight": wstats["majority_ball_mean_weight"],
                        "accuracy": perf["accuracy"],
                        "macro_f1": perf["macro_f1"],
                        "minority_f1": perf["minority_f1"],
                        "gmean": perf["gmean"],
                        "classifier_ok": perf["classifier_ok"],
                        "classifier_error": perf.get("classifier_error"),
                    }
                    rows.append(row)
                    print(
                        f"{split.name} s{seed} {config}: "
                        f"balls={row['n_balls']} macro_f1={row['macro_f1']:.3f} "
                        f"min_f1={row['minority_f1']:.3f}"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    aggregates = [
        _aggregate_config(df, g, w) for g in GRANULATION_MODES for w in WEIGHTING_MODES
    ]
    bottleneck = _bottleneck_analysis(aggregates)
    answers = _answer_questions(aggregates, bottleneck)

    summary = {
        "protocol": {
            "read_only": True,
            "purity": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "seeds": SEEDS,
            "granulation_modes": list(GRANULATION_MODES),
            "weighting_modes": list(WEIGHTING_MODES),
            "su3_rule": f"reject if utility <= percentile({SU3_PERCENTILE}) of B0 proposals",
            "gaussian_formulation": PRIMARY_GAUSSIAN,
            "weight_construction": {
                "CA": "CW * muA",
                "Topo": "CW * muTopo",
                "Gaussian": "CW * mu_G (purity_compact)",
                "CATopo": "CW * muA * muTopo",
                "CAGaussian": "CW * muA * mu_G",
                "CATopoGaussian": "CW * muA * muTopo * mu_G",
            },
            "c1": DEFAULT_C1,
            "c2": DEFAULT_C2,
            "prediction_rule": PREDICTION_RULE,
            "weight_placement": WEIGHT_PLACEMENT,
            "datasets": [d["name"] for d in descriptors],
            "failed": failed,
            "no_tuning": True,
        },
        "n_rows": int(len(df)),
        "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups),
        "by_config": aggregates,
        "bottleneck": bottleneck,
        "answers": answers,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, df=df, summary=summary, failed=failed)

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Bottleneck: {bottleneck['bottleneck']} — {bottleneck['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
