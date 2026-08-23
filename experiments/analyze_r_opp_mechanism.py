"""
Mechanism-only R_opp analysis across completed Topo broad-screening runs.

Recomputes ball-level membership/weights from frozen splits (no classifier
training). Performance deltas are taken from topo_broad_screening_results.csv.

Run from GBLSTSVM root:
  python experiments/analyze_r_opp_mechanism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from src.cbcm import ball_cbcm_weights, safe_correlation  # noqa: E402
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.topology_membership import ball_topo_weights  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "topology"
SCREENING_CSV = OUT_DIR / "topo_broad_screening_results.csv"
OUT_CSV = OUT_DIR / "r_opp_mechanism_analysis.csv"
OUT_SUMMARY = OUT_DIR / "r_opp_mechanism_summary.json"

SEEDS = [42, 0, 1]

# Classification thresholds (fixed, not tuned on outcomes).
R_OPP_MIN_STD = 0.02
TOPO_CA_MIN_REL_L2 = 0.01
HURT_MACRO_F1 = -0.01
HURT_GMEAN = -0.015


def _var_stats(x: np.ndarray) -> dict:
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "var": np.nan}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "var": float(np.var(v)),
    }


def _ball_weight_vector(
    weights_pos: np.ndarray,
    weights_neg: np.ndarray,
    balls,
) -> np.ndarray:
    pos_i = 0
    neg_i = 0
    out = np.empty(len(balls), dtype=float)
    for i, b in enumerate(balls):
        if b.label == 1:
            out[i] = weights_pos[pos_i]
            pos_i += 1
        else:
            out[i] = weights_neg[neg_i]
            neg_i += 1
    return out


def _analyze_run(descriptor: dict, seed: int, perf: dict) -> dict:
    split, err = try_load_descriptor(descriptor, seed=seed)
    if split is None:
        raise RuntimeError(err)

    y_train = normalize_labels(split.y_train)
    np.random.seed(seed)
    balls = generate_granular_balls(
        split.X_train,
        y_train,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    ca_pos, ca_neg, ca_mem, _, _ = ball_cbcm_weights(
        balls, kappa_pos, kappa_neg, variant="competition"
    )
    topo_pos, topo_neg, topo_mem, _, _ = ball_topo_weights(
        balls, kappa_pos, kappa_neg
    )

    ca_w = _ball_weight_vector(ca_pos, ca_neg, balls)
    topo_w = _ball_weight_vector(topo_pos, topo_neg, balls)
    mu_a = ca_mem.mu_a
    r_opp = topo_mem.r_opp
    mu_topo = topo_mem.mu_topo
    comp = topo_mem.comp
    comp_a = topo_mem.ambiguity

    r_stats = _var_stats(r_opp)
    ca_stats = _var_stats(ca_w)
    topo_stats = _var_stats(topo_w)
    comp_stats = _var_stats(comp)
    comp_a_stats = _var_stats(comp_a)

    l2_diff = float(np.linalg.norm(topo_w - ca_w))
    ca_norm = float(np.linalg.norm(ca_w))
    rel_l2 = l2_diff / ca_norm if ca_norm > 0 else float("nan")

    row = {
        "dataset": split.name,
        "seed": seed,
        "n_balls": len(balls),
        "r_opp_mean": r_stats["mean"],
        "r_opp_median": r_stats["median"],
        "r_opp_std": r_stats["std"],
        "r_opp_var": r_stats["var"],
        "ca_weight_var": ca_stats["var"],
        "topo_weight_var": topo_stats["var"],
        "comp_mean": comp_stats["mean"],
        "comp_std": comp_stats["std"],
        "comp_var": comp_stats["var"],
        "compA_mean": comp_a_stats["mean"],
        "compA_std": comp_a_stats["std"],
        "compA_var": comp_a_stats["var"],
        "corr_ca_r_opp": safe_correlation(mu_a, r_opp, "pearson"),
        "corr_ca_topo": safe_correlation(ca_w, topo_w, "pearson"),
        "corr_muA_muTopo": safe_correlation(mu_a, mu_topo, "pearson"),
        "corr_r_opp_topo": safe_correlation(r_opp, mu_topo, "pearson"),
        "l2_topo_minus_ca": l2_diff,
        "rel_l2_topo_minus_ca": rel_l2,
        "frac_r_opp_gt_0": float(np.mean(r_opp > 0)) if r_opp.size else np.nan,
        "delta_macro_f1_topo_vs_cw": perf.get("delta_macro_f1_topo_vs_cw"),
        "delta_macro_f1_topo_vs_ca": perf.get("delta_macro_f1_topo_vs_ca"),
        "delta_minority_f1_topo_vs_cw": perf.get("delta_minority_f1_topo_vs_cw"),
        "delta_minority_f1_topo_vs_ca": perf.get("delta_minority_f1_topo_vs_ca"),
        "delta_gmean_topo_vs_cw": perf.get("delta_gmean_topo_vs_cw"),
        "delta_gmean_topo_vs_ca": perf.get("delta_gmean_topo_vs_ca"),
    }
    row["r_opp_category"] = _categorize(row)
    return row


def _categorize(row: dict) -> str:
    near_constant = (
        not np.isfinite(row["r_opp_std"])
        or row["r_opp_std"] < R_OPP_MIN_STD
        or row.get("frac_r_opp_gt_0", 1.0) <= 0.0
    )
    topo_collapses_to_ca = (
        (np.isfinite(row["rel_l2_topo_minus_ca"]) and row["rel_l2_topo_minus_ca"] < TOPO_CA_MIN_REL_L2)
        or (
            np.isfinite(row["corr_muA_muTopo"])
            and row["corr_muA_muTopo"] > 0.995
            and row["r_opp_std"] < 0.05
        )
    )

    if near_constant or topo_collapses_to_ca:
        return "redundant_r_opp"

    hurts = (
        (row.get("delta_macro_f1_topo_vs_ca") is not None and row["delta_macro_f1_topo_vs_ca"] < HURT_MACRO_F1)
        or (row.get("delta_gmean_topo_vs_ca") is not None and row["delta_gmean_topo_vs_ca"] < HURT_GMEAN)
    )
    differs = (
        np.isfinite(row["rel_l2_topo_minus_ca"])
        and row["rel_l2_topo_minus_ca"] >= TOPO_CA_MIN_REL_L2
    )

    if differs and hurts:
        return "adversarial_r_opp"
    if differs:
        return "informative_r_opp"
    return "redundant_r_opp"


def _correlation_summary(df: pd.DataFrame) -> dict:
    success = df["delta_macro_f1_topo_vs_ca"] > 0
    fail = df["delta_macro_f1_topo_vs_ca"] < 0

    def _mean_col(mask, col):
        if mask.sum() == 0:
            return None
        return float(df.loc[mask, col].mean())

    keys = [
        "r_opp_std",
        "r_opp_var",
        "ca_weight_var",
        "topo_weight_var",
        "corr_ca_r_opp",
        "corr_ca_topo",
        "corr_r_opp_topo",
        "rel_l2_topo_minus_ca",
        "compA_std",
    ]
    out = {}
    for k in keys:
        out[k] = {
            "mean_when_topo_beats_ca": _mean_col(success, k),
            "mean_when_topo_loses_to_ca": _mean_col(fail, k),
        }
    return out


def main() -> int:
    if not SCREENING_CSV.exists():
        raise FileNotFoundError(f"Missing screening results: {SCREENING_CSV}")

    screening = pd.read_csv(SCREENING_CSV)
    topo_perf = screening[screening["method"] == "Topo"].set_index(["dataset", "seed"])

    descriptors = {d["name"]: d for d in discover_available_datasets()}
    rows: list[dict] = []

    for (dataset, seed), perf_row in topo_perf.iterrows():
        desc = descriptors.get(dataset)
        if desc is None:
            continue
        perf = perf_row.to_dict()
        rows.append(_analyze_run(desc, int(seed), perf))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    cat_counts = df["r_opp_category"].value_counts().to_dict()
    corr_summary = _correlation_summary(df)

    # Threshold hints: median split on r_opp_std for wins vs losses
    wins = df[df["delta_macro_f1_topo_vs_ca"] > 0]
    losses = df[df["delta_macro_f1_topo_vs_ca"] < 0]
    threshold_hints = {}
    if len(wins) and len(losses):
        for col in ["r_opp_std", "rel_l2_topo_minus_ca", "corr_ca_r_opp", "topo_weight_var"]:
            w_med = float(wins[col].median())
            l_med = float(losses[col].median())
            threshold_hints[col] = {"wins_median": w_med, "losses_median": l_med}

    # Category vs outcome
    category_outcomes = {}
    for cat in df["r_opp_category"].unique():
        sub = df[df["r_opp_category"] == cat]
        category_outcomes[cat] = {
            "n": int(len(sub)),
            "mean_delta_macro_f1_topo_vs_ca": float(sub["delta_macro_f1_topo_vs_ca"].mean()),
            "mean_delta_gmean_topo_vs_ca": float(sub["delta_gmean_topo_vs_ca"].mean()),
            "frac_topo_beats_ca": float((sub["delta_macro_f1_topo_vs_ca"] > 0).mean()),
        }

    summary = {
        "n_runs": int(len(df)),
        "category_counts": cat_counts,
        "category_outcomes": category_outcomes,
        "correlation_summary": corr_summary,
        "threshold_hints": threshold_hints,
        "thresholds_used": {
            "R_OPP_MIN_STD": R_OPP_MIN_STD,
            "TOPO_CA_MIN_REL_L2": TOPO_CA_MIN_REL_L2,
            "HURT_MACRO_F1": HURT_MACRO_F1,
            "HURT_GMEAN": HURT_GMEAN,
        },
        "patterns": _narrative_patterns(df, cat_counts, corr_summary, threshold_hints),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print("Category counts:", cat_counts)
    return 0


def _narrative_patterns(df, cat_counts, corr_summary, threshold_hints) -> list[str]:
    patterns = []

    n_red = cat_counts.get("redundant_r_opp", 0)
    n_inf = cat_counts.get("informative_r_opp", 0)
    n_adv = cat_counts.get("adversarial_r_opp", 0)
    patterns.append(
        f"{n_red}/{len(df)} runs classified redundant (near-constant R_opp or Topo≈CA); "
        f"{n_inf} informative; {n_adv} adversarial."
    )

    cs = corr_summary.get("r_opp_std", {})
    if cs.get("mean_when_topo_beats_ca") is not None:
        patterns.append(
            f"Runs where Topo beats CA have higher mean r_opp_std "
            f"({cs['mean_when_topo_beats_ca']:.4f}) than losses "
            f"({cs['mean_when_topo_loses_to_ca']:.4f})."
        )

    cs2 = corr_summary.get("rel_l2_topo_minus_ca", {})
    if cs2.get("mean_when_topo_beats_ca") is not None:
        patterns.append(
            f"Successful runs show larger rel_l2_topo_minus_ca "
            f"({cs2['mean_when_topo_beats_ca']:.4f}) vs failed "
            f"({cs2['mean_when_topo_loses_to_ca']:.4f}) — Topo must differ from CA to help."
        )

    cs3 = corr_summary.get("corr_ca_r_opp", {})
    if cs3.get("mean_when_topo_beats_ca") is not None:
        patterns.append(
            f"corr(CA,R_opp) is lower when Topo wins "
            f"({cs3['mean_when_topo_beats_ca']:.3f}) than when it loses "
            f"({cs3['mean_when_topo_loses_to_ca']:.3f}) — R_opp adds most when not redundant with CA."
        )

    adv = df[df["r_opp_category"] == "adversarial_r_opp"]
    if len(adv):
        top_adv = adv.nsmallest(3, "delta_macro_f1_topo_vs_ca")[
            ["dataset", "seed", "delta_macro_f1_topo_vs_ca", "r_opp_std", "rel_l2_topo_minus_ca"]
        ]
        patterns.append(
            "Largest adversarial cases: "
            + ", ".join(
                f"{r.dataset}(s{int(r.seed)}, Δ={r.delta_macro_f1_topo_vs_ca:.3f})"
                for r in top_adv.itertuples()
            )
        )

    inf_wins = df[(df["r_opp_category"] == "informative_r_opp") & (df["delta_macro_f1_topo_vs_ca"] > 0)]
    if len(inf_wins):
        patterns.append(
            f"Informative R_opp runs with Topo>CA: {len(inf_wins)} "
            f"(e.g. {', '.join(inf_wins.nlargest(3, 'delta_macro_f1_topo_vs_ca')['dataset'].tolist())})."
        )

    return patterns


if __name__ == "__main__":
    raise SystemExit(main())
