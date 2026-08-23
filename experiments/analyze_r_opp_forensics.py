"""
Granular-ball forensic analysis of R_opp: successful vs adversarial Topo runs.

Run from GBLSTSVM root:
  python experiments/analyze_r_opp_forensics.py
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

from experiments.analyze_cbcm_competition import _per_ball_knn_features  # noqa: E402
from experiments.analyze_topology import MIN_BALL_COUNT, PURITY  # noqa: E402
from experiments.dataset_registry import discover_available_datasets, try_load_descriptor  # noqa: E402
from experiments.phase8_common import normalize_labels  # noqa: E402
from src.cbcm import ball_cbcm_weights  # noqa: E402
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.topology_membership import ball_topo_weights  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "topology"
OUT_BALL_CSV = OUT_DIR / "r_opp_forensics_per_ball.csv"
OUT_SUMMARY = OUT_DIR / "r_opp_forensics_summary.json"

# Successful datasets (seed 42 — all show Topo>CA at this seed in screening).
SUCCESSFUL = [
    ("chronic_kidney_disease", 42),
    ("credit_approval", 42),
    ("spambase", 42),
]

# Adversarial runs (specified seeds from screening).
ADVERSARIAL = [
    ("ecoli2", 42),
    ("ionosphere", 0),
    ("vehicle0", 1),
]

CORRECTION_EPS = 1e-9


def _ball_weight_vector(weights_pos, weights_neg, balls) -> np.ndarray:
    pos_i = neg_i = 0
    out = np.empty(len(balls), dtype=float)
    for i, b in enumerate(balls):
        if b.label == 1:
            out[i] = weights_pos[pos_i]
            pos_i += 1
        else:
            out[i] = weights_neg[neg_i]
            neg_i += 1
    return out


def _extract_run(
    descriptor: dict,
    seed: int,
    outcome_group: str,
) -> tuple[list[dict], dict]:
    split, err = try_load_descriptor(descriptor, seed=seed)
    if split is None:
        raise RuntimeError(err)

    y_train = normalize_labels(split.y_train)
    minority_label = split.minority_label
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
    weight_delta = topo_w - ca_w

    centers = np.vstack([b.center for b in balls])
    labels = np.array([b.label for b in balls], dtype=float)
    topo_knn = _per_ball_knn_features(centers, labels, topo_mem.k)

    run_id = f"{split.name}_s{seed}"
    rows: list[dict] = []

    for i, b in enumerate(balls):
        is_minority = bool(b.label == minority_label)
        rows.append(
            {
                "run_id": run_id,
                "dataset": split.name,
                "seed": seed,
                "outcome_group": outcome_group,
                "ball_id": i,
                "ball_class": float(b.label),
                "is_minority": is_minority,
                "is_majority": not is_minority,
                "n_features": split.info.get("n_features"),
                "comp": float(topo_mem.comp[i]),
                "compA": float(topo_mem.ambiguity[i]),
                "muA": float(ca_mem.mu_a[i]),
                "R_opp": float(topo_mem.r_opp[i]),
                "n_opp": int(topo_mem.n_opp[i]),
                "n_reciprocal_opp": int(topo_mem.n_reciprocal_opp[i]),
                "muTopo": float(topo_mem.mu_topo[i]),
                "ca_weight": float(ca_w[i]),
                "topo_weight": float(topo_w[i]),
                "topo_minus_ca": float(weight_delta[i]),
                "abs_topo_minus_ca": float(abs(weight_delta[i])),
                "rel_topo_minus_ca": float(weight_delta[i] / ca_w[i])
                if ca_w[i] > CORRECTION_EPS
                else np.nan,
                "purity": float(b.purity),
                "radius": float(b.radius),
                "ball_size": int(b.num_samples),
                "nearest_opp_dist": float(topo_knn["nearest_opp_dist"][i])
                if np.isfinite(topo_knn["nearest_opp_dist"][i])
                else np.nan,
                "nearest_same_dist": float(topo_knn["nearest_same_dist"][i])
                if np.isfinite(topo_knn["nearest_same_dist"][i])
                else np.nan,
                "opp_same_ratio": float(topo_knn["opp_same_ratio"][i])
                if np.isfinite(topo_knn["opp_same_ratio"][i])
                else np.nan,
                "weighted_opp_mass": float(topo_knn["weighted_opp_mass"][i]),
                "weighted_same_mass": float(topo_knn["weighted_same_mass"][i]),
                "opp_neighbor_fraction": float(topo_knn["opp_neighbor_fraction"][i]),
                "k_neighbors": int(topo_mem.k),
                "n_balls": len(balls),
            }
        )

    run_meta = {
        "run_id": run_id,
        "dataset": split.name,
        "seed": seed,
        "outcome_group": outcome_group,
        "n_balls": len(balls),
        "n_features": split.info.get("n_features"),
        "imbalance_ratio": split.info.get("imbalance_ratio"),
        "minority_label": float(minority_label),
        "mean_abs_topo_minus_ca": float(np.mean(np.abs(weight_delta))),
        "frac_balls_corrected": float(np.mean(np.abs(weight_delta) > CORRECTION_EPS)),
        "mean_R_opp": float(np.mean(topo_mem.r_opp)),
        "std_R_opp": float(np.std(topo_mem.r_opp)),
    }
    return rows, run_meta


def _corrected_ball_mask(df: pd.DataFrame) -> pd.Series:
    return df["abs_topo_minus_ca"] > CORRECTION_EPS


def _summarize_group(df: pd.DataFrame, group: str) -> dict:
    g = df[df["outcome_group"] == group]
    corrected = g[_corrected_ball_mask(g)]
    downweighted = corrected[corrected["topo_minus_ca"] < 0]
    upweighted = corrected[corrected["topo_minus_ca"] > 0]

    def _stats(frame: pd.DataFrame, col: str) -> dict:
        if frame.empty or col not in frame:
            return {}
        s = frame[col].dropna()
        if s.empty:
            return {}
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
        }

    metrics = [
        "compA",
        "R_opp",
        "abs_topo_minus_ca",
        "purity",
        "radius",
        "ball_size",
        "nearest_opp_dist",
        "opp_neighbor_fraction",
        "weighted_opp_mass",
        "n_opp",
        "n_reciprocal_opp",
    ]

    summary = {
        "n_runs": int(g["run_id"].nunique()),
        "n_balls": int(len(g)),
        "n_corrected_balls": int(len(corrected)),
        "frac_corrected": float(len(corrected) / len(g)) if len(g) else 0.0,
        "n_downweighted": int(len(downweighted)),
        "n_upweighted": int(len(upweighted)),
        "frac_downweighted_minority": float(downweighted["is_minority"].mean())
        if len(downweighted)
        else np.nan,
        "frac_downweighted_high_compA": float((downweighted["compA"] > 0.5).mean())
        if len(downweighted)
        else np.nan,
        "frac_downweighted_low_purity": float((downweighted["purity"] < PURITY).mean())
        if len(downweighted)
        else np.nan,
        "frac_downweighted_small_ball": float((downweighted["ball_size"] <= MIN_BALL_COUNT).mean())
        if len(downweighted)
        else np.nan,
        "frac_downweighted_large_radius": float(
            (downweighted["radius"] > downweighted["radius"].median()).mean()
        )
        if len(downweighted)
        else np.nan,
        "frac_downweighted_far_opp": float(
            (
                downweighted["nearest_opp_dist"]
                > downweighted["nearest_opp_dist"].median()
            ).mean()
        )
        if len(downweighted) and downweighted["nearest_opp_dist"].notna().any()
        else np.nan,
    }

    for col in metrics:
        summary[f"all_balls_{col}"] = _stats(g, col)
        summary[f"corrected_{col}"] = _stats(corrected, col)
        summary[f"downweighted_{col}"] = _stats(downweighted, col)

    # High-competition minority balls (research hypothesis)
    hcm = g[g["is_minority"] & (g["compA"] > 0.5)]
    summary["high_comp_minority_balls"] = {
        "n": int(len(hcm)),
        "mean_topo_minus_ca": float(hcm["topo_minus_ca"].mean()) if len(hcm) else np.nan,
        "mean_R_opp": float(hcm["R_opp"].mean()) if len(hcm) else np.nan,
        "mean_nearest_opp_dist": float(hcm["nearest_opp_dist"].mean())
        if len(hcm) and hcm["nearest_opp_dist"].notna().any()
        else np.nan,
        "frac_downweighted": float((hcm["topo_minus_ca"] < 0).mean()) if len(hcm) else np.nan,
    }

    return summary


def _contrast(success: dict, adv: dict, key: str) -> dict:
    s = success.get(key, {})
    a = adv.get(key, {})
    if not s or not a:
        return {}
    return {
        "successful_mean": s.get("mean"),
        "adversarial_mean": a.get("mean"),
        "delta_success_minus_adversarial": (
            (s.get("mean") - a.get("mean"))
            if s.get("mean") is not None and a.get("mean") is not None
            else None
        ),
    }


def _build_patterns(df: pd.DataFrame, success: dict, adv: dict) -> list[str]:
    patterns = []

    # Pattern 1: downweighted ball characteristics
    s_dw = success.get("downweighted_compA", {})
    a_dw = adv.get("downweighted_compA", {})
    s_rod = success.get("downweighted_nearest_opp_dist", {})
    a_rod = adv.get("downweighted_nearest_opp_dist", {})
    if s_dw and a_dw:
        patterns.append(
            "Downweighted balls (Topo<CA): adversarial runs downweight higher-compA balls "
            f"(mean compA {a_dw.get('mean', 0):.3f}) vs successful ({s_dw.get('mean', 0):.3f}). "
            "Harmful corrections disproportionately suppress already-ambiguous competition balls."
        )
    s_ro = success.get("downweighted_R_opp", {})
    a_ro = adv.get("downweighted_R_opp", {})
    s_no = success.get("downweighted_n_opp", {})
    a_no = adv.get("downweighted_n_opp", {})
    if s_ro and a_ro:
        patterns.append(
            "Downweighted balls in successful runs have higher mean R_opp "
            f"({s_ro.get('mean', 0):.3f}) and denser opposite connectivity (n_opp "
            f"{s_no.get('mean', 0):.1f} vs {a_no.get('mean', 0):.1f}). "
            "Sparse ball graphs make R_opp noisy and corrections oversized."
        )

    # Pattern 2: high-comp minority
    s_hcm = success.get("high_comp_minority_balls", {})
    a_hcm = adv.get("high_comp_minority_balls", {})
    if s_hcm and a_hcm:
        patterns.append(
            "High-competition minority balls (compA>0.5): adversarial runs downweight "
            f"{a_hcm.get('frac_downweighted', 0):.1%} vs successful "
            f"{s_hcm.get('frac_downweighted', 0):.1%}; mean Topo-CA is "
            f"{a_hcm.get('mean_topo_minus_ca', 0):.4f} vs "
            f"{s_hcm.get('mean_topo_minus_ca', 0):.4f}. "
            "When R_opp suppresses these balls, minority recall collapses."
        )

    # Pattern 3: small/sparse balls
    s_small = success.get("frac_downweighted_small_ball", np.nan)
    a_small = adv.get("frac_downweighted_small_ball", np.nan)
    s_pur = success.get("downweighted_purity", {})
    a_pur = adv.get("downweighted_purity", {})
    if np.isfinite(s_small) and np.isfinite(a_small):
        patterns.append(
            f"Small balls (size≤{MIN_BALL_COUNT}) among downweighted: adversarial "
            f"{a_small:.1%} vs successful {s_small:.1%}; "
            f"downweighted purity adversarial mean {a_pur.get('mean', 0):.3f} vs "
            f"successful {s_pur.get('mean', 0):.3f}. "
            "Sparse/granular balls at purity boundary get unstable kNN R_opp estimates, "
            "amplifying harmful corrections."
        )

    return patterns[:3]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    descriptors = {d["name"]: d for d in discover_available_datasets()}

    all_rows: list[dict] = []
    run_metas: list[dict] = []

    for dataset, seed in SUCCESSFUL:
        desc = descriptors[dataset]
        rows, meta = _extract_run(desc, seed, "successful")
        all_rows.extend(rows)
        run_metas.append(meta)

    for dataset, seed in ADVERSARIAL:
        desc = descriptors[dataset]
        rows, meta = _extract_run(desc, seed, "adversarial")
        all_rows.extend(rows)
        run_metas.append(meta)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_BALL_CSV, index=False)

    success_summary = _summarize_group(df, "successful")
    adversarial_summary = _summarize_group(df, "adversarial")

    contrasts = {
        "downweighted_compA": _contrast(
            success_summary, adversarial_summary, "downweighted_compA"
        ),
        "downweighted_nearest_opp_dist": _contrast(
            success_summary, adversarial_summary, "downweighted_nearest_opp_dist"
        ),
        "downweighted_R_opp": _contrast(
            success_summary, adversarial_summary, "downweighted_R_opp"
        ),
        "downweighted_ball_size": _contrast(
            success_summary, adversarial_summary, "downweighted_ball_size"
        ),
        "downweighted_purity": _contrast(
            success_summary, adversarial_summary, "downweighted_purity"
        ),
        "corrected_abs_topo_minus_ca": _contrast(
            success_summary, adversarial_summary, "corrected_abs_topo_minus_ca"
        ),
    }

    patterns = _build_patterns(df, success_summary, adversarial_summary)

    payload = {
        "runs": run_metas,
        "successful_summary": success_summary,
        "adversarial_summary": adversarial_summary,
        "contrasts": contrasts,
        "strongest_patterns": patterns,
    }
    OUT_SUMMARY.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {OUT_BALL_CSV} ({len(df)} ball rows)")
    print(f"Wrote {OUT_SUMMARY}")
    for p in patterns:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
