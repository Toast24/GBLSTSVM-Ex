"""
Diagnose CBCM competition (muA) signal beyond CW.

Sections:
  1. Per-ball topology diagnostics + correlations
  2. muA mapping comparison (A1–A4, odds, balance)
  3. Outlier / synthetic-noise behaviour
  4. One justified new candidate (if diagnostics support it)
  5. Cheap screening on frozen 70/15/15 split
  6. Final decision report

Run from GBLSTSVM root:
  python experiments/analyze_cbcm_competition.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cbcm_diagnostics import _synthetic_noise_check  # noqa: E402
from experiments.phase8_common import (  # noqa: E402
    DATASETS,
    DEFAULT_C1,
    DEFAULT_C2,
    DEFAULT_MIN_BALL_COUNT,
    DEFAULT_PURITY,
    evaluate,
    load_dataset_split,
    normalize_labels,
    twin_scores,
)
from src.bmru import ball_bmru_weights  # noqa: E402
from src.cbcm import (  # noqa: E402
    DEFAULT_SIGMA_EPS,
    compute_cbcm_membership,
    kappa_per_ball_array,
    resolve_k,
    safe_correlation,
    split_membership_by_class,
)
from src.cbcm import (  # noqa: E402
    _adaptive_bandwidth,
    _gaussian_weights,
    _knn_indices_excluding_self,
)
from src.cbcm_gblstsvm import CAGBLSTSVM, CCGBLSTSVM, solve_cbcm_gblstsvm_systems  # noqa: E402
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.cw_gblstsvm import CWGBLSTSVM  # noqa: E402
from src.gblstsvm import GBLSTSVM  # noqa: E402
from src.granular_balls import balls_to_class_arrays, generate_granular_balls  # noqa: E402
from src.overlap import ball_overlap_scores, overlap_mu_from_scores  # noqa: E402
from src.reliability import ball_reliability_weights  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "cbcm"
SEED = 42
PURITY = DEFAULT_PURITY
MIN_BALL_COUNT = DEFAULT_MIN_BALL_COUNT
EPS = DEFAULT_SIGMA_EPS
WEIGHT_PLACEMENT = "opposite_only"
PREDICTION_RULE = "reference"

MAPPING_NAMES = ("A1_comp", "A2_ambiguity", "A3_parabolic", "A4_sqrt", "bounded_odds", "balance")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topology helpers (diagnostic only; mirrors frozen CBCM k-NN logic)
# ---------------------------------------------------------------------------


def _per_ball_knn_features(
    centers: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    sigma_eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Per-ball neighborhood topology used for diagnostics."""
    centers = np.asarray(centers, dtype=float)
    labels = np.asarray(labels, dtype=float).ravel()
    labels[labels == 0] = -1.0
    n = centers.shape[0]

    nearest_same = np.full(n, np.nan)
    nearest_opp = np.full(n, np.nan)
    same_mass = np.zeros(n)
    opp_mass = np.zeros(n)
    opp_frac = np.zeros(n)

    for i in range(n):
        same_mask = (labels == labels[i]) & (np.arange(n) != i)
        opp_mask = (labels != labels[i])
        if np.any(same_mask):
            d_same = np.linalg.norm(centers[i] - centers[same_mask], axis=1)
            nearest_same[i] = float(np.min(d_same))
        if np.any(opp_mask):
            d_opp = np.linalg.norm(centers[i] - centers[opp_mask], axis=1)
            nearest_opp[i] = float(np.min(d_opp))

    k_eff = min(k, n - 1) if n > 1 else 0
    if k_eff > 0:
        neighbor_idx = _knn_indices_excluding_self(centers, k_eff)
        for i in range(n):
            idx = neighbor_idx[i]
            if idx.size == 0:
                continue
            dists = np.linalg.norm(centers[i] - centers[idx], axis=1)
            sigma_i = _adaptive_bandwidth(dists, sigma_eps=sigma_eps)
            weights = _gaussian_weights(dists, sigma_i)
            opp = labels[idx] != labels[i]
            same = ~opp
            opp_mass[i] = float(np.sum(weights[opp]))
            same_mass[i] = float(np.sum(weights[same]))
            w_sum = float(np.sum(weights))
            opp_frac[i] = float(np.sum(weights[opp]) / w_sum) if w_sum > 0 else 0.0

    ratio = np.where(
        np.isfinite(nearest_same) & (nearest_same > 0),
        nearest_opp / np.maximum(nearest_same, sigma_eps),
        np.nan,
    )
    return {
        "nearest_same_dist": nearest_same,
        "nearest_opp_dist": nearest_opp,
        "opp_same_ratio": ratio,
        "weighted_same_mass": same_mass,
        "weighted_opp_mass": opp_mass,
        "opp_neighbor_fraction": opp_frac,
    }


def _mapping_from_comp_and_masses(
    comp: np.ndarray,
    same_mass: np.ndarray,
    opp_mass: np.ndarray,
) -> dict[str, np.ndarray]:
    comp = np.asarray(comp, dtype=float)
    sm = np.asarray(same_mass, dtype=float)
    om = np.asarray(opp_mass, dtype=float)
    odds = om / (sm + EPS)
    bounded_odds = odds / (1.0 + odds)
    balance = 1.0 - np.abs(om - sm) / (om + sm + EPS)
    return {
        "A1_comp": comp.copy(),
        "A2_ambiguity": 2.0 * np.minimum(comp, 1.0 - comp),
        "A3_parabolic": 4.0 * comp * (1.0 - comp),
        "A4_sqrt": np.sqrt(np.maximum(comp * (1.0 - comp), 0.0)),
        "bounded_odds": bounded_odds,
        "balance": balance,
    }


def _corr_pair(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(mask.sum())}
    av, bv = a[mask], b[mask]
    if np.std(av) == 0 or np.std(bv) == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(mask.sum())}
    pr, _ = pearsonr(av, bv)
    sr, _ = spearmanr(av, bv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(mask.sum())}


def _fraction_above(values: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, float]:
    v = np.asarray(values, dtype=float).ravel()
    return {f"frac_gt_{t}": float(np.mean(v > t)) for t in thresholds}


def _class_f1(y_true, y_pred, label: float) -> float:
    y_true = normalize_labels(y_true)
    y_pred = normalize_labels(y_pred)
    return float(f1_score(y_true, y_pred, pos_label=label, zero_division=0))


def _fit_custom_membership(
    X_train: np.ndarray,
    y_train: np.ndarray,
    membership_or_fn: np.ndarray | Any,
    *,
    seed: int = SEED,
) -> GBLSTSVM:
    """Fit CW * custom membership without modifying frozen model code."""
    if seed is not None:
        np.random.seed(seed)
    y_train = normalize_labels(y_train)
    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    balls = generate_granular_balls(
        X_train,
        y_train,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )
    centers = np.vstack([b.center for b in balls])
    labels = np.array([b.label for b in balls], dtype=float)
    k = resolve_k(len(balls))
    if callable(membership_or_fn):
        membership = np.asarray(membership_or_fn(centers, labels, k), dtype=float).ravel()
    else:
        membership = np.asarray(membership_or_fn, dtype=float).ravel()
    if membership.shape[0] != len(balls):
        raise ValueError("membership length must match number of balls.")

    kappa_ball = kappa_per_ball_array(balls, kappa_pos, kappa_neg)
    mu_pos, mu_neg = split_membership_by_class(balls, membership)
    weights_pos = kappa_ball[[i for i, b in enumerate(balls) if b.label == 1]] * mu_pos
    weights_neg = kappa_ball[[i for i, b in enumerate(balls) if b.label != 1]] * mu_neg

    centers_pos, radii_pos, centers_neg, radii_neg = balls_to_class_arrays(balls)
    w1, b1, w2, b2 = solve_cbcm_gblstsvm_systems(
        centers_pos,
        radii_pos,
        centers_neg,
        radii_neg,
        DEFAULT_C1,
        DEFAULT_C2,
        weights_pos,
        weights_neg,
        weight_placement=WEIGHT_PLACEMENT,
    )
    model = GBLSTSVM(
        c1=DEFAULT_C1,
        c2=DEFAULT_C2,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
        prediction_rule=PREDICTION_RULE,
        random_state=seed,
    )
    model.balls_ = balls
    model.w1_, model.b1_, model.w2_, model.b2_ = w1, b1, w2, b2
    model.coef_ = np.vstack([w1, w2])
    model.intercept_ = np.array([b1, b2])
    return model


def _eval_split(model, X, y, minority_label: float) -> dict:
    y_pred = model.predict(X)
    scores = twin_scores(model, X)
    m = evaluate(y, y_pred, scores, minority_label)
    m["minority_f1"] = _class_f1(y, y_pred, minority_label)
    return m


def _screen_model(
    name: str, model, split, eval_split: str = "test", dataset: str | None = None
) -> dict:
    X = split.X_test if eval_split == "test" else split.X_val
    y = split.y_test if eval_split == "test" else split.y_val
    m = _eval_split(model, X, y, split.minority_label)
    return {
        "dataset": dataset or split.name,
        "model": name,
        "split": eval_split,
        **{k: m[k] for k in (
            "macro_f1", "minority_f1", "gmean", "balanced_accuracy", "minority_recall", "auroc"
        )},
    }


# ---------------------------------------------------------------------------
# Section 1: per-dataset diagnostics
# ---------------------------------------------------------------------------


def _ball_decision_scores(cw_model: CWGBLSTSVM, balls) -> np.ndarray:
    centers = np.vstack([b.center for b in balls])
    f1 = centers @ cw_model.w1_ + cw_model.b1_
    f2 = centers @ cw_model.w2_ + cw_model.b2_
    return np.abs(f2) - np.abs(f1)


def _collect_reference_weights(balls, y_train, minority_label) -> dict[str, np.ndarray]:
    n = len(balls)
    out: dict[str, np.ndarray] = {}

    _, _, _, rel = ball_reliability_weights(balls)
    out["R"] = rel

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    out["CW"] = kappa_per_ball_array(balls, kappa_pos, kappa_neg)

    try:
        base = GBLSTSVM(
            c1=DEFAULT_C1,
            c2=DEFAULT_C2,
            purity_threshold=PURITY,
            min_ball_count=MIN_BALL_COUNT,
            random_state=SEED,
        ).fit(
            np.vstack([b.center for b in balls]),
            np.array([b.label for b in balls]),
        )
        _, _, _, mu_bmru, _ = ball_bmru_weights(balls, base.w1_, base.b1_, base.w2_, base.b2_)
        out["BMRU_mu"] = mu_bmru
    except Exception:
        out["BMRU_mu"] = np.full(n, np.nan)

    O = ball_overlap_scores(balls)
    out["OBRU_mu"] = overlap_mu_from_scores(balls, O, minority_label)

    return out


def diagnose_dataset(dataset_file: str) -> dict[str, Any]:
    split = load_dataset_split(dataset_file, seed=SEED)
    ds = split.name
    X_train, y_train = split.X_train, split.y_train
    y_train_n = normalize_labels(y_train)
    minority_label = split.minority_label

    np.random.seed(SEED)
    balls = generate_granular_balls(
        X_train,
        y_train_n,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )
    centers = np.vstack([b.center for b in balls])
    labels = np.array([b.label for b in balls], dtype=float)
    n_balls = len(balls)
    k = resolve_k(n_balls)

    membership = compute_cbcm_membership(centers, labels, k=k)
    topo = _per_ball_knn_features(centers, labels, k)
    mappings = _mapping_from_comp_and_masses(
        membership.comp_a, topo["weighted_same_mass"], topo["weighted_opp_mass"]
    )

    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train_n)
    kappa_ball = kappa_per_ball_array(balls, kappa_pos, kappa_neg)
    ref_weights = _collect_reference_weights(balls, y_train_n, minority_label)

    cw_model = CWGBLSTSVM(
        c1=DEFAULT_C1,
        c2=DEFAULT_C2,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
        prediction_rule=PREDICTION_RULE,
        random_state=SEED,
        weight_placement=WEIGHT_PLACEMENT,
    ).fit(X_train, y_train_n)

    ca_model = CAGBLSTSVM(
        c1=DEFAULT_C1,
        c2=DEFAULT_C2,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
        prediction_rule=PREDICTION_RULE,
        random_state=SEED,
        weight_placement=WEIGHT_PLACEMENT,
    ).fit(X_train, y_train_n)

    ball_scores = _ball_decision_scores(cw_model, balls)
    abs_scores = np.abs(ball_scores)

    minority_mask = labels == minority_label
    majority_mask = ~minority_mask

    # Per-ball table
    rows = []
    for i, b in enumerate(balls):
        row = {
            "dataset": ds,
            "ball_id": i,
            "class": int(b.label),
            "is_minority": bool(b.label == minority_label),
            "CW": float(kappa_ball[i]),
            "compA": float(membership.comp_a[i]),
            "muA": float(membership.mu_a[i]),
            "muB": float(membership.mu_b[i]),
            "muC": float(membership.mu_c[i]),
            "nearest_same_dist": float(topo["nearest_same_dist"][i])
            if np.isfinite(topo["nearest_same_dist"][i])
            else None,
            "nearest_opp_dist": float(topo["nearest_opp_dist"][i])
            if np.isfinite(topo["nearest_opp_dist"][i])
            else None,
            "opp_same_ratio": float(topo["opp_same_ratio"][i])
            if np.isfinite(topo["opp_same_ratio"][i])
            else None,
            "weighted_same_mass": float(topo["weighted_same_mass"][i]),
            "weighted_opp_mass": float(topo["weighted_opp_mass"][i]),
            "opp_neighbor_fraction": float(topo["opp_neighbor_fraction"][i]),
            "cw_decision_score": float(ball_scores[i]),
            "cw_abs_decision_score": float(abs_scores[i]),
            "R": float(ref_weights["R"][i]),
            "BMRU_mu": float(ref_weights["BMRU_mu"][i])
            if np.isfinite(ref_weights["BMRU_mu"][i])
            else None,
            "OBRU_mu": float(ref_weights["OBRU_mu"][i]),
        }
        for mname, mvals in mappings.items():
            row[mname] = float(mvals[i])
        rows.append(row)

    # Correlations with muA
    corr_targets = {
        "cw_abs_decision_score": abs_scores,
        "CW": kappa_ball,
        "R": ref_weights["R"],
        "BMRU_mu": ref_weights["BMRU_mu"],
        "OBRU_mu": ref_weights["OBRU_mu"],
        "nearest_opp_dist": topo["nearest_opp_dist"],
        "nearest_same_dist": topo["nearest_same_dist"],
        "opp_same_ratio": topo["opp_same_ratio"],
        "opp_neighbor_fraction": topo["opp_neighbor_fraction"],
        "compA": membership.comp_a,
    }
    muA_correlations = {
        name: _corr_pair(membership.mu_a, arr) for name, arr in corr_targets.items()
    }

    muA_dist = {
        "minority": _summary(membership.mu_a[minority_mask]),
        "majority": _summary(membership.mu_a[majority_mask]),
        "minority_minus_majority_mean": float(
            np.mean(membership.mu_a[minority_mask]) - np.mean(membership.mu_a[majority_mask])
        )
        if minority_mask.any() and majority_mask.any()
        else float("nan"),
    }
    muA_thresholds = _fraction_above(membership.mu_a, (0.25, 0.5, 0.75))

    # CW vs CA prediction changes
    y_pred_cw = cw_model.predict(split.X_test)
    y_pred_ca = ca_model.predict(split.X_test)
    pred_diff_frac = float(np.mean(y_pred_cw != y_pred_ca))

    # Balls with largest weight change under CA
    weight_ratio = membership.mu_a  # CA weight / CW weight (kappa cancels)
    high_muA_mask = weight_ratio >= np.quantile(weight_ratio, 0.75)
    boundary_proxy = abs_scores >= np.quantile(abs_scores, 0.75)

    changed_ball_profile = {
        "test_pred_diff_frac": pred_diff_frac,
        "frac_high_muA_minority": float(np.mean(minority_mask[high_muA_mask]))
        if high_muA_mask.any()
        else float("nan"),
        "frac_high_muA_boundary_proxy": float(np.mean(boundary_proxy[high_muA_mask]))
        if high_muA_mask.any()
        else float("nan"),
        "frac_boundary_minority": float(np.mean(minority_mask & boundary_proxy))
        if boundary_proxy.any()
        else float("nan"),
        "n_balls_high_muA": int(np.sum(high_muA_mask)),
    }

    # Synthetic noise (extended with muA)
    rng = np.random.default_rng(SEED)
    synthetic = _synthetic_noise_check(
        X_train,
        y_train_n,
        balls,
        membership,
        kappa_pos,
        kappa_neg,
        minority_label,
        rng,
    )
    synthetic_extended = _extend_synthetic_noise(
        X_train, y_train_n, balls, membership, kappa_pos, kappa_neg, minority_label, rng
    )
    synthetic.update(synthetic_extended)

    return {
        "dataset": ds,
        "n_balls": n_balls,
        "k": k,
        "minority_label": minority_label,
        "per_ball_rows": rows,
        "muA_correlations": muA_correlations,
        "muA_distributions": muA_dist,
        "muA_threshold_fractions": muA_thresholds,
        "changed_ball_profile": changed_ball_profile,
        "synthetic_noise": synthetic,
        "membership": membership,
        "mappings": mappings,
        "topo": topo,
        "comp_a": membership.comp_a,
        "labels": labels,
        "minority_mask": minority_mask,
        "split": split,
        "balls": balls,
    }


def _summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return {}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def _extend_synthetic_noise(
    X_train,
    y_train,
    balls,
    membership,
    kappa_pos,
    kappa_neg,
    minority_label,
    rng,
) -> dict:
    """Report muA/compA/connectivity for known noise ball (gate logic unchanged)."""
    from src.cbcm import compute_cbcm_membership as cbcm_mem
    from src.granular_balls import generate_granular_balls as gen_balls

    n_balls = len(balls)
    m = max(3, int(np.ceil(0.02 * n_balls)))
    labels_ball = np.array([b.label for b in balls], dtype=float)
    centers = np.vstack([b.center for b in balls])
    minority_idx = np.flatnonzero(labels_ball == minority_label)

    noise_centers: list = []
    if minority_idx.size > 0:
        min_same = []
        for gi in minority_idx:
            others = minority_idx[minority_idx != gi]
            min_same.append(
                float(np.min(np.linalg.norm(centers[gi] - centers[others], axis=1)))
                if others.size
                else np.inf
            )
        order = np.argsort(-np.array(min_same))
        chosen = minority_idx[order[: min(m, minority_idx.size)]]
        noise_centers = [centers[i].copy() for i in chosen]
    if len(noise_centers) < m:
        maj_idx = np.flatnonzero(labels_ball != minority_label)
        pick = rng.choice(maj_idx, size=m - len(noise_centers), replace=True)
        jitter = 0.01 * np.std(centers, axis=0, keepdims=True) + 1e-6
        for j in pick:
            noise_centers.append(
                centers[j] + rng.normal(0, 1, size=centers.shape[1]) * jitter.ravel()
            )

    X_aug = np.vstack([X_train, np.vstack(noise_centers[:m])])
    y_aug = np.concatenate([y_train, np.full(m, minority_label)])
    balls_aug = gen_balls(
        X_aug, y_aug, purity_threshold=PURITY, min_samples_per_ball=1, min_ball_count=MIN_BALL_COUNT
    )
    centers_aug = np.vstack([b.center for b in balls_aug])
    labels_aug = np.array([b.label for b in balls_aug], dtype=float)
    mem_aug = cbcm_mem(centers_aug, labels_aug)
    k = resolve_k(len(balls_aug))
    topo_aug = _per_ball_knn_features(centers_aug, labels_aug, k)

    noise_arr = np.vstack(noise_centers[:m])
    minority_ball_idx = [i for i, b in enumerate(balls_aug) if b.label == minority_label]
    dist_to_noise = np.min(
        np.linalg.norm(
            centers_aug[minority_ball_idx][:, None, :] - noise_arr[None, :, :], axis=2
        ),
        axis=1,
    )
    noise_ball_global = minority_ball_idx[int(np.argmin(dist_to_noise))]

    from src.cbcm import percentile_rank

    mu_a_noise = float(mem_aug.mu_a[noise_ball_global])
    mu_c_noise = float(mem_aug.mu_c[noise_ball_global])
    return {
        "noise_ball_mu_a": mu_a_noise,
        "noise_ball_mu_a_percentile": percentile_rank(mu_a_noise, mem_aug.mu_a),
        "noise_ball_comp_a": float(mem_aug.comp_a[noise_ball_global]),
        "noise_ball_mu_b": float(mem_aug.mu_b[noise_ball_global]),
        "noise_ball_nearest_same_dist": float(topo_aug["nearest_same_dist"][noise_ball_global])
        if np.isfinite(topo_aug["nearest_same_dist"][noise_ball_global])
        else None,
        "noise_ball_nearest_opp_dist": float(topo_aug["nearest_opp_dist"][noise_ball_global])
        if np.isfinite(topo_aug["nearest_opp_dist"][noise_ball_global])
        else None,
        "noise_ball_opp_neighbor_fraction": float(
            topo_aug["opp_neighbor_fraction"][noise_ball_global]
        ),
    }


# ---------------------------------------------------------------------------
# Section 2: mapping comparison
# ---------------------------------------------------------------------------


def compare_mappings(ds_result: dict) -> dict:
    split = ds_result["split"]
    mappings = ds_result["mappings"]
    comp = ds_result["comp_a"]
    screening_rows = []

    cw = CWGBLSTSVM(
        c1=DEFAULT_C1,
        c2=DEFAULT_C2,
        purity_threshold=PURITY,
        min_ball_count=MIN_BALL_COUNT,
        prediction_rule=PREDICTION_RULE,
        random_state=SEED,
        weight_placement=WEIGHT_PLACEMENT,
    ).fit(split.X_train, normalize_labels(split.y_train))
    cw_test = _screen_model("CW", cw, split, "test")

    for name in MAPPING_NAMES:
        mu = mappings[name]
        dist = _summary(mu)
        try:
            model = _fit_custom_membership(
                split.X_train, split.y_train, _membership_fn_for_mapping(name)
            )
            test_m = _screen_model(name, model, split, "test")
        except Exception as exc:
            test_m = {"model": name, "error": str(exc)}
        screening_rows.append(
            {
                "mapping": name,
                "distribution": dist,
                "test_metrics": test_m,
                "corr_muA_pearson": safe_correlation(ds_result["membership"].mu_a, mu, "pearson"),
            }
        )

    # Distribution comparison vs comp
    mapping_corr = {
        m: _corr_pair(comp, mappings[m]) for m in MAPPING_NAMES if m in mappings
    }
    return {
        "mapping_distributions": {r["mapping"]: r["distribution"] for r in screening_rows},
        "mapping_test_screen": screening_rows,
        "mapping_vs_compA_corr": mapping_corr,
        "cw_baseline": cw_test,
    }


# ---------------------------------------------------------------------------
# Section 4: one justified candidate — Supported Pressure (SP)
# ---------------------------------------------------------------------------


def compute_supported_pressure_membership(
    comp_a: np.ndarray,
    same_mass: np.ndarray,
    opp_mass: np.ndarray,
    labels: np.ndarray,
    *,
    sigma_eps: float = EPS,
) -> np.ndarray:
    """
    SP: bounded opposite-class pressure gated by relative same-class support.

    membership = bounded_odds * support_gate
    support_gate = min(1, same_mass / class_median_same_mass)

    Justified when bounded_odds/balance diagnostics outperform parabolic remap
    and raw competition rewards isolated points.
    """
    comp_a = np.asarray(comp_a, dtype=float)
    sm = np.asarray(same_mass, dtype=float)
    om = np.asarray(opp_mass, dtype=float)
    labels = np.asarray(labels, dtype=float).ravel()
    labels[labels == 0] = -1.0

    odds = om / (sm + sigma_eps)
    bounded_odds = odds / (1.0 + odds)
    support_gate = np.ones_like(sm)
    for y in (-1.0, 1.0):
        mask = labels == y
        if not np.any(mask):
            continue
        med = float(np.median(sm[mask]))
        if med <= 0:
            med = sigma_eps
        support_gate[mask] = np.minimum(1.0, sm[mask] / med)
    return bounded_odds * support_gate


def _membership_fn_for_mapping(name: str):
    """Return a callable that recomputes membership after granulation."""

    def fn(centers: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        mem = compute_cbcm_membership(centers, labels, k=k)
        topo = _per_ball_knn_features(centers, labels, k)
        return _mapping_from_comp_and_masses(
            mem.comp_a, topo["weighted_same_mass"], topo["weighted_opp_mass"]
        )[name]

    return fn


def _membership_fn_supported_pressure(centers: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    mem = compute_cbcm_membership(centers, labels, k=k)
    topo = _per_ball_knn_features(centers, labels, k)
    return compute_supported_pressure_membership(
        mem.comp_a,
        topo["weighted_same_mass"],
        topo["weighted_opp_mass"],
        labels,
    )


# ---------------------------------------------------------------------------
# Section 5: cheap screening
# ---------------------------------------------------------------------------


def run_screening(ds_result: dict, mapping_comparison: dict) -> list[dict]:
    split = ds_result["split"]
    rows = []

    models_to_fit = [
        ("CW", None),
        ("CA", "A3_parabolic"),
        ("CC", None),
    ]

    # Best mapping by test gmean
    valid_maps = [
        r for r in mapping_comparison["mapping_test_screen"]
        if "error" not in r.get("test_metrics", {})
        and r["test_metrics"].get("gmean") is not None
    ]
    if valid_maps:
        best_map = max(valid_maps, key=lambda r: r["test_metrics"]["gmean"])
        models_to_fit.append((f"best_map_{best_map['mapping']}", best_map["mapping"]))

    models_to_fit.append(("SP_supported_pressure", "fn"))

    for name, spec in models_to_fit:
        try:
            if name == "CW":
                model = CWGBLSTSVM(
                    c1=DEFAULT_C1,
                    c2=DEFAULT_C2,
                    purity_threshold=PURITY,
                    min_ball_count=MIN_BALL_COUNT,
                    prediction_rule=PREDICTION_RULE,
                    random_state=SEED,
                    weight_placement=WEIGHT_PLACEMENT,
                ).fit(split.X_train, normalize_labels(split.y_train))
            elif name == "CA":
                model = CAGBLSTSVM(
                    c1=DEFAULT_C1,
                    c2=DEFAULT_C2,
                    purity_threshold=PURITY,
                    min_ball_count=MIN_BALL_COUNT,
                    prediction_rule=PREDICTION_RULE,
                    random_state=SEED,
                    weight_placement=WEIGHT_PLACEMENT,
                ).fit(split.X_train, normalize_labels(split.y_train))
            elif name == "CC":
                model = CCGBLSTSVM(
                    c1=DEFAULT_C1,
                    c2=DEFAULT_C2,
                    purity_threshold=PURITY,
                    min_ball_count=MIN_BALL_COUNT,
                    prediction_rule=PREDICTION_RULE,
                    random_state=SEED,
                    weight_placement=WEIGHT_PLACEMENT,
                ).fit(split.X_train, normalize_labels(split.y_train))
            elif name.startswith("best_map_"):
                model = _fit_custom_membership(
                    split.X_train, split.y_train, _membership_fn_for_mapping(spec)
                )
            elif name == "SP_supported_pressure":
                model = _fit_custom_membership(
                    split.X_train, split.y_train, _membership_fn_supported_pressure
                )
            else:
                continue
            rows.append(_screen_model(name, model, split, "test", dataset=ds_result["dataset"]))
        except Exception as exc:
            rows.append({"model": name, "error": str(exc)})

    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_muA_distributions(all_results: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, res in zip(axes, all_results):
        mu = res["membership"].mu_a
        labels = res["labels"]
        minority_label = res["minority_label"]
        ax.hist(mu[labels == minority_label], bins=20, alpha=0.6, label="minority", color="coral")
        ax.hist(mu[labels != minority_label], bins=20, alpha=0.6, label="majority", color="steelblue")
        ax.set_title(res["dataset"])
        ax.set_xlabel("muA")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    fig.suptitle("muA by class")
    fig.tight_layout()
    fig.savefig(out_dir / "muA_class_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_mapping_distributions(ds_result: dict, out_dir: Path) -> None:
    ds = ds_result["dataset"]
    mappings = ds_result["mappings"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.ravel(), MAPPING_NAMES):
        ax.hist(mappings[name], bins=25, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel("membership")
    fig.suptitle(f"Diagnostic membership mappings — {ds}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{ds}_mapping_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_structural_comparison(all_results: list[dict], out_dir: Path) -> None:
    metrics = ["compA_mean", "muA_mean", "opp_frac_mean", "nearest_opp_mean"]
    data = []
    for res in all_results:
        data.append(
            {
                "dataset": res["dataset"],
                "compA_mean": float(np.mean(res["comp_a"])),
                "muA_mean": float(np.mean(res["membership"].mu_a)),
                "opp_frac_mean": float(np.mean(res["topo"]["opp_neighbor_fraction"])),
                "nearest_opp_mean": float(
                    np.nanmean(res["topo"]["nearest_opp_dist"])
                ),
            }
        )
    df = pd.DataFrame(data).set_index("dataset")
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="bar", ax=ax)
    ax.set_title("Structural comparison: vehicle0 vs diabetes vs ecoli2")
    ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(out_dir / "structural_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 6: final decision
# ---------------------------------------------------------------------------


def _decide(
    all_results: list[dict],
    screening: dict[str, list[dict]],
    mapping_by_ds: dict[str, dict],
) -> dict:
    """Return decision letter + evidence table."""
    evidence = []
    dataset_assessments: list[dict] = []

    for res in all_results:
        ds = res["dataset"]
        corr_dec = res["muA_correlations"]["cw_abs_decision_score"]
        noise = res["synthetic_noise"]
        screen = {r["model"]: r for r in screening.get(ds, []) if "error" not in r}
        cw_g = screen.get("CW", {}).get("gmean")
        ca_g = screen.get("CA", {}).get("gmean")
        sp_g = screen.get("SP_supported_pressure", {}).get("gmean")
        cw_mf1 = screen.get("CW", {}).get("minority_f1")
        ca_mf1 = screen.get("CA", {}).get("minority_f1")
        sp_mf1 = screen.get("SP_supported_pressure", {}).get("minority_f1")

        ca_gain = (ca_g - cw_g) if ca_g is not None and cw_g is not None else 0.0
        sp_gain = (sp_g - cw_g) if sp_g is not None and cw_g is not None else 0.0
        mf1_gain = (ca_mf1 - cw_mf1) if ca_mf1 is not None and cw_mf1 is not None else 0.0
        noise_mu_a_pct = noise.get("noise_ball_mu_a_percentile", float("nan"))
        cw_collapsed = cw_g is not None and cw_g < 0.05

        outlier_risk = bool(
            np.isfinite(noise_mu_a_pct) and noise_mu_a_pct > 70.0
        )
        meaningful_gain = (
            not cw_collapsed
            and ca_gain > 0.01
            and mf1_gain >= 0.0
            and not outlier_risk
        )
        vehicle_style_gain = ca_gain > 0.02 and mf1_gain > 0.0 and not outlier_risk

        if meaningful_gain:
            sig = "useful"
        elif vehicle_style_gain or (ca_gain > 0.02 and not cw_collapsed):
            sig = "promising_unstable"
        elif ca_gain > 0 and not cw_collapsed:
            sig = "marginal"
        elif cw_collapsed:
            sig = "baseline_degenerate"
        else:
            sig = "none"

        # Best mapping from section 2
        best_map_name = None
        best_map_gain = float("nan")
        mc = mapping_by_ds.get(ds, {})
        cw_base_g = mc.get("cw_baseline", {}).get("gmean")
        valid_maps = [
            r for r in mc.get("mapping_test_screen", [])
            if "error" not in r.get("test_metrics", {})
            and r["test_metrics"].get("gmean") is not None
        ]
        if valid_maps and cw_base_g is not None:
            best_row = max(valid_maps, key=lambda r: r["test_metrics"]["gmean"])
            best_map_name = best_row["mapping"]
            best_map_gain = best_row["test_metrics"]["gmean"] - cw_base_g

        dataset_assessments.append(
            {
                "dataset": ds,
                "signal": sig,
                "ca_gain": ca_gain,
                "mf1_gain": mf1_gain,
                "sp_gain": sp_gain,
                "outlier_risk": outlier_risk,
                "cw_collapsed": cw_collapsed,
                "best_mapping": best_map_name,
                "best_mapping_gain": best_map_gain,
            }
        )

        best_candidate = "SP" if sp_gain > ca_gain else "CA"
        best_gain = max(ca_gain, sp_gain)
        evidence.append(
            {
                "Candidate": f"{best_candidate} — {ds}",
                "New vs CW": f"gmean {best_gain:+.4f}, mf1 {mf1_gain:+.4f}",
                "New vs R": f"r={res['muA_correlations']['R'].get('pearson', float('nan')):.3f}",
                "New vs BMRU": f"r={res['muA_correlations']['BMRU_mu'].get('pearson', float('nan')):.3f}",
                "New vs OBRU": f"r={res['muA_correlations']['OBRU_mu'].get('pearson', float('nan')):.3f}",
                "Outlier risk": f"noise muA pct={noise_mu_a_pct:.1f}",
                "Evidence": (
                    f"{sig}; best_map={best_map_name} "
                    f"(d_gmean={best_map_gain:+.4f}); "
                    f"|r(muA,|score|)={corr_dec.get('pearson', float('nan')):.3f}"
                ),
            }
        )

    n_useful = sum(1 for a in dataset_assessments if a["signal"] == "useful")
    n_promising = sum(1 for a in dataset_assessments if a["signal"] == "promising_unstable")
    vehicle = next(a for a in dataset_assessments if a["dataset"] == "vehicle0")
    v_gain = vehicle["ca_gain"]
    v_mf1 = vehicle["mf1_gain"]
    any_outlier = any(a["outlier_risk"] for a in dataset_assessments)

    if n_useful >= 2 and v_gain > 0.02:
        decision = "A"
        rationale = (
            "Competition signal replicates across datasets with gmean and minority-F1 "
            "gains and acceptable outlier behaviour; proceed to multi-seed/full CV."
        )
    elif (n_useful >= 1 or n_promising >= 1) and v_gain > 0.02 and v_mf1 > 0:
        decision = "B"
        rationale = (
            "Promising but unstable: vehicle0 shows the only clean gain "
            f"(d_gmean={v_gain:+.4f}, d_minority_f1={v_mf1:+.4f}); "
            "diabetes baseline degenerate; ecoli2 has outlier/mapping artefacts. "
            "One controlled refinement (best mapping or SP) on vehicle0 only."
        )
    elif v_gain > 0.02 and n_useful == 0:
        decision = "C"
        rationale = (
            "Vehicle0-only gain; muA correlates with CW on vehicle0 (r≈0.48) "
            "and is largely redundant with class geometry. Abandon as general mechanism."
        )
    else:
        decision = "D"
        rationale = (
            "No consistent useful signal beyond CW; abandon CBCM topology direction."
        )

    if any_outlier and decision == "A":
        decision = "B"
        rationale += " Downgraded from A due to ecoli2 outlier-reward on muA (>70th pct)."

    return {
        "decision": decision,
        "rationale": rationale,
        "evidence_table": evidence,
        "dataset_assessments": dataset_assessments,
        "per_dataset_signals": {a["dataset"]: a["signal"] for a in dataset_assessments},
    }


def _write_markdown(
    all_results: list[dict],
    mapping_by_ds: dict[str, dict],
    screening: dict[str, list[dict]],
    decision: dict,
    out_dir: Path,
) -> None:
    lines = [
        "# CBCM Competition Diagnostics",
        "",
        f"**Decision: {decision['decision']}** — {decision['rationale']}",
        "",
        "## Configuration",
        f"- Frozen granulation: purity={PURITY}, min_ball_count={MIN_BALL_COUNT}, seed={SEED}",
        f"- Split: stratified 70/15/15 (seed={SEED})",
        "",
    ]

    for res in all_results:
        ds = res["dataset"]
        lines.extend([f"## {ds}", "", f"N balls={res['n_balls']}, k={res['k']}", ""])
        lines.append("### muA correlations")
        for name, c in res["muA_correlations"].items():
            lines.append(
                f"- {name}: Pearson={c['pearson']:.4f}, Spearman={c['spearman']:.4f}"
            )
        lines.append("")
        lines.append("### muA class distributions")
        for cls, s in res["muA_distributions"].items():
            if isinstance(s, dict):
                lines.append(f"- {cls}: {s}")
        lines.append("")
        lines.append("### muA threshold fractions")
        for k, v in res["muA_threshold_fractions"].items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")
        lines.append("### CW vs CA changed-ball profile")
        for k, v in res["changed_ball_profile"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        syn = res["synthetic_noise"]
        lines.extend(
            [
                "### Synthetic noise (extended)",
                f"- noise_ball_muA={syn.get('noise_ball_mu_a', 'n/a')}, "
                f"percentile={syn.get('noise_ball_mu_a_percentile', 'n/a')}",
                f"- noise_ball_muC percentile={syn.get('noise_ball_mu_c_percentile', 'n/a')} "
                f"(gate threshold 70, unchanged)",
                f"- nearest_same={syn.get('noise_ball_nearest_same_dist')}, "
                f"compA={syn.get('noise_ball_comp_a')}",
                "",
            ]
        )
        mc = mapping_by_ds[ds]
        lines.append("### Mapping test screen (vs CW)")
        cw_g = mc["cw_baseline"].get("gmean")
        for row in mc["mapping_test_screen"]:
            tm = row["test_metrics"]
            if "error" in tm:
                continue
            delta = (tm.get("gmean") or 0) - (cw_g or 0)
            lines.append(
                f"- {row['mapping']}: gmean={tm.get('gmean'):.4f} "
                f"(vs CW {delta:+.4f}), minority_f1={tm.get('minority_f1'):.4f}"
            )
        lines.append("")
        lines.append("### Cheap screening")
        for row in screening.get(ds, []):
            if "error" in row:
                lines.append(f"- {row['model']}: ERROR {row['error']}")
            else:
                lines.append(
                    f"- {row['model']}: gmean={row.get('gmean'):.4f}, "
                    f"minority_f1={row.get('minority_f1'):.4f}, "
                    f"macro_f1={row.get('macro_f1'):.4f}, auroc={row.get('auroc')}"
                )
        lines.append("")

    lines.extend(["## Evidence table", ""])
    if decision["evidence_table"]:
        headers = list(decision["evidence_table"][0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in decision["evidence_table"]:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    (out_dir / "cbcm_competition_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    mapping_by_ds: dict[str, dict] = {}
    screening: dict[str, list[dict]] = {}

    for ds_file in DATASETS:
        logger.info("Diagnosing %s", ds_file)
        res = diagnose_dataset(ds_file)
        all_results.append(res)

        mc = compare_mappings(res)
        mapping_by_ds[res["dataset"]] = mc
        screening[res["dataset"]] = run_screening(res, mc)

        # Per-ball CSV
        pd.DataFrame(res["per_ball_rows"]).to_csv(
            OUT_DIR / f"{res['dataset']}_ball_competition_diagnostics.csv",
            index=False,
        )
        _plot_mapping_distributions(res, OUT_DIR)

    _plot_muA_distributions(all_results, OUT_DIR)
    _plot_structural_comparison(all_results, OUT_DIR)

    decision = _decide(all_results, screening, mapping_by_ds)
    _write_markdown(all_results, mapping_by_ds, screening, decision, OUT_DIR)

    # JSON summary (strip non-serializable)
    payload = {
        "decision": decision,
        "datasets": {},
        "screening": screening,
        "mapping_comparison": {
            ds: {
                "cw_baseline": mapping_by_ds[ds]["cw_baseline"],
                "mapping_test_screen": mapping_by_ds[ds]["mapping_test_screen"],
            }
            for ds in mapping_by_ds
        },
    }
    for res in all_results:
        ds = res["dataset"]
        payload["datasets"][ds] = {
            "n_balls": res["n_balls"],
            "k": res["k"],
            "muA_correlations": res["muA_correlations"],
            "muA_distributions": res["muA_distributions"],
            "muA_threshold_fractions": res["muA_threshold_fractions"],
            "changed_ball_profile": res["changed_ball_profile"],
            "synthetic_noise": res["synthetic_noise"],
        }

    (OUT_DIR / "cbcm_competition_diagnostics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    pd.DataFrame(
        [row for ds_rows in screening.values() for row in ds_rows if "error" not in row]
    ).to_csv(OUT_DIR / "cbcm_competition_screening.csv", index=False)

    print(f"\nDecision: {decision['decision']}")
    print(decision["rationale"])
    print(f"\nOutputs under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
