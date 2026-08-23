"""
Granulation sensitivity / reliability mechanism validation (Phase 5).

Varies ONLY purity_threshold on fixed synthetic datasets to test whether
lower thresholds produce heterogeneous ball purity/geometry and thereby
give the reliability mechanism dynamic range.

Does NOT tune for performance. Does NOT modify frozen implementations.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.cr_gblstsvm import CRGBLSTSVM
from src.cw_gblstsvm import CWGBLSTSVM
from src.gblstsvm import GBLSTSVM
from src.granular_balls import generate_granular_balls
from src.r_gblstsvm import RGBLSTSVM
from src.reliability import (
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_R,
    normalize_radius,
)

# Fixed experimental constants (Phases 2–4)
SEED = 42
STABILITY_SEEDS = [42, 7, 123, 99]
N_FEATURES = 6
TEST_SIZE = 0.3
C1 = 100.0
C2 = 1e-2
MIN_BALL_COUNT = 2
PREDICTION_RULE = "reference"
CANDIDATE_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98]
WEIGHT_TOLERANCE = 1e-9

SCENARIOS = [
    ("balanced", 100, 100),
    ("imbalanced_1_3", 50, 150),
    ("imbalanced_1_5", 33, 167),
]

MODEL_NAMES = ["GBLSTSVM", "CW-GBLSTSVM", "R-GBLSTSVM", "CR-GBLSTSVM"]
MODEL_CLASSES = {
    "GBLSTSVM": GBLSTSVM,
    "CW-GBLSTSVM": CWGBLSTSVM,
    "R-GBLSTSVM": RGBLSTSVM,
    "CR-GBLSTSVM": CRGBLSTSVM,
}


def make_dataset(
    n_minority: int, n_majority: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_minority + n_majority
    X = rng.standard_normal((n, N_FEATURES))
    X[:n_minority, 0] += 1.5
    X[n_minority:, 0] -= 1.5
    y = np.array([1.0] * n_minority + [-1.0] * n_majority)
    return X, y


def distribution_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def saturation_stats(reliabilities: np.ndarray) -> dict:
    if reliabilities.size == 0:
        return {}
    n = reliabilities.size
    return {
        "frac_ge_0_99": float(np.sum(reliabilities >= 0.99) / n),
        "frac_ge_0_95": float(np.sum(reliabilities >= 0.95) / n),
        "frac_lt_0_95": float(np.sum(reliabilities < 0.95) / n),
        "frac_lt_0_90": float(np.sum(reliabilities < 0.90) / n),
    }


def minority_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true == 1
    if mask.sum() == 0:
        return float("nan")
    return float(recall_score(y_true[mask], y_pred[mask], pos_label=1.0))


def minority_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        precision_score(y_true, y_pred, pos_label=1.0, zero_division=0)
    )


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds = model.predict(X_test)
    y_bin = np.where(y_test > 0, 1.0, -1.0)
    return {
        "accuracy": float(accuracy_score(y_bin, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_bin, preds)),
        "macro_f1": float(f1_score(y_bin, preds, average="macro")),
        "minority_recall": minority_recall(y_bin, preds),
        "minority_precision": minority_precision(y_bin, preds),
    }


def inspect_granulation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    purity_threshold: float,
) -> dict | None:
    """Return ball-structure summary; None if both classes not represented."""
    np.random.seed(SEED)
    try:
        balls = generate_granular_balls(
            X_train,
            y_train,
            purity_threshold=purity_threshold,
            min_ball_count=MIN_BALL_COUNT,
        )
    except Exception:
        return None

    pos = [b for b in balls if b.label == 1]
    neg = [b for b in balls if b.label != 1]
    if not pos or not neg:
        return {"viable": False, "reason": "missing_class", "purity_threshold": purity_threshold}

    purities = np.array([b.purity for b in balls], dtype=float)
    radii = np.array([b.radius for b in balls], dtype=float)
    sizes = np.array([b.num_samples for b in balls], dtype=int)

    return {
        "viable": True,
        "purity_threshold": purity_threshold,
        "n_balls": len(balls),
        "n_balls_pos": len(pos),
        "n_balls_neg": len(neg),
        "purity_std": float(np.std(purities)),
        "purity_mean": float(np.mean(purities)),
        "purity_min": float(np.min(purities)),
        "purity_max": float(np.max(purities)),
        "radius_mean": float(np.mean(radii)),
        "singleton_count": int(np.sum(sizes == 1)),
        "n_le_2": int(np.sum(sizes <= 2)),
        "n_le_5": int(np.sum(sizes <= 5)),
    }


def ball_statistics(balls) -> dict:
    purities = np.array([b.purity for b in balls], dtype=float)
    radii = np.array([b.radius for b in balls], dtype=float)
    sizes = np.array([b.num_samples for b in balls], dtype=int)
    pos = [b for b in balls if b.label == 1]
    neg = [b for b in balls if b.label != 1]
    n = len(balls)

    return {
        "n_balls": n,
        "n_balls_pos": len(pos),
        "n_balls_neg": len(neg),
        "purity": distribution_summary(purities),
        "radius": distribution_summary(radii),
        "singletons": {
            "count": int(np.sum(sizes == 1)),
            "fraction": float(np.sum(sizes == 1) / n) if n else 0.0,
        },
        "small_balls": {
            "n_le_2": int(np.sum(sizes <= 2)),
            "n_le_5": int(np.sum(sizes <= 5)),
        },
    }


def weight_class_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def weight_diagnostics(cr_model: CRGBLSTSVM) -> dict:
    cw_pos = cr_model.kappa_pos_balls_
    cw_neg = cr_model.kappa_neg_balls_
    cr_pos = cr_model.weights_pos_
    cr_neg = cr_model.weights_neg_

    cw_all = np.concatenate([cw_pos, cw_neg])
    cr_all = np.concatenate([cr_pos, cr_neg])

    if cw_all.size >= 2 and np.std(cw_all) > 0 and np.std(cr_all) > 0:
        corr = float(np.corrcoef(cw_all, cr_all)[0, 1])
    else:
        corr = float("nan")

    diff = np.abs(cr_all - cw_all)
    frac_near = float(np.mean(diff < WEIGHT_TOLERANCE))

    return {
        "kappa": {
            "pos": weight_class_stats(cw_pos),
            "neg": weight_class_stats(cw_neg),
            "all": weight_class_stats(cw_all),
        },
        "reliability": distribution_summary(cr_model.reliabilities_),
        "reliability_saturation": saturation_stats(cr_model.reliabilities_),
        "cr_final_weight": {
            "pos": weight_class_stats(cr_pos),
            "neg": weight_class_stats(cr_neg),
            "all": weight_class_stats(cr_all),
        },
        "cw_vs_cr": {
            "correlation": corr,
            "fraction_near_equal": frac_near,
            "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
            "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        },
    }


def per_ball_rows(
    dataset: str,
    seed: int,
    purity_threshold: float,
    balls,
    cr_model: CRGBLSTSVM,
) -> list[dict]:
    radius_scale = cr_model.radius_scale_
    pos_idx = 0
    neg_idx = 0
    rows = []
    for i, ball in enumerate(balls):
        r_tilde = normalize_radius(ball.radius, radius_scale)
        reliability = float(cr_model.reliabilities_[i])

        if ball.label == 1:
            kappa = float(cr_model.kappa_pos_balls_[pos_idx])
            final_w = float(cr_model.weights_pos_[pos_idx])
            pos_idx += 1
        else:
            kappa = float(cr_model.kappa_neg_balls_[neg_idx])
            final_w = float(cr_model.weights_neg_[neg_idx])
            neg_idx += 1

        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "purity_threshold": purity_threshold,
                "class": int(ball.label),
                "ball_size": int(ball.num_samples),
                "purity": float(ball.purity),
                "radius": float(ball.radius),
                "normalized_radius": float(r_tilde),
                "reliability": reliability,
                "kappa": kappa,
                "final_cr_weight": final_w,
            }
        )
    return rows


def run_threshold_experiment(
    dataset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    purity_threshold: float,
    seed: int,
) -> dict | None:
    base_params = {
        "c1": C1,
        "c2": C2,
        "purity_threshold": purity_threshold,
        "min_ball_count": MIN_BALL_COUNT,
        "prediction_rule": PREDICTION_RULE,
        "random_state": seed,
    }
    r_params = {
        **base_params,
        "epsilon": DEFAULT_EPSILON,
        "gamma": DEFAULT_GAMMA,
        "lambda_r": DEFAULT_LAMBDA_R,
    }

    models = {}
    for name, cls in MODEL_CLASSES.items():
        params = r_params if name in ("R-GBLSTSVM", "CR-GBLSTSVM") else base_params
        try:
            models[name] = cls(**params).fit(X_train, y_train)
        except ValueError:
            return None

    cr = models["CR-GBLSTSVM"]
    cw = models["CW-GBLSTSVM"]
    r_model = models["R-GBLSTSVM"]

    metrics = {name: evaluate_model(m, X_test, y_test) for name, m in models.items()}
    cr_m = metrics["CR-GBLSTSVM"]
    cw_m = metrics["CW-GBLSTSVM"]
    r_m = metrics["R-GBLSTSVM"]

    delta_reliability = {
        "balanced_accuracy": cr_m["balanced_accuracy"] - cw_m["balanced_accuracy"],
        "macro_f1": cr_m["macro_f1"] - cw_m["macro_f1"],
        "minority_recall": cr_m["minority_recall"] - cw_m["minority_recall"],
    }
    delta_class = {
        "balanced_accuracy": cr_m["balanced_accuracy"] - r_m["balanced_accuracy"],
        "macro_f1": cr_m["macro_f1"] - r_m["macro_f1"],
        "minority_recall": cr_m["minority_recall"] - r_m["minority_recall"],
    }

    rel = cr.reliabilities_
    ball_stats = ball_statistics(cr.balls_)

    return {
        "dataset": dataset,
        "seed": seed,
        "purity_threshold": purity_threshold,
        "ball_statistics": ball_stats,
        "reliability": {
            **distribution_summary(rel),
            **saturation_stats(rel),
        },
        "weight_diagnostics": weight_diagnostics(cr),
        "metrics": metrics,
        "delta_reliability_cr_minus_cw": delta_reliability,
        "delta_class_cr_minus_r": delta_class,
        "_cr_model": cr,
        "_balls": cr.balls_,
    }


def flatten_for_csv(record: dict) -> dict:
    row = {
        "dataset": record["dataset"],
        "seed": record["seed"],
        "purity_threshold": record["purity_threshold"],
        "n_balls": record["ball_statistics"]["n_balls"],
        "n_balls_pos": record["ball_statistics"]["n_balls_pos"],
        "n_balls_neg": record["ball_statistics"]["n_balls_neg"],
        "purity_mean": record["ball_statistics"]["purity"].get("mean"),
        "purity_std": record["ball_statistics"]["purity"].get("std"),
        "reliability_mean": record["reliability"].get("mean"),
        "reliability_std": record["reliability"].get("std"),
        "frac_a_ge_0_99": record["reliability"].get("frac_ge_0_99"),
        "frac_a_ge_0_95": record["reliability"].get("frac_ge_0_95"),
        "frac_a_lt_0_95": record["reliability"].get("frac_lt_0_95"),
        "singleton_count": record["ball_statistics"]["singletons"]["count"],
        "cw_cr_correlation": record["weight_diagnostics"]["cw_vs_cr"]["correlation"],
        "frac_cr_near_cw": record["weight_diagnostics"]["cw_vs_cr"]["fraction_near_equal"],
        "delta_bal_acc": record["delta_reliability_cr_minus_cw"]["balanced_accuracy"],
        "delta_macro_f1": record["delta_reliability_cr_minus_cw"]["macro_f1"],
        "delta_min_recall": record["delta_reliability_cr_minus_cw"]["minority_recall"],
    }
    for model in MODEL_NAMES:
        m = record["metrics"][model]
        prefix = model.replace("-", "_").lower()
        row[f"{prefix}_acc"] = m["accuracy"]
        row[f"{prefix}_bal_acc"] = m["balanced_accuracy"]
        row[f"{prefix}_macro_f1"] = m["macro_f1"]
        row[f"{prefix}_min_recall"] = m["minority_recall"]
    return row


def compare_threshold_structures(inspections: list[dict]) -> list[dict]:
    """Identify which thresholds produce meaningfully different ball structures."""
    viable = [i for i in inspections if i.get("viable")]
    comparisons = []
    for i, a in enumerate(viable):
        for b in viable[i + 1 :]:
            comparisons.append(
                {
                    "threshold_a": a["purity_threshold"],
                    "threshold_b": b["purity_threshold"],
                    "delta_n_balls": b["n_balls"] - a["n_balls"],
                    "delta_purity_std": b["purity_std"] - a["purity_std"],
                    "delta_purity_min": b["purity_min"] - a["purity_min"],
                    "meaningfully_different": (
                        abs(b["n_balls"] - a["n_balls"]) >= 2
                        or abs(b["purity_std"] - a["purity_std"]) > 0.02
                        or abs(b["purity_min"] - a["purity_min"]) > 0.02
                    ),
                }
            )
    return comparisons


def create_plots(records: list[dict], out_dir: Path) -> None:
    """Create sensitivity plots aggregated by dataset (seed=42 primary runs)."""
    primary = [r for r in records if r["seed"] == SEED]
    datasets = sorted(set(r["dataset"] for r in primary))

    plot_specs = [
        (
            "reliability_std",
            lambda r: r["reliability"].get("std", float("nan")),
            "Reliability std",
            "granulation_sensitivity_plot_reliability_std.png",
        ),
        (
            "frac_a_ge_0_99",
            lambda r: r["reliability"].get("frac_ge_0_99", float("nan")),
            "Fraction(a >= 0.99)",
            "granulation_sensitivity_plot_frac_ge_099.png",
        ),
        (
            "delta_macro_f1",
            lambda r: r["delta_reliability_cr_minus_cw"]["macro_f1"],
            "CR - CW Macro F1",
            "granulation_sensitivity_plot_delta_macro_f1.png",
        ),
        (
            "delta_min_recall",
            lambda r: r["delta_reliability_cr_minus_cw"]["minority_recall"],
            "CR - CW Minority Recall",
            "granulation_sensitivity_plot_delta_min_recall.png",
        ),
    ]

    for _, y_fn, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ds in datasets:
            subset = sorted(
                [r for r in primary if r["dataset"] == ds],
                key=lambda x: x["purity_threshold"],
            )
            xs = [r["purity_threshold"] for r in subset]
            ys = [y_fn(r) for r in subset]
            ax.plot(xs, ys, marker="o", label=ds)
        ax.set_xlabel("purity_threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs purity_threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Inspect granulation across candidate thresholds ---
    granulation_inspection: dict[str, list] = {}
    for ds_name, n_min, n_maj in SCENARIOS:
        X, y = make_dataset(n_min, n_maj, SEED)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )
        inspections = []
        for rho in CANDIDATE_THRESHOLDS:
            result = inspect_granulation(X_train, y_train, rho)
            if result is not None:
                inspections.append(result)
        granulation_inspection[ds_name] = inspections

    # --- Step 2: Full experiments (primary seed + stability seeds) ---
    all_records: list[dict] = []
    per_ball_rows_all: list[dict] = []

    for ds_name, n_min, n_maj in SCENARIOS:
        for seed in STABILITY_SEEDS:
            X, y = make_dataset(n_min, n_maj, seed)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
            )

            for rho in CANDIDATE_THRESHOLDS:
                record = run_threshold_experiment(
                    ds_name, X_train, y_train, X_test, y_test, rho, seed
                )
                if record is None:
                    continue

                cr_model = record.pop("_cr_model")
                balls = record.pop("_balls")
                all_records.append(record)

                if seed == SEED:
                    per_ball_rows_all.extend(
                        per_ball_rows(ds_name, seed, rho, balls, cr_model)
                    )

    # --- Step 3: Threshold structure comparisons ---
    threshold_comparisons = {
        ds: compare_threshold_structures(inspections)
        for ds, inspections in granulation_inspection.items()
    }

    # --- Step 4: Write outputs ---
    json_payload = {
        "experiment_config": {
            "seed_primary": SEED,
            "stability_seeds": STABILITY_SEEDS,
            "candidate_thresholds": CANDIDATE_THRESHOLDS,
            "c1": C1,
            "c2": C2,
            "min_ball_count": MIN_BALL_COUNT,
            "test_size": TEST_SIZE,
            "prediction_rule": PREDICTION_RULE,
            "reliability_defaults": {
                "epsilon": DEFAULT_EPSILON,
                "gamma": DEFAULT_GAMMA,
                "lambda_r": DEFAULT_LAMBDA_R,
            },
        },
        "granulation_inspection": granulation_inspection,
        "threshold_structure_comparisons": threshold_comparisons,
        "experiments": all_records,
    }

    json_path = out_dir / "granulation_sensitivity.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    csv_rows = [flatten_for_csv(r) for r in all_records]
    if csv_rows:
        csv_path = out_dir / "granulation_sensitivity.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    per_ball_path = out_dir / "granulation_sensitivity_per_ball.csv"
    if per_ball_rows_all:
        with per_ball_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_ball_rows_all[0].keys()))
            writer.writeheader()
            writer.writerows(per_ball_rows_all)

    create_plots(all_records, out_dir)

    # --- Console summary ---
    print(f"Wrote {json_path}")
    print(f"Wrote {out_dir / 'granulation_sensitivity.csv'}")
    print(f"Wrote {per_ball_path}")
    print(f"Wrote plots to {out_dir}\n")

    print("=== GRANULATION INSPECTION (train split, seed=42) ===")
    for ds_name, inspections in granulation_inspection.items():
        print(f"\n--- {ds_name} ---")
        for insp in inspections:
            if not insp.get("viable"):
                print(f"  rho={insp['purity_threshold']:.2f}: NOT VIABLE ({insp.get('reason')})")
                continue
            print(
                f"  rho={insp['purity_threshold']:.2f}: "
                f"balls={insp['n_balls']} (pos={insp['n_balls_pos']}, neg={insp['n_balls_neg']}), "
                f"purity std={insp['purity_std']:.3f}, min={insp['purity_min']:.3f}, "
                f"singletons={insp['singleton_count']}"
            )

    print("\n=== PRIMARY RESULTS (seed=42, Delta CR-CW) ===")
    primary = [r for r in all_records if r["seed"] == SEED]
    for ds in sorted(set(r["dataset"] for r in primary)):
        print(f"\n--- {ds} ---")
        for r in sorted(primary, key=lambda x: x["purity_threshold"]):
            if r["dataset"] != ds:
                continue
            dr = r["delta_reliability_cr_minus_cw"]
            wd = r["weight_diagnostics"]["cw_vs_cr"]
            print(
                f"  rho={r['purity_threshold']:.2f}: "
                f"rel_std={r['reliability'].get('std', 0):.4f}, "
                f"frac(a>=0.99)={r['reliability'].get('frac_ge_0_99', 0):.2%}, "
                f"frac_CR~CW={wd['fraction_near_equal']:.2%}, "
                f"dF1={dr['macro_f1']:+.4f}, dRec={dr['minority_recall']:+.4f}"
            )


if __name__ == "__main__":
    main()
