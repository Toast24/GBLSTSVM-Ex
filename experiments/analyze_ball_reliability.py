"""
Diagnostic analysis of granular-ball purity, radius, and reliability.

Run from GBLSTSVM root:
  python experiments/analyze_ball_reliability.py

Inspects ball statistics BEFORE drawing conclusions about whether the
reliability score is useful. Writes JSON + CSV under experiments/output/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.granular_balls import generate_granular_balls
from src.reliability import (
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_R,
    ball_reliability_score,
    compute_radius_scale,
    normalize_radius,
)

SEED = 42
N_FEATURES = 6


def _summary_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
        }
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def make_balanced(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEATURES))
    X[: n // 2, 0] += 1.5
    X[n // 2 :, 0] -= 1.5
    y = np.array([1.0] * (n // 2) + [-1.0] * (n - n // 2))
    return X, y


def make_imbalanced(n_minority: int, n_majority: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_minority + n_majority
    X = rng.standard_normal((n, N_FEATURES))
    X[:n_minority, 0] += 1.5
    X[n_minority:, 0] -= 1.5
    y = np.array([1.0] * n_minority + [-1.0] * n_majority)
    return X, y


def analyze_configuration(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    purity_threshold: float,
    min_ball_count: int,
    epsilon: float = DEFAULT_EPSILON,
    gamma: float = DEFAULT_GAMMA,
    lambda_r: float = DEFAULT_LAMBDA_R,
) -> dict:
    np.random.seed(SEED)
    balls = generate_granular_balls(
        X,
        y,
        purity_threshold=purity_threshold,
        min_ball_count=min_ball_count,
    )

    radii = np.array([b.radius for b in balls], dtype=float)
    purities = np.array([b.purity for b in balls], dtype=float)
    sizes = np.array([b.num_samples for b in balls], dtype=int)
    labels = np.array([b.label for b in balls], dtype=float)

    radius_scale = compute_radius_scale(radii)
    r_tilde = np.array([normalize_radius(b.radius, radius_scale) for b in balls])
    reliabilities = np.array(
        [
            ball_reliability_score(
                b.purity,
                b.radius,
                radius_scale,
                epsilon=epsilon,
                gamma=gamma,
                lambda_r=lambda_r,
                variant="full",
            )
            for b in balls
        ]
    )

    per_ball = []
    for i, b in enumerate(balls):
        per_ball.append(
            {
                "ball_id": i,
                "class": int(b.label),
                "num_samples": int(b.num_samples),
                "purity": float(b.purity),
                "radius": float(b.radius),
                "radius_normalized": float(r_tilde[i]),
                "reliability": float(reliabilities[i]),
            }
        )

    singleton_mask = sizes == 1
    le2_mask = sizes <= 2
    le5_mask = sizes <= 5

    def subset_stats(mask: np.ndarray, label: str) -> dict:
        if mask.sum() == 0:
            return {"count": 0}
        return {
            "count": int(mask.sum()),
            "purity": _summary_stats(purities[mask]),
            "radius": _summary_stats(radii[mask]),
            "reliability": _summary_stats(reliabilities[mask]),
            "label": label,
        }

    return {
        "configuration": name,
        "granulation": {
            "purity_threshold": purity_threshold,
            "min_ball_count": min_ball_count,
            "epsilon": epsilon,
            "gamma": gamma,
            "lambda_r": lambda_r,
            "radius_scale": radius_scale,
        },
        "n_balls": len(balls),
        "purity": _summary_stats(purities),
        "radius": _summary_stats(radii),
        "radius_normalized": _summary_stats(r_tilde),
        "reliability": _summary_stats(reliabilities),
        "small_ball_counts": {
            "singletons_n_eq_1": int(singleton_mask.sum()),
            "n_le_2": int(le2_mask.sum()),
            "n_le_5": int(le5_mask.sum()),
        },
        "small_ball_analysis": {
            "singletons": subset_stats(singleton_mask, "n=1"),
            "n_le_2": subset_stats(le2_mask, "n<=2"),
            "n_le_5": subset_stats(le5_mask, "n<=5"),
        },
        "correlations": {
            "purity_vs_radius": _correlation(purities, radii),
            "purity_vs_reliability": _correlation(purities, reliabilities),
            "radius_vs_reliability": _correlation(radii, reliabilities),
            "ball_size_vs_purity": _correlation(sizes.astype(float), purities),
            "ball_size_vs_radius": _correlation(sizes.astype(float), radii),
            "ball_size_vs_reliability": _correlation(sizes.astype(float), reliabilities),
        },
        "per_ball": per_ball,
        "class_counts": {
            "pos_balls": int(np.sum(labels == 1)),
            "neg_balls": int(np.sum(labels != 1)),
        },
    }


def main() -> None:
    configs = []

    X_bal, y_bal = make_balanced(200, SEED)
    configs.append(
        analyze_configuration(
            "balanced_n200_p0.95_min2",
            X_bal,
            y_bal,
            purity_threshold=0.95,
            min_ball_count=2,
        )
    )
    configs.append(
        analyze_configuration(
            "balanced_n200_p0.95_min0_singletons_visible",
            X_bal,
            y_bal,
            purity_threshold=0.95,
            min_ball_count=0,
        )
    )

    X_imb, y_imb = make_imbalanced(50, 150, SEED)
    configs.append(
        analyze_configuration(
            "imbalanced_1to3_n200_p0.95_min2",
            X_imb,
            y_imb,
            purity_threshold=0.95,
            min_ball_count=2,
        )
    )
    configs.append(
        analyze_configuration(
            "imbalanced_1to3_n200_p0.85_min2",
            X_imb,
            y_imb,
            purity_threshold=0.85,
            min_ball_count=2,
        )
    )

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "analyze_ball_reliability.json"
    summary = [
        {k: v for k, v in cfg.items() if k != "per_ball"} for cfg in configs
    ]
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = out_dir / "analyze_ball_reliability_per_ball.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "configuration",
                "ball_id",
                "class",
                "num_samples",
                "purity",
                "radius",
                "radius_normalized",
                "reliability",
            ],
        )
        writer.writeheader()
        for cfg in configs:
            for row in cfg["per_ball"]:
                writer.writerow({"configuration": cfg["configuration"], **row})

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}\n")

    for cfg in summary:
        print(f"=== {cfg['configuration']} ===")
        print(f"  balls: {cfg['n_balls']} (pos={cfg['class_counts']['pos_balls']}, "
              f"neg={cfg['class_counts']['neg_balls']})")
        print(
            f"  purity: min={cfg['purity']['min']:.3f}, "
            f"mean={cfg['purity']['mean']:.3f}, max={cfg['purity']['max']:.3f}"
        )
        print(
            f"  radius: min={cfg['radius']['min']:.3f}, "
            f"median~q50={cfg['radius']['q50']:.3f}, max={cfg['radius']['max']:.3f}"
        )
        print(
            f"  reliability: min={cfg['reliability']['min']:.3f}, "
            f"mean={cfg['reliability']['mean']:.3f}, max={cfg['reliability']['max']:.3f}"
        )
        sb = cfg["small_ball_counts"]
        print(
            f"  small balls: singletons={sb['singletons_n_eq_1']}, "
            f"n<=2={sb['n_le_2']}, n<=5={sb['n_le_5']}"
        )
        if sb["singletons_n_eq_1"] > 0:
            s = cfg["small_ball_analysis"]["singletons"]
            print(
                f"  singleton reliability: mean={s['reliability']['mean']:.3f}, "
                f"max={s['reliability']['max']:.3f}"
            )
        print()


if __name__ == "__main__":
    main()
