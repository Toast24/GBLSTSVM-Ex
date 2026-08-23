"""
IPA mandatory diagnostic report (run BEFORE classifier evaluation).

Reports per-ball ambiguity, isolation, and muIPA; compares against CA
ambiguity on synthetic noise balls. Does not modify frozen baselines.

Run from GBLSTSVM root:
  python experiments/analyze_ipa.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase8_common import (  # noqa: E402
    DATASETS,
    DEFAULT_MIN_BALL_COUNT,
    DEFAULT_PURITY,
    load_dataset_split,
    normalize_labels,
)
from src.cbcm import kappa_per_ball_array, percentile_rank, safe_correlation  # noqa: E402
from src.class_weights import compute_binary_class_weights  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.ipa import compute_ipa_membership  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "ipa"
REPORT_PATH = OUT_DIR / "ipa_diagnostics_report.md"
JSON_PATH = OUT_DIR / "ipa_diagnostics_report.json"

SEED = 42
PURITY = DEFAULT_PURITY
MIN_BALL_COUNT = DEFAULT_MIN_BALL_COUNT
SUBSTANTIAL_CHANGE_THRESHOLD = 0.1

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
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


def _synthetic_noise_ipa(
    X_train: np.ndarray,
    y_train: np.ndarray,
    balls,
    membership,
    minority_label: float,
    rng: np.random.Generator,
) -> dict:
    """Inject synthetic minority noise; report ambiguity and muIPA percentiles."""
    from src.granular_balls import generate_granular_balls as gen_balls

    n_balls = len(balls)
    m = max(3, int(np.ceil(0.02 * n_balls)))
    labels_ball = np.array([b.label for b in balls], dtype=float)
    centers = np.vstack([b.center for b in balls])
    minority_idx = np.flatnonzero(labels_ball == minority_label)

    noise_centers: list[np.ndarray] = []
    method = "farthest_minority_balls"
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
    else:
        method = "synthetic_majority_region"

    if len(noise_centers) < m:
        maj_idx = np.flatnonzero(labels_ball != minority_label)
        if maj_idx.size == 0:
            return {"error": "no majority balls for synthetic placement"}
        pick = rng.choice(maj_idx, size=m - len(noise_centers), replace=True)
        jitter = 0.01 * np.std(centers, axis=0, keepdims=True) + 1e-6
        for j in pick:
            noise_centers.append(
                centers[j] + rng.normal(0.0, 1.0, size=centers.shape[1]) * jitter.ravel()
            )

    X_aug = np.vstack([X_train, np.vstack(noise_centers[:m])])
    y_aug = np.concatenate([y_train, np.full(m, minority_label, dtype=float)])
    balls_aug = gen_balls(
        X_aug, y_aug, purity_threshold=PURITY, min_samples_per_ball=1, min_ball_count=MIN_BALL_COUNT
    )
    centers_aug = np.vstack([b.center for b in balls_aug])
    labels_aug = np.array([b.label for b in balls_aug], dtype=float)
    mem_aug = compute_ipa_membership(centers_aug, labels_aug)

    noise_arr = np.vstack(noise_centers[:m])
    minority_ball_idx = [i for i, b in enumerate(balls_aug) if b.label == minority_label]
    if not minority_ball_idx:
        return {"error": "no minority balls after augmentation"}

    dist_to_noise = np.min(
        np.linalg.norm(
            centers_aug[minority_ball_idx][:, None, :] - noise_arr[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    noise_ball_global = minority_ball_idx[int(np.argmin(dist_to_noise))]

    amb_noise = float(mem_aug.ambiguity[noise_ball_global])
    ipa_noise = float(mem_aug.mu_ipa[noise_ball_global])
    return {
        "m_noise_points": m,
        "method": method,
        "noise_ball_ambiguity": amb_noise,
        "noise_ball_ambiguity_percentile": percentile_rank(amb_noise, mem_aug.ambiguity),
        "noise_ball_mu_ipa": ipa_noise,
        "noise_ball_mu_ipa_percentile": percentile_rank(ipa_noise, mem_aug.mu_ipa),
        "noise_ball_raw_isolation": float(mem_aug.raw_isolation[noise_ball_global]),
        "noise_ball_nearest_same": float(mem_aug.nearest_same_dist[noise_ball_global])
        if np.isfinite(mem_aug.nearest_same_dist[noise_ball_global])
        else None,
        "noise_ball_comp": float(mem_aug.comp_a[noise_ball_global]),
        "ambiguity_reduction": amb_noise - ipa_noise,
    }


def diagnose_dataset(dataset_file: str) -> dict:
    split = load_dataset_split(dataset_file, seed=SEED)
    ds = split.name
    X_train = split.X_train
    y_train = normalize_labels(split.y_train)
    minority_label = split.minority_label

    np.random.seed(SEED)
    balls = generate_granular_balls(
        X_train,
        y_train,
        purity_threshold=PURITY,
        min_samples_per_ball=1,
        min_ball_count=MIN_BALL_COUNT,
    )
    centers = np.vstack([b.center for b in balls])
    labels = np.array([b.label for b in balls], dtype=float)
    n_balls = len(balls)

    membership = compute_ipa_membership(centers, labels)
    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    kappa_ball = kappa_per_ball_array(balls, kappa_pos, kappa_neg)

    minority_mask = labels == minority_label
    majority_mask = ~minority_mask

    delta = np.abs(membership.mu_ipa - membership.ambiguity)
    substantial_mask = delta > SUBSTANTIAL_CHANGE_THRESHOLD

    rows = []
    for i, b in enumerate(balls):
        rows.append(
            {
                "dataset": ds,
                "ball_id": i,
                "class": int(b.label),
                "is_minority": bool(b.label == minority_label),
                "CW": float(kappa_ball[i]),
                "comp": float(membership.comp_a[i]),
                "ambiguity": float(membership.ambiguity[i]),
                "nearest_same_distance": float(membership.nearest_same_dist[i])
                if np.isfinite(membership.nearest_same_dist[i])
                else None,
                "isolation": float(membership.isolation[i]),
                "raw_isolation": float(membership.raw_isolation[i]),
                "muIPA": float(membership.mu_ipa[i]),
                "delta_from_ambiguity": float(delta[i]),
                "substantially_penalized": bool(substantial_mask[i]),
            }
        )

    rng = np.random.default_rng(SEED)
    synthetic = _synthetic_noise_ipa(
        X_train, y_train, balls, membership, minority_label, rng
    )

    return {
        "dataset": ds,
        "n_balls": n_balls,
        "k": membership.k,
        "lam": membership.lam,
        "per_ball_rows": rows,
        "correlations": {
            "ambiguity_vs_isolation": {
                "pearson": safe_correlation(membership.ambiguity, membership.isolation, "pearson"),
                "spearman": safe_correlation(membership.ambiguity, membership.isolation, "spearman"),
            },
            "muIPA_vs_CW": {
                "pearson": safe_correlation(membership.mu_ipa, kappa_ball, "pearson"),
                "spearman": safe_correlation(membership.mu_ipa, kappa_ball, "spearman"),
            },
            "ambiguity_vs_raw_isolation": {
                "pearson": safe_correlation(membership.ambiguity, membership.raw_isolation, "pearson"),
                "spearman": safe_correlation(membership.ambiguity, membership.raw_isolation, "spearman"),
            },
        },
        "muIPA_distributions": {
            "minority": _summary(membership.mu_ipa[minority_mask]),
            "majority": _summary(membership.mu_ipa[majority_mask]),
        },
        "ambiguity_distributions": {
            "minority": _summary(membership.ambiguity[minority_mask]),
            "majority": _summary(membership.ambiguity[majority_mask]),
        },
        "frac_muIPA_gt_0_5": float(np.mean(membership.mu_ipa > 0.5)),
        "n_substantially_penalized": int(np.sum(substantial_mask)),
        "frac_substantially_penalized": float(np.mean(substantial_mask)),
        "frac_substantial_minority": float(np.mean(substantial_mask & minority_mask))
        if minority_mask.any()
        else float("nan"),
        "synthetic_noise": synthetic,
        "membership": membership,
        "minority_mask": minority_mask,
    }


def _plot_comparison(res: dict, out_dir: Path) -> str:
    ds = res["dataset"]
    m = res["membership"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(m.ambiguity, m.mu_ipa, c=res["minority_mask"], cmap="coolwarm", alpha=0.7, s=30)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="y=x")
    axes[0].set_xlabel("ambiguity")
    axes[0].set_ylabel("muIPA")
    axes[0].set_title(f"{ds}: ambiguity vs muIPA")
    axes[0].legend()

    axes[1].hist(m.ambiguity, bins=20, alpha=0.6, label="ambiguity", color="steelblue")
    axes[1].hist(m.mu_ipa, bins=20, alpha=0.6, label="muIPA", color="coral")
    axes[1].set_xlabel("membership")
    axes[1].set_title(f"{ds}: distributions")
    axes[1].legend()
    fig.tight_layout()
    path = out_dir / f"{ds}_ipa_diagnostics.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _write_markdown(all_results: list[dict]) -> None:
    lines = [
        "# IPA Diagnostic Report",
        "",
        "## Configuration",
        f"- Granulation: purity={PURITY}, min_ball_count={MIN_BALL_COUNT}, seed={SEED}",
        f"- lambda={all_results[0]['lam'] if all_results else 1.0} (fixed)",
        f"- Substantial change threshold: |muIPA - ambiguity| > {SUBSTANTIAL_CHANGE_THRESHOLD}",
        "",
    ]

    for res in all_results:
        ds = res["dataset"]
        lines.extend([f"## {ds}", "", f"N balls={res['n_balls']}, k={res['k']}", ""])
        lines.append("### Correlations")
        for name, c in res["correlations"].items():
            lines.append(f"- {name}: Pearson={c['pearson']:.4f}, Spearman={c['spearman']:.4f}")
        lines.append("")
        lines.append("### muIPA distributions")
        for cls, s in res["muIPA_distributions"].items():
            lines.append(f"- {cls}: {s}")
        lines.append(f"- frac(muIPA > 0.5): {res['frac_muIPA_gt_0_5']:.4f}")
        lines.append("")
        lines.append("### Penalization summary")
        lines.append(f"- n substantially penalized: {res['n_substantially_penalized']}")
        lines.append(f"- frac substantially penalized: {res['frac_substantially_penalized']:.4f}")
        lines.append(f"- frac substantial among minority: {res['frac_substantial_minority']:.4f}")
        lines.append("")
        syn = res["synthetic_noise"]
        lines.extend(
            [
                "### Synthetic noise (ambiguity vs IPA)",
                f"- Method: {syn.get('method', 'n/a')}",
                f"- noise_ball ambiguity: {syn.get('noise_ball_ambiguity', 'n/a')}, "
                f"percentile={syn.get('noise_ball_ambiguity_percentile', float('nan')):.1f}",
                f"- noise_ball muIPA: {syn.get('noise_ball_mu_ipa', 'n/a')}, "
                f"percentile={syn.get('noise_ball_mu_ipa_percentile', float('nan')):.1f}",
                f"- ambiguity reduction on noise ball: {syn.get('ambiguity_reduction', float('nan')):.4f}",
                f"- raw_isolation: {syn.get('noise_ball_raw_isolation', 'n/a')}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for ds_file in DATASETS:
        logger.info("IPA diagnostics: %s", ds_file)
        res = diagnose_dataset(ds_file)
        all_results.append(res)
        pd.DataFrame(res["per_ball_rows"]).to_csv(
            OUT_DIR / f"{res['dataset']}_ball_ipa_diagnostics.csv", index=False
        )
        _plot_comparison(res, OUT_DIR)

    _write_markdown(all_results)

    payload = {
        "config": {
            "seed": SEED,
            "purity": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
            "lambda": all_results[0]["lam"] if all_results else 1.0,
            "substantial_change_threshold": SUBSTANTIAL_CHANGE_THRESHOLD,
        },
        "datasets": {
            r["dataset"]: {
                k: r[k]
                for k in (
                    "n_balls",
                    "k",
                    "lam",
                    "correlations",
                    "muIPA_distributions",
                    "ambiguity_distributions",
                    "frac_muIPA_gt_0_5",
                    "n_substantially_penalized",
                    "frac_substantially_penalized",
                    "frac_substantial_minority",
                    "synthetic_noise",
                )
            }
            for r in all_results
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nReport: {REPORT_PATH}")
    print(f"JSON:   {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
