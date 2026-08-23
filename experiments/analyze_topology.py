"""
Topology membership diagnostics (run BEFORE classifier evaluation).

Reports A2, R_opp, and muTopo; compares synthetic noise balls vs ordinary
balls. Does not modify the formula or frozen baselines.

Run from GBLSTSVM root:
  python experiments/analyze_topology.py
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
from src.topology_membership import compute_topology_membership  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "topology"
REPORT_PATH = OUT_DIR / "topology_diagnostics_report.md"
JSON_PATH = OUT_DIR / "topology_diagnostics_report.json"

SEED = 42
PURITY = DEFAULT_PURITY
MIN_BALL_COUNT = DEFAULT_MIN_BALL_COUNT

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


def _synthetic_noise_topo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    balls,
    minority_label: float,
    rng: np.random.Generator,
) -> dict:
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
    balls_aug = generate_granular_balls(
        X_aug, y_aug, purity_threshold=PURITY, min_samples_per_ball=1, min_ball_count=MIN_BALL_COUNT
    )
    centers_aug = np.vstack([b.center for b in balls_aug])
    labels_aug = np.array([b.label for b in balls_aug], dtype=float)
    mem_aug = compute_topology_membership(centers_aug, labels_aug)

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

    a_noise = float(mem_aug.ambiguity[noise_ball_global])
    r_noise = float(mem_aug.r_opp[noise_ball_global])
    mu_noise = float(mem_aug.mu_topo[noise_ball_global])
    return {
        "m_noise_points": m,
        "method": method,
        "noise_ball_A": a_noise,
        "noise_ball_A_percentile": percentile_rank(a_noise, mem_aug.ambiguity),
        "noise_ball_R_opp": r_noise,
        "noise_ball_R_opp_percentile": percentile_rank(r_noise, mem_aug.r_opp),
        "noise_ball_muTopo": mu_noise,
        "noise_ball_muTopo_percentile": percentile_rank(mu_noise, mem_aug.mu_topo),
        "noise_ball_comp": float(mem_aug.comp[noise_ball_global]),
        "noise_ball_n_opp": int(mem_aug.n_opp[noise_ball_global]),
        "noise_ball_n_recip": int(mem_aug.n_reciprocal_opp[noise_ball_global]),
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

    membership = compute_topology_membership(centers, labels)
    logger.info("%s: N=%d, k=%d", ds, n_balls, membership.k)
    kappa_pos, kappa_neg, _, _, _ = compute_binary_class_weights(y_train)
    kappa_ball = kappa_per_ball_array(balls, kappa_pos, kappa_neg)

    minority_mask = labels == minority_label
    majority_mask = ~minority_mask

    rows = []
    for i, b in enumerate(balls):
        rows.append(
            {
                "dataset": ds,
                "ball_id": i,
                "class": int(b.label),
                "is_minority": bool(b.label == minority_label),
                "CW": float(kappa_ball[i]),
                "comp": float(membership.comp[i]),
                "A": float(membership.ambiguity[i]),
                "R_opp": float(membership.r_opp[i]),
                "n_opp": int(membership.n_opp[i]),
                "n_reciprocal_opp": int(membership.n_reciprocal_opp[i]),
                "muTopo": float(membership.mu_topo[i]),
            }
        )

    rng = np.random.default_rng(SEED)
    synthetic = _synthetic_noise_topo(X_train, y_train, balls, minority_label, rng)

    r_std = float(np.std(membership.r_opp))
    a_vs_r = safe_correlation(membership.ambiguity, membership.r_opp, "pearson")
    degenerate = (r_std == 0.0) or (
        np.isfinite(a_vs_r) and abs(a_vs_r) >= 0.99
    )

    return {
        "dataset": ds,
        "n_balls": n_balls,
        "k": membership.k,
        "per_ball_rows": rows,
        "distributions": {
            "comp": _summary(membership.comp),
            "A": _summary(membership.ambiguity),
            "R_opp": _summary(membership.r_opp),
            "muTopo": _summary(membership.mu_topo),
        },
        "correlations": {
            "A_vs_CW": {
                "pearson": safe_correlation(membership.ambiguity, kappa_ball, "pearson"),
                "spearman": safe_correlation(membership.ambiguity, kappa_ball, "spearman"),
            },
            "R_opp_vs_A": {
                "pearson": safe_correlation(membership.r_opp, membership.ambiguity, "pearson"),
                "spearman": safe_correlation(membership.r_opp, membership.ambiguity, "spearman"),
            },
            "muTopo_vs_CW": {
                "pearson": safe_correlation(membership.mu_topo, kappa_ball, "pearson"),
                "spearman": safe_correlation(membership.mu_topo, kappa_ball, "spearman"),
            },
        },
        "class_means": {
            "A": {
                "minority": float(np.mean(membership.ambiguity[minority_mask]))
                if minority_mask.any()
                else float("nan"),
                "majority": float(np.mean(membership.ambiguity[majority_mask]))
                if majority_mask.any()
                else float("nan"),
            },
            "R_opp": {
                "minority": float(np.mean(membership.r_opp[minority_mask]))
                if minority_mask.any()
                else float("nan"),
                "majority": float(np.mean(membership.r_opp[majority_mask]))
                if majority_mask.any()
                else float("nan"),
            },
            "muTopo": {
                "minority": float(np.mean(membership.mu_topo[minority_mask]))
                if minority_mask.any()
                else float("nan"),
                "majority": float(np.mean(membership.mu_topo[majority_mask]))
                if majority_mask.any()
                else float("nan"),
            },
        },
        "fractions": {
            "A_gt_0_5": float(np.mean(membership.ambiguity > 0.5)),
            "R_opp_gt_0": float(np.mean(membership.r_opp > 0)),
            "muTopo_gt_0_5": float(np.mean(membership.mu_topo > 0.5)),
        },
        "synthetic_noise": synthetic,
        "mechanism_degenerate": degenerate,
        "membership": membership,
        "minority_mask": minority_mask,
    }


def _plot(res: dict, out_dir: Path) -> None:
    ds = res["dataset"]
    m = res["membership"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].scatter(m.ambiguity, m.r_opp, c=res["minority_mask"], cmap="coolwarm", alpha=0.7, s=30)
    axes[0].set_xlabel("A")
    axes[0].set_ylabel("R_opp")
    axes[0].set_title(f"{ds}: A vs R_opp")

    axes[1].scatter(m.ambiguity, m.mu_topo, c=res["minority_mask"], cmap="coolwarm", alpha=0.7, s=30)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_xlabel("A")
    axes[1].set_ylabel("muTopo")
    axes[1].set_title(f"{ds}: A vs muTopo")

    axes[2].hist(m.ambiguity, bins=20, alpha=0.55, label="A")
    axes[2].hist(m.r_opp, bins=20, alpha=0.55, label="R_opp")
    axes[2].hist(m.mu_topo, bins=20, alpha=0.55, label="muTopo")
    axes[2].set_title(f"{ds}: distributions")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{ds}_topology_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(all_results: list[dict]) -> None:
    lines = [
        "# Topology Membership Diagnostic Report",
        "",
        "Formula frozen: muTopo = A2 * R_opp. No post-hoc retuning.",
        "",
        "## Configuration",
        f"- Granulation: purity={PURITY}, min_ball_count={MIN_BALL_COUNT}, seed={SEED}",
        "- k = max(5, ceil(0.05 * N))",
        "",
        "## Key question",
        "",
        "Does reciprocal topology reduce high membership on unsupported noisy balls "
        "without destroying high membership of useful ambiguous balls?",
        "",
    ]

    for res in all_results:
        ds = res["dataset"]
        lines.extend(
            [
                f"## {ds}",
                "",
                f"N balls={res['n_balls']}, k={res['k']}",
                f"degenerate={res['mechanism_degenerate']}",
                "",
                "### Distributions",
            ]
        )
        for name, s in res["distributions"].items():
            lines.append(f"- {name}: {s}")
        lines.append("")
        lines.append("### Correlations")
        for name, c in res["correlations"].items():
            lines.append(f"- {name}: Pearson={c['pearson']:.4f}, Spearman={c['spearman']:.4f}")
        lines.append("")
        lines.append("### Minority vs majority means")
        for name, means in res["class_means"].items():
            lines.append(
                f"- {name}: minority={means['minority']:.4f}, majority={means['majority']:.4f}"
            )
        lines.append("")
        lines.append("### Fractions")
        fr = res["fractions"]
        lines.append(f"- A > 0.5: {fr['A_gt_0_5']:.4f}")
        lines.append(f"- R_opp > 0: {fr['R_opp_gt_0']:.4f}")
        lines.append(f"- muTopo > 0.5: {fr['muTopo_gt_0_5']:.4f}")
        lines.append("")
        syn = res["synthetic_noise"]
        lines.extend(
            [
                "### Synthetic noise",
                f"- Method: {syn.get('method', 'n/a')}",
                f"- A: {syn.get('noise_ball_A', 'n/a')}, "
                f"percentile={syn.get('noise_ball_A_percentile', float('nan')):.1f}",
                f"- R_opp: {syn.get('noise_ball_R_opp', 'n/a')}, "
                f"percentile={syn.get('noise_ball_R_opp_percentile', float('nan')):.1f}",
                f"- muTopo: {syn.get('noise_ball_muTopo', 'n/a')}, "
                f"percentile={syn.get('noise_ball_muTopo_percentile', float('nan')):.1f}",
                f"- n_opp={syn.get('noise_ball_n_opp')}, n_recip={syn.get('noise_ball_n_recip')}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _payload(all_results: list[dict]) -> dict:
    return {
        "config": {
            "seed": SEED,
            "purity": PURITY,
            "min_ball_count": MIN_BALL_COUNT,
        },
        "sensible": (not any(r["mechanism_degenerate"] for r in all_results))
        and any(r["fractions"]["R_opp_gt_0"] > 0 for r in all_results),
        "datasets": {
            r["dataset"]: {
                k: r[k]
                for k in (
                    "n_balls",
                    "k",
                    "distributions",
                    "correlations",
                    "class_means",
                    "fractions",
                    "synthetic_noise",
                    "mechanism_degenerate",
                )
            }
            for r in all_results
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for ds_file in DATASETS:
        logger.info("Topology diagnostics: %s", ds_file)
        res = diagnose_dataset(ds_file)
        all_results.append(res)
        pd.DataFrame(res["per_ball_rows"]).to_csv(
            OUT_DIR / f"{res['dataset']}_ball_topology_diagnostics.csv", index=False
        )
        _plot(res, OUT_DIR)

    _write_markdown(all_results)
    payload = _payload(all_results)
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nSensible for cheap screen: {payload['sensible']}")
    print(f"Report: {REPORT_PATH}")
    print(f"JSON:   {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
