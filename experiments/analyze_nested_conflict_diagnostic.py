"""
Mechanism 2 diagnostic: heterogeneous nested-ball conflicts (read-only).

After frozen recursive KMeans(k=2) granulation completes and BEFORE del_ball,
inspect all ball pairs:

  distance = ||center_i - center_j||
  NESTED if distance <= |radius_i - radius_j|
  HETEROGENEOUS_CONFLICT if NESTED and label_i != label_j

Does NOT modify granulation, del_ball, Topo, or GBLSTSVM.

Run from GBLSTSVM root:
  python experiments/analyze_nested_conflict_diagnostic.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
from src.cbcm import resolve_k  # noqa: E402
from src.granular_balls import BallInfo, GBList  # noqa: E402
from src.topology_membership import compute_topology_membership  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granulation"
PAIR_CSV_PATH = OUT_DIR / "nested_conflict_pairs.csv"
SUMMARY_JSON = OUT_DIR / "nested_conflict_diagnostic_summary.json"
REPORT_PATH = OUT_DIR / "nested_conflict_diagnostic_report.md"

SEEDS = [42, 0, 1]
PURITY = DEFAULT_PURITY
MIN_SAMPLES_PER_BALL = 1
MIN_BALL_COUNT = DEFAULT_MIN_BALL_COUNT

# Concentration thresholds (diagnostic only; not tuned).
SMALL_BALL_MAX = 5
IMPURE_EPS = 1e-12

# Sparse Topo failure cases from Session 3 broad screen (seed-specific).
KNOWN_TOPO_FAILURE_RUNS = {
    ("ecoli2", 42),
    ("ecoli2", 0),
    ("ecoli2", 1),
    ("vehicle0", 1),
}


@dataclass
class BallRecord:
    ball_id: int
    center: np.ndarray
    radius: float
    label: float
    purity: float
    num_samples: int
    would_del_ball: bool
    is_minority_class: bool
    is_small: bool
    is_impure: bool


def _build_indexed_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = normalize_labels(y)
    data = np.column_stack([X, y])
    num = data.shape[0]
    index = np.arange(num).reshape(num, 1)
    return np.hstack([data, index])


def generate_balls_before_del_ball(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    purity_threshold: float,
    min_samples_per_ball: int,
    seed: int,
) -> list[BallRecord]:
    """Replay granulation; stop before del_ball."""
    np.random.seed(seed)
    indexed = _build_indexed_matrix(X_train, y_train)
    gb = GBList(indexed)
    gb.init_granular_balls(
        purity=purity_threshold, min_sample=min_samples_per_ball
    )

    y_train_n = normalize_labels(y_train)
    n_pos = int(np.sum(y_train_n == 1.0))
    n_neg = int(np.sum(y_train_n == -1.0))
    minority_label = 1.0 if n_pos <= n_neg else -1.0

    records: list[BallRecord] = []
    for ball_id, ball in enumerate(gb.granular_balls):
        label = -1.0 if ball.label == -1 else 1.0
        records.append(
            BallRecord(
                ball_id=ball_id,
                center=ball.center.copy(),
                radius=float(ball.r),
                label=label,
                purity=float(ball.purity),
                num_samples=int(ball.num),
                would_del_ball=int(ball.num) < MIN_BALL_COUNT,
                is_minority_class=label == minority_label,
                is_small=int(ball.num) <= SMALL_BALL_MAX,
                is_impure=float(ball.purity) < 1.0 - IMPURE_EPS,
            )
        )
    return records


def _ball_info_list(records: list[BallRecord]) -> list[BallInfo]:
    return [
        BallInfo(
            center=r.center,
            radius=r.radius,
            label=r.label,
            purity=r.purity,
            num_samples=r.num_samples,
        )
        for r in records
    ]


def inspect_pairs(
    records: list[BallRecord],
    *,
    dataset: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(records)
    for i in range(n):
        bi = records[i]
        for j in range(i + 1, n):
            bj = records[j]
            distance = float(np.linalg.norm(bi.center - bj.center))
            radius_diff = abs(bi.radius - bj.radius)
            nested = distance <= radius_diff
            same_label = bi.label == bj.label
            homogeneous_nested = nested and same_label
            heterogeneous_conflict = nested and not same_label

            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "ball_i": bi.ball_id,
                    "ball_j": bj.ball_id,
                    "center_distance": distance,
                    "radius_diff": radius_diff,
                    "nested": nested,
                    "homogeneous_nested": homogeneous_nested,
                    "heterogeneous_conflict": heterogeneous_conflict,
                    "label_i": bi.label,
                    "label_j": bj.label,
                    "num_samples_i": bi.num_samples,
                    "num_samples_j": bj.num_samples,
                    "purity_i": bi.purity,
                    "purity_j": bj.purity,
                    "radius_i": bi.radius,
                    "radius_j": bj.radius,
                    "would_del_ball_i": bi.would_del_ball,
                    "would_del_ball_j": bj.would_del_ball,
                    "either_would_del_ball": bi.would_del_ball or bj.would_del_ball,
                    "both_would_del_ball": bi.would_del_ball and bj.would_del_ball,
                    "either_small_ball": bi.is_small or bj.is_small,
                    "both_small_balls": bi.is_small and bj.is_small,
                    "either_impure": bi.is_impure or bj.is_impure,
                    "both_impure": bi.is_impure and bj.is_impure,
                    "either_minority_class": bi.is_minority_class
                    or bj.is_minority_class,
                    "both_minority_class": bi.is_minority_class
                    and bj.is_minority_class,
                }
            )
    return rows


def summarize_run(
    records: list[BallRecord],
    pair_rows: list[dict[str, Any]],
    *,
    dataset: str,
    seed: int,
    minority_label: float,
) -> dict[str, Any]:
    n_balls = len(records)
    nested_pairs = [r for r in pair_rows if r["nested"]]
    hom_nested = [r for r in pair_rows if r["homogeneous_nested"]]
    het_conflicts = [r for r in pair_rows if r["heterogeneous_conflict"]]

    involved: set[int] = set()
    for row in het_conflicts:
        involved.add(int(row["ball_i"]))
        involved.add(int(row["ball_j"]))

    pct_involved = (len(involved) / n_balls * 100.0) if n_balls else 0.0

    conflict_details = []
    for row in het_conflicts:
        conflict_details.append(
            {
                "ball_i": row["ball_i"],
                "ball_j": row["ball_j"],
                "label_i": row["label_i"],
                "label_j": row["label_j"],
                "num_samples_i": row["num_samples_i"],
                "num_samples_j": row["num_samples_j"],
                "purity_i": row["purity_i"],
                "purity_j": row["purity_j"],
                "radius_i": row["radius_i"],
                "radius_j": row["radius_j"],
                "center_distance": row["center_distance"],
                "either_would_del_ball": row["either_would_del_ball"],
            }
        )

    k = resolve_k(n_balls) if n_balls > 0 else 0
    sparse_graph = n_balls > 0 and n_balls <= max(k, 15)
    known_topo_failure = (dataset, seed) in KNOWN_TOPO_FAILURE_RUNS

    topo_stats: dict[str, Any] = {}
    if n_balls > 1:
        balls = _ball_info_list(records)
        centers = np.vstack([b.center for b in balls])
        labels = np.array([b.label for b in balls], dtype=float)
        topo = compute_topology_membership(centers, labels, k=k)
        topo_stats = {
            "k": int(topo.k),
            "mean_n_opp": float(np.mean(topo.n_opp)),
            "frac_r_opp_zero": float(np.mean(topo.r_opp == 0.0)),
            "mean_r_opp": float(np.mean(topo.r_opp)),
        }
        if involved:
            idx = sorted(involved)
            topo_stats["conflict_balls_mean_n_opp"] = float(
                np.mean(topo.n_opp[idx])
            )
            topo_stats["conflict_balls_frac_r_opp_zero"] = float(
                np.mean(topo.r_opp[idx] == 0.0)
            )
            non_involved = [i for i in range(n_balls) if i not in involved]
            if non_involved:
                topo_stats["other_balls_mean_n_opp"] = float(
                    np.mean(topo.n_opp[non_involved])
                )
                topo_stats["other_balls_frac_r_opp_zero"] = float(
                    np.mean(topo.r_opp[non_involved] == 0.0)
                )

    def frac(condition_key: str, rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return float(np.mean([bool(r[condition_key]) for r in rows]))

    concentration = {
        "heterogeneous_conflicts": len(het_conflicts),
        "frac_either_small_ball": frac("either_small_ball", het_conflicts),
        "frac_both_small_balls": frac("both_small_balls", het_conflicts),
        "frac_either_impure": frac("either_impure", het_conflicts),
        "frac_both_impure": frac("both_impure", het_conflicts),
        "frac_either_minority_class": frac("either_minority_class", het_conflicts),
        "frac_both_minority_class": frac("both_minority_class", het_conflicts),
        "frac_either_would_del_ball": frac("either_would_del_ball", het_conflicts),
        "frac_both_would_del_ball": frac("both_would_del_ball", het_conflicts),
        "balls_small_total": int(sum(r.is_small for r in records)),
        "balls_impure_total": int(sum(r.is_impure for r in records)),
        "balls_minority_total": int(sum(r.is_minority_class for r in records)),
        "balls_would_del_ball_total": int(sum(r.would_del_ball for r in records)),
    }

    return {
        "dataset": dataset,
        "seed": seed,
        "minority_label": minority_label,
        "total_balls": n_balls,
        "total_pairs": len(pair_rows),
        "total_nested_pairs": len(nested_pairs),
        "homogeneous_nested_pairs": len(hom_nested),
        "heterogeneous_nested_conflicts": len(het_conflicts),
        "balls_in_heterogeneous_conflict": len(involved),
        "pct_balls_in_heterogeneous_conflict": pct_involved,
        "heterogeneous_conflict_details": conflict_details,
        "sparse_topo_graph": sparse_graph,
        "known_topo_failure_run": known_topo_failure,
        "topo_graph_stats": topo_stats,
        "concentration": concentration,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    per_run: list[dict[str, Any]],
    concentration_aggregate: dict[str, Any],
) -> None:
    lines = [
        "# Nested-Ball Conflict Diagnostic (Mechanism 2)",
        "",
        "Read-only inspection after granulation, **before** `del_ball`.",
        f"Purity={PURITY}, min_samples_per_ball={MIN_SAMPLES_PER_BALL}, "
        f"min_ball_count={MIN_BALL_COUNT} (used only to flag would-remove).",
        "",
        "## Per-run summary",
        "",
        "| dataset | seed | balls | nested | hom nested | het conflict | "
        "% balls in conflict | del_ball touch | sparse graph | known Topo fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in per_run:
        c = run["concentration"]
        lines.append(
            f"| {run['dataset']} | {run['seed']} | {run['total_balls']} | "
            f"{run['total_nested_pairs']} | {run['homogeneous_nested_pairs']} | "
            f"{run['heterogeneous_nested_conflicts']} | "
            f"{run['pct_balls_in_heterogeneous_conflict']:.1f} | "
            f"{c['frac_either_would_del_ball']:.0%} of conflicts | "
            f"{run['sparse_topo_graph']} | {run['known_topo_failure_run']} |"
        )

    lines.extend(["", "## Heterogeneous conflict details (by run)", ""])
    for run in per_run:
        lines.append(f"### {run['dataset']} seed={run['seed']}")
        details = run["heterogeneous_conflict_details"]
        if not details:
            lines.append("- No heterogeneous nested conflicts.")
        else:
            for d in details:
                lines.append(
                    f"- balls ({d['ball_i']},{d['ball_j']}): "
                    f"labels ({d['label_i']:+.0f},{d['label_j']:+.0f}), "
                    f"sizes ({d['num_samples_i']},{d['num_samples_j']}), "
                    f"purities ({d['purity_i']:.3f},{d['purity_j']:.3f}), "
                    f"radii ({d['radius_i']:.3f},{d['radius_j']:.3f}), "
                    f"dist={d['center_distance']:.3f}, "
                    f"either_del_ball={d['either_would_del_ball']}"
                )
        lines.append("")

    lines.extend(["## Concentration analysis", ""])
    lines.append(
        "Fraction of heterogeneous conflict pairs involving the listed ball property:"
    )
    lines.append("")
    agg = concentration_aggregate
    lines.append(
        f"- Either small ball (n≤{SMALL_BALL_MAX}): "
        f"{agg['overall_frac_either_small_ball']:.1%} "
        f"(sparse-graph runs: {agg['sparse_frac_either_small_ball']:.1%}, "
        f"dense runs: {agg['dense_frac_either_small_ball']:.1%})"
    )
    lines.append(
        f"- Either impure (purity<1): {agg['overall_frac_either_impure']:.1%} "
        f"(sparse: {agg['sparse_frac_either_impure']:.1%}, "
        f"dense: {agg['dense_frac_either_impure']:.1%})"
    )
    lines.append(
        f"- Either minority-class ball: {agg['overall_frac_either_minority_class']:.1%} "
        f"(known Topo failure runs: {agg['topo_fail_frac_either_minority_class']:.1%}, "
        f"other runs: {agg['topo_ok_frac_either_minority_class']:.1%})"
    )
    lines.append(
        f"- Either would be removed by del_ball: "
        f"{agg['overall_frac_either_would_del_ball']:.1%}"
    )
    lines.extend(["", "### Sparse Topo failure runs", ""])
    for run in per_run:
        if not run["known_topo_failure_run"] and not run["sparse_topo_graph"]:
            continue
        t = run["topo_graph_stats"]
        lines.append(
            f"- **{run['dataset']} s{run['seed']}**: "
            f"balls={run['total_balls']}, k={t.get('k', 'n/a')}, "
            f"het_conflicts={run['heterogeneous_nested_conflicts']}, "
            f"mean_n_opp={t.get('mean_n_opp', float('nan')):.2f}, "
            f"frac_R_opp=0={t.get('frac_r_opp_zero', float('nan')):.2f}"
        )
        if "conflict_balls_mean_n_opp" in t:
            lines.append(
                f"  - conflict balls mean n_opp={t['conflict_balls_mean_n_opp']:.2f} "
                f"(others={t.get('other_balls_mean_n_opp', float('nan')):.2f})"
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Pair CSV: `{PAIR_CSV_PATH.name}`",
            f"- Summary JSON: `{SUMMARY_JSON.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_concentration(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    sparse_conflicts: list[dict[str, Any]] = []
    dense_conflicts: list[dict[str, Any]] = []
    topo_fail_conflicts: list[dict[str, Any]] = []
    topo_ok_conflicts: list[dict[str, Any]] = []

    for run in per_run:
        minority = run["minority_label"]
        for d in run["heterogeneous_conflict_details"]:
            either_minority = (
                d["label_i"] == minority or d["label_j"] == minority
            )
            entry = {
                "either_small_ball": (
                    d["num_samples_i"] <= SMALL_BALL_MAX
                    or d["num_samples_j"] <= SMALL_BALL_MAX
                ),
                "either_impure": (
                    d["purity_i"] < 1.0 - IMPURE_EPS
                    or d["purity_j"] < 1.0 - IMPURE_EPS
                ),
                "either_minority_class": either_minority,
                "either_would_del_ball": d["either_would_del_ball"],
            }
            if run["sparse_topo_graph"]:
                sparse_conflicts.append(entry)
            else:
                dense_conflicts.append(entry)
            if run["known_topo_failure_run"]:
                topo_fail_conflicts.append(entry)
            else:
                topo_ok_conflicts.append(entry)

    def mean_flag(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return float(np.mean([bool(r[key]) for r in rows]))

    all_entries = sparse_conflicts + dense_conflicts
    return {
        "overall_frac_either_small_ball": mean_flag(all_entries, "either_small_ball"),
        "overall_frac_either_impure": mean_flag(all_entries, "either_impure"),
        "overall_frac_either_minority_class": mean_flag(
            all_entries, "either_minority_class"
        ),
        "overall_frac_either_would_del_ball": mean_flag(
            all_entries, "either_would_del_ball"
        ),
        "sparse_frac_either_small_ball": mean_flag(
            sparse_conflicts, "either_small_ball"
        ),
        "dense_frac_either_small_ball": mean_flag(dense_conflicts, "either_small_ball"),
        "sparse_frac_either_impure": mean_flag(sparse_conflicts, "either_impure"),
        "dense_frac_either_impure": mean_flag(dense_conflicts, "either_impure"),
        "topo_fail_frac_either_minority_class": mean_flag(
            topo_fail_conflicts, "either_minority_class"
        ),
        "topo_ok_frac_either_minority_class": mean_flag(
            topo_ok_conflicts, "either_minority_class"
        ),
        "n_conflict_pairs_total": len(all_entries),
        "n_conflict_pairs_sparse_graph": len(sparse_conflicts),
        "n_conflict_pairs_dense_graph": len(dense_conflicts),
        "n_conflict_pairs_known_topo_failure_runs": len(topo_fail_conflicts),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pair_rows: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []

    for dataset_file in DATASETS:
        dataset_name = dataset_file.replace(".csv", "")
        for seed in SEEDS:
            split = load_dataset_split(dataset_name, seed=seed)
            records = generate_balls_before_del_ball(
                split.X_train,
                split.y_train,
                purity_threshold=PURITY,
                min_samples_per_ball=MIN_SAMPLES_PER_BALL,
                seed=seed,
            )
            pair_rows = inspect_pairs(records, dataset=dataset_name, seed=seed)
            summary = summarize_run(
                records,
                pair_rows,
                dataset=dataset_name,
                seed=seed,
                minority_label=split.minority_label,
            )
            per_run.append(summary)
            all_pair_rows.extend(pair_rows)

            print(
                f"{dataset_name} seed={seed}: balls={summary['total_balls']}, "
                f"nested={summary['total_nested_pairs']}, "
                f"hom={summary['homogeneous_nested_pairs']}, "
                f"het_conflict={summary['heterogeneous_nested_conflicts']}, "
                f"pct_involved={summary['pct_balls_in_heterogeneous_conflict']:.1f}%"
            )

    write_csv(PAIR_CSV_PATH, all_pair_rows)
    concentration = aggregate_concentration(per_run)
    payload = {
        "protocol": {
            "purity_threshold": PURITY,
            "min_samples_per_ball": MIN_SAMPLES_PER_BALL,
            "min_ball_count_for_del_flag": MIN_BALL_COUNT,
            "datasets": [d.replace(".csv", "") for d in DATASETS],
            "seeds": SEEDS,
            "nested_rule": "center_distance <= abs(radius_i - radius_j)",
            "heterogeneous_conflict_rule": "nested and label_i != label_j",
            "small_ball_max": SMALL_BALL_MAX,
            "known_topo_failure_runs": sorted(
                [f"{d}_s{s}" for d, s in KNOWN_TOPO_FAILURE_RUNS]
            ),
        },
        "concentration_aggregate": concentration,
        "per_run": per_run,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(REPORT_PATH, per_run, concentration)

    print(f"\nWrote {PAIR_CSV_PATH}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
