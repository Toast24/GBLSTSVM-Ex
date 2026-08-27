"""
Mechanism 1 diagnostic: hd2nd singleton classification (read-only).

Replays the frozen recursive KMeans(k=2) granulation loop and, for every
split that produces a singleton child, computes within the PARENT ball only:

  hd1 = nearest opposite-class distance
  hd2 = second-nearest opposite-class distance (strictly > hd1)
  S   = same-class samples with d(x, z) <= hd2

Classification:
  OUTLIER     if |S| == 1
  LEGITIMATE  if |S| > 1

Does NOT modify granulation, del_ball, Topo, or GBLSTSVM.

Run from GBLSTSVM root:
  python experiments/analyze_hd2_singleton_diagnostic.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
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
from src.granular_balls import GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "granulation"
CSV_PATH = OUT_DIR / "hd2_singleton_diagnostic.csv"
SUMMARY_JSON = OUT_DIR / "hd2_singleton_diagnostic_summary.json"
REPORT_PATH = OUT_DIR / "hd2_singleton_diagnostic_report.md"

SEEDS = [42, 0, 1]
PURITY = DEFAULT_PURITY
MIN_SAMPLES_PER_BALL = 1
MIN_BALL_COUNT = DEFAULT_MIN_BALL_COUNT  # recorded only; not applied in trace


def _build_indexed_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = normalize_labels(y)
    data = np.column_stack([X, y])
    num = data.shape[0]
    index = np.arange(num).reshape(num, 1)
    return np.hstack([data, index])


def compute_hd1_hd2(
    x: np.ndarray,
    y_x: float,
    parent_features: np.ndarray,
    parent_labels: np.ndarray,
) -> tuple[float | None, float | None, str | None]:
    """
    Return (hd1, hd2, edge_case).

    edge_case is None when hd2 is defined; otherwise a short code string.
    """
    opp_mask = parent_labels != y_x
    if not np.any(opp_mask):
        return None, None, "no_opposite_class_in_parent"

    opp_dists = np.linalg.norm(parent_features[opp_mask] - x, axis=1)
    hd1 = float(np.min(opp_dists))

    # Strictly greater than hd1 (handles duplicate distances at hd1).
    greater = opp_dists[opp_dists > hd1]
    if greater.size == 0:
        return hd1, None, "no_second_heterogeneous"

    hd2 = float(np.min(greater))
    return hd1, hd2, None


def compute_s_size(
    x: np.ndarray,
    y_x: float,
    hd2: float,
    parent_features: np.ndarray,
    parent_labels: np.ndarray,
) -> int:
    same_mask = parent_labels == y_x
    same_dists = np.linalg.norm(parent_features[same_mask] - x, axis=1)
    return int(np.sum(same_dists <= hd2))


def classify_singleton(
    s_size: int,
    *,
    hd2_defined: bool,
) -> str:
    if not hd2_defined:
        return "UNDEFINED_HD2"
    if s_size == 1:
        return "OUTLIER"
    return "LEGITIMATE"


def collect_singleton_split_events(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    purity_threshold: float,
    min_samples_per_ball: int,
    seed: int,
    dataset: str,
) -> list[dict[str, Any]]:
    """Replay init_granular_balls and record singleton children at each split."""
    np.random.seed(seed)
    indexed = _build_indexed_matrix(X_train, y_train)
    balls: list[GranularBall] = [GranularBall(indexed)]
    events: list[dict[str, Any]] = []
    split_id = 0

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if ball.purity < purity_threshold and ball.num > min_samples_per_ball:
            parent_features = ball.data_no_label.copy()
            parent_labels = ball.data[:, -2].copy()
            parent_indices = ball.data[:, -1].astype(int).copy()
            parent_num = int(ball.num)
            parent_purity = float(ball.purity)
            parent_label = float(ball.label)

            split_balls = ball.split_2balls()
            split_id += 1

            for child_slot, child in enumerate(("first", "second")):
                child_ball = split_balls[0] if child_slot == 0 else split_balls[1]
                if child_ball.num != 1:
                    continue

                x = child_ball.data_no_label[0].copy()
                y_x = float(child_ball.data[0, -2])
                sample_index = int(child_ball.data[0, -1])

                edge_flags: list[str] = []
                if parent_num == 1:
                    edge_flags.append("singleton_parent")
                if parent_num == 2:
                    edge_flags.append("two_sample_parent")

                hd1, hd2, hd_edge = compute_hd1_hd2(
                    x, y_x, parent_features, parent_labels
                )
                if hd_edge is not None:
                    edge_flags.append(hd_edge)

                if hd1 is not None and parent_features.shape[0] > 1:
                    opp_mask = parent_labels != y_x
                    opp_dists = np.linalg.norm(parent_features[opp_mask] - x, axis=1)
                    n_at_hd1 = int(np.sum(np.isclose(opp_dists, hd1)))
                    if n_at_hd1 > 1:
                        edge_flags.append("duplicate_hd1_distance")

                hd2_defined = hd2 is not None
                s_size = (
                    compute_s_size(x, y_x, hd2, parent_features, parent_labels)
                    if hd2_defined
                    else 0
                )
                classification = classify_singleton(s_size, hd2_defined=hd2_defined)

                events.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "split_id": split_id,
                        "parent_index_in_queue": i,
                        "child_slot": child,
                        "parent_num_samples": parent_num,
                        "parent_purity": parent_purity,
                        "parent_majority_label": parent_label,
                        "singleton_sample_index": sample_index,
                        "singleton_label": y_x,
                        "hd1": hd1,
                        "hd2": hd2,
                        "s_size": s_size,
                        "classification": classification,
                        "edge_flags": "|".join(edge_flags) if edge_flags else "",
                        "parent_sample_indices": ",".join(map(str, parent_indices)),
                    }
                )

            balls[i] = split_balls[0]
            balls.append(split_balls[1])
            ll += 1
        else:
            i += 1

    return events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    n_singleton = len(events)
    outliers = [e for e in events if e["classification"] == "OUTLIER"]
    legitimate = [e for e in events if e["classification"] == "LEGITIMATE"]
    undefined = [e for e in events if e["classification"] == "UNDEFINED_HD2"]

    def sample_key(row: dict[str, Any]) -> tuple[str, int, int]:
        return (row["dataset"], int(row["seed"]), int(row["singleton_sample_index"]))

    affected_keys = {sample_key(e) for e in events}
    outlier_keys = {sample_key(e) for e in outliers}
    legit_keys = {sample_key(e) for e in legitimate}

    def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        c = Counter()
        for row in rows:
            label = "+1" if row["singleton_label"] == 1.0 else "-1"
            c[label] += 1
        return dict(sorted(c.items()))

    edge_case_counts = Counter()
    for e in events:
        if e["edge_flags"]:
            for flag in e["edge_flags"].split("|"):
                if flag:
                    edge_case_counts[flag] += 1

    return {
        "n_singleton_children": n_singleton,
        "n_predicted_outliers": len(outliers),
        "n_legitimate_singletons": len(legitimate),
        "n_undefined_hd2": len(undefined),
        "n_samples_affected": len(affected_keys),
        "n_unique_outlier_samples": len(outlier_keys),
        "n_unique_legitimate_samples": len(legit_keys),
        "class_distribution_all": class_counts(events),
        "class_distribution_outliers": class_counts(outliers),
        "class_distribution_legitimate": class_counts(legitimate),
        "edge_case_counts": dict(sorted(edge_case_counts.items())),
    }


def verify_edge_cases(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observed edge-case rows for the report."""
    no_second = [e for e in events if "no_second_heterogeneous" in e["edge_flags"]]
    dup_hd1 = [e for e in events if "duplicate_hd1_distance" in e["edge_flags"]]
    singleton_parent = [e for e in events if "singleton_parent" in e["edge_flags"]]
    two_sample = [e for e in events if "two_sample_parent" in e["edge_flags"]]

    return {
        "no_second_heterogeneous": {
            "count": len(no_second),
            "all_undefined_hd2": all(
                e["classification"] == "UNDEFINED_HD2" for e in no_second
            ),
            "example": no_second[0] if no_second else None,
        },
        "duplicate_hd1_distance": {
            "count": len(dup_hd1),
            "example": dup_hd1[0] if dup_hd1 else None,
        },
        "singleton_parent": {
            "count": len(singleton_parent),
            "example": singleton_parent[0] if singleton_parent else None,
        },
        "two_sample_parent": {
            "count": len(two_sample),
            "example": two_sample[0] if two_sample else None,
        },
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
    all_events: list[dict[str, Any]],
    per_run: list[dict[str, Any]],
    edge_verification: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# hd2 Singleton Diagnostic (Mechanism 1)",
        "",
        "Read-only replay of frozen recursive KMeans(k=2) splits.",
        f"Purity={PURITY}, min_samples_per_ball={MIN_SAMPLES_PER_BALL}.",
        "`del_ball` is **not** applied in this trace.",
        "",
        "## Per-run summary",
        "",
        "| dataset | seed | singletons | outliers | legitimate | undefined hd2 | samples affected |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in per_run:
        s = run["summary"]
        lines.append(
            f"| {run['dataset']} | {run['seed']} | "
            f"{s['n_singleton_children']} | {s['n_predicted_outliers']} | "
            f"{s['n_legitimate_singletons']} | {s['n_undefined_hd2']} | "
            f"{s['n_samples_affected']} |"
        )

    lines.extend(["", "## Class distribution (all runs aggregated)", ""])
    agg = summarize_events(all_events)
    lines.append(f"- All singleton children: {agg['class_distribution_all']}")
    lines.append(f"- Predicted outliers: {agg['class_distribution_outliers']}")
    lines.append(f"- Legitimate singletons: {agg['class_distribution_legitimate']}")

    lines.extend(["", "## Edge-case verification", ""])
    for name, info in edge_verification.items():
        lines.append(f"### {name}")
        lines.append(f"- Count: {info['count']}")
        if "all_undefined_hd2" in info:
            lines.append(f"- All classified UNDEFINED_HD2: {info['all_undefined_hd2']}")
        if info.get("example"):
            ex = info["example"]
            lines.append(
                f"- Example: dataset={ex['dataset']}, seed={ex['seed']}, "
                f"sample_index={ex['singleton_sample_index']}, "
                f"parent_num={ex['parent_num_samples']}, flags={ex['edge_flags']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Outputs",
            "",
            f"- CSV: `{CSV_PATH.name}`",
            f"- Summary JSON: `{SUMMARY_JSON.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_events: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []

    for dataset_file in DATASETS:
        dataset_name = dataset_file.replace(".csv", "")
        for seed in SEEDS:
            split = load_dataset_split(dataset_name, seed=seed)
            events = collect_singleton_split_events(
                split.X_train,
                split.y_train,
                purity_threshold=PURITY,
                min_samples_per_ball=MIN_SAMPLES_PER_BALL,
                seed=seed,
                dataset=dataset_name,
            )
            summary = summarize_events(events)
            edge_info = verify_edge_cases(events)
            per_run.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "summary": summary,
                    "edge_cases": edge_info,
                }
            )
            all_events.extend(events)
            print(
                f"{dataset_name} seed={seed}: "
                f"singletons={summary['n_singleton_children']}, "
                f"outliers={summary['n_predicted_outliers']}, "
                f"legitimate={summary['n_legitimate_singletons']}, "
                f"undefined={summary['n_undefined_hd2']}, "
                f"affected={summary['n_samples_affected']}"
            )

    write_csv(CSV_PATH, all_events)

    aggregated_edge = verify_edge_cases(all_events)
    payload = {
        "protocol": {
            "purity_threshold": PURITY,
            "min_samples_per_ball": MIN_SAMPLES_PER_BALL,
            "min_ball_count_not_applied": MIN_BALL_COUNT,
            "datasets": [d.replace(".csv", "") for d in DATASETS],
            "seeds": SEEDS,
        },
        "aggregated_summary": summarize_events(all_events),
        "per_run": per_run,
        "edge_case_verification": aggregated_edge,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(REPORT_PATH, all_events, per_run, aggregated_edge)

    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
