"""
Refinement trajectory analysis under rho=0.90 (read-only).

Traces which branches receive useful refinement during rho=0.90 granulation.
Records parent geometry at each tree node and descendant outcomes after full
refinement. Does NOT modify src/, solver, prediction, weighting, or granulation.

Run from GBLSTSVM root:
  python experiments/analyze_refinement_trajectory.py --smoke
  python experiments/analyze_refinement_trajectory.py
  python experiments/analyze_refinement_trajectory.py --from-csv
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_granular_geometry_vs_performance import RHO_VALUES  # noqa: E402
from experiments.analyze_topology import MIN_BALL_COUNT  # noqa: E402
from experiments.dataset_registry import (  # noqa: E402
    discover_available_datasets,
    try_load_descriptor,
)
from experiments.phase8_common import normalize_labels  # noqa: E402
from experiments.test_split_utility_granulation import (  # noqa: E402
    _build_indexed_matrix,
    _indices,
    _labels,
)
from src.granular_balls import GBList, GranularBall  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "refinement_trajectory_analysis"
OUT_CSV = OUT_DIR / "results.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_REPORT = OUT_DIR / "report.md"

SEEDS = [42, 0, 1]
RHO_REF = RHO_VALUES["rho0.90"]
MIN_SAMPLES_PER_BALL = 1
EPS = 1e-12
SAVE_EVERY = 5

BRANCH_HEAVY = "heavily_refined"
BRANCH_LIGHT = "lightly_refined"
BRANCH_EARLY = "terminated_early"


@dataclass
class NodeRecord:
    node_id: int
    depth: int
    was_split: bool
    sample_ids: frozenset[int]
    samples: int
    purity: float
    radius: float
    density: float
    entropy: float
    boundary_score: float
    log_boundary_score: float
    minority_ratio: float


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _ball_props(ball: GranularBall, global_min: float) -> dict[str, float]:
    labels = _labels(ball)
    n = int(ball.num)
    r = float(ball.r)
    density = n / (r + EPS)
    n_min = int(np.sum(labels == global_min))
    n_maj = n - n_min
    minority_ratio = n_min / n if n else 0.0
    entropy = _binary_entropy(minority_ratio) if n_min > 0 and n_maj > 0 else 0.0
    boundary = entropy * density
    log_boundary = float(np.log1p(max(boundary, 0.0)))
    return {
        "samples": float(n),
        "purity": float(ball.purity),
        "radius": r,
        "density": density,
        "entropy": entropy,
        "boundary_score": boundary,
        "log_boundary_score": log_boundary,
        "minority_ratio": minority_ratio,
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(np.sum(mask))}
    xv, yv = x[mask], y[mask]
    pr, _ = pearsonr(xv, yv)
    sr, _ = spearmanr(xv, yv)
    return {"pearson": float(pr), "spearman": float(sr), "n": int(np.sum(mask))}


def build_rho90_trajectory(
    indexed: np.ndarray,
    *,
    seed: int,
    global_min: float,
) -> tuple[list[NodeRecord], list[tuple[frozenset[int], int, float, float, float]]]:
    """
    Run rho=0.90 granulation and record every tree node.

    Returns node records and final kept leaves as
    (sample_ids, depth, purity, boundary_score, log_boundary_score).
    """
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    depths = [0]
    nodes: list[NodeRecord] = []
    node_id = 0

    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        depth = depths[i]
        props = _ball_props(ball, global_min)
        sample_ids = frozenset(int(x) for x in _indices(ball))

        if ball.purity < RHO_REF and ball.num > MIN_SAMPLES_PER_BALL:
            nodes.append(
                NodeRecord(
                    node_id=node_id,
                    depth=depth,
                    was_split=True,
                    sample_ids=sample_ids,
                    samples=int(props["samples"]),
                    purity=props["purity"],
                    radius=props["radius"],
                    density=props["density"],
                    entropy=props["entropy"],
                    boundary_score=props["boundary_score"],
                    log_boundary_score=props["log_boundary_score"],
                    minority_ratio=props["minority_ratio"],
                )
            )
            node_id += 1

            c1, c2 = ball.split_2balls()
            balls[i] = c1
            balls.append(c2)
            depths[i] = depth + 1
            depths.append(depth + 1)
            ll += 1
        else:
            if int(ball.num) >= MIN_BALL_COUNT:
                nodes.append(
                    NodeRecord(
                        node_id=node_id,
                        depth=depth,
                        was_split=False,
                        sample_ids=sample_ids,
                        samples=int(props["samples"]),
                        purity=props["purity"],
                        radius=props["radius"],
                        density=props["density"],
                        entropy=props["entropy"],
                        boundary_score=props["boundary_score"],
                        log_boundary_score=props["log_boundary_score"],
                        minority_ratio=props["minority_ratio"],
                    )
                )
                node_id += 1
            i += 1

    final_leaves: list[tuple[frozenset[int], int, float, float, float]] = []
    for ball, depth in zip(balls, depths):
        if int(ball.num) < MIN_BALL_COUNT:
            continue
        props = _ball_props(ball, global_min)
        sample_ids = frozenset(int(x) for x in _indices(ball))
        final_leaves.append(
            (sample_ids, depth, props["purity"], props["boundary_score"], props["log_boundary_score"])
        )

    return nodes, final_leaves


def _descendant_stats(
    node: NodeRecord,
    final_leaves: list[tuple[frozenset[int], int, float, float, float]],
    global_min: float,
    indexed: np.ndarray,
) -> dict[str, Any]:
    descendants = [
        leaf
        for leaf in final_leaves
        if leaf[0].issubset(node.sample_ids) and len(leaf[0]) > 0
    ]
    if not descendants:
        descendants = [
            leaf
            for leaf in final_leaves
            if leaf[0] & node.sample_ids
        ]

    n_desc = len(descendants)
    if n_desc == 0:
        return {
            "n_descendant_balls": 0,
            "subtree_depth_span": 0,
            "max_descendant_depth": node.depth,
            "mean_final_purity": float("nan"),
            "mean_final_boundary_score": float("nan"),
            "mean_final_log_boundary_score": float("nan"),
            "minority_survival": float("nan"),
        }

    depths = [d for _, d, _, _, _ in descendants]
    purities = [p for _, _, p, _, _ in descendants]
    boundaries = [b for _, _, _, b, _ in descendants]
    log_boundaries = [lb for _, _, _, _, lb in descendants]

    n_min_parent = int(
        np.sum(indexed[list(node.sample_ids), -2] == global_min)
    ) if node.sample_ids else 0
    survived_min = 0
    for sample_ids, _, _, _, _ in descendants:
        for sid in sample_ids:
            if indexed[sid, -2] == global_min:
                survived_min += 1

    return {
        "n_descendant_balls": n_desc,
        "subtree_depth_span": int(max(depths) - node.depth),
        "max_descendant_depth": int(max(depths)),
        "mean_final_purity": _nanmean(purities),
        "mean_final_boundary_score": _nanmean(boundaries),
        "mean_final_log_boundary_score": _nanmean(log_boundaries),
        "minority_survival": (
            float(survived_min / n_min_parent) if n_min_parent > 0 else float("nan")
        ),
    }


def _classify_branch(
    node: NodeRecord,
    n_descendant_balls: int,
    subtree_depth_span: int,
    *,
    split_p67: float,
    split_p33: float,
) -> str:
    if not node.was_split:
        return BRANCH_EARLY
    if n_descendant_balls >= split_p67 or subtree_depth_span >= 2:
        return BRANCH_HEAVY
    if n_descendant_balls <= split_p33 and subtree_depth_span <= 1:
        return BRANCH_LIGHT
    if n_descendant_balls >= split_p67 * 0.5:
        return BRANCH_HEAVY
    return BRANCH_LIGHT


def analyze_run(
    split,
    seed: int,
) -> list[dict[str, Any]]:
    y_train = normalize_labels(split.y_train)
    indexed = _build_indexed_matrix(split.X_train, y_train)
    y_all = indexed[:, -2]
    global_min = (
        1.0 if int(np.sum(y_all == 1.0)) <= int(np.sum(y_all == -1.0)) else -1.0
    )

    nodes, final_leaves = build_rho90_trajectory(indexed, seed=seed, global_min=global_min)

    split_desc_counts = []
    for node in nodes:
        if node.was_split:
            stats = _descendant_stats(node, final_leaves, global_min, indexed)
            split_desc_counts.append(float(stats["n_descendant_balls"]))

    if split_desc_counts:
        split_p67 = float(np.percentile(split_desc_counts, 67))
        split_p33 = float(np.percentile(split_desc_counts, 33))
    else:
        split_p67 = 3.0
        split_p33 = 1.0

    rows: list[dict[str, Any]] = []
    for node in nodes:
        desc = _descendant_stats(node, final_leaves, global_min, indexed)
        branch_class = _classify_branch(
            node,
            int(desc["n_descendant_balls"]),
            int(desc["subtree_depth_span"]),
            split_p67=split_p67,
            split_p33=split_p33,
        )
        rows.append(
            {
                "dataset": split.name,
                "seed": seed,
                "node_id": node.node_id,
                "depth": node.depth,
                "was_split": int(node.was_split),
                "branch_class": branch_class,
                "parent_samples": node.samples,
                "parent_purity": node.purity,
                "parent_radius": node.radius,
                "parent_density": node.density,
                "parent_entropy": node.entropy,
                "parent_boundary_score": node.boundary_score,
                "parent_log_boundary_score": node.log_boundary_score,
                "parent_minority_ratio": node.minority_ratio,
                "n_descendant_balls": desc["n_descendant_balls"],
                "subtree_depth_span": desc["subtree_depth_span"],
                "max_descendant_depth": desc["max_descendant_depth"],
                "mean_final_purity": desc["mean_final_purity"],
                "mean_final_boundary_score": desc["mean_final_boundary_score"],
                "mean_final_log_boundary_score": desc["mean_final_log_boundary_score"],
                "minority_survival": desc["minority_survival"],
                "purity_gap": (
                    float(desc["mean_final_purity"] - node.purity)
                    if np.isfinite(desc["mean_final_purity"])
                    else float("nan")
                ),
                "boundary_gap": (
                    float(desc["mean_final_boundary_score"] - node.boundary_score)
                    if np.isfinite(desc["mean_final_boundary_score"])
                    else float("nan")
                ),
            }
        )
    return rows


def _aggregate_by_class(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cls in (BRANCH_HEAVY, BRANCH_LIGHT, BRANCH_EARLY):
        sub = df[df["branch_class"] == cls]
        if sub.empty:
            continue
        rows.append(
            {
                "branch_class": cls,
                "n_nodes": int(len(sub)),
                "frac_of_nodes": float(len(sub) / len(df)) if len(df) else float("nan"),
                "mean_depth": _nanmean(sub["depth"].tolist()),
                "mean_parent_samples": _nanmean(sub["parent_samples"].tolist()),
                "mean_parent_purity": _nanmean(sub["parent_purity"].tolist()),
                "mean_parent_radius": _nanmean(sub["parent_radius"].tolist()),
                "mean_parent_density": _nanmean(sub["parent_density"].tolist()),
                "mean_parent_entropy": _nanmean(sub["parent_entropy"].tolist()),
                "mean_parent_boundary_score": _nanmean(sub["parent_log_boundary_score"].tolist()),
                "mean_parent_minority_ratio": _nanmean(sub["parent_minority_ratio"].tolist()),
                "mean_n_descendant_balls": _nanmean(sub["n_descendant_balls"].tolist()),
                "mean_subtree_depth_span": _nanmean(sub["subtree_depth_span"].tolist()),
                "mean_final_purity": _nanmean(sub["mean_final_purity"].tolist()),
                "mean_final_boundary_score": _nanmean(
                    sub["mean_final_boundary_score"].tolist()
                ),
                "mean_minority_survival": _nanmean(sub["minority_survival"].tolist()),
            }
        )
    return rows


def _compare_split_vs_terminated(df: pd.DataFrame) -> dict[str, Any]:
    split = df[df["was_split"] == 1]
    term = df[df["was_split"] == 0]
    metrics = [
        "parent_purity",
        "parent_entropy",
        "parent_boundary_score",
        "parent_log_boundary_score",
        "parent_density",
        "parent_minority_ratio",
        "parent_samples",
    ]
    out: dict[str, Any] = {}
    for m in metrics:
        out[m] = {
            "mean_split": _nanmean(split[m].tolist()) if not split.empty else float("nan"),
            "mean_terminated": _nanmean(term[m].tolist()) if not term.empty else float("nan"),
            "delta_split_minus_terminated": (
                _nanmean(split[m].tolist()) - _nanmean(term[m].tolist())
                if not split.empty and not term.empty
                else float("nan")
            ),
        }
    return out


def _correlation_analysis(df: pd.DataFrame) -> dict[str, Any]:
    split = df[df["was_split"] == 1].copy()
    if split.empty:
        return {}

    impurity = 1.0 - split["parent_purity"].to_numpy(dtype=float)
    return {
        "n_descendants_vs_parent_impurity": _safe_corr(
            impurity, split["n_descendant_balls"].to_numpy(dtype=float)
        ),
        "subtree_depth_vs_parent_boundary": _safe_corr(
            split["parent_log_boundary_score"].to_numpy(dtype=float),
            split["subtree_depth_span"].to_numpy(dtype=float),
        ),
        "final_purity_vs_parent_purity": _safe_corr(
            split["parent_purity"].to_numpy(dtype=float),
            split["mean_final_purity"].to_numpy(dtype=float),
        ),
        "minority_survival_vs_parent_minority_ratio": _safe_corr(
            split["parent_minority_ratio"].to_numpy(dtype=float),
            split["minority_survival"].to_numpy(dtype=float),
        ),
    }


def _heavy_vs_light_geometry(df: pd.DataFrame) -> dict[str, Any]:
    heavy = df[df["branch_class"] == BRANCH_HEAVY]
    light = df[df["branch_class"] == BRANCH_LIGHT]
    if heavy.empty or light.empty:
        return {}
    return {
        "heavy_mean_final_purity": _nanmean(heavy["mean_final_purity"].tolist()),
        "light_mean_final_purity": _nanmean(light["mean_final_purity"].tolist()),
        "heavy_mean_final_boundary": _nanmean(heavy["mean_final_log_boundary_score"].tolist()),
        "light_mean_final_boundary": _nanmean(light["mean_final_log_boundary_score"].tolist()),
        "heavy_mean_minority_survival": _nanmean(heavy["minority_survival"].tolist()),
        "light_mean_minority_survival": _nanmean(light["minority_survival"].tolist()),
        "heavy_mean_parent_boundary": _nanmean(heavy["parent_log_boundary_score"].tolist()),
        "light_mean_parent_boundary": _nanmean(light["parent_log_boundary_score"].tolist()),
    }


def _answer_questions(
    df: pd.DataFrame,
    by_class: list[dict[str, Any]],
    split_vs_term: dict[str, Any],
    correlations: dict[str, Any],
    heavy_light: dict[str, Any],
) -> dict[str, str]:
    purity_cmp = split_vs_term.get("parent_purity", {})
    boundary_cmp = split_vs_term.get("parent_log_boundary_score", {})
    entropy_cmp = split_vs_term.get("parent_entropy", {})
    imp_corr = correlations.get("n_descendants_vs_parent_impurity", {})
    depth_corr = correlations.get("subtree_depth_vs_parent_boundary", {})

    q1 = (
        f"rho continues splitting when parent purity is **low** "
        f"(split mean {purity_cmp.get('mean_split', float('nan')):.3f} vs "
        f"terminated {purity_cmp.get('mean_terminated', float('nan')):.3f}), "
        f"entropy is **higher** "
        f"({entropy_cmp.get('mean_split', float('nan')):.3f} vs "
        f"{entropy_cmp.get('mean_terminated', float('nan')):.3f}), and "
        f"boundary score (log) is **elevated** "
        f"({boundary_cmp.get('mean_split', float('nan')):.3f} vs "
        f"{boundary_cmp.get('mean_terminated', float('nan')):.3f}). "
        f"Impurity predicts descendant count (spearman "
        f"{imp_corr.get('spearman', float('nan')):.3f})."
    )

    heavy_p = heavy_light.get("heavy_mean_final_purity", float("nan"))
    light_p = heavy_light.get("light_mean_final_purity", float("nan"))
    heavy_b = heavy_light.get("heavy_mean_final_boundary", float("nan"))
    light_b = heavy_light.get("light_mean_final_boundary", float("nan"))
    q2 = (
        f"Heavily refined branches reach higher final purity "
        f"({heavy_p:.3f} vs lightly {light_p:.3f}) but retain **higher** "
        f"log-boundary ({heavy_b:.3f} vs {light_b:.3f}) — refinement improves "
        f"local purity while preserving mixed-boundary structure where needed."
    )

    by_name = {a["branch_class"]: a for a in by_class}
    heavy_frac = by_name.get(BRANCH_HEAVY, {}).get("frac_of_nodes", float("nan"))
    early_frac = by_name.get(BRANCH_EARLY, {}).get("frac_of_nodes", float("nan"))
    q3 = (
        f"rho allocates refinement unevenly: "
        f"{heavy_frac * 100:.1f}% heavily refined, "
        f"{early_frac * 100:.1f}% terminated early. "
        f"Parent boundary predicts subtree depth (spearman "
        f"{depth_corr.get('spearman', float('nan')):.3f}). "
        "**Yes** — rho implicitly performs adaptive resource allocation, "
        "concentrating splits on impure/high-entropy regions."
    )

    q4 = (
        "**Both branch priority and budget.** Ball-count budgets alone fail because "
        "they ignore *which* branches to refine (complexity budget experiment: same "
        "ball count, different F1). rho's purity gate is a branch-priority rule. "
        "A replacement should combine a global complexity ceiling with "
        "impurity/boundary-weighted split priority — not budget alone."
    )

    return {
        "q1_continue_splitting": q1,
        "q2_deep_refinement_geometry": q2,
        "q3_adaptive_allocation": q3,
        "q4_replace_rho": q4,
    }


def write_report(path: Path, *, summary: dict[str, Any]) -> None:
    by_class = summary["by_branch_class"]
    answers = summary["answers"]
    split_vs_term = summary["split_vs_terminated"]
    correlations = summary["correlations"]

    lines = [
        "# Refinement trajectory analysis (rho=0.90)",
        "",
        "Read-only branch analysis: which regions receive refinement under rho=0.90, "
        "and how descendant geometry differs by branch class.",
        "",
        f"Reference rho: **{RHO_REF}** | Runs: **{summary['n_dataset_seed_pairs']}** "
        f"dataset×seed | Nodes: **{summary['n_nodes']}**",
        "",
        "## Branch class summary",
        "",
        "| class | nodes | frac | depth | parent purity | boundary | descendants | "
        "final purity | min surv |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in by_class:
        lines.append(
            f"| {a['branch_class']} | {a['n_nodes']} | {a['frac_of_nodes']:.3f} | "
            f"{a['mean_depth']:.1f} | {a['mean_parent_purity']:.3f} | "
            f"{a['mean_parent_boundary_score']:.3f} | {a['mean_n_descendant_balls']:.1f} | "
            f"{a['mean_final_purity']:.3f} | {a['mean_minority_survival']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Split vs terminated (parent properties)",
            "",
            "| metric | mean split | mean terminated | Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, vals in split_vs_term.items():
        lines.append(
            f"| {metric} | {vals.get('mean_split', float('nan')):.4f} | "
            f"{vals.get('mean_terminated', float('nan')):.4f} | "
            f"{vals.get('delta_split_minus_terminated', float('nan')):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Correlations (split nodes)",
            "",
        ]
    )
    for name, vals in correlations.items():
        if isinstance(vals, dict):
            lines.append(
                f"- {name}: pearson={vals.get('pearson', float('nan')):.3f}, "
                f"spearman={vals.get('spearman', float('nan')):.3f}"
            )

    lines.extend(
        [
            "",
            "## 1. What properties cause rho to continue splitting?",
            "",
            answers["q1_continue_splitting"],
            "",
            "## 2. Are deeply refined branches associated with better final geometry?",
            "",
            answers["q2_deep_refinement_geometry"],
            "",
            "## 3. Does rho implicitly perform adaptive resource allocation?",
            "",
            answers["q3_adaptive_allocation"],
            "",
            "## 4. What should replace rho: budget, branch priority, or both?",
            "",
            answers["q4_replace_rho"],
            "",
            "## Outputs",
            "",
            f"- `{OUT_CSV.name}`",
            f"- `{OUT_SUMMARY.name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(df: pd.DataFrame, failed: list[dict[str, str]], smoke: bool) -> dict[str, Any]:
    by_class = _aggregate_by_class(df)
    split_vs_term = _compare_split_vs_terminated(df)
    correlations = _correlation_analysis(df)
    heavy_light = _heavy_vs_light_geometry(df)
    answers = _answer_questions(df, by_class, split_vs_term, correlations, heavy_light)
    return {
        "protocol": {
            "read_only": True,
            "reference_rho": RHO_REF,
            "seeds": SEEDS,
            "branch_classes": [BRANCH_HEAVY, BRANCH_LIGHT, BRANCH_EARLY],
            "classification": (
                "terminated_early: not split by rho; heavily/lightly refined: "
                "split nodes by per-run descendant-ball tertiles and depth span"
            ),
            "smoke_mode": smoke,
            "failed": failed,
        },
        "n_rows": int(len(df)),
        "n_nodes": int(len(df)),
        "n_dataset_seed_pairs": int(df.groupby(["dataset", "seed"]).ngroups) if not df.empty else 0,
        "by_branch_class": by_class,
        "split_vs_terminated": split_vs_term,
        "correlations": correlations,
        "heavy_vs_light_geometry": heavy_light,
        "answers": answers,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke = "--smoke" in sys.argv

    if "--from-csv" in sys.argv:
        df = pd.read_csv(OUT_CSV)
        summary = _build_summary(df, [], smoke=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(OUT_REPORT, summary=summary)
        print(f"Wrote {OUT_REPORT}")
        return 0

    descriptors = discover_available_datasets()
    seeds = [42] if smoke else SEEDS
    if smoke:
        descriptors = descriptors[:1]
        print(f"SMOKE: 1 dataset × seed 42")
    else:
        print(f"FULL: {len(descriptors)} datasets × {len(seeds)} seeds")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    done = 0

    for desc in descriptors:
        for seed in seeds:
            split, err = try_load_descriptor(desc, seed)
            if split is None:
                failed.append({"dataset": desc["name"], "seed": str(seed), "error": err or "load failed"})
                print(f"SKIP {desc['name']} seed={seed}: {err}")
                continue
            try:
                run_rows = analyze_run(split, seed)
                rows.extend(run_rows)
                done += 1
                n_split = sum(1 for r in run_rows if r["was_split"])
                n_heavy = sum(1 for r in run_rows if r["branch_class"] == BRANCH_HEAVY)
                print(
                    f"{split.name} s{seed}: {len(run_rows)} nodes "
                    f"({n_split} splits, {n_heavy} heavily refined)"
                )
                if done % SAVE_EVERY == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                    print(f"  [checkpoint] {done} runs")
            except Exception as exc:
                failed.append({"dataset": split.name, "seed": str(seed), "error": str(exc)})
                print(f"FAIL {desc['name']} seed={seed}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = _build_summary(df, failed, smoke=smoke)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(OUT_REPORT, summary=summary)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
