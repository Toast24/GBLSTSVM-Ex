"""
Shared helpers for the principled K-selection decision experiment.

Leakage policy:
  - rho90 granulation, ball ranking, compression, and K selection use train_selection only.
  - internal validation (20% of benchmark train) is used ONLY to choose K.
  - benchmark test (15%) is used ONLY for final evaluation after K is frozen.
  - oracle-K diagnostics are POST-HOC and never influence method behaviour.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from experiments.analyze_ball_utility_landscape import compute_ball_features
from experiments.granular_compression_k_common import (
    COLLAPSE_BALL_THRESHOLD,
    build_rho90_from_train_indices,
    deterministic_rng_seed,
)
from experiments.phase8_common import normalize_labels
from experiments.test_adaptive_entropy_granulation_v2 import RHO_REF, _global_min_label
from experiments.test_split_utility_granulation import _build_indexed_matrix
from experiments.validation_selected_k_common import (
    COMPRESSION_REPS,
    DELTA_EPS,
    KValidationSummary,
    assert_partition_disjoint_from_test,
    build_rho90_pool,
    compress_and_evaluate,
    compression_seed,
    eval_full_rho90,
    make_train_val_partition,
    select_k_1se,
    select_k_max,
    summarize_k_validation,
)
from src.granular_balls import GBList

EPS = 1e-12
REPRESENTATION = "rho0.90"
K_CANDIDATES = [2, 4, 6, 8, 10, 12]
SEEDS = list(range(10))
FRACTION_OF_RHO90 = 0.15
SIZE_K_MIN = 6
SIZE_K_MAX = 12

METHOD_FULL_RHO90 = "FULL_RHO90"
METHOD_FIXED_K6 = "FIXED_K6"
METHOD_FIXED_K8 = "FIXED_K8"
METHOD_FIXED_K10 = "FIXED_K10"
METHOD_FIXED_K12 = "FIXED_K12"
METHOD_VAL_MAX = "VALIDATION_SELECTED_MAX"
METHOD_VAL_1SE = "VALIDATION_SELECTED_1SE"
METHOD_PREREG_K10 = "PRE_REGISTERED_K10_RULE"
METHOD_SIZE_RULE = "SIZE_BASED_RULE"
METHOD_FRACTION_RULE = "FRACTION_OF_RHO90_RULE"

FIXED_K_METHODS = {
    6: METHOD_FIXED_K6,
    8: METHOD_FIXED_K8,
    10: METHOD_FIXED_K10,
    12: METHOD_FIXED_K12,
}

ALL_METHODS = (
    METHOD_FULL_RHO90,
    METHOD_FIXED_K6,
    METHOD_FIXED_K8,
    METHOD_FIXED_K10,
    METHOD_FIXED_K12,
    METHOD_VAL_MAX,
    METHOD_VAL_1SE,
    METHOD_PREREG_K10,
    METHOD_SIZE_RULE,
    METHOD_FRACTION_RULE,
)

PRIMARY_COMPARISONS = (
    (METHOD_VAL_MAX, METHOD_FULL_RHO90),
    (METHOD_VAL_MAX, METHOD_FIXED_K6),
    (METHOD_VAL_MAX, METHOD_FIXED_K10),
    (METHOD_VAL_MAX, METHOD_FIXED_K12),
    (METHOD_VAL_MAX, METHOD_SIZE_RULE),
    (METHOD_VAL_MAX, METHOD_FRACTION_RULE),
)

RUN_COLUMNS = (
    "dataset",
    "seed",
    "method",
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "n_train",
    "n_balls_full_rho90",
    "n_balls_final",
    "K_selected",
    "n_splits",
    "collapse",
    "minority_survival",
    "majority_survival",
    "validation_best_k",
    "validation_best_f1",
    "runtime_seconds",
)


def size_based_k(n_train: int) -> int:
    """K = min(12, max(6, round(sqrt(n_train))))."""
    return int(min(SIZE_K_MAX, max(SIZE_K_MIN, round(np.sqrt(n_train)))))


def fraction_of_rho90_k(b90: int) -> int:
    """K = min(12, max(6, round(0.15 * B90)))."""
    return int(min(SIZE_K_MAX, max(SIZE_K_MIN, round(FRACTION_OF_RHO90 * b90))))


def measure_rho90_split_count(indexed: np.ndarray, *, seed: int) -> int:
    """Count KMeans(k=2) splits during standard rho=0.90 granulation."""
    np.random.seed(seed)
    balls = GBList(indexed).granular_balls
    n_splits = 0
    ll = len(balls)
    i = 0
    while i < ll:
        ball = balls[i]
        if float(ball.purity) < RHO_REF and int(ball.num) > 1:
            c1, c2 = ball.split_2balls()
            balls[i] = c1
            balls.append(c2)
            ll += 1
            n_splits += 1
        else:
            i += 1
    return n_splits


def _nanmean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _nanmedian(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _nanstd(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if v.size > 1 else (0.0 if v.size == 1 else float("nan"))


def _cohens_d(deltas: np.ndarray) -> float:
    if deltas.size < 2:
        return float("nan")
    return float(np.mean(deltas) / (np.std(deltas, ddof=1) + 1e-12))


def pack_run_row(
    dataset: str,
    seed: int,
    method: str,
    ev: dict[str, Any],
    *,
    n_train: int,
    n_balls_full_rho90: int,
    K_selected: float | int | None = None,
    n_splits: int | float = float("nan"),
    validation_best_k: float | int | None = None,
    validation_best_f1: float | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    k_val = float("nan")
    if K_selected is not None and not (isinstance(K_selected, float) and np.isnan(K_selected)):
        k_val = float(K_selected)
    vbk = float("nan")
    if validation_best_k is not None and not (
        isinstance(validation_best_k, float) and np.isnan(validation_best_k)
    ):
        vbk = float(validation_best_k)
    return {
        "dataset": dataset,
        "seed": int(seed),
        "method": method,
        "macro_f1": float(ev.get("macro_f1", float("nan"))),
        "accuracy": float(ev.get("accuracy", float("nan"))),
        "balanced_accuracy": float(ev.get("balanced_accuracy", float("nan"))),
        "n_train": int(n_train),
        "n_balls_full_rho90": int(n_balls_full_rho90),
        "n_balls_final": int(ev.get("n_balls", 0)),
        "K_selected": k_val,
        "n_splits": int(n_splits) if np.isfinite(n_splits) else float("nan"),
        "collapse": int(ev.get("collapsed", ev.get("collapse", 0))),
        "minority_survival": float(ev.get("minority_survival", float("nan"))),
        "majority_survival": float(ev.get("majority_survival", float("nan"))),
        "validation_best_k": vbk,
        "validation_best_f1": float(
            validation_best_f1 if validation_best_f1 is not None else float("nan")
        ),
        "runtime_seconds": float(runtime_seconds if runtime_seconds is not None else float("nan")),
    }


def run_validation_k_selection(
    dataset: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    minority_label: float,
    k_candidates: list[int],
    n_reps: int = COMPRESSION_REPS,
) -> tuple[
    dict[int, KValidationSummary],
    int | None,
    int | None,
    float,
    int,
    dict[str, Any],
]:
    """
    Select K using internal validation only (train_selection pool).

    Returns summaries, k_max, k_1se, validation_best_f1, n_pool_sel, meta.
    """
    y = normalize_labels(y_train)

    partition = make_train_val_partition(dataset, seed, X_train, y)
    assert_partition_disjoint_from_test(
        partition, len(X_train), 0, context="validation_selection"
    )

    sel_idx = partition.selection_indices
    val_idx = partition.validation_indices
    y_sel = y[sel_idx]
    y_val = y[val_idx]
    X_val = X_train[val_idx]

    indexed_full = _build_indexed_matrix(X_train, y)
    global_min = _global_min_label(indexed_full)

    balls_sel, kept_sel, _, _ = build_rho90_pool(X_train, y, sel_idx, seed=seed)
    features_sel = compute_ball_features(kept_sel, minority_label)
    n_pool_sel = len(balls_sel)

    k_summaries: dict[int, KValidationSummary] = {}
    for k in k_candidates:
        rep_rows: list[dict[str, Any]] = []
        if n_pool_sel < k:
            skip = {"K": k, "skipped": 1, "skip_reason": f"rho90_pool={n_pool_sel}<K={k}"}
            k_summaries[k] = summarize_k_validation([skip])
            continue
        for rep in range(n_reps):
            rng_seed = compression_seed(dataset, seed, k, rep, phase="selection")
            ev = compress_and_evaluate(
                balls_sel,
                kept_sel,
                features_sel,
                k,
                minority_label=minority_label,
                rng_seed=rng_seed,
                y_train_fit=y_sel,
                X_eval=X_val,
                y_eval=y_val,
                seed=seed,
                indexed=indexed_full,
                global_min=global_min,
                phase="selection",
            )
            rep_rows.append(ev)
        k_summaries[k] = summarize_k_validation(rep_rows)

    k_max, _ = select_k_max(k_summaries)
    k_1se, _, _ = select_k_1se(k_summaries)
    val_best_f1 = float("nan")
    if k_max is not None and k_max in k_summaries:
        val_best_f1 = float(k_summaries[k_max].mean_macro_f1)

    meta = {
        "partition": partition,
        "indexed_full": indexed_full,
        "global_min": global_min,
        "minority_label": minority_label,
        "n_pool_sel": n_pool_sel,
    }
    return k_summaries, k_max, k_1se, val_best_f1, n_pool_sel, meta


def eval_compressed_on_test(
    split,
    seed: int,
    k: int,
    *,
    indexed_full: np.ndarray,
    global_min: float,
    minority_label: float,
    balls_full: list,
    kept_full: list,
    features_full: list,
    dataset: str,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Random ball compression at K on full benchmark train; evaluate on test."""
    if len(balls_full) < k:
        return {
            "skipped": 1,
            "skip_reason": f"pool={len(balls_full)}<K={k}",
            "macro_f1": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "collapsed": 1,
            "n_balls": 0,
            "minority_survival": float("nan"),
            "majority_survival": float("nan"),
        }
    rng_seed = compression_seed(dataset, seed, k, 0, phase="final_test")
    return compress_and_evaluate(
        balls_full,
        kept_full,
        features_full,
        k,
        minority_label=minority_label,
        rng_seed=rng_seed,
        y_train_fit=y_train,
        X_eval=split.X_test,
        y_eval=split.y_test,
        seed=seed,
        indexed=indexed_full,
        global_min=global_min,
        phase="final_test",
    )


def oracle_test_optimal_k(
    split,
    seed: int,
    k_candidates: list[int],
    *,
    indexed_full: np.ndarray,
    global_min: float,
    minority_label: float,
    balls_full: list,
    kept_full: list,
    features_full: list,
    dataset: str,
    y_train: np.ndarray,
) -> tuple[int | None, float, dict[int, float]]:
    """POST-HOC: argmax test macro F1 over candidates; ties → smaller K."""
    test_f1_by_k: dict[int, float] = {}
    for k in k_candidates:
        ev = eval_compressed_on_test(
            split,
            seed,
            k,
            indexed_full=indexed_full,
            global_min=global_min,
            minority_label=minority_label,
            balls_full=balls_full,
            kept_full=kept_full,
            features_full=features_full,
            dataset=dataset,
            y_train=y_train,
        )
        f1 = float(ev.get("macro_f1", float("nan")))
        test_f1_by_k[k] = f1

    eligible = {k: f1 for k, f1 in test_f1_by_k.items() if np.isfinite(f1)}
    if not eligible:
        return None, float("nan"), test_f1_by_k

    best_k = min(
        eligible.keys(),
        key=lambda k: (-eligible[k], k),
    )
    return int(best_k), float(eligible[best_k]), test_f1_by_k


def run_dataset_seed(
    split,
    seed: int,
    *,
    k_candidates: list[int] | None = None,
    n_reps: int = COMPRESSION_REPS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Run all methods for one dataset×seed.

    Returns (all_runs, oracle_rows, failures).
    """
    dataset = split.name
    k_candidates = list(k_candidates or K_CANDIDATES)
    y_train = normalize_labels(split.y_train)
    X_train = split.X_train
    n_train = len(y_train)
    minority_label = split.minority_label

    t0 = time.perf_counter()

    k_summaries, k_max, k_1se, val_best_f1, _, sel_meta = run_validation_k_selection(
        dataset,
        seed,
        X_train,
        y_train,
        minority_label=minority_label,
        k_candidates=k_candidates,
        n_reps=n_reps,
    )
    indexed_full = sel_meta["indexed_full"]
    global_min = sel_meta["global_min"]

    all_idx = np.arange(n_train, dtype=int)
    balls_full, kept_full = build_rho90_from_train_indices(indexed_full, all_idx, seed=seed)
    features_full = compute_ball_features(kept_full, minority_label)
    n_balls_full = len(balls_full)
    n_splits = measure_rho90_split_count(indexed_full, seed=seed)

    k_size = size_based_k(n_train)
    k_fraction = fraction_of_rho90_k(n_balls_full)

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _record(method: str, ev: dict[str, Any], k_sel: float | int | None) -> None:
        if ev.get("skipped"):
            failures.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "K": k_sel,
                    "reason": ev.get("skip_reason", "skipped"),
                }
            )
        runs.append(
            pack_run_row(
                dataset,
                seed,
                method,
                ev,
                n_train=n_train,
                n_balls_full_rho90=n_balls_full,
                K_selected=k_sel,
                n_splits=n_splits,
                validation_best_k=k_max,
                validation_best_f1=val_best_f1,
                runtime_seconds=time.perf_counter() - t0,
            )
        )

    # FULL_RHO90
    ev_full = eval_full_rho90(
        split, seed=seed, phase="final_test", X_eval=split.X_test, y_eval=split.y_test
    )
    _record(METHOD_FULL_RHO90, ev_full, None)

    # FIXED_K*
    fixed_ks = {6, 8, 10, 12}
    fixed_evals: dict[int, dict[str, Any]] = {}
    for fk in fixed_ks:
        ev = eval_compressed_on_test(
            split,
            seed,
            fk,
            indexed_full=indexed_full,
            global_min=global_min,
            minority_label=minority_label,
            balls_full=balls_full,
            kept_full=kept_full,
            features_full=features_full,
            dataset=dataset,
            y_train=y_train,
        )
        fixed_evals[fk] = ev
        _record(FIXED_K_METHODS[fk], ev, fk)

    # PRE_REGISTERED_K10_RULE — identical path to FIXED_K10
    _record(METHOD_PREREG_K10, fixed_evals[10], 10)

    # VALIDATION_SELECTED_*
    for sel_k, method in ((k_max, METHOD_VAL_MAX), (k_1se, METHOD_VAL_1SE)):
        if sel_k is None:
            ev = {
                "skipped": 1,
                "skip_reason": "no_eligible_K",
                "macro_f1": float("nan"),
                "collapsed": 1,
                "n_balls": 0,
            }
            _record(method, ev, None)
            continue
        ev = eval_compressed_on_test(
            split,
            seed,
            int(sel_k),
            indexed_full=indexed_full,
            global_min=global_min,
            minority_label=minority_label,
            balls_full=balls_full,
            kept_full=kept_full,
            features_full=features_full,
            dataset=dataset,
            y_train=y_train,
        )
        _record(method, ev, int(sel_k))

    # Rule-based K (no tuning after seeing results)
    for k_rule, method in ((k_size, METHOD_SIZE_RULE), (k_fraction, METHOD_FRACTION_RULE)):
        ev = eval_compressed_on_test(
            split,
            seed,
            k_rule,
            indexed_full=indexed_full,
            global_min=global_min,
            minority_label=minority_label,
            balls_full=balls_full,
            kept_full=kept_full,
            features_full=features_full,
            dataset=dataset,
            y_train=y_train,
        )
        _record(method, ev, k_rule)

    # Oracle diagnostic (post-hoc only)
    oracle_k, oracle_f1, test_f1_by_k = oracle_test_optimal_k(
        split,
        seed,
        k_candidates,
        indexed_full=indexed_full,
        global_min=global_min,
        minority_label=minority_label,
        balls_full=balls_full,
        kept_full=kept_full,
        features_full=features_full,
        dataset=dataset,
        y_train=y_train,
    )
    max_row = next((r for r in runs if r["method"] == METHOD_VAL_MAX), None)
    one_se_row = next((r for r in runs if r["method"] == METHOD_VAL_1SE), None)
    max_f1 = float(max_row["macro_f1"]) if max_row else float("nan")
    one_se_f1 = float(one_se_row["macro_f1"]) if one_se_row else float("nan")

    oracle_rows: list[dict[str, Any]] = []
    for k, f1 in test_f1_by_k.items():
        val_f1 = (
            float(k_summaries[k].mean_macro_f1)
            if k in k_summaries and not k_summaries[k].skipped
            else float("nan")
        )
        oracle_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "K": k,
                "test_macro_f1": f1,
                "val_mean_macro_f1": val_f1,
                "oracle_k": oracle_k if oracle_k is not None else float("nan"),
                "oracle_test_macro_f1": oracle_f1,
                "K_selected_max": k_max if k_max is not None else float("nan"),
                "K_selected_1se": k_1se if k_1se is not None else float("nan"),
                "K_size_rule": k_size,
                "K_fraction_rule": k_fraction,
                "regret_max": float(oracle_f1 - max_f1)
                if np.isfinite(oracle_f1) and np.isfinite(max_f1)
                else float("nan"),
                "regret_1se": float(oracle_f1 - one_se_f1)
                if np.isfinite(oracle_f1) and np.isfinite(one_se_f1)
                else float("nan"),
                "is_oracle_k": int(k == oracle_k) if oracle_k is not None else 0,
            }
        )

    return runs, oracle_rows, failures


def build_summary_by_method(df) -> Any:
    import pandas as pd

    rows = []
    for method, g in df.groupby("method"):
        rows.append(
            {
                "method": method,
                "n_runs": len(g),
                "mean_macro_f1": _nanmean(g["macro_f1"]),
                "median_macro_f1": _nanmedian(g["macro_f1"]),
                "std_macro_f1": _nanstd(g["macro_f1"]),
                "mean_accuracy": _nanmean(g["accuracy"]),
                "mean_balanced_accuracy": _nanmean(g["balanced_accuracy"]),
                "frac_collapse": float(g["collapse"].mean()),
                "mean_n_balls_final": _nanmean(g["n_balls_final"]),
                "mean_K_selected": _nanmean(g["K_selected"]),
            }
        )
    return pd.DataFrame(rows)


def build_paired_comparisons(df) -> Any:
    import pandas as pd
    from scipy.stats import wilcoxon

    rows = []
    for focal, baseline in PRIMARY_COMPARISONS:
        f = df[df["method"] == focal].set_index(["dataset", "seed"])
        b = df[df["method"] == baseline].set_index(["dataset", "seed"])
        keys = sorted(set(f.index) & set(b.index))
        deltas = []
        for key in keys:
            fv = float(f.loc[key, "macro_f1"])
            bv = float(b.loc[key, "macro_f1"])
            if np.isfinite(fv) and np.isfinite(bv):
                deltas.append(fv - bv)
        if not deltas:
            rows.append(
                {
                    "focal": focal,
                    "baseline": baseline,
                    "n_pairs": 0,
                    "mean_delta_macro_f1": float("nan"),
                    "median_delta_macro_f1": float("nan"),
                    "std_delta_macro_f1": float("nan"),
                    "wins": 0,
                    "ties": 0,
                    "losses": 0,
                    "wilcoxon_p": float("nan"),
                    "cohens_d": float("nan"),
                }
            )
            continue
        arr = np.asarray(deltas)
        wins = int(np.sum(arr > DELTA_EPS))
        ties = int(np.sum(np.abs(arr) <= DELTA_EPS))
        losses = int(np.sum(arr < -DELTA_EPS))
        try:
            wstat, wp = wilcoxon(arr) if not np.allclose(arr, 0) else (0.0, 1.0)
        except ValueError:
            wstat, wp = float("nan"), float("nan")
        rows.append(
            {
                "focal": focal,
                "baseline": baseline,
                "n_pairs": len(deltas),
                "mean_delta_macro_f1": float(np.mean(arr)),
                "median_delta_macro_f1": float(np.median(arr)),
                "std_delta_macro_f1": float(np.std(arr)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_stat": float(wstat) if np.isfinite(wstat) else float("nan"),
                "wilcoxon_p": float(wp) if np.isfinite(wp) else float("nan"),
                "cohens_d": _cohens_d(arr),
            }
        )
    return pd.DataFrame(rows)


def build_k_selection_frequency(df) -> Any:
    import pandas as pd

    rows = []
    for method in (METHOD_VAL_MAX, METHOD_VAL_1SE):
        sub = df[df["method"] == method]["K_selected"].dropna()
        if sub.empty:
            continue
        for k, count in sub.value_counts().sort_index().items():
            rows.append(
                {
                    "method": method,
                    "K": float(k),
                    "count": int(count),
                    "fraction": float(count / len(sub)),
                }
            )
    return pd.DataFrame(rows)


def build_dataset_summary(df) -> Any:
    import pandas as pd

    rows = []
    for (dataset, method), g in df.groupby(["dataset", "method"]):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_seeds": len(g),
                "mean_macro_f1": _nanmean(g["macro_f1"]),
                "std_macro_f1": _nanstd(g["macro_f1"]),
                "frac_collapse": float(g["collapse"].mean()),
                "mean_K_selected": _nanmean(g["K_selected"]),
            }
        )

    out = pd.DataFrame(rows)
    wtl_rows = []
    focal = METHOD_VAL_MAX
    for dataset in df["dataset"].unique():
        for baseline in (
            METHOD_FULL_RHO90,
            METHOD_FIXED_K6,
            METHOD_FIXED_K10,
            METHOD_FIXED_K12,
            METHOD_SIZE_RULE,
            METHOD_FRACTION_RULE,
        ):
            f = df[(df["dataset"] == dataset) & (df["method"] == focal)]
            b = df[(df["dataset"] == dataset) & (df["method"] == baseline)]
            merged = f.merge(b, on=["dataset", "seed"], suffixes=("_f", "_b"))
            if merged.empty:
                continue
            deltas = merged["macro_f1_f"] - merged["macro_f1_b"]
            valid = deltas[np.isfinite(deltas)]
            if valid.empty:
                continue
            wins = int(np.sum(valid > DELTA_EPS))
            ties = int(np.sum(np.abs(valid) <= DELTA_EPS))
            losses = int(np.sum(valid < -DELTA_EPS))
            wtl_rows.append(
                {
                    "dataset": dataset,
                    "focal": focal,
                    "baseline": baseline,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "mean_delta": float(np.mean(valid)),
                }
            )
    wtl = pd.DataFrame(wtl_rows)
    if not wtl.empty and not out.empty:
        return out, wtl
    return out, wtl


def build_validation_test_agreement(df, oracle_df) -> Any:
    import pandas as pd

    rows = []
    max_sub = df[df["method"] == METHOD_VAL_MAX][["dataset", "seed", "K_selected"]].rename(
        columns={"K_selected": "K_max"}
    )
    one_sub = df[df["method"] == METHOD_VAL_1SE][["dataset", "seed", "K_selected"]].rename(
        columns={"K_selected": "K_1se"}
    )
    merged = max_sub.merge(one_sub, on=["dataset", "seed"], how="outer")
    oracle_best = (
        oracle_df.groupby(["dataset", "seed"])["oracle_k"].first().reset_index()
        if not oracle_df.empty
        else pd.DataFrame(columns=["dataset", "seed", "oracle_k"])
    )
    merged = merged.merge(oracle_best, on=["dataset", "seed"], how="left")
    for _, r in merged.iterrows():
        rows.append(
            {
                "dataset": r["dataset"],
                "seed": int(r["seed"]),
                "K_max": r["K_max"],
                "K_1se": r["K_1se"],
                "max_agrees_1se": int(r["K_max"] == r["K_1se"])
                if np.isfinite(r.get("K_max", float("nan")))
                and np.isfinite(r.get("K_1se", float("nan")))
                else 0,
                "oracle_k": r.get("oracle_k", float("nan")),
                "max_agrees_oracle": int(r["K_max"] == r["oracle_k"])
                if np.isfinite(r.get("K_max", float("nan")))
                and np.isfinite(r.get("oracle_k", float("nan")))
                else 0,
                "1se_agrees_oracle": int(r["K_1se"] == r["oracle_k"])
                if np.isfinite(r.get("K_1se", float("nan")))
                and np.isfinite(r.get("oracle_k", float("nan")))
                else 0,
            }
        )
    return pd.DataFrame(rows)


def spearman_k_vs_f1(df) -> dict[str, float]:
    from scipy.stats import spearmanr

    out: dict[str, float] = {}
    sub = df[df["method"] != METHOD_FULL_RHO90].copy()
    sub = sub[np.isfinite(sub["K_selected"]) & np.isfinite(sub["macro_f1"])]
    if len(sub) >= 3:
        rho, p = spearmanr(sub["K_selected"], sub["macro_f1"])
        out["K_selected_vs_macro_f1_rho"] = float(rho)
        out["K_selected_vs_macro_f1_p"] = float(p)
    balls_sub = sub[np.isfinite(sub["n_balls_final"])]
    if len(balls_sub) >= 3:
        rho, p = spearmanr(balls_sub["n_balls_final"], balls_sub["macro_f1"])
        out["n_balls_final_vs_macro_f1_rho"] = float(rho)
        out["n_balls_final_vs_macro_f1_p"] = float(p)
    return out
