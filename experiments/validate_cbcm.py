"""
CBCM 5-arm evaluation (run only after diagnostic gate passes).

Arms:
  1. GBLSTSVM (vanilla) — from frozen Phase 7 reference where available
  2. CW-GBLSTSVM — frozen reference
  3. CA-GBLSTSVM (competition)
  4. CB-GBLSTSVM (connectivity)
  5. CC-GBLSTSVM (combined)

Uses stratified 5-fold CV on full dataset (train protocol aligned with Phase 7
hyperparameters: c1=100, c2=1e-2, purity=0.95, no scaling).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase8_common import (  # noqa: E402
    DATASETS,
    DEFAULT_C1,
    DEFAULT_C2,
    DEFAULT_MIN_BALL_COUNT,
    DEFAULT_PURITY,
    evaluate,
    inspect_dataset,
    map_to_binary,
    normalize_labels,
    twin_scores,
)
from src.cbcm_gblstsvm import CAGBLSTSVM, CBGBLSTSVM, CCGBLSTSVM  # noqa: E402
from src.cw_gblstsvm import CWGBLSTSVM  # noqa: E402
from src.gblstsvm import GBLSTSVM  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "cbcm"
DIAG_JSON = OUT_DIR / "cbcm_diagnostics_report.json"
REPORT_PATH = OUT_DIR / "cbcm_evaluation_report.md"
CSV_PATH = OUT_DIR / "cbcm_evaluation_results.csv"

N_FOLDS = 5
SEED = 42
PREDICTION_RULE = "reference"
WEIGHT_PLACEMENT = "opposite_only"

# Frozen Phase 7 test-set reference (single split, seed=42).
FROZEN_REFERENCE = {
    "diabetes": {
        "GBLSTSVM": {
            "accuracy": 0.646552,
            "gmean": 0.0,
            "minority_f1": 0.392670,
            "majority_f1": 0.500000,
            "auroc": 0.498211,
        },
        "CW-GBLSTSVM": {
            "accuracy": 0.646552,
            "gmean": 0.0,
            "minority_f1": 0.392670,
            "majority_f1": 0.500000,
            "auroc": 0.505366,
        },
    },
    "ecoli2": {
        "GBLSTSVM": {
            "accuracy": 0.901961,
            "gmean": 0.771965,
            "minority_f1": 0.804598,
            "majority_f1": 0.789244,
            "auroc": 0.933140,
        },
        "CW-GBLSTSVM": {
            "accuracy": 0.901961,
            "gmean": 0.771965,
            "minority_f1": 0.804598,
            "majority_f1": 0.789244,
            "auroc": 0.930233,
        },
    },
    "vehicle0": {
        "GBLSTSVM": {
            "accuracy": 0.818898,
            "gmean": 0.568352,
            "minority_f1": 0.678056,
            "majority_f1": 0.651203,
            "auroc": 0.910309,
        },
        "CW-GBLSTSVM": {
            "accuracy": 0.811024,
            "gmean": 0.612583,
            "minority_f1": 0.691748,
            "majority_f1": 0.669072,
            "auroc": 0.924399,
        },
    },
}

CBCM_MODELS = {
    "GBLSTSVM": GBLSTSVM,
    "CW-GBLSTSVM": CWGBLSTSVM,
    "CA-GBLSTSVM": CAGBLSTSVM,
    "CB-GBLSTSVM": CBGBLSTSVM,
    "CC-GBLSTSVM": CCGBLSTSVM,
}

METRICS = ("accuracy", "gmean", "minority_f1", "majority_f1", "auroc")


def _class_f1(y_true, y_pred, label: float) -> float:
    return float(
        f1_score(y_true, y_pred, pos_label=label, zero_division=0)
    )


def _fold_metrics(model, X_test, y_test, minority_label: float) -> dict:
    y_pred = model.predict(X_test)
    scores = twin_scores(model, X_test)
    base = evaluate(y_test, y_pred, scores, minority_label)
    y_true = normalize_labels(y_test)
    y_pred_n = y_pred.copy()
    y_pred_n[y_pred_n == 0] = -1.0
    majority_label = -minority_label
    return {
        "accuracy": base["accuracy"],
        "gmean": base["gmean"],
        "minority_f1": _class_f1(y_true, y_pred_n, minority_label),
        "majority_f1": _class_f1(y_true, y_pred_n, majority_label),
        "auroc": base["auroc"],
    }


def _fit_model(name: str, X_train, y_train):
    params = {
        "c1": DEFAULT_C1,
        "c2": DEFAULT_C2,
        "purity_threshold": DEFAULT_PURITY,
        "min_ball_count": DEFAULT_MIN_BALL_COUNT,
        "prediction_rule": PREDICTION_RULE,
        "random_state": SEED,
    }
    cls = CBCM_MODELS[name]
    if name in ("CW-GBLSTSVM",):
        return cls(**params, weight_placement=WEIGHT_PLACEMENT).fit(X_train, y_train)
    if name in ("CA-GBLSTSVM", "CB-GBLSTSVM", "CC-GBLSTSVM"):
        return cls(**params, weight_placement=WEIGHT_PLACEMENT).fit(X_train, y_train)
    return cls(**params).fit(X_train, y_train)


def _aggregate_fold_metrics(fold_records: list[dict]) -> dict:
    out = {}
    for m in METRICS:
        vals = [r[m] for r in fold_records if r.get(m) is not None]
        if not vals:
            out[f"{m}_mean"] = None
            out[f"{m}_std"] = None
        else:
            arr = np.array(vals, dtype=float)
            out[f"{m}_mean"] = float(np.mean(arr))
            out[f"{m}_std"] = float(np.std(arr))
    return out


def run_cv_evaluation(dataset_file: str) -> dict:
    path = ROOT / "Data" / "Data" / dataset_file
    X, y_raw, info = inspect_dataset(path)
    y, _ = map_to_binary(y_raw, info)
    y_n = normalize_labels(y)
    minority_label = 1.0 if int(np.sum(y_n == 1.0)) <= int(np.sum(y_n == -1.0)) else -1.0

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results: dict[str, list[dict]] = {name: [] for name in CBCM_MODELS}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_n)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_n[train_idx], y_n[test_idx]
        for model_name in CBCM_MODELS:
            try:
                model = _fit_model(model_name, X_train, y_train)
                m = _fold_metrics(model, X_test, y_test, minority_label)
                m["fold"] = fold_idx
                results[model_name].append(m)
            except Exception as exc:  # noqa: BLE001
                results[model_name].append(
                    {"fold": fold_idx, "error": str(exc)}
                )

    aggregated = {
        name: _aggregate_fold_metrics(recs)
        for name, recs in results.items()
    }

    # Wilcoxon: CC vs CW on gmean and minority_f1.
    sig = {}
    for metric in ("gmean", "minority_f1"):
        cc_vals = [r[metric] for r in results["CC-GBLSTSVM"] if metric in r]
        cw_vals = [r[metric] for r in results["CW-GBLSTSVM"] if metric in r]
        if len(cc_vals) == len(cw_vals) and len(cc_vals) >= 2:
            try:
                stat, p = wilcoxon(cc_vals, cw_vals)
                sig[metric] = {"statistic": float(stat), "p_value": float(p)}
            except ValueError:
                sig[metric] = {"statistic": None, "p_value": None, "note": "wilcoxon failed"}
        else:
            sig[metric] = {"statistic": None, "p_value": None}

    ds_name = info["name"]
    return {
        "dataset": ds_name,
        "n_folds": N_FOLDS,
        "fold_results": results,
        "aggregated": aggregated,
        "significance_cc_vs_cw": sig,
        "frozen_reference_test": FROZEN_REFERENCE.get(ds_name, {}),
    }


def _write_report(all_eval: dict, gate_pass: bool, excluded: list[str]) -> None:
    lines = [
        "# CBCM Evaluation Report",
        "",
        f"**Diagnostic gate:** {'PASS' if gate_pass else 'FAIL'}",
        "",
    ]
    if excluded:
        lines.append(f"**Excluded datasets (gate fail):** {', '.join(excluded)}")
        lines.append("")

    lines.extend(
        [
            "## Protocol",
            f"- {N_FOLDS}-fold stratified CV, seed={SEED}",
            f"- c1={DEFAULT_C1}, c2={DEFAULT_C2}, purity={DEFAULT_PURITY}",
            "- No per-dataset tuning of k, alpha, beta, or normalization",
            "",
        ]
    )

    for ds, ev in all_eval.items():
        lines.extend([f"## {ds}", "", "### 5-fold CV (mean ± std)", ""])
        lines.append(
            "| Model | Accuracy | G-mean | Minority F1 | Majority F1 | AUC |"
        )
        lines.append("|-------|----------|--------|-------------|-------------|-----|")
        for model_name in CBCM_MODELS:
            agg = ev["aggregated"][model_name]
            row = [model_name]
            for m in METRICS:
                mean = agg.get(f"{m}_mean")
                std = agg.get(f"{m}_std")
                if mean is None:
                    row.append("n/a")
                else:
                    row.append(f"{mean:.4f} ± {std:.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        sig = ev["significance_cc_vs_cw"]
        lines.append("### Significance (CC vs CW, Wilcoxon across folds)")
        for metric in ("gmean", "minority_f1"):
            s = sig.get(metric, {})
            lines.append(
                f"- **{metric}**: stat={s.get('statistic')}, p={s.get('p_value')}"
            )
        lines.append("")
        ref = ev.get("frozen_reference_test", {})
        if ref:
            lines.append("### Frozen Phase 7 reference (70/15/15 test, seed=42)")
            for model in ("GBLSTSVM", "CW-GBLSTSVM"):
                if model in ref:
                    r = ref[model]
                    lines.append(
                        f"- {model}: acc={r['accuracy']:.4f}, "
                        f"gmean={r['gmean']:.4f}, minF1={r['minority_f1']:.4f}, "
                        f"AUC={r['auroc']:.4f}"
                    )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not DIAG_JSON.exists():
        print("ERROR: Run cbcm_diagnostics.py first.")
        return 2

    diag = json.loads(DIAG_JSON.read_text(encoding="utf-8"))
    gate_pass = diag.get("global_gate_pass", False)
    excluded = [
        ds
        for ds, res in diag.get("datasets", {}).items()
        if not res.get("stop_conditions", {}).get("dataset_gate_pass", False)
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_eval: dict = {}
    rows: list[dict] = []

    for ds_file in DATASETS:
        ds_stem = ds_file.replace(".csv", "")
        if ds_stem in excluded:
            print(f"Skipping {ds_stem} (diagnostic gate FAIL)")
            continue
        print(f"Evaluating {ds_stem}...")
        ev = run_cv_evaluation(ds_file)
        all_eval[ds_stem] = ev
        for model_name, fold_recs in ev["fold_results"].items():
            for rec in fold_recs:
                row = {
                    "dataset": ds_stem,
                    "model": model_name,
                    "fold": rec.get("fold"),
                }
                for m in METRICS:
                    row[m] = rec.get(m)
                rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    payload = {
        "gate_pass": gate_pass,
        "excluded_datasets": excluded,
        "n_folds": N_FOLDS,
        "evaluations": all_eval,
    }
    (OUT_DIR / "cbcm_evaluation_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    _write_report(all_eval, gate_pass, excluded)

    print(f"\nEvaluation report: {REPORT_PATH}")
    if not gate_pass:
        print("WARNING: Diagnostic gate did not pass globally; results flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
