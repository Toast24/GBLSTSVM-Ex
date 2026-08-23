"""
Cheap classifier screen: CW vs CA vs Topo on frozen 70/15/15 split.

Run AFTER analyze_topology.py if diagnostics are sensible.

Run from GBLSTSVM root:
  python experiments/validate_topology.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

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
    load_dataset_split,
    normalize_labels,
    twin_scores,
)
from src.cbcm_gblstsvm import CAGBLSTSVM  # noqa: E402
from src.cw_gblstsvm import CWGBLSTSVM  # noqa: E402
from src.topology_gblstsvm import TopoGBLSTSVM  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "topology"
DIAG_JSON = OUT_DIR / "topology_diagnostics_report.json"
REPORT_PATH = OUT_DIR / "topology_validation_report.md"
CSV_PATH = OUT_DIR / "topology_validation_results.csv"
JSON_PATH = OUT_DIR / "topology_validation_results.json"

SEED = 42
PURITY = DEFAULT_PURITY
MIN_BALL_COUNT = DEFAULT_MIN_BALL_COUNT
WEIGHT_PLACEMENT = "opposite_only"
PREDICTION_RULE = "reference"

MODELS = {
    "CW": CWGBLSTSVM,
    "CA": CAGBLSTSVM,
    "Topo": TopoGBLSTSVM,
}

METRIC_KEYS = (
    "gmean",
    "minority_f1",
    "auroc",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "minority_recall",
)


def _class_f1(y_true, y_pred, label: float) -> float:
    return float(
        f1_score(normalize_labels(y_true), normalize_labels(y_pred), pos_label=label, zero_division=0)
    )


def _fit_model(name: str, X_train, y_train):
    params = {
        "c1": DEFAULT_C1,
        "c2": DEFAULT_C2,
        "purity_threshold": PURITY,
        "min_ball_count": MIN_BALL_COUNT,
        "prediction_rule": PREDICTION_RULE,
        "random_state": SEED,
        "weight_placement": WEIGHT_PLACEMENT,
    }
    return MODELS[name](**params).fit(X_train, normalize_labels(y_train))


def _eval_model(model, X, y, minority_label: float) -> dict:
    y_pred = model.predict(X)
    scores = twin_scores(model, X)
    m = evaluate(y, y_pred, scores, minority_label)
    m["minority_f1"] = _class_f1(y, y_pred, minority_label)
    return {k: m[k] for k in METRIC_KEYS}


def _delta(a: dict, b: dict) -> dict:
    return {
        k: (a[k] - b[k]) if a.get(k) is not None and b.get(k) is not None else None
        for k in METRIC_KEYS
    }


def _r_opp_informative(diag_ds: dict) -> bool:
    corr = diag_ds.get("correlations", {}).get("R_opp_vs_A", {})
    pearson = corr.get("pearson", float("nan"))
    r_frac = diag_ds.get("fractions", {}).get("R_opp_gt_0", 0.0)
    r_std = diag_ds.get("distributions", {}).get("R_opp", {}).get("std", 0.0)
    if r_frac <= 0 or r_std == 0:
        return False
    if np.isfinite(pearson) and abs(pearson) >= 0.99:
        return False
    return True


def _interpret(ds: str, cw: dict, ca: dict, topo: dict, diag_ds: dict) -> dict:
    noise = diag_ds.get("synthetic_noise", {})
    a_pct = noise.get("noise_ball_A_percentile", float("nan"))
    mu_pct = noise.get("noise_ball_muTopo_percentile", float("nan"))
    noise_not_worse = (
        np.isfinite(a_pct) and np.isfinite(mu_pct) and mu_pct <= a_pct + 1e-9
    )
    noise_improved = (
        np.isfinite(a_pct) and np.isfinite(mu_pct) and mu_pct < a_pct
    )

    retains_ca = (
        topo.get("gmean", 0) >= ca.get("gmean", 0) - 0.02
        and topo.get("minority_f1", 0) >= ca.get("minority_f1", 0) - 0.03
    )
    loses_ca_like_ipa = (topo.get("gmean", 0) - ca.get("gmean", 0)) <= -0.015
    beats_cw = topo.get("gmean", 0) > cw.get("gmean", 0) + 0.01
    r_info = _r_opp_informative(diag_ds)

    return {
        "topo_vs_ca": _delta(topo, ca),
        "topo_vs_cw": _delta(topo, cw),
        "ca_vs_cw": _delta(ca, cw),
        "noise_A_percentile": a_pct,
        "noise_muTopo_percentile": mu_pct,
        "noise_not_worse": noise_not_worse,
        "noise_improved": noise_improved,
        "retains_ca_gain": retains_ca,
        "loses_ca_like_ipa": loses_ca_like_ipa,
        "beats_cw": beats_cw,
        "r_opp_informative": r_info,
    }


def _global_decision(interpretations: dict[str, dict]) -> tuple[str, str]:
    vehicle = interpretations.get("vehicle0", {})
    other = {k: v for k, v in interpretations.items() if k != "vehicle0"}

    if vehicle.get("loses_ca_like_ipa") and not vehicle.get("retains_ca_gain"):
        return (
            "ABANDON",
            "Topo loses the CA vehicle0 gain similarly to IPA/SP. "
            "Abandon the topology-competition family. Do not run full CV.",
        )

    promote = (
        vehicle.get("retains_ca_gain")
        and vehicle.get("noise_not_worse")
        and vehicle.get("r_opp_informative")
        and vehicle.get("beats_cw")
    )

    others_ok = all(
        v.get("beats_cw") or (v.get("topo_vs_cw") or {}).get("gmean", 0) >= -0.02
        for v in other.values()
    ) if other else True
    others_unstable = any(
        (v.get("topo_vs_ca") or {}).get("gmean", 0) is not None
        and (v.get("topo_vs_ca")["gmean"] < -0.05 or v.get("topo_vs_cw", {}).get("gmean", 0) < -0.05)
        for v in other.values()
    )

    if promote and others_ok and not others_unstable:
        return (
            "PROMOTE",
            "Cheap screen supports Topo: vehicle0 CA gain largely preserved, "
            "noise membership not worsened, R_opp not redundant. "
            "Recommend a separate multi-seed / full-CV validation stage. "
            "Do not tune k or the formula.",
        )
    if vehicle.get("retains_ca_gain") and vehicle.get("beats_cw"):
        return (
            "LOCALIZED",
            "Topo improves or preserves vehicle0 but remains unstable or unproven elsewhere. "
            "Report as localized evidence. Do not immediately run full CV.",
        )
    return (
        "ABANDON",
        "Cheap screen does not support promotion. Do not run multi-seed/full CV.",
    )


def run_screening() -> dict:
    if not DIAG_JSON.exists():
        return {
            "global_verdict": "SKIPPED",
            "recommendation": "Run experiments/analyze_topology.py first.",
            "per_dataset": {},
            "results": [],
            "skipped": True,
        }

    diag_payload = json.loads(DIAG_JSON.read_text(encoding="utf-8"))
    if not diag_payload.get("sensible", False):
        return {
            "global_verdict": "SKIPPED",
            "recommendation": (
                "Diagnostics not sensible (R_opp degenerate or unused). "
                "Cheap classifier screen was not run."
            ),
            "per_dataset": {},
            "results": [],
            "skipped": True,
            "diagnostics": diag_payload,
        }

    diag = diag_payload.get("datasets", {})
    all_rows: list[dict] = []
    interpretations: dict[str, dict] = {}

    for ds_file in DATASETS:
        split = load_dataset_split(ds_file, seed=SEED)
        ds = split.name
        results: dict[str, dict] = {}
        for name in MODELS:
            model = _fit_model(name, split.X_train, split.y_train)
            results[name] = _eval_model(model, split.X_test, split.y_test, split.minority_label)
            all_rows.append({"dataset": ds, "model": name, **results[name]})

        interpretations[ds] = _interpret(
            ds, results["CW"], results["CA"], results["Topo"], diag.get(ds, {})
        )

    global_verdict, recommendation = _global_decision(interpretations)
    return {
        "global_verdict": global_verdict,
        "recommendation": recommendation,
        "per_dataset": interpretations,
        "results": all_rows,
        "skipped": False,
    }


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(payload: dict) -> None:
    lines = [
        "# Topology Cheap-Screen Evaluation Report",
        "",
        f"**Global verdict: {payload['global_verdict']}**",
        "",
        payload["recommendation"],
        "",
        "## Configuration",
        f"- Split: stratified 70/15/15, seed={SEED}",
        "- Models: CW, CA (frozen parabolic muA), Topo (A2 * R_opp)",
        "- No hyperparameter tuning after seeing results.",
        "",
    ]

    if payload.get("skipped"):
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        return

    rows_by_ds: dict[str, list] = {}
    for row in payload["results"]:
        rows_by_ds.setdefault(row["dataset"], []).append(row)

    for ds, rows in rows_by_ds.items():
        interp = payload["per_dataset"][ds]
        lines.extend([f"## {ds}", ""])
        lines.append("| Model | G-mean | Minority F1 | AUC | Balanced Acc | Accuracy |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['model']} | {_fmt(row['gmean'])} | {_fmt(row['minority_f1'])} | "
                f"{_fmt(row.get('auroc'))} | {_fmt(row['balanced_accuracy'])} | "
                f"{_fmt(row['accuracy'])} |"
            )
        lines.append("")
        for label, d in [("Topo vs CA", interp["topo_vs_ca"]), ("Topo vs CW", interp["topo_vs_cw"])]:
            lines.append(
                f"- {label}: d_gmean={_fmt(d.get('gmean'))}, "
                f"d_minority_f1={_fmt(d.get('minority_f1'))}"
            )
        lines.append(
            f"- Noise A pct={interp['noise_A_percentile']:.1f}, "
            f"muTopo pct={interp['noise_muTopo_percentile']:.1f}, "
            f"not_worse={interp['noise_not_worse']}, improved={interp['noise_improved']}"
        )
        lines.append(
            f"- retains_ca_gain={interp['retains_ca_gain']}, "
            f"r_opp_informative={interp['r_opp_informative']}"
        )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run_screening()
    _write_report(payload)
    if payload.get("results"):
        pd.DataFrame(payload["results"]).to_csv(CSV_PATH, index=False)
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nGlobal verdict: {payload['global_verdict']}")
    print(payload["recommendation"])
    print(f"\nReport: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
