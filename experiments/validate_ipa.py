"""
IPA classifier screening: CW vs CA vs IPA on frozen 70/15/15 split.

Run AFTER analyze_ipa.py diagnostics.

Run from GBLSTSVM root:
  python experiments/validate_ipa.py
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
from src.ipa_gblstsvm import IPAGBLSTSVM  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "ipa"
DIAG_JSON = OUT_DIR / "ipa_diagnostics_report.json"
REPORT_PATH = OUT_DIR / "ipa_validation_report.md"
CSV_PATH = OUT_DIR / "ipa_validation_results.csv"
JSON_PATH = OUT_DIR / "ipa_validation_results.json"

SEED = 42
PURITY = DEFAULT_PURITY
MIN_BALL_COUNT = DEFAULT_MIN_BALL_COUNT
WEIGHT_PLACEMENT = "opposite_only"
PREDICTION_RULE = "reference"

MODELS = {
    "CW": CWGBLSTSVM,
    "CA": CAGBLSTSVM,
    "IPA": IPAGBLSTSVM,
}

METRIC_KEYS = (
    "macro_f1",
    "minority_f1",
    "gmean",
    "balanced_accuracy",
    "minority_recall",
    "auroc",
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


def _interpret(
    ds: str,
    cw: dict,
    ca: dict,
    ipa: dict,
    noise: dict,
) -> dict:
    ipa_vs_ca = _delta(ipa, ca)
    ipa_vs_cw = _delta(ipa, cw)
    ca_vs_cw = _delta(ca, cw)

    amb_pct = noise.get("noise_ball_ambiguity_percentile", float("nan"))
    ipa_pct = noise.get("noise_ball_mu_ipa_percentile", float("nan"))
    noise_reduced = (
        np.isfinite(amb_pct)
        and np.isfinite(ipa_pct)
        and ipa_pct < amb_pct
    )

    # vehicle0-style success criteria (applied uniformly, not tuned)
    retains_ca = (
        ipa.get("gmean", 0) >= ca.get("gmean", 0) - 0.02
        and ipa.get("minority_f1", 0) >= ca.get("minority_f1", 0) - 0.03
    )
    beats_cw = ipa.get("gmean", 0) > cw.get("gmean", 0) + 0.01

    if retains_ca and beats_cw and noise_reduced:
        verdict = "SUCCESS"
    elif noise_reduced and (ipa_vs_ca.get("gmean") or 0) < -0.03:
        verdict = "PARTIAL"
    elif ipa_vs_ca.get("gmean") is not None and ipa_vs_ca["gmean"] >= -0.01 and beats_cw:
        verdict = "SUCCESS"
    elif noise_reduced and not retains_ca:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "ipa_vs_ca": ipa_vs_ca,
        "ipa_vs_cw": ipa_vs_cw,
        "ca_vs_cw": ca_vs_cw,
        "noise_ambiguity_percentile": amb_pct,
        "noise_muIPA_percentile": ipa_pct,
        "noise_percentile_decreased": noise_reduced,
        "retains_ca_gain": retains_ca,
        "beats_cw": beats_cw,
    }


def run_screening() -> dict:
    diag = {}
    if DIAG_JSON.exists():
        diag = json.loads(DIAG_JSON.read_text(encoding="utf-8")).get("datasets", {})

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

        noise = diag.get(ds, {}).get("synthetic_noise", {})
        interpretations[ds] = _interpret(ds, results["CW"], results["CA"], results["IPA"], noise)

    # Global decision
    verdicts = [v["verdict"] for v in interpretations.values()]
    vehicle = interpretations.get("vehicle0", {})
    if verdicts.count("SUCCESS") >= 1 and vehicle.get("verdict") in ("SUCCESS",):
        global_verdict = "PASS"
        recommendation = (
            "Promising controlled refinement: IPA retains CA gain on vehicle0 while "
            "reducing noise-ball membership. Proceed to multi-seed/full CV on vehicle0."
        )
    elif "PARTIAL" in verdicts and vehicle.get("retains_ca_gain"):
        global_verdict = "PARTIAL"
        recommendation = (
            "Isolation penalty reduces noise reward but may erode CA performance. "
            "Do not proceed to full CV; abandon or accept CA without penalty."
        )
    elif vehicle.get("beats_cw") and vehicle.get("verdict") == "SUCCESS":
        global_verdict = "PASS"
        recommendation = "IPA shows vehicle0 promise; proceed to multi-seed CV."
    else:
        global_verdict = "FAIL"
        recommendation = (
            "IPA does not improve on CA/CW consistently. Abandon competition-membership family."
        )

    return {
        "global_verdict": global_verdict,
        "recommendation": recommendation,
        "per_dataset": interpretations,
        "results": all_rows,
    }


def _write_report(payload: dict) -> None:
    lines = [
        "# IPA Validation Report",
        "",
        f"**Global verdict: {payload['global_verdict']}**",
        "",
        payload["recommendation"],
        "",
        "## Configuration",
        f"- Split: stratified 70/15/15, seed={SEED}",
        f"- Models: CW, CA (parabolic muA), IPA (lambda=1)",
        "",
    ]

    rows_by_ds: dict[str, list] = {}
    for row in payload["results"]:
        rows_by_ds.setdefault(row["dataset"], []).append(row)

    for ds, rows in rows_by_ds.items():
        interp = payload["per_dataset"][ds]
        lines.extend([f"## {ds}", "", f"**Verdict: {interp['verdict']}**", ""])
        lines.append("| Model | G-mean | Minority F1 | Macro F1 | Balanced Acc | Minority Rec | AUC |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['model']} | {row['gmean']:.4f} | {row['minority_f1']:.4f} | "
                f"{row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} | "
                f"{row['minority_recall']:.4f} | {row.get('auroc', 'n/a')} |"
            )
        lines.append("")
        lines.append("### IPA vs CA / CW")
        for label, d in [("IPA vs CA", interp["ipa_vs_ca"]), ("IPA vs CW", interp["ipa_vs_cw"])]:
            lines.append(
                f"- {label}: d_gmean={d.get('gmean', float('nan')):+.4f}, "
                f"d_minority_f1={d.get('minority_f1', float('nan')):+.4f}"
            )
        lines.append(
            f"- Noise ambiguity pct={interp['noise_ambiguity_percentile']:.1f}, "
            f"muIPA pct={interp['noise_muIPA_percentile']:.1f}, "
            f"decreased={interp['noise_percentile_decreased']}"
        )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run_screening()
    _write_report(payload)
    pd.DataFrame(payload["results"]).to_csv(CSV_PATH, index=False)
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nGlobal verdict: {payload['global_verdict']}")
    print(payload["recommendation"])
    print(f"\nReport: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
