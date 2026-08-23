"""
Per-sample twin-hyperplane diagnostics for paper vs reference prediction rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.gblstsvm import predict_twin_hyperplanes

PredictionRule = Literal["paper", "reference"]


@dataclass
class PredictionDiagnostics:
    sample_id: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    norm_w1: float
    norm_w2: float
    d1: np.ndarray
    d2: np.ndarray
    reference_prediction: np.ndarray
    paper_prediction: np.ndarray
    disagreement: np.ndarray


def compute_prediction_diagnostics(
    X: np.ndarray,
    w1: np.ndarray,
    b1: float,
    w2: np.ndarray,
    b2: float,
) -> PredictionDiagnostics:
    """Compute f_i, normalized distances, and both prediction rules per sample."""
    X = np.asarray(X, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)

    f1 = X @ w1 + b1
    f2 = X @ w2 + b2
    norm_w1 = float(np.linalg.norm(w1))
    norm_w2 = float(np.linalg.norm(w2))

    d1 = np.abs(f1) / (norm_w1 if norm_w1 > 0 else 1.0)
    d2 = np.abs(f2) / (norm_w2 if norm_w2 > 0 else 1.0)

    reference_prediction = predict_twin_hyperplanes(
        X, w1, b1, w2, b2, rule="reference"
    )
    paper_prediction = predict_twin_hyperplanes(
        X, w1, b1, w2, b2, rule="paper"
    )
    disagreement = reference_prediction != paper_prediction

    return PredictionDiagnostics(
        sample_id=np.arange(X.shape[0]),
        f1=f1,
        f2=f2,
        norm_w1=norm_w1,
        norm_w2=norm_w2,
        d1=d1,
        d2=d2,
        reference_prediction=reference_prediction,
        paper_prediction=paper_prediction,
        disagreement=disagreement,
    )


def diagnostics_to_records(diag: PredictionDiagnostics) -> list[dict]:
    """Flatten diagnostics to row dicts for CSV export."""
    rows = []
    for i in range(len(diag.sample_id)):
        rows.append(
            {
                "sample_id": int(diag.sample_id[i]),
                "f1": float(diag.f1[i]),
                "f2": float(diag.f2[i]),
                "norm_w1": diag.norm_w1,
                "norm_w2": diag.norm_w2,
                "d1": float(diag.d1[i]),
                "d2": float(diag.d2[i]),
                "reference_prediction": float(diag.reference_prediction[i]),
                "paper_prediction": float(diag.paper_prediction[i]),
                "disagreement": bool(diag.disagreement[i]),
            }
        )
    return rows


def summarize_disagreements(diag: PredictionDiagnostics) -> dict:
    """Summary statistics for paper vs reference prediction differences."""
    n = len(diag.sample_id)
    disagree_mask = diag.disagreement
    n_disagree = int(np.sum(disagree_mask))

    abs_f1 = np.abs(diag.f1)
    abs_f2 = np.abs(diag.f2)
    ratio_w = diag.norm_w1 / diag.norm_w2 if diag.norm_w2 > 0 else np.inf

    # Reference +1 when |f2| > |f1|; paper +1 when d1 <= d2.
    ref_prefers_pos = abs_f2 > abs_f1
    paper_prefers_pos = diag.d1 <= diag.d2

    disagree_due_to_norm_only = np.sum(
        disagree_mask & (ref_prefers_pos != paper_prefers_pos)
    )

    return {
        "n_samples": n,
        "n_disagree": n_disagree,
        "disagree_rate": n_disagree / n if n else 0.0,
        "norm_w1": diag.norm_w1,
        "norm_w2": diag.norm_w2,
        "norm_ratio_w1_over_w2": ratio_w,
        "n_reference_zero": int(np.sum(diag.reference_prediction == 0)),
        "n_disagree_where_abs_equal": int(
            np.sum(disagree_mask & np.isclose(abs_f1, abs_f2))
        ),
        "n_disagree_where_abs_unequal": int(
            np.sum(disagree_mask & ~np.isclose(abs_f1, abs_f2))
        ),
    }
