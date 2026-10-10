"""Probability metrics for a validation period (proposal sections 14 and 15).

Every metric is compared with the base-rate reference: always predicting the
positive share of the fold's *training* rows (known before the validation period
starts). A model has learned something only if it beats that reference.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

CALIBRATION_BINS = 10


def calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = CALIBRATION_BINS) -> float:
    """Expected calibration error with equal-count bins.

    Rows are sorted by predicted probability and split into ``bins`` groups of
    (nearly) equal size; the error is the size-weighted mean of
    |mean predicted probability - observed positive share| over the groups.
    """
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(probabilities, kind="mergesort")
    error = 0.0
    for group in np.array_split(order, bins):
        if len(group):
            error += len(group) * abs(probabilities[group].mean() - labels[group].mean())
    return error / len(labels)


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray, base_rate: float) -> dict[str, float]:
    """AUC, log-loss, Brier score and calibration error, with the base-rate reference.

    ``*_skill`` compares with the reference: 1 - model / reference, so 0 means no
    better than always predicting ``base_rate`` and anything above 0 is an edge.
    """
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    reference = np.full_like(probabilities, base_rate)
    model_log_loss = log_loss(labels, probabilities, labels=[0, 1])
    reference_log_loss = log_loss(labels, reference, labels=[0, 1])
    model_brier = brier_score_loss(labels, probabilities)
    reference_brier = brier_score_loss(labels, reference)
    return {
        "rows": int(len(labels)),
        "positive_share": float(labels.mean()),
        "base_rate": float(base_rate),
        "auc": float(roc_auc_score(labels, probabilities)) if 0 < labels.mean() < 1 else float("nan"),
        "log_loss": float(model_log_loss),
        "log_loss_reference": float(reference_log_loss),
        "log_loss_skill": float(1 - model_log_loss / reference_log_loss),
        "brier": float(model_brier),
        "brier_reference": float(reference_brier),
        "brier_skill": float(1 - model_brier / reference_brier),
        "calibration_error": float(calibration_error(labels, probabilities)),
        "mean_probability": float(probabilities.mean()),
    }
