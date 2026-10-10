"""PREDICTOR model types (proposal section 14). XGBoost arrives in Phase 6."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Settings that are not searched (retrain.search.space holds the searched ones).
FIXED_PARAMETERS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"max_iter": 2000},
    "random_forest": {"n_estimators": 300, "max_features": "sqrt"},
}


def build_model(model_type: str, parameters: Mapping[str, Any], seed: int):
    """An unfitted classifier with ``predict_proba``.

    Logistic Regression scales its inputs inside the pipeline, so the scaler is
    fit on each fold's training rows only. Neither model weights classes: the
    calibration step maps scores to probabilities of the defined target.
    """
    if model_type == "logistic_regression":
        model = LogisticRegression(random_state=seed, **FIXED_PARAMETERS[model_type], **parameters)
        return Pipeline([("scale", StandardScaler()), ("model", model)])
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **FIXED_PARAMETERS[model_type], **parameters)
    raise NotImplementedError(f"model type {model_type!r} is not implemented yet")
