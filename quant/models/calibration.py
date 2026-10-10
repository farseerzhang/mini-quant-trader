"""Probability calibration (proposal section 14).

Raw model scores are mapped to probabilities of the defined target on rows the
model was not fit on. A calibrated 0.6 means that about 60% of such rows met the
target in the calibration data; it is not a 60% chance of making money.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPSILON = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPSILON, 1 - _EPSILON)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """``isotonic``: a monotone step function; ``sigmoid`` (Platt): a logistic curve on the
    raw score's log-odds."""

    def __init__(self, method: str) -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError("calibration method must be 'isotonic' or 'sigmoid'")
        self.method = method
        self._model = None

    def fit(self, raw: np.ndarray, labels: np.ndarray) -> "Calibrator":
        raw, labels = np.asarray(raw, dtype=float), np.asarray(labels, dtype=float)
        if self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, labels)
        else:
            self._model = LogisticRegression().fit(_logit(raw), labels)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("calibrator is not fitted")
        raw = np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            return self._model.predict(raw)
        return self._model.predict_proba(_logit(raw))[:, 1]
