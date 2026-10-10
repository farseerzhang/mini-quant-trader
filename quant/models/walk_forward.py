"""Train, calibrate and score a predictor on every walk-forward fold (proposal section 15).

Per fold, the training window is split into a fit part and a calibration part
(the last ``model.calibration_months``), with ``embargo_days`` sessions purged
between them. The model is fit on every ``retrain.sample_every_sessions``-th
session of the fit part; the calibrator on every session of the calibration
part; the fold is then scored on every session of its validation window. The
validation rows of all folds together are the out-of-sample predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant.config.settings import Settings
from quant.models.calibration import Calibrator
from quant.models.dataset import INPUT_COLUMNS, Dataset
from quant.models.evaluation import probability_metrics
from quant.models.folds import Fold, sample_sessions
from quant.models.predictors import build_model


@dataclass(frozen=True)
class FoldSplit:
    fit: pd.DatetimeIndex
    calibration: pd.DatetimeIndex


@dataclass(frozen=True)
class WalkForwardResult:
    model_type: str
    parameters: dict[str, Any]
    # One row per fold: dates, row counts, calibrated and raw probability metrics.
    folds: pd.DataFrame
    # Out-of-sample rows (date, symbol): raw, probability, label, fold.
    predictions: pd.DataFrame

    def summary(self) -> pd.Series:
        """Mean and standard deviation across folds of the main metrics."""
        columns = ["auc", "log_loss_skill", "brier_skill", "calibration_error", "raw_calibration_error"]
        stats = {}
        for column in columns:
            stats[f"{column}_mean"] = float(self.folds[column].mean())
            stats[f"{column}_std"] = float(self.folds[column].std(ddof=0))
        stats["folds_beating_reference"] = int((self.folds["log_loss_skill"] > 0).sum())
        stats["folds"] = int(len(self.folds))
        return pd.Series(stats)


def split_calibration(train: pd.DatetimeIndex, months: int, embargo: int) -> FoldSplit:
    """The last ``months`` of training calibrate; the rest, purged before it, fits."""
    calibration = train[train > train[-1] - pd.DateOffset(months=months)]
    earlier = train[train < calibration[0]]
    fit = earlier[: max(len(earlier) - embargo, 0)]
    if len(fit) == 0:
        raise ValueError("the training window is too short to hold back a calibration period")
    return FoldSplit(fit=fit, calibration=calibration)


def _xy(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset.frame
    return frame.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float), frame["label"].to_numpy(dtype=float)


def run_walk_forward(
    dataset: Dataset,
    folds: list[Fold],
    model_type: str,
    parameters: Mapping[str, Any],
    settings: Settings,
) -> WalkForwardResult:
    """Fit, calibrate and score ``model_type`` with ``parameters`` on every fold."""
    labelled = dataset.labelled()
    anchor = dataset.sessions
    seed = settings.retrain.search.seed
    fold_rows, predictions = [], []
    for fold in folds:
        split = split_calibration(fold.train, settings.model.calibration_months, settings.model.embargo_days)
        fit_rows = labelled.rows_on(sample_sessions(split.fit, settings.retrain.sample_every_sessions, anchor))
        calibration_rows = labelled.rows_on(split.calibration)
        validation_rows = labelled.rows_on(fold.validation)

        x_fit, y_fit = _xy(fit_rows)
        model = build_model(model_type, parameters, seed).fit(x_fit, y_fit)
        x_cal, y_cal = _xy(calibration_rows)
        calibrator = Calibrator(settings.model.calibration).fit(model.predict_proba(x_cal)[:, 1], y_cal)

        x_val, y_val = _xy(validation_rows)
        raw = model.predict_proba(x_val)[:, 1]
        probability = calibrator.transform(raw)
        base_rate = float(y_fit.mean())

        metrics = probability_metrics(y_val, probability, base_rate)
        raw_metrics = probability_metrics(y_val, raw, base_rate)
        fold_rows.append(
            {
                "fold": fold.number,
                "fit_start": split.fit[0].date(),
                "fit_end": split.fit[-1].date(),
                "calibration_start": split.calibration[0].date(),
                "validation_start": fold.validation_start.date(),
                "validation_end": fold.validation_end.date(),
                "fit_rows": int(len(y_fit)),
                "calibration_rows": int(len(y_cal)),
                **metrics,
                "raw_auc": raw_metrics["auc"],
                "raw_calibration_error": raw_metrics["calibration_error"],
            }
        )
        predictions.append(
            pd.DataFrame(
                {"raw": raw, "probability": probability, "label": y_val, "fold": fold.number},
                index=validation_rows.frame.index,
            )
        )
    return WalkForwardResult(
        model_type=model_type,
        parameters=dict(parameters),
        folds=pd.DataFrame(fold_rows).set_index("fold"),
        predictions=pd.concat(predictions) if predictions else pd.DataFrame(),
    )
