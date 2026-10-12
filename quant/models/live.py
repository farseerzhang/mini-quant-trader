"""The ACTIVE predictor in daily use (proposal sections 6.7, 16.3 and 23 steps 7-8).

Every run scores every pool stock on every session after the model's training end up to
the latest session T, and replaces those stored predictions. The whole stretch is needed
because `ml_select` refreshes its selection every few sessions and keeps held stocks
within a rank band: today's holdings depend on earlier predictions, and anchoring the
rebalance schedule at the first session after training keeps it the same on every run.

Inputs are built exactly as in training (stored features, market features and the regime
the model was trained with, its ``depends_on``), all point in time. Each prediction stores
a hash of its input values, so it can be checked and reproduced later (section 32, item 15).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.features.technical import FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
from quant.models.dataset import build_dataset
from quant.models.final import FinalModel, load_final_model
from quant.models.labels import Labels
from quant.persistence import queries
from quant.regime.active import HMM_METHOD, ActiveModelError
from quant.regime.pipeline import RULE_METHOD

PREDICTOR = "PREDICTOR"
COMMAND = "generate-signals"
_NO_LABELS = Labels(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


@dataclass(frozen=True)
class PredictionSummary:
    model_id: str
    model_type: str
    rows: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp


def input_hash(values: np.ndarray) -> str:
    """sha256 of one row's input values as float64, in the model's column order."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def load_active_predictor(row, settings: Settings) -> FinalModel:
    path = Path(row["model_path"])
    if not path.exists():
        raise ActiveModelError(f"ACTIVE predictor file {path} is missing; retrain or promote another model")
    if row["feature_version"] != settings.features.feature_version:
        raise ActiveModelError(
            f"ACTIVE predictor {row['model_id']} was trained on {row['feature_version']}, but the stored features are "
            f"{settings.features.feature_version}; retrain it"
        )
    return load_final_model(path)


def predictor_regimes(engine: Engine, settings: Settings, row) -> pd.Series:
    """The regime the predictor was trained with (its ``depends_on``), as stored."""
    depends_on = row["depends_on"] or settings.regime.rule_version
    if depends_on == settings.regime.rule_version:
        return queries.load_regimes(engine, RULE_METHOD, depends_on)
    regimes = queries.load_regimes(engine, HMM_METHOD, depends_on)
    if regimes.empty:
        raise ActiveModelError(
            f"ACTIVE predictor {row['model_id']} needs the regime {depends_on}, which is not stored; "
            "promote that regime model or retrain the predictor"
        )
    return regimes


def predict_sessions(model: FinalModel, features: pd.DataFrame, market: pd.DataFrame,
                     regimes: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    """Probability and input hash per (date, symbol) from ``start`` on (pure; no database)."""
    dataset = build_dataset(features, market, regimes, _NO_LABELS)
    inputs = dataset.inputs
    inputs = inputs[inputs.index.get_level_values("date") >= start]
    if inputs.empty:
        return pd.DataFrame(columns=["probability", "input_hash"],
                            index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]))
    values = inputs.loc[:, list(model.input_columns)].to_numpy(dtype=float)
    return pd.DataFrame({
        "probability": model.predict(inputs).to_numpy(),
        "input_hash": [input_hash(row) for row in values],
    }, index=inputs.index)


def predict_and_store(engine: Engine, settings: Settings, row) -> PredictionSummary:
    model = load_active_predictor(row, settings)
    stocks = list(settings.market.stocks)
    start = pd.Timestamp(row["training_end"]) + pd.Timedelta(days=1)
    features = queries.load_features_long(engine, settings.features.feature_version, stocks, FEATURE_COLUMNS)
    market = queries.load_market_features(engine, settings.features.market_feature_version, MARKET_FEATURE_COLUMNS)
    frame = predict_sessions(model, features, market, predictor_regimes(engine, settings, row), start)
    if frame.empty:
        raise ActiveModelError(f"no session after {row['training_end']} has complete inputs; run `quant run-daily`")
    target = row["target_definition"] or json.dumps({})
    rows = [
        {"date": date.date(), "symbol": symbol, "probability": float(r.probability),
         "target_definition": target, "input_hash": r.input_hash}
        for (date, symbol), r in frame.iterrows()
    ]
    stored = queries.replace_predictions(engine, row["model_id"], start.date(), rows)
    dates = frame.index.get_level_values("date")
    summary = PredictionSummary(row["model_id"], row["model_type"], stored, dates.min(), dates.max())
    queries.record_pipeline_run(
        engine, command=COMMAND, stage="predictor", status="OK", data_as_of_date=summary.last_date.date(),
        message=f"{stored} predictions by ACTIVE {row['model_id']} for {summary.first_date.date()}.."
                f"{summary.last_date.date()}",
    )
    return summary


def load_probabilities(engine: Engine, model_id: str) -> pd.DataFrame:
    """Stored probabilities of a model, dates x symbols."""
    stored = queries.load_predictions(engine, model_id)
    return stored["probability"].unstack("symbol").sort_index()
