"""`quant generate-signals`: run-daily stages 7-12 (proposal section 23).

7-8.  ACTIVE predictor: predictions after its training end, stored; skipped (logged) without one.
9-11. Strategy, risk layer, positions -> BUY / HOLD / REDUCE / EXIT with reasons.
12.   Persist: the positions snapshot and the signals, each in its own transaction and
      with its own pipeline_runs row, so a re-run of the same session replaces both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import Engine

from quant.config.settings import Settings
from quant.models.live import predict_and_store
from quant.persistence import queries
from quant.signals.engine import Signals, load_signals
from quant.signals.positions import snapshot_rows

log = logging.getLogger(__name__)
COMMAND = "generate-signals"
PREDICTOR = "PREDICTOR"


@dataclass(frozen=True)
class SignalRun:
    signals: Signals
    model_version: str
    positions_stored: int  # 0: no positions file, nothing stored
    signals_stored: int
    predictions: object = None  # PredictionSummary with an ACTIVE predictor


def _number(stocks, symbol: str, column: str) -> float | None:
    if column not in stocks.columns:
        return None
    value = stocks.at[symbol, column]
    return None if value != value else float(value)  # NaN -> None


def signal_rows(signals: Signals) -> list[dict]:
    stocks = signals.targets.stocks
    return [
        {
            "symbol": symbol,
            "regime": signals.targets.regime,
            "target_weight": float(row["target_weight"]),
            "current_weight": float(row["current_weight"]),
            "signal": row["signal"],
            "score": _number(stocks, symbol, "probability"),  # calibrated probability (ml_select)
            "confidence": _number(stocks, symbol, "rank_percentile"),  # 1.0 = ranked first
            "suggested_shares": float(row["suggested_shares"]),
            "close": float(row["close"]),
            "reason": row["reason"],
        }
        for symbol, row in signals.rows.iterrows()
    ]


def generate_and_store_signals(engine: Engine, settings: Settings) -> SignalRun:
    active = queries.active_model(engine, PREDICTOR)
    predictions = predict_and_store(engine, settings, active) if active is not None else None
    signals = load_signals(engine, settings)
    session = signals.session.date()
    for warning in signals.warnings:
        log.warning("%s: %s", session, warning)
    if active is None:
        queries.record_pipeline_run(
            engine, command=COMMAND, stage="predictor", status="SKIPPED", data_as_of_date=session,
            message="no ACTIVE predictor: rule-based signals only",
        )

    positions_stored = 0
    if signals.holdings.path is not None:
        positions_stored = queries.replace_positions(
            engine, session, signals.holdings.source, snapshot_rows(signals.holdings, session)
        )
        message = f"{positions_stored} rows from {signals.holdings.path}; value {signals.portfolio_value:,.2f}"
    else:
        message = f"no positions file: notional all-cash portfolio of {signals.portfolio_value:,.0f}, not stored"
    queries.record_pipeline_run(engine, command=COMMAND, stage="positions", status="OK",
                                data_as_of_date=session, message=message)

    model_version = signals.targets.model_version
    stored = queries.replace_signals(engine, session, signals.targets.strategy, model_version, signal_rows(signals))
    counts = ", ".join(f"{name} {count}" for name, count in signals.counts.items())
    queries.record_pipeline_run(
        engine, command=COMMAND, stage="signals", status="OK", data_as_of_date=session,
        message=f"{signals.targets.strategy} ({model_version}), {signals.targets.regime}: {counts}",
    )
    return SignalRun(signals, model_version, positions_stored, stored, predictions)
