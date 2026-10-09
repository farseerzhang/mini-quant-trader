"""Calculate and persist versioned technical and market features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from quant.config.settings import FeatureSettings
from quant.features.technical import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    calculate_market_features,
    calculate_symbol_features,
    prepare_adjusted_series,
)
from quant.persistence import queries


@dataclass(frozen=True)
class FeatureSummary:
    symbols: int
    symbol_rows: int
    market_rows: int
    feature_version: str
    market_feature_version: str
    data_as_of: date | None


def _records(
    frame: pd.DataFrame,
    key_columns: dict[str, Any],
    version: str,
    columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, values in frame.loc[:, columns].iterrows():
        record = dict(key_columns)
        record["date"] = index.date()
        record["feature_version"] = version
        for column in columns:
            value = values[column]
            record[column] = None if pd.isna(value) or not np.isfinite(value) else float(value)
        rows.append(record)
    return rows


def _load_adjusted(engine: Engine, symbol: str) -> pd.DataFrame | None:
    prices, actions = queries.load_market_data(engine, [symbol])
    if symbol not in prices:
        return None
    empty_actions = pd.DataFrame(columns=["ex_date", "action_type", "value"])
    return prepare_adjusted_series(prices[symbol], actions.get(symbol, empty_actions))


def calculate_and_store_features(engine: Engine, settings: FeatureSettings) -> FeatureSummary:
    """Recompute history, keeping only one symbol's bars in memory at a time."""
    feature_version = settings.feature_version
    market_feature_version = settings.market_feature_version
    symbols = queries.list_price_symbols(engine)
    if not symbols:
        raise ValueError("No stored market data; update market data before calculating features")
    if "SPY" not in symbols:
        raise ValueError("SPY data is required for market and relative-return features")

    spy = _load_adjusted(engine, "SPY")
    qqq = _load_adjusted(engine, "QQQ") if "QQQ" in symbols else None
    vix_symbol = "^VIX" if "^VIX" in symbols else "VIX"
    vix = _load_adjusted(engine, vix_symbol) if vix_symbol in symbols else None
    benchmarks = {"SPY": spy}
    if qqq is not None:
        benchmarks["QQQ"] = qqq
    if vix is not None:
        benchmarks[vix_symbol] = vix

    market_frame = calculate_market_features(benchmarks, settings)
    market_rows = _records(market_frame, {}, market_feature_version, MARKET_FEATURE_COLUMNS)
    _, market_count = queries.save_features(engine, [], market_rows)

    spy_close = spy["total_return_close"]
    symbol_count = 0
    data_as_of: date | None = None
    for symbol in symbols:
        adjusted = spy if symbol == "SPY" else _load_adjusted(engine, symbol)
        if adjusted is None or adjusted.empty:
            continue
        frame = calculate_symbol_features(adjusted, settings, spy_close)
        rows = _records(frame, {"symbol": symbol}, feature_version, FEATURE_COLUMNS)
        inserted, _ = queries.save_features(engine, rows, [])
        symbol_count += inserted
        latest = adjusted.index.max().date()
        data_as_of = latest if data_as_of is None else max(data_as_of, latest)

    queries.record_pipeline_run(
        engine,
        command="calculate-features",
        stage="features",
        status="OK",
        data_as_of_date=data_as_of,
        message=(
            f"Calculated {symbol_count} {feature_version} symbol rows and "
            f"{market_count} {market_feature_version} market rows"
        ),
    )
    return FeatureSummary(
        symbols=len(symbols),
        symbol_rows=symbol_count,
        market_rows=market_count,
        feature_version=feature_version,
        market_feature_version=market_feature_version,
        data_as_of=data_as_of,
    )