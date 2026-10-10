"""A small stored market (3 stocks, SPY, QQQ, ^VIX over 650 sessions) with features and regimes."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from quant.features.pipeline import calculate_and_store_features
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from quant.regime.pipeline import calculate_and_store_regimes
from tests.settings_helper import SETTINGS

POOL = {"information_technology": ["AAPL", "MSFT"], "energy": ["XOM"]}
SESSIONS = 650
DATES = pd.bdate_range("2023-01-02", periods=SESSIONS)


def small_settings(random_runs: int = 3):
    """The project settings with the small pool and a few random-baseline runs."""
    pool = {sector: tuple(stocks) for sector, stocks in POOL.items()}
    return dataclasses.replace(
        SETTINGS,
        market=dataclasses.replace(SETTINGS.market, symbols=pool),
        backtest=dataclasses.replace(SETTINGS.backtest, random_baseline_runs=random_runs),
    )


def _rows(symbol, closes):
    return [
        {"symbol": symbol, "date": stamp.date(), "open": float(c) * 0.999, "high": float(c) * 1.01,
         "low": float(c) * 0.99, "close": float(c), "volume": 1000, "provider": "test"}
        for stamp, c in zip(DATES, closes)
    ]


def build_market_db(path: Path, settings) -> None:
    t = np.arange(SESSIONS)
    series = {
        "SPY": 100 * np.exp(0.0004 * t + 0.03 * np.sin(t / 40)),
        "QQQ": 100 * np.exp(0.0005 * t + 0.04 * np.sin(t / 35)),
        "AAPL": 100 * np.exp(0.0006 * t + 0.08 * np.sin(t / 25)),
        "MSFT": 100 * np.exp(0.0003 * t + 0.06 * np.sin(t / 18 + 1)),
        "XOM": 100 * np.exp(0.0002 * t + 0.07 * np.sin(t / 30 + 2)),
        "^VIX": 18 + 6 * np.sin(t / 50),
    }
    engine = create_db_engine(path, settings.database.busy_timeout_ms)
    init_db(engine)
    queries.upsert_daily_prices(engine, [row for symbol, closes in series.items() for row in _rows(symbol, closes)])
    calculate_and_store_features(engine, settings.features)
    calculate_and_store_regimes(engine, settings.regime, settings.features)
    engine.dispose()
