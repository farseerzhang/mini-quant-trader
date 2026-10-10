import dataclasses
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select, func

from quant.config.settings import ConfigError
from quant.features.pipeline import calculate_and_store_features
from quant.features.technical import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    calculate_market_features,
    calculate_symbol_features,
    prepare_adjusted_series,
)
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine, init_db
from tests.settings_helper import SETTINGS


def _price_frame(dates, close, volume=None):
    close = np.asarray(close, dtype=float)
    if volume is None:
        volume = np.full(len(close), 1000, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates),
    )


class CorporateActionAdjustmentTests(unittest.TestCase):
    def test_split_adjusts_prior_ohlcv_and_dividend_builds_total_return_close(self):
        dates = pd.date_range("2026-01-01", periods=4, freq="D")
        prices = pd.DataFrame(
            {
                "open": [100.0, 100.0, 25.0, 24.0],
                "high": [110.0, 110.0, 27.5, 25.0],
                "low": [90.0, 90.0, 22.5, 23.0],
                "close": [100.0, 100.0, 25.0, 24.0],
                "volume": [100, 100, 400, 400],
            },
            index=dates,
        )
        actions = pd.DataFrame(
            [
                {"ex_date": dates[2].date(), "action_type": "SPLIT", "value": 4.0},
                {"ex_date": dates[3].date(), "action_type": "DIVIDEND", "value": 1.0},
            ]
        )

        adjusted = prepare_adjusted_series(prices, actions)

        self.assertEqual(adjusted.loc[dates[1], "close"], 25.0)
        self.assertEqual(adjusted.loc[dates[0], "volume"], 400.0)
        self.assertEqual(adjusted.loc[dates[2], "close"], 25.0)
        self.assertAlmostEqual(adjusted.loc[dates[2], "total_return_close"], 25.0)
        self.assertAlmostEqual(adjusted.loc[dates[3], "total_return_close"], 25.0)

    def test_dividend_is_rescaled_for_a_later_split(self):
        dates = pd.date_range("2026-01-01", periods=4, freq="D")
        prices = _price_frame(dates, [100, 98, 24.5, 24.5])
        actions = pd.DataFrame(
            [
                {"ex_date": dates[1].date(), "action_type": "DIVIDEND", "value": 2.0},
                {"ex_date": dates[2].date(), "action_type": "SPLIT", "value": 4.0},
            ]
        )
        adjusted = prepare_adjusted_series(prices, actions)
        self.assertAlmostEqual(adjusted.loc[dates[1], "total_return_close"], 25.0)
        self.assertAlmostEqual(adjusted.loc[dates[2], "total_return_close"], 25.0)


class TechnicalFeatureTests(unittest.TestCase):
    def test_warmup_periods_are_null_not_backfilled(self):
        dates = pd.bdate_range("2025-01-01", periods=20)
        adjusted = prepare_adjusted_series(_price_frame(dates, np.arange(100, 120)), pd.DataFrame())
        features = calculate_symbol_features(adjusted, SETTINGS.features)
        self.assertTrue(features.iloc[:19]["price_vs_sma20"].isna().all())
        self.assertTrue(features.iloc[:20]["return_1m"].isna().all())
        self.assertTrue(features.iloc[19]["price_vs_sma20"] > 0)

    def test_truncation_at_t_does_not_change_features_at_t(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        close = 100 * np.exp(np.cumsum(np.sin(np.arange(300) / 13) * 0.002 + 0.0004))
        full_adjusted = prepare_adjusted_series(_price_frame(dates, close), pd.DataFrame())
        cut = dates[247]
        truncated_adjusted = full_adjusted.loc[:cut]
        full = calculate_symbol_features(full_adjusted, SETTINGS.features)
        truncated = calculate_symbol_features(truncated_adjusted, SETTINGS.features)
        pd.testing.assert_series_equal(full.loc[cut], truncated.loc[cut], check_names=False)

    def test_market_relative_features_against_known_values(self):
        sessions = 300
        dates = pd.bdate_range("2024-01-01", periods=sessions)
        spy_returns = 0.01 * np.sin(np.arange(sessions) / 5) + 0.0003
        spy_returns[0] = 0.0
        spy_close = 100 * np.cumprod(1 + spy_returns)
        # The stock moves exactly twice as much as SPY every day: beta 2, correlation 1.
        stock_close = 50 * np.cumprod(1 + 2 * spy_returns)
        spy = prepare_adjusted_series(_price_frame(dates, spy_close), pd.DataFrame())
        stock = prepare_adjusted_series(_price_frame(dates, stock_close), pd.DataFrame())

        features = calculate_symbol_features(stock, SETTINGS.features, spy["total_return_close"])
        last = features.iloc[-1]
        stock_series = pd.Series(stock_close, index=dates)

        self.assertAlmostEqual(last["beta_spy_60d"], 2.0)
        self.assertAlmostEqual(last["corr_spy_60d"], 1.0)
        self.assertAlmostEqual(last["return_5d"], stock_close[-1] / stock_close[-6] - 1)
        self.assertAlmostEqual(
            last["relative_return_3m_spy"],
            (stock_close[-1] / stock_close[-64] - 1) - (spy_close[-1] / spy_close[-64] - 1),
        )
        self.assertAlmostEqual(last["drawdown_252d"], stock_close[-1] / stock_series.iloc[-252:].max() - 1)
        daily = stock_series.pct_change()
        self.assertAlmostEqual(
            last["volatility_ratio_20_60"], daily.iloc[-20:].std() / daily.iloc[-60:].std()
        )
        self.assertTrue(features["beta_spy_60d"].iloc[:60].isna().all())  # 60 returns need 61 sessions
        self.assertTrue(features["drawdown_252d"].iloc[:251].isna().all())

    def test_hmm_market_features_are_log_returns_and_log_levels(self):
        sessions = 60
        dates = pd.bdate_range("2024-01-01", periods=sessions)
        spy_close = 100 + np.arange(sessions) + 3 * np.sin(np.arange(sessions))
        vix_close = 20 + 5 * np.sin(np.arange(sessions) / 4)
        benchmarks = {
            "SPY": prepare_adjusted_series(_price_frame(dates, spy_close), pd.DataFrame()),
            "^VIX": prepare_adjusted_series(_price_frame(dates, vix_close), pd.DataFrame()),
        }
        market = calculate_market_features(benchmarks, SETTINGS.features)

        np.testing.assert_allclose(market["spy_return_1d"].iloc[1:], np.diff(np.log(spy_close)))
        np.testing.assert_allclose(market["log_vix"], np.log(vix_close))
        np.testing.assert_allclose(
            market["log_market_volatility"].dropna(), np.log(market["market_volatility"].dropna())
        )
        self.assertTrue(market["qqq_return_1d"].isna().all())  # QQQ is optional

    def test_truncation_with_spy_does_not_change_relative_features_at_t(self):
        dates = pd.bdate_range("2024-01-01", periods=320)
        spy_close = 100 * np.exp(np.cumsum(np.cos(np.arange(320) / 11) * 0.003 + 0.0003))
        stock_close = 80 * np.exp(np.cumsum(np.sin(np.arange(320) / 7) * 0.004 + 0.0002))
        spy = prepare_adjusted_series(_price_frame(dates, spy_close), pd.DataFrame())
        stock = prepare_adjusted_series(_price_frame(dates, stock_close), pd.DataFrame())
        cut = dates[280]
        full = calculate_symbol_features(stock, SETTINGS.features, spy["total_return_close"])
        truncated = calculate_symbol_features(
            stock.loc[:cut], SETTINGS.features, spy.loc[:cut, "total_return_close"]
        )
        pd.testing.assert_series_equal(full.loc[cut], truncated.loc[cut], check_names=False)
        self.assertFalse(full.loc[cut].isna().any())

    def test_feature_names_match_persisted_model_inputs(self):
        self.assertEqual(tuple(tables.features.c.keys())[3:], FEATURE_COLUMNS)
        self.assertEqual(tuple(tables.market_features.c.keys())[2:], MARKET_FEATURE_COLUMNS)
        self.assertEqual(SETTINGS.features.feature_version, "technical-v1")


class SchemaCompatibilityTests(unittest.TestCase):
    def test_sma_windows_must_match_the_stored_feature_columns(self):
        dates = pd.bdate_range("2025-01-01", periods=30)
        adjusted = prepare_adjusted_series(_price_frame(dates, np.arange(100, 130)), pd.DataFrame())
        mismatched = dataclasses.replace(SETTINGS.features, sma=(10, 50, 200))
        with self.assertRaises(ConfigError):
            calculate_symbol_features(adjusted, mismatched)
        calculate_symbol_features(adjusted, SETTINGS.features)


class FeaturePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_db_engine(
            Path(self.temp_dir.name) / "features.db", SETTINGS.database.busy_timeout_ms
        )
        init_db(self.engine)

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_feature_pipeline_persists_and_reruns_without_duplicates(self):
        dates = pd.bdate_range("2025-01-01", periods=260)
        records = []
        for symbol, multiplier in (("SPY", 1.0), ("QQQ", 1.1), ("AAPL", 0.8)):
            closes = multiplier * (100 + np.arange(len(dates)) * 0.15 + np.sin(np.arange(len(dates)) / 8))
            frame = _price_frame(dates, closes)
            records.extend(
                {
                    "symbol": symbol,
                    "date": stamp.date(),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": int(row.volume),
                    "provider": "test",
                }
                for stamp, row in frame.iterrows()
            )
        queries.upsert_daily_prices(self.engine, records)

        first = calculate_and_store_features(self.engine, SETTINGS.features)
        second = calculate_and_store_features(self.engine, SETTINGS.features)

        expected_symbol_rows = len(dates) * 3
        with self.engine.connect() as conn:
            stored_symbols = conn.execute(select(func.count()).select_from(tables.features)).scalar_one()
            stored_market = conn.execute(select(func.count()).select_from(tables.market_features)).scalar_one()
        self.assertEqual(first.symbol_rows, expected_symbol_rows)
        self.assertEqual(second.symbol_rows, expected_symbol_rows)
        self.assertEqual(stored_symbols, expected_symbol_rows)
        self.assertEqual(stored_market, len(dates))

        with self.engine.connect() as conn:
            aapl_last = conn.execute(
                select(tables.features).where(
                    tables.features.c.symbol == "AAPL",
                    tables.features.c.date == dates[-1].date(),
                    tables.features.c.feature_version == SETTINGS.features.feature_version,
                )
            ).mappings().one()
        self.assertIsNotNone(aapl_last["price_vs_sma200"])
        self.assertIsNotNone(aapl_last["relative_return_1m_spy"])


if __name__ == "__main__":
    unittest.main()
