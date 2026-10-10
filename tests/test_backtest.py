import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest.data import Prices, load_prices
from quant.backtest.engine import BENCHMARK, STRATEGY, run_backtests, simulate, split_periods
from quant.backtest.metrics import period_metrics
from quant.backtest.orders import buy_and_hold, equal_weight_monthly, order_targets, permuted_weights
from quant.features.pipeline import calculate_and_store_features
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from quant.regime.pipeline import calculate_and_store_regimes
from tests.settings_helper import SETTINGS

BT = SETTINGS.backtest  # 1 bp fee, 5 bp slippage
_NO_ORDERS = pd.DataFrame(columns=["Timestamp", "Size", "Price", "Fees"])
_NO_TRADES = pd.DataFrame(columns=["Exit Timestamp", "PnL", "Status"])


def _dates(n, start="2026-01-05"):
    return pd.bdate_range(start, periods=n)


class OrderTargetTests(unittest.TestCase):
    def test_entries_and_exits_always_trade_but_small_resizes_do_not(self):
        weights = pd.DataFrame(
            {"A": [0.0, 0.10, 0.09, 0.07, 0.07, 0.0], "B": [0.0, 0.0, 0.01, 0.01, 0.0, 0.0]}, index=_dates(6)
        )
        orders = order_targets(weights, min_trade_weight=0.02)
        # A: entry 10%; 9% skipped (1 pt); 7% is 3 pts from the last ORDERED 10% -> resize; exit.
        np.testing.assert_array_equal(orders["A"].to_numpy(), [np.nan, 0.10, np.nan, 0.07, np.nan, 0.0])
        # B: a 1% entry and its exit trade even though they are below the minimum.
        np.testing.assert_array_equal(orders["B"].to_numpy(), [np.nan, np.nan, 0.01, np.nan, 0.0, np.nan])

    def test_equal_weight_rebalances_on_the_first_session_of_each_month(self):
        index = pd.bdate_range("2026-01-28", "2026-03-04")
        orders = equal_weight_monthly(index, ["A", "B", "C", "D"])
        placed = orders.dropna(how="all")
        self.assertEqual(list(placed.index.strftime("%Y-%m-%d")), ["2026-01-28", "2026-02-02", "2026-03-02"])
        self.assertTrue((placed == 0.25).all().all())

    def test_permuted_weights_keep_timing_and_sizes(self):
        weights = pd.DataFrame(np.random.default_rng(1).random((20, 6)), index=_dates(20), columns=list("ABCDEF"))
        shuffled = permuted_weights(weights, np.random.default_rng(42))
        pd.testing.assert_series_equal(weights.sum(axis=1), shuffled.sum(axis=1))
        self.assertEqual(list(shuffled.columns), list("ABCDEF"))
        self.assertFalse(weights.equals(shuffled))
        again = permuted_weights(weights, np.random.default_rng(42))
        pd.testing.assert_frame_equal(shuffled, again)  # same seed -> same baseline


class ExecutionTests(unittest.TestCase):
    """Section 21, test 2: a signal at T never fills at or before T's prices."""

    def setUp(self):
        self.index = _dates(5)
        self.prices = Prices(
            open=pd.DataFrame({"A": [99.0, 101.0, 103.0, 104.0, 104.0]}, index=self.index),
            close=pd.DataFrame({"A": [100.0, 102.0, 104.0, 103.0, 105.0]}, index=self.index),
        )

    def test_weight_decided_at_close_t_fills_at_the_open_of_t_plus_1_with_costs(self):
        targets = pd.DataFrame({"A": [0.5, np.nan, 0.0, np.nan, np.nan]}, index=self.index)
        simulation = simulate(targets, self.prices, BT)
        orders = simulation.orders
        self.assertEqual(list(orders["Timestamp"]), [self.index[1], self.index[3]])
        buy, sell = orders.iloc[0], orders.iloc[1]
        self.assertAlmostEqual(buy["Price"], 101.0 * (1 + BT.slippage))   # open of T+1, plus slippage
        self.assertAlmostEqual(sell["Price"], 104.0 * (1 - BT.slippage))  # open of T+1, minus slippage
        self.assertAlmostEqual(buy["Fees"], buy["Size"] * buy["Price"] * BT.fees)
        self.assertAlmostEqual(buy["Size"], BT.initial_cash * 0.5 / 101.0)
        # Valued at the close: on the fill day, cash plus shares at that close.
        cash = BT.initial_cash - buy["Size"] * buy["Price"] - buy["Fees"]
        self.assertAlmostEqual(simulation.value.iloc[1], cash + buy["Size"] * 102.0)
        self.assertAlmostEqual(simulation.value.iloc[0], BT.initial_cash)  # nothing happens on the signal day

    def test_buy_and_hold_value_matches_the_hand_calculation(self):
        simulation = simulate(buy_and_hold(self.index, "A"), self.prices, BT)
        fill = 101.0 * (1 + BT.slippage)
        # A 100% target is limited by cash: it buys what the cash covers after slippage and fees.
        shares = BT.initial_cash / (fill * (1 + BT.fees))
        self.assertAlmostEqual(simulation.orders.iloc[0]["Size"], shares)
        self.assertAlmostEqual(simulation.value.iloc[-1], shares * 105.0, places=6)


class MetricTests(unittest.TestCase):
    def test_known_series_and_chained_periods(self):
        index = _dates(6)
        value = pd.Series([100.0, 110.0, 99.0, 99.0, 108.9, 119.79], index=index)
        invested = pd.Series(1.0, index=index)
        full = period_metrics(value, invested, _NO_ORDERS, _NO_TRADES, index[0], index[-1])
        first = period_metrics(value, invested, _NO_ORDERS, _NO_TRADES, index[0], index[2])
        second = period_metrics(value, invested, _NO_ORDERS, _NO_TRADES, index[3], index[-1])
        self.assertAlmostEqual(full["total_return"], 0.1979)
        self.assertAlmostEqual(full["max_drawdown"], -0.1)
        self.assertAlmostEqual((1 + first["total_return"]) * (1 + second["total_return"]), 1 + full["total_return"])
        self.assertAlmostEqual(second["total_return"], 119.79 / 99.0 - 1)  # measured from the prior close

    def test_holdout_is_the_last_months_and_development_ends_before_it(self):
        index = pd.bdate_range("2020-01-01", "2026-10-09")
        periods = split_periods(index, holdout_months=12)
        self.assertEqual(periods.holdout_start, pd.Timestamp("2025-10-10"))
        self.assertEqual(periods.development_end, pd.Timestamp("2025-10-09"))
        self.assertEqual((periods.start, periods.end), (index[0], index[-1]))


def _rows(symbol, dates, closes):
    return [
        {"symbol": symbol, "date": stamp.date(), "open": float(c) * 0.999, "high": float(c) * 1.01,
         "low": float(c) * 0.99, "close": float(c), "volume": 1000, "provider": "test"}
        for stamp, c in zip(dates, closes)
    ]


class RunBacktestsTests(unittest.TestCase):
    def test_every_strategy_and_benchmark_with_separate_development_and_holdout(self):
        pool = {"information_technology": ("AAPL", "MSFT"), "energy": ("XOM",)}
        settings = dataclasses.replace(
            SETTINGS,
            market=dataclasses.replace(SETTINGS.market, symbols=pool),
            backtest=dataclasses.replace(SETTINGS.backtest, random_baseline_runs=3),
        )
        sessions = 650
        dates = pd.bdate_range("2023-01-02", periods=sessions)
        t = np.arange(sessions)
        series = {
            "SPY": 100 * np.exp(0.0004 * t + 0.03 * np.sin(t / 40)),
            "QQQ": 100 * np.exp(0.0005 * t + 0.04 * np.sin(t / 35)),
            "AAPL": 100 * np.exp(0.0006 * t + 0.08 * np.sin(t / 25)),
            "MSFT": 100 * np.exp(0.0003 * t + 0.06 * np.sin(t / 18 + 1)),
            "XOM": 100 * np.exp(0.0002 * t + 0.07 * np.sin(t / 30 + 2)),
            "^VIX": 18 + 6 * np.sin(t / 50),
        }
        with tempfile.TemporaryDirectory() as directory:
            engine = create_db_engine(Path(directory) / "bt.db", settings.database.busy_timeout_ms)
            init_db(engine)
            queries.upsert_daily_prices(engine, [row for s, c in series.items() for row in _rows(s, dates, c)])
            calculate_and_store_features(engine, settings.features)
            calculate_and_store_regimes(engine, settings.regime, settings.features)
            report = run_backtests(engine, settings)
            prices = load_prices(engine, ["SPY"])
            engine.dispose()

        names = [result.name for result in report.results]
        self.assertEqual(
            names, ["trend", "mean_reversion", "regime_switch", "regime_switch_bear50", "SPY", "equal_weight"]
        )
        self.assertEqual([r.kind for r in report.results], [STRATEGY] * 4 + [BENCHMARK] * 2)
        development, holdout = report.table("development"), report.table("holdout")
        self.assertTrue(development.index.equals(holdout.index))
        self.assertFalse(development.loc[:, ["total_return", "sharpe", "max_drawdown"]].isna().any().any())
        self.assertLess(report.periods.development_end, report.periods.holdout_start)
        self.assertEqual(report.periods.start, dates[settings.features.sma[2] - 1])  # first regime session

        # The two periods chain to the full-history return of each portfolio.
        for result in report.results:
            full = result.value.iloc[-1] / settings.backtest.initial_cash - 1
            chained = (1 + result.development["total_return"]) * (1 + result.holdout["total_return"]) - 1
            self.assertAlmostEqual(chained, full, places=9)

        # SPY buy-and-hold tracks SPY's total return after the first fill.
        spy = report.results[names.index("SPY")].value
        spy_close = prices.close["SPY"].reindex(spy.index)
        np.testing.assert_allclose(spy.pct_change().iloc[2:], spy_close.pct_change().iloc[2:], rtol=1e-9)

        summary = report.random_baseline_summary()
        self.assertEqual(list(summary.index), names[:4])
        self.assertTrue((summary["runs"] == 3).all())


if __name__ == "__main__":
    unittest.main()
