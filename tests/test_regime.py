import dataclasses
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from quant.cli import main
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.features.pipeline import calculate_and_store_features
from quant.persistence import queries, tables
from quant.persistence.database import create_db_engine, init_db
from quant.regime.pipeline import calculate_and_store_regimes
from quant.regime.rule_based import (
    BEAR,
    BULL,
    SIDEWAYS,
    average_dwell,
    calculate_rule_regimes,
    classify_regimes,
    score_inputs,
)
from tests.settings_helper import SETTINGS

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH


def _rules(min_dwell_days=1):
    """Fixed thresholds (E=3, X=1) so state-machine tests do not move with the config."""
    return dataclasses.replace(
        SETTINGS.regime, enter_threshold=3, exit_threshold=1, min_dwell_days=min_dwell_days
    )


def _states(scores, rules):
    dates = pd.bdate_range("2026-01-01", periods=len(scores))
    return list(classify_regimes(pd.Series(scores, index=dates, dtype=float), rules)["regime"])


def _inputs(rows):
    columns = ["spy_trend", "qqq_trend", "spy_sma50_vs_sma200", "spy_return_3m", "vix"]
    return pd.DataFrame(rows, columns=columns, index=pd.bdate_range("2026-01-01", periods=len(rows)))


class ScoreTests(unittest.TestCase):
    def test_each_input_scores_minus_one_zero_or_plus_one(self):
        rules = SETTINGS.regime
        between = (rules.vix_low + rules.vix_high) / 2
        scores = score_inputs(
            _inputs(
                [
                    [0.05, 0.04, 0.10, 0.08, rules.vix_low - 1],    # everything bullish
                    [-0.05, -0.04, -0.10, -0.08, rules.vix_high + 1],  # everything bearish
                    [0.05, -0.04, 0.0, 0.08, between],               # +1 -1 0 +1 0
                ]
            ),
            rules,
        )
        self.assertEqual(list(scores), [5, -5, 1])

    def test_a_missing_input_gives_no_score_instead_of_zero(self):
        scores = score_inputs(_inputs([[0.05, np.nan, 0.10, 0.08, 15.0], [0.05, 0.04, 0.10, 0.08, np.nan]]),
                              SETTINGS.regime)
        self.assertTrue(scores.isna().all())


class StateMachineTests(unittest.TestCase):
    def test_hysteresis_enters_at_e_and_leaves_below_x(self):
        states = _states([0, 3, 2, 1, 0, -1, -3, -2, -1, 0], _rules())
        self.assertEqual(
            states,
            [SIDEWAYS, BULL, BULL, BULL, SIDEWAYS, SIDEWAYS, BEAR, BEAR, BEAR, SIDEWAYS],
        )

    def test_bull_and_bear_can_switch_directly(self):
        self.assertEqual(_states([3, -3, 4], _rules()), [BULL, BEAR, BULL])

    def test_first_session_is_exempt_from_the_dwell_rule(self):
        self.assertEqual(_states([-4], _rules(min_dwell_days=5)), [BEAR])

    def test_minimum_dwell_delays_a_change(self):
        states = _states([3, -3, -3, -3, 0, 0], _rules(min_dwell_days=3))
        self.assertEqual(states, [BULL, BULL, BULL, BEAR, BEAR, BEAR])

    def test_a_suppressed_change_is_not_queued(self):
        # The BEAR proposal on day 2 is blocked; by day 4 the score only supports SIDEWAYS.
        states = _states([3, -3, 0, 0], _rules(min_dwell_days=3))
        self.assertEqual(states, [BULL, BULL, BULL, SIDEWAYS])

    def test_unscorable_days_carry_the_state_and_do_not_count_towards_dwell(self):
        dates = pd.bdate_range("2026-01-01", periods=5)
        scores = pd.Series([3, np.nan, np.nan, 0, 0], index=dates)
        regimes = classify_regimes(scores, _rules(min_dwell_days=2))
        self.assertEqual(list(regimes.index), [dates[0], dates[3], dates[4]])
        self.assertEqual(list(regimes["regime"]), [BULL, BULL, SIDEWAYS])

    def test_score_and_confidence_are_stored(self):
        regimes = classify_regimes(pd.Series([4.0, -2.0], index=pd.bdate_range("2026-01-01", periods=2)), _rules())
        self.assertEqual(list(regimes["regime_score"]), [4, -2])
        self.assertEqual(list(regimes["confidence"]), [0.8, 0.4])

    def test_average_dwell_is_the_mean_run_length(self):
        self.assertEqual(average_dwell(pd.Series([BULL, BULL, BULL, BEAR, SIDEWAYS, SIDEWAYS])), 2.0)


class PointInTimeTests(unittest.TestCase):
    def test_truncation_at_t_does_not_change_the_regime_up_to_t(self):
        days = 400
        phase = np.arange(days) / 17
        inputs = _inputs(
            np.column_stack(
                [np.sin(phase), np.sin(phase + 0.5), np.sin(phase / 2), np.cos(phase), 25 + 12 * np.sin(phase / 3)]
            )
        )
        inputs.iloc[100:103, 1] = np.nan  # a gap in one input
        full = calculate_rule_regimes(inputs, SETTINGS.regime)
        for cut in (inputs.index[150], inputs.index[257], inputs.index[-2]):
            truncated = calculate_rule_regimes(inputs.loc[:cut], SETTINGS.regime)
            pd.testing.assert_frame_equal(full.loc[:cut], truncated)
        self.assertGreater(full["regime"].nunique(), 1)


def _price_rows(symbol, dates, closes):
    return [
        {
            "symbol": symbol,
            "date": stamp.date(),
            "open": float(close),
            "high": float(close) * 1.01,
            "low": float(close) * 0.99,
            "close": float(close),
            "volume": None if symbol == "^VIX" else 1000,
            "provider": "test",
        }
        for stamp, close in zip(dates, closes)
    ]


class RegimePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "regime.db"
        self.engine = create_db_engine(self.db_path, SETTINGS.database.busy_timeout_ms)
        init_db(self.engine)

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _store_rising_market(self, sessions=260):
        dates = pd.bdate_range("2025-01-01", periods=sessions)
        trend = 100 + np.arange(sessions) * 0.2
        rows = (
            _price_rows("SPY", dates, trend)
            + _price_rows("QQQ", dates, trend * 1.5)
            + _price_rows("^VIX", dates, np.full(sessions, SETTINGS.regime.vix_low - 5))
        )
        queries.upsert_daily_prices(self.engine, rows)
        calculate_and_store_features(self.engine, SETTINGS.features)
        return dates

    def test_pipeline_stores_one_regime_per_scorable_session_and_reruns_cleanly(self):
        dates = self._store_rising_market()
        first = calculate_and_store_regimes(self.engine, SETTINGS.regime, SETTINGS.features)
        second = calculate_and_store_regimes(self.engine, SETTINGS.regime, SETTINGS.features)

        # The SMA200-based inputs need 200 sessions of history.
        expected = len(dates) - (SETTINGS.features.sma[2] - 1)
        with self.engine.connect() as conn:
            stored = conn.execute(select(func.count()).select_from(tables.market_regimes)).scalar_one()
            versions = set(conn.execute(select(tables.market_regimes.c.model_version)).scalars())
        self.assertEqual((first.rows, second.rows, stored), (expected, expected, expected))
        self.assertEqual(versions, {SETTINGS.regime.rule_version})
        self.assertEqual(first.first_date, dates[SETTINGS.features.sma[2] - 1].date())
        self.assertEqual((second.latest_regime, second.latest_score), (BULL, 5))

    def test_cli_calculates_the_regime(self):
        self._store_rising_market()
        self.engine.dispose()
        output = StringIO()
        with redirect_stdout(output):
            code = main(["--config", str(PROJECT_SETTINGS), "calculate-regime", "--db", str(self.db_path)])
        self.assertEqual(code, 0)
        self.assertIn("Latest: BULL (score +5)", output.getvalue())

    def test_missing_features_are_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "calculate features"):
            calculate_and_store_regimes(self.engine, SETTINGS.regime, SETTINGS.features)


if __name__ == "__main__":
    unittest.main()
