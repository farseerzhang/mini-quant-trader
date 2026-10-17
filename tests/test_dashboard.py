import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from quant.backtest.engine import run_backtests
from quant.backtest.persist import save_backtest
from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.dashboard import data
from quant.persistence.database import create_db_engine
from quant.signals.pipeline import generate_and_store_signals
from quant.forward.report import build_report
from tests.backtest_fixture import DATES, POOL, build_market_db, small_settings
from tests.test_forward_paper import WINDOW, PaperFixture

ROOT = Path(__file__).resolve().parents[1]
PROJECT_SETTINGS = ROOT / DEFAULT_CONFIG_PATH


class DashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        base = small_settings(random_runs=0)
        cls.positions = cls.root / "positions.csv"
        cls.positions.write_text("symbol,quantity,cost_basis\nAAPL,10,90\nTSLA,1,\nCASH,1000,\n", encoding="utf-8")
        cls.settings = dataclasses.replace(base, positions=dataclasses.replace(base.positions, path=str(cls.positions)))
        cls.db = cls.root / "dash.db"
        build_market_db(cls.db, cls.settings)
        cls.engine = create_db_engine(cls.db, 5000)
        generate_and_store_signals(cls.engine, cls.settings)
        cls.batch = save_backtest(cls.engine, cls.settings, run_backtests(cls.engine, cls.settings), include_holdout=False)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        cls.temp.cleanup()

    def test_market_overview_and_regime_runs(self):
        overview = data.market_overview(self.engine, self.settings)
        self.assertEqual((overview.session, overview.regime_source), (DATES[-1], "rule_based-v1"))
        self.assertEqual(list(overview.quotes.index), ["SPY", "QQQ", "^VIX"])
        self.assertIsNotNone(overview.score)
        runs = data.regime_runs(overview.history)
        self.assertTrue((runs["regime"] != runs["regime"].shift()).iloc[1:].all())  # neighbours differ
        self.assertTrue((runs["start"] < runs["end"]).all())
        labelled = overview.history.dropna(subset=["regime"])
        self.assertEqual(runs["start"].iloc[0], labelled["date"].iloc[0])

    def test_signals_are_ordered_sells_first_and_positions_flag_unmanaged(self):
        session, signals = data.latest_signals(self.engine)
        self.assertEqual(session, DATES[-1])
        order = signals["signal"].map({"EXIT": 0, "REDUCE": 1, "BUY": 2, "HOLD": 3})
        self.assertTrue(order.is_monotonic_increasing)
        _, positions = data.latest_positions(self.engine)
        self.assertEqual(positions.set_index("symbol")["managed"].to_dict(), {"AAPL": 1, "CASH": 1, "TSLA": 0})

    def test_stock_detail_lines_sit_on_the_price(self):
        frame = data.stock_detail(self.engine, self.settings, "AAPL", years=1)
        self.assertLessEqual(frame["date"].iloc[-1] - frame["date"].iloc[0], pd.Timedelta(days=367))
        last = frame.iloc[-1]
        self.assertLess(abs(last["sma50"] / last["close"] - 1), 0.5)
        self.assertTrue(frame["rsi14"].between(0, 100).all())
        self.assertTrue((frame["volume"] > 0).all())

    def test_backtest_equity_starts_at_one_with_drawdowns(self):
        batches = data.backtest_batches(self.engine)
        self.assertEqual(batches["batch_id"].iloc[0], self.batch)
        self.assertEqual(batches["stocks"].iloc[0], 3)
        metrics = data.backtest_metrics(self.engine, self.batch)
        self.assertIn("regime_switch", set(metrics["strategy"]))
        equity = data.backtest_equity(self.engine, self.batch)
        first = equity.groupby("strategy")["value"].first()
        self.assertTrue((first == 1).all())
        self.assertTrue((equity["drawdown"] <= 0).all())
        self.assertTrue(data.backtest_equity(self.engine, "no-such-batch").empty)

    def test_models_on_an_empty_registry(self):
        self.assertTrue(data.models(self.engine, self.root / "reports").empty)


class ForwardPageTests(PaperFixture):
    def test_forward_data_and_page(self):
        from streamlit.testing.v1 import AppTest

        self.freeze(WINDOW)
        runs = data.forward_runs(self.engine)
        self.assertEqual(list(runs["run_id"]), [1])
        curves = data.growth_and_drawdown(build_report(self.engine, self.settings).values)
        first = curves.sort_values("date").groupby("portfolio")["growth"].first()
        self.assertTrue((first == 1).all())
        self.assertTrue((curves["drawdown"] <= 0).all())
        session, headers, targets = data.latest_frozen(self.engine)
        self.assertEqual((session, list(headers["portfolio"])), (WINDOW[-1], ["live"]))
        self.assertAlmostEqual(targets["target_weight"].sum(), headers["invested"].iloc[0])

        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        document["database"]["url"] = f"sqlite:///{Path(str(self.engine.url.database)).as_posix()}"
        document["market"]["symbols"] = {sector: list(stocks) for sector, stocks in POOL.items()}
        document["forward_test"]["shadows"] = {}
        config = Path(self.temp.name) / "settings.yaml"
        config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        argv, sys.argv = sys.argv, ["app.py", "--config", str(config)]
        try:
            at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
            at.run()
            at.sidebar.radio[0].set_value("Forward test").run()
            self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
            self.assertIn("before the 6-month first look", " ".join(i.value for i in at.info))
            self.assertIn("live", [m.label for m in at.metric])
        finally:
            sys.argv = argv
            import gc

            import streamlit as st
            from sqlalchemy.engine import Engine

            st.cache_resource.clear()
            for engine in [o for o in gc.get_objects() if isinstance(o, Engine) and o is not self.engine]:
                engine.dispose()


class DashboardAppTests(unittest.TestCase):
    def test_every_page_renders_without_errors(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = small_settings(random_runs=0)
            build_market_db(root / "app.db", settings)
            document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
            document["database"]["url"] = f"sqlite:///{(root / 'app.db').as_posix()}"
            document["market"]["symbols"] = {sector: list(stocks) for sector, stocks in POOL.items()}
            document["positions"]["path"] = (root / "positions.csv").as_posix()
            config = root / "settings.yaml"
            config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            app = str(ROOT / "dashboard" / "app.py")
            import sys
            argv = sys.argv
            sys.argv = ["app.py", "--config", str(config)]
            try:
                for page in ("Status", "Market", "Signals", "Sentiment", "Forward test", "Broker accounts", "Stock detail", "Models",
                             "Backtest"):
                    with self.subTest(page):
                        at = AppTest.from_file(app, default_timeout=60)
                        at.run()
                        at.sidebar.radio[0].set_value(page).run()
                        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
                        self.assertIn("not advice", " ".join(c.value for c in at.caption))
            finally:
                sys.argv = argv
                import gc

                import streamlit as st
                from sqlalchemy.engine import Engine

                st.cache_resource.clear()
                for engine in [o for o in gc.get_objects() if isinstance(o, Engine)]:
                    engine.dispose()  # the app's cached engine holds the file open (Windows cannot delete it)


if __name__ == "__main__":
    unittest.main()
