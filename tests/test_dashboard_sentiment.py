"""The dashboard's creator-opinion views (Sentiment page, Stock detail section), offline."""

import gc
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from quant.config.settings import DEFAULT_CONFIG_PATH
from quant.dashboard import data
from quant.persistence.database import create_db_engine, init_db
from quant.sentiment.summarize import run_summarize
from test_sentiment_extract import FakeBackend
from test_sentiment_report import DAY, add_opinions, fake_settings, summary_reply

ROOT = Path(__file__).resolve().parents[1]


class SentimentDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = fake_settings(Path(self.temp.name) / "reports")
        self.db = Path(self.temp.name) / "sentiment.db"
        self.engine = create_db_engine(self.db, 1000)
        init_db(self.engine)
        self.verified = add_opinions(self.engine, self.settings)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_day_view(self):
        self.assertEqual(data.sentiment_days(self.engine, self.settings), [DAY])
        view = data.sentiment_day(self.engine, self.settings, DAY)
        self.assertIsNone(view.market)  # not summarized yet
        self.assertEqual((len(view.opinions), len(view.verified)), (5, 3))
        nvda = view.stocks.set_index("Stock").loc["NVDA"]
        self.assertEqual((nvda["Pool"], nvda["Bull"], nvda["Bear"], nvda["Net"]), ("★", 1, 1, 0))
        self.assertEqual(list(view.etfs["Stock"]), ["SPY"])
        first = view.quotes[view.quotes["channel"] == "NaNa说美股"].iloc[0]
        self.assertEqual((first["stance"], first["at"], first["link"]), ("▲ bullish", "00:30", "https://youtu.be/v1?t=30"))

        run_summarize(self.engine, self.settings, FakeBackend([summary_reply(self.verified)]), self.verified, [DAY])
        view = data.sentiment_day(self.engine, self.settings, DAY)
        self.assertEqual(view.market, "Creators lean cautious.")
        self.assertEqual(sorted(view.summaries), ["NVDA", "SPY"])
        self.assertTrue(view.summaries["NVDA"]["disagreement"])

    def test_stock_opinions_and_marks(self):
        opinions = data.stock_opinions(self.engine, self.settings, "NVDA")
        self.assertEqual(sorted(opinions["stance"]), ["▲ bullish", "▼ bearish"])
        self.assertTrue(data.stock_opinions(self.engine, self.settings, "NVDA", DAY + pd.Timedelta(days=1)).empty)
        prices = pd.DataFrame({"date": pd.to_datetime([DAY]), "close": [190.0]})
        marks = data.opinion_marks(opinions, prices)
        self.assertEqual(list(marks["stance"]), ["▲ bullish", "▼ bearish"])
        self.assertTrue((marks["close"] == 190.0).all())
        later = prices.assign(date=prices["date"] - pd.Timedelta(days=3))  # no close on the effective session yet
        self.assertTrue(data.opinion_marks(opinions, later).empty)

    def test_without_sentiment_tables(self):
        engine = create_db_engine(Path(self.temp.name) / "empty.db", 1000)
        try:
            self.assertEqual(data.sentiment_days(engine, self.settings), [])
            self.assertTrue(data.stock_opinions(engine, self.settings, "NVDA").empty)
        finally:
            engine.dispose()

    def test_page(self):
        from streamlit.testing.v1 import AppTest

        run_summarize(self.engine, self.settings, FakeBackend([summary_reply(self.verified)]), self.verified, [DAY])
        document = yaml.safe_load((ROOT / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
        document["database"]["url"] = f"sqlite:///{self.db.as_posix()}"
        document["sentiment"]["extraction"]["model"] = "fake"
        document["sentiment"]["extraction"]["report_models"] = ["fake", "fake2"]
        document["sentiment"]["summary"]["model"] = "fake"
        config = Path(self.temp.name) / "settings.yaml"
        config.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
        argv, sys.argv = sys.argv, ["app.py", "--config", str(config)]
        try:
            at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
            at.run()
            at.sidebar.radio[0].set_value("Sentiment").run()
            self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
            metrics = {m.label: m.value for m in at.metric}
            self.assertEqual((metrics["verified opinions"], metrics["not counted"]), ("3", "2"))
            self.assertIn("Creators lean cautious.", " ".join(m.value for m in at.markdown))
            self.assertIn("Creators disagree on this stock.", [w.value for w in at.warning])
            at.toggle[0].set_value(True).run()  # pool only: NVDA stays, SPY (an ETF) is always shown
            self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        finally:
            sys.argv = argv
            import streamlit as st
            from sqlalchemy.engine import Engine

            st.cache_resource.clear()
            for engine in [o for o in gc.get_objects() if isinstance(o, Engine) and o is not self.engine]:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
