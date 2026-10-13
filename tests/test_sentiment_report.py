"""S3: summaries and daily reports (sentiment_proposal.md 8.1, 8.3), offline."""

import dataclasses
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import select

from quant.persistence.database import create_db_engine, init_db
from quant.sentiment import store, tables
from quant.sentiment.extract import run_extract, videos_to_extract
from quant.sentiment.report import write_reports
from quant.sentiment.summarize import (
    days_to_summarize,
    load_opinions,
    run_summarize,
    summary_input,
    validate_summary,
)
from settings_helper import SETTINGS
from test_sentiment_extract import BOOK, PIECE, FakeBackend, _row

DAY = date(2026, 10, 12)  # Friday-evening videos count from Monday


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        extraction = dataclasses.replace(SETTINGS.sentiment.extraction, model="fake")
        summary = dataclasses.replace(SETTINGS.sentiment.summary, model="fake",
                                      reports_dir=str(Path(self.temp.name) / "reports"))
        self.settings = dataclasses.replace(SETTINGS, sentiment=dataclasses.replace(
            SETTINGS.sentiment, extraction=extraction, summary=summary))
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        store.sync_channels(self.engine, SETTINGS.sentiment.channels)
        segments = [{"start": float(line.start), "duration": 5.0, "text": line.text} for line in PIECE]
        for video_id, handle in (("v1", "@NaNaShuoMeiGu"), ("v2", "@yttalkjun")):
            channel = SETTINGS.sentiment.channel(handle)
            store.insert_video(self.engine, {"video_id": video_id, "channel_id": channel.channel_id,
                                             "title": f"title {video_id}", "published_at": datetime(2026, 10, 9, 22),
                                             "transcript_status": "PENDING"})
            store.save_transcript(self.engine, video_id, "zh-Hans", False, segments, segments, 1)
        backend = FakeBackend([
            {"opinions": [_row(), _row(ticker="SPY", stance="BEARISH", horizon="SHORT", quote="AMD 我暂时不看。"),
                          _row(ticker="ZZZQ", name_as_said="某公司", quote="今天我们来聊聊英伟达。")]},
            {"opinions": [_row(stance="BEARISH", horizon="SHORT", action="NONE", quote="今天我们来聊聊英伟达。"),
                          _row(ticker="AMD", stance="NEUTRAL", quote="完全对不上的一句话")]},
        ])
        run_extract(self.engine, self.settings, backend, BOOK, videos_to_extract(self.engine, "claude_code", "fake", 1))
        self.verified = load_opinions(self.engine, self.settings, verified_only=True)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def ids(self, ticker):
        return [int(i) for i in self.verified[self.verified["ticker"] == ticker]["opinion_id"]]

    def reply(self, nvda_ids=None):
        return {"market": {"summary": "Creators lean cautious.", "opinion_ids": self.ids("SPY")},
                "stocks": [{"ticker": "NVDA", "summary": "NaNa is bullish long term; TALK is bearish short term.",
                            "disagreement": True, "opinion_ids": nvda_ids or self.ids("NVDA")},
                           {"ticker": "SPY", "summary": "NaNa expects a short-term dip.", "disagreement": False,
                            "opinion_ids": self.ids("SPY")}]}

    def test_input_and_validation(self):
        self.assertEqual(sorted(self.verified["ticker"].unique()), ["NVDA", "SPY"])  # AMD unverified, ZZZQ unresolved
        text = summary_input(DAY, self.verified)
        self.assertIn("## NVDA", text)
        self.assertIn("quote: 我觉得英伟达长期还是会涨", text)
        good = validate_summary(self.reply(), DAY, self.verified)
        self.assertEqual(sorted(r["ticker"] for r in good.rows), ["NVDA", "SPY", tables.MARKET])
        bad = validate_summary(self.reply(nvda_ids=self.ids("SPY")), DAY, self.verified)  # cites another stock
        self.assertEqual(bad.dropped, ["NVDA"])

    def test_summarize_once_and_again_when_opinions_change(self):
        days = days_to_summarize(self.engine, self.settings, self.verified)
        self.assertEqual(days, [DAY])
        result = run_summarize(self.engine, self.settings, FakeBackend([self.reply()]), self.verified, days)
        self.assertEqual(len(result.done), 1)
        self.assertEqual(days_to_summarize(self.engine, self.settings, self.verified), [])
        changed = self.verified.iloc[:-1]  # e.g. a re-extraction removed an opinion
        self.assertEqual(days_to_summarize(self.engine, self.settings, changed), [DAY])

    def test_report(self):
        run_summarize(self.engine, self.settings, FakeBackend([self.reply()]), self.verified, [DAY])
        [path] = write_reports(self.engine, self.settings)
        self.assertEqual(path.name, "2026-10-12.md")
        text = path.read_text(encoding="utf-8")
        self.assertIn("# Creator opinions, effective 2026-10-12 (Mon)", text)
        self.assertIn("Creators lean cautious.", text)
        # NVDA: one bull (NaNa, long term, bought) and one bear (TALK, short term); in the pool.
        nvda = next(line for line in text.splitlines() if line.startswith("| NVDA"))
        self.assertIn("| ★ | 1 | 1 | 0 | 0 | +0 |", nvda)
        self.assertIn("NaNa说美股 ▲L (bought)", nvda)
        self.assertIn("投资TALK君 ▼S", nvda)
        self.assertIn("**NVDA** (creators disagree): NaNa is bullish", text)
        self.assertIn("## ETFs and indices", text)
        self.assertIn("[00:30](https://youtu.be/v1?t=30)", text)
        self.assertIn("> 我觉得英伟达长期还是会涨，我又加仓了。", text)
        self.assertIn("1 opinion(s) whose quote did not match the transcript: AMD (投资TALK君)", text)
        self.assertIn("Unresolved names (add real stocks to the alias table): 某公司 (x1)", text)

    def test_one_entry_per_creator_and_no_double_if(self):
        import pandas as pd
        from quant.sentiment.report import stock_table

        rows = pd.DataFrame([
            {"ticker": "AMZN", "channel": "Chip", "stance": "CONDITIONAL", "horizon": "MEDIUM",
             "condition": "If it sells off", "action": "NONE"},
            {"ticker": "AMZN", "channel": "Chip", "stance": "BULLISH", "horizon": "LONG", "condition": "",
             "action": "HOLDING"},
        ])
        creators = stock_table(rows, set()).iloc[0]["Creators"]
        self.assertEqual(creators, "Chip ◆M ▲L (if it sells off; holds)")

    def test_report_without_summaries(self):
        [path] = write_reports(self.engine, self.settings)
        text = path.read_text(encoding="utf-8")
        self.assertIn("_No summary yet: run `quant sentiment summarize`._", text)
        self.assertIn("**NVDA**: _no summary_", text)

    def test_stored_summaries_follow_the_summary_config(self):
        run_summarize(self.engine, self.settings, FakeBackend([self.reply()]), self.verified, [DAY])
        with self.engine.connect() as conn:
            rows = conn.execute(select(tables.sentiment_summaries)).mappings().all()
        self.assertEqual({r["model"] for r in rows}, {"fake"})
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
