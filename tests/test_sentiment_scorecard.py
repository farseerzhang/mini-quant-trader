"""S4 (sentiment_proposal.md 12.6): the evaluation settings, the forward-first caption and extraction order,
the backfill listing and the exploratory scorecard; offline, hand-made data, no YouTube, Yahoo or claude."""

import dataclasses
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant.backtest.data import Prices
from quant.cli import main
from quant.config.settings import DEFAULT_CONFIG_PATH, ConfigError, load_settings
from quant.persistence.database import create_db_engine, init_db
from quant.sentiment import store, tables
from quant.sentiment.extract import videos_to_extract
from quant.sentiment.fetch import fetch
from quant.sentiment.scorecard import (
    ALL_CREATORS,
    _format,
    counted,
    load_scored_opinions,
    score_horizon,
    window_excess,
)
from settings_helper import SETTINGS
from test_sentiment_fetch import CHIP, FakeCaptions, FakeVideos, _captions, _listed, _page, _settings
from tests.backtest_fixture import DATES, POOL, build_market_db, small_settings

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
NOW = datetime(2026, 10, 13, 22, 0)


class EvaluationSettingsTest(unittest.TestCase):
    def variant(self, change):
        document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        change(document["sentiment"]["evaluation"])
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "settings.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return path

    def test_project_settings(self):
        evaluation = SETTINGS.sentiment.evaluation
        self.assertEqual((evaluation.horizons, evaluation.min_samples, evaluation.permutations, evaluation.model),
                         ((5, 20, 60), 30, 10000, "haiku"))
        self.assertEqual((evaluation.scored_until, evaluation.backfill_from), (date(2026, 10, 22), date(2026, 4, 23)))

    def test_forward_opinions_can_never_be_scored(self):
        for change in (lambda e: e.update(scored_until="2026-10-23"),
                       lambda e: e.update(backfill_from="2026-10-22"),
                       lambda e: e.update(horizons=[0]),
                       lambda e: e.update(permutations=10)):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                load_settings(self.variant(change))


class BackfillFetchTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        self.settings = _settings(handles={CHIP})
        evaluation = dataclasses.replace(self.settings.sentiment.evaluation, backfill_from=date(2026, 9, 1))
        self.settings = dataclasses.replace(self.settings, sentiment=dataclasses.replace(
            self.settings.sentiment, evaluation=evaluation, max_captions_per_run=2))

    def tearDown(self):
        self.engine.dispose()

    def statuses(self):
        with self.engine.connect() as conn:
            return {r.video_id: r.transcript_status for r in conn.execute(tables.sentiment_videos.select())}

    def test_the_backfill_listing_reaches_backfill_from_and_forward_videos_come_first(self):
        listing = {CHIP: [_listed("new"), _listed("week_old"), _listed("old1"), _listed("old2"), _listed("too_old")]}
        pages = {"new": _page("new", 5), "week_old": _page("week_old", 24 * 6), "old1": _page("old1", 24 * 20),
                 "old2": _page("old2", 24 * 30), "too_old": _page("too_old", 24 * 60)}
        captions = FakeCaptions({v: _captions() for v in pages})
        fetch(self.engine, self.settings, FakeVideos(listing, pages), captions, now=NOW, backfill=True)
        stored = self.statuses()
        self.assertNotIn("too_old", stored)  # before 2026-09-01
        self.assertEqual([r[0] for r in captions.requests], ["week_old", "new"])  # forward first, oldest first
        self.assertEqual((stored["old1"], stored["old2"]), ("PENDING", "PENDING"))

        # A later ordinary run (no --backfill) still fetches backfill captions, after any forward video.
        later = NOW + timedelta(hours=7)
        listing = {CHIP: [_listed("newer"), *listing[CHIP]]}
        pages["newer"] = _page("newer", -6)  # published after NOW
        captions = FakeCaptions({v: _captions() for v in pages})
        fetch(self.engine, self.settings, FakeVideos(listing, pages), captions, now=later)
        self.assertEqual([r[0] for r in captions.requests], ["newer", "old2"])  # then the oldest backfill video


    def test_a_backfill_listing_reads_10_entries_a_day_and_warns_when_cut_off(self):
        limits = []

        class RecordingVideos(FakeVideos):
            def list_channel(self, handle, limit):
                limits.append(limit)
                return super().list_channel(handle, limit)

        class AnyCaptions(FakeCaptions):
            def fetch(self, video_id, languages):
                return _captions()

        lines = []
        listing = {CHIP: [_listed("a"), _listed("b")]}
        pages = {"a": _page("a", 24), "b": _page("b", 48)}
        fetch(self.engine, self.settings, RecordingVideos(listing, pages), AnyCaptions({}), now=NOW, backfill=True,
              progress=lines.append)
        days = (NOW - datetime(2026, 9, 1)).days + 1
        self.assertEqual(limits, [10 * days + 10])
        self.assertFalse(any("reached its cap" in line for line in lines))  # the listing simply ended

        # A normal run uses the same 10 entries a day (80 for 7 days); a full listing of recent videos ...
        lines.clear()
        listing = {CHIP: [_listed(f"r{i}") for i in range(90)]}  # 90 in 7 days: more than the cap of 80
        pages = {f"r{i}": _page(f"r{i}", 1 + i) for i in range(90)}
        fetch(self.engine, self.settings, FakeVideos(listing, pages), AnyCaptions({}), now=NOW,
              progress=lines.append)
        cut = [line for line in lines if "reached its cap" in line]
        self.assertEqual(len(cut), 1)  # ... is a cut-off when no entry older than the window was reached
        lines.clear()
        listing = {CHIP: [_listed(f"r{i}") for i in range(5)] + [_listed("old")]}  # the window's start is reached
        pages["old"] = _page("old", 24 * 20)
        fetch(self.engine, self.settings, FakeVideos(listing, pages), AnyCaptions({}), now=NOW, days=1,
              progress=lines.append)
        self.assertFalse(any("reached its cap" in line for line in lines))


class ExtractionOrderTest(unittest.TestCase):
    def test_forward_videos_are_extracted_first(self):
        engine = create_db_engine(":memory:", 1000)
        init_db(engine)
        try:
            store.sync_channels(engine, SETTINGS.sentiment.channels)
            channel = SETTINGS.sentiment.channels[0]
            for video_id, published in (("backfill", NOW - timedelta(days=60)), ("forward_late", NOW - timedelta(days=1)),
                                        ("forward_early", NOW - timedelta(days=3))):
                store.insert_video(engine, {"video_id": video_id, "channel_id": channel.channel_id, "title": video_id,
                                            "transcript_status": "PENDING", "published_at": published})
                store.save_transcript(engine, video_id, "en", False, [{"start": 0, "duration": 1, "text": "x"}], None, 1)
            order = videos_to_extract(engine, "claude_code", "haiku", 1, forward_since=NOW - timedelta(days=7))
            self.assertEqual(list(order["video_id"]), ["forward_early", "forward_late", "backfill"])
            self.assertEqual(list(videos_to_extract(engine, "claude_code", "haiku", 1)["video_id"]),
                             ["backfill", "forward_early", "forward_late"])
        finally:
            engine.dispose()


SESSIONS = pd.bdate_range("2024-01-02", periods=400)
STOCKS = [f"S{i}" for i in range(10)]


def prices(rates):
    """Each stock grows at its own constant daily rate; SPY at the average."""
    t = np.arange(len(SESSIONS))[:, None]
    opens = pd.DataFrame(100 * np.exp(np.asarray(rates)[None, :] * t), index=SESSIONS, columns=STOCKS)
    opens["SPY"] = 100 * np.exp(np.mean(rates) * np.arange(len(SESSIONS)))
    return Prices(open=opens, close=opens * 1.001)


def opinion(i, creator, ticker, stance, session):
    return {"opinion_id": i, "item_id": f"v{i}", "published_at": SESSIONS[session] - timedelta(hours=10),
            "effective_date": SESSIONS[session], "ticker": ticker, "asset_type": "STOCK", "stance": stance,
            "horizon": "MEDIUM", "conviction": 2, "creator": creator}


class ScoreTest(unittest.TestCase):
    def test_window_excess_against_the_pool_average(self):
        p = prices(np.linspace(0, 0.009, 10) / 10)
        excess = window_excess(p.open[STOCKS], 20)
        returns = p.open[STOCKS].iloc[20] / p.open[STOCKS].iloc[0] - 1
        self.assertAlmostEqual(excess.iloc[0]["S9"], returns["S9"] - returns.mean())
        self.assertTrue(excess.iloc[-20:].isna().all().all())  # the window has not ended

    def test_counting_one_call_per_window(self):
        frame = pd.DataFrame([opinion(i, "A", "S1", "BULLISH", s) for i, s in enumerate([10, 12, 29, 31, 60])])
        frame["position"] = [10, 12, 29, 31, 60]
        self.assertEqual(list(counted(frame, 20)["position"]), [10, 31, 60])
        frame.loc[1, "stance"] = "BEARISH"  # another stance is another key
        self.assertEqual(list(counted(frame, 20)["position"]), [10, 12, 31, 60])

    def test_a_good_picker_beats_random_picks_and_a_bad_one_does_not(self):
        p = prices(np.linspace(-0.002, 0.002, 10))  # S9 always best, S0 always worst
        rows = [opinion(i, "good", "S9", "BULLISH", 300 + 25 * (i % 3)) for i in range(3)]
        rows += [opinion(10 + i, "good", "S0", "BEARISH", 300 + 25 * (i % 3)) for i in range(3)]
        rows += [opinion(20 + i, "bad", "S0", "BULLISH", 300 + 25 * (i % 3)) for i in range(3)]
        rows += [opinion(30, "good", "S5", "BULLISH", 395)]  # its window has not ended
        scores = score_horizon(pd.DataFrame(rows), p, STOCKS, 20, permutations=2000, seed=1)
        table = scores.table
        self.assertEqual(scores.pending, 1)
        self.assertEqual(int(table.loc["good", "opinions"]), 6)
        self.assertEqual(table.loc["good", "hit rate"], 1.0)
        self.assertLess(table.loc["good", "p (mean)"], 0.01)
        self.assertGreater(table.loc["bad", "p (mean)"], 0.9)
        self.assertEqual(int(table.loc[ALL_CREATORS, "opinions"]), 9)
        formatted = _format(table, min_samples=30)
        self.assertEqual(formatted.loc["good", "hit rate"], "-")  # below 30: the count only
        self.assertEqual(formatted.loc["good", "opinions"], 6)


class ScorecardCommandTest(unittest.TestCase):
    """The command on the small stored market with hand-made opinions, one of them after scored_until."""

    def test_the_scorecard_scores_only_up_to_scored_until(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        db = root / "card.db"
        build_market_db(db, small_settings())
        engine = create_db_engine(db, 5000)
        settings_document = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        settings_document["market"]["symbols"] = POOL
        settings_document["market"]["market_symbols"] = ["SPY", "QQQ", "^VIX"]
        settings_document["sentiment"]["summary"]["reports_dir"] = (root / "reports").as_posix()
        settings_document["sentiment"]["evaluation"].update(scored_until=str(DATES[400].date()),
                                                             backfill_from=str(DATES[100].date()),
                                                             permutations=200)
        config = root / "settings.yaml"
        config.write_text(yaml.safe_dump(settings_document, sort_keys=False), encoding="utf-8")
        settings = load_settings(config)
        try:
            store.sync_channels(engine, settings.sentiment.channels)
            channel = settings.sentiment.channels[0]
            rows = []
            for i, (session, model) in enumerate([(300, "haiku"), (350, "haiku"), (390, "sonnet"), (450, "haiku")]):
                rows.append({"source": "youtube", "item_id": f"v{i}", "channel_id": channel.channel_id,
                             "published_at": DATES[session].to_pydatetime() - timedelta(hours=10),
                             "effective_date": DATES[session].date(), "ticker": "AAPL", "name_as_said": "Apple",
                             "asset_type": "STOCK", "stance": "BULLISH", "horizon": "MEDIUM", "conviction": 2,
                             "action": "NONE", "quote": "q", "verification": "VERIFIED", "backend": "claude_code",
                             "model": model, "prompt_version": 1, "extracted_at": datetime(2026, 10, 1)})
            with engine.begin() as conn:
                conn.execute(tables.sentiment_opinions.insert(), rows)
            loaded = load_scored_opinions(engine, settings)
            self.assertEqual(sorted(loaded["item_id"]), ["v0", "v1"])  # sonnet and after scored_until left out
        finally:
            engine.dispose()
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(["--config", str(config), "sentiment", "scorecard", "--db", str(db)])
        self.assertEqual(code, 0, errors.getvalue())
        reports = list((root / "reports").glob("scorecard-*.md"))
        self.assertEqual(len(reports), 1)
        text = reports[0].read_text(encoding="utf-8")
        for part in ("Exploration only, decides nothing", "## 5-session window", "## 20-session window",
                     "## Creators (descriptive)", "| AAPL | 2 | yes |"):
            self.assertIn(part, text)


if __name__ == "__main__":
    unittest.main()
