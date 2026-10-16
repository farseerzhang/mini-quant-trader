"""`quant sentiment fetch` (sentiment_proposal.md S1): offline, with fake YouTube sources."""

import dataclasses
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest import mock

import yaml
from sqlalchemy import select

from quant import cli
from quant.config.settings import DEFAULT_CONFIG_PATH, ConfigError, load_settings
from quant.persistence import tables as core_tables
from quant.persistence.database import create_db_engine, init_db
from quant.sentiment import tables
from quant.sentiment.fetch import fetch
from quant.sentiment.sources import ListedVideo, MembersOnlyError, SourceError, VideoDetails
from quant.sentiment.transcripts import (
    NoTranscriptError,
    Transcript,
    TranscriptBlockedError,
    TranscriptError,
    to_simplified,
)
from settings_helper import SETTINGS

PROJECT_SETTINGS = Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_PATH
NOW = datetime(2026, 10, 13, 22, 0)
CHIP = "@chipstockinvestor"
NICK = "@nick_stockcafe"


def _settings(**channels_only):
    """Project settings without request delays; optionally only some channels."""
    sentiment = SETTINGS.sentiment
    channels = tuple(c for c in sentiment.channels if not channels_only or c.handle in channels_only["handles"])
    return dataclasses.replace(SETTINGS, sentiment=dataclasses.replace(
        sentiment, channels=channels, request_delay_seconds=(0.0, 0.0)))


class FakeVideos:
    """A channel listing (newest first) and video pages; records every request."""

    def __init__(self, listings, pages):
        self.listings, self.pages = listings, pages
        self.detail_requests = []

    def list_channel(self, handle, limit):
        listing = self.listings.get(handle, [])
        if isinstance(listing, Exception):
            raise listing
        return listing[:limit]

    def details(self, video_id):
        self.detail_requests.append(video_id)
        page = self.pages[video_id]
        if isinstance(page, list):  # a sequence of outcomes, one per request
            page = page.pop(0)
        if isinstance(page, Exception):
            raise page
        return page


class FakeCaptions:
    def __init__(self, outcomes):
        self.outcomes = outcomes  # video_id -> Transcript | Exception | list of these
        self.requests = []

    def fetch(self, video_id, languages):
        self.requests.append((video_id, languages))
        outcome = self.outcomes[video_id]
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _page(video_id, hours_ago, duration=1500, live_status="not_live", channel_id="UC3aD-gfmHV_MhMmcwyIu1wA"):
    return VideoDetails(video_id=video_id, channel_id=channel_id, title=f"title {video_id}",
                        published_at=NOW - timedelta(hours=hours_ago), duration_seconds=duration,
                        live_status=live_status, availability="public")


def _listed(video_id, duration=1500, availability=None):
    return ListedVideo(video_id=video_id, title=f"title {video_id}", duration_seconds=duration,
                       availability=availability)


def _captions(text="I bought more NVDA today", language="en", generated=False):
    return Transcript(language=language, is_generated=generated,
                      segments=[{"start": 0.0, "duration": 4.2, "text": text}])


class SentimentDb(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)

    def tearDown(self):
        self.engine.dispose()

    def videos(self):
        with self.engine.connect() as conn:
            rows = conn.execute(select(tables.sentiment_videos)).mappings().all()
        return {row["video_id"]: dict(row) for row in rows}

    def transcripts(self):
        with self.engine.connect() as conn:
            rows = conn.execute(select(tables.sentiment_transcripts)).mappings().all()
        return {row["video_id"]: dict(row) for row in rows}


class FetchTest(SentimentDb):
    def channel_listing(self):
        return {CHIP: [
            _listed("members", availability="subscriber_only"),
            _listed("new1"),
            _listed("short", duration=45),
            _listed("replay"),
            _listed("upcoming", duration=None),
            _listed("nocaps"),
            _listed("old"),
            _listed("older"),
        ]}

    def channel_pages(self):
        return {
            "new1": _page("new1", 5),
            "replay": _page("replay", 20, live_status="was_live"),
            "upcoming": _page("upcoming", -3, live_status="is_upcoming"),
            "nocaps": _page("nocaps", 30),
            "old": _page("old", 24 * 20),
            "older": _page("older", 24 * 25),
        }

    def run_fetch(self, videos, captions, **kwargs):
        return fetch(self.engine, _settings(handles={CHIP}), videos, captions, now=NOW, **kwargs)

    def test_statuses_after_one_run(self):
        videos = FakeVideos(self.channel_listing(), self.channel_pages())
        captions = FakeCaptions({"new1": _captions(), "nocaps": NoTranscriptError("TranscriptsDisabled")})
        result = self.run_fetch(videos, captions)

        stored = self.videos()
        self.assertEqual({k: v["transcript_status"] for k, v in stored.items()},
                         {"members": "SKIPPED", "new1": "OK", "short": "SKIPPED", "replay": "SKIPPED",
                          "nocaps": "NO_TRANSCRIPT"})
        self.assertEqual(stored["members"]["note"], "members-only")
        self.assertEqual(stored["short"]["note"], "short")
        self.assertEqual(stored["replay"]["note"], "live replay")
        self.assertIsNone(stored["members"]["published_at"])
        self.assertEqual(stored["new1"]["published_at"], NOW - timedelta(hours=5))
        # Members-only videos and Shorts are skipped from the listing alone; the listing stops at "old".
        self.assertEqual(videos.detail_requests, ["new1", "replay", "upcoming", "nocaps", "old"])
        self.assertTrue(result.ok)
        self.assertEqual((result.transcripts_ok, result.no_transcript), (1, 1))

        transcript = self.transcripts()["new1"]
        self.assertEqual(json.loads(transcript["segments_json"])[0]["text"], "I bought more NVDA today")
        self.assertIsNone(transcript["segments_simplified_json"])  # English channel
        self.assertEqual(transcript["char_count"], len("I bought more NVDA today"))
        self.assertEqual(captions.requests[0][1], ("en", "en-US", "en-GB"))

    def test_progress_lines(self):
        lines = []
        captions = FakeCaptions({"new1": _captions(), "nocaps": [TranscriptError("timeout"), _captions()]})
        self.run_fetch(FakeVideos(self.channel_listing(), self.channel_pages()), captions, progress=lines.append)
        text = "\n".join(lines)
        for expected in ("[1/1] @chipstockinvestor (Chip Stock Investor)",
                         "8 listed, 0 already stored",
                         "skipped (members-only): title members",
                         "new: title new1",
                         "is upcoming, later run: title upcoming",
                         "before the lookback: done",
                         "Captions: 2 video(s) waiting; fetching 2 this run",
                         "[1/2] @chipstockinvestor",
                         "failed (timeout), retrying",
                         "[2/2] @chipstockinvestor",
                         "OK (en, 24 characters)"):
            self.assertIn(expected, text)

    def test_rerun_writes_no_duplicates_and_reads_nothing_twice(self):
        captions = FakeCaptions({"new1": _captions(), "nocaps": NoTranscriptError("NoTranscriptFound")})
        self.run_fetch(FakeVideos(self.channel_listing(), self.channel_pages()), captions)
        before = self.videos()

        videos = FakeVideos(self.channel_listing(), self.channel_pages())
        captions.requests.clear()
        lines = []
        result = self.run_fetch(videos, captions, progress=lines.append)
        self.assertIn("    no new videos", lines)
        self.assertFalse(any("reading the new ones" in line for line in lines))

        stable = lambda rows: {k: {c: v for c, v in row.items() if c not in ("attempts", "updated_at")}
                               for k, row in rows.items()}
        self.assertEqual(stable(self.videos()), stable(before))  # same rows, nothing overwritten
        self.assertEqual(self.videos()["nocaps"]["attempts"], 2)  # asked again: still recent
        self.assertEqual(result.channels[0].new, 0)
        # Only the stream that has not happened yet, and the first video past the lookback (where it stops).
        self.assertEqual(videos.detail_requests, ["upcoming", "old"])
        # Captions already stored are never asked again; only the recent video without captions is.
        self.assertEqual([r[0] for r in captions.requests], ["nocaps"])

    def test_no_captions_are_asked_again_only_while_recent(self):
        captions = FakeCaptions({"new1": NoTranscriptError("NoTranscriptFound"),
                                 "nocaps": NoTranscriptError("NoTranscriptFound")})
        self.run_fetch(FakeVideos(self.channel_listing(), self.channel_pages()), captions)
        captions.requests.clear()
        captions.outcomes["new1"] = _captions()  # the captions appeared later

        # Two days later: new1 (2 days 5 h old) is retried and now has captions; nocaps (3 days 6 h) is not.
        later = NOW + timedelta(days=2)
        fetch(self.engine, _settings(handles={CHIP}), FakeVideos(self.channel_listing(), self.channel_pages()),
              captions, now=later)
        self.assertEqual([r[0] for r in captions.requests], ["new1"])
        self.assertEqual(self.videos()["new1"]["transcript_status"], "OK")
        self.assertEqual(self.videos()["nocaps"]["transcript_status"], "NO_TRANSCRIPT")

    def test_retries_come_after_videos_not_tried_yet(self):
        settings = _settings(handles={CHIP})
        settings = dataclasses.replace(settings, sentiment=dataclasses.replace(
            settings.sentiment, max_captions_per_run=1))
        listing = {CHIP: [_listed("fresh"), _listed("nocaps"), _listed("old")]}
        pages = {"fresh": _page("fresh", 1), "nocaps": _page("nocaps", 30), "old": _page("old", 24 * 20)}
        captions = FakeCaptions({"nocaps": NoTranscriptError("NoTranscriptFound"), "fresh": _captions()})
        fetch(self.engine, settings, FakeVideos({CHIP: listing[CHIP][1:]}, pages), captions, now=NOW)
        captions.requests.clear()

        fetch(self.engine, settings, FakeVideos(listing, pages), captions, now=NOW)
        self.assertEqual([r[0] for r in captions.requests], ["fresh"])  # not the older retry

    def test_a_run_after_failures_fills_the_gaps(self):
        pages = self.channel_pages()
        pages["new1"] = [SourceError("timeout"), _page("new1", 5)]
        captions = FakeCaptions({"new1": _captions(),
                                 "nocaps": [TranscriptError("timeout"), TranscriptError("timeout"), _captions()]})
        first = self.run_fetch(FakeVideos(self.channel_listing(), pages), captions)

        self.assertFalse(first.ok)
        self.assertNotIn("new1", self.videos())  # not stored: listed and read again next run
        self.assertEqual(self.videos()["nocaps"]["transcript_status"], "FAILED")
        self.assertEqual(self.videos()["nocaps"]["attempts"], 2)  # one retry

        second = self.run_fetch(FakeVideos(self.channel_listing(), pages), captions)
        self.assertTrue(second.ok)
        self.assertEqual(self.videos()["new1"]["transcript_status"], "OK")
        self.assertEqual(self.videos()["nocaps"]["transcript_status"], "OK")
        self.assertEqual(set(self.transcripts()), {"new1", "nocaps"})

    def test_blocked_stops_the_run_and_keeps_videos_pending(self):
        # Captions are fetched oldest first: nocaps, then new1.
        captions = FakeCaptions({"nocaps": TranscriptBlockedError("IpBlocked"), "new1": _captions()})
        result = self.run_fetch(FakeVideos(self.channel_listing(), self.channel_pages()), captions)

        self.assertIsNotNone(result.blocked)
        self.assertFalse(result.ok)
        self.assertEqual(len(captions.requests), 1)  # nothing after the block
        stored = self.videos()
        self.assertEqual(stored["new1"]["transcript_status"], "PENDING")
        self.assertEqual(stored["nocaps"]["transcript_status"], "PENDING")

    def test_captions_per_run_are_capped_oldest_first(self):
        settings = _settings(handles={CHIP})
        settings = dataclasses.replace(settings, sentiment=dataclasses.replace(
            settings.sentiment, max_captions_per_run=1))
        captions = FakeCaptions({"new1": _captions(), "nocaps": _captions()})
        first = fetch(self.engine, settings, FakeVideos(self.channel_listing(), self.channel_pages()), captions,
                      now=NOW)
        self.assertEqual([r[0] for r in captions.requests], ["nocaps"])  # older than new1
        self.assertEqual(first.captions_left, 1)
        self.assertTrue(first.ok)
        self.assertEqual(self.videos()["new1"]["transcript_status"], "PENDING")

        second = fetch(self.engine, settings, FakeVideos(self.channel_listing(), self.channel_pages()), captions,
                       now=NOW)
        self.assertEqual(second.captions_left, 0)
        self.assertEqual(self.videos()["new1"]["transcript_status"], "OK")

    def test_members_only_found_on_the_video_page(self):
        listing = {CHIP: [_listed("hidden"), _listed("old")]}
        pages = {"hidden": MembersOnlyError("Join this channel"), "old": _page("old", 24 * 20)}
        self.run_fetch(FakeVideos(listing, pages), FakeCaptions({}))
        self.assertEqual(self.videos()["hidden"]["transcript_status"], "SKIPPED")
        self.assertEqual(self.videos()["hidden"]["note"], "members-only")

    def test_listing_failure_is_reported_and_other_channels_continue(self):
        settings = _settings(handles={CHIP, NICK})
        nick_id = settings.sentiment.channel(NICK).channel_id
        videos = FakeVideos({CHIP: SourceError("cannot list"), NICK: [_listed("n1")]},
                            {"n1": _page("n1", 2, channel_id=nick_id)})
        result = fetch(self.engine, settings, videos, FakeCaptions({"n1": _captions("我買了", "zh-TW")}), now=NOW)
        self.assertEqual(result.errors, ["cannot list"])
        self.assertEqual(self.videos()["n1"]["transcript_status"], "OK")

    def test_longer_days_reach_older_videos(self):
        captions = FakeCaptions({"new1": _captions(), "nocaps": _captions(), "old": _captions()})
        self.run_fetch(FakeVideos(self.channel_listing(), self.channel_pages()), captions, days=21)
        self.assertEqual(self.videos()["old"]["transcript_status"], "OK")
        self.assertNotIn("older", self.videos())

    def test_requests_are_paced_after_the_first(self):
        settings = _settings(handles={CHIP})
        settings = dataclasses.replace(settings, sentiment=dataclasses.replace(
            settings.sentiment, request_delay_seconds=(2.0, 5.0)))
        waits = []
        captions = FakeCaptions({"new1": _captions(), "nocaps": _captions()})
        fetch(self.engine, settings, FakeVideos(self.channel_listing(), self.channel_pages()), captions,
              now=NOW, sleep=waits.append, rng=random.Random(1))
        requests = 1 + 5 + 2  # listing, video pages, captions
        self.assertEqual(len(waits), requests - 1)
        self.assertTrue(all(2.0 <= w <= 5.0 for w in waits))


class ChineseTest(SentimentDb):
    def test_traditional_captions_are_also_stored_simplified(self):
        self.assertEqual(to_simplified("美股咖啡館，我買了輝達"), "美股咖啡馆，我买了辉达")
        settings = _settings(handles={NICK})
        nick_id = settings.sentiment.channel(NICK).channel_id
        videos = FakeVideos({NICK: [_listed("n1")]}, {"n1": _page("n1", 2, channel_id=nick_id)})
        captions = FakeCaptions({"n1": _captions("我買了輝達", "zh-TW")})
        fetch(self.engine, settings, videos, captions, now=NOW)

        transcript = self.transcripts()["n1"]
        self.assertEqual(transcript["language"], "zh-TW")
        self.assertEqual(json.loads(transcript["segments_json"])[0]["text"], "我買了輝達")
        self.assertEqual(json.loads(transcript["segments_simplified_json"])[0]["text"], "我买了辉达")
        self.assertIn("zh-TW", captions.requests[0][1])

    def test_channels_are_synchronized_from_config(self):
        fetch(self.engine, _settings(), FakeVideos({}, {}), FakeCaptions({}), now=NOW)
        with self.engine.connect() as conn:
            rows = conn.execute(select(tables.sentiment_channels)).mappings().all()
        self.assertEqual({r["handle"] for r in rows if r["active"]}, {c.handle for c in SETTINGS.sentiment.channels})


class SentimentSettingsTest(unittest.TestCase):
    def variant(self, mutate):
        raw = yaml.safe_load(PROJECT_SETTINGS.read_text(encoding="utf-8"))
        mutate(raw["sentiment"])
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "settings.yaml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        return path

    def test_project_channels(self):
        handles = [c.handle for c in SETTINGS.sentiment.channels]
        self.assertEqual(len(handles), 11)  # six from S0 / S1, five added 2026-10-16
        self.assertEqual(len(set(handles)), len(handles))
        self.assertEqual(SETTINGS.sentiment.channel("@CHIPSTOCKINVESTOR").language, "en")
        with self.assertRaises(ConfigError):
            SETTINGS.sentiment.channel("@nobody")

    def test_invalid_settings_are_rejected(self):
        cases = [
            lambda s: s["channels"][0].update(language="de"),
            lambda s: s["channels"][0].update(handle="chipstockinvestor"),
            lambda s: s["channels"].append(dict(s["channels"][0])),
            lambda s: s["channels"][0].pop("channel_id"),
            lambda s: s.update(request_delay_seconds=[5, 2]),
            lambda s: s.update(lookback_days=0),
            lambda s: s.update(max_captions_per_run=0),
            lambda s: s.update(no_captions_retry_days=-1),
            lambda s: s.update(include_live="no"),
        ]
        for mutate in cases:
            with self.subTest(mutate=mutate), self.assertRaises(ConfigError):
                load_settings(self.variant(mutate))


class SentimentCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name) / "quant.db")

    def tearDown(self):
        import gc

        gc.collect()
        self.temp.cleanup()

    def run_cli(self, *args):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["--config", str(PROJECT_SETTINGS), "sentiment", *args, "--db", self.db])
        return code, out.getvalue(), err.getvalue()

    def test_fetch_one_channel(self):
        listing = {CHIP: [_listed("v1"), _listed("old")]}
        pages = {"v1": _page("v1", 2), "old": _page("old", 24 * 400)}
        videos, captions = FakeVideos(listing, pages), FakeCaptions({"v1": _captions()})
        with mock.patch("quant.sentiment.sources.YtDlpSource", return_value=videos), \
                mock.patch("quant.sentiment.transcripts.YouTubeTranscriptSource", return_value=captions), \
                mock.patch("quant.sentiment.fetch.time.sleep"):
            code, out, err = self.run_cli("fetch", "--channel", CHIP, "--days", "30")
        self.assertEqual(code, 0, err)
        self.assertIn("new   1", out)
        self.assertIn("Captions: 1 fetched", out)
        engine = create_db_engine(self.db, 1000)
        try:
            with engine.connect() as conn:
                run = conn.execute(select(core_tables.pipeline_runs)).mappings().all()[-1]
        finally:
            engine.dispose()
        self.assertEqual((run["command"], run["status"]), ("sentiment-fetch", "OK"))

    def test_run_does_every_step_even_after_a_fetch_problem(self):
        calls = []

        def step(name, code):
            return lambda a, s: calls.append((name, a)) or code

        with mock.patch("quant.sentiment.cli._run_fetch", side_effect=step("fetch", 1)), \
                mock.patch("quant.sentiment.cli._run_extract", side_effect=step("extract", 0)), \
                mock.patch("quant.sentiment.cli._run_summarize", side_effect=step("summarize", 0)), \
                mock.patch("quant.sentiment.cli._run_report", side_effect=step("report", 0)):
            code, out, _ = self.run_cli("run", "--channel", CHIP, "--days", "3", "--limit", "2", "--model", "haiku")
        self.assertEqual(code, 1)  # fetch had a problem
        self.assertEqual([name for name, _ in calls], ["fetch", "extract", "summarize", "report"])
        fetch_args, extract_args = calls[0][1], calls[1][1]
        self.assertEqual((fetch_args.channel, fetch_args.days), (CHIP, 3))
        self.assertEqual((extract_args.limit, extract_args.model, extract_args.reextract), (2, "haiku", False))
        self.assertIn("fetch had problems (see above), extract OK, summarize OK, report OK", out)

    def test_unknown_channel(self):
        code, _, err = self.run_cli("fetch", "--channel", "@nobody")
        self.assertEqual(code, 1)
        self.assertIn("unknown sentiment channel", err)

    def test_missing_packages(self):
        with mock.patch("quant.sentiment.cli.importlib.util.find_spec", return_value=None):
            code, _, err = self.run_cli("fetch")
        self.assertEqual(code, 1)
        self.assertIn("pip install -e .", err)


if __name__ == "__main__":
    unittest.main()
