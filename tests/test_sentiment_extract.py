"""Opinion extraction (sentiment_proposal.md S2 steps 1-5): offline, with a fake backend."""

import dataclasses
import json
import os
import subprocess
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from sqlalchemy import select

from quant.data.calendar import first_session_opening_after
from quant.persistence.database import create_db_engine, init_db
from quant.sentiment import store, tables
from quant.sentiment.aliases import AliasError, Listing, TickerBook, _parse_directory, load_aliases, load_listings
from quant.sentiment.backends import BackendError, ClaudeCodeBackend, Reply, UsageLimitError
from quant.sentiment.extract import (
    InvalidReply,
    Line,
    check_opinion,
    chunks,
    held_back_videos,
    match_quote,
    merge,
    parse_opinions,
    run_extract,
    transcript_lines,
    videos_to_extract,
)
from quant.sentiment.prompts import OPINION_SCHEMA, PROMPTS, system_prompt
from settings_helper import SETTINGS

BOOK = TickerBook({"NVDA": ("英伟达", "Nvidia"), "HESAY": ("爱马仕",), "SPY": ("大盘",)},
                  {"NVDA": Listing("NVIDIA Corp", False), "BRK-B": Listing("Berkshire", False),
                   "SPY": Listing("SPDR S&P 500", True), "AMD": Listing("AMD", False)})
PIECE = [Line(0, "今天我们来聊聊英伟达。"), Line(30, "我觉得英伟达长期还是会涨，我又加仓了。"), Line(60, "AMD 我暂时不看。")]


def _row(**changes):
    row = {"ticker": "NVDA", "name_as_said": "英伟达", "asset_type": "STOCK", "stance": "BULLISH", "horizon": "LONG",
           "condition": "", "conviction": 2, "action": "BOUGHT", "quote": "我觉得英伟达长期还是会涨，我又加仓了。",
           "start_seconds": 30, "reasoning": "AI demand"}
    row.update(changes)
    return row


class PiecesTest(unittest.TestCase):
    def test_lines_and_overlapping_chunks(self):
        segments = [{"start": float(s), "duration": 3.0, "text": f"w{s}"} for s in range(0, 1500, 10)]
        lines = transcript_lines(segments)
        self.assertEqual(lines[0], Line(0, "w0 w10 w20"))
        pieces = chunks(lines, minutes=10, overlap_seconds=30)
        self.assertEqual(len(pieces), 3)
        self.assertEqual((pieces[0][0].start, pieces[0][-1].start), (0, 570))
        self.assertEqual(pieces[1][0].start, 570)  # starts 30 s before the first piece ends
        self.assertEqual(pieces[-1][-1].start, lines[-1].start)

    def test_quote_matching(self):
        self.assertEqual(match_quote("我觉得英伟达长期还是会涨，我又加仓了。", PIECE), (1.0, 30))
        # Punctuation and full-width differences do not matter.
        self.assertEqual(match_quote("我觉得英伟达长期还是会涨 我又加仓了", PIECE)[0], 1.0)
        score, start = match_quote("我觉得英伟达长期还会涨，又加仓了", PIECE)
        self.assertGreater(score, 0.85)
        self.assertEqual(start, 30)
        self.assertLess(match_quote("特斯拉明年肯定翻倍", PIECE)[0], 0.5)


class CheckTest(unittest.TestCase):
    def test_verified(self):
        opinion = check_opinion(_row(), PIECE, BOOK, 0.85)
        self.assertEqual((opinion.verification, opinion.start_seconds, opinion.quote_score), ("VERIFIED", 30, 1.0))

    def test_unverified_and_unresolved(self):
        self.assertEqual(check_opinion(_row(quote="完全不同的一句话说特斯拉"), PIECE, BOOK, 0.85).verification,
                         "UNVERIFIED")
        self.assertEqual(check_opinion(_row(ticker="ZZZZ"), PIECE, BOOK, 0.85).verification, "UNRESOLVED")
        self.assertEqual(check_opinion(_row(ticker="not a ticker"), PIECE, BOOK, 0.85).ticker, "UNKNOWN")
        # Known only through the alias table (an OTC ADR) still counts.
        self.assertEqual(check_opinion(_row(ticker="HESAY"), PIECE, BOOK, 0.85).verification, "VERIFIED")

    def test_ticker_normalized_and_etf(self):
        self.assertEqual(check_opinion(_row(ticker="brk.b"), PIECE, BOOK, 0.85).ticker, "BRK-B")
        self.assertEqual(check_opinion(_row(ticker="$SPY"), PIECE, BOOK, 0.85).asset_type, "ETF")

    def test_parse_rejects_bad_values(self):
        for bad in ({"stance": "CAUTIOUS"}, {"action": "BUY"}, {"conviction": 5}, {"quote": ""}):
            with self.subTest(bad=bad), self.assertRaises(InvalidReply):
                parse_opinions({"opinions": [_row(**bad)]})
        with self.assertRaises(InvalidReply):
            parse_opinions({"answer": []})
        self.assertEqual(parse_opinions({"opinions": [_row(condition="below 100")]})[0]["condition"], "")


class MergeTest(unittest.TestCase):
    def test_last_stance_wins_and_details_are_kept(self):
        early = check_opinion(_row(stance="BEARISH", conviction=3, action="NONE",
                                   quote="今天我们来聊聊英伟达。"), PIECE, BOOK, 0.85)
        late = check_opinion(_row(conviction=2, action="BOUGHT"), PIECE, BOOK, 0.85)
        other = check_opinion(_row(horizon="SHORT", stance="BEARISH", quote="AMD 我暂时不看。"), PIECE, BOOK, 0.85)
        merged = {(o.ticker, o.horizon): o for o in merge([late, early, other])}
        self.assertEqual(len(merged), 2)
        long = merged[("NVDA", "LONG")]
        self.assertEqual((long.stance, long.conviction, long.action), ("BULLISH", 3, "BOUGHT"))
        self.assertEqual(long.other_quotes, [{"stance": "BEARISH", "quote": "今天我们来聊聊英伟达。", "start_seconds": 0}])

    def test_duplicates_from_the_overlap_are_dropped(self):
        same = check_opinion(_row(), PIECE, BOOK, 0.85)
        self.assertEqual(merge([same, dataclasses.replace(same)])[0].other_quotes, [])


class FakeBackend:
    name, model = "claude_code", "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, user, schema):
        self.calls.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Reply(payload=reply, model="claude-fake-1", input_tokens=100, output_tokens=20)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        store.sync_channels(self.engine, SETTINGS.sentiment.channels)
        # Friday 2026-10-09 22:00 UTC: after the close, so it counts from Monday's open.
        self.published = datetime(2026, 10, 9, 22, 0)
        for video_id in ("v1", "v2"):
            self.add(video_id)

    def tearDown(self):
        self.engine.dispose()

    def add(self, video_id):
        channel = SETTINGS.sentiment.channel("@NaNaShuoMeiGu")
        store.insert_video(self.engine, {"video_id": video_id, "channel_id": channel.channel_id, "title": "t",
                                         "published_at": self.published, "transcript_status": "PENDING"})
        segments = [{"start": float(line.start), "duration": 5.0, "text": line.text} for line in PIECE]
        store.save_transcript(self.engine, video_id, "zh-Hans", False, segments, segments, 1)

    def run_videos(self, backend, **kwargs):
        videos = videos_to_extract(self.engine, backend.name, backend.model, 1, **kwargs)
        return run_extract(self.engine, SETTINGS, backend, BOOK, videos)

    def rows(self, table):
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(table)).mappings().all()]

    def test_opinions_and_extractions_are_stored_once(self):
        backend = FakeBackend([{"opinions": [_row()]}, {"opinions": []}])
        result = self.run_videos(backend)
        self.assertEqual(len(result.done), 2)
        opinions = self.rows(tables.sentiment_opinions)
        self.assertEqual(len(opinions), 1)
        self.assertEqual((opinions[0]["item_id"], opinions[0]["verification"], opinions[0]["effective_date"]),
                         ("v1", "VERIFIED", date(2026, 10, 12)))
        extractions = {r["video_id"]: r for r in self.rows(tables.sentiment_extractions)}
        self.assertEqual((extractions["v2"]["status"], extractions["v2"]["opinions"]), ("OK", 0))
        self.assertEqual(extractions["v1"]["model_resolved"], "claude-fake-1")
        self.assertIn("[30] 我觉得英伟达长期还是会涨", backend.calls[0])

        again = FakeBackend([])
        self.assertEqual(self.run_videos(again).planned, 0)  # both done, even the one without opinions

    def test_reextract_replaces_the_rows(self):
        self.run_videos(FakeBackend([{"opinions": [_row()]}, {"opinions": []}]))
        self.run_videos(FakeBackend([{"opinions": [_row(stance="NEUTRAL")]}]), video_ids=["v1"], reextract=True)
        opinions = self.rows(tables.sentiment_opinions)
        self.assertEqual([o["stance"] for o in opinions], ["NEUTRAL"])

    def test_retry_then_failed_and_retried_next_run(self):
        result = self.run_videos(FakeBackend([BackendError("x"), BackendError("y"), {"opinions": []}]))
        self.assertEqual(result.failed, 1)
        statuses = {r["video_id"]: r["status"] for r in self.rows(tables.sentiment_extractions)}
        self.assertEqual(statuses, {"v1": "FAILED", "v2": "OK"})
        self.run_videos(FakeBackend([{"opinions": [_row()]}]))
        statuses = {r["video_id"]: r["status"] for r in self.rows(tables.sentiment_extractions)}
        self.assertEqual(statuses, {"v1": "OK", "v2": "OK"})

    def test_failed_reextract_keeps_the_good_result(self):
        self.run_videos(FakeBackend([{"opinions": [_row()]}, {"opinions": []}]))
        self.run_videos(FakeBackend([BackendError("x"), BackendError("y")]), video_ids=["v1"], reextract=True)
        self.assertEqual(len(self.rows(tables.sentiment_opinions)), 1)

    def test_usage_limit_stops_and_saves_nothing_for_that_video(self):
        result = self.run_videos(FakeBackend([{"opinions": [_row()]}, UsageLimitError("limit")]))
        self.assertIsNotNone(result.stopped)
        self.assertEqual([r["video_id"] for r in self.rows(tables.sentiment_extractions)], ["v1"])

    def test_held_back_until_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "videos.csv").write_text("video_id,status\nv1,\nv2,done\nv3,done-none\n",
                                                     encoding="utf-8-sig")
            self.assertEqual(held_back_videos(directory), {"v1"})
            self.assertEqual(held_back_videos(Path(directory) / "missing"), set())


class EffectiveDateTest(unittest.TestCase):
    def test_next_open(self):
        utc = timezone.utc
        self.assertEqual(first_session_opening_after(datetime(2026, 10, 12, 12, 0, tzinfo=utc), "XNYS"),
                         date(2026, 10, 12))  # before Monday's open
        self.assertEqual(first_session_opening_after(datetime(2026, 10, 12, 15, 0, tzinfo=utc), "XNYS"),
                         date(2026, 10, 13))  # during the session
        self.assertEqual(first_session_opening_after(datetime(2026, 10, 10, 9, 0, tzinfo=utc), "XNYS"),
                         date(2026, 10, 12))  # Saturday


class AliasTest(unittest.TestCase):
    def test_project_alias_table_loads(self):
        aliases = load_aliases(Path(__file__).resolve().parents[1] / SETTINGS.sentiment.extraction.aliases_path)
        self.assertIn("ON", aliases)
        self.assertIn("英伟达", aliases["NVDA"])
        self.assertIn("NVDA: Nvidia", TickerBook(aliases, {}).prompt_table())

    def test_bad_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            for text in ("ON: [onsemi]\n", "NVDA: Nvidia\n", "not a ticker: [x]\n"):
                path = Path(directory, "a.yaml")
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(AliasError):
                    load_aliases(path)

    def test_directory_and_cache(self):
        text = ("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
                "BRK.B|Berkshire|N|BRK.B|N|40|N|BRK.B\nSPY|SPDR|P|SPY|Y|100|N|SPY\nZTST|Test|N|ZTST|N|100|Y|ZTST\n"
                "File Creation Time: 1013202618:00|||||||\n")
        listings = _parse_directory(text)
        self.assertEqual(set(listings), {"BRK-B", "SPY"})
        self.assertTrue(listings["SPY"].etf)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory, "t.json")
            now = datetime(2026, 10, 13, tzinfo=timezone.utc)
            downloads = []
            fetch = lambda: downloads.append(1) or listings  # noqa: E731
            load_listings(cache, now, fetch)
            load_listings(cache, now + timedelta(days=5), fetch)
            self.assertEqual(len(downloads), 1)  # cached
            fresh = load_listings(cache, now + timedelta(days=40), fetch)
            self.assertEqual(len(downloads), 2)  # refreshed after a month
            self.assertIn("BRK-B", fresh)

            def offline():
                raise OSError("no network")
            self.assertIn("SPY", load_listings(cache, now + timedelta(days=90), offline))  # old cache is kept


class ClaudeCodeTest(unittest.TestCase):
    def backend(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            return ClaudeCodeBackend("haiku", 60, executable="claude")

    def test_refuses_with_an_api_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x"}), self.assertRaises(BackendError):
            ClaudeCodeBackend("haiku", 60, executable="claude")

    def test_command_and_reply(self):
        backend = self.backend()
        out = {"is_error": False, "structured_output": {"opinions": []}, "usage": {"input_tokens": 5,
               "cache_read_input_tokens": 10, "output_tokens": 3}, "modelUsage": {"claude-haiku-5-5": {}}}
        done = subprocess.CompletedProcess([], 0, stdout=json.dumps(out), stderr="")
        with mock.patch("quant.sentiment.backends.subprocess.run", return_value=done) as run:
            reply = backend.complete("line one\nS&P \"two\"", "hello", OPINION_SCHEMA)
        command = run.call_args.args[0]
        self.assertIn("--system-prompt-file", command)
        self.assertNotIn("--system-prompt", command)
        prompt_file = command[command.index("--system-prompt-file") + 1]
        self.assertEqual(Path(prompt_file).read_text(encoding="utf-8"), "line one\nS&P \"two\"")
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertEqual(run.call_args.kwargs["input"], "hello")
        self.assertEqual((reply.payload, reply.model, reply.input_tokens), ({"opinions": []}, "claude-haiku-5-5", 15))

    def test_usage_limit_and_errors(self):
        backend = self.backend()
        cases = [
            (subprocess.CompletedProcess([], 1, stdout=json.dumps({"is_error": True, "result": "Claude usage limit "
                                                                   "reached, resets 5pm"}), stderr=""), UsageLimitError),
            (subprocess.CompletedProcess([], 1, stdout="", stderr="boom"), BackendError),
            (subprocess.CompletedProcess([], 0, stdout=json.dumps({"result": "not json"}), stderr=""), BackendError),
        ]
        for done, error in cases:
            with self.subTest(error=error), mock.patch("quant.sentiment.backends.subprocess.run", return_value=done), \
                    self.assertRaises(error):
                backend.complete("s", "u", OPINION_SCHEMA)


class PromptTest(unittest.TestCase):
    def test_versions(self):
        self.assertIn(SETTINGS.sentiment.extraction.prompt_version, PROMPTS)
        text = system_prompt(1, "NVDA: Nvidia")
        self.assertIn("NVDA: Nvidia", text)
        self.assertNotIn("{aliases}", text)
        with self.assertRaises(ValueError):
            system_prompt(99, "")


if __name__ == "__main__":
    unittest.main()
