"""`quant sentiment gold-export` (sentiment_proposal.md section 4.6)."""

import csv
import dataclasses
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from quant.persistence.database import create_db_engine, init_db
from quant.sentiment import store
from quant.sentiment.gold import CSV_ENCODING, OPINION_COLUMNS, export_gold, transcript_lines
from settings_helper import SETTINGS

NOW = datetime(2026, 10, 13, 12, 0)


class GoldExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.labels, self.transcripts = root / "research" / "gold", root / "data" / "gold" / "transcripts"
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        store.sync_channels(self.engine, SETTINGS.sentiment.channels)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def settings(self, videos=10):
        gold = dataclasses.replace(SETTINGS.sentiment.gold, videos=videos, labels_dir=str(self.labels),
                                   transcripts_dir=str(self.transcripts))
        return dataclasses.replace(SETTINGS, sentiment=dataclasses.replace(SETTINGS.sentiment, gold=gold))

    def add_video(self, handle, video_id, text="I bought more NVDA", simplified=None, generated=False):
        channel = SETTINGS.sentiment.channel(handle)
        store.insert_video(self.engine, {"video_id": video_id, "channel_id": channel.channel_id,
                                         "title": f"title {video_id}", "published_at": NOW - timedelta(days=1),
                                         "duration_seconds": 1500, "transcript_status": "PENDING"})
        segments = [{"start": 0.0, "duration": 5.0, "text": text}, {"start": 31.0, "duration": 5.0, "text": "next"}]
        simple = [{**segments[0], "text": simplified}, segments[1]] if simplified else None
        store.save_transcript(self.engine, video_id, "zh-TW" if simplified else "en", generated, segments, simple, 1)

    def read_videos(self):
        with (self.labels / "videos.csv").open(encoding=CSV_ENCODING, newline="") as source:
            return list(csv.DictReader(source))

    def test_channels_take_turns(self):
        for handle, count in (("@FinancialEducation", 3), ("@chipstockinvestor", 3), ("@nick_stockcafe", 1)):
            for i in range(count):
                self.add_video(handle, f"{handle[1:4]}{i}")
        result = export_gold(self.engine, self.settings(videos=5))
        self.assertTrue(result.created)
        channels = [row["channel"] for row in self.read_videos()]
        self.assertEqual(channels, ["@FinancialEducation", "@chipstockinvestor", "@nick_stockcafe",
                                    "@FinancialEducation", "@chipstockinvestor"])
        self.assertEqual(len(list(self.transcripts.glob("*.txt"))), 5)

    def test_files_and_formats(self):
        self.add_video("@nick_stockcafe", "nick1", text="我買了輝達", simplified="我买了辉达")
        self.add_video("@FinancialEducation", "fin1", generated=True)
        export_gold(self.engine, self.settings(videos=2))

        rows = {row["video_id"]: row for row in self.read_videos()}
        self.assertEqual(rows["fin1"]["captions"], "auto")
        self.assertEqual(rows["nick1"]["link"], "https://youtu.be/nick1")
        self.assertEqual(rows["nick1"]["status"], "")
        self.assertTrue((self.labels / "videos.csv").read_bytes().startswith(b"\xef\xbb\xbf"))  # for Excel
        with (self.labels / "opinions.csv").open(encoding=CSV_ENCODING, newline="") as source:
            self.assertEqual(list(csv.reader(source)), [OPINION_COLUMNS])
        self.assertIn("own", (self.labels / "LABELING.md").read_text(encoding="utf-8"))

        text = (self.transcripts / "nick1.txt").read_text(encoding="utf-8")
        self.assertIn("[00:00] 我买了辉达", text)  # Simplified
        self.assertIn("shown in Simplified Chinese", text)
        self.assertIn("[00:31] next", text)

    def test_rerun_keeps_labels_and_rewrites_transcripts(self):
        for i in range(3):
            self.add_video("@chipstockinvestor", f"chip{i}")
        export_gold(self.engine, self.settings(videos=2))
        opinions = self.labels / "opinions.csv"
        opinions.write_text(",".join(OPINION_COLUMNS) + "\nchip0,NVDA,BULLISH,LONG,,2,BOUGHT,03:10,\n",
                            encoding=CSV_ENCODING)
        videos = (self.labels / "videos.csv").read_bytes()
        for path in self.transcripts.glob("*.txt"):
            path.unlink()
        self.add_video("@FinancialEducation", "fin9")  # a new video does not join the set

        result = export_gold(self.engine, self.settings(videos=2))
        self.assertFalse(result.created)
        self.assertEqual((self.labels / "videos.csv").read_bytes(), videos)
        self.assertIn("chip0,NVDA,BULLISH", opinions.read_text(encoding=CSV_ENCODING))
        self.assertEqual(sorted(p.stem for p in self.transcripts.glob("*.txt")), ["chip0", "chip1"])

    def test_add_extends_without_rewriting_your_rows(self):
        for i in range(4):
            self.add_video("@chipstockinvestor", f"chip{i}")
        self.add_video("@nick_stockcafe", "nick0", text="我買了", simplified="我买了")
        export_gold(self.engine, self.settings(videos=2))  # chip0, nick0
        videos = self.labels / "videos.csv"
        before = videos.read_bytes().replace(b",\r\n", b",done\r\n", 1)  # you labeled the first video
        videos.write_bytes(before)

        result = export_gold(self.engine, self.settings(videos=2), add=2)
        self.assertEqual(result.added, ["chip1", "chip2"])
        after = videos.read_bytes()
        self.assertTrue(after.startswith(before))  # earlier rows byte for byte
        self.assertEqual(after.count(b"\xef\xbb\xbf"), 1)
        self.assertEqual([r["video_id"] for r in self.read_videos()], ["chip0", "nick0", "chip1", "chip2"])
        self.assertEqual(self.read_videos()[0]["status"], "done")
        self.assertEqual(len(list(self.transcripts.glob("*.txt"))), 4)

        with self.assertRaisesRegex(ValueError, "only 1 video"):
            export_gold(self.engine, self.settings(videos=2), add=2)

    def test_add_needs_an_existing_set(self):
        self.add_video("@chipstockinvestor", "chip0")
        with self.assertRaisesRegex(ValueError, "no gold set"):
            export_gold(self.engine, self.settings(videos=1), add=1)

    def test_too_few_videos_is_refused(self):
        self.add_video("@chipstockinvestor", "chip0")
        with self.assertRaisesRegex(ValueError, "only 1 video"):
            export_gold(self.engine, self.settings(videos=10))
        self.assertFalse((self.labels / "videos.csv").exists())

    def test_transcript_lines(self):
        segments = [{"start": s, "duration": 2.0, "text": f"w{s}"} for s in (0, 10, 29, 30, 65, 3700)]
        self.assertEqual(transcript_lines(segments),
                         ["[00:00] w0 w10 w29", "[00:30] w30", "[01:05] w65", "[1:01:40] w3700"])


if __name__ == "__main__":
    unittest.main()
