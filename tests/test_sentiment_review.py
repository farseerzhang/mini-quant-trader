"""`quant sentiment review-export`: disagreements between an extractor and the labels."""

import csv
import dataclasses
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from quant.persistence.database import create_db_engine, init_db
from quant.sentiment import store
from quant.sentiment.extract import Line, run_extract, videos_to_extract
from quant.sentiment.gold import CSV_ENCODING, OPINION_COLUMNS
from quant.sentiment.review import build_review, write_review
from settings_helper import SETTINGS
from test_sentiment_extract import BOOK, FakeBackend, _row

TEXT = [Line(0, "今天聊英伟达和AMD。"), Line(30, "我觉得英伟达长期还是会涨，我又加仓了。"), Line(60, "AMD 我暂时不看。"),
        Line(90, "苹果我觉得会跌。")]


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        gold = dataclasses.replace(SETTINGS.sentiment.gold, labels_dir=str(root / "labels"),
                                   transcripts_dir=str(root / "data" / "transcripts"))
        extraction = dataclasses.replace(SETTINGS.sentiment.extraction, model="fake")
        self.settings = dataclasses.replace(SETTINGS, sentiment=dataclasses.replace(
            SETTINGS.sentiment, gold=gold, extraction=extraction))
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)
        store.sync_channels(self.engine, SETTINGS.sentiment.channels)
        channel = SETTINGS.sentiment.channel("@NaNaShuoMeiGu")
        store.insert_video(self.engine, {"video_id": "v1", "channel_id": channel.channel_id, "title": "t",
                                         "published_at": datetime(2026, 10, 9, 22), "transcript_status": "PENDING"})
        segments = [{"start": float(line.start), "duration": 5.0, "text": line.text} for line in TEXT]
        store.save_transcript(self.engine, "v1", "zh-Hans", False, segments, segments, 1)
        labels = root / "labels"
        labels.mkdir()
        (labels / "videos.csv").write_text("video_id,status\nv1,done\n", encoding=CSV_ENCODING)
        with (labels / "opinions.csv").open("w", encoding=CSV_ENCODING, newline="") as target:
            writer = csv.DictWriter(target, fieldnames=OPINION_COLUMNS)
            writer.writeheader()
            writer.writerow({"video_id": "v1", "ticker": "NVDA", "stance": "BULLISH", "horizon": "LONG",
                             "conviction": "2", "action": "BOUGHT", "time": "00:30"})
            writer.writerow({"video_id": "v1", "ticker": "AMD", "stance": "BULLISH", "horizon": "LONG",
                             "conviction": "2", "action": "NONE", "time": "01:00"})
            writer.writerow({"video_id": "v1", "ticker": "TSLA", "stance": "BEARISH", "horizon": "SHORT",
                             "conviction": "1", "action": "NONE", "time": "01:30", "note": "tesla?"})

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def extract(self):
        backend = FakeBackend([{"opinions": [
            _row(),  # NVDA bullish: agrees
            _row(ticker="AMD", stance="NEUTRAL", quote="AMD 我暂时不看。"),  # direction differs
            _row(ticker="AAPL", stance="BEARISH", quote="苹果我觉得会跌。"),  # only the extractor
        ]}])
        videos = videos_to_extract(self.engine, backend.name, backend.model, 1)
        run_extract(self.engine, self.settings, backend, BOOK, videos)

    def test_cases_and_agreement(self):
        self.extract()
        frame, agreement = build_review(self.engine, self.settings)
        kinds = dict(zip(frame["ticker"], frame["kind"]))
        self.assertEqual(kinds, {"AMD": "direction", "AAPL": "only_extractor", "TSLA": "only_reference"})
        self.assertEqual((agreement.reference_pairs, agreement.extractor_pairs, agreement.both,
                          agreement.same_direction), (3, 3, 2, 1))
        amd = frame[frame["ticker"] == "AMD"].iloc[0]
        self.assertEqual(amd["extractor_quote"], "AMD 我暂时不看。")
        self.assertNotIn("nan", amd["extractor"])  # no condition: nothing printed
        self.assertEqual(amd["link"], "https://youtu.be/v1?t=60")
        self.assertIn("[01:00] AMD 我暂时不看。", amd["transcript_around"])
        tsla = frame[frame["ticker"] == "TSLA"].iloc[0]
        self.assertEqual((tsla["reference_note"], tsla["link"]), ("tesla?", "https://youtu.be/v1?t=90"))
        self.assertEqual(list(frame["case"]), [1, 2, 3])

        path = write_review(frame, self.settings, "fake")
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        with self.assertRaisesRegex(ValueError, "already exists"):
            write_review(frame, self.settings, "fake")  # never overwrites your verdicts

    def test_needs_the_extraction_first(self):
        with self.assertRaisesRegex(ValueError, "not extracted"):
            build_review(self.engine, self.settings)


if __name__ == "__main__":
    unittest.main()
