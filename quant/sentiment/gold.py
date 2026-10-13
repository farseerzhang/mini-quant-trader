"""`quant sentiment gold-export`: the hand-labeled answer key (sentiment_proposal.md section 4.6).

The first run picks `sentiment.gold.videos` videos with stored captions and writes:

- <labels_dir>/videos.csv     the picked videos; you set `status` to done / done-none
- <labels_dir>/opinions.csv   empty with headers; you add one row per opinion
- <labels_dir>/LABELING.md    the labeling rules
- <transcripts_dir>/<video_id>.txt  readable transcripts (Chinese in Simplified)

The labels hold no transcript text, so labels_dir is committed; the transcripts are reading
copies in data/ (git-ignored) and can be deleted at any time. Later runs never touch the
label files: they only write the transcripts of the videos already in videos.csv again.

Picking: channels take turns in config order, so every channel is covered; within a channel
the videos are taken in video_id order, which is arbitrary but fixed (not chosen by anyone
who has read them). Label before seeing any extraction output of these videos.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, select

from quant.config.settings import Settings
from quant.sentiment import tables

VIDEO_COLUMNS = ["video_id", "channel", "title", "published", "language", "captions", "minutes", "link",
                 "transcript", "status"]
OPINION_COLUMNS = ["video_id", "ticker", "stance", "horizon", "condition", "conviction", "action", "time", "note"]
BLOCK_SECONDS = 30  # one transcript line per ~30 s
# utf-8 with a byte-order mark, so Excel shows Chinese correctly when you open the CSV files.
CSV_ENCODING = "utf-8-sig"

LABELING = """\
# Labeling the gold set

The answer key for opinion extraction (sentiment_proposal.md section 4.6). Label **before**
you see any extraction output for these videos.

For each video in `videos.csv`:

1. Read `<transcripts_dir>/<video_id>.txt` (the `transcript` column; skim it, and open the
   link if a passage is unclear).
2. Add one row to `opinions.csv` for every opinion **the creator holds themselves**.
3. Set the video's `status` in `videos.csv` to `done`, or `done-none` if it has no opinions.
   Only videos marked done / done-none are scored, so partial work counts.

## opinions.csv

| Column | Values |
|---|---|
| video_id | from videos.csv |
| ticker | US ticker, e.g. NVDA, BRK-B; ETFs and indices too (SPY, QQQ) |
| stance | BULLISH / BEARISH / NEUTRAL / CONDITIONAL |
| horizon | SHORT (weeks) / MEDIUM (months) / LONG (a year or more) / UNSPECIFIED |
| condition | only for CONDITIONAL, e.g. "buy below 140" |
| conviction | 1 mild, 2 clear, 3 strong (how strongly they say it) |
| action | BOUGHT / SOLD / HOLDING / NONE: what the creator says they did themselves |
| time | where it is said, mm:ss (or h:mm:ss) from the transcript |
| note | optional: anything you are unsure about |

## Rules

- Count only the creator's **own** view ("I think", "I'm buying", "I'd avoid").
- Do not count relayed views (analyst ratings, news, viewer questions) or passing mentions
  without a view.
- "Bearish short term, bullish long term" is **two rows** with different horizons.
- One row per stock and horizon per video; if they come back to it, keep their final view.
- When unsure, label your best guess and say so in `note`; do not skip it.

Back this folder up: it is hours of work, and every later prompt or model change is scored
against it. The transcripts folder can be deleted; `quant sentiment gold-export` writes it again.
"""


@dataclass
class GoldExport:
    labels_dir: Path
    transcripts_dir: Path
    created: bool  # False: the labels already existed; only transcripts were written
    video_ids: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # in videos.csv but no transcript stored


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def transcript_lines(segments: list[dict]) -> list[str]:
    """Join caption segments into lines of about BLOCK_SECONDS, each with its start time."""
    lines, start, words = [], None, []
    for segment in segments:
        if start is not None and segment["start"] - start >= BLOCK_SECONDS:
            lines.append(f"[{_clock(start)}] {' '.join(words)}")
            start, words = None, []
        if start is None:
            start = segment["start"]
        words.append(" ".join(segment["text"].split()))
    if words:
        lines.append(f"[{_clock(start)}] {' '.join(words)}")
    return lines


def _videos_with_transcripts(engine: Engine) -> pd.DataFrame:
    v, t, c = tables.sentiment_videos, tables.sentiment_transcripts, tables.sentiment_channels
    stmt = (
        select(v.c.video_id, v.c.title, v.c.published_at, v.c.duration_seconds, c.c.handle, c.c.name,
               c.c.language.label("channel_language"), t.c.language, t.c.is_generated, t.c.segments_json,
               t.c.segments_simplified_json)
        .join(t, t.c.video_id == v.c.video_id)
        .join(c, c.c.channel_id == v.c.channel_id)
        .where(v.c.transcript_status == "OK")
        .order_by(v.c.video_id)
    )
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(stmt).mappings().all(), columns=list(stmt.selected_columns.keys()))


def pick_videos(stored: pd.DataFrame, handles: list[str], count: int) -> pd.DataFrame:
    """Channels take turns in config order; within a channel, video_id order."""
    queues = {handle: list(stored.index[stored["handle"] == handle]) for handle in handles}
    picked: list = []
    while len(picked) < count and any(queues.values()):
        for handle in handles:
            if queues[handle] and len(picked) < count:
                picked.append(queues[handle].pop(0))
    return stored.loc[picked]


def _write_transcript(path: Path, video) -> None:
    simplified = video.segments_simplified_json if isinstance(video.segments_simplified_json, str) else None  # NaN
    segments = json.loads(simplified if simplified else video.segments_json)
    captions = "auto-generated" if video.is_generated else "manual"
    shown = " (shown in Simplified Chinese)" if simplified else ""
    header = [
        f"Title:    {video.title}",
        f"Channel:  {video.name} ({video.handle})",
        f"Published {video.published_at:%Y-%m-%d %H:%M} UTC",
        f"Captions: {video.language}, {captions}{shown}",
        f"Video:    https://youtu.be/{video.video_id}",
        "Label in opinions.csv, with the [mm:ss] times below.",
        "",
    ]
    path.write_text("\n".join(header + transcript_lines(segments)) + "\n", encoding="utf-8")


def _video_row(video, transcripts_dir: Path) -> dict:
    return {
        "video_id": video.video_id, "channel": video.handle, "title": video.title,
        "published": f"{video.published_at:%Y-%m-%d}", "language": video.language,
        "captions": "auto" if video.is_generated else "manual",
        "minutes": round(video.duration_seconds / 60) if video.duration_seconds else "",
        "link": f"https://youtu.be/{video.video_id}",
        "transcript": str(transcripts_dir / f"{video.video_id}.txt"), "status": "",
    }


def read_gold_video_ids(labels_dir: Path) -> list[str]:
    with (labels_dir / "videos.csv").open(encoding=CSV_ENCODING, newline="") as source:
        return [row["video_id"] for row in csv.DictReader(source) if row.get("video_id")]


def export_gold(engine: Engine, settings: Settings) -> GoldExport:
    gold = settings.sentiment.gold
    labels_dir, transcripts_dir = Path(gold.labels_dir), Path(gold.transcripts_dir)
    stored = _videos_with_transcripts(engine)
    exists = (labels_dir / "videos.csv").exists()
    if exists:
        wanted = read_gold_video_ids(labels_dir)
        chosen = stored[stored["video_id"].isin(wanted)]
    else:
        handles = [c.handle for c in settings.sentiment.channels]
        chosen = pick_videos(stored, handles, gold.videos)
        if len(chosen) < gold.videos:
            raise ValueError(f"only {len(chosen)} video(s) with captions are stored, {gold.videos} needed "
                             "(sentiment.gold.videos); run `quant sentiment fetch` first")
        wanted = list(chosen["video_id"])
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    for video in chosen.itertuples(index=False):
        _write_transcript(transcripts_dir / f"{video.video_id}.txt", video)
    if not exists:
        labels_dir.mkdir(parents=True, exist_ok=True)
        with (labels_dir / "videos.csv").open("w", encoding=CSV_ENCODING, newline="") as target:
            writer = csv.DictWriter(target, fieldnames=VIDEO_COLUMNS)
            writer.writeheader()
            writer.writerows(_video_row(video, transcripts_dir) for video in chosen.itertuples(index=False))
        if not (labels_dir / "opinions.csv").exists():
            with (labels_dir / "opinions.csv").open("w", encoding=CSV_ENCODING, newline="") as target:
                csv.writer(target).writerow(OPINION_COLUMNS)
        (labels_dir / "LABELING.md").write_text(LABELING.replace("<transcripts_dir>", str(transcripts_dir)),
                                                encoding="utf-8")
    found = set(chosen["video_id"])
    return GoldExport(labels_dir, transcripts_dir, created=not exists, video_ids=wanted,
                      missing=[video_id for video_id in wanted if video_id not in found])
