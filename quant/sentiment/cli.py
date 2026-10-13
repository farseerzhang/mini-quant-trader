"""`quant sentiment ...` subcommands (sentiment_proposal.md section 7); registered in quant/cli.py.

Its packages (yt-dlp, youtube-transcript-api, opencc) are imported only when a subcommand runs,
so the other commands never depend on them.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys

from quant.config.settings import Settings
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db

log = logging.getLogger("quant.sentiment")

REQUIRED = {"yt_dlp": "yt-dlp", "youtube_transcript_api": "youtube-transcript-api",
            "opencc": "opencc-python-reimplemented"}


def add_parser(commands, db_help: str) -> None:
    sentiment = commands.add_parser(
        "sentiment", help="creator opinion factor (research only; not part of run-daily): fetch, ...")
    sub = sentiment.add_subparsers(dest="sentiment_command", required=True)
    fetch = sub.add_parser("fetch", help="list new videos of the configured channels and fetch their captions")
    fetch.add_argument("--days", type=int, help="look back this many days (default: sentiment.lookback_days)")
    fetch.add_argument("--channel", metavar="HANDLE", help="only this channel, e.g. @chipstockinvestor")
    fetch.add_argument("--db", help=db_help)
    gold = sub.add_parser(
        "gold-export",
        help="pick the gold-set videos and write the label files and readable transcripts (S2); "
        "never overwrites existing labels, only writes the transcripts again")
    gold.add_argument("--db", help=db_help)


def _error(message: str) -> None:
    print(message, file=sys.stderr)
    log.error(message)


def run(args: argparse.Namespace, settings: Settings) -> int:
    missing = [package for module, package in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        _error(f"Missing package(s) {', '.join(missing)}: install them with  pip install -e .")
        return 1
    if args.sentiment_command == "fetch":
        return _run_fetch(args, settings)
    if args.sentiment_command == "gold-export":
        return _run_gold_export(args, settings)
    raise ValueError(f"unknown sentiment command {args.sentiment_command!r}")


def _engine(settings: Settings, db_override: str | None):
    import quant.sentiment.tables  # noqa: F401  registers the sentiment tables before init_db

    engine = create_db_engine(db_override or settings.database.path, settings.database.busy_timeout_ms)
    init_db(engine)
    return engine


def _run_gold_export(args: argparse.Namespace, settings: Settings) -> int:
    from quant.sentiment.gold import export_gold

    engine = _engine(settings, args.db)
    try:
        try:
            result = export_gold(engine, settings)
        except ValueError as exc:
            _error(f"Gold export refused: {exc}")
            return 1
    finally:
        engine.dispose()
    if result.created:
        print(f"Picked {len(result.video_ids)} videos for the gold set:")
        print(f"  labels:      {result.labels_dir / 'videos.csv'}, {result.labels_dir / 'opinions.csv'} "
              f"(rules: {result.labels_dir / 'LABELING.md'})")
    else:
        print(f"Labels already exist in {result.labels_dir} (left unchanged); transcripts written again.")
    print(f"  transcripts: {result.transcripts_dir} ({len(result.video_ids) - len(result.missing)} files)")
    for video_id in result.missing:
        _error(f"  {video_id} is in videos.csv but has no stored transcript (run `quant sentiment fetch`)")
    return 0 if not result.missing else 1


def _run_fetch(args: argparse.Namespace, settings: Settings) -> int:
    from quant.sentiment.fetch import fetch
    from quant.sentiment.sources import YtDlpSource
    from quant.sentiment.transcripts import YouTubeTranscriptSource

    if args.days is not None and args.days < 1:
        _error("--days must be >= 1")
        return 1
    if args.channel:
        settings.sentiment.channel(args.channel)  # unknown handle -> ConfigError, reported by main
    def progress(line: str) -> None:
        print(line, flush=True)
        log.info("fetch: %s", line.strip())

    engine = _engine(settings, args.db)
    try:
        result = fetch(engine, settings, YtDlpSource(), YouTubeTranscriptSource(), days=args.days,
                       handle=args.channel, progress=progress)
        print("\nSummary")
        for channel in result.channels:
            print(f"  {channel.handle:22s} listed {channel.listed:3d}, new {channel.new:3d} "
                  f"(skipped {channel.skipped}: members-only, Shorts, live replays)"
                  + (f", {len(channel.errors)} error(s)" if channel.errors else ""))
        print(f"Captions: {result.transcripts_ok} fetched, {result.no_transcript} without captions, "
              f"{result.transcripts_failed} failed (retried next run)"
              + (f", {result.captions_left} left for the next run" if result.captions_left else ""))
        for error in result.errors:
            _error(f"  {error}")
        if result.blocked:
            _error(result.blocked)
        message = (f"{sum(c.new for c in result.channels)} new videos, {result.transcripts_ok} captions, "
                   f"{result.transcripts_failed} failed, {len(result.errors)} listing errors"
                   + (f"; stopped: {result.blocked}" if result.blocked else ""))
        queries.record_pipeline_run(engine, "sentiment-fetch", "fetch", "OK" if result.ok else "FAILED",
                                    message=message)
        return 0 if result.ok else 1
    finally:
        engine.dispose()
