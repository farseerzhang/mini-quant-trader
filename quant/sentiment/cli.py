"""`quant sentiment ...` subcommands (sentiment_proposal.md section 7); registered in quant/cli.py.

Its packages (yt-dlp, youtube-transcript-api, opencc) are imported only when a subcommand runs,
so the other commands never depend on them.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys

import pandas as pd

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
    gold.add_argument("--add", type=int, default=0, metavar="N",
                      help="extend the existing gold set by N videos not in it yet (appended to videos.csv)")
    gold.add_argument("--db", help=db_help)
    extract = sub.add_parser(
        "extract", help="extract opinions from stored transcripts with sentiment.extraction.backend (S2); "
        "only videos without opinions for the current backend, model and prompt version")
    extract.add_argument("--limit", type=int, metavar="N", help="at most N videos this run")
    extract.add_argument("--video", action="append", metavar="ID", help="only this video (repeatable)")
    extract.add_argument("--model", help="use this model instead of sentiment.extraction.model (e.g. haiku)")
    extract.add_argument("--reextract", action="store_true",
                         help="extract again videos already done with this backend, model and prompt version")
    extract.add_argument("--yes", action="store_true",
                         help="go ahead with more than sentiment.extraction.max_videos_per_run videos")
    extract.add_argument("--db", help=db_help)
    run = sub.add_parser(
        "run", help="fetch, then extract: new videos and captions, then opinions for every video without them "
        "(extract runs even if fetch was partly blocked)")
    run.add_argument("--days", type=int, help="fetch: look back this many days (default: sentiment.lookback_days)")
    run.add_argument("--channel", metavar="HANDLE", help="fetch: only this channel")
    run.add_argument("--limit", type=int, metavar="N", help="extract: at most N videos this run")
    run.add_argument("--model", help="extract: use this model instead of sentiment.extraction.model")
    run.add_argument("--yes", action="store_true",
                     help="extract: go ahead with more than sentiment.extraction.max_videos_per_run videos")
    run.add_argument("--db", help=db_help)


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
    if args.sentiment_command == "extract":
        return _run_extract(args, settings)
    if args.sentiment_command == "run":
        return _run_all(args, settings)
    raise ValueError(f"unknown sentiment command {args.sentiment_command!r}")


def _engine(settings: Settings, db_override: str | None):
    import quant.sentiment.tables  # noqa: F401  registers the sentiment tables before init_db

    engine = create_db_engine(db_override or settings.database.path, settings.database.busy_timeout_ms)
    init_db(engine)
    return engine


def _run_all(args: argparse.Namespace, settings: Settings) -> int:
    """`quant sentiment run`: fetch, then extract. Extract runs even when fetch had problems (a
    YouTube block, a failed listing): the videos already stored still get their opinions."""
    print("===== [1/2] fetch: new videos and captions =====", flush=True)
    fetched = _run_fetch(argparse.Namespace(days=args.days, channel=args.channel, db=args.db), settings)
    print("\n===== [2/2] extract: opinions =====", flush=True)
    extracted = _run_extract(argparse.Namespace(limit=args.limit, video=None, model=args.model, reextract=False,
                                                yes=args.yes, db=args.db), settings)
    words = {0: "OK", 1: "had problems (see above)"}
    print(f"\nsentiment run: fetch {words[fetched]}, extract {words[extracted]}")
    return 0 if fetched == 0 and extracted == 0 else 1


def _run_extract(args: argparse.Namespace, settings: Settings) -> int:
    from quant.sentiment.aliases import AliasError, TickerBook, load_aliases, load_listings
    from quant.sentiment.backends import BackendError, make_backend
    from quant.sentiment.extract import held_back_videos, plan, run_extract, videos_to_extract
    from quant.sentiment.prompts import system_prompt

    config = settings.sentiment.extraction
    model = args.model or config.model
    if args.limit is not None and args.limit < 1:
        _error("--limit must be >= 1")
        return 1
    try:
        book = TickerBook(load_aliases(config.aliases_path), load_listings(config.tickers_cache))
    except AliasError as exc:
        _error(f"Alias table: {exc}")
        return 1
    except OSError as exc:
        _error(f"Cannot download the US ticker list (needed once, then cached): {exc}")
        return 1

    def progress(line: str) -> None:
        print(line, flush=True)
        log.info("extract: %s", line.strip())

    engine = _engine(settings, args.db)
    try:
        videos = videos_to_extract(engine, config.backend, model, config.prompt_version, args.video, args.reextract)
        held = held_back_videos(settings.sentiment.gold.labels_dir)
        held_here = videos["video_id"].isin(held)
        videos = videos[~held_here]
        if args.limit:
            videos = videos.head(args.limit)
        system = system_prompt(config.prompt_version, book.prompt_table())
        pieces, tokens = plan(videos, settings, system)
        print(f"Backend {config.backend}, model {model}, prompt version {config.prompt_version}: "
              f"{len(videos)} video(s), {pieces} call(s), about {tokens:,} input tokens"
              + (f"; {int(held_here.sum())} held back until labeled" if held_here.any() else ""))
        if videos.empty:
            print("Nothing to extract.")
            return 0
        if len(videos) > config.max_videos_per_run and not args.yes:
            _error(f"That is more than sentiment.extraction.max_videos_per_run ({config.max_videos_per_run}); "
                   "use --limit N, or --yes to go ahead")
            return 1
        try:
            backend = make_backend(config.backend, model, config.call_timeout_seconds)
        except BackendError as exc:
            _error(f"Backend {config.backend}: {exc}")
            return 1
        result = run_extract(engine, settings, backend, book, videos, progress)
        opinions = [o for v in result.done for o in v.opinions]
        by = pd.Series([o.verification for o in opinions], dtype=object).value_counts().to_dict()
        print(f"\nSummary: {len(result.done)} of {result.planned} video(s) done, {result.failed} failed; "
              f"{len(opinions)} opinion(s): " + (", ".join(f"{k.lower()} {v}" for k, v in by.items()) or "none"))
        unresolved = sorted({o.name_as_said for o in opinions if o.verification == "UNRESOLVED"})
        if unresolved:
            print("Unresolved names (add them to the alias table if they are real stocks): " + ", ".join(unresolved))
        if result.stopped:
            _error(result.stopped)
        ok = not result.failed and not result.stopped
        queries.record_pipeline_run(engine, "sentiment-extract", "extract", "OK" if ok else "FAILED",
                                    message=f"{config.backend}/{model}/v{config.prompt_version}: "
                                    f"{len(result.done)} videos, {len(opinions)} opinions, {result.failed} failed"
                                    + (f"; {result.stopped}" if result.stopped else ""))
        return 0 if ok else 1
    finally:
        engine.dispose()


def _run_gold_export(args: argparse.Namespace, settings: Settings) -> int:
    from quant.sentiment.gold import export_gold

    if args.add < 0:
        _error("--add must not be negative")
        return 1
    engine = _engine(settings, args.db)
    try:
        try:
            result = export_gold(engine, settings, add=args.add)
        except ValueError as exc:
            _error(f"Gold export refused: {exc}")
            return 1
    finally:
        engine.dispose()
    if result.created:
        print(f"Picked {len(result.video_ids)} videos for the gold set:")
        print(f"  labels:      {result.labels_dir / 'videos.csv'}, {result.labels_dir / 'opinions.csv'} "
              f"(rules: {result.labels_dir / 'LABELING.md'})")
    elif result.added:
        print(f"Added {len(result.added)} videos to {result.labels_dir / 'videos.csv'} (your earlier rows unchanged):")
        for video_id in result.added:
            print(f"  {video_id}")
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
