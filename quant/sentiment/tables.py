"""Tables of the opinion factor (sentiment_proposal.md section 6), on the shared metadata.

quant/persistence/tables.py does not change: these tables are registered when this module
is imported, so `init_db` creates them for the sentiment commands. S1: the fetch tables;
S2: extractions and opinions; summaries and out-of-pool prices follow with S3 and S4.

Times are stored as naive UTC.
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
)

from quant.persistence.tables import metadata

TRANSCRIPT_STATUSES = ("PENDING", "OK", "NO_TRANSCRIPT", "FAILED", "SKIPPED")


# Synchronized from config (sentiment.channels), which is the source of truth, like `symbols`.
sentiment_channels = Table(
    "sentiment_channels",
    metadata,
    Column("channel_id", String, primary_key=True),
    Column("handle", String, nullable=False),
    Column("name", String, nullable=False),
    Column("language", String, nullable=False),  # en | zh
    Column("style_note", String),
    Column("active", Integer, nullable=False, server_default="1"),
    Column("added_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)


sentiment_videos = Table(
    "sentiment_videos",
    metadata,
    Column("video_id", String, primary_key=True),
    Column("channel_id", String, ForeignKey("sentiment_channels.channel_id"), nullable=False),
    Column("title", String),
    # When viewers could first see it (premiere time if any); NULL for videos skipped from
    # the channel listing alone (members-only, Shorts), whose page is never read.
    Column("published_at", DateTime),
    Column("duration_seconds", Integer),
    Column("live_status", String),  # yt-dlp: not_live | was_live | ...
    Column("availability", String),  # yt-dlp: public | subscriber_only | ...
    Column("transcript_status", String, nullable=False),
    Column("note", String),  # why SKIPPED / NO_TRANSCRIPT / FAILED
    Column("attempts", Integer, nullable=False, server_default="0"),  # caption fetch attempts
    Column("fetched_at", DateTime, nullable=False),  # first listed
    Column("updated_at", DateTime, nullable=False),
    CheckConstraint(
        "transcript_status IN (" + ", ".join(f"'{s}'" for s in TRANSCRIPT_STATUSES) + ")",
        name="ck_sentiment_videos_status",
    ),
)


# The raw captions are the source of truth (proposal 4.3): extraction can be re-run from them.
sentiment_transcripts = Table(
    "sentiment_transcripts",
    metadata,
    Column("video_id", String, ForeignKey("sentiment_videos.video_id"), primary_key=True),
    Column("language", String, nullable=False),  # caption language code, e.g. en, zh-Hans, zh-TW
    Column("is_generated", Integer, nullable=False),  # 1 = YouTube auto captions
    Column("segments_json", Text, nullable=False),  # [{start, duration, text}]
    Column("segments_simplified_json", Text),  # Chinese channels: the same in Simplified Chinese
    Column("char_count", Integer, nullable=False),
    Column("fetched_at", DateTime, nullable=False),
)


# One row per video and extractor: done (even with no opinions), or failed (retried next run).
sentiment_extractions = Table(
    "sentiment_extractions",
    metadata,
    Column("video_id", String, ForeignKey("sentiment_videos.video_id"), nullable=False),
    Column("backend", String, nullable=False),
    Column("model", String, nullable=False),  # as configured, e.g. sonnet
    Column("prompt_version", Integer, nullable=False),
    Column("status", String, nullable=False),  # OK | FAILED
    Column("model_resolved", String),  # the model that answered, e.g. claude-sonnet-5-5
    Column("chunks", Integer, nullable=False),
    Column("opinions", Integer, nullable=False),
    Column("input_tokens", Integer, nullable=False),
    Column("output_tokens", Integer, nullable=False),
    Column("error", Text),
    Column("extracted_at", DateTime, nullable=False),
    PrimaryKeyConstraint("video_id", "backend", "model", "prompt_version", name="pk_sentiment_extractions"),
    CheckConstraint("status IN ('OK', 'FAILED')", name="ck_sentiment_extractions_status"),
)


VERIFICATIONS = ("VERIFIED", "UNVERIFIED", "UNRESOLVED")

# Section 6: one merged opinion per video, stock and horizon (section 4.5) for each extractor.
sentiment_opinions = Table(
    "sentiment_opinions",
    metadata,
    Column("opinion_id", Integer, primary_key=True, autoincrement=True),
    Column("source", String, nullable=False),  # youtube | reddit
    Column("item_id", String, nullable=False),  # YouTube video_id or Reddit post id
    Column("channel_id", String, nullable=False),
    Column("author", String),  # Reddit only
    Column("published_at", DateTime, nullable=False),
    Column("effective_date", Date, nullable=False),  # the first NYSE session opening after published_at
    Column("ticker", String, nullable=False),  # UNKNOWN when the extractor could not tell
    Column("name_as_said", String),
    Column("asset_type", String, nullable=False),  # STOCK | ETF
    Column("stance", String, nullable=False),
    Column("horizon", String, nullable=False),
    Column("condition", String),
    Column("conviction", Integer, nullable=False),
    Column("action", String, nullable=False),
    Column("quote", Text, nullable=False),
    Column("start_seconds", Integer),
    Column("quote_score", Float),  # best match of the quote against its transcript piece, 0..1
    Column("reasoning", Text),
    Column("other_quotes_json", Text),  # earlier statements merged into this opinion (section 4.5)
    Column("verification", String, nullable=False),
    Column("backend", String, nullable=False),
    Column("model", String, nullable=False),
    Column("prompt_version", Integer, nullable=False),
    Column("extracted_at", DateTime, nullable=False),
    UniqueConstraint("source", "item_id", "ticker", "name_as_said", "horizon", "backend", "model",
                     "prompt_version", name="uq_sentiment_opinions"),
    CheckConstraint("verification IN ('VERIFIED', 'UNVERIFIED', 'UNRESOLVED')", name="ck_sentiment_opinions_verif"),
)
