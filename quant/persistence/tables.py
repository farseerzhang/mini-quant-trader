"""SQLAlchemy Core table definitions for the local data store.

The schema follows section 6 of the proposal. Only the raw market-data
foundation is defined here (sections 6.1, 6.2, 6.3, 6.12, 6.13); derived
tables (features, predictions, signals, ...) are added in later phases.
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
)

metadata = MetaData()


# 6.1 symbols — the universe, synchronized from config (config is the source of truth).
symbols = Table(
    "symbols",
    metadata,
    Column("symbol", String, primary_key=True),
    Column("name", String),
    Column("asset_type", String),
    Column("exchange", String),
    Column("currency", String),
    Column("sector", String),
    Column("industry", String),
    Column("active", Integer, nullable=False, server_default="1"),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)


# 6.2 daily_prices — unadjusted OHLCV, as originally traded. The source of truth.
daily_prices = Table(
    "daily_prices",
    metadata,
    Column("symbol", String, nullable=False),
    Column("date", Date, nullable=False),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    # volume may be NULL for instruments such as ^VIX.
    Column("volume", Integer),
    Column("provider", String, nullable=False),
    Column("ingested_at", DateTime, nullable=False),
    # PRIMARY KEY (symbol, date) — idempotent, upsert-friendly (sections 4.4 / 6.2).
    PrimaryKeyConstraint("symbol", "date", name="pk_daily_prices"),
    CheckConstraint(
        "volume IS NULL OR volume >= 0", name="ck_daily_prices_volume_non_negative"
    ),
)


# 6.3 corporate_actions — splits and dividends, stored separately from prices.
corporate_actions = Table(
    "corporate_actions",
    metadata,
    Column("symbol", String, nullable=False),
    Column("ex_date", Date, nullable=False),
    Column("action_type", String, nullable=False),  # SPLIT | DIVIDEND
    Column("value", Float, nullable=False),
    Column("provider", String, nullable=False),
    Column("ingested_at", DateTime, nullable=False),
    PrimaryKeyConstraint(
        "symbol", "ex_date", "action_type", name="pk_corporate_actions"
    ),
    CheckConstraint(
        "action_type IN ('SPLIT', 'DIVIDEND')", name="ck_corporate_actions_type"
    ),
)


# 6.13 data_quarantine — rows that fail validation, never silently dropped.
data_quarantine = Table(
    "data_quarantine",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String, nullable=False),
    Column("date", Date),
    Column("provider", String, nullable=False),
    Column("reason", String, nullable=False),
    Column("raw", Text),  # JSON snapshot of the offending row
    Column("quarantined_at", DateTime, nullable=False),
)


# 6.12 pipeline_runs — restartability and data-freshness tracking.
pipeline_runs = Table(
    "pipeline_runs",
    metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("command", String, nullable=False),
    Column("stage", String, nullable=False),
    Column("status", String, nullable=False),  # STARTED | OK | FAILED | SKIPPED
    Column("data_as_of_date", Date),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime),
    Column("message", Text),
)
