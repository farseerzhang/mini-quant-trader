"""SQLAlchemy Core table definitions for the local data store.

The schema follows section 6 of the proposal. It defines the raw market-data
foundation plus versioned per-symbol and date-keyed market feature tables.
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


# 6.1 symbols â€” the universe, synchronized from config (config is the source of truth).
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


# 6.2 daily_prices â€” unadjusted OHLCV, as originally traded. The source of truth.
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
    # PRIMARY KEY (symbol, date) â€” idempotent, upsert-friendly (sections 4.4 / 6.2).
    PrimaryKeyConstraint("symbol", "date", name="pk_daily_prices"),
    CheckConstraint(
        "volume IS NULL OR volume >= 0", name="ck_daily_prices_volume_non_negative"
    ),
)


# 6.3 corporate_actions â€” splits and dividends, stored separately from prices.
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


# 6.13 data_quarantine â€” rows that fail validation, never silently dropped.
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


# 6.12 pipeline_runs â€” restartability and data-freshness tracking.
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

# 6.4 market_features â€” date-keyed market context, versioned for reproducibility.
market_features = Table(
    "market_features",
    metadata,
    Column("date", Date, nullable=False),
    Column("feature_version", String, nullable=False),
    Column("spy_trend", Float),
    Column("qqq_trend", Float),
    Column("vix", Float),
    Column("vix_change", Float),
    Column("market_volatility", Float),
    # Daily log returns and log levels: observations for the HMM regime (section 13).
    Column("spy_return_1d", Float),
    Column("qqq_return_1d", Float),
    Column("log_vix", Float),
    Column("log_market_volatility", Float),
    PrimaryKeyConstraint("date", "feature_version", name="pk_market_features"),
)


# 6.5 per-symbol, scale-free technical features.
features = Table(
    "features",
    metadata,
    Column("symbol", String, nullable=False),
    Column("date", Date, nullable=False),
    Column("feature_version", String, nullable=False),
    Column("price_vs_sma20", Float),
    Column("price_vs_sma50", Float),
    Column("price_vs_sma200", Float),
    Column("sma50_vs_sma200", Float),
    Column("return_1m", Float),
    Column("return_3m", Float),
    Column("return_6m", Float),
    Column("return_12m", Float),
    Column("rsi14", Float),
    Column("macd_over_price", Float),
    Column("macd_signal_over_price", Float),
    Column("macd_hist_over_price", Float),
    Column("atr14_over_price", Float),
    Column("realized_volatility_20d", Float),
    Column("volume_ratio_20d", Float),
    Column("relative_return_1m_spy", Float),
    Column("return_5d", Float),
    Column("relative_return_3m_spy", Float),
    Column("relative_return_6m_spy", Float),
    Column("beta_spy_60d", Float),
    Column("corr_spy_60d", Float),
    Column("volatility_ratio_20_60", Float),
    Column("drawdown_252d", Float),
    PrimaryKeyConstraint(
        "symbol", "date", "feature_version", name="pk_features"
    ),
)


# 6.6 market_regimes — one point-in-time regime per session, per method and version.
market_regimes = Table(
    "market_regimes",
    metadata,
    Column("date", Date, nullable=False),
    Column("regime_method", String, nullable=False),  # RULE | HMM
    Column("regime", String, nullable=False),  # BULL | SIDEWAYS | BEAR
    Column("regime_score", Float),
    Column("confidence", Float),
    Column("model_version", String, nullable=False),
    Column("created_at", DateTime, nullable=False),
    PrimaryKeyConstraint("date", "regime_method", "model_version", name="pk_market_regimes"),
    CheckConstraint("regime_method IN ('RULE', 'HMM')", name="ck_market_regimes_method"),
    CheckConstraint("regime IN ('BULL', 'SIDEWAYS', 'BEAR')", name="ck_market_regimes_regime"),
)
