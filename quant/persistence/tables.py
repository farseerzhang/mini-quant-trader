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
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
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


# 6.7 predictions - the ACTIVE predictor's calibrated probability per stock and session.
# A model output only, never a BUY/HOLD label; input_hash fingerprints the exact inputs used.
predictions = Table(
    "predictions",
    metadata,
    Column("symbol", String, nullable=False),
    Column("date", Date, nullable=False),
    Column("model_version", String, nullable=False),  # the predictor's model_id
    Column("probability", Float, nullable=False),
    Column("target_definition", Text, nullable=False),  # JSON, as registered with the model
    Column("input_hash", String, nullable=False),  # sha256 of the input values (float64, model column order)
    Column("created_at", DateTime, nullable=False),
    PrimaryKeyConstraint("symbol", "date", "model_version", name="pk_predictions"),
)


# 6.8 positions — snapshots of the holdings the signals were computed against.
# One snapshot per as-of session and source; re-running a session replaces it.
positions = Table(
    "positions",
    metadata,
    Column("symbol", String, nullable=False),  # a stock, or CASH (quantity = cash amount)
    Column("as_of_date", Date, nullable=False),  # session T the holdings are valued at
    Column("quantity", Float, nullable=False),
    Column("cost_basis", Float),  # per share, optional
    Column("managed", Integer, nullable=False),  # 0: outside the stock pool, no signal
    Column("source", String, nullable=False),  # MANUAL | PAPER
    Column("created_at", DateTime, nullable=False),
    PrimaryKeyConstraint("symbol", "as_of_date", "source", name="pk_positions"),
    CheckConstraint("source IN ('MANUAL', 'PAPER')", name="ck_positions_source"),
    CheckConstraint("quantity >= 0", name="ck_positions_quantity_non_negative"),
)


# 6.9 signals — BUY / HOLD / REDUCE / EXIT at the close of T, with the reason (section 18.1).
# A re-run of the same session, strategy and version replaces that day's rows.
signals = Table(
    "signals",
    metadata,
    Column("symbol", String, nullable=False),
    Column("date", Date, nullable=False),  # session T the signal is computed at; fills at the open of T+1
    Column("regime", String),
    Column("strategy", String, nullable=False),
    Column("target_weight", Float, nullable=False),
    Column("current_weight", Float, nullable=False),
    Column("signal", String, nullable=False),  # BUY | HOLD | REDUCE | EXIT
    Column("score", Float),  # empty while there is no ACTIVE predictor
    Column("confidence", Float),
    Column("model_version", String, nullable=False),  # rule_based-v<strategies.version>
    Column("suggested_shares", Float, nullable=False),  # estimate at the close of T; negative = sell
    Column("close", Float),  # as-traded close of T used for the estimate
    Column("reason", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
    PrimaryKeyConstraint("symbol", "date", "strategy", "model_version", name="pk_signals"),
    CheckConstraint("signal IN ('BUY', 'HOLD', 'REDUCE', 'EXIT')", name="ck_signals_signal"),
)


# Forward test (Phase 10): what each portfolio targeted at the close of a session, frozen on
# that session's evening and never updated or deleted. forward_sessions has one row per
# (session, portfolio), also when it targets nothing; forward_targets one per held stock.
forward_sessions = Table(
    "forward_sessions",
    metadata,
    Column("session", Date, nullable=False),
    Column("portfolio", String, nullable=False),  # live | a shadow name
    Column("strategy", String, nullable=False),
    Column("regime", String),
    Column("regime_source", String, nullable=False),  # rule_based-v<N> or a REGIME model id
    Column("model_version", String, nullable=False),  # rule_based-v<N> or a PREDICTOR model id
    Column("invested", Float, nullable=False),
    Column("positions", Integer, nullable=False),
    Column("frozen_at", DateTime, nullable=False),
    PrimaryKeyConstraint("session", "portfolio", name="pk_forward_sessions"),
)

forward_targets = Table(
    "forward_targets",
    metadata,
    Column("session", Date, nullable=False),
    Column("portfolio", String, nullable=False),
    Column("symbol", String, nullable=False),
    Column("target_weight", Float, nullable=False),
    Column("close", Float),  # as-traded close of the session
    Column("score", Float),  # calibrated probability (ml_select)
    PrimaryKeyConstraint("session", "portfolio", "symbol", name="pk_forward_targets"),
)


# 6.11 backtest_runs — one row per portfolio and period of one `quant backtest` invocation (a batch).
# HOLDOUT rows exist only when the holdout was explicitly evaluated (section 15.2).
backtest_runs = Table(
    "backtest_runs",
    metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", String, nullable=False),
    Column("strategy", String, nullable=False),
    Column("kind", String, nullable=False),  # STRATEGY | BENCHMARK
    Column("strategy_version", String),
    Column("description", Text),
    Column("period", String, nullable=False),  # DEVELOPMENT | HOLDOUT
    Column("universe", Text, nullable=False),  # JSON list of the pool stocks
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=False),
    Column("initial_cash", Float, nullable=False),
    Column("fee_bps", Float, nullable=False),
    Column("slippage_bps", Float, nullable=False),
    Column("min_trade_weight", Float, nullable=False),
    Column("execution_convention", String, nullable=False),
    Column("feature_version", String, nullable=False),
    Column("model_version", String),  # regime version the strategies used
    Column("benchmark", String),
    Column("total_return", Float),
    Column("cagr", Float),
    Column("volatility", Float),
    Column("sharpe", Float),
    Column("sortino", Float),
    Column("max_drawdown", Float),
    Column("avg_invested", Float),
    Column("turnover", Float),
    Column("num_trades", Integer),
    Column("win_rate", Float),
    Column("profit_factor", Float),
    Column("fees_paid", Float),
    Column("benchmark_total_return", Float),
    Column("benchmark_sharpe", Float),
    Column("random_runs", Integer),
    Column("random_sharpe_median", Float),
    Column("sharpe_percentile", Float),
    Column("random_cagr_median", Float),
    Column("cagr_percentile", Float),
    Column("parameters", Text),  # JSON: strategies and risk settings
    Column("config_hash", String, nullable=False),
    Column("git_commit", String),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("batch_id", "strategy", "period", name="uq_backtest_runs"),
    CheckConstraint("kind IN ('STRATEGY', 'BENCHMARK')", name="ck_backtest_runs_kind"),
    CheckConstraint("period IN ('DEVELOPMENT', 'HOLDOUT')", name="ck_backtest_runs_period"),
)


# Daily equity curve of each backtest run (for the dashboard).
backtest_equity = Table(
    "backtest_equity",
    metadata,
    Column("run_id", Integer, ForeignKey("backtest_runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("date", Date, nullable=False),
    Column("value", Float, nullable=False),
    Column("invested", Float),
    PrimaryKeyConstraint("run_id", "date", name="pk_backtest_equity"),
)


# 6.10 model_versions — every trained model; at most one ACTIVE per role (section 16.1).
model_versions = Table(
    "model_versions",
    metadata,
    Column("model_id", String, primary_key=True),
    Column("role", String, nullable=False),  # PREDICTOR | REGIME
    Column("model_type", String, nullable=False),
    Column("version", Integer, nullable=False),  # 1, 2, ... per model_type
    Column("status", String, nullable=False),  # CANDIDATE | ACTIVE | RETIRED
    Column("depends_on", String),  # e.g. the regime version used as an input (rule_based-v1)
    Column("training_start", Date, nullable=False),
    Column("training_end", Date, nullable=False),
    Column("feature_version", String, nullable=False),
    Column("target_definition", Text, nullable=False),  # JSON
    Column("hyperparameters", Text, nullable=False),  # JSON
    Column("metrics", Text, nullable=False),  # JSON: walk-forward and backtest summary
    Column("gates_passed", Integer, nullable=False),  # 1 / 0
    Column("gate_results", Text, nullable=False),  # JSON: gate -> status, observed, required
    Column("trial_count", Integer, nullable=False),
    Column("holdout_evaluated", Integer, nullable=False),
    Column("validation_report_path", String),
    Column("model_path", String, nullable=False),
    Column("library_versions", Text, nullable=False),  # JSON
    Column("config_hash", String, nullable=False),
    Column("git_commit", String),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("model_type", "version", name="uq_model_versions_type_version"),
    CheckConstraint("role IN ('PREDICTOR', 'REGIME')", name="ck_model_versions_role"),
    CheckConstraint("status IN ('CANDIDATE', 'ACTIVE', 'RETIRED')", name="ck_model_versions_status"),
)

# At most one ACTIVE model per role, enforced by the database (section 16.1).
one_active_per_role = Index(
    "uq_model_versions_one_active_per_role",
    model_versions.c.role,
    unique=True,
    sqlite_where=model_versions.c.status == "ACTIVE",
)
