# Quant Trading System — Application Developer Proposal

# 1. Project Goal

Build a local, personal quantitative trading research and signal-generation application for US equities.

The application should:

- Import market data from external APIs.
- Store historical market data locally.
- Calculate technical and market features.
- Detect the current market regime: Bull, Sideways, Bear.
- Generate stock/ETF predictions using machine-learning models.
- Generate interpretable target exposures and BUY / HOLD / REDUCE signals.
- Backtest strategies with VectorBT against explicit benchmarks.
- Store predictions, signals, model versions, and backtest results.
- Run on demand from a command line, with a **manual** command to retrain models.
- Provide a Streamlit dashboard for monitoring.
- Be designed so future paper trading / broker integration can be added without redesigning the core.

This is a research and decision-support system. It must not assume that model predictions are reliable or that a signal is a recommendation to trade.

---

# 2. Technology Stack

## Core

- Python (a version confirmed compatible with VectorBT in the Phase 0 spike)
- SQLite (WAL mode)
- SQLAlchemy (Core is sufficient; ORM optional)
- pandas, NumPy
- `exchange_calendars` (or `pandas_market_calendars`) for the NYSE trading calendar

## Market Data

Initial implementation: `yfinance`.

`yfinance` is an unofficial scraper: it can break, rate-limit, and revise data. It must be isolated behind a provider interface so it can later be replaced (Polygon, Tiingo, IBKR, etc.).

## Quant / ML

- VectorBT — backtesting
- scikit-learn — Logistic Regression, Random Forest, calibration
- XGBoost — gradient-boosted trees
- hmmlearn — HMM regime detection
- joblib — model serialization for sklearn models; XGBoost native format for XGBoost models

## Operations

- A CLI (`argparse` or `typer`). **No scheduler.**

## UI

- Streamlit

## Packaging / Testing

- `pyproject.toml`
- pytest
- Alembic (optional, only if the schema starts changing often)

## Containerization

No Docker in V1. The application runs directly on the local machine.

---

# 3. High-Level Architecture

```text
                    External Market Data API
                              |
                              v
                    +--------------------+
                    | Market Data Layer  |
                    | Incremental Import |
                    | + Corporate Actions|
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    |       SQLite       |
                    | Raw (unadjusted)   |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | Feature Engineering|
                    | Scale-free features|
                    +---------+----------+
                              |
              +---------------+----------------+
              v                                v
   +--------------------+           +--------------------+
   | Market Regime      |           | ML Prediction      |
   | Rule-based / HMM   |---------->| LR / RF / XGBoost  |
   | (filtered only)    |           | (calibrated)       |
   +---------+----------+           +---------+----------+
             |                                |
             +---------------+----------------+
                             v
                    +--------------------+
                    | Strategy Engine    |
                    +---------+----------+
                              v
                    +--------------------+
                    | Risk Management    |
                    | -> Target Weights  |
                    +---------+----------+
                              v
                    +--------------------+
                    | Signals (vs        |
                    | positions)         |
                    +---------+----------+
                              v
                    +--------------------+
                    | SQLite             |
                    | Signals / Results  |
                    +---------+----------+
                              v
                    +--------------------+
                    | Streamlit Dashboard|
                    +--------------------+

   Triggered manually via CLI:
   quant run-daily | quant retrain | quant promote | quant backtest
```

---

# 4. Architecture Principles

## 4.1 Python-first

The complete V1 system uses Python. Do not introduce Java for data downloading, SQLite access, feature engineering, or orchestration. Java is not justified unless a later production trading-service requirement exists.

## 4.2 SQLite as a local file

```text
data/quant.db
```

SQLite is sufficient: single user, local, daily data, low concurrency.

Even so, two processes touch the file (CLI commands write, Streamlit reads). Therefore:

- Enable WAL mode (`PRAGMA journal_mode=WAL`).
- Set a busy timeout.
- Keep write transactions short.

Access goes through a thin data-access layer. Plain functions over SQLAlchemy Core are preferred to one repository class per table.

## 4.3 Raw data is the source of truth

Raw market data stays logically separate from derived features. Changing RSI parameters, SMA windows or model algorithms must not require downloading market data again.

## 4.4 Incremental data ingestion

The application must NOT re-download the entire history every run.

```text
Initial run:     configured_initial_date -> last completed session
Subsequent runs: latest stored date + 1  -> last completed session
```

Rules:

- "Last completed session" is determined with the exchange calendar in `America/New_York`. The current day's partial bar is never ingested before the close.
- If the app was not run for several days, the missing range is filled automatically.
- Ingestion is idempotent: `PRIMARY KEY (symbol, date)` on prices, upsert semantics.
- Corporate actions (splits, dividends) are fetched and stored. When a **new** corporate action appears for a symbol, the symbol's affected history is re-validated and adjustment factors are recomputed (see 6.3).

## 4.5 Time and execution convention

```text
Bar date T      = a completed daily bar.
Signal at T     = computed only from information available at or before the close of T.
Execution       = at the open of T+1 (or a configurable alternative such as T+1 close).
```

This convention applies identically to backtests, paper trading and forward testing. Backtest code must shift signals accordingly; VectorBT must not be allowed to fill at the same bar the signal was computed on.

---

# 5. Expected Data Volume

Daily US equity data for a personal universe is small (e.g. 1,000 symbols x 252 days x 20 years is roughly 5 million rows). SQLite handles this comfortably. Do not optimize for distributed databases. Intraday/tick data is a future concern.

---

# 6. Database Schema

All derived tables must have a natural unique key and use upserts, so re-running any stage never creates duplicates.

## 6.1 symbols

```text
symbol (PK), name, asset_type, exchange, currency,
sector, industry, active, created_at, updated_at
```

`sector` / `industry` are needed for sector concentration limits. The config file lists the universe; the `symbols` table is synchronized from it at the start of each run (config is the single source of truth). `market.symbols` maps GICS sectors to stocks; `update-market-data` writes each pool stock with `asset_type = STOCK` and its sector, each market symbol with `asset_type = BENCHMARK`, and marks symbols no longer in the config inactive (their history is kept).

## 6.2 daily_prices (unadjusted)

```sql
CREATE TABLE daily_prices (
                              symbol TEXT NOT NULL,
                              date DATE NOT NULL,
                              open REAL, high REAL, low REAL, close REAL,   -- as traded (split-unadjusted)
                              volume INTEGER,                                -- NULL allowed (e.g. ^VIX)
                              provider TEXT NOT NULL,
                              ingested_at TIMESTAMP NOT NULL,
                              PRIMARY KEY (symbol, date)
);
```

- Store prices **as originally traded**. Do not store the provider's `adjusted_close` as the source of truth, because it is rewritten retroactively whenever a dividend or split occurs.
- Never silently overwrite historical rows. A correction requires an explicit, logged data-correction path.

## 6.3 corporate_actions

```text
symbol, ex_date, action_type (SPLIT | DIVIDEND), value, provider, ingested_at
PRIMARY KEY (symbol, ex_date, action_type)
```

Adjusted prices and total-return series are **computed** from `daily_prices` + `corporate_actions` (a view or a function), so every consumer (features, labels, backtests) uses one consistent adjustment basis. Features that need adjusted series (long moving averages, momentum, labels) must read the computed series, never raw closes across split dates.

The total-return series covers the open as well as the close, because labels (section 14) and backtest fills (section 20) use the open of T+1:

```text
total_return_close[T] = total_return_close[T-1] * (close[T] + dividend[T]) / close[T-1]   (split-adjusted values)
total_return_open[T]  = open[T] * total_return_close[T] / close[T]
```

Holding overnight into an ex-date therefore earns the dividend, and buying at the ex-date open does not. Dividends are treated as reinvested at the ex-date close; the error against reinvesting at the open is the dividend yield times that day's open-to-close move (about 0.005% for a 0.5% dividend and a 1% move).

## 6.4 market_features

Market-level features keyed by date only (not duplicated per symbol):

```text
date, spy_trend, qqq_trend, vix, vix_change, market_volatility,
spy_return_1d, qqq_return_1d, log_vix, log_market_volatility
feature_version
```

## 6.5 features

Per-symbol derived features:

```text
symbol, date, feature_version, <feature columns...>
PRIMARY KEY (symbol, date, feature_version)
```

Features must be scale-free so a single model can be trained across symbols (see section 11).

## 6.6 market_regimes

```text
date, regime_method (RULE | HMM), regime, regime_score, confidence,
model_version, created_at
PRIMARY KEY (date, regime_method, model_version)
```

Historical regime values must be point-in-time (filtered), never smoothed over future data. Rule-based regimes use `model_version = rule_based-v<regime.version>` (section 12.3).

## 6.7 predictions

```text
symbol, date, model_version, probability, target_definition, created_at
PRIMARY KEY (symbol, date, model_version)
```

A prediction is a **model output only**: the calibrated probability of the defined target. It is not a BUY/HOLD/REDUCE label.

Also snapshot (or hash) the feature vector used, so the prediction can be reproduced even if upstream data is later corrected.

## 6.8 positions

```text
symbol, as_of_date, quantity, cost_basis (optional), managed, source (MANUAL | PAPER), created_at
PRIMARY KEY (symbol, as_of_date, source)
```

Required to turn target weights into BUY / HOLD / REDUCE. Initially populated by manual entry: `run-daily` reads `positions.path` (a CSV of `symbol,quantity,cost_basis` with one `CASH` row; `quant/signals/positions.py`) and stores it as the snapshot for session T, replacing any earlier snapshot of T. Holdings are valued at the as-traded close of T. A stock outside the pool is stored with `managed = 0` and gets no signal; without a file the portfolio is all cash at the notional `backtest.initial_cash` and nothing is stored.

## 6.9 signals

```text
symbol, date, regime, strategy, target_weight, current_weight,
signal (BUY | HOLD | REDUCE | EXIT), score, confidence,
model_version, reason, created_at
PRIMARY KEY (symbol, date, strategy, model_version)
```

Semantics:

```text
target_weight > current_weight  -> BUY (increase)
target_weight ~= current_weight -> HOLD
0 < target_weight < current     -> REDUCE (to target_weight)
target_weight = 0 and held      -> EXIT
```

`reason` is human-readable and lists the main contributing factors.

## 6.10 model_versions

```text
model_id (PK), role (REGIME | PREDICTOR), model_type, version,
status (CANDIDATE | ACTIVE | RETIRED), depends_on_model_id,
training_start, training_end, feature_version, target_definition,
hyperparameters, metrics, validation_report_path, gates_passed, trial_count,
holdout_evaluated, library_versions, config_hash, git_commit,
model_path, created_at
```

`role` is what the model is used for (see section 16.1). At most one model per role is `ACTIVE` at a time. `depends_on_model_id` records which model supplied an input at training time (for example, the regime model whose output a predictor used as a feature).

The serialized model lives on disk; SQLite holds metadata and the path:

```text
models/
  xgb/xgb_20261001_v01.ubj
  random_forest/rf_20261001_v01.joblib
  hmm/hmm_20261001_v01.joblib
```

Avoid raw `pickle` where a native or joblib format is available.

As implemented (`quant/models/final.py`, `registry.py`, `report.py`): the chosen configuration is refit on every purged development session (sampled like the folds) and calibrated on the walk-forward out-of-sample predictions, since a refit on all data leaves nothing unseen to calibrate on. The model and its calibrator are saved together with joblib as `models/<model_type>/<model_id>.joblib`; the report is `reports/<model_type>/<model_id>.md`, mirroring `models/<model_type>/` (gates, walk-forward backtest against `regime_switch`, per-fold probability quality, the search table, coefficients or importances, leak checks, caveats). Both folders are generated and git-ignored. `model_id` is `<model_type>-<UTC timestamp>-<random>`; `version` counts up per model type; `depends_on` holds the regime version whose indicator columns are inputs; `trial_count` is the number of search candidates of that run (earlier research trials are listed in section 15.4). Registration inserts one CANDIDATE row and never changes another row.

First registration (2026-10-11): `logistic_regression-20261011T090716Z-67f9c0`, version 1, C = 0.01, refit 2006-01-03 .. 2025-09-29 (47,712 rows), gates FAILED, holdout not evaluated.

## 6.11 backtest_runs

```text
run_id, strategy, strategy_version, universe, start_date, end_date,
initial_cash, fee_model, slippage_model, execution_convention,
model_version, walk_forward_config, benchmark,
total_return, cagr, sharpe, sortino, max_drawdown, win_rate,
num_trades, profit_factor, benchmark_total_return, benchmark_sharpe,
parameters, config_hash, git_commit, created_at
```

Runs are portfolio-level; per-symbol breakdowns may be stored separately.

As implemented, one `quant backtest` invocation is a batch (`batch_id`), with one row per portfolio (`kind` STRATEGY or BENCHMARK) and period (`period` DEVELOPMENT, plus HOLDOUT only when the holdout was explicitly evaluated), unique on `(batch_id, strategy, period)`. Rows also store volatility, average invested, turnover, fees and the random-baseline ranking (development only). Costs are stored as `fee_bps`, `slippage_bps` and `min_trade_weight`. `git_commit` ends in `+dirty` when the working tree had uncommitted changes. `backtest_equity (run_id, date, value, invested)` holds each run's daily equity curve for the dashboard; it stops at the end of development unless the holdout was evaluated.

## 6.12 pipeline_runs

```text
run_id, command, stage, status (STARTED | OK | FAILED | SKIPPED),
data_as_of_date, started_at, finished_at, message
```

Used for restartability and to show data freshness / last successful run on the dashboard.

## 6.13 data_quarantine

Rows that fail validation are written here with a reason instead of entering `daily_prices`, and never silently dropped.

---

# 7. Suggested Project Structure

```text
quant-system/
├── pyproject.toml
├── README.md
├── config/settings.yaml
├── data/quant.db
├── models/{xgb,random_forest,logistic,hmm}/
├── src/quant/
│   ├── cli.py                 # entry points (python -m quant ...)
│   ├── config/settings.py
│   ├── data/                  # provider.py, yfinance_provider.py,
│   │                          # downloader.py, calendar.py, data_quality.py,
│   │                          # adjustments.py
│   ├── persistence/           # database.py, tables.py, queries.py
│   ├── features/              # technical.py, momentum.py, volatility.py, market.py
│   ├── regime/                # rule_based.py, hmm.py
│   ├── models/                # training.py, validation.py, calibration.py,
│   │                          # registry.py
│   ├── strategies/            # trend.py, mean_reversion.py, selector.py
│   ├── risk/                  # position_sizing.py, portfolio_risk.py
│   ├── signals/               # engine.py, weights_to_signals.py
│   ├── backtest/              # vectorbt_engine.py, walk_forward.py, benchmarks.py
│   └── pipeline/              # daily.py, retrain.py
├── dashboard/app.py
└── tests/{data,features,regime,models,strategies,signals,backtest}/
```

The structure may be simplified wherever it becomes heavier than the problem requires. The provider abstraction is the one abstraction that is clearly worth keeping.

---

# 8. Market Data Provider Abstraction

```python
class MarketDataProvider(Protocol):
    def get_daily_data(self, symbol: str, start: date, end: date) -> DataFrame: ...
    def get_corporate_actions(self, symbol: str, start: date, end: date) -> DataFrame: ...
```

```text
MarketDataProvider
   +-- YFinanceProvider
   +-- (future) PolygonProvider / IBKRProvider
```

The downloader depends on the abstraction only. The provider must return **unadjusted** prices and corporate actions separately.

**Phase 0 finding — yfinance is not as traded.** With `auto_adjust=False`, Yahoo still rescales prices, volumes and dividends onto the share basis of every split known at fetch time; only the dividend adjustment (`Adj Close`) is left out. Examples: AAPL's close on 2020-08-28 (before the 4:1 split) comes back as 124.81 instead of 499.23, the 2014-06-06 close as 23.06 instead of 645.57 (7:1 and 4:1), and the 2020-08-07 dividend as 0.205 instead of 0.82. Storing this would break section 6.2 twice over: the rows would change every time a new split happens, and the local adjustment (section 6.3) would apply older splits a second time.

`YFinanceProvider` therefore reverses the rescaling before returning data. It reads the symbol's full split history, and for each bar multiplies prices by the product of all split ratios with an ex-date after the bar, divides the volume by that product, and multiplies dividends by it. Split ratios are returned unchanged. Any future provider must likewise return values as traded.

---

# 9. Incremental Downloader

```text
last_session = calendar.last_completed_session(now, tz="America/New_York")

for symbol in universe:
    latest = latest_stored_date(symbol)
    start = initial_date if latest is None else next_session(latest)
    if start > last_session: continue

    prices  = provider.get_daily_data(symbol, start, last_session)
    actions = provider.get_corporate_actions(symbol, start, last_session)

    good, bad = validate(prices)
    quarantine(bad)
    upsert(good)
    upsert(actions)
    if new_corporate_action_detected: flag symbol for adjustment recompute
```

Requirements:

- Idempotent; no duplicate rows.
- Uses the exchange calendar (holidays and early closes) instead of guessing from weekdays.
- Retry failed API requests with backoff.
- A failing symbol never aborts the whole batch.
- Log failed symbols; print a final ingestion summary.
- Detect missing or abnormal data against the calendar.

---

# 10. Data Quality Checks

Before writing to `daily_prices`:

```text
- Duplicate rows
- Missing symbol/date
- Invalid OHLC relationships (high >= max(open, close, low); low <= min(open, close, high))
- Negative volume; non-positive prices
- Missing sessions versus the exchange calendar
- Extreme price changes (cross-checked against corporate_actions so legitimate
  splits are not rejected and unexplained jumps are flagged)
```

Failures go to `data_quarantine` and the log; they never silently enter the main tables.

---

# 11. Feature Engineering

All features are computed from the adjusted series (section 6.3) using only data at or before T.

Features must be **scale-free** (comparable across symbols and over time):

### Trend

```text
price_vs_sma20, price_vs_sma50, price_vs_sma200, sma50_vs_sma200
```

### Momentum

```text
return_5d, return_1m, return_3m, return_6m, return_12m
RSI14
MACD / price, MACD signal / price (or normalized MACD histogram)
drawdown_252d (close vs its 252-session high)
```

`return_5d` matches the default 5-day target horizon (short-term reversal).

### Risk / volatility

```text
ATR14 / price
realized_volatility (e.g. 20d, annualized)
volatility_ratio_20_60 (20d vs 60d volatility: rising or falling risk)
volume_ratio (volume / rolling average volume)
```

### Relative to the market (per symbol, stored in features)

```text
symbol return minus SPY return over 1m, 3m, 6m
beta and correlation of daily returns to SPY over 60 sessions
```

The default target is excess return over SPY (section 14), so relative strength and market sensitivity are the most direct inputs.

### Market-level (stored in market_features)

```text
SPY trend, QQQ trend, VIX level and change, market realized volatility
SPY and QQQ daily log total return, log VIX, log market volatility
```

The daily log returns and log levels are the HMM observations (section 13). The trend and momentum columns change slowly (day-to-day autocorrelation about 0.99), and raw VIX is strongly right-skewed (skewness about 2.6, against 0.9 for log VIX), so neither suits a Gaussian HMM well.

Raw SMA/ATR values may be stored for charting but are not model inputs.

### Feature research (2026-10-11, development period only; `quant/models/feature_research.py`)

Rank IC = the daily cross-sectional Spearman correlation between a feature at T and the 5-day excess return (the label's), over 2006-01-03 .. 2025-10-01; the t-statistic uses every 5th session (non-overlapping labels). |t| < ~2 is indistinguishable from noise.

```text
feature (current / candidate)         mean IC    t      years with the same sign
illiquidity_20d (candidate)           +0.027    +5.3    17/20   survivorship bias, see below
log_dollar_volume_20d (candidate)     -0.026    -4.9    15/20   survivorship bias, see below
overnight_gap (candidate)             -0.008    -2.9    16/20   short-term reversal
return_5d (current)                   -0.017    -2.7    16/20   short-term reversal
sector_relative_return_5d (candidate) -0.014    -2.5    14/20   short-term reversal
return_1d (candidate)                 -0.015    -2.2    16/20   short-term reversal
max_return_20d, return_12m, momentum_12_1, idiosyncratic_volatility_60d    |t| 1.6 - 1.8
the other 19 current stock features   |t| <= 1.5 (rsi14, drawdown_252d, volatility_ratio_20_60, price_vs_sma50, corr_spy_60d: |t| <= 0.5)
all 9 market features and the 3 regime columns: no IC (identical for every stock on a day)
```

- **Market features and the regime columns cannot rank stocks:** they are the same for all 48 stocks on a session, so for the selection target they add only noise (a model can at most use them to change how other inputs are weighted).
- **Most current features are noise for this target;** pairs such as `return_1m` and `relative_return_1m_spy` have identical rank IC (subtracting SPY's return does not change a ranking within a day).
- **The one robust effect is short-term reversal:** stocks that rose over the last 1-5 sessions (or gapped up overnight) tend to lag over the next 5, consistently in 14-16 of 20 years.
- **Size and liquidity look strongest but are survivorship bias:** the pool is today's large caps, so the stocks that were small in 2006 are exactly the ones that grew most (2006 dollar volume vs 2006-2025 return: rank correlation -0.32; NFLX x322). These features must not be traded on with a pool chosen today.

**Removed from V1:**

- **Fear & Greed**: no reliable point-in-time history; revisions and look-ahead risk. It may be logged going forward only (see section 30).
- **Market breadth**: requires point-in-time index constituents, which V1 does not have.

Warm-up periods (e.g. first 200 days for SMA200) must be handled explicitly (NaN, not backfilled).

---

# 12. Market Regime Detection — Rule-Based

Regimes: `BULL`, `SIDEWAYS`, `BEAR`. One regime per NYSE session, computed after the close of T from stored features of the configured `features.version` (section 11), so it inherits their point-in-time guarantee.

## 12.1 Score

Five inputs, each scored `-1`, `0` or `+1`. The score is their sum, an integer from `-5` to `+5`.

| # | Input | Source at T | +1 | 0 | -1 |
|---|---|---|---|---|---|
| 1 | SPY trend | `market_features.spy_trend` (SPY vs its SMA of `features.sma[2]`) | > 0 | = 0 | < 0 |
| 2 | QQQ trend | `market_features.qqq_trend` | > 0 | = 0 | < 0 |
| 3 | SPY long-term trend | `features.sma50_vs_sma200` for SPY | > 0 | = 0 | < 0 |
| 4 | SPY momentum | `features.return_3m` for SPY (63 sessions, total return) | > 0 | = 0 | < 0 |
| 5 | VIX level | `market_features.vix` | < `vix_low` | between | > `vix_high` |

Every input counts the same. Weights are not fitted; with five coarse inputs, weights tuned on history would mostly fit noise.

**Missing inputs.** If any input is NULL at T (warm-up of the SMA200 or the 3-month return, missing QQQ or VIX data), no regime is produced for T. A missing input is never scored as `0`, because that would silently change the score's range. The state below carries over unchanged across such a day, and the day does not count towards the dwell time.

## 12.2 State machine (hysteresis and dwell time)

With `E = regime.enter_threshold` and `X = regime.exit_threshold` (`0 < X < E <= 5`), the thresholds are symmetric for BULL and BEAR. For each scorable session, in date order, the first matching row gives the **proposed** state:

```text
condition                          proposed state
score >= +E                        BULL
score <= -E                        BEAR
current = BULL and score <  +X     SIDEWAYS
current = BEAR and score >  -X     SIDEWAYS
otherwise                          current (no change)
```

- **Hysteresis:** entering BULL needs `score >= E`, but staying in BULL only needs `score >= X` (BEAR mirrors this). The gap between `E` and `X` stops daily flip-flopping near a boundary. BULL and BEAR can switch directly into each other when the score crosses the far threshold.
- **Minimum dwell:** a proposed change takes effect only once the current state has been held for at least `regime.min_dwell_days` scorable sessions; otherwise the current state is kept. A suppressed proposal is not queued; the next session is evaluated afresh. The first scorable session starts in `SIDEWAYS` and is exempt from the dwell rule. The dwell rule also delays entering BEAR in a fast crash by up to `min_dwell_days - 1` sessions; whether that is worth the stability is tested, not assumed.
- **Point in time:** the state at T is a forward-only recursion over scores up to T. It does depend on where the history starts, so the truncation test (section 21) cuts only the end of the history, never the start.

## 12.3 Output

Stored in `market_regimes` (section 6.6):

```text
regime_method = RULE
regime        = BULL | SIDEWAYS | BEAR
regime_score  = the integer score (-5 .. +5)
confidence    = |score| / 5
model_version = rule_based-v<regime.version>
```

Bump `regime.version` whenever the rules or any `regime` setting change, so stored regimes from different rules never mix (as with `features.version`).

## 12.4 Validation

- The configured values (`E = 3`, `X = 1`, `min_dwell_days = 5`, `vix_low = 20`, `vix_high = 30`) are starting points, not established optima.
- The rule-based regime has nothing to fit. `quant retrain --role regime` with `method: rule_based` does no search: it evaluates the configured rules on the walk-forward folds against the regime gates (section 27) and registers a `CANDIDATE` with no model file.
- Changing a threshold is a manual research trial: bump `regime.version`, record it in the trial log, and compare out-of-sample. Tuning thresholds on the full history is in-sample.
- The rule-based regime is the baseline that HMM must beat to be adopted.

---

# 13. Market Regime Detection — HMM

Observations (market-level): SPY/QQQ returns, realized volatility, VIX, momentum, price-vs-SMA200.

Critical rules:

1. **No look-ahead in state estimation.** hmmlearn's `predict()` (Viterbi) and `predict_proba()` (smoothed posteriors) use the whole sequence, including the future. For the regime at date T, use only the **filtered** state probability computed from observations up to T (a forward-only pass, or equivalently the last-step posterior from a fit on data up to T).
2. **Fit walk-forward.** Refit the HMM on an expanding or rolling window ending at the training cutoff; do not fit once on the full history and use it for the past.
3. **State mapping.** HMM state numbers have no meaning. Map states to BULL / SIDEWAYS / BEAR from each fit's observed return/volatility characteristics, and check the mapping stays stable between refits.
4. Use a long enough history (see section 27: `market.initial_history`) for the model to see more than one or two bear phases.
5. Treat results skeptically; compare against the rule-based baseline before adopting.

---

## 13.1 Phase 7 pre-registration (2026-10-11, before any HMM result)

Decisions fixed before fitting anything on the real data:

- **Observations (per session T):** `spy_return_1d`, `log_market_volatility`, `log_vix` from `market_features`, standardized with the mean and standard deviation of each fit's own training window. The slow indicators (SPY vs SMA200, 3-month momentum; day-to-day autocorrelation about 0.99) are left out, so the HMM is not a copy of the rule-based regime.
- **Model:** Gaussian HMM, 3 states, diagonal covariances, 5 random starts (seeds from `retrain.search.seed`), 200 EM iterations; the start with the highest training log-likelihood is kept.
- **Point in time:** only **filtered** state probabilities (a forward pass over sessions up to T with the parameters of the latest fit made on data before T), computed by our own code from the fitted parameters; never `predict` / `predict_proba`, which are smoothed over the whole sequence.
- **Walk-forward:** refit once a year on an expanding window, using the same fold rule as the predictors (`retrain.validation`: 8 training years, 12-month windows, purged, anchored to the development end); the holdout is not used.
- **State labels:** per fit, the states are ordered by their mean `spy_return_1d`: highest = BULL, middle = SIDEWAYS, lowest = BEAR. The session's regime is the label of the most probable filtered state, then the same minimum dwell as the rule-based regime (`regime.min_dwell_days`, section 12.2).
- **Gates (`retrain.gates.regime`):** `filtered_only_test_pass`: the regime at T is unchanged when later sessions are added; `state_mapping_stable_across_refits`: two consecutive fits give the same label to at least **80%** of the sessions they both cover; `min_avg_regime_dwell_days: 10`; `beats_rule_based_regime_after_costs`: `regime_switch` traded with the HMM regime has a higher Sharpe after costs than with the rule-based regime, over the same walk-forward window.

Phase 0 finding for this phase: `hmmlearn` 0.3.3 installs and runs on Python 3.12.10 / numpy 2.5 / scipy 1.18 (5 fits of 5,000 sessions in 0.4 s; 99.9% recovery of known synthetic states; reproducible with fixed seeds), but its own code triggers a numpy 2.5 deprecation warning (`a_sum.shape = shape`). It is therefore used only for fitting; the forward filter uses only the fitted parameters.

### First fit (2026-10-11; `quant/regime/hmm.py`)

The settings above are `regime.hmm` in the configuration. One fit on the first fold's training window (2006-01-03 .. 2014-09-17, 2,192 sessions; all 5 starts reached the same log-likelihood, -2.3965 per session) gives three clearly different states:

```text
            daily return (annualized)   daily sd   market vol   VIX    stay probability   expected length
BULL        +0.095% (+27%)              0.62%      10%          14.1   0.988              81 sessions
SIDEWAYS    -0.003% (-1%)               1.14%      17%          21.8   0.978              46 sessions
BEAR        -0.104% (-23%)              2.78%      38%          38.8   0.988              81 sessions
```

Within that (in-sample) window the filtered regime is BEAR for 89% of Sep-Dec 2008, 70% of the 2010 flash-crash period and 85% of the 2011 debt-ceiling sell-off, and BULL for 94% of 2013. Whether it helps out of sample is step 3 onwards.

### Walk-forward HMM regime (2026-10-11; `quant/regime/hmm_walk_forward.py`)

11 yearly fits on the predictors' folds (validation 2014-09-26 .. 2025-10-01, holdout untouched). Each fit's states stay recognizable (BULL +21% to +27% a year, BEAR -20% to -26%, BEAR VIX 34-39), and consecutive fits label their shared history the same way on **95.1%-99.9%** of sessions (pre-registered minimum 80%). In later fits only 1-2 of the 5 starts reach the best log-likelihood, so the best-of-5 rule matters.

Out of sample, against the rule-based regime on the same 2,770 sessions (a session's SPY return is attributed to the regime known at the previous close):

```text
              BULL   SIDEWAYS   BEAR   avg length   changes   SPY a year in BULL / SIDEWAYS / BEAR
HMM           50%    40%        10%    32           86        +9% / +13% / +43%
rule-based    78%     8%        14%    49           55        +11% / -16% / +50%
```

First BEAR session in each sell-off (SPY's fall from its peak by then):

```text
2018 Q4        HMM 2018-12-24 (-19%)   rule 2018-12-18 (-13%)
2020 COVID     HMM 2020-02-28 (-12%)   rule 2020-03-09 (-19%)
2022           HMM 2022-02-25 (-8%)    rule 2022-01-27 (-10%)
2025 tariffs   HMM 2025-04-03 (-12%)   rule 2025-04-08 (-19%)
```

The HMM reacts faster to sudden volatility shocks (2020, 2025) and slower to grinding declines (2018, 2022). Its BEAR is a high-volatility state, and volatility stays high through the rebounds, so SPY still rose 43% a year during its BEAR sessions: the late-exit problem of the rule-based regime remains. It also calls 40% of sessions SIDEWAYS, when SPY rose 13% a year.

### Trading comparison (2026-10-11; `quant/regime/comparison.py`)

The same strategies, settings and costs, once with each regime, from fresh cash over the walk-forward window 2014-09-26 .. 2025-10-01:

```text
                              CAGR    Sharpe   Sortino   max DD    vol     invested   turnover/yr
regime_switch [rule]          10.3%   0.8644   1.18      -21.3%    12.3%   75%        3.9x
regime_switch [hmm]            7.5%   0.8630   1.22      -13.5%     8.9%   55%        6.5x
regime_switch_bear50 [rule]   12.2%   0.9616   1.32      -25.0%    12.9%   82%        4.8x
regime_switch_bear50 [hmm]     9.2%   0.9604   1.36      -12.2%     9.6%   60%        7.6x
SPY                           13.7%   0.81     1.14      -33.7%    17.8%   100%       -
equal_weight                  18.3%   1.07     1.52      -31.7%    17.2%   100%       0.3x
```

With the HMM regime the strategies earn the same return per unit of risk (Sharpe lower by 0.0015 and 0.0012) at a lower level of risk: about 20 percentage points less invested, roughly a third less volatility and a much shallower drawdown (-13.5% against -21.3%), and therefore about 3 points less return a year. Year by year the HMM version has the higher Sharpe in 5 of 11 years (bear50: 7 of 11); it does best where the rule-based regime struggled (2021-22: Sharpe 1.18 vs 0.09; calendar 2022: +1% vs -10%) and worst in the 2020-21 recovery (Sharpe -0.62 vs 1.60), which it spent largely out of BULL. By the pre-registered criterion (a higher Sharpe after costs) the HMM regime does not beat the rule-based one; it is a lower-risk alternative of equal risk-adjusted quality, not an improvement.

### Regime gates (2026-10-11; `quant/regime/gates.py`)

Keys of `retrain.gates.regime` are validated when the settings load and evaluated in config order. `filtered_only_test_pass` re-runs the whole walk-forward on the real observations cut at two sessions (refitting only the folds that start before each cut) and requires identical filtered probabilities, raw regimes and dwelled regimes up to each cut.

```text
filtered_only_test_pass               PASS   2/2 cuts identical (2018-04-02: 884 sessions; 2025-04-03: 2,646 sessions)
state_mapping_stable_across_refits    PASS   lowest agreement 95.1% over 10 consecutive pairs (need >= 80%)
min_avg_regime_dwell_days             PASS   31.8 sessions (need >= 10)
beats_rule_based_regime_after_costs   FAIL   regime_switch Sharpe 0.8630 vs 0.8644 with the rule-based regime
verdict                               FAILED
```

The HMM regime is sound (point in time, stable, persistent) but does not beat the rule-based regime by the pre-registered criterion, so the rule-based regime remains the one in use.

### Registration (2026-10-11)

`quant retrain --role regime` (`quant/regime/retrain.py`, `report.py`) runs the steps above, fits the final HMM on all purged development sessions (2006-01-03 .. 2025-10-01), saves it as `models/hmm/<model_id>.joblib`, writes `reports/hmm/<model_id>.md` and registers a REGIME CANDIDATE in `model_versions` (`trial_count` 1: one pre-registered configuration). First registration: `hmm-20261011T150715Z-ba9d41`, gates FAILED. Since 2026-10-11 the report also embeds a regime timeline (`quant/regime/chart.py`; `<model_id>_regimes.svg` plus an interactive `.html`): SPY (log scale) above one ribbon per regime from the first session with both regimes (2005-10-17) to the end of the walk-forward window. Before the first out-of-sample session the HMM ribbon shows the final fit's filtered regime, which has seen those years; it is drawn faded and marked in sample. Colors: categorical slots 1-3 (bull blue, bear orange, sideways aqua), validated for both modes. Phase 7 outcome: the HMM regime is a sound, lower-risk alternative of equal risk-adjusted quality, but it does not beat the rule-based regime by the pre-registered criterion, so the rule-based regime remains in use.

# 14. ML Models

## Candidates

- Logistic Regression (baseline)
- Random Forest (benchmark)
- XGBoost (candidate)

RF and XGBoost are competitors; the final system does not need both unless an ensemble shows a robust benefit.

## Model scope

A **single pooled model** across all symbols (with scale-free features and optionally a symbol-agnostic design). Per-symbol models are not used in V1: with a small universe they overfit.

## Target

### Default: volatility-scaled excess return vs the benchmark

```text
entry          = open of T+1                 (execution convention, section 4.5)
exit           = open of T+1+h               (h = horizon in trading days, default 5)
stock_return   = adjusted return from entry to exit
bench_return   = the same window for the benchmark (default SPY)
excess         = stock_return - bench_return
sigma_T        = trailing daily volatility of the stock, using data <= T only
threshold      = max( k * sigma_T * sqrt(h), min_threshold )
target         = 1 if excess > threshold else 0
```

Reasons:

- It removes the market's overall move. The predictor answers "is this stock stronger than the market?" (stock selection), while the `REGIME` role handles market direction. The two roles do not duplicate each other.
- Scaling the threshold by volatility makes the target comparable across stocks (a move that is large for a low-volatility stock is not for a high-volatility one) and keeps the share of positive labels more stable over time.
- `min_threshold` is a floor that must be above the round-trip trading cost (fees plus slippage), so a "hit" is worth something after costs.

Limitations to keep in mind:

- In a falling market a stock can beat SPY and still lose money. A high probability is therefore never a reason to buy by itself; the regime and risk layers decide exposure.
- A tiny threshold turns the label into noise. Treat `k` and `min_threshold` as parameters to test, not as established values.

### Implementation rules

- Labels start from the real entry price (open of T+1), not the close of T, so they match what can actually be traded.
- Returns use the adjusted series (section 6.3), including dividends.
- `sigma_T` and every other input to the label threshold use data available at or before T.
- A label for row T uses prices up to `T+1+h`, so the label window spans `h+1` trading days. The embargo must be at least `h+1` days (section 15).
- The horizon `h` is a parameter. Longer horizons (10 or 20 days) mean lower turnover; test them rather than assuming 5.

### Measured label statistics (2026-10-10, development period only)

Implemented in `quant/models/labels.py`. Over 249,888 stock-days (48 stocks, 2005-10 to 2025-10) with `k = 0.25` and a 0.5% floor:

```text
positive share     36.7% overall; 35-39% in every year; BULL 36.9%, SIDEWAYS 36.0%, BEAR 35.6% (regime at T)
threshold          median 0.77% (10%-90%: 0.50%-1.57%); at the 0.5% floor on 16.6% of stock-days
5-day excess       median +0.02%, standard deviation 3.52%
by stock           DIS 32.6% (lowest) to NVDA 41.9% (highest)
```

The volatility scaling does what section 14 intends: the positive share is stable across years and regimes, so a model cannot score well just by learning when the market is calm. A model that always predicted 36.7% is the reference its log-loss and Brier score must beat.

### Alternatives to compare

```text
fixed_return        future return over h days > fixed threshold (e.g. 2%)   # reference only
cross_sectional_rank  each day, top ~30% = 1, bottom ~30% = 0, middle dropped
```

The rank target gives cleaner labels but needs a universe large enough for a daily ranking to mean something (dozens of symbols). It is not useful with a handful of stocks.

Targets are research parameters. The target definition is stored with every model and prediction, and every variant tried is counted in the trial log. **Do not compare AUC or log-loss across different targets**, because they are different questions. Compare targets by out-of-sample, after-cost backtest results against the baselines, using the same walk-forward folds.

## Probability calibration

Raw tree-model probabilities are not calibrated. Fit a calibration step (Platt/sigmoid, the configured choice; isotonic is available but overfits here, see below) on data the model was not fit on, and track Brier score and log-loss in addition to AUC. A probability of 0.73 means 73% **under the defined target and model assumptions**, not a 73% chance of making money.

### Measured (2026-10-10, development folds only, before any hyperparameter search)

Per fold the calibrator is fit on the last `model.calibration_months` (12) of the training window, which the model is not fit on (purged by `embargo_days`). Mean over the 11 folds, compared with always predicting the fold's training base rate:

```text
                      calibration   log-loss skill   Brier skill   calib. error   AUC     folds beating the reference
logistic_regression   none (raw)    +0.28%           +0.36%        0.020          0.534   9/11
                      isotonic      -0.93%           +0.06%        0.026          0.533   2/11
                      sigmoid       +0.23%           +0.29%        0.022          0.534   9/11
random_forest         none (raw)    +0.28%           +0.37%        0.019          0.534   9/11
                      isotonic      -0.71%           +0.03%        0.028          0.530   2/11
                      sigmoid       +0.19%           +0.25%        0.023          0.534   9/11
```

With a weak signal and one year of calibration data, isotonic regression overfits: its step function pushes probabilities to the extremes (on average 0.05-0.76 against raw 0.16-0.55), and log-loss punishes the confident misses, so it does worse than no calibration. Sigmoid (Platt) calibration, with two parameters, keeps nearly all of the raw scores' accuracy. These three variants count as calibration trials (section 15.4). **Decision (2026-10-10): `model.calibration: sigmoid`.**

## Regime as an input

The rule-based regime may be an input feature. HMM regime becomes an input only after it passes the filtered, walk-forward tests in section 13.

---

# 15. ML Validation Rules

## 15.1 Scheme: expanding walk-forward plus a final holdout

A single train / validate / test split is not enough: one validation year may contain only one kind of market, and a short final test period is too noisy to tell skill from luck. The default scheme is:

```text
Fold 1: train [start .. Y1]   -> validate Y1+1
Fold 2: train [start .. Y2]   -> validate Y2+1
Fold 3: train [start .. Y3]   -> validate Y3+1
  ...
Fold n: train [start .. Yn]   -> validate Yn+1
Final holdout: the most recent period (never used for tuning)
```

Rules:

- **Never** use random train/test splits. Training data always precedes validation data in time.
- Validation folds must cover different market conditions (for example 2020 and 2022 style volatility, not only calm bull years). Require a minimum number of folds and a minimum initial training length (see section 27).
- **Purge and embargo:** because the target looks ahead, labels in adjacent rows overlap (a label spans `h+1` trading days, see section 14). Leave a gap of at least `h+1` trading days at **every** boundary: between training and validation, and between validation and the holdout.
- Select hyperparameters by the **average and the stability across all folds**, not by the best single fold.
- Dates are never hardcoded. Fold length, minimum training length and holdout length are configured as relative rules, so the windows move forward by themselves on the next retrain.

### 15.1.1 Implementation (`quant/models/folds.py`, `quant/models/dataset.py`)

- **Rows:** one per (session, pool stock); SPY, QQQ and ^VIX are not rows. Inputs at T: the 23 stock features, the 9 market features and the rule-based regime at T as three indicator columns (35 inputs). Rows with a missing input (warm-up) are dropped; rows whose label is not known yet are kept for scoring.
- **Purging:** `embargo_days = h + 1` sessions are dropped before every boundary (training -> validation, and development -> holdout), so the last kept session's exit open is the session before the boundary.
- **Anchoring:** the last validation window ends at the last purged development session, and windows step back `fold_length_months` at a time; a fold is kept only if at least `min_train_years` of history precede it. Nothing is hardcoded, so the windows move forward on the next retrain.
- **Sampling:** training uses every `retrain.sample_every_sessions`-th session (counted on all sessions, so every fold picks the same calendar sessions); validation scores every session.

Measured on 2026-10-10 (48 stocks, 250,800 rows from 2006-01-03): 11 folds validating 2014-09-26 .. 2025-10-01, training from 2006-01-03 on 21,072 sampled rows (fold 1) up to 45,264 (fold 11), each validating about 12,000 rows; the holdout starts 2025-10-10, six sessions after the last validation day.

## 15.2 Final holdout

- The holdout is evaluated only when explicitly requested (`--evaluate-holdout`). It is never part of the normal retrain loop.
- Every holdout evaluation is logged (timestamp, model id, result). Reports show how many times the holdout has been used.
- If the holdout result is used to change the model, features, target or parameters, that period is consumed and is no longer an out-of-sample test. Treat it as ordinary validation data from then on and rely on a later period (and forward testing) for new evidence.
- When the holdout window rolls forward on a later retrain, the previous holdout period becomes ordinary data.

## 15.3 Backtests of ML strategies

The ML backtest is itself walk-forward: at each point in time the signal comes from a model trained only on data available before that point (plus the embargo). It is never one model trained once and replayed over the whole period.

### 15.3.1 Implementation and first result (`quant/models/ml_backtest.py`, `quant/strategies/ml.py`)

The fold predictions are stitched into one out-of-sample probability panel (a guard rejects any prediction not from the fold whose validation window contains it, or from a fold trained less than `embargo_days` before that window) and traded as **`ml_select`**: the regime sets exposure every session exactly as in `regime_switch` (BEAR: the baseline bear exposure), and every `strategies.ml_select.rebalance_every_sessions` (5) sessions the `top_n` (10) stocks with the highest calibrated probability are chosen, sized with the usual caps. All portfolios start from fresh cash on the first validation session and stop at the last one, so nothing in the holdout is used.

Measured on 2026-10-10 (2014-09-26 .. 2025-10-01, best search parameters, after costs):

```text
                         CAGR    Sharpe   max DD    turnover/yr   higher Sharpe than regime_switch   random baseline (Sharpe beats)
ml_logistic_regression   12.5%   0.92     -24.5%    16.5x         6 of 11 folds (55%)                97% of 100 runs
ml_random_forest         10.0%   0.77     -26.7%    19.5x         3 of 11 folds (27%)                79% of 100 runs
regime_switch            10.3%   0.86     -21.3%     3.9x         -
SPY                      13.7%   0.81     -33.7%     -
equal_weight             18.3%   1.07     -31.7%     0.3x
```

The Logistic Regression's stock selection is real (its picks beat 97% of random picks with the same timing and sizes), but it trades about four times as much as `regime_switch` (about 2% a year in costs at 6 bp per side), is ahead in only 6 of 11 years and has a deeper drawdown. Random Forest is worse than the baseline. A lower-turnover way to trade the probabilities would be a new, pre-registered trial.

## 15.4 Trials and pass criteria

- Keep a **trial log** (how many models, targets and parameter sets were tried) in `model_versions` and the validation report. The more variants tried, the less any single good result means.
- Pass criteria ("gates") are defined in configuration **before** a run, and are not changed after seeing the results. Typical gates: leakage tests pass; the strategy beats the rule-based baseline after costs in at least a minimum fraction of folds; maximum drawdown is not worse than the baseline; calibration error is below a limit; the holdout (if evaluated) is not worse than the baseline.
- Judge models by out-of-sample, after-cost trading metrics, stability across folds, sub-periods and regimes, and complexity, never training accuracy.
- Compare against: rule-based baseline, Logistic Regression, SPY buy-and-hold, and a random / shuffled-signal baseline.

### 15.4.1 Implementation and first search (`quant/models/search.py`)

Candidates are every combination in `retrain.search.space` when there are at most `max_trials`, otherwise `max_trials` drawn at random with `seed`. Each runs the full walk-forward. The selection score is the **mean minus one standard deviation of the per-fold log-loss skill** (log-loss relative to that fold's base-rate reference, so folds of different difficulty are comparable); the first of equal scores wins. Trading results are left to the gates.

Measured on 2026-10-10 (11 development folds, sigmoid calibration; 12 trials):

```text
logistic_regression   C = 0.01 / 0.1 / 1.0          log-loss skill +0.25% / +0.23% / +0.22% (std ~0.23%)   best: C = 0.01
random_forest         max_depth 4-8 x leaf 20-100    log-loss skill +0.14% .. +0.21% (std 0.31-0.41%)       best: depth 8, leaf 20
all candidates        AUC 0.532-0.535, calibration error 0.022-0.025, better than the reference in 9 of 11 folds
```

The differences between candidates are far smaller than the year-to-year spread, so the hyperparameters barely matter here, and Random Forest is not better than Logistic Regression. The best Random Forest sits at the edge of the grid (deepest trees, smallest leaves); widening the grid after seeing this would be a new trial.

Trial log so far: 12 search trials, plus 6 earlier calibration comparisons (none / isotonic / sigmoid for each model type).

### 15.4.2 Gate evaluation (`quant/models/gates.py`)

Each key of `retrain.gates.predictor` is evaluated in config order and reported as PASS, FAIL, DISABLED (`false` in the config) or NOT EVALUATED (the holdout gate without `--evaluate-holdout`). A key without an evaluator, a misspelled key or a value of the wrong type is rejected when the settings load. A candidate passes when no gate FAILs. The leak gate re-checks the actual run: the training -> validation purge in every fold, the development -> holdout purge, the fit -> calibration purge, and that every prediction comes from the fold whose validation window contains it.

How each gate is measured (walk-forward ML backtest, after costs, against `regime_switch`): `beats_baseline_after_costs` = higher overall Sharpe; `min_fraction_of_folds_beating_baseline` = share of validation years with a higher Sharpe; `max_drawdown_not_worse_than_baseline` = drawdown no deeper; `max_calibration_error` = mean over folds of the equal-count calibration error; `holdout_not_worse_than_baseline` = holdout Sharpe no lower.

First evaluation (2026-10-11, gates as configured before any result):

```text
                                         ml_logistic_regression          ml_random_forest
leak_tests_pass                          PASS (4/4)                      PASS (4/4)
beats_baseline_after_costs               PASS (0.92 vs 0.86)             FAIL (0.77 vs 0.86)
min_fraction_of_folds_beating_baseline   FAIL (6/11 = 55% < 60%)         FAIL (3/11 = 27%)
max_drawdown_not_worse_than_baseline     FAIL (-24.5% vs -21.3%)         FAIL (-26.7% vs -21.3%)
max_calibration_error                    PASS (0.022)                    PASS (0.022)
holdout_not_worse_than_baseline          NOT EVALUATED                   NOT EVALUATED
verdict                                  FAILED                          FAILED
```

Under the research-validity gate (section 32), the rule-based `regime_switch` remains what ships.

### 15.4.3 Pre-registered trials: horizon and keep band (registered 2026-10-11, before running)

Motivation (from the first evaluation, section 15.4.2): the edge in `ml_select` is real but small, and trading it every 5 sessions costs about 2% a year. Two changes target the noise and the turnover:

- **Longer label horizon** (section 14 asks for 10 and 20 to be tested): the label spans h sessions, the embargo is h + 1, and the selection is refreshed every h sessions.
- **Keep band:** a held stock is kept while it ranks within `keep_rank`; only free places are refilled from the top `top_n` (10).

Grid, fixed now: Logistic Regression only (the better model so far), `C = 0.01` (the search showed C barely matters), horizons **5 / 10 / 20** x `keep_rank` **10 / 20**. Everything else is unchanged, including `sample_every_sessions: 5`, `top_n: 10`, sigmoid calibration, the folds rule and **the gates**. Horizon 5 / keep 10 is the existing result, so the grid adds **5 trials** (trial log: 12 search + 6 calibration + 5 = 23). Each trial is judged by the gates as configured; picking the best of six afterwards is itself a selection and is reported as such. The holdout is not used.

Result (2026-10-11, development folds, after costs; implemented in `quant/models/experiments.py`, keep band in `quant/strategies/ml.py`):

```text
trial        AUC     log-loss skill   CAGR    Sharpe (base)   max DD (base)     turnover   years > base   random baseline   verdict
h5  keep10   0.535   +0.25%           12.5%   0.92 (0.86)     -24.5% (-21.3%)   16.5x      6/11           97%               FAILED
h5  keep20   0.535   +0.25%           12.1%   0.90 (0.86)     -24.0% (-21.3%)    8.8x      6/11           88%               FAILED
h10 keep10   0.534   +0.20%           10.6%   0.82 (0.85)     -20.0% (-21.3%)   11.1x      6/11           73%               FAILED
h10 keep20   0.534   +0.20%           10.4%   0.81 (0.85)     -23.0% (-21.3%)    7.1x      5/11           57%               FAILED
h20 keep10   0.520   +0.03%            9.1%   0.73 (0.83)     -19.1% (-21.3%)    8.2x      6/11           45%               FAILED
h20 keep20   0.520   +0.03%            8.1%   0.65 (0.83)     -20.2% (-21.3%)    6.0x      6/11           26%               FAILED
```

No trial passes; every one fails `min_fraction_of_folds_beating_baseline` (at most 6 of 11 years). The keep band works mechanically (median holding 10 -> 20 sessions, changes per rebalance 3.7 -> 1.9) and halves the costs, but the stocks it keeps are lower-ranked, so the selection edge shrinks by about as much as the costs fall (random-baseline percentile 97% -> 88%). Longer horizons lose the signal itself: the skill falls from +0.25% (5 days) to +0.20% (10) and +0.03% (20), so whatever the model knows is short-term. Holding longer, by either route, removes edge as fast as it removes cost. (The baseline Sharpe differs slightly per horizon because the purge, h + 1 sessions, shifts the evaluation window.)

### 15.4.4 Phase 6: XGBoost (registered 2026-10-11, before running)

XGBoost (`xgboost` 3.4.1, `tree_method: hist`, `objective: binary:logistic`) runs through the same chain as Logistic Regression and Random Forest: the same label (5-day horizon), dataset, folds, sigmoid calibration, `ml_select` trading rule (`top_n: 10`, `keep_rank: 10`, every 5 sessions) and gates. Its search space is the one in `retrain.search.space.xgboost`, which was written into the configuration before any result existed: 432 combinations, of which `max_trials: 30` are drawn at random with `seed: 42`. This adds 30 trials (trial log: 53). The final model is stored in XGBoost's native format (`.ubj`) next to the joblib file holding its calibrator (section 6.10).

Result (2026-10-11): best of 30 trials `max_depth 2, learning_rate 0.1, n_estimators 300, min_child_weight 5, reg_lambda 5` (the shallowest trees; the deepest, fastest-learning settings scored worst, AUC down to 0.514). Registered as `xgboost-20261011T110317Z-42e83b`, gates FAILED.

**Phase 6 comparison** (the same 11 development folds, 2014-09-26 .. 2025-10-01, after costs):

```text
                      AUC     log-loss skill   CAGR    Sharpe   max DD    turnover/yr   years > regime_switch   gates
logistic_regression   0.535   +0.25%           12.5%   0.92     -24.5%    16.5x         6/11                    FAILED (2 of 5)
random_forest         0.534   +0.21%           10.0%   0.77     -26.7%    19.5x         3/11                    FAILED (3 of 5)
xgboost               0.531   +0.19%            9.3%   0.72     -30.5%    22.5x         4/11                    FAILED (3 of 5)
regime_switch (rule)  -       -                10.3%   0.86     -21.3%     3.9x         -                       baseline
```

The more flexible the model, the worse it does: each step from linear to bagged trees to boosted trees adds a little noise-fitting and turnover and loses a little skill, and within XGBoost the simplest settings win. XGBoost is not better than the linear baseline (section 14 says it is not assumed to win), Random Forest remains only a benchmark, and no predictor beats the rule-based baseline. Under section 32 the rule-based `regime_switch` is what ships.

### 15.4.5 Pre-registered trials: feature sets (registered 2026-10-11, before running)

Motivation: the feature research in section 11 found that the market and regime columns cannot rank stocks and that most stock features are noise for the 5-day target. Two trials, Logistic Regression `C = 0.01`, everything else as in the Phase 5 baseline (5-day label, sigmoid calibration, `ml_select` top 10 every 5 sessions, same folds and gates):

- **A. Stock features only:** the 23 stock features; the 9 market features and 3 regime columns are removed.
- **B. Selection inside each fold:** for each fold, the rank IC of every feature in a pool is computed on that fold's fit sessions only (never its calibration or validation sessions), and the features with |t| >= 2 are kept (at least the strongest one). Pool: the 23 stock features plus the candidates of section 11 except the survivorship-biased size/liquidity features (`log_dollar_volume_20d`, `illiquidity_20d`). The chosen inputs are recorded per fold.

This adds 2 trials (trial log: 55). The holdout is not used.

Result (2026-10-11, development folds, after costs):

```text
trial                          AUC     log-loss skill   folds > reference   CAGR    Sharpe (base)   max DD (base)     turnover   years > base   random baseline   verdict
baseline (35 inputs)           0.535   +0.25%            9/11               12.5%   0.92 (0.86)     -24.5% (-21.3%)   16.5x      6/11           97%               FAILED
A: stock features only (23)    0.537   +0.29%           10/11               11.7%   0.86 (0.86)     -24.7% (-21.3%)   16.5x      6/11           91%               FAILED
B: selected in each fold       0.497   -0.03%            2/11                9.0%   0.66 (0.86)     -27.9% (-21.3%)   29.6x      5/11           81%               FAILED
```

- **A** confirms that the market and regime columns were noise for the probabilities: without them the model is slightly more accurate and beats the reference in 10 of 11 years, the best probability quality so far. The trading result is no better (Sharpe 0.86, equal to the baseline), so it still fails three gates.
- **B** chose 1-4 short-term reversal inputs in every fold (`return_1d`, `overnight_gap`, `return_5d`, `sector_relative_return_5d`; `macd_hist_over_price` in 2018-2020), but its predictions are no better than chance in the following years (AUC below 0.5 in 9 of 11 folds; rank IC of its probabilities about 0) and a one-day signal churns the holdings (29.6x a year). A reversal effect visible over 20 years (t about 2-3) is too weak and too unstable to be learned from the preceding years and traded after costs: in the labels, the day's biggest fallers meet the target 37.3% of the time against 35.9% for the biggest risers.

Cleaner inputs improve the probabilities a little, but not enough to beat `regime_switch` after costs; the binding constraints remain the weak signal and the cost of trading it.

## 15.5 Final refit

After a model type and its hyperparameters are chosen, the deployed model is refit with those fixed parameters on **all data up to the cutoff** (including the most recent years), so it has seen recent market conditions. A refit model has no untouched out-of-sample data left. Its evidence comes from forward testing: logging live signals and scoring them against realized outcomes (Phase 10).

---

# 16. Retraining Policy — Manual Trigger

There is **no scheduler** and no automatic retraining or parameter optimization. Reasons: re-fitting on recent data automatically invites overfitting to noise, and a local personal tool does not need unattended jobs.

Operating rhythm (all manual):

```text
Whenever you want fresh signals:   quant run-daily
Occasionally, when you decide:     quant backtest
When you decide to retrain:        quant retrain
When a candidate looks good:       quant promote <model_id>
```

## 16.1 Model roles

The system has more than one kind of model, each with a different job. Each kind is a **role**, and each role has its own `ACTIVE` model.

| Role | Question it answers | Candidate models | ACTIVE at a time |
|---|---|---|---|
| `REGIME` | What state is the market in? | Rule-based, HMM | one |
| `PREDICTOR` | How likely is this stock to meet the target over the horizon? | Logistic Regression, Random Forest, XGBoost | one |

Rules:

- `quant promote` only affects the role of the candidate being promoted. Promoting a new HMM does not touch the active XGBoost model, and the other way round.
- Candidates **within** a role compete: Logistic Regression, Random Forest and XGBoost are compared for the `PREDICTOR` role, and the winner is promoted. Combining several into an ensemble is a later research decision.
- Roles are validated differently. A `PREDICTOR` is judged by out-of-sample predictive quality, calibration and after-cost trading results (section 15). A `REGIME` model is judged by: filtered-only (no look-ahead) tests, stability of the state mapping across refits, reasonable average dwell time, and whether downstream strategies do better after costs than with the rule-based regime. Each role has its own gates in configuration.
- **Dependency:** if the regime is an input feature of the predictor, a new `REGIME` model changes the predictor's inputs. When a regime model is promoted, the active predictor (whose `depends_on_model_id` points to the old regime model) is marked stale. `quant status` and `quant run-daily` show a warning until the predictor is retrained and validated on the new regime history. The old predictor keeps running meanwhile.

## 16.2 What `quant retrain` automates

1. Builds the dataset from stored data up to a cutoff (default: last completed session).
2. Generates the folds and the holdout window from configuration, applying the embargo at every boundary.
3. Trains, calibrates and evaluates the configured model type(s) on every fold, drawing hyperparameter sets from `retrain.search.space` (random search, fixed seed). The number of trials is recorded.
4. Runs the leakage tests (section 21). Any failure aborts the run.
5. Runs the walk-forward backtest with fees and slippage and compares it with the rule-based baseline, Logistic Regression, SPY buy-and-hold and the random baseline, with breakdowns by fold, year and market regime.
6. Evaluates the pre-defined gates and reports PASS or FAIL for each.
7. Only if `--evaluate-holdout` is passed: evaluates the holdout once and logs it.
8. Refits the chosen configuration on all data up to the cutoff (section 15.5).
9. Registers a new model as `CANDIDATE` with metrics, validation report, gate results, trial count, feature version, library versions, config hash and git commit.
10. Prints a summary against the currently `ACTIVE` model of the same role.
11. **Does not change any active model.**

## 16.3 What stays manual

- Defining the gates and the validation configuration (before looking at results).
- Reading the report and deciding whether a candidate is worth activating.
- `quant promote <model_id>`, which sets a candidate `ACTIVE` for its role (the previous active model of that role becomes `RETIRED`; other roles are untouched). Promotion refuses a candidate whose gates failed, unless `--override "<reason>"` is given; the override and reason are logged.
- Research decisions: changing features, the target definition or the model type.
- Deciding to evaluate the holdout.

`quant run-daily` never trains; it only loads the `ACTIVE` model of each role. If no active `REGIME` model exists, it uses the rule-based regime. If no active `PREDICTOR` exists, it uses rule-based signals only. In both cases it says so.

Retrain parameters (window type, fold length, horizon, embargo, search space, gates) come from configuration, with CLI overrides.

---

# 17. Strategy Layer

The strategy layer combines:

```text
Market regime + technical indicators + ML probability + risk constraints
```

Phase 4 implements the rule-based baselines below (`quant/strategies/`). Each is a pure function from the stored features and regime at the close of T to a **target weight** per stock; the backtest fills the change at the open of T+1 (section 4.5). A missing input (warm-up) never counts as a signal. They are baselines for testing, not trading rules.

## 17.1 Rules

| Strategy | Holds a stock at T when | Exit |
|---|---|---|
| `trend` | entry: `price_vs_sma200 > 0` and `sma50_vs_sma200 > 0` and `macd_hist_over_price > 0` (MACD above its signal line) | when `price_vs_sma200 <= 0` or `sma50_vs_sma200 <= 0`; the MACD is an entry trigger only |
| `mean_reversion` | entry: `price_vs_sma200 > 0` and `rsi14 < rsi_entry` and the MACD histogram above its value at T-1 | `rsi14 > rsi_exit`, or after `max_holding_days` sessions; no re-entry on the exit session |
| `regime_switch` | BULL: `trend`; SIDEWAYS: `mean_reversion`, with new entries only on SIDEWAYS sessions; BEAR: `trend` scaled by `bear_exposure`; no regime: nothing | follows the rules above |

`strategies.regime_switch.bear_exposures` lists the BEAR exposure of the baseline first (`0`: all cash) and then the pre-registered variants (`0.5`: trend at half size, reported as `regime_switch_bear50`). Variants are fixed in configuration before any result is seen; adding one later is a new trial (section 15.4).

## 17.2 Sizing (section 19)

1. Each held stock gets `1 / (number held)`, capped at `risk.max_position_size`.
2. A sector whose total exceeds `risk.max_sector_weight` is scaled down proportionally to the cap. The excess stays in cash; it is not moved to other sectors.
3. The result is multiplied by the regime exposure (regime_switch only).
4. If the invested total is above `risk.max_portfolio_exposure`, every weight is scaled down proportionally (section 19.1; the default 1.0 never binds). The backtest and the signal engine share this code, so they stay identical.

Whatever does not add up to 1 is cash. Sectors come from `market.symbols`, which maps GICS sectors to stocks (today's classification, not point-in-time).

## 17.3 Signal frequency (measured 2026-10-10, 48 stocks, 2005-2026)

Measured before any backtest result, so the rule changes below are design, not tuning on performance:

```text
                 first version                         current rules
trend            ~14 stocks, 80% invested, 30x/yr      ~28 stocks, 93% invested, 6.6x/yr
                 (MACD > signal required every session: it flips ~20 times per stock
                 per year, median holding 8 sessions -> now an entry trigger only,
                 median holding 17 sessions, mean 72)
mean_reversion   rsi_entry 30: ~0.2 stocks, 2% invested  rsi_entry 40: ~1.7 stocks, 16% invested
                 (RSI < 30 above SMA200 on only 0.35% of stock-days)
regime_switch    ~12 stocks, 65% invested, 26x/yr      ~24 stocks, 74% invested, 5.9x/yr
```

About 2.7x of trend's turnover is resizing the other holdings when the number held changes; the backtest removes most of it with a minimum trade size (section 18: ignore tiny weight changes).

Once a strategy has been backtested, any change to its rules or settings is a new trial (section 15.4).

---

# 18. Signal Engine

The final signal must not simply equal the ML prediction.

```text
Regime + Technical score + ML probability + Risk limits
        |
        v
  Target weight per symbol
        |
        v
  Compare with current positions
        |
        v
  BUY / HOLD / REDUCE / EXIT  +  reason
```

Hysteresis around weight changes (ignore tiny differences) avoids churn.

Example stored reason:

```text
Bull regime. Price above SMA200. SMA50 above SMA200. MACD positive.
Calibrated model probability 0.64. Volatility within risk limits.
Target weight 6% vs current 4% -> BUY (increase).
```

## 18.1 Phase 8 pre-registration (2026-10-11, before any signal code)

Fixed before implementation; changing any rule below after signals have been generated is a design change that is logged here, not a silent edit.

**Which strategy.** `signals.strategy: regime_switch` (baseline, all cash in BEAR). Under section 32 no predictor and no HMM regime passed their gates, so the rule-based strategy with the rule-based regime is what ships. `regime_switch_bear50` stays a backtest variant.

**Inputs at the close of T.** The regime comes from the ACTIVE REGIME model, else the rule-based regime (today: rule-based). Predictions come from the ACTIVE PREDICTOR; with none (today) the prediction stages are skipped and logged, and signals are rule-based only (`score`, `confidence` empty; `model_version = rule_based-v<strategies.version>`).

**Live = backtest.** The target weight for T is the strategy's weight at T, computed over the full stored history (trend holdings persist from their entry; mean reversion counts holding days) and then passed through the risk layer (19.1). With the default risk settings it must equal the backtest's weight at T exactly; a test enforces this. Any risk rule that would change target weights is off by default; switching it on is a new trial and needs a backtest first (section 15.4).

**Current weight.** `quantity x close(T) / (sum of quantity x close(T) over held stocks + cash)`, from the positions snapshot (section 6.8).

**Signal mapping, with hold band `h = signals.hold_band = 0.02`** (equal to `backtest.min_trade_weight`, so live trades only what the backtest would):

| Current weight | Target weight | Signal |
|---|---|---|
| 0 | > 0 | BUY (new position; always trades) |
| > 0 | 0 | EXIT (always trades) |
| > 0 | > 0, target - current >= h | BUY (increase) |
| > 0 | > 0, current - target >= h | REDUCE (to target) |
| > 0 | > 0, \|target - current\| < h | HOLD |
| 0 | 0 | no signal row |

A stock held but outside the configured universe gets a warning ("not managed") and no signal. Suggested quantity: `floor(|target - current| x portfolio value / close(T))` shares, an estimate only, because the order fills at the open of T+1 (section 4.5).

**Reason.** Built from the factors that actually decided the weight, in this order: regime (and its source), the strategy rule that holds or exits the stock (with the indicator values), each risk limit that bound, the informational risk flags, then `target x% vs current y% -> SIGNAL`.

---

# 19. Risk Management

Risk management is independent of the prediction model and can override it.

- Maximum position size
- Maximum portfolio exposure
- Volatility-based position sizing
- ATR-based stop distance
- Maximum drawdown protection
- Sector concentration limits (uses `symbols.sector`)
- Cash allocation
- Regime exposure limits

A high ML probability must never bypass risk controls.

## 19.1 Phase 8 risk layer (pre-registered 2026-10-11)

Applied in this order to the strategy's weights at T. "Binds" means it changes a weight; "flag" means it is reported in the reason only.

| Limit | Setting | Default | Effect |
|---|---|---|---|
| Maximum position size | `risk.max_position_size` | 0.10 | binds (already part of sizing, section 17.2) |
| Sector concentration | `risk.max_sector_weight` | 0.30 | binds (already part of sizing) |
| Regime exposure | `strategies.regime_switch` | BULL/SIDEWAYS 1, BEAR 0 | binds (already part of the strategy) |
| Maximum portfolio exposure | `risk.max_portfolio_exposure` | 1.0 | binds: if invested weight exceeds it, all weights scale down proportionally; 1.0 never binds |
| Cash allocation | (derived) | 1 - invested | whatever is not invested is cash |
| ATR stop distance | `risk.atr_stop_multiple` | 2.0 | flag: stop price `close(T) - k x ATR14` in the reason; never trades |
| High volatility | `risk.high_volatility` | 0.60 | flag when `realized_volatility_20d` is above it; never resizes |

Not in Phase 8:

- **Volatility-based position sizing** changes the strategy's weights, so it is a new trial with its own backtest, not a default.
- **Maximum drawdown protection** needs the live portfolio's value history, which only exists once forward testing starts (Phase 10).

---

# 20. Backtesting

Use VectorBT, portfolio-level.

Backtests must include:

```text
- Transaction fees and slippage assumptions
- Position sizing and portfolio constraints
- Execution convention from section 4.5 (signals shifted, no same-bar fills)
- Entry/exit rules
- Market regime
- Walk-forward / out-of-sample periods, plus the final holdout
- Benchmarks: SPY buy-and-hold and equal-weight universe
- A random / shuffled-signal baseline
```

Report:

```text
Total return, CAGR, Sharpe, Sortino, Max drawdown, Win rate,
Number of trades, Turnover, Profit factor, and all of the above for the benchmarks
```

Never judge a strategy by return alone. A strategy that does not beat the benchmark on a risk-adjusted, after-cost basis is not an improvement.

Known limitation to document in every report: the universe is chosen today, so results suffer from **survivorship bias**; and the history may be dominated by bull markets.

## 20.1 Implementation (`quant/backtest/`)

- **Execution:** target weights decided at the close of T are shifted one session and placed as `targetpercent` orders at the total-return open of T+1 (section 6.3). Buys fill at `open x (1 + slippage)`, sells at `open x (1 - slippage)`, the fee is charged on the filled amount, cash is shared across stocks, sells go before buys, and the portfolio is valued at the close. A buy is limited by the cash left, so a fully invested target can end slightly below its weight. Cash earns nothing.
- **Minimum trade size (section 18):** entries and exits always trade; a held position is resized only when its target moves at least `backtest.min_trade_weight` from the last *ordered* target.
- **Benchmarks:** `SPY` buy-and-hold (one buy at the first open), and `equal_weight` (every pool stock at 1/N, set at the start and rebalanced on the first session of each month), both with the same costs.
- **Random baseline:** `random_baseline_runs` portfolios per strategy (seed `random_seed`) that keep the strategy's weights, and therefore its timing, number held, exposure and turnover, but assign each stock's weight series to a randomly permuted stock (sector caps are not re-applied). Where the strategy ranks among them shows whether its stock selection added anything beyond its timing and sizing.
- **Periods:** every portfolio is simulated once over the whole history, starting at the first session with a stored regime (when every SMA200-based input exists). Metrics are reported separately for the development period and for the holdout (the last `retrain.validation.holdout_months`); the two chain to the full-history return. Development results also break down by calendar year and by regime (a session's return is attributed to the regime known at the previous close). The holdout is not shown while rules are being designed (section 15.2).
- **Metrics:** daily returns of the value at the close, zero risk-free rate, 252 sessions a year; turnover is one-way (traded amount / 2 / average value, per year). Win rate and profit factor use VectorBT exit trades and only mean something for strategies: a monthly rebalance sells what has risen, so the equal-weight benchmark's "trades" are profitable almost by construction.

---

# 21. Research / Backtest / Live Separation and Look-Ahead Tests

Distinguish:

```text
Research
Backtest
Live signal generation
```

At date T:

```text
features use data <= T
regime uses data <= T (filtered)
the model used was trained on data available <= T (minus the embargo)
signal uses information <= T
```

Mandatory tests:

1. **Truncation test.** Compute features, regime and signals on the full dataset, then on data truncated at T. Values at T must be identical. Run for every feature and for the regime.
2. **Execution test.** A signal at T never produces a fill at or before T's price in the backtest.
3. **Embargo test.** No training label window overlaps the validation period.

Production predictions are a point-in-time log. Historical performance of ML models is established by **walk-forward simulation** (retraining at each historical step), not by replaying the latest model over the past.

---

# 22. Model Versioning

Every trained model has a unique version, stored in `model_versions` (section 6.10). Every prediction records `model_version`. Models are never overwritten; a new training run creates a new version.

Record library versions, config hash and git commit so a model and its results can be reproduced.

---

# 23. Daily Pipeline (`quant run-daily`)

Run manually, whenever fresh signals are wanted. Because ingestion fills any gap, it is safe to run after days off, and safe to run twice.

```text
1.  Load/sync configured symbol universe
2.  Update market data and corporate actions (to last completed session)
3.  Run data-quality checks (quarantine bad rows)
4.  Recompute adjustments if new corporate actions appeared
5.  Calculate/update features and market features
6.  Calculate market regime (ACTIVE regime model; fallback: rule-based)
7.  Load the ACTIVE predictor (fallback: rule-based signals only)
8.  Generate predictions
9.  Run strategy selection
10. Apply risk management -> target weights
11. Compare with positions -> BUY/HOLD/REDUCE/EXIT signals
12. Persist results
13. Print summary and write log
```

Each stage writes a `pipeline_runs` row. Stages persist inside their own transaction, so a failure at a later stage (e.g. prediction) cannot corrupt data written by earlier stages, and a rerun resumes safely.

---

# 24. Operations: CLI

There is no APScheduler and no background process.

```text
python -m quant update-market-data     # steps 1-3 only
python -m quant calculate-features     # step 5 only
python -m quant calculate-regime       # step 6 only (rule-based regime)
python -m quant run-daily              # full daily pipeline (implemented so far: steps 1-6)
python -m quant retrain --role predictor|regime [--model logistic_regression|random_forest|xgboost] [--evaluate-holdout]
                                       # MANUAL; validates, then registers a CANDIDATE
                                       # (implemented: --role predictor with logistic_regression / random_forest / xgboost,
                                       #  --role regime fits the HMM; holdout count shared with backtest; the regime
                                       #  role has no holdout gate, so --evaluate-holdout is rejected there)
python -m quant promote <model_id> [--override "<reason>"]
                                       # activate a candidate for its role (refuses failed gates)
python -m quant backtest [--strategy NAME ...] [--evaluate-holdout]
                                       # rule-based strategies vs SPY, equal weight, random baseline;
                                       # holdout only with the flag, every evaluation logged
python -m quant status                 # last runs, data freshness, active model per role, stale warnings
streamlit run dashboard/app.py         # dashboard (separate process)
```

If the user later wants automation, a scheduler can be added in a later phase without changing the pipeline, because every command is idempotent and restartable (section 30).

---

# 25. Streamlit Dashboard

Read-only view of the database.

**Status:** last successful `run-daily`, data as-of date, active model version per role, warnings (stale data, quarantined rows).

**Market Overview:** current regime and score, SPY, QQQ, VIX.

**Signals table:** symbol, signal, current vs target weight, score, ML probability, regime, strategy, reason.

**Stock Detail:** price, SMA50, SMA200, RSI, MACD, volume (from adjusted series).

**Model:** version, status, training period, validation metrics, calibration plot, feature importance, comparison with baseline.

**Backtest:** equity curve vs benchmarks, drawdown, Sharpe, CAGR, trades.

The dashboard should show a clear note that outputs are research signals, not advice. (The dashboard does not trigger retraining in V1; the CLI does.)

---

# 26. Logging

Use Python's `logging`. Log data ingestion, API failures, quality warnings, feature calculation, model training/loading, predictions, signals, backtests and every CLI command. Prefer structured, readable logs.

---

# 27. Configuration

```yaml
database:
  url: sqlite:///data/quant.db
  busy_timeout_ms: 5000

market:
  provider: yfinance
  request_timeout_seconds: 20
  calendar: XNYS                   # exchange_calendars name for the NYSE trading calendar
  initial_history: 2005-01-01      # regime work needs more than one or two bear phases
  timezone: America/New_York
  execution: next_open
  # Stock pool by GICS sector (sector -> stocks), used for the sector cap (risk.max_sector_weight).
  # Large caps listed before 2005, avoiding stocks with large spin-offs (spin-offs are not
  # modelled as corporate actions). Chosen today, and sectors are today's classification,
  # so backtests carry survivorship bias (section 20).
  symbols:
    information_technology: [AAPL, MSFT, NVDA, ORCL, CSCO, INTC, ADBE, TXN, QCOM, AMD, MU, PLTR, RMBS, LITE, FIG]
    communication_services: [GOOGL, DIS, NFLX]
    consumer_discretionary: [AMZN, HD, NKE, SBUX, LOW, TJX, TSLA, LULU, SE, CCL]
    consumer_staples: [PG, KO, PEP, WMT, COST]
    health_care: [JNJ, UNH, AMGN, LLY, TMO, TEM]
    financials: [JPM, BAC, WFC, GS, BLK]
    industrials: [CAT, UNP, LMT, DE, SPCX]
    energy: [XOM, CVX, SLB]
    materials: [SHW, NEM]
    utilities: [NEE, SO, AEP]
    real_estate: [O, PSA]
  market_symbols: [SPY, QQQ, ^VIX]

features:
  version: 1
  sma: [20, 50, 200]               # must match the price_vs_sma<N> columns of the features table
  rsi_period: 14
  atr_period: 14

regime:
  version: 1                       # stays 1 during development; bump on rule/value changes once models are registered (Phase 5)
  method: rule_based               # REGIME role: type trained by `retrain` (rule_based | hmm); run-daily uses the ACTIVE model
  enter_threshold: 3               # rule score runs from -5 to +5 (section 12)
  exit_threshold: 1                # hysteresis: must satisfy 0 < exit_threshold < enter_threshold
  min_dwell_days: 5                # sessions a regime is held before it may change
  vix_low: 20                      # VIX below this scores +1
  vix_high: 30                     # VIX above this scores -1
  hmm:                             # REGIME candidate (section 13.1, pre-registered); fit only by `retrain --role regime`
    observations: [spy_return_1d, log_market_volatility, log_vix]   # market_features columns, standardized per fit
    states: 3                      # BULL / SIDEWAYS / BEAR, ordered by mean spy_return_1d
    covariance: diag               # diag | full
    starts: 5                      # random starts (seeds from retrain.search.seed); best log-likelihood kept
    iterations: 200

strategies:                        # rule-based baselines (section 17); weights at the close of T
  version: 1                       # stays 1 during development; bump on rule/value changes once models are registered (Phase 5)
  mean_reversion:
    rsi_entry: 40                  # enter when RSI14 is below this (and above SMA200, MACD histogram rising)
    rsi_exit: 50                   # exit when RSI14 is above this ...
    max_holding_days: 10           # ... or after this many sessions
  regime_switch:                   # BULL -> trend, SIDEWAYS -> mean reversion, BEAR -> trend x bear_exposure
    bear_exposures: [0.0, 0.5]     # first = baseline (all cash); others are variants fixed before any results
  ml_select:                       # regime timing as regime_switch (baseline bear exposure), stocks chosen by the PREDICTOR
    top_n: 10                      # hold the N stocks with the highest calibrated probability (BULL / SIDEWAYS)
    keep_rank: 10                  # keep a held stock while it ranks within this (= top_n: no keep band)
    rebalance_every_sessions: 5    # refresh the selection every N sessions (= the target horizon); exposure is daily

model:
  type: xgboost                    # PREDICTOR role: default type trained by `retrain`; run-daily uses the ACTIVE model
  target:
    kind: excess_vs_benchmark_vol_scaled   # | fixed_return | cross_sectional_rank
    benchmark: SPY
    horizon_days: 5
    k: 0.25                                # volatility multiple; a parameter to test
    volatility_window_days: 20
    min_threshold: 0.005                   # floor; must exceed round-trip costs
  embargo_days: 6                          # must be >= target.horizon_days + 1
  calibration: sigmoid                     # sigmoid | isotonic; isotonic overfits the weak signal (section 14)
  calibration_months: 12                   # per fold: calibrate on the last N months of training,
                                           # which the model is not fit on (purged)

retrain:                           # used only by the manual `quant retrain`
  validation:
    scheme: expanding_walk_forward # expanding_walk_forward | rolling_walk_forward
    rolling_years: 5               # only for rolling_walk_forward
    min_train_years: 8             # length of the first training window
    fold_length_months: 12
    min_folds: 5
    holdout_months: 12             # most recent N months; rolls forward with each retrain
    holdout_requires_flag: true    # evaluated only with --evaluate-holdout
    # embargo: uses model.embargo_days (must be >= model.target.horizon_days + 1)
  sample_every_sessions: 5         # training rows: every Nth session (labels of neighbouring sessions overlap);
                                   # validation still scores every session
  search:
    method: random                 # random search over `space`, not a full grid
    max_trials: 30                 # number of hyperparameter sets tried (recorded in the trial log)
    seed: 42                       # same data + config + seed -> same metrics
    space:                         # illustrative values; tune to your own experiments
      xgboost:
        max_depth: [2, 3, 4, 5]
        learning_rate: [0.01, 0.03, 0.05, 0.1]
        n_estimators: [100, 300, 600]
        min_child_weight: [5, 20, 50]
        reg_lambda: [1, 5, 20]
      random_forest:
        max_depth: [4, 6, 8]
        min_samples_leaf: [20, 50, 100]
      logistic_regression:
        C: [0.01, 0.1, 1.0]
  gates:                           # fix these BEFORE running; do not edit after seeing results
    predictor:
      leak_tests_pass: true
      beats_baseline_after_costs: true
      min_fraction_of_folds_beating_baseline: 0.6
      max_drawdown_not_worse_than_baseline: true
      max_calibration_error: 0.05
      holdout_not_worse_than_baseline: true   # checked only if the holdout was evaluated
    regime:
      filtered_only_test_pass: true           # no look-ahead in state estimates
      state_mapping_stable_across_refits: true
      min_avg_regime_dwell_days: 10
      beats_rule_based_regime_after_costs: true
  final_refit: true
  models_dir: models               # model files: <models_dir>/<model_type>/<model_id>.joblib (git-ignored)
  reports_dir: reports             # validation reports: <reports_dir>/<model_type>/<model_id>.md (git-ignored)

backtest:                          # signal at the close of T, fill at the open of T+1 (section 20)
  initial_cash: 100000
  fee_bps: 1                       # per trade side, on the filled amount
  slippage_bps: 5                  # per trade side: buys fill at open x (1 + slippage), sells at open x (1 - slippage)
  min_trade_weight: 0.02           # resize a held position only if its target moves at least this much
                                   # (entries and exits always trade; section 18)
  benchmarks: [SPY, equal_weight]  # SPY buy-and-hold; all pool stocks, equal weight, rebalanced monthly
  random_baseline_runs: 100        # the strategy's weights reassigned to randomly permuted stocks
  random_seed: 42

risk:                              # section 19.1; applied in this order to the strategy's weights at T
  max_position_size: 0.10          # binds: per-stock cap (sizing, section 17.2)
  max_sector_weight: 0.30          # binds: per-sector cap; the excess stays in cash
  max_portfolio_exposure: 1.0      # binds: scales all weights down if invested weight is above it; 1.0 never binds
  atr_stop_multiple: 2.0           # flag only: stop price close(T) - k x ATR14 in the signal reason
  high_volatility: 0.60            # flag only: realized_volatility_20d above this is noted; never resizes

positions:                         # section 6.8; your holdings, read by run-daily (data/ is git-ignored)
  path: data/positions.csv         # columns symbol,quantity,cost_basis; one CASH row holds the cash amount

signals:                           # section 18.1; BUY / HOLD / REDUCE / EXIT at the close of T
  strategy: regime_switch          # a rule-based strategy from section 17.1 (baseline or a variant)
  hold_band: 0.02                  # HOLD while |target - current| is below this; = backtest.min_trade_weight
```

`config/settings.yaml` holds exactly this block. Every section and key is required and unknown keys are rejected at startup, so the two must be changed together.

There is no `schedule` section. Secrets/API keys must not be committed; use environment variables or a local secrets mechanism.

---

# 28. Testing Requirements

## Unit tests

- Incremental date calculation (with exchange calendar and holidays)
- Upsert behavior and duplicate prevention
- Corporate action adjustment (split and dividend cases)
- Feature calculations and warm-up handling
- Regime classification and hysteresis
- Weights-to-signals mapping (BUY / HOLD / REDUCE / EXIT)
- Position sizing and risk limits
- Data validation and quarantine

## Integration tests

- Provider -> downloader -> SQLite (using a fake provider)
- Feature pipeline -> SQLite
- Prediction -> signal pipeline
- `retrain` registers a candidate and does not change the active model
- `promote` switches the active model for its role only and leaves the other role untouched
- Promoting a new regime model marks the dependent predictor stale and shows a warning
- `promote` refuses a candidate whose gates failed unless `--override` is given, and logs the override
- `run-daily` twice produces no duplicates

## Look-ahead / correctness tests

- Truncation test (section 21)
- Execution timing and fees on small deterministic datasets
- Embargo test for training/validation windows
- Fold generator: training always precedes validation, folds do not overlap, the embargo is applied at every boundary, no label window crosses into the holdout
- Holdout guard: not evaluated without `--evaluate-holdout`; every evaluation is logged
- Gate evaluation is deterministic for a given report
- Reproducibility: the same data, config and seed produce the same metrics
- HMM: the state at T is unchanged when later data is appended

---

# 29. Implementation Phases

Each stage must be validated against the previous one with walk-forward, out-of-sample backtests.

## Phase 0 — Spikes

- Confirm VectorBT installs and runs on the target Python version.
- Confirm how `yfinance` returns unadjusted prices and corporate actions.
- Confirm the exchange calendar library covers the needed history.

Findings (2026-10-10):

- **VectorBT:** 1.1.2 installs and runs on Python 3.12.10 with numba 0.68, numpy 2.5 and pandas 3.0, without warnings. `Portfolio.from_orders` with `size_type="targetpercent"`, a separate fill `price` (the open), `fees`, `slippage`, `cash_sharing=True` and `call_seq="auto"` (sells before buys) does what section 20 needs: a weight decided at the close of T and shifted one row fills at the open of T+1 at `open * (1 + slippage)` for buys and `open * (1 - slippage)` for sells, the fee is charged on the filled notional, a NaN size places no order, and the portfolio is valued at the close. Order sizes are computed from the price before slippage, and buys are limited by the cash left, so a fully invested target can end slightly below its weight. 100 portfolios of 48 symbols over 5,500 sessions run in about 9 seconds (including numba compilation), which is fast enough for the random baseline.
- **yfinance:** see section 8 (prices are not as traded; the provider reverses Yahoo's split rescaling).
- **Calendar:** `exchange_calendars` 4.13 builds XNYS only about 20 years back by default (first session 2006-10-10 as of 2026-10), which is shorter than `market.initial_history`, and raises `DateOutOfBounds` for earlier dates. Requested with an explicit `start`, it covers 2005 fully: its 5,477 sessions from 2005-01-03 to 2026-10-09 match the 5,477 bars stored per stock. `quant/data/calendar.py` therefore always passes the start of the range it needs.

## Phase 1 — Data foundation

SQLite (WAL), `symbols`, `daily_prices`, `corporate_actions`, provider interface, `YFinanceProvider`, incremental downloader, calendar, data validation, quarantine, `pipeline_runs`.

Acceptance: initial download works; second run downloads only missing sessions; no duplicates; failed symbols reported; a new split/dividend triggers an adjustment recompute; the database survives restart; no partial current-day bar is stored.

## Phase 2 — Features

SMA/RSI/MACD/ATR/momentum/volatility (scale-free), market features.

Acceptance: reproducible; truncation test passes; warm-up periods handled.

## Phase 3 — Rule-based regime

Rule-based BULL/SIDEWAYS/BEAR with hysteresis. Serves as the baseline.

## Phase 4 — Rule-based strategies + backtesting

Trend and mean-reversion strategies in VectorBT with fees, slippage, sizing, next-open execution, benchmarks and a random baseline. This produces the **baseline performance** that later stages must beat.

## Phase 5 — ML baselines

Logistic Regression and Random Forest with purged walk-forward validation, calibration and the final holdout. Random Forest first uses the rule-based regime as an input.

## Phase 6 — XGBoost

Compare Rule-Based vs Logistic vs Random Forest vs XGBoost on out-of-sample, after-cost results and stability. XGBoost is not assumed to win; RF may remain only as a benchmark.

## Phase 7 — HMM regime

Filtered, walk-forward HMM; state mapping; compare to the rule-based regime. Adopt only if it adds out-of-sample value.

## Phase 8 — Integrated signal engine + risk

Combine regime, indicators, ML probability and risk into target weights; add `positions`; derive BUY/HOLD/REDUCE/EXIT with reasons. Rules pre-registered in sections 18.1 and 19.1.

## Phase 9 — CLI operations and dashboard

`run-daily`, `retrain` (manual), `promote`, `backtest`, `status`; Streamlit monitoring.

## Phase 10 — Forward testing, then paper trading

Log live signals and score them against realized outcomes (true out-of-sample evidence). Only after that, consider paper trading:

```text
Signal Engine -> Paper Trading -> Broker API
```

Do not implement live trading in V1.

---

# 30. Future Extensions

Not part of V1 unless required:

- Optional scheduler/automation (cron, systemd, launchd, or APScheduler) on top of the idempotent CLI
- Polygon / IBKR data provider; intraday data
- Point-in-time index constituents and market breadth
- Fear & Greed or other sentiment (logged going forward only, until a reliable history exists)
- Fundamental, sector/industry, macro and credit-spread features
- Portfolio optimization, risk parity, Black-Litterman, Kelly sizing
- Paper trading, broker integration, automated execution
- PostgreSQL, Docker, cloud deployment

---

# 31. Important Engineering Constraints

Priorities:

1. Correctness over complexity.
2. Reproducibility.
3. No look-ahead bias.
4. No data leakage.
5. Idempotent ingestion and idempotent pipeline stages.
6. Clear separation of raw and derived data.
7. Model versioning.
8. Testability.
9. Configuration over hardcoded parameters.
10. Simple local deployment.

Avoid: microservices, Kafka, Kubernetes, PostgreSQL, Docker, live broker trading, schedulers, unnecessary abstractions.

The V1 objective is a reliable local research and signal-generation platform.

---

# 32. Definition of Done for V1

V1 is complete when the developer can run:

```bash
python -m quant run-daily
```

and the system can:

```text
 1. Load configured US stocks.
 2. Download missing daily data and corporate actions (completed sessions only).
 3. Persist them to SQLite; quarantine invalid rows.
 4. Validate the data.
 5. Calculate technical features.
 6. Detect market regime (point-in-time).
 7. Load the active ML model (or fall back to rule-based).
 8. Generate calibrated predictions.
 9. Generate target weights and BUY/HOLD/REDUCE/EXIT signals.
10. Persist signals with reasons.
11. Run a VectorBT backtest against benchmarks.
12. Display results in Streamlit.
13. Retrain a model on demand with `python -m quant retrain`: it runs the walk-forward
    validation and the pre-defined pass/fail gates, writes a report, and registers a
    candidate. Nothing changes until `promote`.
14. Re-run safely without creating duplicate data.
15. Reproduce historical predictions from stored model versions and feature snapshots.
```

Research-validity gate: if the ML layer does not beat the rule-based baseline net of costs across out-of-sample periods, the rule-based system is what ships.

---

# 33. Recommended First Implementation

First working vertical slice:

```text
AAPL -> yfinance -> SQLite (unadjusted + corporate actions)
     -> SMA50 / SMA200 / RSI14 / MACD
     -> simple Bull/Bear regime
     -> simple trend strategy
     -> VectorBT (next-open execution, vs SPY buy-and-hold)
     -> backtest result
```

Then expand: SPY/QQQ, multiple stocks, VIX, rule-based regime with hysteresis, Logistic/Random Forest, XGBoost, HMM, risk management.

Prefer this incremental approach to building the whole architecture at once.

---

# 34. Key Design Decision Summary

| Area | V1 Decision |
|---|---|
| Language | Python |
| Database | SQLite file, WAL mode |
| DB access | SQLAlchemy (Core is enough) |
| Market data | yfinance behind provider abstraction |
| Price storage | Unadjusted OHLCV + corporate_actions; adjustments computed locally |
| Data frequency | Daily |
| Data ingestion | Incremental, completed sessions only |
| Execution convention | Signal at close of T, fill at open of T+1 |
| Features | pandas / NumPy, scale-free |
| Regime | Rule-based first (with hysteresis), filtered HMM later |
| ML | Logistic Regression, Random Forest, XGBoost; pooled; calibrated |
| Validation | Expanding walk-forward with embargo + final holdout + trial log; automated inside `retrain` against pre-defined gates |
| Backtest | VectorBT, portfolio-level, with benchmarks |
| Output | Target weights -> BUY/HOLD/REDUCE/EXIT vs positions |
| Scheduler | **None** |
| Retraining | **Manual** (`quant retrain`), candidate then explicit `quant promote` |
| UI | Streamlit |
| Model storage | Local files (native/joblib) + SQLite metadata |
| Container | None |
| PostgreSQL | Future |
| Java | Not needed |
| Broker execution / live trading | Future / not V1 |

---

# 35. Instructions to the Coding Agent

Implement the system incrementally. Before adding a dependency or architectural component, verify it is necessary for the current phase.

Do not introduce Java, Docker, PostgreSQL, Kafka, Redis, Kubernetes, microservices, a scheduler, or broker execution unless explicitly requested.

Start with Phase 0 and Phase 1, and make the data foundation fully testable before moving to ML.

For every feature/model/strategy:

- Write tests, including the look-ahead tests in section 21.
- Document assumptions.
- Keep parameters configurable.
- Store versions where reproducibility matters.
- Never train, retrain or promote a model from `run-daily`; those are explicit manual commands.

The most important invariant:

```text
A historical signal at time T must only depend on information
that was available at or before the close of T.
```

Favor a simple, deterministic, reproducible research pipeline over architectural complexity.

---

# Appendix — Algorithm Responsibility

| Component | Purpose | Output |
|---|---|---|
| SMA / RSI / MACD / ATR | Describe technical conditions | Features |
| Rule-Based Regime | Simple, explainable baseline | Bull / Sideways / Bear |
| Logistic Regression / Random Forest | Baseline supervised predictors | Calibrated probability |
| XGBoost | Stronger boosting candidate | Calibrated probability |
| HMM (filtered) | Detect latent market states | Market regime |
| VectorBT | Test whether strategies work historically, after costs, against benchmarks | Backtest metrics |

**Important:** More algorithms do not automatically mean a better trading system. Every added model must demonstrate incremental value through out-of-sample and walk-forward testing, and must not introduce look-ahead bias or data leakage.