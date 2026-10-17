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

Implementation (Phase 9, `quant/models/live.py`): columns `symbol, date, model_version, probability, target_definition, input_hash, created_at`; `input_hash` is the sha256 of the input values (float64, in the model's column order). With an ACTIVE predictor, `generate-signals` scores every pool stock on every session after the model's training end up to T and replaces those rows, so a re-run never duplicates and gives the same probabilities and hashes. The whole stretch is scored because `ml_select` keeps held stocks within a rank band and rebalances every few sessions; anchoring the schedule at the first session after training keeps it identical on every run. Inputs use the regime the model was trained with (`depends_on`).

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
model_version, suggested_shares, close, reason, created_at
PRIMARY KEY (symbol, date, strategy, model_version)
```

Semantics:

```text
target_weight > current_weight  -> BUY (increase)
target_weight ~= current_weight -> HOLD
0 < target_weight < current     -> REDUCE (to target_weight)
target_weight = 0 and held      -> EXIT
```

`reason` is human-readable and lists the main contributing factors. The exact mapping (with the hold band) is in section 18.1. `suggested_shares` (negative = sell) is an estimate at the as-traded `close` of T. `quant generate-signals` (and `run-daily`) store one session's signals per strategy and version, replacing any earlier run of that session (`quant/signals/pipeline.py`); `score` and `confidence` stay empty until a predictor is ACTIVE.

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

## 12.5 Pre-registered trial: bond-market fear inputs (registered 2026-10-15, before any data or result)

**Why not the Fear & Greed index itself.** It stays excluded (section 11): no official point-in-time history (an unofficial CNN endpoint from about 2011, unpublished method, possible revisions). Of its seven components, two are already in the score (SPY vs a moving average, VIX), two need point-in-time index constituents, which V1 does not have (52-week highs vs lows, breadth; section 11), and the put/call ratio has no reliable free history. The other two come from the bond market, which the regime knows nothing about yet: **junk-bond demand** and **safe-haven demand**. They can be rebuilt from ETF total-return closes, point in time at the close of T.

**Question.** Does adding these two inputs to the rule score improve `regime_switch` after costs out of sample?

**Data.**
- Add TLT (since 2002-07), IEF (since 2002-07) and HYG (since 2007-04) to `market.market_symbols`, with the same provider, quality checks and total-return closes as SPY / QQQ.
- Before any result, check coverage and gaps. This check is descriptive and decides nothing.

**Inputs** (at the close of T; 20 sessions, as in CNN's components). Two new `market_features` columns; the existing columns do not change.

| # | Input | Definition at T | +1 | 0 | -1 |
|---|---|---|---|---|---|
| 6 | Junk-bond demand | `credit_appetite_20d` = 20-session total return of HYG minus IEF's | > 0 | = 0 | < 0 |
| 7 | Safe-haven demand | `stock_vs_bond_20d` = 20-session total return of SPY minus TLT's | > 0 | = 0 | < 0 |

- **Direction:** risk-on scores +1, the same direction as inputs 1-5. The contrarian reading ("extreme fear is a buying chance") is not tested in this round.
- **Missing inputs:** as in section 12.1, a NULL input means no regime for T.

**Candidate rule.**
- The score of the seven inputs runs from -7 to +7.
- `E = 4`, `X = 1`. These are the same fractions of the range as today's 3/5 and 1/5, rounded (4.2 -> 4, 1.4 -> 1). They are fixed now and no other values are tried.
- `min_dwell_days`, `vix_low` and `vix_high` are unchanged.
- The candidate is research only, as `rule_based-v2`. `rule_based-v1` stays ACTIVE.

**Test.**
- **Window:** the walk-forward validation window of section 13, 2014-09-26 .. 2025-10-01. The holdout (from 2025-10-10) stays untouched.
- **Nothing is fitted.** Both regimes run their recursion from the same first session with all seven inputs (about 2007-05), so the history start is the same for both.
- **Trading:** `regime_switch` (baseline bear exposure 0) on the current 81-stock pool, same settings and costs, from fresh cash, once with v1 and once with the candidate.

**Pass criteria** (all must hold; `retrain.gates.regime` where they apply):
1. `beats_rule_based_regime_after_costs`: the candidate's Sharpe after costs is higher than v1's on the same window.
2. `min_avg_regime_dwell_days`: at least 10 sessions.
3. Point in time: the regime at T does not change when later sessions are added (truncation test at two cuts, as in section 13).

`state_mapping_stable_across_refits` does not apply (nothing is fitted).

This is **one trial**. If it fails, no input is promoted, and the two inputs are not redefined or tested again on the same data. Other windows, thresholds, weights or the contrarian direction would each be a new trial with its own pre-registration.

**Descriptive only** (reported, decides nothing):
- Each input alone (six inputs, -6 .. +6, `E = 4`, `X = 1` by the same rounding).
- `regime_switch_bear50` with the candidate.
- Per regime: share of sessions, number of changes, average length, SPY's return a year.
- The first BEAR session in the 2018 Q4, 2020, 2022 and 2025 sell-offs.
- Year-by-year Sharpe (years higher, of 11) and max drawdown.
- The rank correlation of each raw input with SPY's next-21-session return (non-overlapping sessions).

**If it passes**, the live regime still does not change:
- Forward-test run 1 continues with v1, because switching the live regime would end the run.
- Next steps, each pre-registered separately: a holdout check (as section 13.2), then the candidate as a shadow (`forward_test.shadows`). Live only after forward evidence.

**Confirmed as drafted (2026-10-15):**
- (a) the seven-input score with `E = 4`, `X = 1` (not the five-input score with the two inputs as a veto on BULL);
- (b) the 20-session window;
- (c) the baseline bear exposure 0, with bear50 descriptive only;
- (d) "higher Sharpe" with no minimum margin, as for the HMM. An 11-year Sharpe difference below about 0.1 is within noise, so a narrow pass would be weak evidence and is reported as such.

### 12.5.1 As built (2026-10-15, before the run)

- **Data:** TLT, IEF and HYG are in `market.market_symbols` and loaded by `update-market-data`. TLT and IEF run from 2005-01-03, HYG from 2007-04-11; there are no missing SPY sessions inside their ranges and nothing is quarantined. All seven inputs exist from 2007-05-09, and on all 2,770 sessions of the window.
- **One implementation change to the wording above:** the two inputs are computed in the trial code from the stored total-return closes, not stored as new `market_features` columns. Adding columns would change a table that the running forward test reads, and the DB can no longer be rebuilt freely. The definitions are unchanged.
- **Code:** `quant/regime/bond_trial.py` holds the inputs, the candidate and single-input scores, the regimes from a common first session, the point-in-time check (inputs and regimes recomputed from data ending at the two `default_cuts`), the three gates and the report. Comparisons use `compare_regimes` and sell-offs use `sell_off_reactions`, as for the HMM. Tests are in `tests/test_regime_bond_trial.py`.
- **Command:** `python -m quant regime-bond-trial` is the one-time run. It writes `reports/regime/bond-trial-<time>.md` and stores no regime.

### 12.5.2 Result (2026-10-15; `reports/regime/bond-trial-20261015T092130Z.md`)

The run was a single one, made by you. The window was **2014-10-01 .. 2025-10-06** (11 validation years), not the registered 2014-09-26 .. 2025-10-01. The folds come from the fixed fold rule (`retrain.validation`) on the current data, and the development end has moved since the HMM trial, so the window shifted by three to five sessions. The rule is unchanged and v1 and v2 were judged on identical sessions. Every recursion starts 2007-05-09.

```text
gate                                  status  observed
beats_rule_based_regime_after_costs   PASS    regime_switch Sharpe 1.0152 vs 0.9841 with rule_based-v1
min_avg_regime_dwell_days             PASS    39.6 sessions (need >= 10)
point_in_time                         PASS    2/2 cuts identical (2018-04-05, 2025-04-08)
verdict                               PASSED
```

```text
regime_switch                  CAGR    Sharpe   max DD   invested   turnover   years higher
[rule_based-v1]                13.4%   0.9841   -25.0%   76%        4.1x       -
[rule_based-v2]                13.7%   1.0152   -24.8%   73%        4.3x       6/11
```

- **Weak evidence (confirmation (d)).** The margin is +0.031 Sharpe, a third of the about 0.1 that 11 years can tell apart. v2 is higher in 6 of 11 years, with large swings both ways: 2014-15 0.54 -> 1.02 and 2017-18 1.85 -> 2.33, but 2015-16 0.30 -> -0.19 and 2018-19 -0.74 -> -1.01. Max drawdown is about the same.
- **Descriptive, decides nothing:**
  - **Earlier BEAR in three of four sell-offs:** 2018 Q4 at -7% from the peak (v1 -13%), 2022 at -8% (-10%), 2025 at -9% (-19%). 2020 is the same for all (-19%).
  - **v2's extra SIDEWAYS sessions are partly false alarms.** SIDEWAYS is 13% of sessions (v1 8%), and SPY rose 9% a year in v2's SIDEWAYS against -14% in v1's. v2 also changes regime more often: 69 changes against 55.
  - **The late BEAR exit remains:** SPY rose 50% a year in v2's BEAR sessions (v1: 58%).
  - **It does not hold for bear50:** `regime_switch_bear50` is lower with v2 (1.0613 vs 1.0853).
  - **Each bond input alone scored higher than both together:** credit 1.0491, safe haven 1.0486. Choosing one now would be a selection after seeing results, so it would need a new pre-registration and would count as a weaker, second trial.
  - **The raw inputs' rank correlation with SPY's next 21 sessions is negative** (credit -0.073, t -0.84; safe haven -0.175, t -2.03; 132 non-overlapping sessions). That is the opposite of the risk-on direction scored. The inputs seem to help as an early warning in sell-offs, not as a forecast of returns.

**Outcome of the section 12.5 trial: PASSED, weak evidence.**
- Nothing changes live: `rule_based-v1` stays ACTIVE, and forward-test run 1 continues unchanged.
- The trial is not repeated with other definitions on this data.
- Possible next step, only if you decide on it, and pre-registered before the holdout is viewed: a holdout check of v2 against v1 (as section 13.2). Only if that holds, v2 as a forward-test shadow.

### 12.5.3 Holdout check of rule_based-v2 (registered 2026-10-15, before viewing)

Decided by you on 2026-10-15 (option A): the holdout check first, the shadow only after it. This will be **holdout evaluation #3** (#1: section 15.4.8, #2: section 13.2).

**What is already known about the holdout**, so the check is not blind on every side:
- `regime_switch` with `rule_based-v1`: +26.4%, Sharpe 1.39, max DD -10.7% with holdings carried in (section 13.2).
- v1's regimes there: BULL 228, SIDEWAYS 10, BEAR 7 sessions.
- v2's regimes, scores and results on the holdout have **not** been looked at.

**Compared.**
- `regime_switch` (bear exposure 0) with `rule_based-v2` vs with `rule_based-v1`, exactly as defined in section 12.5. Both recursions start 2007-05-09 and run forward through the holdout; nothing is refit or changed.
- The same 81 stocks, sizing, costs and minimum trade size as `quant backtest`.
- Simulated over the full history so that holdings carry into the window, as in section 13.2. Metrics are on the holdout sessions only.

**Window:** 2025-10-10 (the fixed holdout start) .. **2026-10-09**, the last session before forward-test run 1 started. Holdout and forward evidence therefore do not overlap.

**Decided before looking:**

| Holdout result (v2 vs v1, `regime_switch`) | Action |
|---|---|
| Sharpe at least as high **and** max drawdown no deeper | Add v2 to forward-test run 1 as shadow `shadow_rule_bonds` |
| One better, one worse | Add `shadow_rule_bonds` as well: one year cannot settle it, and the forward test is the stronger test |
| Both worse | No shadow. v2 is closed: not changed and not tested again on this data |

**Descriptive only:**
- The same comparison started from cash on 2025-10-10.
- v2's regime days in the window, and the sessions where v1 and v2 differ.
- `regime_switch_bear50` with each.

**Limits, stated now:**
- One year is a short test: one year's Sharpe has an uncertainty of about +/-1.
- v1 had only 7 BEAR sessions in the window, so the result will turn on a few sessions where the regimes differ.
- The check can stop v2, but it cannot confirm it.

**If a shadow follows:** a new shadow type that computes `rule_based-v2` each evening from the stored inputs and the TLT / IEF / HYG closes (the code of section 12.5), with tests. It is recorded as a design change of run 1 (Phase 10), frozen from the first evening after it is added, with no backfill; its missed sessions are counted.

**Confirmed as drafted (2026-10-15):**
- (a) the decision table, in particular "one better, one worse -> shadow";
- (b) the window ends 2026-10-09, before the forward test;
- (c) holdings carried in (as section 13.2), with the fresh start descriptive only.

**As built (2026-10-15, before the run):**
- **Code:** `run_holdout_check` and `write_holdout_report` in `quant/regime/bond_trial.py`.
  - The window is fixed in code: `HOLDOUT_START` 2025-10-10, `HOLDOUT_END` 2026-10-09.
  - The decision table is `holdout_decision`: only "both worse" closes v2.
  - The shadow's name was changed from `shadow_v2` to `shadow_rule_v2` on 2026-10-15, before the run, to make it clearer, and to `shadow_rule_bonds` later that day (see the rename note below). These are names only; the decision rules are unchanged.
- **Tests:** in `tests/test_regime_bond_trial.py`.
- **Command:** `python -m quant regime-rule-v2-holdout` is the one-time run. It writes `reports/regime/rule-v2-holdout-<time>.md` and stores nothing.

**Result (2026-10-15, holdout evaluation #3; `reports/regime/rule-v2-holdout-20261015T142851Z.md`).** The run was a single one, made by you. Window 2025-10-10 .. 2026-10-09, 251 sessions; after costs.

```text
regime_switch, holdings carried in   CAGR    Sharpe   max DD   invested   turnover
[rule_based-v1]                      26.3%   1.39     -10.7%   90%        4.3x
[rule_based-v2]                      33.0%   1.51     -11.4%   90%        4.1x
regime days: v1 BULL 234 / SIDEWAYS 10 / BEAR 7; v2 BULL 226 / SIDEWAYS 13 / BEAR 12
```

- **Decision: one better, one worse -> add the shadow** (the report says `shadow_rule_v2`, its name at the time; now `shadow_rule_bonds`). The Sharpe ratio is higher; the max drawdown is 0.7 points deeper.
- v1's figures reproduce section 13.2 (+26.4%, Sharpe 1.39), so the setup is consistent.
- **The result rests on one episode.** The regimes differ on 23 sessions, all in March-April 2026:
  - 2026-03-03 .. 03-19: v1 BULL, v2 SIDEWAYS (v2 turned cautious earlier);
  - 03-20 .. 03-26: v1 SIDEWAYS, v2 BEAR;
  - 04-08 .. 04-14: v1 SIDEWAYS, v2 BULL. v2 returned to BULL earlier, unlike the late exits seen on development data.
- **Descriptive:**
  - From a fresh cash start: Sharpe 1.81 vs 1.70.
  - `regime_switch_bear50`: Sharpe 1.52 vs 1.57, max DD -8.3% vs -10.7%.
- As stated before the run, one year and one episode can stop v2 but cannot confirm it. The forward test decides.

**Shadow added (2026-10-15).**
- `forward_test.shadows.shadow_rule_bonds: {strategy: regime_switch, regime_rule: rule_bonds-v1}`.
- **New shadow option `regime_rule`** (`REGIME_RULES` in the settings; only `rule_bonds-v1` so far): each evening the freeze computes the regime with `rule_bonds_regime` (`quant/regime/rule_bonds.py`). It uses the stored inputs and the TLT / IEF / HYG closes, from the trial's first session, 2007-05-09, and no regime is stored.
- **If a bond close is missing for the session**, the shadow is not frozen that evening ("its data ends ..."), so it is never frozen on stale inputs.
- Recorded as a design change of forward-test run 1 (Phase 10 pre-registration).
- **Tests:** in `tests/test_forward_freeze.py`.

**Rename (2026-10-15, before the shadow's first freeze): `rule_based-v2` -> `rule_bonds-v1`, `shadow_rule_v2` -> `shadow_rule_bonds`.**
- The ACTIVE rules are named `rule_based-v<regime.version>`. Bumping `regime.version` to 2, as section 12.4 asks for any rule change, would have produced a second, different `rule_based-v2`.
- The rules are unchanged. The trial, combination and holdout reports and the results above keep the name they were made with (`rule_based-v2`, `shadow_rule_v2`).
- The rule's code moved from `quant/regime/bond_trial.py` to `quant/regime/rule_bonds.py`, because the nightly forward test depends on it. `bond_trial.py` keeps only the one-time trial, the holdout check and their reports.
- Nothing had been frozen under the old names.


## 12.6 Spike: Treasury yields (2026-10-16; data only, no returns looked at)

Question raised: should Treasury yields (the level, or the slope of the curve) be a market input? Nothing is stored and no setting is changed; the spike scripts were kept outside the repository.

**Sources compared over 2005-01-03 .. 2026-10-15** (5,481 SPY sessions):

| Source | Series | Coverage on SPY sessions | Notes |
|---|---|---|---|
| Yahoo (yfinance, the project's provider) | `^IRX` 13-week bill, `^FVX` 5y, `^TNX` 10y, `^TYX` 30y | from 2005-01-03; 5 sessions missing, never two in a row | Quoted **in percent**: `^TNX` / FRED `DGS10` median ratio 0.998, absolute difference median 0.007, 95th percentile 0.03, max 0.10 points. (`^TNX` used to be 10x the yield; it no longer is.) |
| FRED (Federal Reserve, CSV without a key) | `DGS3MO`, `DGS2`, `DGS10` | 41 sessions missing: Columbus Day and Veterans Day, when the bond market is closed and NYSE is open | Published with a lag: the latest value on 2026-10-16 was for 2026-10-14. **Not usable at the close of the same session** without a one-day lag |

**Findings:**
- On those bond-market holidays Yahoo still has a value (36 of 41), close to the previous session's. It would be treated as a session with an unchanged yield.
- `^IRX` was negative on 7 sessions (2020-03-19 .. 03-27, as low as -0.105%); `^IRX` / `DGS3MO` ratio 0.969 (discount rate vs bond-equivalent yield). This is harmless for a slope computed in differences.
- The largest daily 10y moves: +0.47 points (2009-03-18, the Fed's QE announcement), 0.33 (2008-09-19), 0.32 (2022-11-10, CPI).
- **10y minus 3m (Yahoo)** is inverted on 15% of sessions, in 19 separate spells. By year: 2006 40%, 2007 37%, 2019 33%, 2020 3%, 2022 14%, **2023 100%, 2024 92%**, 2025 3%.

**Assessment:**
- **Usable data:** Yahoo, same provider, same-session availability, full coverage from 2005. If a trial follows, the inputs would be `^TNX` and `^IRX` in `market.market_symbols`, like TLT / IEF / HYG.
- **Inversion itself is a poor regime input.** It lasted almost all of 2023-2024, a strong rally, and its known lead on recessions (12-18 months) is far longer than the regime's horizon.
- **The candidate worth a trial is the change in the slope**, e.g. the 20-session change of 10y minus 3m. A falling slope points to an expected easing; a rising one, to tightening or a long-end sell-off. It overlaps least with `rule_bonds-v1`'s bond-price inputs. A yield level is not stationary (0.5% to 5.3% here), so it would only be used as a change, not a level.
- A trial would be a new, separately pre-registered question. It counts as a further regime trial after sections 12.5 and 13.3, both of which moved the Sharpe ratio by only about 0.03.

## 12.7 Pre-registered trial: interest-rate shock inputs (registered 2026-10-16, before any data or result)

**A change from section 12.6, made while drafting, before any result.** Section 12.6 suggested the 20-session change of the curve slope. Its direction is ambiguous for a regime: the slope steepens both when long yields rise in a recovery and when the Fed cuts short rates in a crash (2007-08, 2008, 2020, a "bull steepening"). A +1 for steepening would then score crashes as risk-on. Rates are therefore used here only as **danger signals**, each scored **0 or -1**, never +1.

**Inputs** (at the close of T, from Yahoo `^IRX` and `^TNX` in percentage points; a missing session takes the previous session's value, at most one session, covering bond-market holidays and Yahoo's 5 gaps):

| # | Input | -1 when | Otherwise | Idea |
|---|---|---|---|---|
| 6 | `short_rate_drop_20d` | the 3-month yield fell more than **0.50** points over 20 sessions | 0 | The market prices emergency easing within a month (at least two 25 bp cuts) |
| 7 | `long_rate_jump_60d` | the 10-year yield rose more than **0.75** points over 60 sessions | 0 | A rate shock (2013 taper, 2022) that reprices stocks |

The thresholds are round numbers chosen from those meanings. They are fixed now; how often they trigger has not been looked at.

**Candidate rule `rule_rate_shocks-v1`:**
- The five v1 inputs plus inputs 6 and 7; the score runs from -7 to +5.
- `E = 3`, `X = 1`, the dwell and the VIX thresholds are **unchanged**. The new inputs can only add caution, so the BULL side keeps v1's thresholds.
- Research only; `rule_based-v1` stays ACTIVE.

**Test** (as section 13.3):
- **Window:** the walk-forward validation window from the fold rule at run time; the report states its dates. The holdout is untouched.
- **History start:** every recursion starts on the first session with all seven inputs.
- **Trading:** `regime_switch` (bear exposure 0), 81 stocks, the same settings and costs, from fresh cash, once with v1 and once with the candidate.

**Pass criteria** (all must hold; the stricter set of section 13.3). The thresholds were chosen with known episodes in mind (2008, 2020, 2022), all inside the window, so a pass is partly in sample:
1. Sharpe after costs **higher** than v1;
2. max drawdown **no deeper** than v1;
3. a higher Sharpe in **at least 6 of the 11** validation years;
4. average regime length at least **10 sessions**;
5. point in time: inputs and regimes are unchanged when later sessions are added (two cuts).

This is **one trial**. If it fails, these inputs are not redefined or re-tested on this data.

**Descriptive only** (decides nothing):
- each input alone;
- `regime_switch_bear50` with the candidate;
- regime shares, number of changes and SPY's return a year per regime;
- BEAR entry and exit in the 2018 Q4, 2020, 2022 and 2025 sell-offs;
- how often each input is -1, by year.

**If it passes:** no live change. A holdout check would follow, pre-registered separately as in section 12.5.3, and only then a forward-test shadow (`shadow_rule_rate_shocks`).

**Data needed first:** `^TNX` and `^IRX` in `market.market_symbols`, loaded by `update-market-data` (no new table; the inputs are computed in code, like `rule_bonds-v1`).

**Design choice (2026-10-16, before any result): option A.** You asked whether splitting a slope change by which end moves could tell a good steepening from a bad one. The four textbook cases:

| Case | Driver | Stocks |
|---|---|---|
| Bull steepening | Short yields falling fast (emergency easing) | Clearly a danger signal |
| Bear flattening | Short yields rising fast (hikes) | Usually a danger signal |
| Bear steepening | Long yields rising (recovery, or a term-premium jump as in Aug-Oct 2023) | Unclear |
| Bull flattening | Long yields falling (growth scare) | Unclear |

Two designs were considered:
- **A:** keep the two danger inputs above. Input 6 is about bull steepening, input 7 about long-end shocks.
- **B:** classify all four cases, scoring bull steepening and bear flattening -1. That needs a rule for "which end leads" and more thresholds.

You chose **A**: fewer thresholds and fewer researcher choices. Only one design is tested; B (or A plus a bear-flattening input) would be a separate trial.

**Confirmed (2026-10-16):**
- (a) the change from section 12.6: danger-only inputs instead of the slope change (option A, chosen 2026-10-16);
- (b) the two inputs and their thresholds (0.50 points in 20 sessions, 0.75 points in 60 sessions);
- (c) v1's thresholds unchanged (E = 3, X = 1) with a -7 .. +5 score;
- (d) the strict pass criteria 1-5;
- (e) the name `rule_rate_shocks-v1` (and `shadow_rule_rate_shocks` if it ever becomes a shadow).
- **No bond inputs:** the candidate is v1 plus the two rate inputs, compared with `rule_based-v1`, not with `rule_bonds-v1`. One change at a time, against the ACTIVE baseline: `rule_bonds-v1` is only weak evidence and is still being tested as a shadow. Combining the two would be a separate trial.

### 12.7.1 As built (2026-10-16, before the run)

- **Data:** `^TNX` and `^IRX` are in `market.market_symbols` and loaded by `update-market-data` (5,477 sessions each, from 2005-01-03).
  - **Quality-check change:** the first load quarantined 9 `^IRX` days (2020-03-18 .. 03-30) as a "non-positive price"; the 3-month yield really was slightly below zero. A yield is not a price, so `quant/data/data_quality.py` now lets the yield indexes in `YIELD_SYMBOLS` (`^IRX`, `^FVX`, `^TNX`, `^TYX`) be zero or negative. All other checks still apply to them.
  - `^IRX` was then removed and reloaded whole; nothing quarantined. The quarantine rows of the first load stay as a record.
- **Inputs:** all seven inputs exist from 2005-10-17, the same first session as v1. So v1 in this trial is exactly the ACTIVE regime's history. There are no incomplete sessions over 2014-09 .. 2025-10 (2,810 sessions).
- **Code:**
  - `quant/regime/rule_rate_shocks.py`: the inputs, the score and the candidate regime.
  - `quant/regime/rate_shocks_trial.py`: the trial, the five gates, the point-in-time check (rate inputs and every regime recomputed from data ending at the two `default_cuts`), and the report.
  - Tests: `tests/test_regime_rate_shocks.py`.
- **Command:** `python -m quant regime-rate-shocks-trial` is the one-time run. It writes `reports/regime/rate-shocks-trial-<time>.md` and stores nothing.

### 12.7.2 Result (2026-10-16; `reports/regime/rate-shocks-trial-20261016T203411Z.md`)

The run was a single one, made by you. The window from the fold rule was 2014-10-03 .. 2025-10-07; every recursion starts 2005-10-17.

```text
gate                         status  observed
sharpe_higher                FAIL    regime_switch Sharpe 0.9726 vs 0.9733 with rule_based-v1
max_drawdown_no_deeper       PASS    -25.0% vs -25.0%
years_higher                 FAIL    1 of 11 validation years (10 identical)
min_avg_regime_dwell_days    PASS    49.4 sessions
point_in_time                PASS    2/2 cuts identical (2018-04-09, 2025-04-09)
verdict                      FAILED
```

**The inputs change almost nothing.**
- The candidate's regime has the same share per state as v1's (BULL 78%, SIDEWAYS 8%, BEAR 14%) and the same 55 changes.
- It enters and leaves BEAR on exactly the same days in all four sell-offs.
- 10 of the 11 validation years are identical; in the other (2023-24) the Sharpe is 2.22 vs 2.20.

**Why** (descriptive):
- `short_rate_drop_20d` fired only in 2020 (9% of sessions), when v1 was already in BEAR on its price inputs.
- `long_rate_jump_60d` fired mostly in 2022 (40%), with a little in 2016, 2021 and 2023-25. In 2022 v1 had already left BULL.
- The rate shocks arrived together with, or after, the breaks in price trend that v1 already scores, so the extra -1 rarely moved the score across a threshold.
- With `regime_switch_bear50` the candidate is marginally higher (1.0813 vs 1.0750). This is descriptive and decides nothing.

The baseline's Sharpe (0.9733) differs from section 12.5's v1 (0.9841) because the window moved by a few sessions and the recursion starts in 2005, not 2007.

**Outcome of the section 12.7 trial: FAILED.**
- Nothing changes: `rule_based-v1` stays ACTIVE, and no holdout check or shadow follows.
- These inputs are not redefined or re-tested on this data. Option B of section 12.7, or a bear-flattening input, would be a separate trial, and this result is a reason to expect little from it.
- `^TNX` and `^IRX` stay in `market.market_symbols` for reading.

**Three regime trials** (12.5, 13.3, 12.7):
- The Sharpe moved by +0.031, -0.023 and -0.0007.
- Market information beyond prices and VIX (credit, bonds, the HMM, rates) has so far changed when BEAR starts more than it has changed risk-adjusted returns.

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

## 13.3 Pre-registered trial: the rule-based regime with an HMM BEAR trigger (registered 2026-10-15, before any result)

**Where the idea comes from, and what that means for the evidence.** The idea comes from the sell-off table in section 13. The HMM entered BEAR earlier in the sudden shocks (2020: -12% vs -19%; 2025: -12% vs -19%), the rule-based regime earlier in the grinding declines (2018 Q4, 2022). All four sell-offs lie inside the test window below, so the idea was formed on part of the data that tests it. A pass on the development window is therefore **partly in sample**. That is why the bar below is stricter than for earlier regime trials, and why the forward test, not this backtest, is meant to decide.

**The holdout cannot help either.** It has been viewed twice for these two regimes (sections 15.4.8 and 13.2), and the HMM had no BEAR session in it (BULL 77, SIDEWAYS 168). The combination below therefore equals the rule-based regime on every holdout session, so a holdout check would show nothing.

**Question.** Does `regime_switch` do better when the HMM's BEAR is added to the rule-based regime as an extra BEAR trigger?

**The combined regime** (per session T, from the two regimes as each already exists at T):

```text
rule-based BEAR or HMM BEAR      -> BEAR
otherwise                        -> the rule-based regime (BULL or SIDEWAYS)
```

- **Rule-based:** `rule_based-v1`, the ACTIVE one: five inputs, E = 3, X = 1, dwell 5 (sections 12.1-12.2). It is not v2: v2 passed only weakly and is not adopted (section 12.5.2).
- **HMM:** the walk-forward regime of section 13: yearly fits on the development folds, filtered (forward only), the `regime.hmm` configuration, the same dwell. Out of sample in every validation year.
- **The HMM's SIDEWAYS is ignored.** It was a poor signal on development data (SPY rose 13% a year in it) and on the holdout (two thirds of a rally).
- **No dwell on top.** Each input regime already has its own dwell; adding a third would be a new parameter. The combined regime's own average length is checked by a gate instead.
- **Known risk, stated before the run:** BEAR now ends only when **both** regimes have left it. The late BEAR exit (SPY +43% to +58% a year during BEAR sessions) may get worse.

**Test.**
- **Window:** the walk-forward validation window as produced by the fold rule (`retrain.validation`) on the stored data at run time, holdout excluded. The dates are not written in advance; the report states them. (Section 12.5.2 shows why: the window moves with the development end.)
- **Trading:** `regime_switch` (bear exposure 0) on the 81-stock pool, the same settings and costs, from fresh cash, once with `rule_based-v1` and once with the combination.
- **Point in time:** the HMM part is checked as in section 13 (refit up to two cuts, identical filtered regimes). The rule-based part is a forward recursion. The combined regime at T must not change when later sessions are added.

**Pass criteria** (all must hold; stricter than sections 12.5 and 13 because the idea came from this data):
1. Sharpe after costs **higher** than with `rule_based-v1`.
2. Max drawdown **no deeper** than with `rule_based-v1`.
3. Higher Sharpe in **at least 6 of the 11** validation years.
4. Average regime length of the combination at least **10 sessions**.
5. Point in time, as above.

This is **one trial**. If it fails, no other combination rule is tried on this data: "both must say BEAR", a HMM-only exit, an extra dwell or v2 instead of v1 would each be a new pre-registration and a further trial.

**Descriptive only** (reported, decides nothing):
- `regime_switch_bear50` with the combination.
- The variant where both must say BEAR.
- Per regime: share of sessions, changes, average length, SPY's return a year.
- The first BEAR session **and the first session after it that is no longer BEAR** in each sell-off (2018 Q4, 2020, 2022, 2025).
- Year-by-year Sharpe and max drawdown.

**If it passes** (the backtest only says the idea is worth a forward test):
- No change to `live` or to the ACTIVE regime.
- The combination becomes a forward-test shadow, `shadow_combo`, as a logged design change of forward-test run 1 (Phase 10 pre-registration). It is frozen from the first evening after it is added, with no backfill, and its missed sessions before that are counted.
- Its 12-month comparison with `live` follows the Phase 10 rule: Sharpe higher **and** max drawdown no deeper. It runs from the date it was added, so its verdict comes later than run 1's.
- **If it fails:** no shadow; nothing changes.

**Confirmed as drafted (2026-10-15):**
- (a) the combination rule: either BEAR -> BEAR, otherwise the rule-based regime;
- (b) `rule_based-v1`, not v2, as the rule-based part;
- (c) the stricter pass criteria 1-5;
- (d) a pass leads only to a forward shadow, never to a live change from the backtest.

### 13.3.1 As built (2026-10-15, before the run)

- **Code:** `quant/regime/combo_trial.py`.
  - The combination and the descriptive "both must say BEAR" variant: BEAR only when both say BEAR; a rule-based BEAR alone becomes SIDEWAYS.
  - `rule_based-v1` recomputed from the stored inputs over the whole history, and the walk-forward HMM from `run_hmm_walk_forward`, as in section 13.
  - The point-in-time checks: the HMM part with section 13's `point_in_time_checks`; the rule-based part and the combination recomputed from inputs ending at the same two cuts.
  - The five gates, and the report. The sell-off table shows each BEAR spell's entry, measured from the peak, and its exit, measured off the low.
- **Tests:** `tests/test_regime_combo_trial.py`.
- **Command:** `python -m quant regime-combo-trial` is the one-time run. It writes `reports/regime/combo-trial-<time>.md` and stores nothing.

### 13.3.2 Result (2026-10-15; `reports/regime/combo-trial-20261015T141233Z.md`)

The run was a single one, made by you. The window from the fold rule was 2014-10-01 .. 2025-10-06 (11 validation years).

```text
gate                         status  observed
sharpe_higher                FAIL    regime_switch Sharpe 0.9610 vs 0.9841 with rule_based-v1
max_drawdown_no_deeper       PASS    -19.6% vs -25.0%
years_higher                 PASS    7 of 11 validation years (need >= 6)
min_avg_regime_dwell_days    PASS    42.6 sessions (need >= 10)
point_in_time                PASS    HMM 2/2, rule-based and combined 2/2 cuts identical (2018-04-05, 2025-04-08)
verdict                      FAILED
```

```text
regime_switch     CAGR    Sharpe   max DD   vol     invested   turnover
[rule_based-v1]   13.4%   0.9841   -25.0%   13.8%   76%        4.1x
[combined]        12.2%   0.9610   -19.6%   12.9%   75%        4.3x
```

- **Lower risk, not better risk-adjusted.** The drawdown is 5.4 points shallower, but the return is 1.2 points a year lower and the Sharpe ratio 0.023 lower, the same pattern as the HMM alone (section 13).
- **The HMM trigger works as expected on entry.** BEAR came earlier in the sudden sell-offs: 2020 at -9% from the peak (v1 -19%), 2025 at -12% (-19%). In 2018 Q4 and 2022 the combination equals v1, which was in BEAR first.
- **The late exit got worse, as stated before the run.** In 2020 BEAR ended 2020-06-02, at +38% off the low (v1: 2020-05-26, +34%). An earlier entry and a later exit roughly cancel.
- **Descriptive, decides nothing:**
  - `regime_switch_bear50` is higher with the combination: Sharpe 1.1371 vs 1.0853, max DD -26.1% vs -30.8%.
  - The "both must say BEAR" variant has Sharpe 1.0301, with the same max DD as v1 (-25.0%).
  - Both were seen only after the run. Choosing either now would be a selection on this data; it would need a new pre-registration and would be weaker evidence for it.

**Outcome of the section 13.3 trial: FAILED.**
- No shadow is added, and nothing changes: `rule_based-v1` stays ACTIVE, and forward-test run 1 continues unchanged.
- No other combination rule is tried on this data.
- Forward evidence on the HMM accumulates anyway, through `shadow_hmm`.

**Sections 12.5 and 13.3 together:**
- Both moved `regime_switch`'s Sharpe by about 0.03 (+0.031 and -0.023), well inside the about 0.1 that 11 years can tell apart.
- On development data, these changes shift when BEAR starts and ends, but they do not reliably improve the risk-adjusted return. `rule_based-v1` remains the baseline to beat.

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

### 15.4.6 Retrain on the 81-stock pool (registered 2026-10-12, before running)

The pool grew from 48 to 81 stocks (section 17.1.3). A cross-sectional predictor has more stocks to rank, which is a reason to expect a weak signal to trade better; the regime model trades a different pool. One retrain per model type with **everything else unchanged**: the configured search spaces, `max_trials`, folds, embargo, target, features, `ml_select` (top 10, keep 10, every 5 sessions), the HMM settings, and the gates in `retrain.gates` exactly as they are. The baseline is `regime_switch` on the same 81 stocks and folds. No holdout is used here (step 3 decides that, once, for the best candidate). Each model type's trial count rises by its search trials; nothing is promoted by this step.

**Result (2026-10-12; walk-forward 2014-09-26 .. 2025-10-01, after costs; baseline `regime_switch` on 81 stocks: CAGR 13.2%, Sharpe 0.97, max DD -25.0%, turnover 4.1x):**

```text
candidate                                     CAGR  Sharpe  max DD  turnover  folds beating  gates
hmm-20261012T174554Z-5f979a (regime_switch)  10.4%   0.976  -19.9%     8.2x   -              PASSED (4/4)
logistic_regression-20261012T174649Z-dd9714  17.3%   1.04   -23.3%    17.4x   6/11 = 55%     FAILED (folds)
xgboost-20261012T174934Z-9f4b96              14.9%   0.96   -33.1%    29.6x   7/11 = 64%     FAILED (Sharpe, DD)
random_forest-20261012T175038Z-1b254a        13.7%   0.84   -26.6%    18.1x   4/11 = 36%     FAILED (3 gates)
```

- **HMM passes every gate**, but narrowly: Sharpe 0.9757 vs 0.9712 with the rule-based regime. The fitted HMM is the same as on 48 stocks (it sees only SPY and VIX; same 95.1% agreement, 31.8-session dwell); what changed is the pool it trades. It trades lower risk for lower return (max DD -19.9% vs -25.0%, invested 58% vs 76%, CAGR 10.4% vs 13.2%) at twice the turnover. With bear exposure 0.5 the same pattern holds (Sharpe 1.09 vs 1.07, max DD -18.4% vs -30.8%).
- **Logistic regression** clears the Sharpe and drawdown gates with the 81-stock pool (it did not with 48: 0.92 and -24.5% vs 0.86 and -21.3%) and fails only the consistency gate, by one fold (6 of 11, 7 needed). The signal is still weak (AUC 0.535) and the strategy trades 17x a year against 4x.
- XGBoost and Random Forest stay below the baseline after costs, as on 48 stocks.

### 15.4.7 Pre-registered trial C: fold-wise selection among stock features (registered 2026-10-12, before running)

Motivated by a development-period check of the 23 stock features (2006-2025, holdout excluded): 11 have |t| below 1 against the 5-day excess return, the three `relative_return_*_spy` columns rank stocks exactly as `return_*` do (SPY's return is the same for every stock on a day), and the market and regime inputs are the same for every stock on a day, so they cannot change a linear model's ranking. Because that check used the same years the trial is tested on, the trial does **not** use its result to pick features; each fold picks its own.

- **Pool:** the 23 stock features minus `relative_return_1m_spy`, `relative_return_3m_spy`, `relative_return_6m_spy` (20 features). No market or regime inputs.
- **Selection inside each fold** (`ic_selector`, as in trial B): keep the pool features whose rank IC against the 5-day excess return on that fold's fit rows has |t| >= 2 on non-overlapping sessions; at least the single strongest. Calibration and validation data are never seen by the choice.
- **Models and fixed hyperparameters** (the best of the 81-stock search, 15.4.6; no new search): logistic regression C = 1.0; random forest max_depth 8, min_samples_leaf 20; XGBoost learning_rate 0.1, max_depth 5, min_child_weight 5, n_estimators 300, reg_lambda 1.
- **Everything else unchanged:** 81 stocks, folds, embargo, target, calibration, `ml_select` (top 10, keep 10, every 5 sessions), baseline `regime_switch` with the rule-based regime, and the gates. No holdout. One trial per model type (3 in all). A passing configuration would still need a `quant retrain` with this input selection before it could be registered or promoted.

**Result (2026-10-12; walk-forward 2014-09-26 .. 2025-10-01, after costs; baseline `regime_switch` Sharpe 0.97, max DD -25.0%):**

```text
                     with selection (trial C)                 all 35 inputs (15.4.6)
                     CAGR   Sharpe  max DD  turnover  AUC     Sharpe  max DD  AUC
logistic_regression  15.0%   0.86   -35.4%   21.0x   0.504    1.04   -23.3%  0.535
random_forest        12.5%   0.75   -33.6%   22.2x   0.508    0.84   -26.6%
xgboost              12.2%   0.79   -31.5%   31.6x   0.501    0.96   -33.1%
```

Every model fails three gates (Sharpe, 4/11 folds, drawdown). The folds chose the same 1-5 inputs for all three models: `return_5d` and `return_12m` (9 of 11 folds each), `return_6m` (6), `sma50_vs_sma200` (5), `price_vs_sma200` (1); fewer in the early folds (only `return_5d` in folds 1-2).

- Screening features one at a time throws away information the models use **in combination**: features that are noise on their own (|t| < 1) help when read together with others, e.g. a stock far above its 200-day average *given* its 6-month return (the full model's `price_vs_sma200` coefficient is negative while its own IC is positive). With 1-5 inputs the probabilities are no better than chance (AUC about 0.50) and the selection churns more.
- Dropping the market and regime inputs cannot by itself change a linear model's daily ranking, so the loss comes from the stock features removed.
- The configured 35 inputs stay. Univariate screening is not repeated; a multivariate alternative (e.g. L1-regularised logistic regression, which drops inputs inside the fit) would be a new pre-registered trial.

### 15.4.8 Holdout check of the logistic regression (registered 2026-10-12, before viewing the holdout)

The first evaluation of the holdout (section 15.2): `quant retrain --role predictor --model logistic_regression --evaluate-holdout`, with the 15.4.6 configuration unchanged (81 stocks, 35 inputs, the same search). The holdout is the last 12 months of stored sessions; the comparison is `ml_logistic_regression` against `regime_switch` with the rule-based regime on the same sessions, after costs. Decided before looking:

| Holdout result | Action |
|---|---|
| Sharpe at least as high **and** max drawdown no deeper | Candidate for promotion with `--override` (the consistency gate still failed), with the rule-based selection tracked in shadow (Phase 10) |
| One better, one worse | No promotion; the logistic regression runs in shadow in Phase 10 |
| Both worse | No promotion; shadow optional; the rule-based stock selection stays |

Whatever the result, the model is not changed and re-checked on this holdout; later research is judged on development data and forward testing. One year is a short test (a single year's Sharpe has an uncertainty of about +/-1), so the result is weak evidence either way.

**Result (2026-10-12, holdout evaluation #1; holdout 2025-10-10 .. 2026-10-01, after costs; candidate `logistic_regression-20261012T183319Z-cf2d50`, the same configuration and walk-forward result as 15.4.6):**

```text
                        CAGR   Sharpe  max DD  invested  turnover
ml_logistic_regression  17.0%   0.86   -14.5%     75%      17.1x
regime_switch           35.4%   1.70   -10.7%     91%       4.6x
SPY                     17.3%   1.32    -8.9%    100%
equal_weight            33.8%   2.08    -8.4%     99%
```

**Both worse** (Sharpe 0.86 vs 1.70, max DD -14.5% vs -10.7%): under the rule above, no promotion, and the rule-based stock selection stays; shadow tracking of the logistic regression in Phase 10 is optional. The `holdout_not_worse_than_baseline` gate fails as well. In a strong, broad rally the model's weekly picks trailed the pool (equal weight +33.8%) while trading almost four times as often.

### 13.2 Holdout check of the ACTIVE HMM regime (registered 2026-10-12, before viewing)

The HMM `hmm-20261012T174554Z-5f979a` was promoted on its development gates (15.4.6), which never use the holdout. Its labels over the holdout year (seen while comparing regimes, no returns) disagree with the rule-based regime on most days (it called 151 of the rule-based regime's 234 BULL days SIDEWAYS), so its trading result there is checked once (holdout evaluation #2; the predictor's check, 15.4.8, was #1).

- **Compared:** `regime_switch` (baseline bear exposure 0) with the HMM's stored regime vs with the rule-based regime; same 81 stocks, sizing, costs and minimum trade size as `quant backtest`; simulated over the full history so holdings carry into the window; metrics on the holdout sessions only (2025-10-10 .. latest). The HMM's regime there is its final fit (trained to 2025-10-01) applied forward: out of sample.
- **Decided before looking:**

| Holdout result (HMM vs rule-based, regime_switch) | Action |
|---|---|
| Sharpe at least as high **and** max drawdown no deeper | Keep the HMM ACTIVE |
| One better, one worse | Keep it ACTIVE, with the rule-based regime in shadow in Phase 10 |
| Both worse | `quant retire` it; the rule-based regime returns, with the HMM in shadow in Phase 10 |

One year is a short test; the HMM is not refit or changed because of this result.

**Result (2026-10-12, holdout evaluation #2; 2025-10-10 .. 2026-10-01, after costs; `regime_switch`, 81 stocks):**

```text
                       as registered (holdings carried in)      started from cash on 2025-10-10
regime               CAGR   Sharpe  max DD  invested turnover    CAGR   Sharpe  max DD
rule-based          +26.4%   1.39   -10.7%    90%      4.3x     +35.4%   1.70   -10.7%
HMM (ACTIVE)         +8.5%   0.73    -7.2%    54%     12.5x     +10.0%   0.87    -7.2%
regime days in the window: rule-based BULL 228 / SIDEWAYS 10 / BEAR 7; HMM BULL 77 / SIDEWAYS 168
```

- **Mixed** under the rule above (Sharpe worse, drawdown shallower): the HMM stays ACTIVE and the rule-based regime runs in shadow in Phase 10.
- The two halves are not of equal size: the HMM gave up about 18 points of return and half the Sharpe ratio for a drawdown 3.5 points shallower, because it called two thirds of a steady rally SIDEWAYS (regime_switch then holds mostly cash) and switched often (12.5x turnover). The rule did not weigh the two sizes; it is not changed after the fact, and keeping or retiring the HMM remains the owner's decision.
- **Decision (2026-10-12, owner):** retired (`quant retire`, logged with the reason). The rule-based regime is back in use; the HMM is to run in shadow in Phase 10. `quant promote hmm-20261012T174554Z-5f979a` would restore it (its development gates passed, so no override is needed).
- The fresh-start columns reproduce the predictor check's baseline (15.4.8, +35.4%, Sharpe 1.70); the gap between the two starts is path dependence: a fresh start buys the whole target list at equal weight on day 1, while carried positions drift within the 2-point minimum trade size.

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

Implementation (Phase 9): with an ACTIVE HMM, `calculate-regime` also runs its final fit forward (filtered probabilities, the same dwell) and stores it in `market_regimes` as method `HMM`, `model_version` = the model id (`quant/regime/active.py`); the signals then use that regime. With an ACTIVE predictor, `ml_select` drives the signals automatically (printed), with `score` = calibrated probability and `confidence` = rank percentile (1.0 = ranked first), and the signals are stored under the predictor's model id. A model whose inputs no longer match the configuration (HMM observations, feature version, a regime it depends on that is not stored) or whose file is missing is refused with a message. `quant status` names the strategy and regime in use and warns when the ACTIVE models' outputs are behind.

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

### 17.1.3 Stock pool expansion by rule (registered 2026-10-12, before any result with it)

The pool grew by hand (48 stocks, then 13 added on 2026-10-11 after their strong runs, 61 in all). Twenty more are added by a fixed rule, so this step picks no winners: **the 20 largest S&P 100 members by market cap that are not already in the pool**, one share class per company (GOOG skipped: GOOGL is in). Constituents from the S&P 100 components table (101 members, Wikipedia, read 2026-10-12); market caps from Yahoo `fast_info` on 2026-10-12; sectors as listed there. Added (largest first): META, AVGO, V, MA, ABBV, AMAT, LRCX, DELL, MRK, PANW, GE, PM, MS, ANET, GEV, RTX, LIN, C, SNDK, IBM (the 20th, IBM, $213B; the next, AXP, $210B). BNY did not resolve on Yahoo and is far below the cut.

- The new pool (81 stocks) is a new trial for every strategy and model; the 61-stock results stay as they are.
- Still survivorship-biased: today's largest companies are, by construction, ones that did well. GEV (2024) and SNDK (2025) have short histories and contribute little to backtests.
- Hindsight check, as before: results are also reported for the 48-stock pool and the 68 stocks without the 13 hand-picked additions of 2026-10-11.

**Result (2026-10-12, development 2005-10-17 .. 2025-10-09, after costs; 81-stock batch `20261012T174259Z-994221`; the two checks without the random baseline):**

```text
                     81 stocks (all)          68 (no hand-picked)      48 (original)
                     CAGR  Sharpe  max DD     CAGR  Sharpe  max DD     CAGR  Sharpe  max DD
regime_switch        13.2%  1.03  -25.0%      12.0%  1.00  -24.1%      11.2%  0.97  -21.3%
regime_switch_bear50 14.0%  1.03  -30.8%      12.8%  1.00  -27.4%      11.7%  0.96  -27.0%
trend                15.6%  0.97  -39.0%      14.3%  0.95  -37.3%      12.2%  0.86  -39.1%
equal_weight         16.9%  0.97  -41.2%      16.8%  0.94  -43.0%      16.8%  0.90  -45.1%
SPY                  11.1%  0.64  -55.2%      (same)
```

- `regime_switch` stays the best risk-adjusted strategy on every pool; the rule-based 20 add a little (Sharpe 0.97 -> 1.00) and the 13 hand-picked stocks a little more (-> 1.03), with a somewhat deeper drawdown (-21% -> -25%).
- On 81 stocks it beats 70% of the random baselines on Sharpe and 100% on CAGR (48 stocks: 94%); a larger pool of mostly strong stocks makes random picks from it harder to beat, so the stock choice adds less than the regime timing.
- `regime_switch` remains the baseline that the models (next) must beat.

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

Added during implementation (2026-10-11, step 4; before any signal was acted on):

- A held stock **without data at T** gets HOLD with a warning, not EXIT. A missing input never counts as a signal (section 17), and a gap in today's data is more likely a data problem than a trend break. The backtest never meets this case (a held stock always has data).
- EXIT sells the whole holding, including a fractional remainder; BUY and REDUCE round down to whole shares and REDUCE never sells more than is held. A BUY or REDUCE below one share is flagged and suggests 0 shares.
- Rows are ordered EXIT, REDUCE, BUY, HOLD (sells fund the buys), then by target weight.

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
python -m quant generate-signals [--reasons]
                                       # steps 7-12: BUY/HOLD/REDUCE/EXIT for the latest session vs positions.csv
python -m quant run-daily [--reasons]  # full daily pipeline (steps 1-13)
python -m quant retrain --role predictor|regime [--model logistic_regression|random_forest|xgboost] [--evaluate-holdout]
                                       # MANUAL; validates, then registers a CANDIDATE
                                       # (implemented: --role predictor with logistic_regression / random_forest / xgboost,
                                       #  --role regime fits the HMM; holdout count shared with backtest; the regime
                                       #  role has no holdout gate, so --evaluate-holdout is rejected there)
python -m quant promote <model_id> [--override "<reason>"]
                                       # activate a candidate for its role (refuses failed gates)
python -m quant retire <model_id> [--reason "<why>"]
                                       # ACTIVE -> RETIRED; the role falls back to rule-based (logged)
python -m quant backtest [--strategy NAME ...] [--evaluate-holdout]
                                       # rule-based strategies vs SPY, equal weight, random baseline;
                                       # holdout only with the flag, every evaluation logged
python -m quant status                 # last runs, data freshness, active model per role, stale warnings
streamlit run dashboard/app.py         # dashboard (separate process)
```

If the user later wants automation, a scheduler can be added in a later phase without changing the pipeline, because every command is idempotent and restartable (section 30).

---

# 25. Streamlit Dashboard

Read-only view of the database. Implemented (Phase 9) as `dashboard/app.py` with pages Status, Market, Signals, Stock detail, Models and Backtest; the calibration plot and feature importance are in the validation reports, which the Models page shows.

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

Implementation (`quant/logs.py`, Phase 9): every command logs to `<database directory>/logs/quant.log` (`data/logs/quant.log`, git-ignored with `data/`), rotated at `logging.max_megabytes` with `logging.backups` old files, at `logging.level`, with UTC timestamps. Each command logs its start, its finish (exit code, duration) or its crash with the traceback; every message printed to stderr is also an ERROR line; every `pipeline_runs` row (ingestion per symbol, features, regime, signal stages, backtests, holdout views, retrains, promotions with any override and reason, run-daily) is also a log line; download failures, quarantined rows and signal warnings are WARNING lines. Terminal output is unchanged. The log sits next to the database it describes, so `--db` and test databases log next to their own file.

---

# 27. Configuration

*Removed 2026-10-15 (cleanup), because no code read them, so changing them had no effect:*
- `market.timezone` and `market.execution`. The trading calendar and its time zone come from `market.calendar` (`exchange_calendars`). Filling at the next open is built into the backtest and the signals (section 4.5).
- `altdata.sec.signal_codes` and `altdata.sec.exclude_10b5_1_sales` (`altdata_proposal.md` section 13.2).

```yaml
database:
  url: sqlite:///data/quant.db
  busy_timeout_ms: 30000
  backup_directory: data/backups   # `quant backup-database` writes dated copies here (git-ignored with data/)
  backups_to_keep: 5              # the newest 5 copies are kept, older ones deleted

market:
  provider: yfinance
  request_timeout_seconds: 20
  calendar: XNYS                   # exchange_calendars name for the NYSE trading calendar
  initial_history: 2005-01-01      # regime work needs more than one or two bear phases
  # Stock pool by GICS sector (sector -> stocks), used for the sector cap (risk.max_sector_weight).
  # Large caps listed before 2005, avoiding stocks with large spin-offs (spin-offs are not
  # modelled as corporate actions). Chosen today, and sectors are today's classification,
  # so backtests carry survivorship bias (section 20).
  symbols:
    information_technology: [AAPL, MSFT, NVDA, ORCL, CSCO, INTC, ADBE, TXN, QCOM, AMD, MU, PLTR, RMBS, LITE, FIG, AVGO, AMAT, LRCX, DELL, PANW, ANET, SNDK, IBM]
    communication_services: [GOOGL, DIS, NFLX, META]
    consumer_discretionary: [AMZN, HD, NKE, SBUX, LOW, TJX, TSLA, LULU, SE, CCL]
    consumer_staples: [PG, KO, PEP, WMT, COST, PM]
    health_care: [JNJ, UNH, AMGN, LLY, TMO, TEM, ABBV, MRK]
    financials: [BRK-B, XYZ, JPM, BAC, WFC, GS, BLK, V, MA, MS, C]
    industrials: [CAT, UNP, LMT, DE, SPCX, GE, GEV, RTX]
    energy: [XOM, CVX, SLB]
    materials: [SHW, NEM, LIN]
    utilities: [NEE, SO, AEP]
    real_estate: [O, PSA]
  market_symbols: [SPY, QQQ, ^VIX, TLT, IEF, HYG, ^TNX, ^IRX, EURUSD=X]   # TLT / IEF / HYG: bond inputs (12.5); ^TNX / ^IRX: Treasury yields in percent (12.7); EURUSD=X: USD per EUR for the broker snapshots (run 2)

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

# Strategies (section 17). `quant backtest` runs: trend (no settings; SMA windows from features.sma),
# mean_reversion, regime_switch and regime_switch_bear50 (from regime_switch.bear_exposures), all with the
# rule-based regime. ml_select needs a PREDICTOR and is backtested by `quant retrain --role predictor`.
# The keys below are settings sections, not a list of strategies.
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
                                   # (once the forward test starts it is read only once, as its start)
  trades_path: data/trades.csv     # forward test: date,symbol,side,quantity,price,fees,amount after the start

signals:                           # section 18.1; BUY / HOLD / REDUCE / EXIT at the close of T
  strategy: regime_switch          # a rule-based strategy from section 17.1 (baseline or a variant)
  hold_band: 0.02                  # HOLD while |target - current| is below this; = backtest.min_trade_weight

logging:                           # section 26; written to <database directory>/logs/quant.log
  level: INFO                      # DEBUG | INFO | WARNING | ERROR
  max_megabytes: 5                 # rotate the file at this size ...
  backups: 5                       # ... keeping this many old files

forward_test:                      # Phase 10 pre-registration (section 29); frozen once per session, never traded
  shadows:                         # portfolio name -> strategy, and the regime model or predictor it uses
    shadow_hmm: {strategy: regime_switch, regime_model: hmm-20261012T174554Z-5f979a}
    shadow_lr: {strategy: ml_select, predictor: logistic_regression-20261012T183319Z-cf2d50}
    shadow_rule_bonds: {strategy: regime_switch, regime_rule: rule_bonds-v1}   # added 2026-10-15 (section 12.5.3)

sentiment:                         # creator opinion factor (sentiment_proposal.md); research only, not in run-daily
  lookback_days: 14                # `quant sentiment fetch` looks this far back (--days overrides)
  min_duration_seconds: 180        # shorter videos (Shorts) are skipped
  include_live: false              # live-stream replays
  request_delay_seconds: [8, 15]   # random wait between requests to YouTube (shorter waits got the IP blocked)
  max_captions_per_run: 10         # captions fetched per run, oldest first; the rest waits for the next run
  no_captions_retry_days: 3        # a video without captions is asked again while younger than this
  gold:                            # hand-labeled answer key for extraction (sentiment_proposal.md 4.6)
    videos: 10                     # how many videos `quant sentiment gold-export` picks
    labels_dir: research/sentiment_ai_reference   # labels (videos.csv, opinions.csv); committed. Now an AI
                                                  # reference (GPT / MAI), not human labels: see SOURCE.md
    transcripts_dir: data/sentiment_gold/transcripts  # reading copies; git-ignored, delete any time
  extraction:                      # opinions from transcripts (sentiment_proposal.md section 4)
    backend: claude_code           # claude_code (your Pro subscription, `claude -p`); gemini comes later
    model: sonnet                  # placeholder until the S2 comparison; `extract --model haiku` overrides
    report_models: [sonnet, haiku] # reports use, per video, the first of these that extracted it
    prompt_version: 1              # bump when the prompt changes; each version's opinions are kept apart
    chunk_minutes: 10              # transcripts are sent in pieces of this length ...
    chunk_overlap_seconds: 30      # ... each starting this much before the previous one ends
    quote_match_threshold: 0.85    # starting value; set per language in S2
    max_videos_per_run: 40         # more needs --yes
    call_timeout_seconds: 300
    aliases_path: config/sentiment_aliases.yaml        # names per ticker, kept by hand
    tickers_cache: data/sentiment/us_tickers.json      # US-listed symbols (Nasdaq Trader), refreshed monthly
  summary:                         # daily report: a short summary per stock from VERIFIED opinions only
    backend: claude_code
    model: haiku                   # the input is short and already clean, so a cheaper model is enough
    prompt_version: 1
    reports_dir: reports/sentiment # one Markdown report per trading day: reports/sentiment/<date>.md

  channels:                        # chosen after S0 (sentiment_proposal.md section 12.1)
    - {handle: "@FinancialEducation", channel_id: UCnMn36GT_H0X-w5_ckLtlgQ, name: "Financial Education", language: en, style: "aggressive growth"}
    - {handle: "@chipstockinvestor", channel_id: UC3aD-gfmHV_MhMmcwyIu1wA, name: "Chip Stock Investor", language: en, style: "semiconductors / AI"}
    - {handle: "@NaNaShuoMeiGu", channel_id: UCFhJ8ZFg9W4kLwFTBBNIjOw, name: "NaNa说美股", language: zh, style: "daily market wrap"}
    - {handle: "@yttalkjun", channel_id: UCRBrH2qS7yKGMNSmjnj8gcw, name: "投资TALK君", language: zh, style: "macro + sectors"}
    - {handle: "@Zac-Stock-Journal", channel_id: UCawshLsWXrFnYhAlnmn0h2A, name: "Zac美股笔记", language: zh, style: "technical buy points, own holdings"}
    - {handle: "@nick_stockcafe", channel_id: UCjrP2TtSTifuRJ76hW2IW1A, name: "Nick 美股咖啡館", language: zh, style: "own buys, Traditional Chinese"}
    - {handle: "@laotangmeigu", channel_id: UCeTEWFsNC3eeUsn9hvsDALQ, name: "老汤美股财经", language: zh, style: "single-stock analysis"}
    - {handle: "@EthanFinanceNote", channel_id: UCvf_c9B7iL0x7aqdhxI4zRg, name: "Ethan的美股笔记", language: zh, style: "own trades"}
    - {handle: "@parkevtatevosiancfa9544", channel_id: UCwKB_00dPL3x5XmHF9IJCrg, name: "Parkev Tatevosian, CFA", language: en, style: "one stock, buy or sell verdict"}
    - {handle: "@EverythingMoney", channel_id: UChBVf9YnourrEDTsbbwJPRA, name: "Everything Money", language: en, style: "valuation, buy / sell"}
    - {handle: "@DividendTalks", channel_id: UClLyc3C2K3C-RAh7eEJAwyg, name: "Dividend Talks", language: en, style: "dividend stocks"}

altdata:                           # free structured alternative data (altdata_proposal.md); research only, not in run-daily
  version: 1
  user_agent: "mini-quant-trader Chenghao Zhang chenghao.zhang@outlook.de"  # sent to the SEC with every request
  sec:                             # EDGAR: Form 4 insider trades, Schedule 13D activist stakes
    max_requests_per_second: 5     # the SEC allows 10
    insider_backfill_from: 2006-01-01   # set from the A0 spike
    extra_ciks:                    # earlier CIKs whose filings belong to a pool stock (A1 finding)
      XOM: [34088]                 # Exxon Mobil Corp; company_tickers.json now gives the new holding company
      GOOGL: [1288776]             # Google Inc. (before Alphabet, 2015)
      AVGO: [1441634, 1649338]     # Avago Technologies, Broadcom Ltd (before Broadcom Inc., 2018)
      DIS: [1001039]               # the old Walt Disney Co (before the 2019 holding company)
      LIN: [884905]                # Praxair (before the 2018 Linde merger; LIN's prices go back through it)
      BLK: [1364742]               # the old BlackRock, Inc. (before the 2024 holding company)
      PSA: [318380]                # Public Storage Inc (before the 2007 reorganization)
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

Done (2026-10-11): `quant/signals/` (positions, targets, engine, reasons, pipeline), the `positions` and `signals` tables, `quant generate-signals` and `run-daily` stages 7-12. The live target at T equals the backtest's weight at T for every rule-based strategy, and a database that ends at T gives exactly the signals of the full history cut at T (`tests/test_signal_leaks.py`). The ML probability input waits for an ACTIVE predictor; none passed its gates.

## Phase 9 — CLI operations and dashboard

`run-daily`, `retrain` (manual), `promote`, `backtest`, `status`; Streamlit monitoring.

Done (2026-10-12): `quant status` and `quant promote` (`quant/ops/`; one ACTIVE model per role enforced by a unique index), logging (section 26), ACTIVE models in daily use (an HMM regime stored and used by the signals; an ACTIVE predictor's predictions stored with input hashes and `ml_select` driving the signals; section 16.3), and the dashboard (`dashboard/app.py`, its data in `quant/dashboard/data.py`; every page smoke-tested with Streamlit's test runner). No model has passed its gates, so the rule-based path is what runs.

## Phase 10 — Forward testing, then paper trading

Log live signals and score them against realized outcomes (true out-of-sample evidence). Only after that, consider paper trading:

```text
Signal Engine -> Paper Trading -> Broker API
```

Do not implement live trading in V1.

Built (2026-10-12): `quant/forward/` (freeze, paper, real, report), `quant forward-test [--start | --end --reason]`, `quant forward-report [--run N]`, `run-daily` stage 5, the `status` checks and the dashboard's Forward test page (steps 1-8). The forward test itself begins when the owner runs `--start`; the first look is after about 126 forward sessions.

### Phase 10 pre-registration (2026-10-12, before any forward data)

Fixed before the first forward session; a change after forward data exists is logged here as a design change, never a silent edit.

**Start.** The first session frozen by `run-daily` after Phase 10 is built. *Design change (2026-10-12, step 6, before any forward data):* the start fixes the real account's starting positions and value for good, so it is an explicit one-time command, `quant forward-test --start`, run after a close and before the next open with the real account in `data/positions.csv`; until then `run-daily`'s forward stage only reports "not started", and starting after a session's deadline is refused. After the start, `run-daily` freezes every evening (stage 5, `forward-test`) and updates `forward_equity`; `quant status` counts frozen and missed sessions and warns when tonight's session is not frozen yet, when the trade log has an error, and when the positions file is edited after the start (it is no longer read). *Design change (2026-10-12, before any forward data): runs.* A forward test is a numbered run (`forward_runs`; at most one open, enforced by a unique index). `quant forward-test --end --reason "<why>"` ends the open run: it is kept, never deleted, with its end date and reason, and logged; `--start` then begins a new run from that evening's session with the positions file as its start. `quant forward-report` shows the open run and lists every other run with its dates and end reason (`--run N` shows one); an ended run is reported up to its last frozen session. The trade log is one file across runs: a run counts only trades dated after its start (the report says how many it left out). Restarting is for a wrong start, a changed setup or a long gap, never because results look bad; every restart stays visible. No backfill: a session the system did not freeze on its own evening is never added later (a missed session is counted, and paper portfolios keep their last targets over it).

**Portfolios** (all from the same stored data each evening):

| Name | What | Role |
|---|---|---|
| `live` | `signals.strategy` (`regime_switch`) with the regime in use (rule-based) | what the signals show |
| `shadow_hmm` | `regime_switch` with the regime of `hmm-20261012T174554Z-5f979a` (RETIRED; its final fit applied forward, filtered, same dwell) | shadow, never traded |
| `shadow_lr` | `ml_select` with `logistic_regression-20261012T183319Z-cf2d50` (35 inputs; probabilities from its saved final model), rule-based regime for exposure | shadow, never traded |
| `shadow_rule_bonds` | `regime_switch` with `rule_bonds-v1` (section 12.5, tested as `rule_based-v2`), computed each evening; added 2026-10-15 (design change below) | shadow, never traded |
| `real` | the owner's account: `data/positions.csv` once at the start, then `data/trades.csv` replayed | what was actually held |
| `SPY` | buy-and-hold from the start | benchmark |

Shadows are listed in `forward_test.shadows`; they never change the signals.

*Decision (2026-10-16, owner): run 1 is a paper test; `live` is the evidence.*
- The owner does not trade the signals or record fills, and `data/trades.csv` stays empty. The real account held about 8% of the live targets in the first sessions.
- `real` therefore shows the owner's own account, valued from the start positions; it is **not** the strategy as traded. The followed share and the execution cost measure nothing in this run.
- The 12-month comparison uses `live` and the shadows only.
- The checkpoints and the rules are unchanged.

*Design change (2026-10-15, during run 1, after three forward sessions): `shadow_rule_bonds` added*, as decided by the holdout check of section 12.5.3.
- Run 1 is not restarted, and `live`, the other shadows and `real` are unchanged.
- The new shadow is frozen from the first evening after it was added (2026-10-15), with no backfill. Its earlier sessions (2026-10-12 .. 10-14) count as missed, and its paper portfolio holds cash over them.
- Its 12-month comparison with `live` uses the rule below and runs to about 252 sessions after 2026-10-15, so it ends a few days after run 1's verdict.

**Frozen log.** Each evening, once per session, the targets of `live` and every shadow (weight per stock, the as-traded close of T) are written to a log that is never updated or deleted; a re-run of a frozen session leaves it as it is.

Implementation (step 1, `quant/forward/freeze.py`): tables `forward_sessions` (one row per session and portfolio, also when it targets nothing: strategy, regime, regime source, model version, invested, positions, frozen_at) and `forward_targets` (one row per held stock: target weight, as-traded close, probability for `ml_select`). "Its own evening" is operational: a session is frozen only before the next market open (NYSE calendar), while its signals can still be acted on; after that the run logs it as SKIPPED (no backfill). Shadows are computed from the same stored data as the signals: `shadow_hmm` runs the retired model's final fit forward (filtered, same dwell) without storing it in `market_regimes`; `shadow_lr` scores every session after its training end from its saved model.

**Paper portfolios** (`live`, shadows, `SPY`) start from cash equal to the `real` portfolio's value at the start session and trade their frozen targets with the backtest engine: fill at the next open, `backtest.fee_bps` and `slippage_bps`, `backtest.min_trade_weight`.

Implementation (step 2, `quant/forward/paper.py`): the start session and start value are written once at the first freeze (`forward_start`; the positions file's managed holdings plus cash at that close, as the signals value it; the notional `backtest.initial_cash` without a file) and never move. Every run re-simulates each paper portfolio from the start with the backtest engine on its frozen targets (a missed session keeps the previous targets) and replaces `forward_equity` (date, portfolio, value, invested): the frozen targets are the record, the values are derived from them and the prices. Paper values use total-return prices, as the backtest; dividends the real account receives count only if recorded (step 4).

**Real portfolio.** Holdings at each close are the starting positions plus every trade dated on or before that session (`date, symbol, side, quantity, price`, optional `fees`); cash follows from the trades. Valued at the as-traded close. Deposits and withdrawals are not tracked (owner's choice): any would show up as return. A trade added later with its real date corrects the real history; frozen targets are never changed by it.

Implementation (step 4, `quant/forward/real.py`): `positions.trades_path` (`data/trades.csv`) has `date,symbol,side,quantity,price,fees,amount`; `side` is BUY, SELL or **DIVIDEND** (`amount` = cash received: part of the return, not a deposit). The positions file is read once at the first freeze and stored (`forward_start_positions`); from then on the **signals also take the current holdings from the start plus the trade log**, so the positions file is not edited again. Every holding counts, in the pool or not, at as-traded closes; one without stored prices keeps its last trade price or cost basis (with a warning). Refused with the line: a trade on or before the start session, after the latest stored session, or selling more than is held; negative cash is a warning. Followed share on session T compares the close of T with the `live` target frozen on T-1 (the target that day's trades were meant to reach). Execution cost per BUY / SELL: the fill price against the as-traded open of its date, fees included, in basis points.

**Scores** (`quant forward-report`, dashboard): for every portfolio since the start: total return, annualised volatility, Sharpe (daily, no risk-free rate), max drawdown; sessions covered and missed. For `real` also: the **followed share** each day, the sum over stocks of min(actual weight, `live` target weight); and the **cost of execution**, the owner's fill prices against the next open after the signal.

Implementation (step 5, `quant/forward/report.py`, `quant forward-report`): read-only; the paper portfolios are re-simulated from the frozen targets and the real account replayed from its trade log in memory. It prints the start, the checkpoint (sessions to the 6-month first look and the 12-month verdict), each portfolio's return, volatility, Sharpe, max drawdown and missed sessions, the average and latest followed share, and the notional-weighted execution cost against the open next to the backtest's assumed fee + slippage.

**Checkpoints.** A first look after **6 months** (about 126 forward sessions); no promotion, retirement or rule change is made on forward evidence before it. A verdict after **12 months** (about 252 sessions). One year's Sharpe has an uncertainty of about +/-1, so even then a difference is evidence, not proof. A shadow becomes a candidate to replace `live` only if, over the 12 months, its Sharpe is higher **and** its max drawdown no deeper than `live`'s; the decision stays the owner's.

### Forward-test run 2 design: `real` read from the brokers (registered 2026-10-17, before run 2 starts)

**Why a new run.** Run 1 needs the start positions and every trade typed in by hand (`data/positions.csv`, `data/trades.csv`). Run 2 reads the owner's two accounts through their read-only APIs (Trading 212, EUR; IBKR, USD; connection checked 2026-10-17). This is a changed setup, which the restart rule allows; the results of run 1 play no part in the decision.

**What `real` means in run 2 (owner's decisions, 2026-10-17):**
- `real` is the owner's **own, discretionary portfolio**: both accounts **combined, in USD**. The owner does **not** follow the signals.
- Run 2 compares that portfolio with what the system would have done (`live`) and with the shadows. It does not measure execution of the signals.
- **Daily holdings snapshots only**: no trade import, so no execution cost. The followed share stays as a descriptive overlap measure (how much the owner's holdings resemble the live targets).

**Rules** (confirmed by the owner 2026-10-17):
1. **Snapshot per session.** Positions, cash and each broker's own valuation for session T. The two brokers report at different times:
   - Trading 212 answers with live prices, so it is read after the US close of T.
   - IBKR's Flex statement for T is only available the next morning (its "Last Business Day" report covers the previous session).
   - The `real` value of T therefore exists once both snapshots of T are stored, at the latest before the next open. A broker missing for a session carries its last value forward, and the session is counted as incomplete.
2. **Currency.** Trading 212 values are converted to USD with the close of `EURUSD=X` on T, stored with the snapshot. The report also shows the part of `real`'s return that comes from EUR/USD (the EUR value times the rate change), so FX is visible rather than mixed in.
3. **Deposits and withdrawals** must not count as gains or losses.
   - Read them from the brokers: Trading 212's transaction history, and IBKR's Flex section "Deposits & Withdrawals" (to add to the query).
   - The daily return of `real` is (value at T - net deposits on T) / value at T-1 - 1 (time-weighted). Dividends and interest stay part of the return.
4. **Holdings outside the pool** (European listings, ETFs, BEKE) count in `real`'s value at the broker's prices; they are unmanaged.
   - **They are not added to the stock pool** (owner's decision, 2026-10-17). The system is built for US listings: NYSE calendar, USD, next-open execution, SPY benchmark, US sectors. The UCITS ETFs SXR8 and EQQQ duplicate SPY and QQQ. Adding BEKE because it is held would break the pool-expansion rule (section 17.1.3). Changing the pool would also change `live`, the system being compared with.
   - **`real` split into three parts, descriptive.** The report shows each part's value, weight and return contribution per session:
     - **in the pool**: holdings the system could also hold (e.g. MSFT, META, XYZ, BRK-B), to compare stock choices on the same universe;
     - **outside the pool**: the European listings, the ETFs, BEKE;
     - **cash**, in both accounts, in USD.
   - This shows whether `real`'s lead or lag against `live` comes from choices inside the pool or from assets the system never holds. The split decides nothing.
5. **Start.** The start value of the paper portfolios (`live`, the shadows, SPY) is `real`'s USD value at the start session, as in run 1. The ending of run 1 and the start of run 2 happen on the same evening, once the new code is built and tested; until then run 1 continues unchanged.
6. **Unchanged:** the freeze, the shadows, the checkpoints (6 and 12 months), and the comparison rule for the shadows.

**To build:**
- tables for the broker snapshots, the FX rate and the cash flows (new tables only);
- `quant broker sync-accounts` (run with or after `run-daily`, and once more before the next open for IBKR);
- `real` computed from the snapshots instead of from the trade log;
- the report and the dashboard updated, plus the "Broker accounts" page.

**As built (2026-10-17, before run 2 starts):**
- **Clients** (read-only): `quant/broker/trading212.py` and `quant/broker/ibkr_flex.py`.
  - Credentials come from `.env` (template `.env.example`).
  - Trading 212's instrument metadata gives each holding's current ticker (FB_US_EQ -> META) and currency; it is cached in `data/broker/`.
- **Snapshots:** `quant/broker/tables.py` adds three new tables, `broker_account_snapshots`, `broker_position_snapshots` and `broker_cash_flows`; `quant/broker/sync.py` fills them via `quant broker sync-accounts`.
  - **Trading 212** is stored only between the US close and 07:00 UTC of the next weekday. Outside that window it is refused, and IBKR is still stored.
  - **USD rates:** `EURUSD=X` was added to `market.market_symbols`. 126 of its 5,652 Yahoo days were quarantined (a close outside the day's high/low); the rate lookup takes the latest stored day within 5 days.
- **`real`:** `quant/forward/broker_real.py` holds the start, the value, the time-weighted returns, the three parts, the FX contribution, the carried-forward sessions, and the holdings for the signals.
  - A run is a broker run when its `value_source` is `BROKER_VALUE_SOURCE`, so no existing table changes.
  - It is started with `quant forward-test --start`. Since 2026-10-17 every new run starts this way (owner's decision: only one kind of forward test from run 2 on); run 1's positions file and trade log stay only for reading run 1.
- **First real snapshot (session 2026-10-16):**
  - Trading 212 $12,399 (EUR 11,058 at 1.1213), IBKR $17,455, so $29,854 combined.
  - Positions plus cash match each broker's own total exactly for IBKR, and within $10 (0.08%, the brokers' own exchange rates) for Trading 212.
  - 15 deposits and withdrawals were stored (2020-11 .. 2026-10).
  - The parts: in the pool $7,159 (24%), outside the pool $17,373 (58%), cash $5,322 (18%).
- **In `run-daily`:** stage 4 of 6, after the regime and before the signals. It never stops the run; without credentials it is skipped. The run-daily tests replace it with a stand-in: a first version without that let the test suite call the real broker APIs and rewrite the 2026-10-16 snapshot (with identical values) four times; fixed the same day.
- **Dashboard:** a new "Broker accounts" page. The Forward test page shows the parts for a broker run.
- **Tests:** `test_broker_clients.py`, `test_broker_sync.py`, `test_forward_broker_real.py`, and a broker-run start in `test_forward_freeze.py`.

**Run 1 ended, run 2 started (2026-10-17, owner):**
- **Run 1** (2026-10-12 .. 10-16, 5 frozen sessions, 0 missed) was ended with the reason "real now read from the broker APIs". It is kept (`quant forward-report --run 1`).
- **Run 2** started on session **2026-10-16** with **29,854.19 USD**: Trading 212 $12,399 plus IBKR $17,455, both snapshots of 2026-10-16. 2026-10-16 was frozen for `live` and the three shadows before the deadline (2026-10-19 13:30 UTC). A database backup was recommended before the switch.
- **Daily routine:** one `run-daily` every evening after the close. It stores that session's Trading 212 snapshot and the IBKR statement of the session before, so `real` is complete one day later.

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