# Mini Quant Trader

A local quantitative trading research and signal-generation application for US equities.

## Overview

Mini Quant Trader is a personal research system designed to:
- Import market data from external APIs (currently Yahoo Finance)
- Store historical market data locally in SQLite
- Calculate technical and market features
- Detect market regimes (Bull, Sideways, Bear)
- Generate stock/ETF predictions using machine learning models
- Generate interpretable target exposures and BUY/HOLD/REDUCE signals
- Backtest strategies against explicit benchmarks
- Provide a Streamlit dashboard for monitoring

**Important**: This is a research and decision-support system. Model predictions are not reliable trading signals and should not be treated as recommendations to trade.

## Current Status

**Phase 7 - HMM regime** (Done: no ML predictor and no HMM regime passes its gates, so the rule-based system ships; next: Phase 8 - signal engine and risk)

The project is currently in early development. Implemented features include:
- SQLite database with WAL mode for concurrent access
- Market data provider abstraction (YFinance)
- Incremental market data downloader
- Data validation and quarantine
- Corporate actions tracking
- CLI interface for data import and storage
- Split-adjusted OHLCV and dividend-aware total-return price series
- Versioned technical and market feature calculation
- Rule-based BULL/SIDEWAYS/BEAR market regime with hysteresis and minimum dwell time
- Rule-based trend, mean-reversion and regime-switch strategies with position and sector caps
- VectorBT backtests against SPY, equal weight and a random baseline, saved per run
- A filtered, walk-forward HMM market regime as a REGIME candidate (`quant retrain --role regime`)
- Logistic Regression, Random Forest and XGBoost predictors with purged walk-forward validation, calibration, pre-defined gates and a model registry (`quant retrain`)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mini-quant-trader
```

2. Install dependencies:
```bash
pip install -e .
```

This will install the required packages:
- pandas >= 2.0
- numpy >= 1.24
- yfinance >= 0.2.54
- SQLAlchemy >= 2.0
- exchange-calendars >= 4.5
- PyYAML >= 6.0

## Usage

### Command Line Interface

The application provides a CLI for market data operations.

#### Daily Pipeline

Run every stage in one step:

```bash
python -m quant run-daily
```

This runs `update-market-data`, `calculate-features` and `calculate-regime` in that order (they are described below). A stage that fails stops the run, because the next stage reads its output. Symbols that fail to download do not stop it: features and the regime are calculated from the stored data, and the command still exits with code 1. Every stage is safe to rerun. It accepts `--initial-date` and `--db` like `update-market-data`.

#### Update the Configured Symbol Pool

Fetch missing daily bars for `market.symbols` and `market.market_symbols` (from `config/settings.yaml`) through the last completed NYSE session:

```powershell
python -m quant update-market-data
```

New symbols start from `market.initial_history` (2005-01-01 by default; override once with `--initial-date YYYY-MM-DD`); existing symbols only fetch sessions after their last stored bar. Edit `market.symbols` in `config/settings.yaml` to change the pool.

#### Calculate Features

After updating the configured symbol pool, calculate and persist features from the stored prices:

```bash
python -m quant calculate-features
```

This recomputes feature history (including adjustment changes from stored corporate actions) and updates the `features` and `market_features` tables in `data/quant.db`. Warm-up rows remain NULL. SPY data is required for relative returns and market features; QQQ and ^VIX provide their respective optional market indicators.

#### Calculate the Market Regime

After calculating features, classify every session as BULL, SIDEWAYS or BEAR:

```bash
python -m quant calculate-regime
```

The score at each session sums five -1/0/+1 inputs (SPY trend, QQQ trend, SPY SMA50 vs SMA200, SPY 3-month return, VIX level), and a state machine with hysteresis (`regime.enter_threshold` / `regime.exit_threshold`) and a minimum dwell time (`regime.min_dwell_days`) turns scores into regimes; the full rules are in section 12 of `proposal.md`. Sessions missing any input get no regime. The whole history is recomputed and replaces the stored rows of the `market_regimes` table for `rule_based-v<regime.version>`; bump `regime.version` when changing any `regime` setting.

#### Backtest the Strategies

After `run-daily`, compare the rule-based strategies with SPY buy-and-hold, an equal-weight portfolio of the pool and a random baseline:

```bash
python -m quant backtest
python -m quant backtest --strategy trend --strategy regime_switch
python -m quant backtest --evaluate-holdout
```

Orders are decided at the close and filled at the next open, with fees, slippage and a minimum trade size from the `backtest` settings; the strategies are described in section 17 of `proposal.md` and the engine in section 20.1. The report shows the development period. The holdout (the last `retrain.validation.holdout_months`) is shown only with `--evaluate-holdout`; every evaluation is logged in `pipeline_runs`, and the report says how many times the holdout has been viewed. Each run is saved as a batch in `backtest_runs` (metrics, costs, parameters, config hash, git commit) and `backtest_equity` (daily value), with holdout rows only when it was evaluated. A run takes about 40 seconds, mostly the random baseline.

#### Retrain a Predictor (manual)

Train and validate a machine-learning stock predictor, and register it as a candidate:

```bash
python -m quant retrain --role predictor --model logistic_regression
python -m quant retrain --role predictor --model random_forest
python -m quant retrain --role predictor --model xgboost
python -m quant retrain --role predictor --model logistic_regression --evaluate-holdout
```

The market regime has its own candidate:

```bash
python -m quant retrain --role regime
```

It fits a 3-state hidden Markov model on daily SPY returns, market volatility and VIX (`regime.hmm`) once a year on the same walk-forward folds, uses filtered (point-in-time) probabilities only, trades `regime_switch` with it and with the rule-based regime, evaluates `retrain.gates.regime`, and registers the final fit as a REGIME candidate (about a minute). The rule-based regime stays in use unless a candidate is promoted.

This builds the dataset (label: stock vs SPY over the next `model.target.horizon_days` sessions, section 14 of `proposal.md`), searches the hyperparameters in `retrain.search.space` on purged walk-forward folds, backtests the best candidate walk-forward as `ml_select` against `regime_switch` after costs, evaluates the gates in `retrain.gates.predictor`, refits the model on all development data and registers it as a CANDIDATE in `model_versions`. The model file goes to `retrain.models_dir` and a readable validation report to `retrain.reports_dir` (both git-ignored). It never changes an active model, and the holdout is used only with `--evaluate-holdout` (logged; the count is shared with `backtest`). Logistic Regression takes about 20 seconds, Random Forest about 3 minutes, XGBoost longer (30 random search trials).

Every command accepts `--db PATH` to use another SQLite file instead of `database.url` from `config/settings.yaml`.

### Configuration

All settings live in one file, `config/settings.yaml`, laid out as in section 27 of `proposal.md`: database, market (provider, symbols, calendar, initial history), features, regime, model, retrain, backtest and risk. Use another file with `python -m quant --config path\to\settings.yaml <command>`. Every section and key is required, and unknown or invalid settings are rejected at startup; there are no defaults in code, so the file is the only place a value is defined. `market.symbols` maps each GICS sector to its stocks (e.g. `energy: [XOM, CVX, SLB]`); sectors feed the sector cap `risk.max_sector_weight`, and `update-market-data` writes them into the `symbols` table. The `strategies` and `risk` sections drive the rule-based strategies (section 17 of `proposal.md`); `model`, `retrain` and `backtest` are loaded but not used until those phases are built.

`features.sma` must match the `price_vs_sma<N>` columns of the `features` table (checked when features are calculated), because those columns are named after the windows. Other indicator parameters (MACD 12/26/9, return and volatility windows) are part of the feature definition, so changing them means bumping `features.version`.

### Database

The application uses SQLite with WAL mode for local data storage. The database file is set by `database.url` (default `data/quant.db`).

`daily_prices` holds prices, volumes and dividends as originally traded. Yahoo returns them rescaled for later splits, and `YFinanceProvider` reverses that (see section 8 of `proposal.md`). Databases built before this fix hold Yahoo's split-rescaled values, and stored rows are never overwritten, so rebuild such a database: move `data/quant.db` aside, then run `python -m quant update-market-data` and `python -m quant calculate-features`.

## Project Structure

```
mini-quant-trader/

```

## Architecture Principles

- **Python-first**: The complete system uses Python
- **SQLite as a local file**: Sufficient for single-user, local, daily data
- **Raw data is the source of truth**: Derived features are computed from raw data
- **Incremental data ingestion**: Only downloads missing data, not full history
- **Time and execution convention**: Signals at T execute at T+1 open

## Planned Features

See [quant_app_developer_proposal_final.md](proposal.md) for the complete design specification.

Future phases will include:
- Feature engineering (SMA, RSI, MACD, ATR, momentum, volatility)
- Market regime detection (rule-based and HMM)
- ML models (Logistic Regression, Random Forest, XGBoost)
- Strategy engine and risk management
- Signal generation (BUY/HOLD/REDUCE/EXIT)
- Backtesting with VectorBT
- Streamlit dashboard
- Manual retriggering and model promotion

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows standard Python conventions. Contributions should maintain the existing code structure and style.

## Data Provider

The current implementation uses `yfinance` as the market data provider. This is an unofficial scraper that can break, rate-limit, and revise data. The provider is isolated behind an abstraction interface to allow future replacement with professional data sources (Polygon, Tiingo, IBKR, etc.).

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines here]
