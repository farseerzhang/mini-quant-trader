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

**Phase 2 - Features** (In Progress)

The project is currently in early development. Implemented features include:
- SQLite database with WAL mode for concurrent access
- Market data provider abstraction (YFinance)
- Incremental market data downloader
- Data validation and quarantine
- Corporate actions tracking
- CLI interface for data import and storage
- Split-adjusted OHLCV and dividend-aware total-return price series
- Versioned technical and market feature calculation

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

The application provides a CLI for market data operations:

#### Import Market Data (Preview)

Fetch daily market data from Yahoo Finance without saving it locally:

```bash
python -m quant fetch-market-data --symbol AAPL --start 2026-09-25 --end 2026-10-02
```

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



#### Store Market Data

Fetch a date range and store it locally in SQLite (idempotent). With no arguments it stores every symbol in `market.symbols` and `market.market_symbols`, from `market.initial_history` through the last completed session:

```bash
python -m quant store-market-data
python -m quant store-market-data --symbol AAPL --start 2020-01-01 --end 2026-10-02
```

Unlike `update-market-data`, this always re-requests the whole range (nothing is duplicated or overwritten). Use `update-market-data` for routine incremental refreshes.

Options (all optional):
- `--symbol`: One ticker (default: the whole pool)
- `--start`: Inclusive start date, YYYY-MM-DD (default: `market.initial_history`)
- `--end`: Exclusive end date, YYYY-MM-DD (default: the day after the last completed session)
- `--db`: Path to SQLite database (default: `database.url` in `config/settings.yaml`)

### Configuration

All settings live in one file, `config/settings.yaml`, laid out as in section 27 of `proposal.md`: database, market (provider, symbols, calendar, initial history), features, regime, model, retrain, backtest and risk. Use another file with `python -m quant --config path\to\settings.yaml <command>`. Every section and key is required, and unknown or invalid settings are rejected at startup; there are no defaults in code, so the file is the only place a value is defined. The `model`, `retrain`, `backtest` and `risk` sections are loaded but not used until those phases are built.

`features.sma` must match the `price_vs_sma<N>` columns of the `features` table (checked when features are calculated), because those columns are named after the windows. Other indicator parameters (MACD 12/26/9, return and volatility windows) are part of the feature definition, so changing them means bumping `features.version`.

### Database

The application uses SQLite with WAL mode for local data storage. The database file is set by `database.url` (default `data/quant.db`).

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
