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

## Usage

### Command Line Interface

The application provides a CLI for market data operations:

#### Import Market Data (Preview)

Fetch daily market data from Yahoo Finance without saving it locally:

```bash
python -m quant fetch-market-data --symbol AAPL --start 2026-09-25 --end 2026-10-02
```

#### Update the Configured Symbol Pool

Fetch missing daily bars for `config/symbols.txt` through the last completed NYSE session:

```powershell
python -m quant update-market-data
```

New symbols start from 2015-01-01 by default; existing symbols only fetch sessions after their last stored bar. Edit `config/symbols.txt` to change the pool (one ticker per line).

#### Calculate Features

After updating the configured symbol pool, calculate and persist features from the stored prices:

```bash
python -m quant calculate-features
```

This recomputes feature history (including adjustment changes from stored corporate actions) and updates the `features` and `market_features` tables in `data/quant.db`. Warm-up rows remain NULL. SPY data is required for relative returns and market features; QQQ and ^VIX provide their respective optional market indicators.



#### Store Market Data

Fetch daily market data and store it locally in SQLite (idempotent):

```bash
python -m quant store-market-data --symbol AAPL --start 2020-01-01 --end 2026-10-02
```

Options:
- `--symbol`: Ticker symbol (e.g., AAPL)
- `--start`: Inclusive start date (YYYY-MM-DD format)
- `--end`: Exclusive end date (YYYY-MM-DD format)
- `--db`: Path to SQLite database (default: `data/quant.db`)

### Database

The application uses SQLite with WAL mode for local data storage. The database file is located at `data/quant.db` by default.

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
