import argparse
import sys
from datetime import date
from pathlib import Path

from quant.data.store import store_market_data, sync_market_data
from quant.data.yfinance_provider import MarketDataError, YFinanceProvider
from quant.features.pipeline import calculate_and_store_features
from quant.persistence.database import DEFAULT_DB_PATH, create_db_engine, init_db

DEFAULT_UNIVERSE_PATH = Path("config") / "symbols.txt"
DEFAULT_INITIAL_DATE = date(2015, 1, 1)

def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_command = commands.add_parser(
        "fetch-market-data",
        help="fetch daily market data from Yahoo Finance without saving it locally",
    )
    fetch_command.add_argument("--symbol", required=True, help="ticker symbol, e.g. AAPL")
    fetch_command.add_argument("--start", required=True, type=_iso_date, help="inclusive start date")
    fetch_command.add_argument("--end", required=True, type=_iso_date, help="exclusive end date")

    store_command = commands.add_parser(
        "store-market-data",
        help="fetch daily market data and store it locally in SQLite (idempotent)",
    )
    store_command.add_argument("--symbol", required=True, help="ticker symbol, e.g. AAPL")
    store_command.add_argument("--start", required=True, type=_iso_date, help="inclusive start date")
    store_command.add_argument("--end", required=True, type=_iso_date, help="exclusive end date")
    store_command.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")

    update_command = commands.add_parser(
        "update-market-data",
        help="incrementally update the configured stock pool through the last completed NYSE session",
    )
    update_command.add_argument("--symbols-file", default=str(DEFAULT_UNIVERSE_PATH))
    update_command.add_argument("--initial-date", type=_iso_date, default=DEFAULT_INITIAL_DATE)
    update_command.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")

    feature_command = commands.add_parser(
        "calculate-features",
        help="calculate and persist versioned technical and market features",
    )
    feature_command.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    return parser


def _run_fetch(args: argparse.Namespace) -> int:
    try:
        prices = YFinanceProvider().get_daily_data(args.symbol, args.start, args.end)
    except MarketDataError as exc:
        print(f"Market data request failed: {exc}", file=sys.stderr)
        return 1
    print(f"Received {len(prices)} daily bars for {args.symbol.upper()} from Yahoo Finance:")
    print(prices.to_string())
    return 0


def _run_store(args: argparse.Namespace) -> int:
    engine = create_db_engine(args.db)
    init_db(engine)
    try:
        summary = store_market_data(engine, YFinanceProvider(), args.symbol, args.start, args.end)
    except MarketDataError as exc:
        print(f"Market data request failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print(
        f"Stored {summary.symbol} into {args.db}: {summary.stored_prices} new price rows "
        f"(fetched {summary.fetched_prices}, quarantined {summary.quarantined}), "
        f"{summary.stored_actions} new corporate actions."
    )
    return 0


def _read_symbols(path: str) -> list[str]:
    symbols: list[str] = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            line = line.split("#", 1)[0]
            symbols.extend(part.strip().upper() for part in line.replace(",", " ").split())
    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    return list(dict.fromkeys(symbols))


def _run_update(args: argparse.Namespace) -> int:
    try:
        symbols = _read_symbols(args.symbols_file)
    except (OSError, ValueError) as exc:
        print(f"Cannot read symbol pool: {exc}", file=sys.stderr)
        return 1
    engine = create_db_engine(args.db)
    init_db(engine)
    try:
        report = sync_market_data(engine, YFinanceProvider(), symbols, initial_date=args.initial_date)
    except (MarketDataError, ValueError) as exc:
        print(f"Market data update failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print(f"Market data updated through {report.as_of} ({args.db})")
    for summary in report.summaries:
        print(
            f"{summary.symbol}: fetched {summary.fetched_prices}, stored {summary.stored_prices}, "
            f"quarantined {summary.quarantined}; {summary.stored_actions} new corporate actions"
        )
    for symbol in report.skipped:
        print(f"{symbol}: already current")
    for failure in report.failures:
        print(f"{failure.symbol}: FAILED: {failure.error}", file=sys.stderr)
    return 1 if report.failures else 0


def _run_features(args: argparse.Namespace) -> int:
    engine = create_db_engine(args.db)
    init_db(engine)
    try:
        summary = calculate_and_store_features(engine)
    except ValueError as exc:
        print(f"Feature calculation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print(
        f"Calculated features through {summary.data_as_of} for {summary.symbols} symbols "
        f"({summary.symbol_rows} symbol rows, {summary.market_rows} market rows; "
        f"versions {summary.feature_version}/{summary.market_feature_version})."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ("fetch-market-data", "store-market-data") and args.start >= args.end:
        parser.error("--start must be earlier than --end")
    if args.command == "fetch-market-data":
        return _run_fetch(args)
    if args.command == "store-market-data":
        return _run_store(args)
    if args.command == "update-market-data":
        return _run_update(args)
    if args.command == "calculate-features":
        return _run_features(args)
    parser.error(f"unknown command: {args.command}")
    return 2
