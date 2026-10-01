import argparse
import sys
from datetime import date

from quant.data.store import store_market_data
from quant.data.yfinance_provider import MarketDataError, YFinanceProvider
from quant.persistence.database import DEFAULT_DB_PATH, create_db_engine, init_db


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant")
    commands = parser.add_subparsers(dest="command", required=True)

    # python -m quant import-market-data --symbol AAPL --start 2026-09-25 --end 2026-10-02
    import_command = commands.add_parser(
        "import-market-data",
        help="fetch daily market data from Yahoo Finance without saving it locally",
    )
    import_command.add_argument("--symbol", required=True, help="ticker symbol, e.g. AAPL")
    import_command.add_argument("--start", required=True, type=_iso_date, help="inclusive start date")
    import_command.add_argument("--end", required=True, type=_iso_date, help="exclusive end date")

    # python -m quant store-market-data --symbol AAPL --start 2020-09-25 --end 2026-10-02
    store_command = commands.add_parser(
        "store-market-data",
        help="fetch daily market data and store it locally in SQLite (idempotent)",
    )
    store_command.add_argument("--symbol", required=True, help="ticker symbol, e.g. AAPL")
    store_command.add_argument("--start", required=True, type=_iso_date, help="inclusive start date")
    store_command.add_argument("--end", required=True, type=_iso_date, help="exclusive end date")
    store_command.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"path to the SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    return parser


def _run_import(args: argparse.Namespace) -> int:
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
        summary = store_market_data(
            engine, YFinanceProvider(), args.symbol, args.start, args.end
        )
    except MarketDataError as exc:
        print(f"Market data request failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    print(
        f"Stored {summary.symbol} into {args.db}: "
        f"{summary.stored_prices} new price rows "
        f"(fetched {summary.fetched_prices}, quarantined {summary.quarantined}), "
        f"{summary.stored_actions} new corporate actions "
        f"(fetched {summary.fetched_actions})."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start >= args.end:
        parser.error("--start must be earlier than --end")

    if args.command == "import-market-data":
        return _run_import(args)
    if args.command == "store-market-data":
        return _run_store(args)
    parser.error(f"unknown command: {args.command}")
    return 2
