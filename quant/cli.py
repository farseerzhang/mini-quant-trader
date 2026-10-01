import argparse
import sys
from datetime import date

from quant.data.yfinance_provider import MarketDataError, YFinanceProvider


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant")
    commands = parser.add_subparsers(dest="command", required=True)
    import_command = commands.add_parser(
        "import-market-data",
        help="fetch daily market data from Yahoo Finance without saving it locally",
    )
    import_command.add_argument("--symbol", required=True, help="ticker symbol, e.g. AAPL")
    import_command.add_argument("--start", required=True, type=_iso_date, help="inclusive start date")
    import_command.add_argument("--end", required=True, type=_iso_date, help="exclusive end date")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start >= args.end:
        parser.error("--start must be earlier than --end")

    try:
        prices = YFinanceProvider().get_daily_data(args.symbol, args.start, args.end)
    except MarketDataError as exc:
        print(f"Market data request failed: {exc}", file=sys.stderr)
        return 1

    print(f"Received {len(prices)} daily bars for {args.symbol.upper()} from Yahoo Finance:")
    print(prices.to_string())
    return 0
