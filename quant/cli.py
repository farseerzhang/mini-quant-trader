import argparse
import sys
from datetime import date, timedelta

from quant.config.settings import DEFAULT_CONFIG_PATH, ConfigError, Settings, load_settings
from quant.data.calendar import last_completed_session
from quant.data.store import store_market_data, sync_market_data
from quant.data.yfinance_provider import MarketDataError, YFinanceProvider
from quant.features.pipeline import calculate_and_store_features
from quant.persistence.database import create_db_engine, init_db


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"path to the settings file (default: {DEFAULT_CONFIG_PATH})",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    db_help = "SQLite database path (overrides database.url in the settings file)"

    store_command = commands.add_parser(
        "store-market-data",
        help="fetch a date range and store it locally in SQLite (idempotent); defaults come from the settings file",
    )
    store_command.add_argument(
        "--symbol", help="ticker symbol, e.g. AAPL (default: every symbol in market.symbols and market.market_symbols)"
    )
    store_command.add_argument(
        "--start", type=_iso_date, help="inclusive start date (default: market.initial_history)"
    )
    store_command.add_argument(
        "--end", type=_iso_date, help="exclusive end date (default: the day after the last completed session)"
    )
    store_command.add_argument("--db", help=db_help)

    update_command = commands.add_parser(
        "update-market-data",
        help="incrementally update market.symbols and market.market_symbols through the last completed session",
    )
    update_command.add_argument(
        "--initial-date",
        type=_iso_date,
        help="start date for symbols not yet stored (overrides market.initial_history)",
    )
    update_command.add_argument("--db", help=db_help)

    feature_command = commands.add_parser(
        "calculate-features",
        help="calculate and persist versioned technical and market features",
    )
    feature_command.add_argument("--db", help=db_help)
    return parser


def _provider(settings: Settings) -> YFinanceProvider:
    if settings.market.provider != "yfinance":
        raise ConfigError(f"unsupported market.provider: {settings.market.provider!r}")
    return YFinanceProvider(request_timeout_seconds=settings.market.request_timeout_seconds)


def _engine(settings: Settings, db_override: str | None):
    engine = create_db_engine(
        db_override or settings.database.path, settings.database.busy_timeout_ms
    )
    init_db(engine)
    return engine


def _run_store(args: argparse.Namespace, settings: Settings) -> int:
    symbols = [args.symbol.strip().upper()] if args.symbol else settings.market.universe
    start = args.start or settings.market.initial_history
    end = args.end or last_completed_session(None, settings.market.calendar) + timedelta(days=1)
    if start >= end:
        print(f"Cannot store market data: start {start} must be earlier than end {end}", file=sys.stderr)
        return 1

    engine = _engine(settings, args.db)
    failures = 0
    try:
        provider = _provider(settings)
        for symbol in symbols:
            try:
                summary = store_market_data(
                    engine, provider, symbol, start, end, provider_name=settings.market.provider
                )
            except MarketDataError as exc:
                print(f"{symbol}: FAILED: {exc}", file=sys.stderr)
                failures += 1
                continue
            print(
                f"{summary.symbol}: stored {summary.stored_prices} new price rows "
                f"(fetched {summary.fetched_prices}, quarantined {summary.quarantined}), "
                f"{summary.stored_actions} new corporate actions"
            )
    finally:
        engine.dispose()
    return 1 if failures else 0


def _run_update(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        report = sync_market_data(
            engine,
            _provider(settings),
            settings.market.universe,
            initial_date=args.initial_date or settings.market.initial_history,
            provider_name=settings.market.provider,
            calendar_name=settings.market.calendar,
        )
    except (MarketDataError, ValueError) as exc:
        print(f"Market data update failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print(f"Market data updated through {report.as_of}")
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


def _run_features(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        summary = calculate_and_store_features(engine, settings.features)
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
    try:
        settings = load_settings(args.config)
        if args.command == "store-market-data":
            return _run_store(args, settings)
        if args.command == "update-market-data":
            return _run_update(args, settings)
        if args.command == "calculate-features":
            return _run_features(args, settings)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2