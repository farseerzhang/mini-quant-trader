import argparse
import sys
from datetime import date

from quant.config.settings import DEFAULT_CONFIG_PATH, ConfigError, Settings, load_settings
from quant.data.store import sync_market_data
from quant.data.yfinance_provider import MarketDataError, YFinanceProvider
from quant.features.pipeline import calculate_and_store_features
from quant.persistence.database import create_db_engine, init_db
from quant.regime.pipeline import calculate_and_store_regimes


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

    regime_command = commands.add_parser(
        "calculate-regime",
        help="calculate and persist the rule-based BULL/SIDEWAYS/BEAR regime from stored features",
    )
    regime_command.add_argument("--db", help=db_help)
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


def _run_regime(args: argparse.Namespace, settings: Settings) -> int:
    engine = _engine(settings, args.db)
    try:
        summary = calculate_and_store_regimes(engine, settings.regime, settings.features)
    except ValueError as exc:
        print(f"Regime calculation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    counts = ", ".join(f"{name} {count}" for name, count in summary.sessions_by_regime.items())
    print(
        f"Calculated {summary.rows} {summary.model_version} regimes "
        f"from {summary.first_date} through {summary.last_date}."
    )
    print(f"Latest: {summary.latest_regime} (score {summary.latest_score:+d})")
    print(f"Sessions: {counts}; average dwell {summary.average_dwell_days:.1f} sessions")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command == "update-market-data":
            return _run_update(args, settings)
        if args.command == "calculate-features":
            return _run_features(args, settings)
        if args.command == "calculate-regime":
            return _run_regime(args, settings)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2