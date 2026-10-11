import argparse
import sys
from datetime import date

from quant.backtest.engine import run_backtests
from quant.backtest.persist import save_backtest
from quant.backtest.report import development_report, holdout_report, metrics_table
from quant.config.settings import DEFAULT_CONFIG_PATH, ConfigError, Settings, load_settings
from quant.data.store import sync_market_data, sync_symbol_table
from quant.data.yfinance_provider import MarketDataError, YFinanceProvider
from quant.features.pipeline import calculate_and_store_features
from quant.models.ml_backtest import BASELINE
from quant.models.retrain import run_retrain
from quant.persistence import queries
from quant.persistence.database import create_db_engine, init_db
from quant.regime.pipeline import calculate_and_store_regimes
from quant.strategies import build_strategies


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

    backtest_command = commands.add_parser(
        "backtest",
        help="backtest the rule-based strategies against SPY, equal weight and a random baseline, and save the results",
    )
    backtest_command.add_argument(
        "--strategy",
        action="append",
        metavar="NAME",
        help="run only this strategy (repeatable; default: all). Benchmarks always run",
    )
    backtest_command.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="also show and save the holdout period; every evaluation is logged (section 15.2)",
    )
    backtest_command.add_argument("--db", help=db_help)

    retrain_command = commands.add_parser(
        "retrain",
        help="MANUAL: search, validate walk-forward, evaluate the gates and register a CANDIDATE "
        "(never changes an active model)",
    )
    retrain_command.add_argument("--role", choices=["predictor", "regime"], required=True)
    retrain_command.add_argument(
        "--model", help="model type to train (default: model.type in the settings file)"
    )
    retrain_command.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="also trade the holdout and check its gate; every evaluation is logged (section 15.2)",
    )
    retrain_command.add_argument("--db", help=db_help)

    daily_command = commands.add_parser(
        "run-daily",
        help="update market data, then calculate features and the regime (stops at the first failed stage)",
    )
    daily_command.add_argument(
        "--initial-date",
        type=_iso_date,
        help="start date for symbols not yet stored (overrides market.initial_history)",
    )
    daily_command.add_argument("--db", help=db_help)
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


# Outcome of the market-data stage: some symbols failing still leaves usable data.
_OK, _PARTIAL, _FAILED = "ok", "partial", "failed"


def _run_update(args: argparse.Namespace, settings: Settings) -> int:
    return 0 if _update_market_data(args, settings) == _OK else 1


def _update_market_data(args: argparse.Namespace, settings: Settings) -> str:
    engine = _engine(settings, args.db)
    try:
        sync_symbol_table(engine, settings.market.sectors, settings.market.market_symbols)
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
        return _FAILED
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
    return _PARTIAL if report.failures else _OK


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


def _run_backtest(args: argparse.Namespace, settings: Settings) -> int:
    strategies = build_strategies(settings)
    if args.strategy:
        known = {strategy.name for strategy in strategies}
        unknown = [name for name in args.strategy if name not in known]
        if unknown:
            print(
                f"Unknown strategy: {', '.join(unknown)}; choose from {', '.join(sorted(known))}",
                file=sys.stderr,
            )
            return 1
        strategies = [strategy for strategy in strategies if strategy.name in args.strategy]
    show_holdout = args.evaluate_holdout or not settings.holdout_requires_flag

    engine = _engine(settings, args.db)
    try:
        report = run_backtests(engine, settings, strategies)
        previous_evaluations = queries.count_holdout_evaluations(engine)
        batch_id = save_backtest(engine, settings, report, include_holdout=show_holdout)
        periods = report.periods
        names = ", ".join(result.name for result in report.results)
        if show_holdout:
            queries.record_pipeline_run(
                engine,
                command="backtest",
                stage="holdout",
                status="OK",
                data_as_of_date=periods.end.date(),
                message=f"batch {batch_id}: holdout {periods.holdout_start.date()}..{periods.end.date()} "
                f"evaluated for {names}",
            )
        queries.record_pipeline_run(
            engine,
            command="backtest",
            stage="backtest",
            status="OK",
            data_as_of_date=periods.end.date(),
            message=f"batch {batch_id}: {names}",
        )
    except ValueError as exc:
        print(f"Backtest failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    print(development_report(report))
    print()
    if show_holdout:
        print(holdout_report(report, previous_evaluations + 1))
    else:
        print(
            f"HOLDOUT {periods.holdout_start.date()} .. {periods.end.date()}: not shown "
            f"(evaluated {previous_evaluations} time(s) so far; use --evaluate-holdout to view it)"
        )
    print()
    print(f"Saved as batch {batch_id} in backtest_runs and backtest_equity.")
    return 0


def _print_retrain(outcome) -> None:
    folds, best = outcome.folds, outcome.search
    name = f"ml_{outcome.model_type}"
    print(
        f"{outcome.model_type}: {len(folds)} walk-forward folds validating "
        f"{folds[0].validation_start.date()} .. {folds[-1].validation_end.date()}; "
        f"holdout from {outcome.holdout_start.date()}"
    )
    print(f"\nSearch ({best.trial_count} trials; best {best.best_parameters}):")
    trials = best.trials
    params = [c for c in trials.columns if c.startswith("param_")]
    view = trials[params].rename(columns=lambda c: c[len("param_"):])
    view["score"] = trials["score"].map(lambda v: f"{v:+.3%}")
    view["log-loss skill"] = trials["log_loss_skill_mean"].map(lambda v: f"{v:+.2%}")
    view["AUC"] = trials["auc_mean"].map(lambda v: f"{v:.3f}")
    print(view.to_string())

    print(f"\nWalk-forward backtest {outcome.backtest.start.date()} .. {outcome.backtest.end.date()} (after costs):")
    print(metrics_table(outcome.backtest.overall))
    if outcome.holdout is not None:
        print(
            f"\nHOLDOUT {outcome.holdout.start.date()} .. {outcome.holdout.end.date()} "
            f"(evaluation #{outcome.holdout_evaluation_number}; logged in pipeline_runs):"
        )
        print(metrics_table(outcome.holdout.overall))
        print("If this result changes a rule or setting, the holdout is consumed (section 15.2).")

    print(f"\nGates: {'PASSED' if outcome.gates.passed else 'FAILED'}")
    print(outcome.gates.table().to_string())
    registration = outcome.registration
    print(
        f"\nRegistered {registration.model_id} (version {registration.version}) as CANDIDATE."
        f"\n  model:  {registration.model_path}\n  report: {registration.report_path}"
    )
    active = outcome.active
    if active is None:
        print("No ACTIVE predictor: run-daily uses rule-based signals only. Nothing was changed.")
    else:
        print(
            f"ACTIVE predictor stays {active['model_id']} ({active['model_type']} v{active['version']}, "
            f"gates {'passed' if active['gates_passed'] else 'failed'}). Nothing was changed; "
            "promoting a candidate is `quant promote` (Phase 9)."
        )
    if not outcome.gates.passed:
        print(f"The candidate failed its gates; under section 32 the rule-based {BASELINE} remains what ships.")


def _run_retrain(args: argparse.Namespace, settings: Settings) -> int:
    if args.role == "regime":
        print("retrain --role regime is not available yet: the REGIME role (HMM) arrives in Phase 7", file=sys.stderr)
        return 1
    model_type = args.model or settings.model.type
    evaluate_holdout = args.evaluate_holdout or not settings.holdout_requires_flag
    engine = _engine(settings, args.db)
    try:
        outcome = run_retrain(engine, settings, model_type, evaluate_holdout)
    except (NotImplementedError, ValueError) as exc:
        print(f"Retrain failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    _print_retrain(outcome)
    return 0


def _run_daily(args: argparse.Namespace, settings: Settings) -> int:
    """Run the pipeline stages in order (proposal section 23, steps 1-6 so far).

    Every stage persists its own results and pipeline_runs row, so a failure
    leaves earlier stages intact and a rerun resumes safely. A stage that fails
    outright stops the run, because later stages read its output. Symbols that
    fail to download do not stop it: features and the regime are calculated
    from the data that is stored, and the exit code still reports the failure.
    """
    print("[1/3] update-market-data")
    market_data = _update_market_data(args, settings)
    if market_data == _FAILED:
        print("run-daily stopped: market data update failed", file=sys.stderr)
        return 1
    for step, name, run in (
        (2, "calculate-features", _run_features),
        (3, "calculate-regime", _run_regime),
    ):
        print(f"[{step}/3] {name}")
        if run(args, settings) != 0:
            print(f"run-daily stopped: {name} failed", file=sys.stderr)
            return 1
    if market_data == _PARTIAL:
        print("run-daily finished, but some symbols failed to update (see above)", file=sys.stderr)
        return 1
    print("run-daily finished")
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
        if args.command == "run-daily":
            return _run_daily(args, settings)
        if args.command == "backtest":
            return _run_backtest(args, settings)
        if args.command == "retrain":
            return _run_retrain(args, settings)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2