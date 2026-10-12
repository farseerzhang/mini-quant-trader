"""What the dashboard shows, as plain DataFrames (proposal section 25).

Every function only reads the database (the dashboard never writes or triggers anything;
the CLI does) and returns data the app renders as it is, so all of it is unit-tested here
and the app stays a thin layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from quant.backtest.data import load_prices
from quant.config.settings import Settings
from quant.ops.status import StatusReport, collect_status
from quant.persistence import queries, tables
from quant.regime.active import regime_in_use
from quant.regime.pipeline import RULE_METHOD
from quant.signals.positions import MANUAL

MARKET_SYMBOLS = ("SPY", "QQQ", "^VIX")
NOTICE = "Research output, not advice. Signals are computed at the close and meant for the next open."


def status(engine: Engine, settings: Settings) -> StatusReport:
    return collect_status(engine, settings)


@dataclass(frozen=True)
class MarketOverview:
    session: pd.Timestamp | None
    regime: str | None
    regime_source: str
    score: float | None  # rule-based score (-5 .. +5); None for an HMM regime
    quotes: pd.DataFrame  # per market symbol: as-traded close; 1 day, 1 month, 1 year total-return change
    history: pd.DataFrame  # date, close (SPY total return), regime


def _change(series: pd.Series, sessions: int) -> float:
    series = series.dropna()
    return float(series.iloc[-1] / series.iloc[-1 - sessions] - 1) if len(series) > sessions else np.nan


def market_overview(engine: Engine, settings: Settings, years: int = 5) -> MarketOverview:
    in_use = regime_in_use(engine, settings)
    regimes = in_use.regimes
    available = [s for s in MARKET_SYMBOLS if s in queries.latest_price_dates(engine)]
    prices = load_prices(engine, available) if available else None
    traded = queries.load_raw_closes(engine, available) if available else pd.DataFrame()
    rows = {}
    for symbol in available:
        close = prices.close[symbol].dropna()  # total return: the changes include dividends
        rows[symbol] = {"close": float(traded[symbol].dropna().iloc[-1]), "1 day": _change(close, 1),
                        "1 month": _change(close, 21), "1 year": _change(close, 252)}
    quotes = pd.DataFrame.from_dict(rows, orient="index").rename_axis("symbol")
    score = None
    if in_use.method == RULE_METHOD and not regimes.empty:
        stmt = select(tables.market_regimes.c.regime_score).where(
            tables.market_regimes.c.date == regimes.index[-1].date(),
            tables.market_regimes.c.regime_method == RULE_METHOD,
            tables.market_regimes.c.model_version == in_use.model_version)
        with engine.connect() as conn:
            score = conn.execute(stmt).scalar_one_or_none()
    history = pd.DataFrame(columns=["date", "close", "regime"])
    if prices is not None and "SPY" in prices.close:
        spy = prices.close["SPY"].dropna()
        spy = spy[spy.index >= spy.index[-1] - pd.DateOffset(years=years)]
        history = pd.DataFrame({"date": spy.index, "close": spy.to_numpy(),
                                "regime": regimes.reindex(spy.index).to_numpy()})
    return MarketOverview(
        session=None if regimes.empty else regimes.index[-1],
        regime=None if regimes.empty else regimes.iloc[-1],
        regime_source=in_use.model_version, score=score, quotes=quotes, history=history,
    )


def regime_runs(history: pd.DataFrame) -> pd.DataFrame:
    """Consecutive sessions with the same regime: start, end, regime (for the ribbon)."""
    frame = history.dropna(subset=["regime"])
    if frame.empty:
        return pd.DataFrame(columns=["start", "end", "regime"])
    block = (frame["regime"] != frame["regime"].shift()).cumsum()
    runs = frame.groupby(block).agg(start=("date", "first"), end=("date", "last"), regime=("regime", "first"))
    runs["end"] = runs["end"] + pd.Timedelta(days=1)
    return runs.reset_index(drop=True)


def latest_signals(engine: Engine) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """The most recent session's stored signals (every strategy and model version run that day)."""
    session = queries.latest_date(engine, tables.signals.c.date)
    if session is None:
        return None, pd.DataFrame(columns=[c.name for c in tables.signals.columns])
    frame = queries.load_signals(engine, session)
    order = {"EXIT": 0, "REDUCE": 1, "BUY": 2, "HOLD": 3}
    frame = frame.assign(_order=frame["signal"].map(order)).sort_values(
        ["_order", "target_weight", "symbol"], ascending=[True, False, True]).drop(columns="_order")
    return pd.Timestamp(session), frame.reset_index(drop=True)


def latest_positions(engine: Engine) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    session = queries.latest_date(engine, tables.positions.c.as_of_date, tables.positions.c.source == MANUAL)
    if session is None:
        return None, pd.DataFrame(columns=[c.name for c in tables.positions.columns])
    return pd.Timestamp(session), queries.load_positions(engine, session, MANUAL)


def stock_detail(engine: Engine, settings: Settings, symbol: str, years: int = 2) -> pd.DataFrame:
    """Per session: adjusted close with SMA50 / SMA200, RSI14, MACD histogram and volume.

    The averages are recomputed from the total-return close like the stored features (the
    features table keeps them as ratios), so the lines sit on the price.
    """
    prices = load_prices(engine, [symbol])
    close = prices.close[symbol].dropna()
    panel = queries.load_feature_panel(engine, settings.features.feature_version, [symbol],
                                       ["rsi14", "macd_hist_over_price"])
    with engine.connect() as conn:
        volume = pd.Series(dict(conn.execute(
            select(tables.daily_prices.c.date, tables.daily_prices.c.volume)
            .where(tables.daily_prices.c.symbol == symbol)).all()), dtype=float)
    volume.index = pd.DatetimeIndex(volume.index)
    frame = pd.DataFrame({
        "close": close,
        "sma50": close.rolling(settings.features.sma[1], min_periods=settings.features.sma[1]).mean(),
        "sma200": close.rolling(settings.features.sma[2], min_periods=settings.features.sma[2]).mean(),
        "rsi14": panel["rsi14"][symbol].reindex(close.index),
        "macd_hist": (panel["macd_hist_over_price"][symbol] * close).reindex(close.index),
        "volume": volume.reindex(close.index),
    })
    frame = frame[frame.index >= frame.index[-1] - pd.DateOffset(years=years)]
    return frame.rename_axis("date").reset_index()


def models(engine: Engine, reports_dir: str | Path) -> pd.DataFrame:
    """Every registered model with its gates and where its report is."""
    columns = ["model_id", "role", "model_type", "version", "status", "gates_passed", "training_start",
               "training_end", "created_at", "validation_report_path", "gate_results"]
    with engine.connect() as conn:
        rows = conn.execute(select(*[tables.model_versions.c[c] for c in columns])
                            .order_by(tables.model_versions.c.created_at.desc())).mappings().all()
    frame = pd.DataFrame([dict(r) for r in rows], columns=columns)

    def failed(text):
        try:
            results = json.loads(text or "{}")
        except json.JSONDecodeError:
            return ""
        return ", ".join(name for name, r in results.items() if r.get("status") == "FAIL")

    frame["failed_gates"] = frame["gate_results"].map(failed)
    frame["report_exists"] = frame["validation_report_path"].map(lambda p: bool(p) and Path(p).exists())
    return frame.drop(columns="gate_results")


def gate_table(engine: Engine, model_id: str) -> pd.DataFrame:
    with engine.connect() as conn:
        text = conn.execute(select(tables.model_versions.c.gate_results)
                            .where(tables.model_versions.c.model_id == model_id)).scalar_one_or_none()
    results = json.loads(text or "{}")
    return pd.DataFrame.from_dict(results, orient="index").rename_axis("gate").reset_index()


def backtest_batches(engine: Engine) -> pd.DataFrame:
    runs = tables.backtest_runs
    with engine.connect() as conn:
        rows = conn.execute(select(runs.c.batch_id, runs.c.created_at, runs.c.strategy, runs.c.universe)
                            .where(runs.c.period == "DEVELOPMENT").order_by(runs.c.run_id)).all()
    frame = pd.DataFrame(rows, columns=["batch_id", "created_at", "strategy", "universe"])
    if frame.empty:
        return pd.DataFrame(columns=["batch_id", "created_at", "portfolios", "stocks"])

    def stocks(universe):
        try:
            return len(json.loads(universe))
        except (TypeError, json.JSONDecodeError):
            return np.nan

    out = frame.groupby("batch_id", sort=False).agg(
        created_at=("created_at", "min"), portfolios=("strategy", "count"), stocks=("universe", "first"))
    out["stocks"] = out["stocks"].map(stocks)
    return out.sort_values("created_at", ascending=False).reset_index()


METRICS = ["strategy", "kind", "cagr", "sharpe", "sortino", "max_drawdown", "volatility", "avg_invested",
           "turnover", "num_trades", "win_rate", "random_sharpe_median", "sharpe_percentile"]


def backtest_metrics(engine: Engine, batch_id: str, period: str = "DEVELOPMENT") -> pd.DataFrame:
    runs = tables.backtest_runs
    with engine.connect() as conn:
        rows = conn.execute(select(runs.c.run_id, *[runs.c[c] for c in METRICS])
                            .where(runs.c.batch_id == batch_id, runs.c.period == period)
                            .order_by(runs.c.run_id)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows], columns=["run_id", *METRICS])


def backtest_equity(engine: Engine, batch_id: str, period: str = "DEVELOPMENT") -> pd.DataFrame:
    """Long frame: date, strategy, value (indexed to 1 at the start), drawdown."""
    metrics = backtest_metrics(engine, batch_id, period)
    if metrics.empty:
        return pd.DataFrame(columns=["date", "strategy", "value", "drawdown"])
    eq = tables.backtest_equity
    with engine.connect() as conn:
        rows = conn.execute(select(eq.c.run_id, eq.c.date, eq.c.value)
                            .where(eq.c.run_id.in_(metrics["run_id"].tolist()))).all()
    frame = pd.DataFrame(rows, columns=["run_id", "date", "value"])
    frame["date"] = pd.DatetimeIndex(frame["date"])
    frame = frame.merge(metrics[["run_id", "strategy"]], on="run_id").sort_values(["strategy", "date"])
    first = frame.groupby("strategy")["value"].transform("first")
    peak = frame.groupby("strategy")["value"].cummax()
    frame["drawdown"] = frame["value"] / peak - 1
    frame["value"] = frame["value"] / first
    return frame[["date", "strategy", "value", "drawdown"]].reset_index(drop=True)
