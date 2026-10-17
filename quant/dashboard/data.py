"""What the dashboard shows, as plain DataFrames (proposal section 25).

Every function only reads the database (the dashboard never writes or triggers anything;
the CLI does) and returns data the app renders as it is, so all of it is unit-tested here
and the app stays a thin layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, inspect, select

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


@dataclass(frozen=True)
class RegimeChoice:
    label: str  # shown in the chooser and on the ribbon
    regimes: pd.Series  # regime per session
    note: str  # how it is computed, and from when it is out of sample


def regime_choices(engine: Engine, settings: Settings) -> list[RegimeChoice]:
    """The regime in use plus the regime of every forward-test shadow that has its own (read-only).

    The HMM is its saved final fit run forward over the stored observations (filtered, the same
    dwell as everywhere): what the shadow sees. That fit was trained up to its training end, so
    its regime before that date is in sample. Rule-based variants have nothing fitted.
    """
    from quant.forward.freeze import _model_row
    from quant.regime.active import active_hmm_regime, load_active_hmm
    from quant.regime.hmm import load_observations

    in_use = regime_in_use(engine, settings)
    choices = [RegimeChoice(f"{in_use.model_version} (in use)", in_use.regimes,
                            "the regime the signals use; rule-based, nothing fitted")]
    for name, shadow in settings.forward_test.shadows.items():
        try:  # a shadow whose model or data is not available here is left out, not an error
            if shadow.regime_model is not None:
                row = _model_row(engine, shadow.regime_model)
                fit = load_active_hmm(row, settings)
                regimes, _ = active_hmm_regime(fit, load_observations(engine, settings),
                                               settings.regime.min_dwell_days)
                choices.append(RegimeChoice(
                    f"HMM {shadow.regime_model} ({name})", regimes,
                    f"the HMM's final fit, filtered; trained up to {row['training_end']}, so in sample before "
                    "that date"))
            elif shadow.regime_rule is not None:
                from quant.regime.rule_bonds import rule_bonds_regime

                choices.append(RegimeChoice(f"{shadow.regime_rule} ({name})", rule_bonds_regime(engine, settings),
                                            "rule-based with the bond inputs of section 12.5; nothing fitted"))
        except ValueError:  # ActiveModelError is a ValueError
            continue
    return choices


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


def forward_runs(engine: Engine) -> pd.DataFrame:
    """Every forward-test run, newest first (Phase 10)."""
    return queries.list_forward_runs(engine).sort_values("run_id", ascending=False).reset_index(drop=True)


def forward_report(engine: Engine, settings: Settings, run_id: int | None = None):
    """The forward report of a run (default: the running one): see quant/forward/report.py."""
    from quant.forward.report import build_report

    return build_report(engine, settings, run_id)


def growth_and_drawdown(values: pd.DataFrame) -> pd.DataFrame:
    """Long frame: date, portfolio, growth (value / first value), drawdown."""
    if values.empty:
        return pd.DataFrame(columns=["date", "portfolio", "growth", "drawdown"])
    growth = values / values.iloc[0]
    drawdown = values / values.cummax() - 1
    frame = growth.rename_axis("date").reset_index().melt(id_vars="date", var_name="portfolio", value_name="growth")
    dd = drawdown.rename_axis("date").reset_index().melt(id_vars="date", var_name="portfolio", value_name="drawdown")
    return frame.merge(dd, on=["date", "portfolio"]).dropna(subset=["growth"])


def latest_frozen(engine: Engine, run_id: int | None = None) -> tuple[pd.Timestamp | None, pd.DataFrame, pd.DataFrame]:
    """The latest frozen session of a run: one row per portfolio, and the held targets."""
    sessions = queries.frozen_sessions(engine, run_id=run_id)
    if sessions.empty:
        return None, sessions, pd.DataFrame(columns=["portfolio", "symbol", "target_weight", "close", "score"])
    last = sessions["session"].max()
    headers = sessions[sessions["session"] == last].reset_index(drop=True)
    ft = tables.forward_targets
    run = queries.forward_run(engine, run_id)["run_id"]
    with engine.connect() as conn:
        rows = conn.execute(select(ft.c.portfolio, ft.c.symbol, ft.c.target_weight, ft.c.close, ft.c.score)
                            .where(ft.c.run_id == run, ft.c.session == last)
                            .order_by(ft.c.portfolio, ft.c.target_weight.desc(), ft.c.symbol)).mappings().all()
    return pd.Timestamp(last), headers, pd.DataFrame([dict(r) for r in rows],
                                                     columns=["portfolio", "symbol", "target_weight", "close", "score"])


# Creator opinions (sentiment_proposal.md 8.1): the same data as `quant sentiment report` / `stock`.

def _has_opinions(engine: Engine) -> bool:
    return inspect(engine).has_table("sentiment_opinions")  # not created until `quant sentiment ...` runs


def sentiment_days(engine: Engine, settings: Settings) -> list[date]:
    """Effective sessions with opinions of the configured extractor (report_models), newest first."""
    if not _has_opinions(engine):
        return []
    from quant.sentiment.summarize import load_opinions

    return sorted(set(load_opinions(engine, settings)["effective_date"]), reverse=True)


@dataclass(frozen=True)
class SentimentDay:
    day: date
    opinions: pd.DataFrame  # every opinion effective that day (verified or not)
    verified: pd.DataFrame
    market: str | None  # the market-mood paragraph
    summaries: dict[str, dict]  # ticker -> stored summary (written from exactly these verified opinions)
    stocks: pd.DataFrame  # the report's table: Stock, Pool, Bull, Bear, Cond, Neutral, Net, Creators
    etfs: pd.DataFrame
    quotes: pd.DataFrame  # one row per verified opinion, with a link to the second in the video


def quote_rows(opinions: pd.DataFrame) -> pd.DataFrame:
    from quant.sentiment.report import ACTION, MARKS, _clock, _condition, _link

    columns = ["date", "ticker", "channel", "stance", "horizon", "conviction", "condition", "action", "quote",
               "reasoning", "at", "link", "title", "model"]
    if opinions.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({
        "date": pd.to_datetime(opinions["effective_date"]),
        "ticker": opinions["ticker"], "channel": opinions["channel"],
        "stance": [f"{MARKS[s]} {s.lower()}" for s in opinions["stance"]],
        "horizon": opinions["horizon"].str.lower(), "conviction": opinions["conviction"],
        "condition": opinions["condition"].map(_condition), "action": opinions["action"].map(ACTION),
        "quote": opinions["quote"], "reasoning": opinions["reasoning"].fillna(""),
        "at": opinions["start_seconds"].map(_clock),
        "link": [_link(o) for o in opinions.itertuples(index=False)], "title": opinions["title"],
        "model": opinions["model"],
    })[columns].reset_index(drop=True)


def sentiment_day(engine: Engine, settings: Settings, day: date) -> SentimentDay:
    from quant.sentiment import tables as st
    from quant.sentiment.report import stock_table
    from quant.sentiment.summarize import inputs_hash, load_opinions, load_summaries

    opinions = load_opinions(engine, settings, day, day)
    verified = opinions[opinions["verification"] == "VERIFIED"]
    summaries = load_summaries(engine, settings, day, inputs_hash(verified, settings)) if len(verified) else {}
    pool = set(settings.market.stocks)
    tables_ = {kind: stock_table(verified[verified["asset_type"] == kind], pool) for kind in ("STOCK", "ETF")}
    market = summaries.pop(st.MARKET, {}).get("summary")
    return SentimentDay(day, opinions, verified, market, summaries, tables_["STOCK"], tables_["ETF"],
                        quote_rows(verified))


def stock_opinions(engine: Engine, settings: Settings, symbol: str, start: date | None = None) -> pd.DataFrame:
    """Verified opinions on one stock from `start` on, oldest first (quote_rows columns)."""
    if not _has_opinions(engine):
        return quote_rows(pd.DataFrame())
    from quant.sentiment.report import stock_history

    return quote_rows(stock_history(engine, settings, symbol, start))


def opinion_marks(opinions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One mark per effective session and stance, placed on that session's close (stock_detail frame).

    Opinions effective after the last stored close (published tonight) have no mark yet.
    """
    columns = ["date", "stance", "close", "count", "creators"]
    if opinions.empty or prices.empty:
        return pd.DataFrame(columns=columns)
    grouped = opinions.groupby(["date", "stance"], as_index=False).agg(
        count=("channel", "size"), creators=("channel", lambda c: ", ".join(dict.fromkeys(c))))
    marks = grouped.merge(prices[["date", "close"]], on="date", how="inner")
    return marks[columns].sort_values(["date", "stance"]).reset_index(drop=True)
