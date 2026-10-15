"""Mini Quant Trader dashboard (proposal section 25). Read-only; the CLI does all the work.

    streamlit run dashboard/app.py
    streamlit run dashboard/app.py -- --config path/to/settings.yaml --db path/to/quant.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without installing

from quant.config.settings import DEFAULT_CONFIG_PATH, load_settings  # noqa: E402
from quant.dashboard import data  # noqa: E402
from quant.persistence.database import create_db_engine  # noqa: E402
from quant.sentiment.report import models_used  # noqa: E402

# Regime colors: the validated palette of the report chart (quant/regime/chart.py).
REGIME_COLORS = alt.Scale(domain=["BULL", "SIDEWAYS", "BEAR"], range=["#2a78d6", "#1baf7a", "#eb6834"])
SIGNAL_ICONS = {"BUY": "▲ BUY", "REDUCE": "▼ REDUCE", "EXIT": "✕ EXIT", "HOLD": "● HOLD"}
# Fixed color per portfolio, so a color always means the same portfolio whatever is selected.
PORTFOLIO_COLORS = {"regime_switch": "#2a78d6", "SPY": "#52514e", "equal_weight": "#eb6834", "trend": "#1baf7a",
                    "mean_reversion": "#9a55d4", "regime_switch_bear50": "#d4a20b"}
OTHER_COLORS = ["#c94f8a", "#4a9fb5", "#7d7a6f"]
PAGES = ("Status", "Market", "Signals", "Sentiment", "Forward test", "Stock detail", "Models", "Backtest")
# Creator stances: the report's marks; up/down in the green/orange of the MACD bars.
STANCES = ["▲ bullish", "▼ bearish", "◆ conditional", "● neutral"]
STANCE_COLORS = alt.Scale(domain=STANCES, range=["#1baf7a", "#eb6834", "#d4a20b", "#898781"])
STANCE_SHAPES = alt.Scale(domain=STANCES, range=["triangle-up", "triangle-down", "diamond", "circle"])
# Fixed colors per forward-test portfolio (a color always means the same portfolio).
FORWARD_COLORS = {"live": "#2a78d6", "real": "#eb6834", "SPY": "#52514e", "shadow_hmm": "#1baf7a",
                  "shadow_lr": "#9a55d4", "shadow_rule_bonds": "#d4a20b"}


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--db")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


@st.cache_resource
def _engine(db_path: str, busy_timeout_ms: int):
    return create_db_engine(db_path, busy_timeout_ms)


def _pct(value, digits=1):
    return "-" if value is None or pd.isna(value) else f"{value:.{digits}%}"


def page_status(engine, settings):
    report = data.status(engine, settings)
    cols = st.columns(4)
    cols[0].metric("Last completed session", str(report.expected_session or "unknown"))
    cols[1].metric("Prices to", str(report.prices_as_of or "none"))
    cols[2].metric(f"Regime on {report.regime_as_of or '-'}", f"{report.regime or 'none'}")
    counts = ", ".join(f"{k} {v}" for k, v in sorted(report.signal_counts.items())) or "none"
    cols[3].metric(f"Signals on {report.signals_as_of or '-'}", str(sum(report.signal_counts.values())))
    cols[3].caption(counts)
    if report.warnings:
        for warning in report.warnings:
            st.warning(warning)
    else:
        st.success("No warnings: data, regime and signals are current.")
    st.subheader("Last successful runs")
    stages = pd.DataFrame([{"stage": s.label, "finished (UTC)": s.last_ok, "data to": s.data_as_of,
                            "last run failed": "yes" if s.failed_since else ""} for s in report.stages])
    stages.loc[len(stages)] = ["run-daily", report.last_run_daily, None, ""]
    stages["finished (UTC)"] = stages["finished (UTC)"].map(
        lambda v: v.strftime("%Y-%m-%d %H:%M") if v is not None and pd.notna(v) else "never")
    stages["data to"] = stages["data to"].map(lambda v: str(v) if v is not None and pd.notna(v) else "never")
    stages.loc[stages["stage"] == "run-daily", "data to"] = ""
    st.dataframe(stages, hide_index=True, width="stretch")
    st.subheader("Models")
    rows = []
    for role in ("REGIME", "PREDICTOR"):
        model = report.active[role]
        candidates, passed = report.candidates[role]
        rows.append({"role": role, "ACTIVE": model["model_id"] if model else "none (rule-based)",
                     "candidates": candidates, "passed gates": passed})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Signals use {report.strategy_in_use} with the {report.regime_in_use} regime. "
               f"Holdout evaluations so far: {report.holdout_evaluations}. "
               f"Quarantined rows: {report.quarantined_total}.")


def _ribbon_chart(history: pd.DataFrame, title: str):
    runs = data.regime_runs(history)
    low, high = float(history["close"].min()) * 0.95, float(history["close"].max()) * 1.05
    base = alt.Chart(history).encode(x=alt.X("date:T", title=None))
    line = base.mark_line(strokeWidth=2, color="#52514e").encode(
        y=alt.Y("close:Q", scale=alt.Scale(type="log", domain=[low, high], nice=False), title=title),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("close:Q", format=",.2f"), alt.Tooltip("regime:N")])
    ribbon = alt.Chart(runs).mark_rect(height=18).encode(
        x=alt.X("start:T", axis=None), x2="end:T", color=alt.Color("regime:N", scale=REGIME_COLORS, title="Regime"),
        tooltip=[alt.Tooltip("regime:N"), alt.Tooltip("start:T"), alt.Tooltip("end:T")])
    return alt.vconcat(line.properties(height=300), ribbon.properties(height=24), spacing=4).resolve_scale(x="shared")


def page_market(engine, settings):
    overview = data.market_overview(engine, settings)
    cols = st.columns(1 + len(overview.quotes))
    cols[0].metric(f"Regime ({overview.regime_source})", overview.regime or "none")
    if overview.score is not None:
        cols[0].caption(f"score {overview.score:+.0f} of ±5")
    for col, (symbol, row) in zip(cols[1:], overview.quotes.iterrows()):
        col.metric(symbol, f"{row['close']:,.2f}", _pct(row["1 day"], 2),
                   delta_color="inverse" if symbol == "^VIX" else "normal")
    st.dataframe(overview.quotes.style.format({"close": "{:,.2f}", "1 day": "{:+.2%}", "1 month": "{:+.1%}",
                                               "1 year": "{:+.1%}"}), width="stretch")
    st.subheader("SPY with the regime in use, last 5 years")
    st.altair_chart(_ribbon_chart(overview.history, "SPY total-return price (log)"), width="stretch")
    st.caption("Changes and the chart include dividends (total return); the close is as traded.")


def page_signals(engine, settings):
    session, signals = data.latest_signals(engine)
    if session is None:
        st.info("No signals yet: run `python -m quant run-daily`.")
        return
    st.subheader(f"Signals for {session.date()}: fill at the next open")
    counts = signals["signal"].value_counts()
    cols = st.columns(4)
    for col, name in zip(cols, ("BUY", "REDUCE", "EXIT", "HOLD")):
        col.metric(SIGNAL_ICONS[name], int(counts.get(name, 0)))
    strategies = sorted(signals["strategy"].unique())
    if len(strategies) > 1:
        chosen = st.selectbox("Strategy", strategies)
        signals = signals[signals["strategy"] == chosen]
    show = signals.assign(signal=signals["signal"].map(SIGNAL_ICONS),
                          suggested_shares=signals["suggested_shares"].map(lambda v: f"{v:+g}" if v else "0"))
    columns = ["symbol", "signal", "current_weight", "target_weight", "suggested_shares", "close", "score",
               "regime", "strategy", "model_version", "reason"]
    if show["score"].isna().all():  # no ACTIVE predictor
        columns.remove("score")
    st.dataframe(
        show[columns],
        hide_index=True, width="stretch",
        column_config={
            "current_weight": st.column_config.NumberColumn("current", format="percent"),
            "target_weight": st.column_config.NumberColumn("target", format="percent"),
            "suggested_shares": st.column_config.TextColumn("shares"),
            "close": st.column_config.NumberColumn("close", format="%.2f"),
            "score": st.column_config.NumberColumn("ML probability", format="%.2f"),
            "reason": st.column_config.TextColumn("reason", width="large"),
        },
    )
    symbol = st.selectbox("Read a reason", signals["symbol"])
    st.write(signals.set_index("symbol").loc[symbol, "reason"])
    as_of, positions = data.latest_positions(engine)
    st.subheader("Positions snapshot" + (f" ({as_of.date()})" if as_of is not None else ""))
    if positions.empty:
        st.info("No positions snapshot stored (no positions file at the last run).")
    else:
        shown = positions.assign(managed=positions["managed"].map({1: "yes", 0: "no: outside the pool"}))
        st.dataframe(shown[["symbol", "quantity", "cost_basis", "managed"]], hide_index=True)


def _quotes_table(quotes: pd.DataFrame, show_ticker: bool = True):
    columns = ["date", "ticker", "channel", "stance", "horizon", "conviction", "condition", "action", "quote",
               "link", "reasoning", "title", "model"]
    if not show_ticker:
        columns.remove("ticker")
    st.dataframe(quotes[columns], hide_index=True, width="stretch", column_config={
        "date": st.column_config.DateColumn("effective"),
        "channel": st.column_config.TextColumn("creator"),
        "conviction": st.column_config.NumberColumn(format="%d", help="1 low .. 3 high"),
        "quote": st.column_config.TextColumn(width="large"),
        "link": st.column_config.LinkColumn("video", display_text="▶ open at the quote"),
        "reasoning": st.column_config.TextColumn("reason"),
    })


def page_sentiment(engine, settings):
    days = data.sentiment_days(engine, settings)
    if not days:
        st.info("No creator opinions yet: run `python -m quant sentiment run` (fetch, extract, summarize, report).")
        return
    day = st.selectbox("Effective session", days, format_func=lambda d: f"{d:%Y-%m-%d (%a)}")
    view = data.sentiment_day(engine, settings, day)
    extraction, summary = settings.sentiment.extraction, settings.sentiment.summary
    st.caption(f"Opinions published after the previous session's open count from this session's open. "
               f"Extraction {extraction.backend} {models_used(view.opinions)}, prompt v{extraction.prompt_version} "
               f"({extraction.model} accepted provisionally; per video the first of "
               f"{', '.join(extraction.report_models)}); "
               f"summaries {summary.backend}/{summary.model}, written from verified opinions only. "
               "Research only: a creator's opinion is not a signal.")
    verified, opinions = view.verified, view.opinions
    with st.container(horizontal=True):
        st.metric("verified opinions", len(verified), border=True)
        st.metric("creators", verified["channel"].nunique(), border=True)
        st.metric("tickers", verified["ticker"].nunique(), border=True)
        st.metric("not counted", len(opinions) - len(verified), border=True,
                  help="quotes that did not match the transcript, and names that are not a known ticker")

    st.subheader("Market mood")
    if view.market:
        st.write(view.market)
    else:
        st.caption("No summary for this day yet: run `python -m quant sentiment summarize`.")

    pool_only = st.toggle("Only stocks in my pool", help="ETFs and indices are always shown")
    config = {"Pool": st.column_config.TextColumn("pool", help="★ = in your stock pool"),
              "Net": st.column_config.NumberColumn(format="%+d", help="creators bullish minus creators bearish"),
              "Creators": st.column_config.TextColumn(width="large", help="▲ bullish ▼ bearish ◆ conditional "
                                                      "● neutral; S/M/L = short / medium / long term")}
    stocks = view.stocks[view.stocks["Pool"] == "★"] if pool_only and not view.stocks.empty else view.stocks
    for title, table in (("Stocks", stocks), ("ETFs and indices", view.etfs)):
        if table.empty:
            continue
        st.subheader(title)
        st.dataframe(table, hide_index=True, width="stretch", column_config=config)
    shown = [t for table in (stocks, view.etfs) if not table.empty for t in table["Stock"]]
    if not shown:
        st.caption("No verified opinions on stocks in your pool this day." if pool_only
                   else "No verified opinions this day.")
    else:
        ticker = st.selectbox("Summary and quotes for", shown)
        entry = view.summaries.get(ticker)
        if entry is None:
            st.caption("No summary for this stock yet.")
        else:
            if entry["disagreement"]:
                st.warning("Creators disagree on this stock.")
            st.write(entry["summary"])
        _quotes_table(view.quotes[view.quotes["ticker"] == ticker], show_ticker=False)
        with st.expander(f"All {len(view.quotes)} verified quotes of the day"):
            _quotes_table(view.quotes)

    unverified = opinions[opinions["verification"] == "UNVERIFIED"]
    unresolved = opinions[opinions["verification"] == "UNRESOLVED"]
    if len(unverified) or len(unresolved):
        with st.expander(f"Not counted ({len(unverified) + len(unresolved)})"):
            if len(unverified):
                st.markdown("**Quotes that did not match the transcript**")
                st.dataframe(unverified[["ticker", "channel", "stance", "quote"]], hide_index=True, width="stretch")
            if len(unresolved):
                st.markdown("**Names that are not a known ticker** (add real stocks to "
                            "`config/sentiment_aliases.yaml`)")
                names = unresolved["name_as_said"].value_counts().rename_axis("name").reset_index()
                st.dataframe(names, hide_index=True)
    st.caption(f"The same data as `{Path(summary.reports_dir).as_posix()}/{day:%Y-%m-%d}.md`; one stock over time: Stock detail, or "
               "`python -m quant sentiment stock <ticker>`.")


def page_stock(engine, settings):
    symbol = st.selectbox("Stock", list(settings.market.stocks))
    years = st.radio("Period", [1, 2, 5], index=1, horizontal=True, format_func=lambda y: f"{y} year" + ("s" if y > 1 else ""))
    frame = data.stock_detail(engine, settings, symbol, years)
    if frame.empty:
        st.info(f"No prices stored for {symbol}.")
        return
    long = frame.melt(id_vars="date", value_vars=["close", "sma50", "sma200"], var_name="line")
    price = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("value:Q", scale=alt.Scale(zero=False), title="total-return price", axis=alt.Axis(minExtent=70)),
        color=alt.Color("line:N", scale=alt.Scale(domain=["close", "sma50", "sma200"],
                                                   range=["#0b0b0b", "#2a78d6", "#eb6834"]), title=None,
                        legend=alt.Legend(orient="top")),
        tooltip=["date:T", "line:N", alt.Tooltip("value:Q", format=",.2f")]).properties(height=280)
    opinions = data.stock_opinions(engine, settings, symbol, frame["date"].iloc[0].date())
    marks = data.opinion_marks(opinions, frame)
    if not marks.empty:  # creator opinions on the close of their effective session
        price = alt.layer(price, alt.Chart(marks).mark_point(filled=True, size=110, opacity=0.9).encode(
            x="date:T", y="close:Q",
            color=alt.Color("stance:N", scale=STANCE_COLORS, title=None, legend=alt.Legend(orient="top")),
            shape=alt.Shape("stance:N", scale=STANCE_SHAPES, legend=None),  # the labels carry the shapes
            tooltip=["date:T", "stance:N", alt.Tooltip("count:Q", title="opinions"), "creators:N"],
        )).resolve_scale(color="independent")
    rsi =alt.Chart(frame).mark_line(strokeWidth=2, color="#2a78d6").encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(labels=False, ticks=False)), y=alt.Y("rsi14:Q", scale=alt.Scale(domain=[0, 100]), title="RSI 14", axis=alt.Axis(minExtent=70)),
        tooltip=["date:T", alt.Tooltip("rsi14:Q", format=".0f")]).properties(height=110)
    rules = alt.Chart(pd.DataFrame({"y": [settings.strategies.mean_reversion.rsi_entry,
                                          settings.strategies.mean_reversion.rsi_exit]})).mark_rule(
        strokeDash=[3, 3], color="#898781").encode(y="y:Q")
    macd = alt.Chart(frame).mark_bar().encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(labels=False, ticks=False)), y=alt.Y("macd_hist:Q", title="MACD hist", axis=alt.Axis(minExtent=70)),
        color=alt.condition("datum.macd_hist >= 0", alt.value("#1baf7a"), alt.value("#eb6834")),
        tooltip=["date:T", alt.Tooltip("macd_hist:Q", format=".2f")]).properties(height=110)
    volume = alt.Chart(frame).mark_bar(color="#c3c2b7").encode(
        x=alt.X("date:T", title=None), y=alt.Y("volume:Q", title="volume", axis=alt.Axis(format="~s", minExtent=70)),
        tooltip=["date:T", alt.Tooltip("volume:Q", format=",.0f")]).properties(height=90)
    for chart in (price, rsi + rules, macd, volume):  # separate full-width panels on the same dates
        st.altair_chart(chart, width="stretch")
    st.caption(f"Dashed lines: the dip-buy RSI entry ({settings.strategies.mean_reversion.rsi_entry:g}) "
               f"and exit ({settings.strategies.mean_reversion.rsi_exit:g}).")

    st.subheader("Creator opinions")
    if opinions.empty:
        st.caption(f"No verified creator opinions on {symbol} in this period "
                   "(`python -m quant sentiment run` collects them).")
        return
    counts = opinions["stance"].value_counts()
    with st.container(horizontal=True):
        for stance in STANCES:
            st.metric(stance, int(counts.get(stance, 0)), border=True)
    if marks["date"].nunique() < opinions["date"].nunique():
        st.caption("Opinions effective after the last stored close are listed but not yet on the chart.")
    _quotes_table(opinions.sort_values("date", ascending=False), show_ticker=False)
    st.caption("Marks sit on the close of the session the opinion counts from (its effective date). "
               "Research only: a creator's opinion is not a signal.")


def page_models(engine, settings):
    frame = data.models(engine, settings.retrain.reports_dir)
    if frame.empty:
        st.info("No models registered: `python -m quant retrain --role predictor` or `--role regime`.")
        return
    st.dataframe(frame[["model_id", "role", "model_type", "version", "status", "gates_passed", "failed_gates",
                        "training_start", "training_end", "created_at"]],
                 hide_index=True, width="stretch",
                 column_config={"gates_passed": st.column_config.CheckboxColumn("gates passed")})
    model_id = st.selectbox("Model", frame["model_id"])
    row = frame.set_index("model_id").loc[model_id]
    st.dataframe(data.gate_table(engine, model_id), hide_index=True, width="stretch")
    if row["report_exists"]:
        report = Path(row["validation_report_path"])
        st.caption(f"Report: {report}")
        with st.expander("Validation report", expanded=False):
            text = report.read_text(encoding="utf-8")
            svg = report.parent / f"{model_id}_regimes.svg"
            if svg.exists():
                st.image(str(svg), width="stretch")
            st.markdown(text.split("## Regime timeline")[0] if svg.exists() else text)
    st.caption("Promotion is manual: `python -m quant promote <model_id>` (refuses failed gates without --override).")


def page_backtest(engine, settings):
    batches = data.backtest_batches(engine)
    if batches.empty:
        st.info("No backtests saved: `python -m quant backtest`.")
        return
    label = batches.apply(lambda r: f"{r['batch_id']} ({r['portfolios']} portfolios, {r['stocks']:.0f} stocks)", axis=1)
    batch = batches["batch_id"][st.selectbox("Batch", range(len(batches)), format_func=lambda i: label[i])]
    metrics = data.backtest_metrics(engine, batch)
    st.dataframe(metrics.drop(columns="run_id"), hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(name.replace("_", " "), format="percent")
        for name in ("cagr", "max_drawdown", "volatility", "avg_invested", "win_rate", "sharpe_percentile")})
    equity = data.backtest_equity(engine, batch)
    chosen = st.multiselect("Portfolios", sorted(equity["strategy"].unique()),
                            default=[s for s in ("regime_switch", "SPY", "equal_weight") if s in set(equity["strategy"])])
    equity = equity[equity["strategy"].isin(chosen)]
    if equity.empty:
        st.info("Choose at least one portfolio.")
        return
    names = sorted(equity["strategy"].unique())
    others = iter(OTHER_COLORS * 3)
    colors = alt.Scale(domain=names, range=[PORTFOLIO_COLORS.get(n) or next(others) for n in names])
    low, high = float(equity["value"].min()) * 0.95, float(equity["value"].max()) * 1.05
    value = alt.Chart(equity).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("value:Q", scale=alt.Scale(type="log", domain=[low, high], nice=False), title="growth of 1 (log)"),
        color=alt.Color("strategy:N", title=None, scale=colors, legend=alt.Legend(orient="top")),
        tooltip=["date:T", "strategy:N", alt.Tooltip("value:Q", format=".2f")]).properties(height=300)
    drawdown = alt.Chart(equity).mark_line(strokeWidth=1.5).encode(
        x=alt.X("date:T", title=None), y=alt.Y("drawdown:Q", axis=alt.Axis(format="%"), title="drawdown"),
        color=alt.Color("strategy:N", legend=None, scale=colors),
        tooltip=["date:T", "strategy:N", alt.Tooltip("drawdown:Q", format=".1%")]).properties(height=160)
    charts = [c.properties(width="container") for c in (value, drawdown)]
    st.altair_chart(alt.vconcat(*charts, spacing=6).resolve_scale(x="shared"), width="stretch")
    st.caption("Development period only, after costs. The stock pool is chosen today (survivorship bias); "
               "cash earns nothing.")


def _forward_scale(names):
    others = iter(["#c94f8a", "#4a9fb5", "#d4a20b", "#7d7a6f"] * 3)
    return alt.Scale(domain=list(names), range=[FORWARD_COLORS.get(n) or next(others) for n in names])


def page_forward(engine, settings):
    runs = data.forward_runs(engine)
    if runs.empty:
        st.info("The forward test has not started. Put your real account in `data/positions.csv`, then run "
                "`python -m quant forward-test --start` after a close (before the next open). From then on "
                "`run-daily` freezes every evening and you record fills in `data/trades.csv`.")
        return
    labels = {int(r.run_id): f"Run {int(r.run_id)}: since {r.started_on}"
              + (f" (ended {r.ended_at:%Y-%m-%d})" if pd.notna(r.ended_at) else " (running)")
              for r in runs.itertuples()}
    run_id = st.selectbox("Run", list(labels), format_func=labels.get)
    report = data.forward_report(engine, settings, run_id)
    if report.ended:
        st.warning(f"Run {run_id} ended {report.ended[0]:%Y-%m-%d}: {report.ended[1]}")
    st.caption(f"Since {report.started_on.date()} with {report.start_value:,.2f}; through "
               f"{report.last_session.date()}.")
    st.info(report.checkpoint)

    with st.container(horizontal=True):
        for name, row in report.scores.iterrows():
            st.metric(name, f"{row['total_return']:+.2%}", border=True,
                      help=f"max drawdown {row['max_drawdown']:.1%}"
                           + ("" if pd.isna(row["missed"]) else f"; {int(row['missed'])} missed session(s)"))
    table = report.scores.rename(columns={"total_return": "return", "max_drawdown": "max drawdown"})
    table["missed"] = table["missed"].map(lambda v: "-" if pd.isna(v) else str(int(v)))  # paper portfolios only
    st.dataframe(table, width="stretch", column_config={
        "return": st.column_config.NumberColumn(format="percent"),
        "volatility": st.column_config.NumberColumn(format="percent"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "max drawdown": st.column_config.NumberColumn(format="percent"),
    })

    curves = data.growth_and_drawdown(report.values)
    if len(report.values) > 1:
        colors = _forward_scale(report.values.columns)
        growth = alt.Chart(curves).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("growth:Q", scale=alt.Scale(zero=False), title="growth of 1"),
            color=alt.Color("portfolio:N", scale=colors, title=None, legend=alt.Legend(orient="top")),
            tooltip=["date:T", "portfolio:N", alt.Tooltip("growth:Q", format=".4f")]).properties(height=280)
        drawdown = alt.Chart(curves).mark_line(strokeWidth=1.5).encode(
            x=alt.X("date:T", title=None), y=alt.Y("drawdown:Q", axis=alt.Axis(format="%"), title="drawdown"),
            color=alt.Color("portfolio:N", scale=colors, legend=None),
            tooltip=["date:T", "portfolio:N", alt.Tooltip("drawdown:Q", format=".1%")]).properties(height=140)
        st.altair_chart(growth, width="stretch")
        st.altair_chart(drawdown, width="stretch")
    else:
        st.caption("The curves appear after the first forward session.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Followed share of the live targets**")
        if report.followed.empty:
            st.caption("Not yet: it compares each close with the previous evening's live target.")
        else:
            st.metric("average", f"{report.followed.mean():.0%}", border=True,
                      chart_data=report.followed.tolist(), chart_type="line",
                      help="sum over stocks of min(your weight, the live target weight)")
    with right:
        st.markdown("**Execution against the open**")
        if report.costs.empty:
            st.caption("No trades recorded in `data/trades.csv` for this run.")
        else:
            costs = report.costs.assign(date=report.costs["date"].dt.date)
            weighted = (costs["cost_bps"] * costs["notional"]).sum() / costs["notional"].sum()
            st.metric("average cost", f"{weighted:+.1f} bp", border=True,
                      help=f"fees included; the backtest assumes {report.assumed_cost_bps:g} bp")
            st.dataframe(costs, hide_index=True, width="stretch", column_config={
                "notional": st.column_config.NumberColumn(format="%.2f"),
                "cost_bps": st.column_config.NumberColumn("cost (bp)", format="%+.1f")})

    session, headers, targets = data.latest_frozen(engine, run_id)
    if session is not None:
        st.subheader(f"Frozen on {session.date()}")
        names = list(headers["portfolio"])
        for tab, name in zip(st.tabs(names), names):
            with tab:
                head = headers.set_index("portfolio").loc[name]
                st.caption(f"{head['strategy']} · {head['regime']} regime ({head['regime_source']}) · "
                           f"model {head['model_version']} · {head['positions']} positions, "
                           f"{head['invested']:.0%} invested")
                held = targets[targets["portfolio"] == name].drop(columns="portfolio")
                if held.empty:
                    st.caption("All cash.")
                else:
                    held = held.drop(columns=[c for c in ("score", "close") if held[c].isna().all()])
                    st.dataframe(held, hide_index=True, width="stretch", column_config={
                        "target_weight": st.column_config.NumberColumn("target", format="percent"),
                        "close": st.column_config.NumberColumn(format="%.2f"),
                        "score": st.column_config.NumberColumn("probability", format="%.2f")})
    for warning in report.warnings:
        st.warning(warning)
    st.caption("Paper portfolios trade their frozen targets at the next open with the backtest's costs; `real` is "
               "your trade log at as-traded closes. A short forward record is mostly noise.")


def main():
    st.set_page_config(page_title="Mini Quant Trader", layout="wide")
    args = _arguments()
    settings = load_settings(args.config)
    engine = _engine(args.db or settings.database.path, settings.database.busy_timeout_ms)
    st.sidebar.title("Mini Quant Trader")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    st.sidebar.caption(data.NOTICE)
    st.sidebar.caption("Read-only. Refresh after `python -m quant run-daily`.")
    st.title(page)
    st.caption(data.NOTICE)
    {"Status": page_status, "Market": page_market, "Signals": page_signals, "Sentiment": page_sentiment,
     "Stock detail": page_stock,
     "Models": page_models, "Backtest": page_backtest, "Forward test": page_forward}[page](engine, settings)


main()
