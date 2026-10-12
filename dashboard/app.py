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

# Regime colors: the validated palette of the report chart (quant/regime/chart.py).
REGIME_COLORS = alt.Scale(domain=["BULL", "SIDEWAYS", "BEAR"], range=["#2a78d6", "#1baf7a", "#eb6834"])
SIGNAL_ICONS = {"BUY": "▲ BUY", "REDUCE": "▼ REDUCE", "EXIT": "✕ EXIT", "HOLD": "● HOLD"}
# Fixed color per portfolio, so a color always means the same portfolio whatever is selected.
PORTFOLIO_COLORS = {"regime_switch": "#2a78d6", "SPY": "#52514e", "equal_weight": "#eb6834", "trend": "#1baf7a",
                    "mean_reversion": "#9a55d4", "regime_switch_bear50": "#d4a20b"}
OTHER_COLORS = ["#c94f8a", "#4a9fb5", "#7d7a6f"]
PAGES = ("Status", "Market", "Signals", "Stock detail", "Models", "Backtest")


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--db")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


@st.cache_resource
def _engine(db_path: str):
    return create_db_engine(db_path, 5000)


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
    rsi = alt.Chart(frame).mark_line(strokeWidth=2, color="#2a78d6").encode(
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


def main():
    st.set_page_config(page_title="Mini Quant Trader", layout="wide")
    args = _arguments()
    settings = load_settings(args.config)
    engine = _engine(args.db or settings.database.path)
    st.sidebar.title("Mini Quant Trader")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    st.sidebar.caption(data.NOTICE)
    st.sidebar.caption("Read-only. Refresh after `python -m quant run-daily`.")
    st.title(page)
    st.caption(data.NOTICE)
    {"Status": page_status, "Market": page_market, "Signals": page_signals, "Stock detail": page_stock,
     "Models": page_models, "Backtest": page_backtest}[page](engine, settings)


main()
