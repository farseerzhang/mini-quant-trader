"""Which stocks each rule-based strategy holds at the close of T (proposal section 17).

Inputs are a feature panel: one DataFrame per feature column, dates x stocks, as
stored for T. Outputs are boolean DataFrames of the same shape. A missing input
(warm-up) never counts as a signal. Every rule uses data at or before T only;
the backtest fills at the open of T+1.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from quant.config.settings import MeanReversionSettings

Panel = Mapping[str, pd.DataFrame]

TREND_INPUTS = ("price_vs_sma200", "sma50_vs_sma200", "macd_hist_over_price")
MEAN_REVERSION_INPUTS = ("price_vs_sma200", "rsi14", "macd_hist_over_price")
PANEL_COLUMNS = tuple(dict.fromkeys((*TREND_INPUTS, *MEAN_REVERSION_INPUTS)))


def _require(panel: Panel, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in panel]
    if missing:
        raise ValueError(f"feature panel is missing: {', '.join(missing)}")


def trend_holdings(panel: Panel) -> pd.DataFrame:
    """Enter on an uptrend confirmed by momentum; hold until the trend itself breaks.

    Entry at T: price > SMA200, SMA50 > SMA200 and the MACD above its signal line.
    Hold while price > SMA200 and SMA50 > SMA200. The MACD is an entry trigger
    only: it crosses its signal line about 20 times per stock per year, so
    requiring it every session made the median holding 8 sessions.
    """
    _require(panel, TREND_INPUTS)
    in_trend = ((panel["price_vs_sma200"] > 0) & (panel["sma50_vs_sma200"] > 0)).to_numpy()
    entry = in_trend & (panel["macd_hist_over_price"] > 0).to_numpy()

    held = np.zeros(entry.shape[1], dtype=bool)
    out = np.zeros(entry.shape, dtype=bool)
    for t in range(entry.shape[0]):
        held = entry[t] | (held & in_trend[t])
        out[t] = held
    return pd.DataFrame(out, index=panel["price_vs_sma200"].index, columns=panel["price_vs_sma200"].columns)


def mean_reversion_holdings(
    panel: Panel, settings: MeanReversionSettings, allow_entry: pd.Series | None = None
) -> pd.DataFrame:
    """Buy a dip inside an uptrend; hold until RSI recovers or the time limit passes.

    Entry at T: price > SMA200, RSI14 < rsi_entry and the MACD histogram rising
    (above its value at T-1). Exit at T: RSI14 > rsi_exit, or the position has
    been held for max_holding_days sessions. A stock cannot re-enter on the
    session it exits. ``allow_entry`` (a boolean per date) restricts new entries,
    e.g. to SIDEWAYS days in regime_switch; exits are never restricted.
    """
    _require(panel, MEAN_REVERSION_INPUTS)
    rsi = panel["rsi14"]
    histogram = panel["macd_hist_over_price"]
    entry = (panel["price_vs_sma200"] > 0) & (rsi < settings.rsi_entry) & (histogram > histogram.shift(1))
    if allow_entry is not None:
        allowed = allow_entry.reindex(entry.index).fillna(False).astype(bool).to_numpy()
        entry = entry & allowed[:, None]
    exit_signal = (rsi > settings.rsi_exit).to_numpy()
    entry_signal = entry.to_numpy()

    held = np.zeros(entry.shape[1], dtype=bool)
    age = np.zeros(entry.shape[1], dtype=int)  # sessions since the entry signal
    out = np.zeros(entry.shape, dtype=bool)
    for t in range(entry.shape[0]):
        age[held] += 1
        leaving = held & (exit_signal[t] | (age >= settings.max_holding_days))
        held &= ~leaving
        entering = ~held & ~leaving & entry_signal[t]
        held |= entering
        age[entering] = 0
        out[t] = held
    return pd.DataFrame(out, index=entry.index, columns=entry.columns)
