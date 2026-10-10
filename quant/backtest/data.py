"""Total-return open and close prices for the backtest (proposal sections 6.3 and 20)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sqlalchemy import Engine

from quant.features.technical import prepare_adjusted_series
from quant.persistence import queries

_NO_ACTIONS = pd.DataFrame(columns=["ex_date", "action_type", "value"])


@dataclass(frozen=True)
class Prices:
    """Dates x symbols. Orders fill at ``open``; the portfolio is valued at ``close``."""

    open: pd.DataFrame
    close: pd.DataFrame


def load_prices(engine: Engine, symbols: Sequence[str]) -> Prices:
    """Split- and dividend-adjusted (total-return) open and close for each symbol."""
    stored, actions = queries.load_market_data(engine, symbols)
    missing = [symbol for symbol in symbols if symbol not in stored]
    if missing:
        raise ValueError(f"no stored prices for: {', '.join(missing)}; update market data first")
    opens, closes = {}, {}
    for symbol in symbols:
        adjusted = prepare_adjusted_series(stored[symbol], actions.get(symbol, _NO_ACTIONS))
        opens[symbol] = adjusted["total_return_open"]
        closes[symbol] = adjusted["total_return_close"]
    return Prices(open=pd.DataFrame(opens), close=pd.DataFrame(closes))
