"""Performance metrics for one portfolio over one period (proposal section 20).

All return-based metrics use daily returns of the portfolio value at the close,
with a zero risk-free rate and 252 sessions a year. Cash earns nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSIONS_PER_YEAR = 252

METRIC_NAMES = (
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "avg_invested",
    "turnover",
    "num_trades",
    "win_rate",
    "profit_factor",
    "fees_paid",
)


def period_metrics(
    value: pd.Series,
    invested: pd.Series,
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float]:
    """Metrics for sessions in [start, end].

    ``value`` and ``invested`` are daily portfolio value and invested fraction,
    ``orders`` has Timestamp / Size / Price / Fees per fill, ``trades`` has Exit
    Timestamp / PnL / Status per trade (VectorBT exit trades). The first return
    of the period is measured from the close before ``start``, so consecutive
    periods chain to the full-history return.
    """
    returns = value.pct_change().loc[start:end].fillna(0.0)
    sessions = len(returns)
    if sessions == 0:
        return {name: float("nan") for name in METRIC_NAMES}
    years = sessions / SESSIONS_PER_YEAR
    growth = float((1.0 + returns).prod())
    equity = (1.0 + returns).cumprod()
    downside = np.sqrt(float((returns.clip(upper=0.0) ** 2).mean()))
    volatility = float(returns.std(ddof=0))
    mean = float(returns.mean())

    in_period = orders["Timestamp"].between(start, end) if len(orders) else pd.Series(dtype=bool)
    period_orders = orders[in_period] if len(orders) else orders
    traded = float((period_orders["Size"] * period_orders["Price"]).sum()) if len(orders) else 0.0
    average_value = float(value.loc[start:end].mean())

    closed = trades[(trades["Status"] == "Closed") & trades["Exit Timestamp"].between(start, end)] if len(trades) else trades
    wins = closed["PnL"][closed["PnL"] > 0].sum() if len(closed) else 0.0
    losses = -closed["PnL"][closed["PnL"] < 0].sum() if len(closed) else 0.0

    return {
        "total_return": growth - 1.0,
        "cagr": growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0,
        "volatility": volatility * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": mean / volatility * np.sqrt(SESSIONS_PER_YEAR) if volatility > 0 else float("nan"),
        "sortino": mean / downside * np.sqrt(SESSIONS_PER_YEAR) if downside > 0 else float("nan"),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "avg_invested": float(invested.loc[start:end].mean()),
        # One-way turnover: traded amount / average portfolio value, per year.
        "turnover": traded / average_value / 2.0 / years if average_value > 0 else float("nan"),
        "num_trades": int(len(period_orders)),
        "win_rate": float((closed["PnL"] > 0).mean()) if len(closed) else float("nan"),
        "profit_factor": float(wins / losses) if losses > 0 else float("nan"),
        "fees_paid": float(period_orders["Fees"].sum()) if len(orders) else 0.0,
    }


def yearly_returns(value: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Calendar-year returns within [start, end] (partial years at the edges)."""
    returns = value.pct_change().loc[start:end].fillna(0.0)
    return (1.0 + returns).groupby(returns.index.year).prod() - 1.0


def regime_returns(
    value: pd.Series, regimes: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Annualized return and volatility of the sessions in each regime.

    The return of session T (close T-1 -> close T) is attributed to the regime
    known at the close of T-1, which is the regime the positions were chosen under.
    """
    returns = value.pct_change().loc[start:end].fillna(0.0)
    known = regimes.reindex(value.index).shift(1).loc[start:end]
    rows = {}
    for regime, group in returns.groupby(known):
        rows[regime] = {
            "sessions": int(len(group)),
            "annual_return": float((1.0 + group).prod() ** (SESSIONS_PER_YEAR / len(group)) - 1.0),
            "volatility": float(group.std(ddof=0) * np.sqrt(SESSIONS_PER_YEAR)),
        }
    return pd.DataFrame(rows).T
