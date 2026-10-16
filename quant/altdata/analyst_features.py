"""A4 round 2: the three analyst features (altdata_proposal.md section 8.2), computed in memory.

Per pool stock and NYSE session T, from the changes of the 2026-10-16 download with
`available_on` <= T; one change per firm, stock and session (the latest by timestamp):

- target_revision_20d: mean of ln(current / prior target), clipped to +/- ln 2, over the changes
  with both targets that became available in the 20 sessions up to T; 0 if none. Changes
  available within 5 sessions after a split of the stock are left out.
- net_rating_changes_20d: `up` minus `down` actions in the 20 sessions up to T; 0 if none.
- target_upside: median over firms of each firm's latest target (from a change available in the
  250 sessions up to T), split-adjusted to T, divided by the as-traded close of T, minus 1;
  NULL if no firm has a target in the window.

All three are NULL before the stock's first change in the table.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, and_, select

from quant.altdata.tables import altdata_analyst_changes
from quant.config.settings import Settings
from quant.persistence import queries, tables

AS_OF = date(2026, 10, 16)  # the download the research uses (8.2 (b))
FEATURES = ("target_revision_20d", "net_rating_changes_20d", "target_upside")
WINDOW, TARGET_WINDOW, AFTER_SPLIT = 20, 250, 5
CLIP = np.log(2.0)


def load_changes(engine: Engine, as_of: date = AS_OF) -> pd.DataFrame:
    """That download's view, one change per firm, stock and session (the latest by timestamp)."""
    a = altdata_analyst_changes
    stmt = select(a.c.symbol, a.c.grade_time, a.c.firm, a.c.action, a.c.current_target, a.c.prior_target,
                  a.c.available_on).where(and_(a.c.first_seen <= as_of, a.c.last_seen >= as_of))
    with engine.connect() as conn:
        frame = pd.DataFrame(conn.execute(stmt).mappings().all(),
                             columns=["symbol", "grade_time", "firm", "action", "current_target", "prior_target",
                                      "available_on"])
    frame["available_on"] = pd.to_datetime(frame["available_on"])
    frame = frame.sort_values("grade_time")
    return frame.drop_duplicates(["symbol", "firm", "available_on"], keep="last").reset_index(drop=True)


def load_splits(engine: Engine, symbols: list[str]) -> pd.DataFrame:
    ca = tables.corporate_actions
    with engine.connect() as conn:
        rows = conn.execute(select(ca.c.symbol, ca.c.ex_date, ca.c.value).where(and_(
            ca.c.action_type == "SPLIT", ca.c.symbol.in_(symbols)))).all()
    frame = pd.DataFrame(rows, columns=["symbol", "ex_date", "ratio"])
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    return frame


def split_factor(splits: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.Series:
    """Cumulative product of one stock's split ratios with ex-date <= each session (1 before any)."""
    ratio = pd.Series(1.0, index=sessions)
    for ex_date, value in zip(splits["ex_date"], splits["ratio"]):
        position = sessions.searchsorted(ex_date)
        if position < len(sessions):
            ratio.iloc[position] *= value
    return ratio.cumprod()


def stock_features(changes: pd.DataFrame, splits: pd.DataFrame, close: pd.Series,
                   sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """The three features of one stock on every session (columns FEATURES)."""
    out = pd.DataFrame(np.nan, index=sessions, columns=list(FEATURES))
    changes = changes[changes["available_on"].isin(sessions)]
    if changes.empty:
        return out
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    started = sessions >= changes["available_on"].min()

    # 1. Target revisions, without the changes just after a split.
    factor = split_factor(splits, sessions)
    split_positions = [p for p in (sessions.searchsorted(d) for d in splits["ex_date"]) if p < len(sessions)]
    pos = position.loc[changes["available_on"]].to_numpy()
    near_split = np.zeros(len(changes), dtype=bool)
    for p in split_positions:
        near_split |= (pos >= p) & (pos < p + AFTER_SPLIT)
    both = changes["current_target"].notna() & changes["prior_target"].notna() & ~near_split
    revisions = np.clip(np.log(changes.loc[both, "current_target"] / changes.loc[both, "prior_target"]), -CLIP, CLIP)
    day_sum = revisions.groupby(changes.loc[both, "available_on"]).sum().reindex(sessions, fill_value=0.0)
    day_count = both[both].groupby(changes.loc[both, "available_on"]).size().reindex(sessions, fill_value=0)
    total, count = day_sum.rolling(WINDOW, min_periods=1).sum(), day_count.rolling(WINDOW, min_periods=1).sum()
    out["target_revision_20d"] = (total / count.where(count > 0)).fillna(0.0).where(started)

    # 2. Net rating changes.
    signed = changes["action"].map({"up": 1, "down": -1}).fillna(0)
    net = signed.groupby(changes["available_on"]).sum().reindex(sessions, fill_value=0)
    out["net_rating_changes_20d"] = net.rolling(WINDOW, min_periods=1).sum().where(started).astype(float)

    # 3. Upside to the median of each firm's latest target, split-adjusted to T.
    targets = changes[changes["current_target"].notna()]
    if len(targets):
        base = targets["current_target"].to_numpy() * factor.loc[targets["available_on"]].to_numpy()
        latest = (pd.DataFrame({"session": targets["available_on"].to_numpy(), "firm": targets["firm"].to_numpy(),
                                "base": base})
                  .pivot_table(index="session", columns="firm", values="base", aggfunc="last")
                  .reindex(sessions).ffill(limit=TARGET_WINDOW - 1))
        median = latest.median(axis=1) / factor
        out["target_upside"] = (median / close.reindex(sessions) - 1.0).where(started)
    return out


def compute_analyst_features(engine: Engine, settings: Settings, sessions: pd.DatetimeIndex,
                             as_of: date = AS_OF) -> dict[str, pd.DataFrame]:
    """name -> sessions x symbols, for the pool stocks."""
    stocks = list(settings.market.stocks)
    changes = load_changes(engine, as_of)
    splits = load_splits(engine, stocks)
    closes = queries.load_raw_closes(engine, stocks)
    closes.index = pd.DatetimeIndex(closes.index)
    per_stock = {}
    for symbol in stocks:
        per_stock[symbol] = stock_features(changes[changes["symbol"] == symbol], splits[splits["symbol"] == symbol],
                                           closes[symbol] if symbol in closes else pd.Series(dtype=float), sessions)
    return {name: pd.DataFrame({s: f[name] for s, f in per_stock.items()}, index=sessions) for name in FEATURES}


def percentile_ranks(feature: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 inputs (8.2 (c)): each session's cross-sectional percentile rank, ties averaged."""
    return feature.rank(axis=1, pct=True, method="average")
