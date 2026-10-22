"""`quant sentiment scorecard`: the exploratory creator scorecard of S4 (sentiment_proposal.md 8.2, 12.6).

Scored: VERIFIED BULLISH / BEARISH opinions on pool stocks, extracted with extraction.backend,
evaluation.model and extraction.prompt_version, from videos published **up to the end of
evaluation.scored_until** (never after the S6 registration date: forward opinions are scored only by
S6, section 9.1). Exploration only: the creators were chosen today and videos may have been deleted,
so nothing here decides anything.

For each window of h sessions (evaluation.horizons) and opinion with effective date d:
- excess return = the stock's total return from the open of d to the open h sessions later, minus the
  equal-weight return of the pool stocks with a price over the same sessions; beta-adjusted alongside
  (stock return minus beta x SPY's, beta from 252 daily returns before d);
- signed by stance (+ BULLISH, - BEARISH); a hit is a positive signed excess return;
- counting: per creator, stock and stance, an opinion counts only if it starts at least h sessions
  after the last counted one;
- permutation test: each counted opinion keeps its creator, date and stance, and its stock is replaced
  by a random pool stock with a price over its window; p = (1 + draws at least as good) / (1 + draws),
  one-sided, uncorrected, per creator and pooled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, and_, select

from quant.backtest.data import Prices, load_prices
from quant.config.settings import S6_REGISTRATION_DATE, Settings
from quant.sentiment import tables

STANCE_SIGN = {"BULLISH": 1.0, "BEARISH": -1.0}
ALL_CREATORS = "all creators"
BETA_SESSIONS = 252
MIN_BETA_RETURNS = 200


def scored_until(settings: Settings) -> date:
    """The last publication day scored: evaluation.scored_until, never after the S6 registration date."""
    return min(settings.sentiment.evaluation.scored_until, S6_REGISTRATION_DATE)


def load_scored_opinions(engine: Engine, settings: Settings) -> pd.DataFrame:
    """Every VERIFIED opinion of the scored extraction published up to the end of ``scored_until``,
    with the creator's name; all stances and asset types (the descriptive tables use them all)."""
    o, c = tables.sentiment_opinions, tables.sentiment_channels
    config, evaluation = settings.sentiment.extraction, settings.sentiment.evaluation
    end = datetime.combine(scored_until(settings) + timedelta(days=1), datetime.min.time())
    stmt = (select(o.c.opinion_id, o.c.item_id, o.c.published_at, o.c.effective_date, o.c.ticker, o.c.asset_type,
                   o.c.stance, o.c.horizon, o.c.conviction, c.c.name.label("creator"))
            .join(c, c.c.channel_id == o.c.channel_id)
            .where(and_(o.c.source == "youtube", o.c.verification == "VERIFIED", o.c.backend == config.backend,
                        o.c.model == evaluation.model, o.c.prompt_version == config.prompt_version,
                        o.c.published_at < end)))
    with engine.connect() as conn:
        frame = pd.DataFrame(conn.execute(stmt).mappings().all(), columns=list(stmt.selected_columns.keys()))
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    frame["published_at"] = pd.to_datetime(frame["published_at"])
    return frame


def window_excess(opens: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per session and pool stock: the return from that open to the open ``horizon`` sessions later,
    minus the equal-weight return of the pool stocks with a price over the same window (NaN where the
    window is not complete or the stock has no price)."""
    returns = opens.shift(-horizon) / opens - 1.0
    return returns.sub(returns.mean(axis=1, skipna=True), axis=0)


def betas(closes: pd.DataFrame, market: pd.Series, session_positions: np.ndarray, stocks: list[str]) -> np.ndarray:
    """Beta of each (session position, stock) pair against SPY from the 252 daily returns before it
    (NaN with fewer than 200)."""
    returns = closes.pct_change()
    market_returns = market.pct_change()
    out = np.full(len(stocks), np.nan)
    for i, (position, stock) in enumerate(zip(session_positions, stocks)):
        start = max(position - BETA_SESSIONS, 1)
        pair = pd.concat([returns[stock].iloc[start:position], market_returns.iloc[start:position]], axis=1).dropna()
        if len(pair) >= MIN_BETA_RETURNS and pair.iloc[:, 1].var() > 0:
            out[i] = pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var()
    return out


def counted(opinions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The counting rule: per creator, stock and stance, keep an opinion only if it starts at least
    ``horizon`` sessions after the last kept one (``position`` = the effective date's session index)."""
    keep = []
    for _, group in opinions.sort_values(["position", "opinion_id"]).groupby(["creator", "ticker", "stance"],
                                                                            sort=False):
        last = None
        for index, position in zip(group.index, group["position"]):
            if last is None or position >= last + horizon:
                keep.append(index)
                last = position
    return opinions.loc[sorted(keep)]


@dataclass(frozen=True)
class HorizonScores:
    horizon: int
    table: pd.DataFrame  # creator (and "all creators") x n, hit rate, mean excess, mean beta excess, p values
    pending: int  # pool opinions whose window has not ended yet


def score_horizon(opinions: pd.DataFrame, prices: Prices, stocks: list[str], horizon: int, permutations: int,
                  seed: int) -> HorizonScores:
    opens = prices.open.loc[:, stocks]
    excess = window_excess(opens, horizon)
    sessions = opens.index
    frame = opinions.copy()
    frame["position"] = sessions.get_indexer(frame["effective_date"])
    after_prices = frame["effective_date"] > sessions[-1]
    pending = int(after_prices.sum() + ((frame["position"] >= 0) & (frame["position"] + horizon >= len(sessions))).sum())
    frame = frame[frame["position"] >= 0]
    rows = excess.to_numpy()
    complete = np.array([np.isfinite(rows[p, stocks.index(t)]) if p + horizon < len(sessions) else False
                         for p, t in zip(frame["position"], frame["ticker"])], dtype=bool)
    frame = counted(frame[complete], horizon)
    columns = ["opinions", "hit rate", "mean excess", "mean beta-adj. excess", "p (mean)", "p (hit rate)"]
    if frame.empty:
        return HorizonScores(horizon, pd.DataFrame(columns=columns).rename_axis("creator"), pending)

    positions = frame["position"].to_numpy()
    tickers = frame["ticker"].tolist()
    signs = frame["stance"].map(STANCE_SIGN).to_numpy()
    observed = np.array([rows[p, stocks.index(t)] for p, t in zip(positions, tickers)]) * signs
    stock_returns = (opens.shift(-horizon) / opens - 1.0).to_numpy()
    market = prices.open["SPY"]
    market_returns = (market.shift(-horizon) / market - 1.0).to_numpy()
    beta = betas(prices.close.loc[:, stocks], prices.close["SPY"], positions, tickers)
    beta_excess = np.array([stock_returns[p, stocks.index(t)] for p, t in zip(positions, tickers)]) \
        - beta * market_returns[positions]
    beta_excess = beta_excess * signs

    groups = {name: np.flatnonzero(frame["creator"].to_numpy() == name) for name in sorted(frame["creator"].unique())}
    groups[ALL_CREATORS] = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    sums = {name: np.zeros(permutations) for name in groups}
    hits = {name: np.zeros(permutations) for name in groups}
    member = [[creator, ALL_CREATORS] for creator in frame["creator"]]
    for i, position in enumerate(positions):
        available = rows[position][np.isfinite(rows[position])]
        draws = available[rng.integers(0, len(available), size=permutations)] * signs[i]
        for name in member[i]:
            sums[name] += draws
            hits[name] += draws > 0

    table = {}
    for name, idx in groups.items():
        n = len(idx)
        mean, hit = observed[idx].mean(), (observed[idx] > 0).mean()
        table[name] = {
            "opinions": n,
            "hit rate": hit,
            "mean excess": mean,
            "mean beta-adj. excess": np.nanmean(beta_excess[idx]) if np.isfinite(beta_excess[idx]).any() else np.nan,
            "p (mean)": (1 + np.sum(sums[name] / n >= mean - 1e-12)) / (1 + permutations),
            "p (hit rate)": (1 + np.sum(hits[name] / n >= hit - 1e-12)) / (1 + permutations),
        }
    return HorizonScores(horizon, pd.DataFrame(table).T.loc[:, columns].rename_axis("creator"), pending)


@dataclass(frozen=True)
class Scorecard:
    scored_until: date
    first_published: pd.Timestamp | None
    prices_until: pd.Timestamp
    opinions: pd.DataFrame  # every scored-extraction VERIFIED opinion up to scored_until
    horizons: list[HorizonScores]


def build_scorecard(engine: Engine, settings: Settings) -> Scorecard:
    evaluation = settings.sentiment.evaluation
    stocks = list(settings.market.stocks)
    opinions = load_scored_opinions(engine, settings)
    prices = load_prices(engine, [*stocks, "SPY"])
    pool = opinions[(opinions["asset_type"] == "STOCK") & opinions["ticker"].isin(stocks)
                    & opinions["stance"].isin(list(STANCE_SIGN))]
    horizons = [score_horizon(pool, prices, stocks, h, evaluation.permutations, evaluation.seed)
                for h in evaluation.horizons]
    first = opinions["published_at"].min() if not opinions.empty else None
    return Scorecard(scored_until(settings), first, prices.close.index[-1], opinions, horizons)


def _format(table: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Numbers only where at least ``min_samples`` opinions count; counts everywhere."""
    out = table.copy().astype(object)
    for name, row in table.iterrows():
        small = row["opinions"] < min_samples
        out.loc[name, "opinions"] = int(row["opinions"])
        for key in ("hit rate", "mean excess", "mean beta-adj. excess"):
            out.loc[name, key] = "-" if small or pd.isna(row[key]) else f"{row[key]:.1%}"
        for key in ("p (mean)", "p (hit rate)"):
            out.loc[name, key] = "-" if small else f"{row[key]:.3f}"
    return out


def describe_creators(opinions: pd.DataFrame, pool: set[str]) -> pd.DataFrame:
    """Per creator: opinions, the bullish share of directional calls, and where the calls go (descriptive)."""
    rows = {}
    for creator, group in opinions.groupby("creator"):
        directional = group[group["stance"].isin(list(STANCE_SIGN))]
        rows[creator] = {
            "videos": group["item_id"].nunique(),
            "opinions": len(group),
            "bullish share": f"{(directional['stance'] == 'BULLISH').mean():.0%}" if len(directional) else "-",
            "pool stocks": int((group["ticker"].isin(pool) & (group["asset_type"] == "STOCK")).sum()),
            "other stocks": int((~group["ticker"].isin(pool) & (group["asset_type"] == "STOCK")).sum()),
            "ETFs": int((group["asset_type"] == "ETF").sum()),
        }
    return pd.DataFrame(rows).T.rename_axis("creator")


def _table(frame: pd.DataFrame) -> list[str]:
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join([frame.index.name or ""] + columns) + " |", "|" + "---|" * (len(columns) + 1)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in frame.iterrows()]
    return lines


def write_scorecard(card: Scorecard, settings: Settings, now: datetime) -> Path:
    evaluation = settings.sentiment.evaluation
    pool = set(settings.market.stocks)
    opinions = card.opinions
    first = f"{card.first_published:%Y-%m-%d}" if card.first_published is not None else "-"
    lines = [
        f"# Creator scorecard, exploratory ({now:%Y-%m-%d %H:%M} UTC)",
        "",
        f"S4 (sentiment_proposal.md 8.2, 12.6). Opinions published {first} .. {card.scored_until} (never after the "
        f"S6 registration date {S6_REGISTRATION_DATE}: forward opinions are scored only by S6), extracted with "
        f"{settings.sentiment.extraction.backend} / {evaluation.model} / prompt version "
        f"{settings.sentiment.extraction.prompt_version}; prices to {card.prices_until.date()}.",
        "",
        "**Exploration only, decides nothing.** The creators were chosen in 2026 and may still be around because "
        "they were right; losing calls may have been deleted. P-values are one-sided and uncorrected: with "
        f"{len(evaluation.horizons)} windows, two measures and every creator, several small p-values are expected "
        f"by chance. Cells with fewer than {evaluation.min_samples} counted opinions show the count only.",
        "",
    ]
    for scores in card.horizons:
        lines += [f"## {scores.horizon}-session window", "",
                  "Signed excess return against the pool's equal weight (bullish +, bearish -); hit = positive. "
                  f"{scores.pending} pool opinion(s) whose window has not ended yet are left out.", ""]
        lines += _table(_format(scores.table, evaluation.min_samples)) if not scores.table.empty \
            else ["No counted opinions yet."]
        lines.append("")
    lines += ["## Creators (descriptive)", ""]
    lines += _table(describe_creators(opinions, pool)) if not opinions.empty else ["No opinions."]
    named = opinions[opinions["asset_type"] == "STOCK"]["ticker"].value_counts().head(15)
    lines += ["", "## Stocks named most often (descriptive)", ""]
    lines += _table(pd.DataFrame({"opinions": named, "in pool": ["yes" if t in pool else "no" for t in named.index]})
                    .rename_axis("ticker")) if len(named) else ["None."]
    for column, title in (("stance", "Stances"), ("horizon", "Horizons"), ("conviction", "Conviction")):
        counts = opinions[column].value_counts().sort_index()
        lines += ["", f"## {title} (descriptive)", ""]
        lines += _table(pd.DataFrame({"opinions": counts}).rename_axis(column)) if len(counts) else ["None."]
    lines += ["", "Research output, not advice."]
    directory = Path(settings.sentiment.summary.reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"scorecard-{now:%Y-%m-%d}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
