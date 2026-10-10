"""The pooled training dataset: one row per (session, stock) (proposal sections 11, 14).

Inputs at T are the stock's stored features, the date's market features and the
rule-based regime at T as three indicator columns (BULL / SIDEWAYS / BEAR); all
are point-in-time. The target columns (label, excess, threshold) look past T and
are never inputs. Benchmarks and indicators (SPY, QQQ, ^VIX) are not rows: the
pool stocks are what the model selects among. Rows with any missing input
(warm-up) are dropped; rows whose label is not known yet are kept, so the same
rows can be scored for live predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine

from quant.backtest.data import load_prices
from quant.config.settings import Settings
from quant.features.technical import FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
from quant.models.labels import Labels, make_labels
from quant.persistence import queries
from quant.regime.pipeline import RULE_METHOD
from quant.regime.rule_based import BEAR, BULL, SIDEWAYS

REGIME_COLUMNS = ("regime_bull", "regime_sideways", "regime_bear")
INPUT_COLUMNS = (*FEATURE_COLUMNS, *MARKET_FEATURE_COLUMNS, *REGIME_COLUMNS)
TARGET_COLUMNS = ("label", "excess", "threshold")


@dataclass(frozen=True)
class Dataset:
    """``frame`` is indexed by (date, symbol) with INPUT_COLUMNS then TARGET_COLUMNS."""

    frame: pd.DataFrame

    @property
    def inputs(self) -> pd.DataFrame:
        return self.frame.loc[:, list(INPUT_COLUMNS)]

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index.get_level_values("date").unique()).sort_values()

    def labelled(self) -> "Dataset":
        """Only rows whose label is known (for training and scoring)."""
        return Dataset(self.frame[self.frame["label"].notna()])

    def rows_on(self, sessions: pd.DatetimeIndex) -> "Dataset":
        dates = self.frame.index.get_level_values("date")
        return Dataset(self.frame[dates.isin(sessions)])


def _long(frame: pd.DataFrame, name: str) -> pd.Series:
    """dates x symbols -> Series indexed by (date, symbol), keeping missing values."""
    wide = frame.rename_axis(index="date", columns="symbol")
    return wide.reset_index().melt(id_vars="date", var_name="symbol", value_name=name).set_index(["date", "symbol"])[name]


def build_dataset(
    features: pd.DataFrame, market: pd.DataFrame, regimes: pd.Series, labels: Labels
) -> Dataset:
    """Assemble rows from point-in-time inputs and the labels (pure; no database)."""
    dates = features.index.get_level_values("date")
    frame = features.loc[:, list(FEATURE_COLUMNS)].copy()
    frame = frame.join(market.loc[:, list(MARKET_FEATURE_COLUMNS)], on="date")

    regime = regimes.reindex(dates).to_numpy()
    for column, name in zip(REGIME_COLUMNS, (BULL, SIDEWAYS, BEAR)):
        frame[column] = (regime == name).astype(float)
    has_regime = pd.notna(regime)

    for column in TARGET_COLUMNS:
        frame[column] = _long(getattr(labels, column), column).reindex(frame.index)

    complete = frame.loc[:, list(INPUT_COLUMNS)].notna().all(axis=1) & has_regime
    return Dataset(frame[complete].sort_index())


def load_dataset(engine: Engine, settings: Settings) -> Dataset:
    """Every pool stock's rows from the stored features, regimes and prices."""
    stocks = list(settings.market.stocks)
    benchmark = settings.model.target.benchmark
    features = queries.load_features_long(engine, settings.features.feature_version, stocks, FEATURE_COLUMNS)
    market = queries.load_market_features(engine, settings.features.market_feature_version, MARKET_FEATURE_COLUMNS)
    regimes = queries.load_regimes(engine, RULE_METHOD, settings.regime.rule_version)
    if features.empty or market.empty or regimes.empty:
        raise ValueError("features or regimes are missing; run `python -m quant run-daily` first")
    prices = load_prices(engine, [*stocks, benchmark])
    labels = make_labels(prices.open[stocks], prices.close[stocks], prices.open[benchmark], settings.model.target)
    return build_dataset(features, market, regimes, labels)
