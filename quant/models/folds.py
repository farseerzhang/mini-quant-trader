"""Purged walk-forward folds plus the final holdout (proposal section 15).

Fold k trains on sessions before its validation window and validates on the next
``fold_length_months``. The label of session T uses opens up to T+1+h, so every
boundary is purged: a session is dropped from the earlier side if its label
window would reach the later side. With ``embargo_days = h + 1`` sessions dropped
before each boundary, the last kept session T has its exit open at boundary - 1.

Windows are anchored to the end of the development period and step back one
fold length at a time, so on the next retrain they move forward by themselves;
no date is hardcoded. A fold is kept only if its training window spans at least
``min_train_years``. The holdout (the last ``holdout_months``) is never part of a
fold; ``holdout_fold`` trains on all usable development sessions before it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.config.settings import ValidationSettings


@dataclass(frozen=True)
class Fold:
    number: int
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train[-1]

    @property
    def validation_start(self) -> pd.Timestamp:
        return self.validation[0]

    @property
    def validation_end(self) -> pd.Timestamp:
        return self.validation[-1]


def _before(sessions: pd.DatetimeIndex, boundary: pd.Timestamp, embargo: int) -> pd.DatetimeIndex:
    """Sessions before ``boundary`` whose label window ends before it: drop the last ``embargo``."""
    earlier = sessions[sessions < boundary]
    return earlier[: max(len(earlier) - embargo, 0)]


def development_sessions(sessions: pd.DatetimeIndex, holdout_start: pd.Timestamp, embargo: int) -> pd.DatetimeIndex:
    """Sessions usable for folds: before the holdout, purged so no label reaches into it."""
    return _before(sessions, holdout_start, embargo)


def walk_forward_folds(
    sessions: pd.DatetimeIndex,
    holdout_start: pd.Timestamp,
    validation: ValidationSettings,
    embargo: int,
) -> list[Fold]:
    """Expanding (or rolling) yearly folds over the development sessions, oldest first."""
    sessions = pd.DatetimeIndex(sessions).sort_values()
    usable = development_sessions(sessions, holdout_start, embargo)
    if len(usable) == 0:
        raise ValueError("no development sessions before the holdout")
    first = usable[0]
    length = pd.DateOffset(months=validation.fold_length_months)

    folds: list[Fold] = []
    remaining = usable
    while len(remaining):
        end = remaining[-1]
        window = remaining[remaining > end - length]
        train = _before(usable, window[0], embargo)
        # A fold needs min_train_years of history before it (from the first session).
        if len(train) == 0 or train[-1] < first + pd.DateOffset(years=validation.min_train_years):
            break
        if validation.scheme == "rolling_walk_forward":
            train = train[train > window[0] - pd.DateOffset(years=validation.rolling_years)]
        folds.append(Fold(number=0, train=train, validation=window))
        remaining = remaining[remaining < window[0]]

    folds.reverse()
    folds = [Fold(number=i + 1, train=f.train, validation=f.validation) for i, f in enumerate(folds)]
    if len(folds) < validation.min_folds:
        raise ValueError(
            f"only {len(folds)} walk-forward fold(s) fit before the holdout; retrain.validation.min_folds is "
            f"{validation.min_folds} (min_train_years {validation.min_train_years}, "
            f"fold_length_months {validation.fold_length_months}); more history is needed"
        )
    return folds


def holdout_fold(sessions: pd.DatetimeIndex, holdout_start: pd.Timestamp, embargo: int) -> Fold:
    """Train on every usable development session; validate on the holdout."""
    sessions = pd.DatetimeIndex(sessions).sort_values()
    return Fold(
        number=0,
        train=development_sessions(sessions, holdout_start, embargo),
        validation=sessions[sessions >= holdout_start],
    )


def sample_sessions(sessions: pd.DatetimeIndex, every: int, anchor: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Every ``every``-th session, counted on the positions of ``anchor`` (all sessions), so
    the same calendar sessions are chosen in every fold."""
    positions = anchor.get_indexer(sessions)
    if (positions < 0).any():
        raise ValueError("sessions must be a subset of the anchor sessions")
    return sessions[positions % every == 0]
