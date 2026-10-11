"""Walk-forward HMM regime: refit yearly, filter forward, label, dwell (proposal sections 13, 13.1).

For each fold the HMM is fit on that fold's training sessions only. Its filtered
probabilities for the fold's validation sessions come from a forward pass that
starts at the fit's first training session and ends at the session itself, so the
regime at T uses observations up to T and the parameters of a fit made before T.
The stitched validation windows give one regime per session; the minimum dwell of
the rule-based regime (``regime.min_dwell_days``) is then applied over the whole
stitched sequence. The holdout is never part of a fold.

Mapping stability: two consecutive fits label the sessions they both cover (the
earlier fit's training window) by their most probable filtered state; the share
of sessions with the same label is their agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.config.settings import Settings
from quant.models.folds import Fold
from quant.regime.hmm import FittedHMM, fit_hmm

# Pre-registered (section 13.1): consecutive fits must agree on at least this share.
MIN_MAPPING_AGREEMENT = 0.80


@dataclass(frozen=True)
class HMMWalkForward:
    fits: list[FittedHMM]
    folds: list[Fold]
    # Validation sessions: filtered probability of each regime, and the fold that made it.
    probabilities: pd.DataFrame
    fold_of_session: pd.Series
    raw_regime: pd.Series  # most probable filtered state, before the dwell rule
    regime: pd.Series  # after the minimum dwell
    agreement: pd.Series  # per consecutive pair of fits (indexed by the later fold)


def apply_min_dwell(proposed: pd.Series, min_dwell_days: int) -> pd.Series:
    """Hold each regime at least ``min_dwell_days`` sessions before it may change.

    Same rule as the rule-based regime (section 12.2): the first session is exempt; a
    change is taken only once the current regime has been held long enough; a
    suppressed change is not queued.
    """
    state, held, out = None, 0, []
    for value in proposed:
        if state is None:
            state, held = value, 0
        elif value != state and held >= min_dwell_days:
            state, held = value, 0
        held += 1
        out.append(state)
    return pd.Series(out, index=proposed.index, name="regime", dtype=object)


def mapping_agreement(earlier: FittedHMM, later: FittedHMM, observations: pd.DataFrame) -> float:
    """Share of the earlier fit's training sessions both fits give the same label."""
    common = observations.loc[earlier.train_start : earlier.train_end]
    first = earlier.filtered(common).idxmax(axis=1)
    second = later.filtered(observations.loc[later.train_start : earlier.train_end]).idxmax(axis=1)
    return float((first == second.reindex(first.index)).mean())


def run_hmm_walk_forward(observations: pd.DataFrame, folds: list[Fold], settings: Settings) -> HMMWalkForward:
    """Fit on each fold's training sessions, filter its validation sessions, stitch, dwell."""
    hmm = settings.regime.hmm
    seed = settings.retrain.search.seed
    fits, parts, owners = [], [], []
    for fold in folds:
        train = observations.loc[observations.index.isin(fold.train)]
        fit = fit_hmm(train, hmm, seed)
        path = observations.loc[fit.train_start : fold.validation_end]  # forward pass up to each session
        probabilities = fit.filtered(path)
        validation = probabilities.loc[probabilities.index.isin(fold.validation)]
        fits.append(fit)
        parts.append(validation)
        owners.append(pd.Series(fold.number, index=validation.index))

    probabilities = pd.concat(parts)
    raw = probabilities.idxmax(axis=1).rename("raw_regime")
    agreement = pd.Series(
        {folds[i].number: mapping_agreement(fits[i - 1], fits[i], observations) for i in range(1, len(fits))},
        name="agreement",
    )
    return HMMWalkForward(
        fits=fits,
        folds=folds,
        probabilities=probabilities,
        fold_of_session=pd.concat(owners).rename("fold"),
        raw_regime=raw,
        regime=apply_min_dwell(raw, settings.regime.min_dwell_days),
        agreement=agreement,
    )
