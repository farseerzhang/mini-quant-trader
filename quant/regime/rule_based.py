"""Rule-based BULL / SIDEWAYS / BEAR regime with hysteresis (proposal section 12).

The score at T is a sum of five -1/0/+1 inputs taken from the stored features
at T. The regime is a forward-only state machine over those scores, so the
regime at T depends only on sessions up to and including T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config.settings import RULE_SCORE_MAX, RegimeSettings

BULL = "BULL"
SIDEWAYS = "SIDEWAYS"
BEAR = "BEAR"

# Inputs scored by their sign (section 12.1, inputs 1-4).
SIGN_INPUTS = ("spy_trend", "qqq_trend", "spy_sma50_vs_sma200", "spy_return_3m")
INPUT_COLUMNS = (*SIGN_INPUTS, "vix")
REGIME_COLUMNS = ("regime", "regime_score", "confidence")


def score_inputs(inputs: pd.DataFrame, settings: RegimeSettings) -> pd.Series:
    """Integer score from -5 to +5 per date; NaN where any input is missing.

    A missing input is never scored as 0, because that would silently shrink
    the score's range.
    """
    missing = [column for column in INPUT_COLUMNS if column not in inputs]
    if missing:
        raise ValueError(f"regime inputs missing columns: {', '.join(missing)}")

    values = inputs.loc[:, INPUT_COLUMNS].astype(float)
    score = np.sign(values.loc[:, SIGN_INPUTS]).sum(axis=1)
    vix = values["vix"]
    score += np.where(vix < settings.vix_low, 1, np.where(vix > settings.vix_high, -1, 0))
    return score.where(values.notna().all(axis=1))


def _proposed_state(current: str, score: int, enter: float, exit_: float) -> str:
    if score >= enter:
        return BULL
    if score <= -enter:
        return BEAR
    if current == BULL and score < exit_:
        return SIDEWAYS
    if current == BEAR and score > -exit_:
        return SIDEWAYS
    return current


def classify_regimes(scores: pd.Series, settings: RegimeSettings) -> pd.DataFrame:
    """Run the hysteresis and minimum-dwell state machine over scores in date order.

    Returns one row per scorable date (dates with a NaN score are left out; the
    state carries over them and they do not count towards the dwell time).
    """
    scores = scores.sort_index().dropna()
    state = SIDEWAYS
    # The first scorable session is exempt from the dwell rule.
    held = settings.min_dwell_days
    regimes: list[str] = []
    for score in scores.astype(int):
        proposed = _proposed_state(state, score, settings.enter_threshold, settings.exit_threshold)
        if proposed != state and held >= settings.min_dwell_days:
            state = proposed
            held = 0
        held += 1
        regimes.append(state)

    result = pd.DataFrame(index=scores.index)
    result["regime"] = regimes
    result["regime_score"] = scores.astype(int)
    result["confidence"] = scores.abs() / RULE_SCORE_MAX
    return result.loc[:, REGIME_COLUMNS]


def calculate_rule_regimes(inputs: pd.DataFrame, settings: RegimeSettings) -> pd.DataFrame:
    """Score the inputs and classify them: the full rule-based regime history."""
    return classify_regimes(score_inputs(inputs, settings), settings)


def average_dwell(regimes: pd.Series) -> float:
    """Average length, in sessions, of consecutive runs of the same regime."""
    if regimes.empty:
        return 0.0
    runs = int((regimes != regimes.shift()).sum())
    return len(regimes) / runs
