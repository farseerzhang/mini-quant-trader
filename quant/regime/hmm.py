"""Hidden Markov model regime: point-in-time (filtered) state probabilities (proposal section 13).

hmmlearn's ``predict`` (Viterbi) and ``predict_proba`` (smoothed posteriors) use
the whole observation sequence, including sessions after T, so they must never
be used for the regime at T. The forward filter below uses only observations up
to and including T:

    alpha_0  ~  start * b(x_0)
    alpha_t  ~  (alpha_{t-1} @ transition) * b(x_t)        (normalized to sum to 1)

where b(x) is each state's Gaussian density of the observation. It runs in log
space, so long histories do not underflow, and needs only the four public
fitted parameters (hmmlearn is used for fitting alone).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from quant.features.technical import MARKET_FEATURE_COLUMNS
from quant.persistence import queries


@dataclass(frozen=True)
class HMMParameters:
    """A fitted Gaussian HMM: ``covars`` are full matrices, one per state."""

    start: np.ndarray  # (states,)
    transition: np.ndarray  # (states, states); rows sum to 1
    means: np.ndarray  # (states, features)
    covars: np.ndarray  # (states, features, features)

    @property
    def states(self) -> int:
        return len(self.start)

    @classmethod
    def from_hmmlearn(cls, model) -> "HMMParameters":
        """Copy the public parameters of a fitted ``hmmlearn.hmm.GaussianHMM``."""
        return cls(
            start=np.asarray(model.startprob_, dtype=float),
            transition=np.asarray(model.transmat_, dtype=float),
            means=np.asarray(model.means_, dtype=float),
            covars=np.asarray(model.covars_, dtype=float),
        )


def state_log_likelihood(parameters: HMMParameters, observations: np.ndarray) -> np.ndarray:
    """log b_k(x_t) for every session t and state k: (sessions, states)."""
    observations = np.atleast_2d(np.asarray(observations, dtype=float))
    return np.column_stack(
        [
            multivariate_normal(mean=parameters.means[k], cov=parameters.covars[k], allow_singular=True).logpdf(observations)
            .reshape(len(observations))
            for k in range(parameters.states)
        ]
    )


def filtered_probabilities(parameters: HMMParameters, observations: np.ndarray) -> np.ndarray:
    """P(state at t | observations 0..t) for every session t: (sessions, states).

    Row t depends only on observations 0..t, so appending later sessions never
    changes it.
    """
    log_b = state_log_likelihood(parameters, observations)
    with np.errstate(divide="ignore"):
        log_start = np.log(parameters.start)
        log_transition = np.log(parameters.transition)
    log_alpha = np.empty_like(log_b)
    current = log_start + log_b[0]
    log_alpha[0] = current - logsumexp(current)
    for t in range(1, len(log_b)):
        current = logsumexp(log_alpha[t - 1][:, None] + log_transition, axis=0) + log_b[t]
        log_alpha[t] = current - logsumexp(current)
    return np.exp(log_alpha)


# --- Fitting (section 13.1) --------------------------------------------------------------

LABELS_BY_MEAN_RETURN = ("BULL", "SIDEWAYS", "BEAR")  # highest -> lowest mean of the return observation


@dataclass(frozen=True)
class FittedHMM:
    """One fit: its standardization, parameters and state -> regime label mapping."""

    columns: tuple[str, ...]
    mean: np.ndarray  # training-window mean of each observation
    std: np.ndarray  # training-window standard deviation
    parameters: HMMParameters
    labels: tuple[str, ...]  # labels[k] = regime of state k
    log_likelihood: float  # per training session, of the kept start
    start_log_likelihoods: tuple[float, ...]  # every start, for the record
    train_start: pd.Timestamp
    train_end: pd.Timestamp

    def standardize(self, observations: pd.DataFrame) -> np.ndarray:
        return (observations.loc[:, list(self.columns)].to_numpy(dtype=float) - self.mean) / self.std

    def filtered(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Filtered probability of each regime per session (columns BULL, SIDEWAYS, BEAR)."""
        probabilities = filtered_probabilities(self.parameters, self.standardize(observations))
        frame = pd.DataFrame(probabilities, index=observations.index, columns=list(self.labels))
        return frame.loc[:, list(LABELS_BY_MEAN_RETURN)]

    def describe(self) -> pd.DataFrame:
        """Each regime's mean and standard deviation of every observation, in original units."""
        rows = {}
        for k, label in enumerate(self.labels):
            means = self.parameters.means[k] * self.std + self.mean
            sds = np.sqrt(np.diag(self.parameters.covars[k])) * self.std
            rows[label] = {**{f"{c} mean": m for c, m in zip(self.columns, means)},
                           **{f"{c} sd": d for c, d in zip(self.columns, sds)}}
        rows = pd.DataFrame(rows).T.loc[list(LABELS_BY_MEAN_RETURN)]
        stay = {label: self.parameters.transition[k, k] for k, label in enumerate(self.labels)}
        rows["stay probability"] = pd.Series(stay)
        rows["expected sessions"] = 1 / (1 - rows["stay probability"])
        return rows


def fit_hmm(observations: pd.DataFrame, settings, seed: int) -> FittedHMM:
    """Fit on ``observations`` (the training window only): standardize with its own mean and
    standard deviation, run ``settings.starts`` random starts, keep the highest log-likelihood,
    and label the states by their mean of the first observation (the return)."""
    columns = tuple(settings.observations)
    data = observations.loc[:, list(columns)].dropna()
    if len(data) < 50 * settings.states:
        raise ValueError(f"too few sessions to fit an HMM: {len(data)}")
    mean, std = data.mean().to_numpy(), data.std(ddof=0).to_numpy()
    x = (data.to_numpy(dtype=float) - mean) / std

    fits = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # hmmlearn on numpy 2.5 (section 13.1)
        for start in range(settings.starts):
            model = GaussianHMM(
                n_components=settings.states, covariance_type=settings.covariance,
                n_iter=settings.iterations, random_state=seed + start,
            ).fit(x)
            fits.append((model.score(x) / len(x), model))
    scores = tuple(score for score, _ in fits)
    best_score, best = max(fits, key=lambda item: item[0])  # first of equal scores
    parameters = HMMParameters.from_hmmlearn(best)

    order = np.argsort(-parameters.means[:, 0], kind="stable")  # highest mean return first
    labels = [""] * settings.states
    for rank, state in enumerate(order):
        labels[state] = LABELS_BY_MEAN_RETURN[rank]
    return FittedHMM(
        columns=columns, mean=mean, std=std, parameters=parameters, labels=tuple(labels),
        log_likelihood=float(best_score), start_log_likelihoods=scores,
        train_start=data.index[0], train_end=data.index[-1],
    )


def load_observations(engine, settings) -> pd.DataFrame:
    """The configured HMM observations per session from ``market_features``."""
    columns = list(settings.regime.hmm.observations)
    unknown = [c for c in columns if c not in MARKET_FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"regime.hmm.observations: not market features: {', '.join(unknown)}")
    frame = queries.load_market_features(engine, settings.features.market_feature_version, columns)
    return frame.dropna()
