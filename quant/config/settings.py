"""Load and validate application settings from config/settings.yaml (proposal section 27)."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

DEFAULT_CONFIG_PATH = Path("config") / "settings.yaml"


class ConfigError(ValueError):
    """Raised when the settings file is missing or invalid."""


@dataclass(frozen=True)
class DatabaseSettings:
    url: str
    busy_timeout_ms: int

    def __post_init__(self) -> None:
        try:
            parsed = make_url(self.url)
        except ArgumentError as exc:
            raise ConfigError(f"database.url is not a valid database URL: {self.url!r}") from exc
        if parsed.get_backend_name() != "sqlite" or not parsed.database:
            raise ConfigError(f"database.url must look like 'sqlite:///path/to/file.db', got {self.url!r}")
        if self.busy_timeout_ms < 0:
            raise ConfigError("database.busy_timeout_ms must not be negative")

    @property
    def path(self) -> str:
        return make_url(self.url).database


# The 11 GICS sectors; market.symbols is keyed by these names.
GICS_SECTORS = (
    "energy",
    "materials",
    "industrials",
    "consumer_discretionary",
    "consumer_staples",
    "health_care",
    "financials",
    "information_technology",
    "communication_services",
    "utilities",
    "real_estate",
)


@dataclass(frozen=True)
class MarketSettings:
    provider: str
    request_timeout_seconds: float
    calendar: str
    initial_history: date
    timezone: str
    execution: str
    # Stock pool by GICS sector, in config order: sector -> stocks.
    symbols: Mapping[str, tuple[str, ...]]
    market_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.request_timeout_seconds <= 0:
            raise ConfigError("market.request_timeout_seconds must be positive")
        if not (self.stocks or self.market_symbols):
            raise ConfigError("market.symbols and market.market_symbols must not both be empty")
        both = sorted(set(self.stocks) & set(self.market_symbols))
        if both:
            raise ConfigError(
                f"{', '.join(both)} listed in both market.symbols and market.market_symbols; "
                "the stock pool holds tradable stocks, market_symbols holds benchmarks and indicators"
            )

    @property
    def stocks(self) -> tuple[str, ...]:
        """The stock pool as one flat tuple, in config order."""
        return tuple(symbol for members in self.symbols.values() for symbol in members)

    @property
    def sectors(self) -> dict[str, str]:
        """Each pool stock's GICS sector (today's classification, not point-in-time)."""
        return {symbol: sector for sector, members in self.symbols.items() for symbol in members}

    @property
    def universe(self) -> list[str]:
        """Every symbol to keep up to date: the stock pool plus market-level symbols."""
        return list(dict.fromkeys((*self.stocks, *self.market_symbols)))


@dataclass(frozen=True)
class FeatureSettings:
    version: int
    sma: tuple[int, int, int]
    rsi_period: int
    atr_period: int

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ConfigError("features.version must be >= 1")
        if len(self.sma) != 3 or self.sma[0] < 2 or list(self.sma) != sorted(set(self.sma)):
            raise ConfigError("features.sma must be three increasing windows, each >= 2")
        if self.rsi_period < 2 or self.atr_period < 2:
            raise ConfigError("features.rsi_period and features.atr_period must be >= 2")

    @property
    def feature_version(self) -> str:
        return f"technical-v{self.version}"

    @property
    def market_feature_version(self) -> str:
        return f"market-v{self.version}"


# The rule-based regime sums five inputs scored -1, 0 or +1 (proposal section 12).
RULE_SCORE_MAX = 5


HMM_COVARIANCES = ("diag", "full")


@dataclass(frozen=True)
class HMMSettings:
    observations: tuple[str, ...]  # market_features columns
    states: int
    covariance: str
    starts: int  # random starts; the best training log-likelihood is kept
    iterations: int

    def __post_init__(self) -> None:
        if not self.observations:
            raise ConfigError("regime.hmm.observations must list at least one market feature")
        if self.states != 3:
            raise ConfigError("regime.hmm.states must be 3 (BULL / SIDEWAYS / BEAR)")
        if self.covariance not in HMM_COVARIANCES:
            raise ConfigError(f"regime.hmm.covariance must be one of {', '.join(HMM_COVARIANCES)}")
        if self.starts < 1 or self.iterations < 1:
            raise ConfigError("regime.hmm.starts and regime.hmm.iterations must be >= 1")


@dataclass(frozen=True)
class RegimeSettings:
    version: int
    method: str
    enter_threshold: float
    exit_threshold: float
    min_dwell_days: int
    vix_low: float
    vix_high: float
    hmm: HMMSettings

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ConfigError("regime.version must be >= 1")
        if self.method not in ("rule_based", "hmm"):
            raise ConfigError("regime.method must be 'rule_based' or 'hmm'")
        if not 0 < self.exit_threshold < self.enter_threshold <= RULE_SCORE_MAX:
            raise ConfigError(
                "regime thresholds must satisfy "
                f"0 < exit_threshold < enter_threshold <= {RULE_SCORE_MAX} (the maximum rule score)"
            )
        if self.min_dwell_days < 1:
            raise ConfigError("regime.min_dwell_days must be >= 1")
        if not 0 < self.vix_low < self.vix_high:
            raise ConfigError("regime VIX levels must satisfy 0 < vix_low < vix_high")

    @property
    def hmm_version(self) -> str:
        """Version label of the HMM regime definition (model ids are per fit)."""
        return f"hmm-v{self.version}"

    @property
    def rule_version(self) -> str:
        """model_version stored with rule-based regimes (market_regimes.model_version)."""
        return f"rule_based-v{self.version}"


@dataclass(frozen=True)
class RiskSettings:
    """Risk layer (section 19.1). The first three bind (change weights); the last two only
    flag in the signal reason."""

    max_position_size: float
    max_sector_weight: float
    max_portfolio_exposure: float
    atr_stop_multiple: float
    high_volatility: float

    def __post_init__(self) -> None:
        if not 0 < self.max_position_size <= 1:
            raise ConfigError("risk.max_position_size must be in (0, 1]")
        if not self.max_position_size <= self.max_sector_weight <= 1:
            raise ConfigError("risk.max_sector_weight must be in [max_position_size, 1]")
        if not 0 < self.max_portfolio_exposure <= 1:
            raise ConfigError("risk.max_portfolio_exposure must be in (0, 1]; 1.0 never binds")
        if self.atr_stop_multiple <= 0:
            raise ConfigError("risk.atr_stop_multiple must be positive")
        if self.high_volatility <= 0:
            raise ConfigError("risk.high_volatility must be positive (annualised, e.g. 0.60)")


@dataclass(frozen=True)
class PositionSettings:
    # CSV of current holdings (section 6.8): symbol,quantity,cost_basis plus one CASH row.
    # Once the forward test has started it is its start; holdings then come from the trade log.
    path: str
    trades_path: str  # forward test (Phase 10): every BUY / SELL / DIVIDEND after the start

    def __post_init__(self) -> None:
        if not str(self.path).strip() or not str(self.trades_path).strip():
            raise ConfigError("positions.path and positions.trades_path must not be empty")


@dataclass(frozen=True)
class SignalSettings:
    strategy: str  # which rule-based strategy drives the signals (section 18.1)
    hold_band: float  # HOLD while |target - current| is below this

    def __post_init__(self) -> None:
        if not 0 <= self.hold_band < 1:
            raise ConfigError("signals.hold_band must be in [0, 1)")


@dataclass(frozen=True)
class MeanReversionSettings:
    rsi_entry: float
    rsi_exit: float
    max_holding_days: int

    def __post_init__(self) -> None:
        if not 0 < self.rsi_entry < self.rsi_exit < 100:
            raise ConfigError("strategies.mean_reversion must satisfy 0 < rsi_entry < rsi_exit < 100")
        if self.max_holding_days < 1:
            raise ConfigError("strategies.mean_reversion.max_holding_days must be >= 1")


@dataclass(frozen=True)
class RegimeSwitchSettings:
    # The first value is the baseline; any others are variants fixed before seeing results.
    bear_exposures: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.bear_exposures:
            raise ConfigError("strategies.regime_switch.bear_exposures must list at least the baseline")
        if any(not 0 <= value <= 1 for value in self.bear_exposures):
            raise ConfigError("strategies.regime_switch.bear_exposures must be in [0, 1]")
        if len(set(self.bear_exposures)) != len(self.bear_exposures):
            raise ConfigError("strategies.regime_switch.bear_exposures must not repeat a value")


@dataclass(frozen=True)
class MLSelectSettings:
    top_n: int
    # A held stock is kept while it ranks within keep_rank; keep_rank = top_n means
    # the selection is simply the current top_n.
    keep_rank: int
    rebalance_every_sessions: int

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ConfigError("strategies.ml_select.top_n must be >= 1")
        if self.keep_rank < self.top_n:
            raise ConfigError("strategies.ml_select.keep_rank must be >= top_n")
        if self.rebalance_every_sessions < 1:
            raise ConfigError("strategies.ml_select.rebalance_every_sessions must be >= 1")


@dataclass(frozen=True)
class StrategySettings:
    version: int
    mean_reversion: MeanReversionSettings
    regime_switch: RegimeSwitchSettings
    ml_select: MLSelectSettings

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ConfigError("strategies.version must be >= 1")

    @property
    def rule_names(self) -> tuple[str, ...]:
        """Names of the rule-based strategies, in report order (section 17.1)."""
        variants = [bear_variant_name(value) for value in self.regime_switch.bear_exposures[1:]]
        return ("trend", "mean_reversion", "regime_switch", *variants)


def bear_variant_name(bear_exposure: float) -> str:
    """Name of a regime_switch variant with a non-baseline BEAR exposure, e.g. regime_switch_bear50."""
    return f"regime_switch_bear{round(bear_exposure * 100):d}"


MODEL_TYPES = ("logistic_regression", "random_forest", "xgboost")
TARGET_KINDS = ("excess_vs_benchmark_vol_scaled", "fixed_return", "cross_sectional_rank")
CALIBRATIONS = ("isotonic", "sigmoid")


@dataclass(frozen=True)
class TargetSettings:
    kind: str
    benchmark: str
    horizon_days: int
    k: float
    volatility_window_days: int
    min_threshold: float

    def __post_init__(self) -> None:
        if self.kind not in TARGET_KINDS:
            raise ConfigError(f"model.target.kind must be one of {', '.join(TARGET_KINDS)}")
        if self.horizon_days < 1:
            raise ConfigError("model.target.horizon_days must be >= 1")
        if self.k < 0:
            raise ConfigError("model.target.k must not be negative")
        if self.volatility_window_days < 2:
            raise ConfigError("model.target.volatility_window_days must be >= 2")
        if self.min_threshold < 0:
            raise ConfigError("model.target.min_threshold must not be negative")


@dataclass(frozen=True)
class ModelSettings:
    type: str
    target: TargetSettings
    embargo_days: int
    calibration: str
    # Each fold's calibration is fit on the last calibration_months of its training window,
    # which the model itself is not fit on (purged by embargo_days).
    calibration_months: int

    def __post_init__(self) -> None:
        if self.calibration_months < 1:
            raise ConfigError("model.calibration_months must be >= 1")
        if self.type not in MODEL_TYPES:
            raise ConfigError(f"model.type must be one of {', '.join(MODEL_TYPES)}")
        if self.calibration not in CALIBRATIONS:
            raise ConfigError(f"model.calibration must be one of {', '.join(CALIBRATIONS)}")
        # A label for row T spans the opens of T+1 .. T+1+h (section 14).
        if self.embargo_days < self.target.horizon_days + 1:
            raise ConfigError("model.embargo_days must be >= model.target.horizon_days + 1 (section 15)")


VALIDATION_SCHEMES = ("expanding_walk_forward", "rolling_walk_forward")


@dataclass(frozen=True)
class ValidationSettings:
    scheme: str
    rolling_years: int
    min_train_years: int
    fold_length_months: int
    min_folds: int
    holdout_months: int
    holdout_requires_flag: bool

    def __post_init__(self) -> None:
        if self.scheme not in VALIDATION_SCHEMES:
            raise ConfigError(f"retrain.validation.scheme must be one of {', '.join(VALIDATION_SCHEMES)}")
        for name in ("rolling_years", "min_train_years", "fold_length_months", "min_folds", "holdout_months"):
            if getattr(self, name) < 1:
                raise ConfigError(f"retrain.validation.{name} must be >= 1")
        if not isinstance(self.holdout_requires_flag, bool):
            raise ConfigError("retrain.validation.holdout_requires_flag must be true or false")


@dataclass(frozen=True)
class SearchSettings:
    method: str
    max_trials: int
    seed: int
    space: Mapping[str, Mapping[str, tuple]]

    def __post_init__(self) -> None:
        if self.method != "random":
            raise ConfigError("retrain.search.method must be 'random'")
        if self.max_trials < 1:
            raise ConfigError("retrain.search.max_trials must be >= 1")
        unknown = sorted(set(self.space) - set(MODEL_TYPES))
        if unknown:
            raise ConfigError(f"retrain.search.space: unknown model type(s) {', '.join(unknown)}")


# Pass criteria for the PREDICTOR role (section 15.4); quant/models/gates.py evaluates each.
# true / a number = checked; false = disabled.
PREDICTOR_GATES = {
    "leak_tests_pass": bool,
    "beats_baseline_after_costs": bool,
    "min_fraction_of_folds_beating_baseline": float,
    "max_drawdown_not_worse_than_baseline": bool,
    "max_calibration_error": float,
    "holdout_not_worse_than_baseline": bool,
}


# Pass criteria for the REGIME role (sections 13.1 and 16.1); quant/regime/gates.py evaluates each.
REGIME_GATES = {
    "filtered_only_test_pass": bool,
    "state_mapping_stable_across_refits": bool,
    "min_avg_regime_dwell_days": float,
    "beats_rule_based_regime_after_costs": bool,
}


def _check_gate_section(gates: Any, role: str, known: Mapping[str, type]) -> None:
    section = gates.get(role) if isinstance(gates, Mapping) else None
    if not isinstance(section, Mapping):
        raise ConfigError(f"retrain.gates.{role} must be a mapping of gate -> value")
    unknown = sorted(set(section) - set(known))
    if unknown:
        raise ConfigError(f"retrain.gates.{role}: unknown gate(s) {', '.join(unknown)}; known: {', '.join(known)}")
    for name, value in section.items():
        kind = known[name]
        if kind is bool and not isinstance(value, bool):
            raise ConfigError(f"retrain.gates.{role}.{name} must be true or false")
        if kind is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ConfigError(f"retrain.gates.{role}.{name} must be a number")


def _check_gates(gates: Any) -> None:
    _check_gate_section(gates, "predictor", PREDICTOR_GATES)
    _check_gate_section(gates, "regime", REGIME_GATES)


@dataclass(frozen=True)
class RetrainSettings:
    validation: ValidationSettings
    # Training rows: every Nth session. Labels of neighbouring sessions overlap (h + 1
    # sessions), so consecutive rows are near-duplicates; validation scores every session.
    sample_every_sessions: int
    search: SearchSettings
    gates: Mapping[str, Any]
    final_refit: bool
    # Where `quant retrain` writes model files and validation reports (relative to the run directory).
    models_dir: str
    reports_dir: str

    def __post_init__(self) -> None:
        if not self.final_refit:
            raise ConfigError("retrain.final_refit must be true (section 15.5); there is no other final model")
        if not str(self.models_dir).strip() or not str(self.reports_dir).strip():
            raise ConfigError("retrain.models_dir and retrain.reports_dir must not be empty")
        if self.sample_every_sessions < 1:
            raise ConfigError("retrain.sample_every_sessions must be >= 1")
        _check_gates(self.gates)


BENCHMARKS = ("SPY", "equal_weight")


@dataclass(frozen=True)
class BacktestSettings:
    initial_cash: float
    fee_bps: float
    slippage_bps: float
    min_trade_weight: float
    benchmarks: tuple[str, ...]
    random_baseline_runs: int
    random_seed: int

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ConfigError("backtest.initial_cash must be positive")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ConfigError("backtest.fee_bps and backtest.slippage_bps must not be negative")
        if not 0 <= self.min_trade_weight < 1:
            raise ConfigError("backtest.min_trade_weight must be in [0, 1)")
        unknown = sorted(set(self.benchmarks) - set(BENCHMARKS))
        if unknown:
            raise ConfigError(f"backtest.benchmarks: unknown {', '.join(unknown)}; use {', '.join(BENCHMARKS)}")
        if self.random_baseline_runs < 0:
            raise ConfigError("backtest.random_baseline_runs must not be negative")

    @property
    def fees(self) -> float:
        return self.fee_bps / 10_000

    @property
    def slippage(self) -> float:
        return self.slippage_bps / 10_000


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    max_megabytes: float
    backups: int

    def __post_init__(self) -> None:
        if self.level not in LOG_LEVELS:
            raise ConfigError(f"logging.level must be one of {', '.join(LOG_LEVELS)}")
        if self.max_megabytes <= 0:
            raise ConfigError("logging.max_megabytes must be positive")
        if self.backups < 0:
            raise ConfigError("logging.backups must not be negative")


RESERVED_PORTFOLIOS = ("live", "real", "SPY")
REGIME_RULES = ("rule_based-v2",)  # forward_test.shadows.<name>.regime_rule (proposal section 12.5.3)


@dataclass(frozen=True)
class ShadowSettings:
    """A forward-test shadow: a strategy run every evening and frozen, never traded (Phase 10)."""

    strategy: str
    regime_model: str | None = None  # a REGIME model id; None: the rule-based regime
    regime_rule: str | None = None  # a research rule regime computed each evening (REGIME_RULES), not a model
    predictor: str | None = None  # a PREDICTOR model id; required by ml_select


@dataclass(frozen=True)
class ForwardTestSettings:
    shadows: Mapping[str, ShadowSettings]

    def __post_init__(self) -> None:
        for name, shadow in self.shadows.items():
            if name in RESERVED_PORTFOLIOS:
                raise ConfigError(f"forward_test.shadows: {name!r} is reserved ({', '.join(RESERVED_PORTFOLIOS)})")
            if (shadow.strategy == "ml_select") != (shadow.predictor is not None):
                raise ConfigError(f"forward_test.shadows.{name}: ml_select needs a predictor, and only ml_select uses one")
            if shadow.regime_rule is not None and shadow.regime_model is not None:
                raise ConfigError(f"forward_test.shadows.{name}: regime_model or regime_rule, not both")
            if shadow.regime_rule is not None and shadow.regime_rule not in REGIME_RULES:
                raise ConfigError(f"forward_test.shadows.{name}: regime_rule must be one of {', '.join(REGIME_RULES)}")


def _as_shadows(value: Any) -> dict[str, ShadowSettings]:
    if not isinstance(value, Mapping):
        raise ConfigError("forward_test.shadows must map portfolio names to {strategy, regime_model | regime_rule | predictor}")
    shadows = {}
    for name, spec in value.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"forward_test.shadows.{name} must be a mapping")
        unknown = sorted(set(spec) - {"strategy", "regime_model", "regime_rule", "predictor"})
        if unknown or "strategy" not in spec:
            raise ConfigError(f"forward_test.shadows.{name}: needs 'strategy'; unknown key(s): {', '.join(unknown) or '-'}")
        shadows[str(name)] = ShadowSettings(**{k: (str(v) if v is not None else None) for k, v in spec.items()})
    return shadows


SENTIMENT_LANGUAGES = ("en", "zh")


@dataclass(frozen=True)
class ChannelSettings:
    """A YouTube creator followed by the opinion factor (sentiment_proposal.md section 10)."""

    handle: str  # e.g. "@chipstockinvestor"
    channel_id: str  # verified in S0 (sentiment_proposal.md section 12.1)
    name: str
    language: str  # en | zh: picks the caption languages
    style: str

    def __post_init__(self) -> None:
        if not self.handle.startswith("@") or len(self.handle) < 2:
            raise ConfigError(f"sentiment.channels: handle must look like '@name', got {self.handle!r}")
        if not self.channel_id.startswith("UC"):
            raise ConfigError(f"sentiment.channels.{self.handle}: channel_id must start with 'UC'")
        if self.language not in SENTIMENT_LANGUAGES:
            raise ConfigError(f"sentiment.channels.{self.handle}: language must be one of "
                              f"{', '.join(SENTIMENT_LANGUAGES)}")


@dataclass(frozen=True)
class GoldSettings:
    """The hand-labeled answer key for opinion extraction (sentiment_proposal.md section 4.6)."""

    videos: int  # how many videos `quant sentiment gold-export` picks
    labels_dir: str  # videos.csv + opinions.csv: your labels; committed, never in data/
    transcripts_dir: str  # reading copies of the transcripts; git-ignored, deletable

    def __post_init__(self) -> None:
        if self.videos < 1:
            raise ConfigError("sentiment.gold.videos must be >= 1")
        if not str(self.labels_dir).strip() or not str(self.transcripts_dir).strip():
            raise ConfigError("sentiment.gold.labels_dir and transcripts_dir must not be empty")
        if Path(self.labels_dir).resolve() == Path(self.transcripts_dir).resolve():
            raise ConfigError("sentiment.gold.labels_dir and transcripts_dir must differ")


EXTRACTION_BACKENDS = ("claude_code",)  # gemini: S2 step 8


@dataclass(frozen=True)
class ExtractionSettings:
    """Opinion extraction (sentiment_proposal.md section 4)."""

    backend: str
    model: str  # passed to the backend, e.g. sonnet / haiku for claude_code
    # Reports, summaries and the dashboard: per video the opinions of the first of these models
    # that extracted it (so a video extracted by two models counts once).
    report_models: tuple[str, ...]
    prompt_version: int  # bump when the prompt changes; opinions of each version are kept apart
    chunk_minutes: float  # transcripts are sent in pieces of this length ...
    chunk_overlap_seconds: float  # ... each starting this much before the previous one ends
    quote_match_threshold: float  # a quote must match its transcript piece at least this well
    max_videos_per_run: int  # more needs --yes
    call_timeout_seconds: float
    aliases_path: str  # hand-kept names per ticker (config/sentiment_aliases.yaml)
    tickers_cache: str  # US-listed symbols (Nasdaq Trader directory), refreshed monthly

    def __post_init__(self) -> None:
        if self.backend not in EXTRACTION_BACKENDS:
            raise ConfigError(f"sentiment.extraction.backend must be one of {', '.join(EXTRACTION_BACKENDS)}")
        if not str(self.model).strip():
            raise ConfigError("sentiment.extraction.model must not be empty")
        if not self.report_models or not all(self.report_models) or len(set(self.report_models)) < len(
                self.report_models):
            raise ConfigError("sentiment.extraction.report_models must list distinct, non-empty model names")
        if self.prompt_version < 1:
            raise ConfigError("sentiment.extraction.prompt_version must be >= 1")
        if self.chunk_minutes <= 0 or not 0 <= self.chunk_overlap_seconds < self.chunk_minutes * 60:
            raise ConfigError("sentiment.extraction: chunk_minutes must be positive and "
                              "0 <= chunk_overlap_seconds < chunk length")
        if not 0 < self.quote_match_threshold <= 1:
            raise ConfigError("sentiment.extraction.quote_match_threshold must be in (0, 1]")
        if self.max_videos_per_run < 1 or self.call_timeout_seconds <= 0:
            raise ConfigError("sentiment.extraction: max_videos_per_run >= 1 and call_timeout_seconds > 0")


@dataclass(frozen=True)
class SummarySettings:
    """Daily summaries per stock, written from verified opinions only (sentiment_proposal.md 8.3)."""

    backend: str
    model: str
    prompt_version: int
    reports_dir: str  # reports/sentiment/<date>.md

    def __post_init__(self) -> None:
        if self.backend not in EXTRACTION_BACKENDS:
            raise ConfigError(f"sentiment.summary.backend must be one of {', '.join(EXTRACTION_BACKENDS)}")
        if not str(self.model).strip() or not str(self.reports_dir).strip():
            raise ConfigError("sentiment.summary.model and reports_dir must not be empty")
        if self.prompt_version < 1:
            raise ConfigError("sentiment.summary.prompt_version must be >= 1")


@dataclass(frozen=True)
class SentimentSettings:
    """Creator opinion factor (sentiment_proposal.md); research only, never part of run-daily."""

    lookback_days: int
    min_duration_seconds: int  # shorter videos (Shorts) are skipped
    include_live: bool  # live-stream replays
    request_delay_seconds: tuple[float, float]  # random wait between requests to YouTube
    max_captions_per_run: int  # the rest waits for the next run, so YouTube is not asked too often
    # A video without captions is asked again on later runs while younger than this (captions
    # often appear hours after publishing); after that it stays NO_TRANSCRIPT.
    no_captions_retry_days: int
    gold: GoldSettings
    extraction: ExtractionSettings
    summary: SummarySettings
    channels: tuple[ChannelSettings, ...]

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ConfigError("sentiment.lookback_days must be >= 1")
        if self.no_captions_retry_days < 0:
            raise ConfigError("sentiment.no_captions_retry_days must not be negative (0: never retry)")
        if self.max_captions_per_run < 1:
            raise ConfigError("sentiment.max_captions_per_run must be >= 1")
        if self.min_duration_seconds < 0:
            raise ConfigError("sentiment.min_duration_seconds must not be negative")
        if not isinstance(self.include_live, bool):
            raise ConfigError("sentiment.include_live must be true or false")
        low, high = self.request_delay_seconds
        if not 0 <= low <= high:
            raise ConfigError("sentiment.request_delay_seconds must be [low, high] with 0 <= low <= high")
        for attribute in ("handle", "channel_id"):
            values = [getattr(channel, attribute) for channel in self.channels]
            if len(set(values)) != len(values):
                raise ConfigError(f"sentiment.channels: a {attribute} is listed twice")

    def channel(self, handle: str) -> ChannelSettings:
        for channel in self.channels:
            if channel.handle.lower() == handle.lower():
                return channel
        raise ConfigError(f"unknown sentiment channel {handle!r}; configured: "
                          f"{', '.join(c.handle for c in self.channels) or '-'}")


def _as_delay_range(value: Any) -> tuple[float, float]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("sentiment.request_delay_seconds must be a list of two numbers")
    return float(value[0]), float(value[1])


def _as_channels(value: Any) -> tuple[ChannelSettings, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError("sentiment.channels must be a list of {handle, channel_id, name, language, style}")
    channels = []
    for spec in value:
        if not isinstance(spec, Mapping):
            raise ConfigError("sentiment.channels: each channel must be a mapping")
        channels.append(_build(ChannelSettings, "sentiment.channels",
                               {k: str(v) for k, v in spec.items()}))
    return tuple(channels)


@dataclass(frozen=True)
class SecSettings:
    max_requests_per_second: float  # the SEC allows 10; stay well below
    insider_backfill_from: date
    signal_codes: tuple[str, ...]  # Form 4 transaction codes that are decisions (P purchase, S sale)
    exclude_10b5_1_sales: bool  # pre-planned sales are not discretionary
    # Earlier SEC CIKs of a pool company, when its filings moved to a new CIK (e.g. a new holding
    # company): symbol -> CIKs, in addition to the one company_tickers.json gives today.
    extra_ciks: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if not 0 < self.max_requests_per_second <= 10:
            raise ConfigError("altdata.sec.max_requests_per_second must be in (0, 10] (the SEC's limit)")
        if not self.signal_codes:
            raise ConfigError("altdata.sec.signal_codes must list at least one transaction code")


def _as_extra_ciks(value: Any) -> dict[str, tuple[int, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError("altdata.sec.extra_ciks must map symbols to lists of CIKs, e.g. 'XOM: [34088]'")
    out = {}
    for symbol, ciks in value.items():
        if isinstance(ciks, (str, int)) or not isinstance(ciks, (list, tuple)):
            raise ConfigError(f"altdata.sec.extra_ciks.{symbol} must be a list of CIKs")
        out[str(symbol).strip().upper()] = tuple(int(c) for c in ciks)
    return out


@dataclass(frozen=True)
class AltdataSettings:
    """Free structured alternative data (altdata_proposal.md); research only, not part of run-daily."""

    version: int
    user_agent: str  # sent with every request to the SEC: a name and an email address
    sec: SecSettings

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ConfigError("altdata.version must be >= 1")
        if "@" not in self.user_agent or len(self.user_agent.split()) < 3:
            raise ConfigError("altdata.user_agent must look like 'mini-quant-trader <name> <email>' (SEC rule)")


@dataclass(frozen=True)
class Settings:
    database: DatabaseSettings
    market: MarketSettings
    features: FeatureSettings
    regime: RegimeSettings
    strategies: StrategySettings
    risk: RiskSettings
    backtest: BacktestSettings
    model: ModelSettings
    retrain: RetrainSettings
    positions: PositionSettings
    signals: SignalSettings
    logging: LoggingSettings
    forward_test: ForwardTestSettings
    sentiment: SentimentSettings
    altdata: AltdataSettings

    def __post_init__(self) -> None:
        for name, shadow in self.forward_test.shadows.items():
            if shadow.strategy != "ml_select" and shadow.strategy not in self.strategies.rule_names:
                raise ConfigError(f"forward_test.shadows.{name}: unknown strategy {shadow.strategy!r}")
        if self.signals.strategy not in self.strategies.rule_names:
            raise ConfigError(
                f"signals.strategy {self.signals.strategy!r} is not a rule-based strategy; "
                f"use one of {', '.join(self.strategies.rule_names)}"
            )
        if self.signals.hold_band != self.backtest.min_trade_weight:
            warnings.warn(
                f"signals.hold_band ({self.signals.hold_band:g}) differs from backtest.min_trade_weight "
                f"({self.backtest.min_trade_weight:g}): live signals will not trade like the backtest",
                stacklevel=2,
            )
        round_trip = 2 * (self.backtest.fees + self.backtest.slippage)
        if self.model.target.min_threshold <= round_trip:
            raise ConfigError(
                f"model.target.min_threshold ({self.model.target.min_threshold:g}) must exceed the round-trip "
                f"trading cost ({round_trip:g} from backtest.fee_bps and slippage_bps), so a hit is worth "
                "something after costs (section 14)"
            )

    @property
    def holdout_months(self) -> int:
        """retrain.validation.holdout_months: the most recent period kept out of all tuning."""
        return self.retrain.validation.holdout_months

    @property
    def holdout_requires_flag(self) -> bool:
        """retrain.validation.holdout_requires_flag: show the holdout only on explicit request."""
        return self.retrain.validation.holdout_requires_flag


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if value is None:
        raise ConfigError(f"missing settings section: {name}")
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping")
    return value


def _build(cls, name: str, values: Mapping[str, Any], converters: Mapping[str, Any] | None = None):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(f"unknown setting(s) in '{name}': {', '.join(unknown)}")
    missing = sorted(known - set(values))
    if missing:
        raise ConfigError(f"missing setting(s) in '{name}': {', '.join(missing)}")
    converters = converters or {}
    try:
        kwargs = {key: converters.get(key, lambda v: v)(value) for key, value in values.items()}
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid value in '{name}': {exc}") from exc


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of symbols")
    if any(isinstance(item, (list, tuple, Mapping)) for item in value):
        raise ValueError("expected a flat list of symbols")
    return tuple(dict.fromkeys(str(item).strip().upper() for item in value if str(item).strip()))


def _as_sector_pool(value: Any) -> dict[str, tuple[str, ...]]:
    """market.symbols: a mapping of GICS sector -> list of stocks; each stock in one sector."""
    if not isinstance(value, Mapping):
        raise ConfigError("market.symbols must map GICS sectors to stock lists, e.g. 'energy: [XOM, CVX]'")
    pool: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}
    for sector, members in value.items():
        if sector not in GICS_SECTORS:
            raise ConfigError(f"market.symbols: unknown sector {sector!r}; use one of {', '.join(GICS_SECTORS)}")
        try:
            stocks = _as_str_tuple(members)
        except ValueError as exc:
            raise ConfigError(f"market.symbols.{sector}: {exc}") from exc
        if not stocks:
            raise ConfigError(f"market.symbols.{sector} is empty; remove the sector or add stocks")
        for stock in stocks:
            if stock in seen:
                raise ConfigError(f"{stock} is listed under both {seen[stock]} and {sector}")
            seen[stock] = sector
        pool[sector] = stocks
    return pool


def _as_name_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of names")
    return tuple(str(item).strip() for item in value)


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of numbers")
    return tuple(float(item) for item in value)


def _as_mapping(name: str):
    def convert(value: Any) -> dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"'{name}' must be a mapping")
        return dict(value)

    return convert


def _as_search_space(value: Any) -> dict[str, dict[str, tuple]]:
    """retrain.search.space: model type -> hyperparameter -> list of candidate values."""
    if not isinstance(value, Mapping):
        raise ConfigError("retrain.search.space must map model types to hyperparameter lists")
    space: dict[str, dict[str, tuple]] = {}
    for model_type, parameters in value.items():
        if not isinstance(parameters, Mapping):
            raise ConfigError(f"retrain.search.space.{model_type} must map hyperparameters to lists")
        space[model_type] = {}
        for name, candidates in parameters.items():
            if isinstance(candidates, str) or not isinstance(candidates, (list, tuple)) or not candidates:
                raise ConfigError(f"retrain.search.space.{model_type}.{name} must be a non-empty list")
            space[model_type][name] = tuple(candidates)
    return space


def _subsection(cls, name: str, converters: Mapping[str, Any] | None = None):
    """Converter for a nested mapping such as strategies.mean_reversion."""

    def convert(value: Any):
        if not isinstance(value, Mapping):
            raise ConfigError(f"'{name}' must be a mapping")
        return _build(cls, name, value, converters)

    return convert


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of integers")
    return tuple(int(item) for item in value)


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """Read settings.yaml. Every section and key is required; unknown keys are rejected."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with config_path.open(encoding="utf-8") as source:
            raw = yaml.safe_load(source) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"settings file not found: {config_path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read settings file {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path} must contain a mapping of settings sections")

    sections = {f.name for f in fields(Settings)}
    unknown = sorted(set(raw) - sections)
    if unknown:
        raise ConfigError(f"unknown settings section(s): {', '.join(unknown)}")

    return Settings(
        database=_build(DatabaseSettings, "database", _section(raw, "database")),
        market=_build(
            MarketSettings,
            "market",
            _section(raw, "market"),
            {"initial_history": _as_date, "symbols": _as_sector_pool, "market_symbols": _as_str_tuple},
        ),
        features=_build(FeatureSettings, "features", _section(raw, "features"), {"sma": _as_int_tuple}),
        regime=_build(
            RegimeSettings,
            "regime",
            _section(raw, "regime"),
            {"hmm": _subsection(HMMSettings, "regime.hmm", {"observations": _as_name_tuple})},
        ),
        strategies=_build(
            StrategySettings,
            "strategies",
            _section(raw, "strategies"),
            {
                "mean_reversion": _subsection(MeanReversionSettings, "strategies.mean_reversion"),
                "regime_switch": _subsection(
                    RegimeSwitchSettings, "strategies.regime_switch", {"bear_exposures": _as_float_tuple}
                ),
                "ml_select": _subsection(MLSelectSettings, "strategies.ml_select"),
            },
        ),
        risk=_build(RiskSettings, "risk", _section(raw, "risk")),
        backtest=_build(BacktestSettings, "backtest", _section(raw, "backtest"), {"benchmarks": _as_name_tuple}),
        model=_build(
            ModelSettings,
            "model",
            _section(raw, "model"),
            {"target": _subsection(TargetSettings, "model.target")},
        ),
        retrain=_build(
            RetrainSettings,
            "retrain",
            _section(raw, "retrain"),
            {
                "validation": _subsection(ValidationSettings, "retrain.validation"),
                "search": _subsection(SearchSettings, "retrain.search", {"space": _as_search_space}),
                "gates": _as_mapping("retrain.gates"),
            },
        ),
        positions=_build(PositionSettings, "positions", _section(raw, "positions")),
        signals=_build(SignalSettings, "signals", _section(raw, "signals")),
        logging=_build(LoggingSettings, "logging", _section(raw, "logging")),
        forward_test=_build(ForwardTestSettings, "forward_test", _section(raw, "forward_test"),
                            {"shadows": _as_shadows}),
        sentiment=_build(SentimentSettings, "sentiment", _section(raw, "sentiment"),
                         {"request_delay_seconds": _as_delay_range, "channels": _as_channels,
                          "gold": _subsection(GoldSettings, "sentiment.gold"),
                          "extraction": _subsection(ExtractionSettings, "sentiment.extraction",
                                                    {"report_models": _as_name_tuple}),
                          "summary": _subsection(SummarySettings, "sentiment.summary")}),
        altdata=_build(AltdataSettings, "altdata", _section(raw, "altdata"),
                       {"sec": _subsection(SecSettings, "altdata.sec", {"insider_backfill_from": _as_date,
                                                                         "signal_codes": _as_str_tuple,
                                                                         "extra_ciks": _as_extra_ciks})}),
    )