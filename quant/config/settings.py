"""Load and validate application settings from config/settings.yaml (proposal section 27)."""

from __future__ import annotations

import os
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


@dataclass(frozen=True)
class RegimeSettings:
    version: int
    method: str
    enter_threshold: float
    exit_threshold: float
    min_dwell_days: int
    vix_low: float
    vix_high: float

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
    def rule_version(self) -> str:
        """model_version stored with rule-based regimes (market_regimes.model_version)."""
        return f"rule_based-v{self.version}"


@dataclass(frozen=True)
class RiskSettings:
    max_position_size: float
    max_sector_weight: float

    def __post_init__(self) -> None:
        if not 0 < self.max_position_size <= 1:
            raise ConfigError("risk.max_position_size must be in (0, 1]")
        if not self.max_position_size <= self.max_sector_weight <= 1:
            raise ConfigError("risk.max_sector_weight must be in [max_position_size, 1]")


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
class StrategySettings:
    version: int
    mean_reversion: MeanReversionSettings
    regime_switch: RegimeSwitchSettings

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ConfigError("strategies.version must be >= 1")


@dataclass(frozen=True)
class Settings:
    database: DatabaseSettings
    market: MarketSettings
    features: FeatureSettings
    regime: RegimeSettings
    strategies: StrategySettings
    risk: RiskSettings
    # Sections for later phases are loaded as-is and validated when those phases land.
    model: Mapping[str, Any]
    retrain: Mapping[str, Any]
    backtest: Mapping[str, Any]


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


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of numbers")
    return tuple(float(item) for item in value)


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
        regime=_build(RegimeSettings, "regime", _section(raw, "regime")),
        strategies=_build(
            StrategySettings,
            "strategies",
            _section(raw, "strategies"),
            {
                "mean_reversion": _subsection(MeanReversionSettings, "strategies.mean_reversion"),
                "regime_switch": _subsection(
                    RegimeSwitchSettings, "strategies.regime_switch", {"bear_exposures": _as_float_tuple}
                ),
            },
        ),
        risk=_build(RiskSettings, "risk", _section(raw, "risk")),
        model=dict(_section(raw, "model")),
        retrain=dict(_section(raw, "retrain")),
        backtest=dict(_section(raw, "backtest")),
    )