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


@dataclass(frozen=True)
class MarketSettings:
    provider: str
    request_timeout_seconds: float
    calendar: str
    initial_history: date
    timezone: str
    execution: str
    symbols: tuple[str, ...]
    market_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.request_timeout_seconds <= 0:
            raise ConfigError("market.request_timeout_seconds must be positive")
        if not (self.symbols or self.market_symbols):
            raise ConfigError("market.symbols and market.market_symbols must not both be empty")

    @property
    def universe(self) -> list[str]:
        """Every symbol to keep up to date: the stock pool plus market-level symbols."""
        return list(dict.fromkeys((*self.symbols, *self.market_symbols)))


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
class Settings:
    database: DatabaseSettings
    market: MarketSettings
    features: FeatureSettings
    regime: RegimeSettings
    # Sections for later phases are loaded as-is and validated when those phases land.
    model: Mapping[str, Any]
    retrain: Mapping[str, Any]
    backtest: Mapping[str, Any]
    risk: Mapping[str, Any]


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
    """A list of symbols, optionally grouped into sub-lists (e.g. one per sector)."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of symbols")
    items: list[Any] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            items.extend(item)
        else:
            items.append(item)
    if any(isinstance(item, (list, tuple, dict)) for item in items):
        raise ValueError("symbol groups must not be nested more than one level")
    return tuple(dict.fromkeys(str(item).strip().upper() for item in items if str(item).strip()))


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
            {"initial_history": _as_date, "symbols": _as_str_tuple, "market_symbols": _as_str_tuple},
        ),
        features=_build(FeatureSettings, "features", _section(raw, "features"), {"sma": _as_int_tuple}),
        regime=_build(RegimeSettings, "regime", _section(raw, "regime")),
        model=dict(_section(raw, "model")),
        retrain=dict(_section(raw, "retrain")),
        backtest=dict(_section(raw, "backtest")),
        risk=dict(_section(raw, "risk")),
    )