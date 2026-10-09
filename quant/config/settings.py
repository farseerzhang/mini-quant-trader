"""Load and validate application settings from config/settings.yaml (proposal section 27)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("config") / "settings.yaml"
_SQLITE_PREFIX = "sqlite:///"
_SUPPORTED_SMA_WINDOWS = (20, 50, 200)


class ConfigError(ValueError):
    """Raised when the settings file is missing or invalid."""


@dataclass(frozen=True)
class DatabaseSettings:
    url: str = "sqlite:///data/quant.db"
    busy_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if not self.url.startswith(_SQLITE_PREFIX) or len(self.url) == len(_SQLITE_PREFIX):
            raise ConfigError(f"database.url must look like 'sqlite:///path/to/file.db', got {self.url!r}")
        if self.busy_timeout_ms < 0:
            raise ConfigError("database.busy_timeout_ms must not be negative")

    @property
    def path(self) -> str:
        return self.url[len(_SQLITE_PREFIX):]


@dataclass(frozen=True)
class MarketSettings:
    provider: str = "yfinance"
    request_timeout_seconds: float = 20
    calendar: str = "XNYS"
    initial_history: date = date(2005, 1, 1)
    timezone: str = "America/New_York"
    execution: str = "next_open"
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "SPY", "QQQ")
    market_symbols: tuple[str, ...] = ("SPY", "QQQ", "^VIX")

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
    version: int = 1
    sma: tuple[int, int, int] = _SUPPORTED_SMA_WINDOWS
    rsi_period: int = 14
    atr_period: int = 14

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ConfigError("features.version must be >= 1")
        if tuple(self.sma) != _SUPPORTED_SMA_WINDOWS:
            raise ConfigError(
                f"features.sma must be {list(_SUPPORTED_SMA_WINDOWS)}: stored feature columns "
                "(price_vs_sma20, ...) are named after these windows"
            )
        if self.rsi_period < 2 or self.atr_period < 2:
            raise ConfigError("features.rsi_period and features.atr_period must be >= 2")

    @property
    def feature_version(self) -> str:
        return f"technical-v{self.version}"

    @property
    def market_feature_version(self) -> str:
        return f"market-v{self.version}"


@dataclass(frozen=True)
class RegimeSettings:
    method: str = "rule_based"
    enter_threshold: float = 5
    exit_threshold: float = 3
    min_dwell_days: int = 5

    def __post_init__(self) -> None:
        if self.method not in ("rule_based", "hmm"):
            raise ConfigError("regime.method must be 'rule_based' or 'hmm'")
        if not 0 < self.exit_threshold < self.enter_threshold:
            raise ConfigError("regime thresholds must satisfy 0 < exit_threshold < enter_threshold")
        if self.min_dwell_days < 1:
            raise ConfigError("regime.min_dwell_days must be >= 1")


@dataclass(frozen=True)
class Settings:
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    market: MarketSettings = field(default_factory=MarketSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    regime: RegimeSettings = field(default_factory=RegimeSettings)
    # Sections for later phases are loaded as-is and validated when those phases land.
    model: Mapping[str, Any] = field(default_factory=dict)
    retrain: Mapping[str, Any] = field(default_factory=dict)
    backtest: Mapping[str, Any] = field(default_factory=dict)
    risk: Mapping[str, Any] = field(default_factory=dict)


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping")
    return value


def _build(cls, name: str, values: Mapping[str, Any], converters: Mapping[str, Any] | None = None):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(f"unknown setting(s) in '{name}': {', '.join(unknown)}")
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
    return tuple(str(item).strip().upper() for item in value if str(item).strip())


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of integers")
    return tuple(int(item) for item in value)


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """Read settings.yaml. Missing sections fall back to defaults; unknown keys are rejected."""
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