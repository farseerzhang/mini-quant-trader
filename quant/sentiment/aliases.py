"""Which tickers exist (sentiment_proposal.md sections 4.3-4.4).

A ticker from the extractor counts as known if it is in the hand-kept alias table
(config/sentiment_aliases.yaml) or listed on a US exchange. The exchange list comes from
the Nasdaq Trader symbol directory (every Nasdaq, NYSE and other US-listed stock and ETF;
public, no key), cached locally and refreshed monthly. The SEC list named in the proposal
needs a contact email in every request, so it is not used.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import yaml

DIRECTORY = ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
             "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")
REFRESH_AFTER = timedelta(days=30)
TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,5}(-[A-Z])?$")


class AliasError(ValueError):
    """The alias table cannot be read."""


def normalize_ticker(value: str) -> str:
    """Upper case, no $ prefix; class shares with a dash as in the stock pool (BRK.B -> BRK-B)."""
    ticker = value.strip().upper().lstrip("$")
    return re.sub(r"[./]([A-Z])$", r"-\1", ticker)


def load_aliases(path: str | Path) -> dict[str, tuple[str, ...]]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise AliasError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise AliasError(f"{path} must map tickers to lists of names")
    aliases = {}
    for key, names in raw.items():
        if not isinstance(key, str):  # e.g. a bare ON or NO, which YAML reads as a boolean
            raise AliasError(f"{path}: ticker {key!r} is not text; put it in quotes")
        ticker = normalize_ticker(key)
        if not TICKER.match(ticker):
            raise AliasError(f"{path}: {key!r} does not look like a ticker")
        if isinstance(names, str) or not isinstance(names, list):
            raise AliasError(f"{path}: {key} must have a list of names")
        aliases[ticker] = tuple(str(name).strip() for name in names if str(name).strip())
    return aliases


@dataclass(frozen=True)
class Listing:
    name: str
    etf: bool


def _parse_directory(text: str) -> dict[str, Listing]:
    lines = text.splitlines()
    header = lines[0].split("|")
    symbol_column = 0  # "Symbol" (nasdaqlisted) or "ACT Symbol" (otherlisted)
    etf_column = header.index("ETF")
    test_column = header.index("Test Issue")
    listings = {}
    for line in lines[1:]:
        fields = line.split("|")
        if len(fields) != len(header) or fields[test_column] == "Y":
            continue
        ticker = normalize_ticker(fields[symbol_column])
        if TICKER.match(ticker):  # skips the closing "File Creation Time" line and preferred-share codes
            listings[ticker] = Listing(name=fields[1], etf=fields[etf_column] == "Y")
    return listings


def download_listings(timeout: float = 30) -> dict[str, Listing]:
    listings: dict[str, Listing] = {}
    for url in DIRECTORY:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            listings.update(_parse_directory(response.read().decode("utf-8", errors="replace")))
    return listings


def load_listings(cache: str | Path, now: datetime | None = None, download=download_listings) -> dict[str, Listing]:
    """The cached US listings, downloaded again when missing or older than REFRESH_AFTER.
    If the download fails, an older cache is used rather than nothing."""
    cache, now = Path(cache), now or datetime.now(timezone.utc)
    stored = None
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if now - datetime.fromisoformat(stored["fetched_at"]) < REFRESH_AFTER:
            return {k: Listing(**v) for k, v in stored["listings"].items()}
    try:
        listings = download()
    except OSError:
        if stored is None:
            raise
        return {k: Listing(**v) for k, v in stored["listings"].items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"fetched_at": now.isoformat(), "listings": {k: vars(v) for k, v in listings.items()}}),
                     encoding="utf-8")
    return listings


class TickerBook:
    """Known tickers: the alias table plus US listings."""

    def __init__(self, aliases: Mapping[str, tuple[str, ...]], listings: Mapping[str, Listing]) -> None:
        self.aliases, self.listings = dict(aliases), dict(listings)

    def known(self, ticker: str) -> bool:
        return ticker in self.aliases or ticker in self.listings

    def is_etf(self, ticker: str) -> bool:
        return ticker in self.listings and self.listings[ticker].etf

    def prompt_table(self) -> str:
        """The alias table as prompt text: one `TICKER: name, name` line per ticker."""
        return "\n".join(f"{ticker}: {', '.join(names)}" for ticker, names in self.aliases.items())
