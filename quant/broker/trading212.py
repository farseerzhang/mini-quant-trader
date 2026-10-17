"""Trading 212 public API (beta), read-only: account info, cash and open positions.

Credentials come from the environment (the project's `.env`, git-ignored), never from code or
settings:

- TRADING212_API_KEY (required) and TRADING212_API_SECRET (if your key has one): created in the
  Trading 212 app under Settings -> API. Give the key read permissions only (account data,
  portfolio, history); never order permissions. With a secret the request uses HTTP Basic
  authentication (key:secret); without one, the key alone in the Authorization header.
- TRADING212_ENVIRONMENT: live (default) or demo (the practice account).

The API has strict rate limits per endpoint (the portfolio about one request every 5 s), so
calls are spaced and a 429 answer is retried after a pause.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field

import requests

BASE_URLS = {"live": "https://live.trading212.com/api/v0", "demo": "https://demo.trading212.com/api/v0"}
PAUSE_BETWEEN_REQUESTS_SECONDS = 5.5
RETRY_PAUSES_SECONDS = (10, 30)


class Trading212Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Trading212Credentials:
    api_key: str
    api_secret: str | None
    environment: str

    @classmethod
    def from_environment(cls) -> "Trading212Credentials":
        from dotenv import load_dotenv

        load_dotenv()
        key = os.environ.get("TRADING212_API_KEY", "").strip()
        if not key:
            raise Trading212Error("TRADING212_API_KEY is not set (add it to the project's .env file)")
        environment = os.environ.get("TRADING212_ENVIRONMENT", "live").strip().lower() or "live"
        if environment not in BASE_URLS:
            raise Trading212Error(f"TRADING212_ENVIRONMENT must be live or demo, got {environment!r}")
        return cls(key, os.environ.get("TRADING212_API_SECRET", "").strip() or None, environment)

    def authorization_header(self) -> str:
        if self.api_secret:
            token = base64.b64encode(f"{self.api_key}:{self.api_secret}".encode()).decode()
            return f"Basic {token}"
        return self.api_key


@dataclass(frozen=True)
class Trading212Position:
    ticker: str  # Trading 212's own format, e.g. AAPL_US_EQ
    quantity: float
    average_price: float | None  # in the instrument's currency
    current_price: float | None
    profit_loss_account_currency: float | None  # "ppl", in the account currency
    fx_profit_loss: float | None  # "fxPpl": the part of it caused by the exchange rate


@dataclass(frozen=True)
class Trading212Account:
    account_id: str | None
    currency: str | None
    cash_free: float | None
    cash_total: float | None  # the account value in its currency (cash + invested + result)
    positions: list[Trading212Position] = field(default_factory=list)


class Trading212Client:
    def __init__(self, credentials: Trading212Credentials, session: requests.Session | None = None,
                 pause_seconds: float = PAUSE_BETWEEN_REQUESTS_SECONDS, sleep=time.sleep):
        self.credentials = credentials
        self.session = session or requests.Session()
        self.pause_seconds, self.sleep = pause_seconds, sleep
        self._requests = 0

    def get(self, path: str):
        url = BASE_URLS[self.credentials.environment] + path
        for attempt in range(len(RETRY_PAUSES_SECONDS) + 1):
            if self._requests:
                self.sleep(self.pause_seconds)
            self._requests += 1
            response = self.session.get(url, headers={"Authorization": self.credentials.authorization_header()},
                                        timeout=30)
            if response.status_code == 429 and attempt < len(RETRY_PAUSES_SECONDS):
                self.sleep(RETRY_PAUSES_SECONDS[attempt])
                continue
            if response.status_code in (401, 403):
                raise Trading212Error(f"{path}: refused ({response.status_code}); check the key, the secret, the "
                                      "environment (live/demo) and that the key has read permission for it")
            if response.status_code != 200:
                raise Trading212Error(f"{path}: HTTP {response.status_code}: {response.text[:200]}")
            return response.json()
        raise Trading212Error(f"{path}: still rate-limited after {len(RETRY_PAUSES_SECONDS)} retries")

    def account(self) -> Trading212Account:
        info = self.get("/equity/account/info")
        cash = self.get("/equity/account/cash")
        portfolio = self.get("/equity/portfolio")
        return Trading212Account(
            account_id=str(info.get("id")) if info.get("id") is not None else None,
            currency=info.get("currencyCode"),
            cash_free=_number(cash.get("free")), cash_total=_number(cash.get("total")),
            positions=[Trading212Position(
                ticker=str(p.get("ticker")), quantity=_number(p.get("quantity")) or 0.0,
                average_price=_number(p.get("averagePrice")), current_price=_number(p.get("currentPrice")),
                profit_loss_account_currency=_number(p.get("ppl")), fx_profit_loss=_number(p.get("fxPpl")))
                for p in portfolio or []])


def _number(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# Trading 212 still lists some US stocks under the ticker they had before a rename (seen in the
# first connection check, 2026-10-17): FB_US_EQ is Meta Platforms (META), SQ_US_EQ is Block (XYZ).
RENAMED_US_TICKERS = {"FB": "META", "SQ": "XYZ"}


def trading212_symbol(ticker: str) -> str:
    """AAPL_US_EQ -> AAPL; BRK_B_US_EQ -> BRK-B (Yahoo's form); renamed stocks get today's ticker
    (FB_US_EQ -> META). Non-US listings (e.g. BMWd_EQ on Xetra) keep their full Trading 212 code."""
    parts = ticker.split("_")
    if len(parts) >= 3 and parts[-2] == "US" and parts[-1] == "EQ":
        symbol = "-".join(parts[:-2])
        return RENAMED_US_TICKERS.get(symbol, symbol)
    return ticker
