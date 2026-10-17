"""IBKR Flex Web Service, read-only: open positions and cash from an Activity Flex Query.

Flex reports cannot place orders and need no running TWS or gateway. One-time setup in the IBKR
portal (Client Portal -> Performance & Reports -> Flex Queries):

1. Create an Activity Flex Query with the sections Open Positions (Summary), Cash Report and
   Account Information; format XML; period "Last Business Day". Note its Query ID.
2. Under Flex Web Service configuration, enable the service and generate a token.

Credentials come from the environment (the project's `.env`, git-ignored): IBKR_FLEX_TOKEN and
IBKR_FLEX_QUERY_ID.

A request is two steps: SendRequest returns a reference code, then GetStatement returns the
report; while IBKR is still generating it, GetStatement answers "in progress" and is asked again.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field

import requests

SERVICE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
VERSION = "3"
STATEMENT_WAITS_SECONDS = (5, 10, 20, 30, 60)
IN_PROGRESS_CODES = {"1019", "1018"}  # statement generation in progress / too many requests, try again
HEADERS = {"User-Agent": "mini-quant-trader (read-only Flex report)"}


class IbkrFlexError(RuntimeError):
    pass


@dataclass(frozen=True)
class IbkrFlexCredentials:
    token: str
    query_id: str

    @classmethod
    def from_environment(cls) -> "IbkrFlexCredentials":
        from dotenv import load_dotenv

        load_dotenv()
        token = os.environ.get("IBKR_FLEX_TOKEN", "").strip()
        query_id = os.environ.get("IBKR_FLEX_QUERY_ID", "").strip()
        missing = [name for name, value in (("IBKR_FLEX_TOKEN", token), ("IBKR_FLEX_QUERY_ID", query_id)) if not value]
        if missing:
            raise IbkrFlexError(f"{' and '.join(missing)} not set (add to the project's .env file)")
        return cls(token, query_id)


@dataclass(frozen=True)
class IbkrPosition:
    symbol: str
    asset_category: str | None  # STK, OPT, ...
    quantity: float
    mark_price: float | None
    position_value: float | None
    cost_basis_price: float | None
    currency: str | None


@dataclass(frozen=True)
class IbkrAccount:
    account_id: str | None
    base_currency: str | None
    report_date: str | None  # the statement's "to" date, YYYYMMDD
    ending_cash_base: float | None  # cash in the base currency (the BASE_SUMMARY line of the Cash Report)
    positions: list[IbkrPosition] = field(default_factory=list)


def _number(value) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def parse_statement(xml_text: str) -> IbkrAccount:
    """The first FlexStatement of a FlexQueryResponse (one account)."""
    root = ElementTree.fromstring(xml_text)
    statement = root.find(".//FlexStatement")
    if statement is None:
        raise IbkrFlexError("the Flex report has no FlexStatement; check the query's sections")
    info = statement.find("AccountInformation")
    positions = []
    for p in statement.findall("./OpenPositions/OpenPosition"):
        if p.get("levelOfDetail") not in (None, "", "SUMMARY"):
            continue  # lot lines repeat the summary line
        positions.append(IbkrPosition(
            symbol=p.get("symbol", ""), asset_category=p.get("assetCategory"),
            quantity=_number(p.get("position")) or 0.0, mark_price=_number(p.get("markPrice")),
            position_value=_number(p.get("positionValue")), cost_basis_price=_number(p.get("costBasisPrice")),
            currency=p.get("currency")))
    cash = None
    for line in statement.findall("./CashReport/CashReportCurrency"):
        if line.get("currency") == "BASE_SUMMARY":
            cash = _number(line.get("endingCash"))
    return IbkrAccount(
        account_id=statement.get("accountId") or (info.get("accountId") if info is not None else None),
        base_currency=info.get("currency") if info is not None else None,
        report_date=statement.get("toDate"), ending_cash_base=cash, positions=positions)


class IbkrFlexClient:
    def __init__(self, credentials: IbkrFlexCredentials, session: requests.Session | None = None, sleep=time.sleep):
        self.credentials = credentials
        self.session = session or requests.Session()
        self.sleep = sleep

    def _call(self, endpoint: str, code: str) -> str:
        response = self.session.get(f"{SERVICE_URL}/{endpoint}", params={"t": self.credentials.token, "q": code,
                                                                         "v": VERSION}, headers=HEADERS, timeout=60)
        if response.status_code != 200:
            raise IbkrFlexError(f"{endpoint}: HTTP {response.status_code}")
        return response.text

    def statement_xml(self) -> str:
        sent = ElementTree.fromstring(self._call("SendRequest", self.credentials.query_id))
        if (sent.findtext("Status") or "") != "Success":
            raise IbkrFlexError(f"SendRequest: {sent.findtext('ErrorCode')} {sent.findtext('ErrorMessage')}")
        reference = sent.findtext("ReferenceCode")
        for wait in STATEMENT_WAITS_SECONDS:
            self.sleep(wait)
            text = self._call("GetStatement", reference)
            if "<FlexQueryResponse" in text:
                return text
            answer = ElementTree.fromstring(text)
            if (answer.findtext("ErrorCode") or "") not in IN_PROGRESS_CODES:
                raise IbkrFlexError(f"GetStatement: {answer.findtext('ErrorCode')} {answer.findtext('ErrorMessage')}")
        raise IbkrFlexError("the Flex report was still being generated after waiting; try again in a few minutes")

    def account(self) -> IbkrAccount:
        return parse_statement(self.statement_xml())
