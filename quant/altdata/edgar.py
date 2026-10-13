"""SEC EDGAR: Form 4 insider transactions (altdata_proposal.md section 3.1).

Two sources with the same output:

- the quarterly Insider Transactions Data Sets (every Form 3/4/5 of the quarter for all US
  issuers, 12-17 MB zips, 2006 onward): the backfill; only pool issuers' Form 4 / 4/A rows
  are kept. They carry a filing date but no acceptance time.
- live filings after the newest loaded quarter: each pool company's submissions list
  (data.sec.gov), then each new Form 4's XML. These carry the exact acceptance time.

Every request sends altdata.user_agent and respects altdata.sec.max_requests_per_second
(the SEC allows 10); 429 / 403 / 5xx are retried with a growing pause.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable

import pandas as pd

INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FORM4_TYPES = ("4", "4/A")
RETRY_PAUSES = (10, 30, 90)  # seconds, after 429 / 403 / 5xx


class SecError(Exception):
    """A request to the SEC failed after its retries."""


class SecClient:
    def __init__(self, user_agent: str, max_requests_per_second: float,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 opener: Callable = urllib.request.urlopen) -> None:
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
        self.interval = 1 / max_requests_per_second
        self.sleep, self.clock, self.opener = sleep, clock, opener
        self.last = None
        self.requests = 0

    def get(self, url: str, timeout: float = 120) -> bytes:
        for attempt in range(len(RETRY_PAUSES) + 1):
            if self.last is not None:
                wait = self.interval - (self.clock() - self.last)
                if wait > 0:
                    self.sleep(wait)
            self.last = self.clock()
            self.requests += 1
            try:
                with self.opener(urllib.request.Request(url, headers=self.headers), timeout=timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429) or exc.code >= 500:
                    if attempt < len(RETRY_PAUSES):
                        self.sleep(RETRY_PAUSES[attempt])
                        continue
                raise SecError(f"{url}: HTTP {exc.code}") from exc
            except OSError as exc:  # network errors
                if attempt < len(RETRY_PAUSES):
                    self.sleep(RETRY_PAUSES[attempt])
                    continue
                raise SecError(f"{url}: {exc}") from exc
        raise SecError(f"{url}: gave up")  # pragma: no cover

    def json(self, url: str):
        return json.loads(self.get(url))


# --- entities ------------------------------------------------------------------------------

def pool_ciks(tickers: dict, pool: list[str]) -> tuple[dict[str, tuple[int, str]], list[str]]:
    """Pool symbol -> (CIK, company name), from company_tickers.json; and the symbols not found."""
    by_ticker = {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in tickers.values()}
    found, missing = {}, []
    for symbol in pool:
        hit = next((by_ticker[c] for c in (symbol, symbol.replace("-", "."), symbol.replace("-", ""))
                    if c in by_ticker), None)
        if hit:
            found[symbol] = hit
        else:
            missing.append(symbol)
    return found, missing


# --- parsed filings -----------------------------------------------------------------------

@dataclass
class Filing:
    accession_no: str
    cik: int
    document_type: str
    filing_date: date
    period_of_report: date | None = None
    date_of_original: date | None = None
    accepted_at: datetime | None = None  # naive UTC
    is_10b5_1: int | None = None
    owners: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)


def flag(value) -> int | None:
    """true / 1 -> 1, false / 0 -> 0, empty -> None (the data sets mix the spellings)."""
    text = str(value if value is not None else "").strip().lower()
    return 1 if text in ("1", "true", "y", "yes") else 0 if text in ("0", "false", "n", "no") else None


def _number(value) -> float | None:
    try:
        text = str(value).strip().replace(",", "")
        return float(text) if text else None
    except ValueError:
        return None


def _date(value) -> date | None:
    text = str(value or "").strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _owner(cik, name, relationship: set[str], title) -> dict:
    return {"owner_cik": int(cik), "owner_name": name or None, "is_director": int("director" in relationship),
            "is_officer": int("officer" in relationship), "is_ten_percent_owner": int("tenpercentowner" in relationship),
            "is_other": int("other" in relationship), "officer_title": (title or "").strip() or None}


# --- the quarterly data sets ----------------------------------------------------------------

def dataset_links(index_html: str) -> dict[str, str]:
    """Quarter key (e.g. 2026q2) -> absolute URL of its zip, read from the index page (the
    newest quarters live under a different folder than the older ones)."""
    links = {}
    for href in re.findall(r'href="([^"]*?(\d{4}q[1-4])_form345\.zip)"', index_html):
        url, key = href
        links[key] = url if url.startswith("http") else "https://www.sec.gov" + url
    return dict(sorted(links.items()))


def quarter_end(key: str) -> date:
    year, quarter = int(key[:4]), int(key[5])
    return (pd.Timestamp(year=year, month=3 * quarter, day=1) + pd.offsets.MonthEnd(0)).date()


def parse_dataset(zip_bytes: bytes, issuers: set[int]) -> list[Filing]:
    """Form 4 / 4/A filings of these issuer CIKs from one quarterly zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = {name.rsplit("/", 1)[-1].upper(): name for name in archive.namelist()}

        def table(name: str) -> pd.DataFrame:
            with archive.open(names[name]) as raw:
                # Tab-separated without quoting (QUOTE_NONE): a stray quote in a name must not swallow rows.
                return pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False, quoting=3,
                                   encoding="utf-8", encoding_errors="replace")

        sub = table("SUBMISSION.TSV")
        sub = sub[sub["DOCUMENT_TYPE"].isin(FORM4_TYPES)]
        sub = sub[pd.to_numeric(sub["ISSUERCIK"], errors="coerce").isin(issuers)]
        if sub.empty:
            return []
        accessions = set(sub["ACCESSION_NUMBER"])
        owners = table("REPORTINGOWNER.TSV")
        owners = owners[owners["ACCESSION_NUMBER"].isin(accessions)]
        lines = table("NONDERIV_TRANS.TSV")
        lines = lines[lines["ACCESSION_NUMBER"].isin(accessions)]
    filings = {}
    for row in sub.to_dict("records"):
        filings[row["ACCESSION_NUMBER"]] = Filing(
            accession_no=row["ACCESSION_NUMBER"], cik=int(row["ISSUERCIK"]), document_type=row["DOCUMENT_TYPE"],
            filing_date=_date(row["FILING_DATE"]), period_of_report=_date(row.get("PERIOD_OF_REPORT")),
            date_of_original=_date(row.get("DATE_OF_ORIG_SUB")), is_10b5_1=flag(row.get("AFF10B5ONE")))
    for row in owners.to_dict("records"):
        relationship = {part.strip().lower() for part in row.get("RPTOWNER_RELATIONSHIP", "").split(",")}
        if str(row.get("RPTOWNERCIK", "")).strip().isdigit():
            filings[row["ACCESSION_NUMBER"]].owners.append(
                _owner(row["RPTOWNERCIK"], row.get("RPTOWNERNAME"), relationship, row.get("RPTOWNER_TITLE")))
    lines = lines.assign(_order=pd.to_numeric(lines.get("NONDERIV_TRANS_SK"), errors="coerce"))
    for accession, group in lines.sort_values("_order").groupby("ACCESSION_NUMBER", sort=False):
        for number, row in enumerate(group.to_dict("records"), start=1):
            filings[accession].lines.append({
                "line_no": number, "security_title": row.get("SECURITY_TITLE") or None,
                "transaction_date": _date(row.get("TRANS_DATE")), "transaction_code": row.get("TRANS_CODE") or None,
                "acquired_disposed": row.get("TRANS_ACQUIRED_DISP_CD") or None,
                "shares": _number(row.get("TRANS_SHARES")), "price": _number(row.get("TRANS_PRICEPERSHARE")),
                "shares_owned_after": _number(row.get("SHRS_OWND_FOLWNG_TRANS")),
                "direct_indirect": row.get("DIRECT_INDIRECT_OWNERSHIP") or None,
            })
    return [f for f in filings.values() if f.filing_date is not None]


# --- live filings -------------------------------------------------------------------------

def recent_form4(submissions: dict, after: date) -> list[dict]:
    """Form 4 / 4/A entries of a submissions list filed after `after`."""
    recent = submissions.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(recent.get("form", [])):
        filed = _date(recent["filingDate"][i])
        if form in FORM4_TYPES and filed and filed > after:
            out.append({"accession_no": recent["accessionNumber"][i], "form": form, "filing_date": filed,
                        "accepted": recent["acceptanceDateTime"][i], "document": recent["primaryDocument"][i]})
    return out


def document_url(cik: int, accession_no: str, primary_document: str) -> str:
    """The raw XML: the primary document without its xsl rendering folder (e.g. xslF345X06/)."""
    return (f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no.replace('-', '')}/"
            f"{primary_document.rsplit('/', 1)[-1]}")


def _utc(text: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).replace(
        tzinfo=None)


def parse_form4_xml(xml: bytes, entry: dict) -> Filing | None:
    """One live Form 4; None if it is not a Form 4 ownership document."""
    root = ET.fromstring(xml)
    if root.tag != "ownershipDocument":
        return None
    text = lambda node, path: (node.findtext(path) or "").strip()  # noqa: E731
    filing = Filing(
        accession_no=entry["accession_no"], cik=int(text(root, "issuer/issuerCik") or 0),
        document_type=text(root, "documentType") or entry["form"], filing_date=entry["filing_date"],
        period_of_report=_date(text(root, "periodOfReport")), accepted_at=_utc(entry["accepted"]),
        is_10b5_1=flag(text(root, "aff10b5One")))
    for owner in root.findall("reportingOwner"):
        relationship = {name.lower() for name, tag in (("director", "isDirector"), ("officer", "isOfficer"),
                                                        ("tenpercentowner", "isTenPercentOwner"), ("other", "isOther"))
                        if flag(text(owner, f"reportingOwnerRelationship/{tag}"))}
        cik = text(owner, "reportingOwnerId/rptOwnerCik")
        if cik.isdigit():
            filing.owners.append(_owner(cik, text(owner, "reportingOwnerId/rptOwnerName"), relationship,
                                        text(owner, "reportingOwnerRelationship/officerTitle")))
    for number, line in enumerate(root.findall("nonDerivativeTable/nonDerivativeTransaction"), start=1):
        filing.lines.append({
            "line_no": number, "security_title": text(line, "securityTitle/value") or None,
            "transaction_date": _date(text(line, "transactionDate/value")[:10]),
            "transaction_code": text(line, "transactionCoding/transactionCode") or None,
            "acquired_disposed": text(line, "transactionAmounts/transactionAcquiredDisposedCode/value") or None,
            "shares": _number(text(line, "transactionAmounts/transactionShares/value")),
            "price": _number(text(line, "transactionAmounts/transactionPricePerShare/value")),
            "shares_owned_after": _number(text(line, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")),
            "direct_indirect": text(line, "ownershipNature/directOrIndirectOwnership/value") or None,
        })
    return filing
