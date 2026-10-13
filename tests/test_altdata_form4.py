"""SEC Form 4 (altdata_proposal.md A1): parsing, availability, idempotency, quality; offline."""

import io
import json
import unittest
import urllib.error
import zipfile
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import select

import quant.altdata.tables as tables
from quant.altdata.edgar import (
    SecClient,
    SecError,
    dataset_links,
    document_url,
    flag,
    parse_dataset,
    parse_form4_xml,
    quarter_end,
    recent_form4,
)
from quant.altdata.form4 import Availability, coverage, store_filings, update_form4
from quant.data.calendar import first_session_after, first_session_closing_after
from quant.persistence import tables as core_tables
from quant.persistence.database import create_db_engine, init_db
from settings_helper import SETTINGS

AAPL, OTHER = 320193, 999999
POOL = list(SETTINGS.market.stocks)
TICKERS = {str(i): {"cik_str": (AAPL if s == "AAPL" else 100000 + i), "ticker": s.replace("-", "."), "title": s}
           for i, s in enumerate(POOL)}


def _tsv(rows: list[dict]) -> str:
    columns = list(rows[0])
    return "\n".join(["\t".join(columns)] + ["\t".join(str(r[c]) for c in columns) for r in rows]) + "\n"


def dataset_zip() -> bytes:
    sub = [
        {"ACCESSION_NUMBER": "0001-26-000001", "FILING_DATE": "09-OCT-2026", "PERIOD_OF_REPORT": "07-OCT-2026",
         "DATE_OF_ORIG_SUB": "", "DOCUMENT_TYPE": "4", "ISSUERCIK": AAPL, "AFF10B5ONE": "true"},
        {"ACCESSION_NUMBER": "0001-26-000002", "FILING_DATE": "09-OCT-2026", "PERIOD_OF_REPORT": "08-OCT-2026",
         "DATE_OF_ORIG_SUB": "01-OCT-2026", "DOCUMENT_TYPE": "4/A", "ISSUERCIK": AAPL, "AFF10B5ONE": "0"},
        {"ACCESSION_NUMBER": "0001-26-000003", "FILING_DATE": "09-OCT-2026", "PERIOD_OF_REPORT": "",
         "DATE_OF_ORIG_SUB": "", "DOCUMENT_TYPE": "3", "ISSUERCIK": AAPL, "AFF10B5ONE": ""},  # Form 3: skipped
        {"ACCESSION_NUMBER": "0001-26-000004", "FILING_DATE": "09-OCT-2026", "PERIOD_OF_REPORT": "",
         "DATE_OF_ORIG_SUB": "", "DOCUMENT_TYPE": "4", "ISSUERCIK": OTHER, "AFF10B5ONE": ""},  # not the pool
    ]
    owners = [
        {"ACCESSION_NUMBER": "0001-26-000001", "RPTOWNERCIK": 11, "RPTOWNERNAME": "Cook Tim",
         "RPTOWNER_RELATIONSHIP": "Director,Officer", "RPTOWNER_TITLE": "CEO"},
        {"ACCESSION_NUMBER": "0001-26-000002", "RPTOWNERCIK": 12, "RPTOWNERNAME": "Fund LP",
         "RPTOWNER_RELATIONSHIP": "TenPercentOwner", "RPTOWNER_TITLE": ""},
        {"ACCESSION_NUMBER": "0001-26-000002", "RPTOWNERCIK": 13, "RPTOWNERNAME": "Fund GP",
         "RPTOWNER_RELATIONSHIP": "TenPercentOwner", "RPTOWNER_TITLE": ""},
    ]

    def line(accession, sk, day, code, shares, price, ad="D"):
        return {"ACCESSION_NUMBER": accession, "NONDERIV_TRANS_SK": sk, "SECURITY_TITLE": "Common Stock",
                "TRANS_DATE": day, "TRANS_CODE": code, "TRANS_SHARES": shares, "TRANS_PRICEPERSHARE": price,
                "TRANS_ACQUIRED_DISP_CD": ad, "SHRS_OWND_FOLWNG_TRANS": "1000", "DIRECT_INDIRECT_OWNERSHIP": "D"}

    lines = [
        line("0001-26-000001", 7, "07-OCT-2026", "S", "100", "250.5"),
        line("0001-26-000001", 5, "07-OCT-2026", "A", "50", "", "A"),  # grant, no price: fine
        line("0001-26-000002", 9, "08-OCT-2026", "P", "10", "0", "A"),  # P without price: quarantined
        line("0001-26-000002", 10, "12-OCT-2026", "P", "10", "250", "A"),  # after the filing date: quarantined
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SUBMISSION.tsv", _tsv(sub))
        archive.writestr("REPORTINGOWNER.tsv", _tsv(owners))
        archive.writestr("NONDERIV_TRANS.tsv", _tsv(lines))
    return buffer.getvalue()


FORM4_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType><periodOfReport>2026-10-12</periodOfReport>
  <issuer><issuerCik>0000320193</issuerCik></issuer>
  <aff10b5One>false</aff10b5One>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000021</rptOwnerCik><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CFO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <securityTitle><value>Common Stock</value></securityTitle>
    <transactionDate><value>2026-10-12</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>200</value></transactionShares>
      <transactionPricePerShare><value>251.00</value></transactionPricePerShare>
      <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
    <postTransactionAmounts><sharesOwnedFollowingTransaction><value>5200</value></sharesOwnedFollowingTransaction>
    </postTransactionAmounts>
    <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""

INDEX = ('<a href="/files/structureddata/data/insider-transactions-data-sets/2026q3_form345.zip">x</a>'
         '<a href="/files/structureddata/data/insider-transactions-data-sets/2005q4_form345.zip">old</a>')


class FakeClient:
    def __init__(self, submissions=None):
        self.urls = []
        self.requests = 0
        self.submissions = submissions or {}

    def get(self, url, timeout=120):
        self.urls.append(url)
        self.requests += 1
        if url.endswith("insider-transactions-data-sets"):
            return INDEX.encode()
        if url.endswith("_form345.zip"):
            return dataset_zip()
        if url.endswith(".xml"):
            return FORM4_XML
        raise SecError(f"unexpected {url}")

    def json(self, url):
        self.urls.append(url)
        if url.endswith("company_tickers.json"):
            return TICKERS
        cik = int(url.rsplit("CIK", 1)[1].split(".")[0])
        return self.submissions.get(cik, {"filings": {"recent": {"form": [], "accessionNumber": [], "filingDate": [],
                                                                  "acceptanceDateTime": [], "primaryDocument": []}}})


def _submissions(entries):
    keys = ("form", "accessionNumber", "filingDate", "acceptanceDateTime", "primaryDocument")
    return {"filings": {"recent": {k: [e[i] for e in entries] for i, k in enumerate(keys)}}}


class ParseTest(unittest.TestCase):
    def test_dataset(self):
        filings = {f.accession_no: f for f in parse_dataset(dataset_zip(), {AAPL})}
        self.assertEqual(set(filings), {"0001-26-000001", "0001-26-000002"})  # no Form 3, no other issuer
        first = filings["0001-26-000001"]
        self.assertEqual((first.filing_date, first.is_10b5_1, first.document_type), (date(2026, 10, 9), 1, "4"))
        self.assertEqual([line["transaction_code"] for line in first.lines], ["A", "S"])  # in SK order
        self.assertEqual(first.lines[1]["price"], 250.5)
        self.assertEqual(first.owners[0]["is_officer"] + first.owners[0]["is_director"], 2)
        amended = filings["0001-26-000002"]
        self.assertEqual((amended.date_of_original, amended.is_10b5_1, len(amended.owners)),
                         (date(2026, 10, 1), 0, 2))

    def test_flags_links_quarters(self):
        self.assertEqual([flag(v) for v in ("true", "1", "false", "0", "", None)], [1, 1, 0, 0, None, None])
        links = dataset_links(INDEX)
        self.assertEqual(list(links), ["2005q4", "2026q3"])
        self.assertTrue(links["2026q3"].startswith("https://www.sec.gov/files/"))
        self.assertEqual(quarter_end("2026q3"), date(2026, 9, 30))
        self.assertEqual(document_url(AAPL, "0001140361-26-039158", "xslF345X06/form4.xml"),
                         "https://www.sec.gov/Archives/edgar/data/320193/000114036126039158/form4.xml")

    def test_live(self):
        entries = recent_form4(_submissions([
            ("4", "0002-26-000001", "2026-10-12", "2026-10-12T21:05:00.000Z", "xslF345X06/form4.xml"),
            ("8-K", "0002-26-000002", "2026-10-12", "2026-10-12T21:05:00.000Z", "x.htm"),
            ("4", "0002-26-000003", "2026-09-30", "2026-09-30T21:05:00.000Z", "form4.xml"),
        ]), after=date(2026, 9, 30))
        self.assertEqual([e["accession_no"] for e in entries], ["0002-26-000001"])
        filing = parse_form4_xml(FORM4_XML, entries[0])
        self.assertEqual((filing.cik, filing.is_10b5_1, filing.accepted_at), (AAPL, 0, datetime(2026, 10, 12, 21, 5)))
        self.assertEqual(filing.lines[0]["transaction_code"], "P")
        self.assertEqual(filing.owners[0]["officer_title"], "CFO")
        self.assertIsNone(parse_form4_xml(b"<edgarSubmission/>", entries[0]))


class AvailabilityTest(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(first_session_after(date(2026, 10, 9), "XNYS"), date(2026, 10, 12))  # Fri -> Mon
        utc = timezone.utc
        # Before Monday's 16:00 ET close (20:00 UTC): usable at that close; after it: Tuesday.
        self.assertEqual(first_session_closing_after(datetime(2026, 10, 12, 19, 0, tzinfo=utc), "XNYS"),
                         date(2026, 10, 12))
        self.assertEqual(first_session_closing_after(datetime(2026, 10, 12, 21, 0, tzinfo=utc), "XNYS"),
                         date(2026, 10, 13))
        availability = Availability("XNYS")
        self.assertEqual(availability.dataset(date(2026, 10, 9)), date(2026, 10, 12))
        self.assertEqual(availability.live(datetime(2026, 10, 12, 21, 5)), date(2026, 10, 13))


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(":memory:", 1000)
        init_db(self.engine)

    def tearDown(self):
        self.engine.dispose()

    def rows(self, table):
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(table)).mappings().all()]

    def test_update_backfill_then_live_and_idempotent(self):
        submissions = {AAPL: _submissions([
            ("4", "0002-26-000001", "2026-10-12", "2026-10-12T21:05:00.000Z", "xslF345X06/form4.xml"),
            ("4", "0001-26-000001", "2026-10-09", "2026-10-09T20:30:00.000Z", "form4.xml"),  # already in the data set
        ])}
        client = FakeClient(submissions)
        result = update_form4(self.engine, SETTINGS, client)
        self.assertEqual(result.quarters_loaded, ["2026q3"])  # 2005q4 is before insider_backfill_from
        self.assertEqual(result.live_filings, 1)
        self.assertEqual(result.quarantined, 2)  # the P without a price and the one after its filing date
        filings = {r["accession_no"]: r for r in self.rows(tables.altdata_insider_filings)}
        self.assertEqual(set(filings), {"0001-26-000001", "0001-26-000002", "0002-26-000001"})
        self.assertEqual(filings["0001-26-000001"]["available_on"], date(2026, 10, 12))  # data set: next session
        self.assertEqual(filings["0002-26-000001"]["available_on"], date(2026, 10, 13))  # live: after the close
        self.assertEqual(filings["0001-26-000001"]["symbol"], "AAPL")
        self.assertEqual(len(self.rows(tables.altdata_insider_owners)), 4)
        lines = self.rows(tables.altdata_insider_transactions)
        self.assertEqual(sorted(l["transaction_code"] for l in lines), ["A", "P", "S"])
        with self.engine.connect() as conn:
            reasons = [r.reason for r in conn.execute(select(core_tables.data_quarantine.c.reason))]
        self.assertEqual(len(reasons), 2)

        again = update_form4(self.engine, SETTINGS, FakeClient(submissions))
        self.assertEqual((again.quarters_loaded, again.live_filings), ([], 0))  # nothing new
        self.assertEqual(len(self.rows(tables.altdata_insider_filings)), 3)
        self.assertEqual(len(self.rows(tables.altdata_insider_transactions)), 3)

        reloaded = update_form4(self.engine, SETTINGS, FakeClient(submissions), reload=True, live=False)
        self.assertEqual(reloaded.quarters_loaded, ["2026q3"])
        self.assertEqual(len(self.rows(tables.altdata_insider_filings)), 3)  # replaced, not doubled

        table = coverage(self.engine)
        self.assertEqual(int(table["filings"].sum()), 3)
        self.assertEqual(int(table["P"].sum()), 1)

    def test_live_waits_until_the_backfill_is_complete(self):
        global INDEX
        old, INDEX = INDEX, INDEX + '<a href="/files/x/insider-transactions-data-sets/2026q2_form345.zip">y</a>'
        try:
            client = FakeClient({AAPL: _submissions([
                ("4", "0002-26-000001", "2026-10-12", "2026-10-12T21:05:00.000Z", "form4.xml")])})
            result = update_form4(self.engine, SETTINGS, client, max_quarters=1)
            self.assertEqual(result.quarters_loaded, ["2026q2"])
            self.assertEqual(result.live_filings, 0)
            self.assertFalse(any("submissions" in url for url in client.urls))  # no live requests at all
        finally:
            INDEX = old

    def test_live_keeps_its_exact_time_when_the_quarter_arrives(self):
        from quant.altdata.edgar import Filing

        live = Filing(accession_no="0001-26-000001", cik=AAPL, document_type="4", filing_date=date(2026, 10, 9),
                      accepted_at=datetime(2026, 10, 9, 19, 0), lines=[])
        availability = Availability("XNYS")
        store_filings(self.engine, [live], {AAPL: "AAPL"}, "live", "live", availability, pd.DataFrame())
        store_filings(self.engine, parse_dataset(dataset_zip(), {AAPL}), {AAPL: "AAPL"}, "dataset", "2026q3",
                      availability, pd.DataFrame())
        filings = {r["accession_no"]: r for r in self.rows(tables.altdata_insider_filings)}
        self.assertEqual(filings["0001-26-000001"]["source"], "live")
        self.assertEqual(filings["0001-26-000001"]["available_on"], date(2026, 10, 9))  # before the close

    def test_price_far_from_the_close_is_quarantined(self):
        closes = pd.DataFrame({"AAPL": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-10-07")]))
        _, lines, bad = store_filings(self.engine, parse_dataset(dataset_zip(), {AAPL}), {AAPL: "AAPL"}, "dataset",
                                      "2026q3", Availability("XNYS"), closes)
        self.assertEqual(bad, 3)  # the S at 250.5 vs a close of 100, plus the two bad P lines
        self.assertEqual(lines, 1)  # only the grant


class ClientTest(unittest.TestCase):
    def test_rate_limit_and_retry(self):
        sleeps, now = [], [0.0]
        responses = [urllib.error.HTTPError("u", 429, "slow down", {}, None), b"ok", b"ok2"]

        class Response:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return self.body

        def opener(request, timeout):
            self.assertIn("chenghao", request.headers["User-agent"])
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return Response(item)

        client = SecClient(SETTINGS.altdata.user_agent, 5, sleep=lambda s: sleeps.append(s) or None,
                           clock=lambda: now[0], opener=opener)
        self.assertEqual(client.get("https://www.sec.gov/x"), b"ok")
        self.assertEqual(client.get("https://www.sec.gov/y"), b"ok2")
        self.assertIn(10, sleeps)  # the pause after the 429
        self.assertTrue(any(abs(s - 0.2) < 1e-9 for s in sleeps))  # 5 requests per second

    def test_gives_up(self):
        def opener(request, timeout):
            raise urllib.error.HTTPError("u", 404, "missing", {}, None)

        with self.assertRaises(SecError):
            SecClient("ua x@y", 5, sleep=lambda s: None, opener=opener).get("https://www.sec.gov/x")


if __name__ == "__main__":
    unittest.main()
