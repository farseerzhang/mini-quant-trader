"""Read-only broker clients (Trading 212, IBKR Flex); offline, with fake HTTP answers."""

import base64
import unittest

from quant.broker.ibkr_flex import IbkrFlexClient, IbkrFlexCredentials, IbkrFlexError, parse_statement
from quant.broker.trading212 import (
    Trading212Client,
    Trading212Credentials,
    Trading212Error,
    trading212_symbol,
)


class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, headers, params))
        return self.answers.pop(0)


class Trading212Test(unittest.TestCase):
    def test_account_positions_and_auth(self):
        session = FakeSession([
            FakeResponse(200, {"id": 123, "currencyCode": "EUR"}),
            FakeResponse(429),  # rate-limited once
            FakeResponse(200, {"free": 100.5, "total": 2500.0}),
            FakeResponse(200, [{"ticker": "AAPL_US_EQ", "quantity": 3.5, "averagePrice": 150, "currentPrice": 180,
                                "ppl": 90.1, "fxPpl": -2.0}]),
        ])
        sleeps = []
        client = Trading212Client(Trading212Credentials("KEY", "SECRET", "live"), session, sleep=sleeps.append)
        account = client.account()
        self.assertEqual((account.currency, account.cash_total, len(account.positions)), ("EUR", 2500.0, 1))
        self.assertEqual(account.positions[0].quantity, 3.5)
        expected = "Basic " + base64.b64encode(b"KEY:SECRET").decode()
        self.assertTrue(all(headers["Authorization"] == expected for _, headers, _ in session.calls))
        self.assertTrue(session.calls[0][0].startswith("https://live.trading212.com/api/v0/equity/account/info"))
        self.assertIn(10, sleeps)  # the retry pause after the 429

    def test_key_without_secret_and_refusal(self):
        self.assertEqual(Trading212Credentials("KEY", None, "demo").authorization_header(), "KEY")
        client = Trading212Client(Trading212Credentials("KEY", None, "demo"), FakeSession([FakeResponse(401)]),
                                  sleep=lambda s: None)
        with self.assertRaisesRegex(Trading212Error, "refused"):
            client.get("/equity/account/info")

    def test_symbols(self):
        self.assertEqual(trading212_symbol("AAPL_US_EQ"), "AAPL")
        self.assertEqual(trading212_symbol("BRK_B_US_EQ"), "BRK-B")
        self.assertEqual(trading212_symbol("FB_US_EQ"), "META")  # renamed stocks: today's ticker
        self.assertEqual(trading212_symbol("SQ_US_EQ"), "XYZ")
        self.assertEqual(trading212_symbol("BMWd_EQ"), "BMWd_EQ")  # Xetra listing: kept
        self.assertEqual(trading212_symbol("VUSAl_EQ"), "VUSAl_EQ")  # not a US listing: kept as it is


STATEMENT = """<FlexQueryResponse queryName="positions" type="AF"><FlexStatements count="1">
<FlexStatement accountId="U1234567" fromDate="20261016" toDate="20261016">
<AccountInformation accountId="U1234567" currency="USD"/>
<OpenPositions>
<OpenPosition symbol="NVDA" assetCategory="STK" position="10" markPrice="190.5" positionValue="1905"
  costBasisPrice="120" currency="USD" levelOfDetail="SUMMARY"/>
<OpenPosition symbol="NVDA" assetCategory="STK" position="10" markPrice="190.5" positionValue="1905"
  costBasisPrice="120" currency="USD" levelOfDetail="LOT"/>
<OpenPosition symbol="BRK B" assetCategory="STK" position="2" markPrice="480" positionValue="960" currency="USD"/>
</OpenPositions>
<CashReport><CashReportCurrency currency="BASE_SUMMARY" endingCash="5000.25"/>
<CashReportCurrency currency="USD" endingCash="5000.25"/></CashReport>
</FlexStatement></FlexStatements></FlexQueryResponse>"""


class IbkrFlexTest(unittest.TestCase):
    def test_parse_statement(self):
        account = parse_statement(STATEMENT)
        self.assertEqual((account.account_id, account.base_currency, account.report_date),
                         ("U1234567", "USD", "20261016"))
        self.assertEqual([p.symbol for p in account.positions], ["NVDA", "BRK B"])  # lot lines skipped
        self.assertEqual(account.ending_cash_base, 5000.25)

    def test_a_long_period_reads_the_end_and_every_deposit(self):
        statement = (STATEMENT.replace('fromDate="20261016"', 'fromDate="20251017"')
                     .replace("</CashReport>", "</CashReport><CashTransactions>"
                              '<CashTransaction type="Deposits/Withdrawals" amount="2916.19" currency="USD" '
                              'dateTime="20261008" transactionID="43299206406" levelOfDetail="DETAIL"/>'
                              '<CashTransaction type="Deposits/Withdrawals" amount="2916.19" currency="USD" '
                              'dateTime="20261008" levelOfDetail="SUMMARY"/>'
                              '<CashTransaction type="Dividends" amount="12" currency="USD" dateTime="20260601" '
                              'transactionID="9" levelOfDetail="DETAIL"/>'
                              "</CashTransactions>"))
        account = parse_statement(statement)
        self.assertEqual(account.report_date, "20261016")  # the snapshot is the period's last day
        self.assertEqual(account.ending_cash_base, 5000.25)
        self.assertEqual([(f.reference, f.date, f.amount) for f in account.cash_flows],
                         [("43299206406", "20261008", 2916.19)])  # no summary line, no dividend

    def test_two_steps_with_a_wait_while_generating(self):
        session = FakeSession([
            FakeResponse(200, text="<FlexStatementResponse><Status>Success</Status><ReferenceCode>99</ReferenceCode>"
                                   "</FlexStatementResponse>"),
            FakeResponse(200, text="<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode>"
                                   "<ErrorMessage>Statement generation in progress</ErrorMessage></FlexStatementResponse>"),
            FakeResponse(200, text=STATEMENT),
        ])
        account = IbkrFlexClient(IbkrFlexCredentials("TOKEN", "777"), session, sleep=lambda s: None).account()
        self.assertEqual(len(account.positions), 2)
        self.assertEqual(session.calls[0][2]["q"], "777")
        self.assertEqual(session.calls[1][2]["q"], "99")  # GetStatement asks with the reference code

    def test_a_refused_request(self):
        session = FakeSession([FakeResponse(200, text="<FlexStatementResponse><Status>Fail</Status><ErrorCode>1015"
                                                      "</ErrorCode><ErrorMessage>Token is invalid.</ErrorMessage>"
                                                      "</FlexStatementResponse>")])
        with self.assertRaisesRegex(IbkrFlexError, "1015 Token is invalid"):
            IbkrFlexClient(IbkrFlexCredentials("BAD", "777"), session, sleep=lambda s: None).account()


if __name__ == "__main__":
    unittest.main()
