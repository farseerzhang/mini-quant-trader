import dataclasses
import math
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd

from quant.forward.report import FIRST_LOOK, build_report, format_report, score
from quant.persistence import queries
from tests.test_forward_paper import WINDOW, PaperFixture
from tests.test_settings import PROJECT_SETTINGS


class ScoreTests(unittest.TestCase):
    def test_score_of_a_known_series(self):
        value = pd.Series([100.0, 110.0, 99.0, 108.9])
        result = score(value)
        returns = value.pct_change().dropna()
        self.assertAlmostEqual(result["total_return"], 0.089)
        self.assertAlmostEqual(result["max_drawdown"], 99 / 110 - 1)
        self.assertAlmostEqual(result["sharpe"], returns.mean() / returns.std() * math.sqrt(252))
        self.assertTrue(math.isnan(score(pd.Series([100.0]))["sharpe"]))


class ReportTests(PaperFixture):
    """The paper fixture (a frozen 30-session window) plus a real account with a trade log."""

    def setUp(self):
        super().setUp()
        self.trades = self.settings.positions.path.replace("positions.csv", "trades.csv")
        self.settings = dataclasses.replace(
            self.settings, positions=dataclasses.replace(self.settings.positions, trades_path=self.trades))

    def test_report_scores_every_portfolio(self):
        missed = WINDOW[10]
        self.freeze([s for s in WINDOW if s != missed])
        opens = queries.load_raw_opens(self.engine, ["AAPL"])
        day = WINDOW[5]
        with open(self.trades, "w", encoding="utf-8") as f:
            f.write("date,symbol,side,quantity,price,fees,amount\n")
            f.write(f"{day.date()},AAPL,BUY,100,{opens.loc[day, 'AAPL'] * 1.001:.6f},1,\n")
            f.write(f"{WINDOW[8].date()},AAPL,DIVIDEND,,,,10\n")
        report = build_report(self.engine, self.settings)
        self.assertEqual((report.started_on, report.sessions), (WINDOW[0], len(WINDOW) - 1))
        self.assertEqual(list(report.scores.index), ["live", "real", "SPY"])
        self.assertEqual(report.scores.loc["live", "missed"], 1)
        self.assertTrue(np.isnan(report.scores.loc["real", "missed"]))
        expected = report.values["live"].iloc[-1] / 50_000.0 - 1
        self.assertAlmostEqual(report.scores.loc["live", "total_return"], expected)
        self.assertEqual(len(report.costs), 1)
        self.assertAlmostEqual(report.costs["cost_bps"].iloc[0], (0.001 + 1 / (100 * opens.loc[day, "AAPL"] * 1.001))
                               * 10_000, places=3)
        self.assertTrue(((report.followed >= 0) & (report.followed <= 1 + 1e-9)).all())
        self.assertEqual(report.followed.index[0], WINDOW[1])  # day 1 compares with the start's target
        text = format_report(report)
        for part in ("Forward test run 1 since", f"{FIRST_LOOK - report.sessions} to go", "Followed share",
                     "Execution: 1 trades", "the backtest assumes 6 bp", "not advice"):
            self.assertIn(part, text)

    def test_command_before_the_start(self):
        from quant import cli

        output = StringIO()
        with redirect_stdout(output):
            code = cli.main(["--config", str(PROJECT_SETTINGS), "forward-report", "--db", str(self.engine.url.database)])
        self.assertEqual(code, 0)
        self.assertIn("is not running", output.getvalue())


if __name__ == "__main__":
    unittest.main()
