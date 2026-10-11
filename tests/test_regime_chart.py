import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from quant.regime.chart import Ribbon, regime_chart_html, regime_chart_svg, runs



def _data():
    index = pd.bdate_range("2010-01-04", periods=600)
    price = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0004, 0.01, 600))), index=index)
    labels = np.array(["BULL"] * 200 + ["SIDEWAYS"] * 100 + ["BEAR"] * 50 + ["BULL"] * 250)
    regime = pd.Series(labels, index=index, dtype=object)
    return price, regime, index


class RunTests(unittest.TestCase):
    def test_consecutive_sessions_merge_into_runs(self):
        index = pd.bdate_range("2020-01-01", periods=6)
        regime = pd.Series(["BULL", "BULL", "BEAR", "BEAR", "BEAR", "BULL"], index=index)
        self.assertEqual(runs(regime), [(index[0], index[1], "BULL"), (index[2], index[4], "BEAR"), (index[5], index[5], "BULL")])


class ChartTests(unittest.TestCase):
    def test_svg_is_valid_with_one_rectangle_per_run_and_an_in_sample_marker(self):
        price, regime, index = _data()
        ribbons = [Ribbon("HMM regime", regime, in_sample_until=index[300]), Ribbon("Rule-based", regime)]
        svg = regime_chart_svg(price, ribbons, [("2011-01-03", "event")])
        root = ET.fromstring(svg)  # well-formed XML
        rects = [r for r in root.iter("{http://www.w3.org/2000/svg}rect") if "f-" in (r.get("class") or "")]
        self.assertEqual(len(rects), 2 * len(runs(regime)))
        faded = [r for r in rects if "faded" in r.get("class")]
        self.assertEqual(len(faded), 2)  # the HMM ribbon's two runs that start before the cut
        texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
        self.assertIn("in sample", texts)
        self.assertIn(f"out of sample from {index[300].date()}", texts)
        self.assertIn("event", texts)
        self.assertIn("2011", texts)  # year axis
        self.assertIn("@media (prefers-color-scheme: dark)", svg)

    def test_html_embeds_the_svg_legend_and_tooltip_data(self):
        price, regime, index = _data()
        page = regime_chart_html(price, [Ribbon("HMM regime", regime)], title="Test")
        self.assertIn("<svg", page)
        self.assertIn("Bull: HMM regime", page)
        self.assertIn('"ribbons":[{"label":"HMM regime"', page)
        self.assertNotIn("NaN", page)


if __name__ == "__main__":
    unittest.main()
