"""Regime timeline chart for the REGIME report: SPY (log scale) above one ribbon per regime.

``regime_chart_svg`` returns a static SVG (embedded in the Markdown report; it
follows the viewer's light/dark setting through a media query). ``regime_chart_html``
wraps the same SVG with a legend and a hover tooltip. Colors are the first three
slots of the validated categorical palette (blue, orange, aqua; checked with the
dataviz validator in both modes): BULL blue, BEAR orange, SIDEWAYS aqua. A ribbon's
sessions before ``in_sample_until`` are drawn faded and marked as in sample.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass

import pandas as pd

LABELS = ("BULL", "SIDEWAYS", "BEAR")
NAMES = {"BULL": "Bull", "SIDEWAYS": "Sideways", "BEAR": "Bear"}
LIGHT = {"BULL": "#2a78d6", "SIDEWAYS": "#1baf7a", "BEAR": "#eb6834",
         "line": "#0b0b0b", "grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781", "ink2": "#52514e"}
DARK = {"BULL": "#3987e5", "SIDEWAYS": "#199e70", "BEAR": "#d95926",
        "line": "#ffffff", "grid": "#2c2c2a", "axis": "#383835", "muted": "#898781", "ink2": "#c3c2b7"}

WIDTH, LEFT, RIGHT, TOP, PLOT_BOTTOM, RIBBON_HEIGHT, RIBBON_GAP = 900, 96, 884, 44, 284, 22, 10
EVENT_ROWS = (14, 30)  # event labels alternate rows when they would overlap
LABEL_WIDTH_PER_CHAR = 6.2  # approximate width of an 11px label character


@dataclass(frozen=True)
class Ribbon:
    label: str
    regime: pd.Series  # regime per session (BULL / SIDEWAYS / BEAR)
    in_sample_until: pd.Timestamp | None = None  # sessions before this are in sample


def runs(regime: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Consecutive sessions with the same regime: (first, last, regime)."""
    out: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    regime = regime.dropna()
    for stamp, value in regime.items():
        if out and out[-1][2] == value:
            out[-1] = (out[-1][0], stamp, value)
        else:
            out.append((stamp, stamp, value))
    return out


def _weekly(price: pd.Series) -> pd.Series:
    calendar = price.index.isocalendar()
    return price.groupby([calendar.year.to_numpy(), calendar.week.to_numpy()]).tail(1)


def _geometry(price: pd.Series, ribbons: list[Ribbon]):
    start, end = price.index[0], price.index[-1]
    span = (end - start).total_seconds()
    x = lambda stamp: LEFT + (pd.Timestamp(stamp) - start).total_seconds() / span * (RIGHT - LEFT)  # noqa: E731
    low, high = math.log(price.min() * 0.92), math.log(price.max() * 1.05)
    y = lambda value: PLOT_BOTTOM - (math.log(value) - low) / (high - low) * (PLOT_BOTTOM - TOP)  # noqa: E731
    height = PLOT_BOTTOM + 22 + len(ribbons) * (RIBBON_HEIGHT + RIBBON_GAP) + 26
    return x, y, height


def _price_ticks(low: float, high: float) -> list[float]:
    ticks = []
    for magnitude in range(int(math.floor(math.log10(low))), int(math.ceil(math.log10(high))) + 1):
        for step in (1, 2, 5):
            value = step * 10**magnitude
            if low <= value <= high:
                ticks.append(value)
    return ticks


def regime_chart_svg(price: pd.Series, ribbons: list[Ribbon], events: list[tuple[str, str]] = ()) -> str:
    """Static SVG: ``price`` (e.g. SPY total-return close) and one ribbon per regime series."""
    price = price.dropna()
    weekly = _weekly(price)
    x, y, height = _geometry(price, ribbons)
    one_day = pd.Timedelta(days=1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:.0f}" width="{WIDTH}" '
        f'role="img" aria-label="SPY price with market regime ribbons">',
        "<title>SPY price (log scale) with market regime ribbons</title>",
        "<style>",
        *[f".f-{k}{{fill:{LIGHT[k]}}}" for k in LABELS],
        f".line{{stroke:{LIGHT['line']}}} .grid{{stroke:{LIGHT['grid']}}} .axis{{stroke:{LIGHT['axis']}}} "
        f".muted{{fill:{LIGHT['muted']}}} .ink2{{fill:{LIGHT['ink2']}}} .bg{{fill:#fcfcfb}}",
        "@media (prefers-color-scheme: dark){",
        *[f".f-{k}{{fill:{DARK[k]}}}" for k in LABELS],
        f".line{{stroke:{DARK['line']}}} .grid{{stroke:{DARK['grid']}}} .axis{{stroke:{DARK['axis']}}} "
        f".muted{{fill:{DARK['muted']}}} .ink2{{fill:{DARK['ink2']}}} .bg{{fill:#1a1a19}}",
        "}",
        "text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}",
        ".faded{opacity:.4}",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{WIDTH}" height="{height:.0f}"/>',
    ]
    low, high = price.min() * 0.92, price.max() * 1.05
    for value in _price_ticks(low, high):
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{RIGHT}" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke-width="1"/>')
        parts.append(f'<text class="muted" x="{LEFT - 8}" y="{y(value) + 4:.1f}" text-anchor="end" font-size="11">{value:g}</text>')
    parts.append(f'<line class="axis" x1="{LEFT}" x2="{RIGHT}" y1="{PLOT_BOTTOM}" y2="{PLOT_BOTTOM}" stroke-width="1"/>')
    row_end = [-1e9] * len(EVENT_ROWS)  # right edge of the last label in each row
    for stamp, name in events:
        if price.index[0] <= pd.Timestamp(stamp) <= price.index[-1]:
            xx = x(stamp)
            half = len(name) * LABEL_WIDTH_PER_CHAR / 2
            row = next((i for i, end in enumerate(row_end) if xx - half > end + 8), len(EVENT_ROWS) - 1)
            row_end[row] = xx + half
            label_y = EVENT_ROWS[row]
            parts.append(f'<text class="muted" x="{xx:.1f}" y="{label_y}" text-anchor="middle" font-size="11">{html.escape(name)}</text>')
            parts.append(f'<line class="axis" x1="{xx:.1f}" x2="{xx:.1f}" y1="{label_y + 3}" y2="{TOP - 4}" stroke-width="1"/>')
    path = "".join(f"{'L' if i else 'M'}{x(s):.1f} {y(v):.1f}" for i, (s, v) in enumerate(weekly.items()))
    parts.append(f'<path class="line" d="{path}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')

    ribbon_top = PLOT_BOTTOM + 22
    for i, ribbon in enumerate(ribbons):
        top = ribbon_top + i * (RIBBON_HEIGHT + RIBBON_GAP)
        parts.append(f'<text class="ink2" x="{LEFT - 8}" y="{top + RIBBON_HEIGHT / 2 + 4:.1f}" text-anchor="end" font-size="12">{html.escape(ribbon.label)}</text>')
        for first, last, value in runs(ribbon.regime.loc[price.index[0] : price.index[-1]]):
            xa, xb = x(first), x(last + one_day)
            faded = ribbon.in_sample_until is not None and first < ribbon.in_sample_until
            parts.append(f'<rect class="f-{value}{" faded" if faded else ""}" x="{xa:.2f}" y="{top}" '
                         f'width="{max(xb - xa - 1, 1):.2f}" height="{RIBBON_HEIGHT}"/>')
        if ribbon.in_sample_until is not None and price.index[0] < ribbon.in_sample_until <= price.index[-1]:
            # The divider runs through the ribbon; its labels sit in the gap above it, off the colors.
            xx = x(ribbon.in_sample_until)
            parts.append(f'<line class="line" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top - 12}" y2="{top + RIBBON_HEIGHT + 2}" stroke-width="1.5"/>')
            parts.append(f'<text class="ink2" x="{xx - 6:.1f}" y="{top - 4}" text-anchor="end" font-size="11">in sample</text>')
            parts.append(f'<text class="ink2" x="{xx + 6:.1f}" y="{top - 4}" font-size="11">out of sample from {ribbon.in_sample_until.date()}</text>')

    axis_y = ribbon_top + len(ribbons) * (RIBBON_HEIGHT + RIBBON_GAP)
    for year in range(price.index[0].year + 1, price.index[-1].year + 1):
        xx = x(pd.Timestamp(f"{year}-01-01"))
        parts.append(f'<line class="axis" x1="{xx:.1f}" x2="{xx:.1f}" y1="{axis_y - 6}" y2="{axis_y - 2}" stroke-width="1"/>')
        parts.append(f'<text class="muted" x="{xx:.1f}" y="{axis_y + 12}" text-anchor="middle" font-size="11">{year}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def regime_chart_html(price: pd.Series, ribbons: list[Ribbon], events: list[tuple[str, str]] = (), title: str = "") -> str:
    """The SVG with a legend and a hover tooltip (date, SPY, every ribbon's regime)."""
    price = price.dropna()
    weekly = _weekly(price)
    x, y, _ = _geometry(price, ribbons)
    shares = {r.label: {k: float((r.regime.loc[price.index[0]:price.index[-1]] == k).mean()) for k in LABELS} for r in ribbons}
    data = {
        "points": [[s.strftime("%Y-%m-%d"), round(float(v), 2), round(x(s), 2), round(y(v), 2)] for s, v in weekly.items()],
        "ribbons": [{"label": r.label, "runs": [[a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), k]
                                                 for a, b, k in runs(r.regime.loc[price.index[0]:price.index[-1]])],
                     "inSampleUntil": r.in_sample_until.strftime("%Y-%m-%d") if r.in_sample_until is not None else None}
                    for r in ribbons],
        "left": LEFT, "right": RIGHT, "width": WIDTH, "top": TOP,
    }
    legend = "".join(
        f'<span><i class="sw f-{k}"></i>{NAMES[k]}: '
        + " · ".join(f"{html.escape(label)} {share[k]:.0%}" for label, share in shares.items())
        + " of days</span>"
        for k in LABELS
    )
    swatch_css = "".join(f".sw.f-{k}{{background:{LIGHT[k]}}}" for k in LABELS)
    swatch_dark = "".join(f".sw.f-{k}{{background:{DARK[k]}}}" for k in LABELS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title or 'Market regimes')}</title>
<style>
body{{margin:24px;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#f9f9f7;color:#0b0b0b}}
h1{{font-size:18px;font-weight:500;margin:0 0 4px}} p{{color:#52514e;font-size:13px;margin:0 0 12px}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:#52514e;margin:0 0 8px}}
.legend span{{display:flex;align-items:center;gap:6px}} .sw{{width:10px;height:10px;border-radius:2px;display:inline-block}}
{swatch_css}
.wrap{{position:relative;max-width:{WIDTH}px}} .wrap svg{{width:100%;height:auto;display:block}}
.tip{{position:absolute;pointer-events:none;background:#fff;border:1px solid rgba(11,11,11,.2);border-radius:8px;padding:8px 10px;font-size:12px;line-height:1.5;white-space:nowrap;display:none}}
@media (prefers-color-scheme: dark){{body{{background:#0d0d0d;color:#fff}} p,.legend{{color:#c3c2b7}} .tip{{background:#1a1a19;border-color:rgba(255,255,255,.2)}} {swatch_dark}}}
</style></head><body>
<h1>{html.escape(title or 'Market regimes')}</h1>
<p>SPY total-return price (log scale) with one ribbon per regime. Hover for the week's values. Faded ribbon sections are in sample.</p>
<div class="legend">{legend}</div>
<div class="wrap" id="wrap">{regime_chart_svg(price, ribbons, events)}<div class="tip" id="tip"></div></div>
<script>
const D={json.dumps(data, separators=(",", ":"))};
const svg=document.querySelector("#wrap svg"),tip=document.getElementById("tip"),NS="http://www.w3.org/2000/svg";
const cross=document.createElementNS(NS,"line");cross.setAttribute("class","axis");cross.setAttribute("stroke-width","1");cross.style.display="none";svg.appendChild(cross);
const day=s=>Date.parse(s+"T00:00:00Z");
const at=(runs,v)=>{{for(const [a,b,k] of runs){{if(v>=day(a)&&v<=day(b)+864e5)return k;}}return null;}};
const N={{BULL:"Bull",SIDEWAYS:"Sideways",BEAR:"Bear"}};
svg.addEventListener("mousemove",e=>{{const r=svg.getBoundingClientRect(),vb=svg.viewBox.baseVal,sx=(e.clientX-r.left)*vb.width/r.width;
if(sx<D.left||sx>D.right){{tip.style.display="none";cross.style.display="none";return;}}
let best=D.points[0];for(const p of D.points)if(Math.abs(p[2]-sx)<Math.abs(best[2]-sx))best=p;
cross.setAttribute("x1",best[2]);cross.setAttribute("x2",best[2]);cross.setAttribute("y1",D.top);cross.setAttribute("y2",vb.height-26);cross.style.display="";
const rows=D.ribbons.map(rb=>{{const k=at(rb.runs,day(best[0]));const ins=rb.inSampleUntil&&best[0]<rb.inSampleUntil?" (in sample)":"";return `<div>${{rb.label}}: ${{k?`<i class="sw f-${{k}}" style="display:inline-block;margin-right:4px"></i>${{N[k]}}`:"-"}}${{ins}}</div>`;}}).join("");
tip.innerHTML=`<div>${{best[0]}}</div><div>SPY ${{best[1].toFixed(0)}}</div>${{rows}}`;tip.style.display="block";
const px=best[2]*r.width/vb.width;tip.style.left=(px>r.width*.6?px-tip.offsetWidth-12:px+12)+"px";tip.style.top="8px";}});
svg.addEventListener("mouseleave",()=>{{tip.style.display="none";cross.style.display="none";}});
</script></body></html>
"""
