# Labeling the gold set

The answer key for opinion extraction (sentiment_proposal.md section 4.6). Label **before**
you see any extraction output for these videos.

For each video in `videos.csv`:

1. Read `data\sentiment_gold\transcripts/<video_id>.txt` (the `transcript` column; skim it, and open the
   link if a passage is unclear).
2. Add one row to `opinions.csv` for every opinion **the creator holds themselves**.
3. Set the video's `status` in `videos.csv` to `done`, or `done-none` if it has no opinions.
   Only videos marked done / done-none are scored, so partial work counts.

## opinions.csv

| Column | Values |
|---|---|
| video_id | from videos.csv |
| ticker | US ticker, e.g. NVDA, BRK-B; ETFs and indices too (SPY, QQQ) |
| stance | BULLISH / BEARISH / NEUTRAL / CONDITIONAL |
| horizon | SHORT (weeks) / MEDIUM (months) / LONG (a year or more) / UNSPECIFIED |
| condition | only for CONDITIONAL, e.g. "buy below 140" |
| conviction | 1 mild, 2 clear, 3 strong (how strongly they say it) |
| action | BOUGHT / SOLD / HOLDING / NONE: what the creator says they did themselves |
| time | where it is said, mm:ss (or h:mm:ss) from the transcript |
| note | optional: anything you are unsure about |

## Rules

- Count only the creator's **own** view ("I think", "I'm buying", "I'd avoid").
- Do not count relayed views (analyst ratings, news, viewer questions) or passing mentions
  without a view.
- "Bearish short term, bullish long term" is **two rows** with different horizons.
- One row per stock and horizon per video; if they come back to it, keep their final view.
- When unsure, label your best guess and say so in `note`; do not skip it.

Back this folder up: it is hours of work, and every later prompt or model change is scored
against it. The transcripts folder can be deleted; `quant sentiment gold-export` writes it again.
