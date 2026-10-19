# Mini Quant Trader — working notes for Claude

A local research and signal app for US equities (Python 3.12, pandas, SQLAlchemy Core, SQLite WAL, Streamlit, yfinance). It never places orders. The owner commits code; Claude does not commit unless asked.

## Where things are written down

- `proposal.md`: the main design, every pre-registered trial and its result, and the forward-test runs (Phase 10; run 2 since 2026-10-16, `real` read from the brokers).
- `altdata_proposal.md`: alternative data (Form 4, analyst changes, option chains). `sentiment_proposal.md`: YouTube creator opinions.
- `README.md`: commands and daily routine. Read the relevant section before changing a module.

## How to work

- **Plan first.** Write the plan or pre-registration into the proposal, wait for the owner's confirmation, then build.
- **Pre-registered trials are never changed after results are seen.** Record the result as it is, pass or fail, and do not re-test the same idea on the same data. Do not edit a registered section's text when compacting documents.
- **The holdout start is fixed at 2025-10-10.** The backtest's split rolls with the data, so a trial must not let viewed holdout sessions move into development.
- **Names:** long and descriptive, not abbreviated.
- **Language:** answer in the owner's language (English or Chinese); "in chinese" means explain again in Chinese. Documents and code stay in English.

## Data safety

- **`data/quant.db` is not disposable** (forward test since 2026-10-12): never rebuild it, never alter existing tables; new tables only. Run `python -m quant backup-database` before larger changes. Deleting rows needs the owner's explicit yes.
- **Broker access is read-only.** API keys live only in `.env` (git-ignored). Never ask for, read, print or store them; the owner must not paste them into chat.
- **Tests never call real services** (brokers, Yahoo, YouTube, SEC, `claude -p`) **and never write the real database.** Use fakes and in-memory databases; patch run-daily's broker and option stages (see `tests/test_cli_daily.py`).

## Commands

```bash
python -m quant run-daily                  # every evening after the US close: 7 stages; the last (option chains) is research only
python -m quant forward-report             # the forward test
python -m quant status
python -m quant backup-database
streamlit run dashboard/app.py             # restart it after changes in quant/ (Streamlit reloads only app.py)
```

Tests (PowerShell): `$env:PYTHONPATH="tests"; python -m unittest discover -s tests`. Run the full suite after changes; it takes about two minutes.

## Editing

- Windows, PowerShell. Some files use CRLF, others LF: keep each file's line endings (edit with a script that preserves them when the Edit tool would change them).
- The sentiment and altdata modules are research only. Nothing live reads them, and a failure there must never stop `run-daily`.
