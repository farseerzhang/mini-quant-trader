"""NYSE session-date helpers for completed daily market data."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import exchange_calendars as xcals
import pandas as pd


# The calendar starts this much before the first date asked for: exchange_calendars rejects a
# range starting before the calendar's first session, so a calendar starting exactly on a
# weekend or holiday (e.g. "10 days ago" on a Saturday) would refuse that same date.
CALENDAR_MARGIN = timedelta(days=14)


def _calendar(calendar_name: str, start: date) -> xcals.ExchangeCalendar:
    """The calendar covering ``start`` onwards (from a margin before it).

    exchange_calendars only builds about 20 years back by default (2006-10 as of
    2026-10), which is shorter than market.initial_history, so ask for the range.
    """
    return xcals.get_calendar(calendar_name, start=pd.Timestamp((start - CALENDAR_MARGIN).isoformat()))


def last_completed_session(now: datetime | None, calendar_name: str) -> date:
    """Get the last NYSE session whose close has passed, including early closes."""
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now_utc.astimezone(timezone.utc)
    calendar = _calendar(calendar_name, now_utc.date() - timedelta(days=10))
    start = pd.Timestamp((now_utc.date() - timedelta(days=10)).isoformat())
    end = pd.Timestamp(now_utc.date().isoformat())
    sessions = calendar.sessions_in_range(start, end)
    completed = [
        session
        for session in sessions
        if calendar.session_close(session).to_pydatetime() <= now_utc
    ]
    if not completed:
        raise ValueError("Unable to determine the last completed NYSE session")
    return completed[-1].date()


def next_session_after(session_date: date, through: date, calendar_name: str) -> date | None:
    """Return the first NYSE session after session_date, up to and including through."""
    if session_date >= through:
        return None
    calendar = _calendar(calendar_name, session_date)
    sessions = calendar.sessions_in_range(
        pd.Timestamp((session_date + timedelta(days=1)).isoformat()),
        pd.Timestamp(through.isoformat()),
    )
    return sessions[0].date() if len(sessions) else None


def next_open_after(session_date: date, calendar_name: str) -> datetime:
    """The UTC time of the first market open after ``session_date``'s session.

    A signal computed at that session's close can be acted on until then; a forward test
    freezes a session only before this moment (Phase 10: no backfill).
    """
    calendar = _calendar(calendar_name, session_date)
    following = calendar.sessions_in_range(
        pd.Timestamp((session_date + timedelta(days=1)).isoformat()),
        pd.Timestamp((session_date + timedelta(days=15)).isoformat()),
    )
    if not len(following):
        raise ValueError(f"no {calendar_name} session found after {session_date}")
    return calendar.session_open(following[0]).to_pydatetime().astimezone(timezone.utc)


def sessions_between(first: date, last: date, calendar_name: str) -> list[date]:
    """Every session from ``first`` to ``last``, both included."""
    calendar = _calendar(calendar_name, first)
    return [s.date() for s in calendar.sessions_in_range(pd.Timestamp(first.isoformat()), pd.Timestamp(last.isoformat()))]
