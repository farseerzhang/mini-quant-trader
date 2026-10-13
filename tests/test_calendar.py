"""NYSE session helpers: ranges that start on a weekend or holiday."""

import unittest
from datetime import date, datetime, timezone

from quant.data.calendar import last_completed_session, next_open_after, next_session_after, sessions_between

XNYS = "XNYS"


class WeekendStartTest(unittest.TestCase):
    def test_ten_days_back_is_a_saturday(self):
        # 2026-10-13 minus 10 days is Saturday 2026-10-03; this used to raise in run-daily.
        now = datetime(2026, 10, 13, 22, 0, tzinfo=timezone.utc)  # after the close
        self.assertEqual(last_completed_session(now, XNYS), date(2026, 10, 13))
        before_close = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(last_completed_session(before_close, XNYS), date(2026, 10, 12))

    def test_ranges_starting_on_a_weekend(self):
        saturday = date(2026, 10, 3)
        self.assertEqual(sessions_between(saturday, date(2026, 10, 6), XNYS), [date(2026, 10, 5), date(2026, 10, 6)])
        self.assertEqual(next_session_after(saturday, date(2026, 10, 9), XNYS), date(2026, 10, 5))
        self.assertEqual(next_open_after(saturday, XNYS), datetime(2026, 10, 5, 13, 30, tzinfo=timezone.utc))

    def test_every_start_day_of_a_month(self):
        for day in range(1, 32):
            now = datetime(2026, 10, day, 23, 0, tzinfo=timezone.utc)
            with self.subTest(day=day):
                session = last_completed_session(now, XNYS)
                self.assertLessEqual(session, now.date())


if __name__ == "__main__":
    unittest.main()
