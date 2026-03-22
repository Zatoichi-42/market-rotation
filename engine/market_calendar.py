"""
Market calendar — determines trading day status, staleness, next open.
Uses pandas business day logic + hardcoded NYSE holidays for 2025-2027.
"""
from datetime import date, timedelta
import pandas as pd

# NYSE holidays (2025-2027) — update annually
_NYSE_HOLIDAYS = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}

def is_trading_day(d: date) -> bool:
    """Check if a date is a NYSE trading day."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in _NYSE_HOLIDAYS:
        return False
    return True

def get_last_close(as_of: date) -> date:
    """Get the most recent trading day close on or before as_of."""
    d = as_of
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d

def get_next_open(as_of: date) -> date:
    """Get the next trading day after as_of."""
    d = as_of + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d

def get_market_status(as_of: date) -> dict:
    """
    Return market status for a given date.
    """
    trading = is_trading_day(as_of)
    last_close = get_last_close(as_of) if not trading else as_of
    next_open = get_next_open(as_of) if not trading else get_next_open(as_of)

    cal_days = (as_of - last_close).days
    # Trading days stale
    trading_days = 0
    d = last_close
    while d < as_of:
        d += timedelta(days=1)
        if is_trading_day(d):
            trading_days += 1

    if trading:
        reason = "trading_day"
        note = f"Market open. Data is live intraday."
    elif as_of.weekday() >= 5:
        reason = "weekend"
        note = f"Weekend. Data reflects {last_close.strftime('%A %Y-%m-%d')} close. Next open {next_open.strftime('%A %Y-%m-%d')}."
    else:
        reason = "holiday"
        note = f"Market holiday. Data reflects {last_close.strftime('%A %Y-%m-%d')} close. Next open {next_open.strftime('%A %Y-%m-%d')}."

    return {
        "is_trading_day": trading,
        "last_close": last_close.isoformat(),
        "next_open": next_open.isoformat(),
        "staleness_calendar_days": cal_days,
        "staleness_trading_days": trading_days,
        "reason": reason,
        "note": note,
    }
