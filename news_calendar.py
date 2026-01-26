# news_calendar.py
# Static news blackout filter for high-impact USD events

from datetime import datetime, time
from typing import List
import config

def get_first_friday(year: int, month: int) -> int:
    """Return the day of the first Friday of the given month."""
    from calendar import monthcalendar, FRIDAY
    cal = monthcalendar(year, month)
    # First week that has a Friday
    for week in cal:
        if week[FRIDAY] != 0:
            return week[FRIDAY]
    return 1  # Fallback

def is_nfp_day(dt: datetime) -> bool:
    """Check if the given date is NFP release day (first Friday of month)."""
    return dt.day == get_first_friday(dt.year, dt.month) and dt.weekday() == 4

def is_fomc_day(dt: datetime) -> bool:
    """Check if the given date is an FOMC announcement day."""
    date_str = dt.strftime('%Y-%m-%d')
    return date_str in config.FOMC_DATES

def is_in_news_blackout(dt: datetime) -> bool:
    """
    Check if the given datetime falls within a news blackout window.
    
    Args:
        dt: Datetime in UTC
        
    Returns:
        True if trading should be avoided, False otherwise
    """
    hour = dt.hour
    
    # Check FOMC days
    if is_fomc_day(dt):
        if config.NEWS_BLACKOUT_HOURS['fomc_buffer_start'] <= hour <= config.NEWS_BLACKOUT_HOURS['fomc_buffer_end']:
            return True
    
    # Check NFP days (first Friday of month)
    if is_nfp_day(dt):
        if config.NEWS_BLACKOUT_HOURS['nfp_buffer_start'] <= hour <= config.NEWS_BLACKOUT_HOURS['nfp_buffer_end']:
            return True
    
    # CPI/PPI typically mid-month - simplified check
    # In production, integrate with actual economic calendar
    if dt.day in [10, 11, 12, 13, 14, 15]:  # Approximate CPI/PPI window
        if config.NEWS_BLACKOUT_HOURS['cpi_buffer_start'] <= hour <= config.NEWS_BLACKOUT_HOURS['cpi_buffer_end']:
            # Only on weekdays
            if dt.weekday() < 5:
                return True
    
    return False

def get_blackout_dates_in_range(start: datetime, end: datetime) -> List[datetime]:
    """
    Return all dates with news blackouts in the given range.
    Useful for backtest visualization.
    """
    blackout_dates = []
    current = start
    while current <= end:
        if is_fomc_day(current) or is_nfp_day(current):
            blackout_dates.append(current)
        current = current + timedelta(days=1)
    return blackout_dates

# Fix missing import
from datetime import timedelta
