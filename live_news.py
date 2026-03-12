# live_news.py
"""
Live Economic Calendar — MT5 Calendar API for NASDAQ-relevant events.
Fetches real-time US economic events and scores risk for trading decisions.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger('live_news')

# NASDAQ-critical US economic events (keywords to match)
HIGH_IMPACT_KEYWORDS = [
    'nonfarm', 'non-farm', 'nfp', 'payroll',
    'fomc', 'federal funds rate', 'interest rate decision',
    'cpi', 'consumer price index', 'inflation',
    'gdp', 'gross domestic product',
    'pce', 'core pce',
    'unemployment', 'jobless claims',
    'retail sales',
    'ppi', 'producer price',
    'ism manufacturing', 'ism services',
    'fed chair', 'powell',
]

MEDIUM_IMPACT_KEYWORDS = [
    'durable goods', 'housing starts', 'building permits',
    'consumer confidence', 'michigan sentiment',
    'trade balance', 'industrial production',
    'existing home', 'new home sales',
    'jolts', 'adp employment',
    'treasury', 'bond auction',
    'crude oil inventories', 'eia',
]


@dataclass
class LiveEvent:
    """A live economic calendar event."""
    timestamp: datetime
    name: str
    country: str
    impact: str       # HIGH, MEDIUM, LOW
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


class LiveNewsManager:
    """
    Real-time economic calendar using MT5's built-in calendar API.
    Falls back to keyword-based impact scoring if MT5 calendar unavailable.
    """

    def __init__(self):
        self.events: List[LiveEvent] = []
        self.last_refresh: Optional[datetime] = None
        self.refresh_interval = timedelta(hours=1)  # Refresh every hour
        self.mt5_available = False

        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.mt5_available = True
        except ImportError:
            logger.warning("MT5 not available — live news disabled")

    def refresh_events(self, force: bool = False) -> int:
        """Fetch upcoming events from MT5 calendar."""
        if not self.mt5_available:
            return 0

        now = datetime.now()
        if not force and self.last_refresh:
            if now - self.last_refresh < self.refresh_interval:
                return len(self.events)

        try:
            # Fetch events for next 24 hours
            from_date = datetime.utcnow()
            to_date = from_date + timedelta(hours=24)

            calendar_events = self.mt5.copy_ticks_from(
                "US100Cash", from_date, 1, self.mt5.COPY_TICKS_ALL
            )

            # Try MT5 calendar API
            try:
                raw_events = self._fetch_mt5_calendar(from_date, to_date)
            except Exception:
                raw_events = []

            if raw_events:
                self.events = raw_events
            else:
                # Fallback: use web scraping
                self.events = self._fetch_from_web()

            self.last_refresh = now
            logger.info("📅 Calendar refreshed: %d events in next 24h", len(self.events))

            # Log upcoming high-impact events
            for ev in self.events:
                if ev.impact == 'HIGH':
                    hours_until = (ev.timestamp - now).total_seconds() / 3600
                    if hours_until > 0:
                        logger.info("  ⚠️ HIGH: %s in %.1fh", ev.name, hours_until)

            return len(self.events)

        except Exception as e:
            logger.warning("Calendar refresh failed: %s", e)
            return 0

    def _fetch_mt5_calendar(self, from_date, to_date) -> List[LiveEvent]:
        """Fetch from MT5's built-in economic calendar."""
        events = []

        try:
            # MT5 calendar_get requires country code
            raw = self.mt5.calendar_get(from_date, to_date)
            if raw is None:
                return []

            for item in raw:
                country = getattr(item, 'country_id', 0)
                # Filter: US events only (country_id for US varies by broker)
                name = getattr(item, 'event_name', '') or getattr(item, 'name', '')
                time_val = getattr(item, 'time', None)

                if time_val is None:
                    continue

                impact = self._classify_impact(name)
                if impact == 'LOW':
                    continue  # Skip low-impact

                events.append(LiveEvent(
                    timestamp=datetime.fromtimestamp(time_val) if isinstance(time_val, (int, float)) else time_val,
                    name=name,
                    country='US',
                    impact=impact,
                    actual=str(getattr(item, 'actual_value', '')),
                    forecast=str(getattr(item, 'forecast_value', '')),
                    previous=str(getattr(item, 'prev_value', '')),
                ))

        except Exception as e:
            logger.debug("MT5 calendar_get error: %s", e)

        return events

    def _fetch_from_web(self) -> List[LiveEvent]:
        """Fallback: fetch from ForexFactory or similar (scraping)."""
        events = []
        try:
            import urllib.request
            import json

            # Use a free economic calendar API
            today = datetime.utcnow().strftime('%Y-%m-%d')
            url = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"

            req = urllib.request.Request(url, headers={
                'User-Agent': 'TradeBot/1.0'
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            now = datetime.utcnow()

            for item in data:
                country = item.get('country', '')
                if country != 'USD':
                    continue  # Only US events

                title = item.get('title', '')
                impact_str = item.get('impact', '').upper()

                # Parse date
                date_str = item.get('date', '')
                try:
                    event_time = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
                    event_time = event_time.replace(tzinfo=None)  # Remove tz for comparison
                except ValueError:
                    continue

                # Only future events
                if event_time < now - timedelta(hours=1):
                    continue

                # Map impact
                if impact_str in ('HIGH', 'HOLIDAY'):
                    impact = 'HIGH'
                elif impact_str == 'MEDIUM':
                    impact = 'MEDIUM'
                else:
                    # Also check by keyword
                    impact = self._classify_impact(title)
                    if impact == 'LOW':
                        continue

                events.append(LiveEvent(
                    timestamp=event_time,
                    name=title,
                    country='US',
                    impact=impact,
                    forecast=item.get('forecast', ''),
                    previous=item.get('previous', ''),
                ))

            events.sort(key=lambda e: e.timestamp)
            logger.info("Fetched %d USD events from web calendar", len(events))

        except Exception as e:
            logger.warning("Web calendar fetch failed: %s", e)

        return events

    def _classify_impact(self, event_name: str) -> str:
        """Classify event impact by keyword matching."""
        name_lower = event_name.lower()

        for keyword in HIGH_IMPACT_KEYWORDS:
            if keyword in name_lower:
                return 'HIGH'

        for keyword in MEDIUM_IMPACT_KEYWORDS:
            if keyword in name_lower:
                return 'MEDIUM'

        return 'LOW'

    def calculate_risk_score(self, current_time: datetime = None) -> Tuple[float, List[LiveEvent]]:
        """
        Calculate risk score based on upcoming events.
        Returns (score, list_of_upcoming_events).

        Score thresholds (same as backtest):
          < 4.0 → Normal (full size)
          4.0-6.9 → Reduced (half size)
          ≥ 7.0 → Blocked (no entry)
        """
        if current_time is None:
            current_time = datetime.now()

        # Auto-refresh if needed
        self.refresh_events()

        upcoming = []
        total_score = 0.0

        for event in self.events:
            hours_until = (event.timestamp - current_time).total_seconds() / 3600

            if hours_until < -1 or hours_until > 4:
                continue  # Only events in [-1h, +4h] window

            upcoming.append(event)

            # Time decay
            if hours_until < 1.0:
                time_mult = 3.0  # Immediate
            elif hours_until < 2.0:
                time_mult = 2.0  # Near
            else:
                time_mult = 1.0  # Far

            # Impact weight
            impact_mult = 2.0 if event.impact == 'HIGH' else 1.0

            total_score += time_mult * impact_mult

        return total_score, upcoming

    def get_status_line(self) -> str:
        """One-line status for logging."""
        score, events = self.calculate_risk_score()
        if not events:
            return "📅 No upcoming events"

        next_event = events[0]
        hours = (next_event.timestamp - datetime.now()).total_seconds() / 3600
        risk = "🔴 BLOCKED" if score >= 7.0 else "🟡 REDUCED" if score >= 4.0 else "🟢 NORMAL"

        return f"📅 {risk} (score={score:.1f}) | Next: {next_event.name} in {hours:.1f}h"


# Singleton
_live_news: Optional[LiveNewsManager] = None


def get_live_news() -> LiveNewsManager:
    global _live_news
    if _live_news is None:
        _live_news = LiveNewsManager()
    return _live_news
