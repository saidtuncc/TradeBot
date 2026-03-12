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

# ═══════════════════════════════════════════════════════════════════
# NASDAQ-SPECIFIC IMPACT CLASSIFICATION
# Not all USD events affect NASDAQ equally. Fed/CPI/GDP are huge,
# but Building Permits or Trade Balance barely move tech stocks.
# ═══════════════════════════════════════════════════════════════════

# DEV impact → score multiplier 3.0 (tradı tamamen durdur)
NASDAQ_DEV_KEYWORDS = [
    'federal funds rate', 'interest rate decision', 'fomc',
    'fed chair', 'powell speaks', 'powell press',
    'nonfarm', 'non-farm', 'nfp', 'payroll',
    'cpi', 'consumer price index', 'core cpi',
    'core pce', 'pce price index',
]

# HIGH impact → score multiplier 2.0 (bekle, sonra momentum trade)
NASDAQ_HIGH_KEYWORDS = [
    'gdp', 'gross domestic product',
    'retail sales',
    'ppi', 'producer price',
    'unemployment', 'jobless claims', 'initial claims',
    'ism manufacturing', 'ism services', 'ism non-manufacturing',
    'michigan sentiment', 'consumer sentiment',
    'jolts', 'job openings',
]

# MEDIUM impact → score multiplier 1.0 (volume küçült)
NASDAQ_MEDIUM_KEYWORDS = [
    'adp employment', 'adp nonfarm',
    'consumer confidence',
    'industrial production',
    'existing home sales', 'new home sales',
    'crude oil inventories', 'eia',
    'philly fed', 'empire state',
]

# IGNORED — these barely affect NASDAQ:
# building permits, housing starts, trade balance,
# durable goods (core), treasury auctions, bond auctions


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
        """Fetch from ForexFactory calendar (faireconomy.media JSON API)."""
        events = []
        try:
            import urllib.request
            import ssl
            import json

            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

            # SSL context — try verified first, fallback to unverified
            try:
                ctx = ssl.create_default_context()
                req = urllib.request.Request(url, headers={'User-Agent': 'TradeBot/2.0'})
                with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                    data = json.loads(resp.read().decode())
            except ssl.SSLError:
                # VPS might lack root certificates — bypass SSL verification
                ctx = ssl._create_unverified_context()
                req = urllib.request.Request(url, headers={'User-Agent': 'TradeBot/2.0'})
                with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                    data = json.loads(resp.read().decode())

            now = datetime.utcnow()

            for item in data:
                country = item.get('country', '')
                if country != 'USD':
                    continue

                title = item.get('title', '')
                impact_raw = item.get('impact', '').strip()
                date_str = item.get('date', '')

                # Parse date — ForexFactory uses various formats
                event_time = None
                for fmt in ['%Y-%m-%dT%H:%M:%S%z',
                            '%Y-%m-%dT%H:%M:%S',
                            '%b %d, %Y %I:%M%p']:
                    try:
                        event_time = datetime.strptime(date_str.strip(), fmt)
                        if event_time.tzinfo:
                            event_time = event_time.replace(tzinfo=None)
                        break
                    except ValueError:
                        continue

                if event_time is None:
                    continue

                # Skip old events (>1h ago)
                if event_time < now - timedelta(hours=1):
                    continue

                # Use NASDAQ-specific classification (not FF generic impact)
                impact = self._classify_impact(title)
                if impact == 'LOW':
                    continue

                events.append(LiveEvent(
                    timestamp=event_time,
                    name=title,
                    country='US',
                    impact=impact,
                    actual=item.get('actual', ''),
                    forecast=item.get('forecast', ''),
                    previous=item.get('previous', ''),
                ))

            events.sort(key=lambda e: e.timestamp)
            if events:
                logger.info("📅 Fetched %d USD events from ForexFactory", len(events))

        except Exception as e:
            logger.warning("Web calendar fetch failed: %s", e)

        return events

    def _classify_impact(self, event_name: str) -> str:
        """Classify event impact on NASDAQ specifically."""
        name_lower = event_name.lower()

        for keyword in NASDAQ_DEV_KEYWORDS:
            if keyword in name_lower:
                return 'DEV'

        for keyword in NASDAQ_HIGH_KEYWORDS:
            if keyword in name_lower:
                return 'HIGH'

        for keyword in NASDAQ_MEDIUM_KEYWORDS:
            if keyword in name_lower:
                return 'MEDIUM'

        return 'LOW'

    def calculate_risk_score(self, current_time: datetime = None) -> Tuple[float, List[LiveEvent]]:
        """
        NASDAQ-specific risk score.

        Score thresholds:
          < 3.0 → Normal trading
          3.0-5.9 → Reduced volume (half)
          ≥ 6.0 → Wait for post-event momentum
        """
        if current_time is None:
            current_time = datetime.now()

        self.refresh_events()

        upcoming = []
        total_score = 0.0

        for event in self.events:
            hours_until = (event.timestamp - current_time).total_seconds() / 3600

            if hours_until < -1 or hours_until > 4:
                continue

            upcoming.append(event)

            # Time decay
            if hours_until < 1.0:
                time_mult = 3.0
            elif hours_until < 2.0:
                time_mult = 2.0
            else:
                time_mult = 1.0

            # NASDAQ impact weight
            if event.impact == 'DEV':
                impact_mult = 3.0
            elif event.impact == 'HIGH':
                impact_mult = 2.0
            else:
                impact_mult = 1.0

            total_score += time_mult * impact_mult

        return total_score, upcoming

    def get_status_line(self) -> str:
        """One-line status for logging."""
        score, events = self.calculate_risk_score()
        if not events:
            return "📅 No upcoming events"

        next_event = events[0]
        hours = (next_event.timestamp - datetime.now()).total_seconds() / 3600
        risk = "🔴 WAIT" if score >= 6.0 else "🟡 REDUCED" if score >= 3.0 else "🟢 CLEAR"

        return f"📅 {risk} (score={score:.1f}) | Next: {next_event.name} [{next_event.impact}] in {hours:.1f}h"


# Singleton
_live_news: Optional[LiveNewsManager] = None


def get_live_news() -> LiveNewsManager:
    global _live_news
    if _live_news is None:
        _live_news = LiveNewsManager()
    return _live_news
