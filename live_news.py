# live_news.py
"""
Live Economic Calendar — ForexFactory API for NASDAQ-relevant events.
Fetches weekly US economic events and scores risk for trading decisions.

Timezone handling:
  - FF API returns dates in US Eastern (-04:00/-05:00)
  - All internal comparisons use UTC
  - Display converts to local time
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger('live_news')

# ═══════════════════════════════════════════════════════════════════
# NASDAQ-SPECIFIC IMPACT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

# DEV impact → score multiplier 3.0 (trade'i durdur)
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
    'president', 'trump speaks',
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


@dataclass
class LiveEvent:
    """A live economic calendar event."""
    timestamp: datetime       # UTC
    timestamp_local: datetime  # Local time
    name: str
    country: str
    impact: str       # DEV, HIGH, MEDIUM
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


class LiveNewsManager:
    """
    Economic calendar using ForexFactory JSON API.
    All times stored internally as UTC, displayed as local.
    """

    def __init__(self):
        self.events: List[LiveEvent] = []
        self.last_refresh: Optional[datetime] = None
        self.refresh_interval = timedelta(minutes=30)  # Refresh every 30 min

    def refresh_events(self, force: bool = False) -> int:
        """Fetch upcoming events from ForexFactory."""
        now = datetime.now(timezone.utc)
        if not force and self.last_refresh:
            if now - self.last_refresh < self.refresh_interval:
                return len(self.events)

        try:
            self.events = self._fetch_from_web()
            self.last_refresh = now

            logger.info("📅 Calendar refreshed: %d NASDAQ-relevant events", len(self.events))

            # Log upcoming high-impact events
            now_local = datetime.now()
            for ev in self.events:
                hours_until = (ev.timestamp_local - now_local).total_seconds() / 3600
                if 0 < hours_until < 24:
                    emoji = "🔴" if ev.impact == 'DEV' else "🟡" if ev.impact == 'HIGH' else "🟢"
                    logger.info("  %s %s [%s] in %.1fh (%s)",
                                 emoji, ev.name, ev.impact,
                                 hours_until, ev.timestamp_local.strftime('%H:%M'))

            return len(self.events)

        except Exception as e:
            logger.warning("Calendar refresh failed: %s", e)
            return 0

    def _fetch_from_web(self) -> List[LiveEvent]:
        """Fetch from ForexFactory calendar (faireconomy.media JSON API)."""
        events = []
        try:
            import urllib.request
            import ssl
            import json

            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

            try:
                ctx = ssl._create_unverified_context()
            except Exception:
                ctx = None

            req = urllib.request.Request(url, headers={'User-Agent': 'TradeBot/3.0'})
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = json.loads(resp.read().decode())

            now_utc = datetime.now(timezone.utc)
            now_local = datetime.now()
            # Offset between UTC and local
            utc_offset = now_local - now_utc.replace(tzinfo=None)

            for item in data:
                country = item.get('country', '')
                if country != 'USD':
                    continue

                title = item.get('title', '')
                date_str = item.get('date', '')

                # Parse timezone-aware date from FF API
                event_utc = self._parse_ff_date(date_str)
                if event_utc is None:
                    continue

                # Convert to local time
                event_local = event_utc + utc_offset

                # Skip old events (>2h ago)
                if event_local < now_local - timedelta(hours=2):
                    continue

                # Skip events more than 7 days out
                if event_local > now_local + timedelta(days=7):
                    continue

                # NASDAQ-specific classification (not FF generic)
                impact = self._classify_impact(title)
                if impact == 'LOW':
                    continue

                events.append(LiveEvent(
                    timestamp=event_utc,
                    timestamp_local=event_local,
                    name=title,
                    country='US',
                    impact=impact,
                    actual=item.get('actual', ''),
                    forecast=item.get('forecast', ''),
                    previous=item.get('previous', ''),
                ))

            events.sort(key=lambda e: e.timestamp)
            if events:
                logger.info("📅 Fetched %d NASDAQ events from ForexFactory", len(events))

        except Exception as e:
            logger.warning("Web calendar fetch failed: %s", e)

        return events

    def _parse_ff_date(self, date_str: str) -> Optional[datetime]:
        """Parse FF date string to UTC datetime."""
        if not date_str:
            return None

        # FF format: "2026-03-09T17:30:00-04:00"
        try:
            dt = datetime.fromisoformat(date_str)
            # Convert to UTC
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except (ValueError, TypeError):
            pass

        # Fallback formats
        for fmt in ['%Y-%m-%dT%H:%M:%S', '%b %d, %Y %I:%M%p']:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None

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
        NASDAQ-specific risk score using LOCAL time.

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
            hours_until = (event.timestamp_local - current_time).total_seconds() / 3600

            # Window: 1h ago to 4h ahead
            if hours_until < -1 or hours_until > 4:
                continue

            upcoming.append(event)

            # Time proximity multiplier
            if hours_until < 0.5:
                time_mult = 3.0    # Çok yakın (30dk içinde veya geçmiş)
            elif hours_until < 1.0:
                time_mult = 2.5
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
        hours = (next_event.timestamp_local - datetime.now()).total_seconds() / 3600
        risk = "🔴 WAIT" if score >= 6.0 else "🟡 REDUCED" if score >= 3.0 else "🟢 CLEAR"
        time_str = next_event.timestamp_local.strftime('%H:%M')

        return f"📅 {risk} (score={score:.1f}) | {next_event.name} [{next_event.impact}] {time_str} ({hours:+.1f}h)"

    def get_next_events(self, count: int = 5) -> List[LiveEvent]:
        """Get next N events for display."""
        self.refresh_events()
        now = datetime.now()
        future = [e for e in self.events
                  if e.timestamp_local > now - timedelta(minutes=30)]
        return future[:count]


# Singleton
_live_news: Optional[LiveNewsManager] = None


def get_live_news() -> LiveNewsManager:
    global _live_news
    if _live_news is None:
        _live_news = LiveNewsManager()
    return _live_news
