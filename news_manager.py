# news_manager.py
# System v0.9: Optimized News Risk Manager
# Uses bisect for O(log n) event lookup instead of O(n) linear scan

import os
import logging
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass
import config

logger = logging.getLogger(__name__)


@dataclass
class EconomicEvent:
    """Represents a single economic news event."""
    timestamp: datetime
    event_name: str
    impact: str  # HIGH, MEDIUM


class NewsManager:
    """
    News Risk Score Manager using density-based scoring.
    
    v0.9 Optimization:
    - Events stored sorted with timestamps extracted for bisect lookup
    - calculate_risk_score() is now O(log n + k) where k = events in window
    - Previously O(n) per call → ~1000x faster for 19K events
    """
    
    def __init__(self):
        self.events: List[EconomicEvent] = []
        self._timestamps: List[datetime] = []  # Parallel list for bisect
        self.enabled = config.ENABLE_NEWS_FILTER
        self._file_loaded = False
        
        if self.enabled:
            self._load_events()
    
    def _load_events(self) -> None:
        """Load events from CSV file with graceful error handling."""
        csv_path = config.NEWS_CSV_PATH
        
        if not os.path.exists(csv_path):
            logger.warning("NewsManager: CSV not found (%s), scoring disabled", csv_path)
            self.enabled = False
            return
        
        file_size = os.path.getsize(csv_path)
        if file_size < 20:
            logger.warning("NewsManager: CSV appears empty (%d bytes), scoring disabled", file_size)
            self.enabled = False
            return
        
        try:
            self._parse_csv(csv_path)
            self._file_loaded = True
            logger.info("NewsManager: Loaded %d events from %s", len(self.events), csv_path)
        except Exception as e:
            logger.warning("NewsManager: Failed to parse CSV (%s), scoring disabled", e)
            self.enabled = False
    
    def _parse_csv(self, filepath: str) -> None:
        """Parse CSV file into EconomicEvent objects."""
        import csv
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    timestamp = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M:%S")
                    impact = row.get('impact', 'MEDIUM').upper()
                    if impact not in ('HIGH', 'MEDIUM'):
                        impact = 'MEDIUM'
                    
                    self.events.append(EconomicEvent(
                        timestamp=timestamp,
                        event_name=row.get('event', 'Unknown'),
                        impact=impact
                    ))
                except (KeyError, ValueError):
                    continue
        
        # Sort and build parallel timestamp list for bisect
        self.events.sort(key=lambda e: e.timestamp)
        self._timestamps = [e.timestamp for e in self.events]
    
    def calculate_risk_score(self, current_time: datetime) -> float:
        """
        Calculate news risk score using binary search.
        
        O(log n + k) where k = events in window.
        """
        if not self.enabled:
            return 0.0
        
        window_end = current_time + timedelta(hours=config.NEWS_LOOKFORWARD_HOURS)
        
        # Binary search for window boundaries
        start_idx = bisect_right(self._timestamps, current_time)
        end_idx = bisect_right(self._timestamps, window_end)
        
        if start_idx >= end_idx:
            return 0.0
        
        total_score = 0.0
        
        for i in range(start_idx, end_idx):
            event = self.events[i]
            time_until = (event.timestamp - current_time).total_seconds() / 3600
            
            # Time decay multiplier
            if time_until < 1.0:
                time_mult = config.NEWS_TIME_DECAY_IMMEDIATE
            elif time_until < 2.0:
                time_mult = config.NEWS_TIME_DECAY_NEAR
            else:
                time_mult = config.NEWS_TIME_DECAY_FAR
            
            # Impact multiplier
            impact_mult = config.NEWS_IMPACT_HIGH if event.impact == 'HIGH' else config.NEWS_IMPACT_MEDIUM
            
            total_score += 1.0 * time_mult * impact_mult
        
        return total_score
    
    def get_upcoming_events(self, current_time: datetime, hours: int = 4) -> List[EconomicEvent]:
        """Get upcoming events using binary search."""
        if not self.enabled:
            return []
        
        window_end = current_time + timedelta(hours=hours)
        start_idx = bisect_right(self._timestamps, current_time)
        end_idx = bisect_right(self._timestamps, window_end)
        
        return self.events[start_idx:end_idx]
    
    def get_risk_level(self, score: float) -> str:
        """Convert score to human-readable risk level."""
        if score >= config.NEWS_RISK_THRESHOLD_HIGH:
            return "HIGH (BLOCKED)"
        elif score >= config.NEWS_RISK_THRESHOLD_MED:
            return "MEDIUM (REDUCED)"
        return "LOW (NORMAL)"


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_news_manager: Optional[NewsManager] = None


def get_news_manager() -> NewsManager:
    """Get or create global NewsManager instance."""
    global _news_manager
    if _news_manager is None:
        _news_manager = NewsManager()
    return _news_manager


def reset_news_manager() -> None:
    """Reset global instance (for testing)."""
    global _news_manager
    _news_manager = None
