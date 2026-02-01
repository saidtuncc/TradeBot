# news_manager.py
# System v0.8: Data-Driven News Risk Manager
# Calculates risk score based on upcoming economic events

import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass
import config


@dataclass
class EconomicEvent:
    """Represents a single economic news event."""
    timestamp: datetime
    event_name: str
    impact: str  # HIGH, MEDIUM


class NewsManager:
    """
    News Risk Score Manager using density-based scoring.
    
    Scoring Algorithm:
    - Scans events within NEWS_LOOKFORWARD_HOURS window
    - Applies time decay (closer events = higher risk)
    - Applies impact weight (HIGH = 2x, MEDIUM = 1x)
    - Returns cumulative score (0.0 to 10.0+)
    
    Behavior:
    - If ENABLE_NEWS_FILTER = False: returns 0.0 (no blocking)
    - If CSV missing: logs warning, returns 0.0 (graceful degradation)
    """
    
    def __init__(self):
        self.events: List[EconomicEvent] = []
        self.enabled = config.ENABLE_NEWS_FILTER
        self._file_loaded = False
        
        if self.enabled:
            self._load_events()
    
    def _load_events(self) -> None:
        """Load events from CSV file with graceful error handling."""
        csv_path = config.NEWS_CSV_PATH
        
        # Check file exists
        if not os.path.exists(csv_path):
            print(f"⚠️ NewsManager: CSV not found ({csv_path}), scoring disabled")
            self.enabled = False
            return
        
        # Check file is not empty
        file_size = os.path.getsize(csv_path)
        if file_size < 20:
            print(f"⚠️ NewsManager: CSV appears empty ({file_size} bytes), scoring disabled")
            self.enabled = False
            return
        
        # Load events
        try:
            self._parse_csv(csv_path)
            self._file_loaded = True
            print(f"✅ NewsManager: Loaded {len(self.events)} events from {csv_path}")
        except Exception as e:
            print(f"⚠️ NewsManager: Failed to parse CSV ({e}), scoring disabled")
            self.enabled = False
    
    def _parse_csv(self, filepath: str) -> None:
        """Parse CSV file into EconomicEvent objects."""
        import csv
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    # Parse datetime
                    timestamp = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M:%S")
                    
                    # Get impact (default to MEDIUM if not HIGH)
                    impact = row.get('impact', 'MEDIUM').upper()
                    if impact not in ('HIGH', 'MEDIUM'):
                        impact = 'MEDIUM'
                    
                    event = EconomicEvent(
                        timestamp=timestamp,
                        event_name=row.get('event', 'Unknown'),
                        impact=impact
                    )
                    self.events.append(event)
                except (KeyError, ValueError):
                    continue  # Skip malformed rows
        
        # Sort by timestamp for efficient scanning
        self.events.sort(key=lambda e: e.timestamp)
    
    def calculate_risk_score(self, current_time: datetime) -> float:
        """
        Calculate news risk score for the current time.
        
        Scoring:
        - Base score = 1.0 per event
        - Time decay: <1hr = 3x, 1-2hr = 2x, 2-4hr = 1x
        - Impact: HIGH = 2x, MEDIUM = 1x
        
        Returns:
            Float score (0.0 = safe, 4.0+ = reduce size, 7.0+ = block)
        """
        if not self.enabled:
            return 0.0
        
        window_end = current_time + timedelta(hours=config.NEWS_LOOKFORWARD_HOURS)
        
        total_score = 0.0
        
        for event in self.events:
            # Only consider future events within window
            if event.timestamp <= current_time:
                continue
            if event.timestamp > window_end:
                break  # Events are sorted, no need to check further
            
            # Calculate time until event
            time_until = (event.timestamp - current_time).total_seconds() / 3600  # hours
            
            # Time decay multiplier
            if time_until < 1.0:
                time_mult = config.NEWS_TIME_DECAY_IMMEDIATE  # 3.0
            elif time_until < 2.0:
                time_mult = config.NEWS_TIME_DECAY_NEAR      # 2.0
            else:
                time_mult = config.NEWS_TIME_DECAY_FAR       # 1.0
            
            # Impact multiplier
            if event.impact == 'HIGH':
                impact_mult = config.NEWS_IMPACT_HIGH        # 2.0
            else:
                impact_mult = config.NEWS_IMPACT_MEDIUM      # 1.0
            
            # Add to total score
            event_score = 1.0 * time_mult * impact_mult
            total_score += event_score
        
        return total_score
    
    def get_upcoming_events(self, current_time: datetime, hours: int = 4) -> List[EconomicEvent]:
        """Get list of upcoming events within specified hours."""
        if not self.enabled:
            return []
        
        window_end = current_time + timedelta(hours=hours)
        upcoming = []
        
        for event in self.events:
            if event.timestamp <= current_time:
                continue
            if event.timestamp > window_end:
                break
            upcoming.append(event)
        
        return upcoming
    
    def get_risk_level(self, score: float) -> str:
        """Convert score to human-readable risk level."""
        if score >= config.NEWS_RISK_THRESHOLD_HIGH:
            return "HIGH (BLOCKED)"
        elif score >= config.NEWS_RISK_THRESHOLD_MED:
            return "MEDIUM (REDUCED)"
        else:
            return "LOW (NORMAL)"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instance (lazy initialization)
_news_manager: Optional[NewsManager] = None


def get_news_manager() -> NewsManager:
    """Get or create global NewsManager instance."""
    global _news_manager
    if _news_manager is None:
        _news_manager = NewsManager()
    return _news_manager


def calculate_risk_score(current_time: datetime) -> float:
    """Quick access to risk score calculation."""
    return get_news_manager().calculate_risk_score(current_time)


def reset_news_manager() -> None:
    """Reset global instance (for testing)."""
    global _news_manager
    _news_manager = None
