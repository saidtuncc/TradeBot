# ai_interface.py
# System v0.8: AI Integration Interface (Placeholder)
#
# This module provides a clean interface for future AI/ML integration.
# Currently implements a STUB that returns neutral signals.
# Ready for n8n/ML connection when enabled.

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import config


class SignalType(Enum):
    """Types of AI signals."""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"
    SCALE_IN = "scale_in"
    ACTIVATE_HEDGE = "hedge"
    NO_ACTION = "no_action"
    BLOCK_ENTRY = "block_entry"


@dataclass
class AISignal:
    """AI recommendation signal."""
    signal_type: SignalType
    confidence: float
    reason: str
    metadata: Optional[Dict[str, Any]] = None


class IntelligenceLayer:
    """
    AI Integration Layer for trading decisions.
    
    This is a STUB implementation. All methods return neutral signals.
    Replace with actual AI/ML model calls in production.
    
    The baseline strategy ALWAYS runs. AI can only:
    1. Block entries (not force entries)
    2. Suggest scaling in (requires PYRAMIDING_ENABLED)
    3. Suggest hedging (requires HEDGING_ENABLED)
    """
    
    def __init__(self):
        self.enabled = config.ENABLE_AI_WRAPPER
        self._call_count = 0
        
        if self.enabled:
            print("🤖 AI Wrapper: ENABLED (stub mode)")
    
    def check_entry_decision(
        self,
        proposed_direction: str,
        market_data: Dict[str, float],
        timestamp: datetime
    ) -> AISignal:
        """Check if AI approves or blocks a proposed entry."""
        if not self.enabled:
            return self._neutral_signal("AI disabled")
        
        self._call_count += 1
        # STUB: Always approve
        return AISignal(
            signal_type=SignalType.NO_ACTION,
            confidence=0.0,
            reason="Stub: Approving baseline entry"
        )
    
    def check_scale_in_decision(
        self,
        current_pnl_atr: float,
        position_direction: str,
        market_data: Dict[str, float],
        timestamp: datetime
    ) -> AISignal:
        """Check if AI recommends scaling into position."""
        if not self.enabled or not config.PYRAMIDING_ENABLED:
            return self._neutral_signal("Scaling disabled")
        
        self._call_count += 1
        return AISignal(
            signal_type=SignalType.NO_ACTION,
            confidence=0.0,
            reason="Stub: Not recommending scale-in"
        )
    
    def check_hedge_decision(
        self,
        candle_range: float,
        atr: float,
        position_direction: str,
        timestamp: datetime
    ) -> AISignal:
        """Check if AI recommends activating hedge."""
        if not self.enabled or not config.HEDGING_ENABLED:
            return self._neutral_signal("Hedging disabled")
        
        self._call_count += 1
        return AISignal(
            signal_type=SignalType.NO_ACTION,
            confidence=0.0,
            reason="Stub: Not recommending hedge"
        )
    
    def _neutral_signal(self, reason: str) -> AISignal:
        """Create a neutral (no action) signal."""
        return AISignal(
            signal_type=SignalType.NO_ACTION,
            confidence=0.0,
            reason=reason
        )
    
    @property
    def call_count(self) -> int:
        """Number of AI inference calls made."""
        return self._call_count


# Global instance
_intelligence_layer: Optional[IntelligenceLayer] = None


def get_intelligence_layer() -> IntelligenceLayer:
    """Get or create global IntelligenceLayer instance."""
    global _intelligence_layer
    if _intelligence_layer is None:
        _intelligence_layer = IntelligenceLayer()
    return _intelligence_layer


def reset_intelligence_layer() -> None:
    """Reset global instance (for testing)."""
    global _intelligence_layer
    _intelligence_layer = None
