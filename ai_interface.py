# ai_interface.py
# System v0.9: AI Integration Interface (Placeholder)
# Ready for future ML/n8n connection

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import config

logger = logging.getLogger(__name__)


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
    AI Integration Layer — STUB implementation.
    All methods return neutral signals until real model is connected.
    """
    
    def __init__(self):
        self.enabled = config.ENABLE_AI_WRAPPER
        self._call_count = 0
        if self.enabled:
            logger.info("AI Wrapper: ENABLED (stub mode)")
    
    def check_entry_decision(self, proposed_direction: str,
                             market_data: Dict[str, float],
                             timestamp: datetime) -> AISignal:
        if not self.enabled:
            return self._neutral("AI disabled")
        self._call_count += 1
        return AISignal(SignalType.NO_ACTION, 0.0, "Stub: Approving baseline entry")
    
    def check_scale_in_decision(self, current_pnl_atr: float,
                                position_direction: str,
                                market_data: Dict[str, float],
                                timestamp: datetime) -> AISignal:
        if not self.enabled or not config.PYRAMIDING_ENABLED:
            return self._neutral("Scaling disabled")
        self._call_count += 1
        return AISignal(SignalType.NO_ACTION, 0.0, "Stub: No scale-in")
    
    def check_hedge_decision(self, candle_range: float, atr: float,
                             position_direction: str,
                             timestamp: datetime) -> AISignal:
        if not self.enabled or not config.HEDGING_ENABLED:
            return self._neutral("Hedging disabled")
        self._call_count += 1
        return AISignal(SignalType.NO_ACTION, 0.0, "Stub: No hedge")
    
    def _neutral(self, reason: str) -> AISignal:
        return AISignal(SignalType.NO_ACTION, 0.0, reason)
    
    @property
    def call_count(self) -> int:
        return self._call_count


_intelligence_layer: Optional[IntelligenceLayer] = None


def get_intelligence_layer() -> IntelligenceLayer:
    global _intelligence_layer
    if _intelligence_layer is None:
        _intelligence_layer = IntelligenceLayer()
    return _intelligence_layer


def reset_intelligence_layer() -> None:
    global _intelligence_layer
    _intelligence_layer = None
