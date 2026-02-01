# config.py
# System v0.8: Data-Driven News Risk System
# NASDAQ H1 Trend-Following with News Risk Scoring
#
# Architecture:
# - Baseline strategy runs by default
# - News Risk Score controls position sizing and entry blocking
# - Advanced features (pyramiding, hedging) require AI wrapper

# =============================================================================
# === SECTION 1: BASELINE STRATEGY (Active) ===
# =============================================================================
# Core v0.6 parameters that run by default

# --- Indicators ---
EMA_PERIOD = 50              # Trend direction
ATR_PERIOD = 14              # Volatility measurement
RSI_PERIOD = 14              # Momentum filter
ATR_AVG_PERIOD = 50          # For volatility normalization

# --- Entry Filters ---
RSI_LONG_MIN = 40            # Min RSI for long entry
RSI_LONG_MAX = 70            # Max RSI for long entry
RSI_SHORT_MIN = 30           # Min RSI for short entry
RSI_SHORT_MAX = 60           # Max RSI for short entry
ATR_FILTER_RATIO = 0.5       # Min volatility (ATR > 0.5 × ATR_AVG)

# --- Exit Rules ---
STOP_LOSS_ATR_MULT = 2.0     # SL = Entry ± 2×ATR
TAKE_PROFIT_ATR_MULT = 3.0   # TP = Entry ± 3×ATR
MAX_BARS_IN_TRADE = 10       # Time exit after 10 bars

# --- Risk Management ---
RISK_PER_TRADE = 0.005       # 0.5% risk per trade
MAX_DAILY_LOSS = 0.01        # 1% daily loss limit
INITIAL_CAPITAL = 100000     # Starting capital

# =============================================================================
# === SECTION 2: NEWS RISK SCORING (Data-Driven) ===
# =============================================================================
# News Risk Score replaces simple boolean filter with density-based scoring

# --- Master Switch ---
ENABLE_NEWS_FILTER = True    # Enable news risk scoring

# --- Data Source ---
NEWS_CSV_PATH = "economic_events.csv"

# --- Risk Thresholds ---
NEWS_RISK_THRESHOLD_HIGH = 7.0   # Block trades completely above this score
NEWS_RISK_THRESHOLD_MED = 4.0    # Reduce position size 50% above this score

# --- Scoring Parameters ---
NEWS_LOOKFORWARD_HOURS = 4       # Scan window for upcoming events

# Time Decay Multipliers (closer events = higher risk)
NEWS_TIME_DECAY_IMMEDIATE = 3.0  # Events < 1 hour away
NEWS_TIME_DECAY_NEAR = 2.0       # Events 1-2 hours away
NEWS_TIME_DECAY_FAR = 1.0        # Events 2-4 hours away

# Impact Multipliers
NEWS_IMPACT_HIGH = 2.0           # HIGH impact events
NEWS_IMPACT_MEDIUM = 1.0         # MEDIUM impact events

# =============================================================================
# === SECTION 3: AI CONTROLS (Dormant) ===
# =============================================================================
# AI wrapper for future ML/n8n integration

ENABLE_AI_WRAPPER = False    # Enable AI signal processing
AI_SIGNAL_TIMEOUT_SEC = 5    # Max wait for AI response

# =============================================================================
# === SECTION 4: ADVANCED/DORMANT (Future Use - AI Controlled) ===
# =============================================================================
# These features are PRESERVED but LOCKED. Only AI can trigger them.

# --- Feature Enable Flags ---
PYRAMIDING_ENABLED = False   # Requires ENABLE_AI_WRAPPER = True
HEDGING_ENABLED = False      # Requires ENABLE_AI_WRAPPER = True

# --- Pyramiding (Scaling In) ---
MAX_LAYERS = 3               # Max position layers
SCALE_IN_PROFIT_ATR = 1.5    # Scale after 1.5 ATR profit
SCALE_IN_SIZE_RATIO = 0.5    # Each layer = 50% of initial
MOVE_SL_TO_BE_ON_SCALE = True  # Move SL to breakeven on scale

# --- Hedging (Defense Mode) ---
HEDGE_RATIO = 0.5            # Hedge 50% of position
HEDGE_MAX_BARS = 3           # Max hedge duration

# --- ADX Trend Strength (Dormant) ---
ADX_PERIOD = 14
ADX_MIN_THRESHOLD = 0        # 0 = disabled

# =============================================================================
# === SECTION 5: SYSTEM PATHS ===
# =============================================================================
DATA_FILE_PATH = "USATECHIDXUSD60.csv"
TRADE_LOG_PATH = "trade_log.csv"
