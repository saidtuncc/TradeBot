# config.py
# System v0.9.1: Strategy Enhanced
# NASDAQ H1 Trend-Following with ADX Filter + ATR Trailing Stop
#
# Architecture:
# - Baseline strategy runs by default
# - News Risk Score controls position sizing and entry blocking
# - Advanced features (pyramiding, hedging) require AI wrapper
# - Commission + slippage simulation for realistic backtesting

import logging

# =============================================================================
# === SECTION 1: BASELINE STRATEGY (Active) ===
# =============================================================================

# --- Indicators ---
EMA_PERIOD = 50
ATR_PERIOD = 14
RSI_PERIOD = 14
ATR_AVG_PERIOD = 50
ADX_PERIOD = 14

# --- Entry Filters ---
RSI_LONG_MIN = 40
RSI_LONG_MAX = 70
RSI_SHORT_MIN = 30
RSI_SHORT_MAX = 60
ATR_FILTER_RATIO = 0.5
ADX_MIN_THRESHOLD = 25           # Only trade when ADX > 25 (strong trend)
EMA_SLOPE_LOOKBACK = 5           # Bars to measure EMA slope
EMA_SLOPE_MIN = 0.0005           # Min EMA slope (% change) to confirm trend

# --- Exit Rules ---
STOP_LOSS_ATR_MULT = 2.0         # ML-driven: tighter SL for controlled risk
TAKE_PROFIT_ATR_MULT = 3.0       # Asymmetric 3:2 R:R → each win > each loss
MAX_BARS_IN_TRADE = 48           # Max 48 H1 bars = 2 days
TRAILING_STOP_ENABLED = True
BREAKEVEN_TRIGGER_ATR = 1.5      # Move SL to BE after 1.5×ATR profit
TRAILING_STOP_ATR_MULT = 2.5     # Trail SL at 2.5×ATR behind price

# --- Multi-Position ---
MAX_CONCURRENT_POSITIONS = 3     # Max open positions at once
PYRAMIDING_ENABLED = True        # Add to winning positions
PYRAMID_MIN_PROFIT_ATR = 1.0     # Add layer when pos is +1 ATR in profit
PYRAMID_MAX_LAYERS = 2           # Max additional layers per direction
PYRAMID_SIZE_DECAY = 0.5         # Each layer = 50% of previous size

# --- Risk Management ---
RISK_PER_TRADE = 0.005
MAX_DAILY_LOSS = 0.01
INITIAL_CAPITAL = 100000

# --- Drawdown Control ---
DD_REDUCE_LEVEL = 0.05            # DD > 5% → halve position size
DD_STOP_LEVEL = 0.10              # DD > 10% → stop trading
DD_RECOVERY_BUFFER = 0.02         # Resume at DD < 8% (stop-2%)

# =============================================================================
# === SECTION 2: TRADING COSTS (Realistic Simulation) ===
# =============================================================================

SPREAD_POINTS = 1.5          # Typical NASDAQ CFD spread
SLIPPAGE_POINTS = 0.5        # Average execution slippage
COMMISSION_PER_TRADE = 0.0   # Most CFD brokers: zero commission
# Total cost per trade entry: SPREAD + SLIPPAGE = 2.0 points

# =============================================================================
# === SECTION 3: NEWS RISK SCORING (Data-Driven) ===
# =============================================================================

ENABLE_NEWS_FILTER = True
NEWS_CSV_PATH = "data/economic_events.csv"

NEWS_RISK_THRESHOLD_HIGH = 7.0
NEWS_RISK_THRESHOLD_MED = 4.0
NEWS_LOOKFORWARD_HOURS = 4

NEWS_TIME_DECAY_IMMEDIATE = 3.0
NEWS_TIME_DECAY_NEAR = 2.0
NEWS_TIME_DECAY_FAR = 1.0

NEWS_IMPACT_HIGH = 2.0
NEWS_IMPACT_MEDIUM = 1.0

# =============================================================================
# === SECTION 4: ML INTEGRATION (Active) ===
# =============================================================================

ENABLE_ML = True                    # Master switch for ML signals
ML_LONG_THRESHOLD = 0.29            # Optimized on 2024-2025 data
ML_SHORT_THRESHOLD = 0.33           # Optimized on 2024-2025 data
ML_CONFIDENCE_SIZING = True         # Scale position size by ML confidence
ML_KELLY_FRACTION = 0.25            # Quarter-Kelly for conservative sizing
ML_MIN_CONFIDENCE = 0.15            # Below this → no trade regardless

# Legacy
ENABLE_AI_WRAPPER = True
AI_SIGNAL_TIMEOUT_SEC = 5

# =============================================================================
# === SECTION 5: ADVANCED/DORMANT (Future Use - AI Controlled) ===
# =============================================================================

PYRAMIDING_ENABLED = False
HEDGING_ENABLED = False

MAX_LAYERS = 3
SCALE_IN_PROFIT_ATR = 1.5
SCALE_IN_SIZE_RATIO = 0.5
MOVE_SL_TO_BE_ON_SCALE = True

HEDGE_RATIO = 0.5
HEDGE_MAX_BARS = 3

# =============================================================================
# === SECTION 6: WALK-FORWARD VALIDATION ===
# =============================================================================

WALK_FORWARD_ENABLED = True
WF_TEST_YEARS = 2              # Last N years as out-of-sample
WF_MIN_TRAIN_YEARS = 5         # Minimum training period

# =============================================================================
# === SECTION 7: SYSTEM PATHS & LOGGING ===
# =============================================================================

DATA_FILE_PATH = "data/USATECHIDXUSD60.csv"
TRADE_LOG_PATH = "trade_log.csv"

LOG_LEVEL = logging.INFO
LOG_FILE = "tradebot.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# === SECTION 8: TELEGRAM NOTIFICATIONS ===
# 1. Telegram'da @BotFather'a /newbot yaz → token al
# 2. @userinfobot'a yaz → chat_id al
TELEGRAM_TOKEN = ""       # BotFather'dan aldığın token
TELEGRAM_CHAT_ID = ""     # Senin chat ID'n
