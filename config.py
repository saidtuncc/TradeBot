# config.py
# System v0.7: Dynamic Position Management Configuration
# NASDAQ H1 Trend-Following with Pyramiding
# 
# PERFORMANCE NOTES:
# - Pyramiding adds complexity but marginal improvement
# - ADX filter tested, found to hurt performance (disabled)
# - Volatility anomaly filter more aggressive than news calendar
# - Consider reverting to v0.6 news calendar if performance drops

# =============================================================================
# INDICATOR PARAMETERS
# =============================================================================
EMA_PERIOD = 50          # Trend indicator
ATR_PERIOD = 14          # Volatility indicator
RSI_PERIOD = 14          # Momentum indicator
ATR_AVG_PERIOD = 50      # For volatility filter baseline
ADX_PERIOD = 14          # Trend strength indicator

# =============================================================================
# ENTRY FILTERS
# =============================================================================
RSI_LONG_MIN = 40        
RSI_LONG_MAX = 70        
RSI_SHORT_MIN = 30       
RSI_SHORT_MAX = 60       
ATR_FILTER_RATIO = 0.5   # Min volatility filter

# ADX Chop Filter - SET TO 0 TO DISABLE
# Testing showed ADX filter hurt overall performance
ADX_MIN_THRESHOLD = 0    # Set >0 to enable (e.g., 20)
ADX_STRONG_TREND = 30    # ADX level for scaling authorization

# =============================================================================
# VOLATILITY ANOMALY FILTER
# =============================================================================
# Replaces news calendar with price-derived filter
# NOTE: More aggressive than news calendar, may block good entries
VOLATILITY_FILTER_ENABLED = False  # Disabled for baseline comparison
CANDLE_ANOMALY_MULT = 2.5    # Block if candle range > ATR × this
ATR_SPIKE_MULT = 2.0         # Block if ATR > ATR_AVG × this

# =============================================================================
# POSITION MANAGEMENT - PYRAMIDING
# =============================================================================
PYRAMIDING_ENABLED = True    # Enable scaling into winning trades
MAX_LAYERS = 2               # Max layers (initial + 1)
SCALE_IN_PROFIT_ATR = 2.0    # Scale after 2.0 ATR profit
SCALE_IN_SIZE_RATIO = 0.5    # Scale-in = 50% of initial
MOVE_SL_TO_BE_ON_SCALE = False  # Don't move to breakeven (reduces whipsaws)

# =============================================================================
# POSITION MANAGEMENT - HEDGING
# =============================================================================
HEDGING_ENABLED = False      # Testing showed hedging hurt PnL
HEDGE_TRIGGER_MULT = 3.0     
HEDGE_RATIO = 0.5            
HEDGE_EXIT_MULT = 1.5        
HEDGE_MAX_BARS = 3           

# =============================================================================
# EXIT PARAMETERS
# =============================================================================
STOP_LOSS_ATR_MULT = 2.0     
TAKE_PROFIT_ATR_MULT = 3.0   
MAX_BARS_IN_TRADE = 8        # Extended for intraday

# High volatility modifications
HIGH_VOL_MAX_BARS = 5        
HIGH_VOL_SL_MULT = 1.5       

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
RISK_PER_TRADE = 0.005       # 0.5% per trade
MAX_DAILY_LOSS = 0.01        # 1% daily loss limit
INITIAL_CAPITAL = 100000     
MAX_POSITION_RISK = 0.0075   # 0.75% max exposure (1.5 layers)
