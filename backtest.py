# backtest.py
# System v0.9: Foundation Hardened
# NASDAQ H1 Trend-Following with News Risk Scoring
#
# v0.9 Changes:
# - Commission + Slippage simulation
# - RSI: vectorized ewm (was slow .iloc loop)
# - Sharpe Ratio: corrected annualization
# - Daily loss limit: based on current capital (was initial)
# - Bar-based equity curve for accurate drawdown
# - Walk-forward validation
# - Logging instead of print()

import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import config
from news_manager import get_news_manager, reset_news_manager
from ai_interface import get_intelligence_layer, reset_intelligence_layer

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging for the application."""
    root_logger = logging.getLogger()
    root_logger.setLevel(config.LOG_LEVEL)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(config.LOG_FORMAT))
    root_logger.addHandler(console)
    
    # File handler
    try:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        root_logger.addHandler(file_handler)
    except (PermissionError, OSError):
        pass  # Skip file logging if not writable

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

@dataclass
class Layer:
    """Single entry layer in a pyramided position."""
    entry_price: float
    size: float
    entry_time: datetime
    atr_at_entry: float
    layer_id: int

@dataclass 
class TradeRecord:
    """Complete record of a closed trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    avg_entry_price: float
    exit_price: float
    initial_size: float
    max_size: float
    final_size: float
    num_layers: int
    gross_pnl: float
    trading_costs: float
    net_pnl: float
    exit_reason: str
    bars_held: int
    hedge_activated: bool
    scaled_in: bool
    news_risk_score: float
    size_reduced: bool

# =============================================================================
# POSITION MANAGER
# =============================================================================

class PositionManager:
    """State machine for position management."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.direction: Direction = Direction.FLAT
        self.layers: List[Layer] = []
        self.current_sl: float = 0.0
        self.current_tp: float = 0.0
        self.entry_bar_idx: int = 0
        self.bars_held: int = 0
        
        self.hedge_active: bool = False
        self.hedge_size: float = 0.0
        self.hedge_bars: int = 0
        
        self.initial_size: float = 0.0
        self.max_size_reached: float = 0.0
        self.hedge_was_activated: bool = False
        self.scaled_in: bool = False
        self.initial_entry_price: float = 0.0
        
        self.entry_risk_score: float = 0.0
        self.size_was_reduced: bool = False
    
    @property
    def is_open(self) -> bool:
        return self.direction != Direction.FLAT and len(self.layers) > 0
    
    @property
    def total_size(self) -> float:
        return sum(layer.size for layer in self.layers)
    
    @property
    def net_exposure(self) -> float:
        gross = self.total_size
        if self.hedge_active:
            return max(0, gross - self.hedge_size)
        return gross
    
    @property
    def avg_entry_price(self) -> float:
        if not self.layers:
            return 0.0
        total_value = sum(l.entry_price * l.size for l in self.layers)
        total_size = self.total_size
        return total_value / total_size if total_size > 0 else 0.0
    
    @property
    def num_layers(self) -> int:
        return len(self.layers)
    
    def open_position(self, direction: Direction, price: float, size: float,
                     atr: float, time: datetime, bar_idx: int,
                     risk_score: float = 0.0, size_reduced: bool = False) -> None:
        self.reset()
        self.direction = direction
        self.entry_bar_idx = bar_idx
        self.initial_size = size
        self.initial_entry_price = price
        self.max_size_reached = size
        self.entry_risk_score = risk_score
        self.size_was_reduced = size_reduced
        
        self.layers.append(Layer(
            entry_price=price, size=size, entry_time=time,
            atr_at_entry=atr, layer_id=1
        ))
        
        if direction == Direction.LONG:
            self.current_sl = price - config.STOP_LOSS_ATR_MULT * atr
            self.current_tp = price + config.TAKE_PROFIT_ATR_MULT * atr
        else:
            self.current_sl = price + config.STOP_LOSS_ATR_MULT * atr
            self.current_tp = price - config.TAKE_PROFIT_ATR_MULT * atr
    
    def add_layer(self, price: float, size: float, atr: float, time: datetime) -> bool:
        """DORMANT: Requires PYRAMIDING + AI_WRAPPER enabled."""
        if not config.PYRAMIDING_ENABLED or not config.ENABLE_AI_WRAPPER:
            return False
        if not self.is_open or self.num_layers >= config.MAX_LAYERS:
            return False
        
        self.layers.append(Layer(
            entry_price=price, size=size, entry_time=time,
            atr_at_entry=atr, layer_id=self.num_layers + 1
        ))
        self.scaled_in = True
        self.max_size_reached = max(self.max_size_reached, self.total_size)
        
        if config.MOVE_SL_TO_BE_ON_SCALE:
            be = self.avg_entry_price
            if self.direction == Direction.LONG and be > self.current_sl:
                self.current_sl = be
            elif self.direction == Direction.SHORT and be < self.current_sl:
                self.current_sl = be
        return True
    
    def activate_hedge(self) -> bool:
        """DORMANT: Requires HEDGING + AI_WRAPPER enabled."""
        if not config.HEDGING_ENABLED or not config.ENABLE_AI_WRAPPER:
            return False
        if self.hedge_active:
            return False
        
        self.hedge_size = self.total_size * config.HEDGE_RATIO
        self.hedge_active = True
        self.hedge_bars = 0
        self.hedge_was_activated = True
        return True
    
    def deactivate_hedge(self) -> None:
        self.hedge_active = False
        self.hedge_size = 0.0
        self.hedge_bars = 0
    
    def close_position(self, exit_price: float, exit_time: datetime,
                       exit_reason: str) -> TradeRecord:
        if self.direction == Direction.LONG:
            price_change = exit_price - self.avg_entry_price
        else:
            price_change = self.avg_entry_price - exit_price
        
        gross_pnl = price_change * self.net_exposure
        
        # Trading costs: spread + slippage on entry AND exit
        cost_per_unit = (config.SPREAD_POINTS + config.SLIPPAGE_POINTS) * 2  # round-trip
        trading_costs = cost_per_unit * self.net_exposure + config.COMMISSION_PER_TRADE * 2
        net_pnl = gross_pnl - trading_costs
        
        record = TradeRecord(
            entry_time=self.layers[0].entry_time,
            exit_time=exit_time,
            direction=self.direction.name,
            entry_price=self.initial_entry_price,
            avg_entry_price=self.avg_entry_price,
            exit_price=exit_price,
            initial_size=self.initial_size,
            max_size=self.max_size_reached,
            final_size=self.net_exposure,
            num_layers=self.num_layers,
            gross_pnl=gross_pnl,
            trading_costs=trading_costs,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            bars_held=self.bars_held,
            hedge_activated=self.hedge_was_activated,
            scaled_in=self.scaled_in,
            news_risk_score=self.entry_risk_score,
            size_reduced=self.size_was_reduced
        )
        
        self.reset()
        return record

# =============================================================================
# INDICATORS (Fully Vectorized)
# =============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI using ewm — fully vectorized, no loops.
    v0.9: Replaced slow .iloc loop with ewm(alpha=1/period).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength (not direction). ADX > 25 = strong trend.
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx

# =============================================================================
# BACKTESTING ENGINE v0.9.1
# =============================================================================

class Backtester:
    """
    Event-driven backtester with ML integration.
    
    v1.0 ML-Integrated:
    - ML dual LONG/SHORT models drive entry direction
    - Kelly criterion position sizing from ML probability
    - Rule-based filters as safety guardrails
    - ATR trailing stop + asymmetric 3:2 R:R
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = None):
        self.data = data.copy()
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.capital = self.initial_capital
        self.trade_records: List[TradeRecord] = []
        self.position = PositionManager()
        self.daily_pnl = 0.0
        self.current_date: Optional[datetime] = None
        self.stopped_for_day = False
        
        # Bar-based equity tracking
        self.equity_curve: List[float] = []
        self.peak_equity = self.initial_capital
        self.dd_halted = False  # True when DD > stop level
        
        self.stats = {
            'total_bars': 0,
            'entry_signals': 0,
            'news_blocks': 0,
            'size_reductions': 0,
            'normal_entries': 0,
            'daily_stops': 0,
            'total_trading_costs': 0.0,
        }
        
        self.news_manager = get_news_manager()
        self.ai_layer = get_intelligence_layer()
        
        # ML predictor
        self.ml_predictor = None
        if getattr(config, 'ENABLE_ML', False):
            try:
                from ml.predictor import get_predictor
                self.ml_predictor = get_predictor()
                if not self.ml_predictor.loaded:
                    logger.warning("ML models not found, falling back to rule-based")
                    self.ml_predictor = None
            except Exception as e:
                logger.warning("ML init failed: %s, falling back to rule-based", e)
        
        self._prepare_data()
    
    def _prepare_data(self):
        df = self.data
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        # Basic indicators (always needed for rule-based)
        df['ema50'] = calculate_ema(df['close'], config.EMA_PERIOD)
        df['atr14'] = calculate_atr(df['high'], df['low'], df['close'], config.ATR_PERIOD)
        df['atr_avg'] = df['atr14'].rolling(window=config.ATR_AVG_PERIOD).mean()
        df['rsi14'] = calculate_rsi(df['close'], config.RSI_PERIOD)
        df['adx'] = calculate_adx(df['high'], df['low'], df['close'], config.ADX_PERIOD)
        df['ema_slope'] = df['ema50'].pct_change(config.EMA_SLOPE_LOOKBACK).abs()
        
        # ML features: pre-compute full feature set if ML enabled
        if self.ml_predictor and self.ml_predictor.loaded:
            try:
                from ml.feature_engine import build_full_feature_set
                logger.info("Pre-computing ML features for backtest...")
                ml_df = build_full_feature_set('.', add_news=True)
                
                # Align ML features with backtest data by datetime
                ml_df = ml_df.set_index('datetime')
                
                # Get ML feature columns from saved model
                for direction in ['long', 'short']:
                    selected = self.ml_predictor.models[direction].get('features', [])
                    if selected:
                        for col in selected:
                            if col in ml_df.columns and col not in df.columns:
                                df[col] = ml_df[col].reindex(df.index)
                
                ml_feature_count = len([c for c in df.columns if c not in
                    ['open', 'high', 'low', 'close', 'volume', 'ema50', 'atr14',
                     'atr_avg', 'rsi14', 'adx', 'ema_slope']])
                logger.info("ML features added: %d columns total in backtest data", ml_feature_count)
            except Exception as e:
                logger.warning("Failed to add ML features: %s", e)
        
        self.data = df.dropna()
    
    def _check_daily_reset(self, current_time: datetime):
        current_day = current_time.date()
        if self.current_date != current_day:
            self.current_date = current_day
            self.daily_pnl = 0.0
            self.stopped_for_day = False
    
    def _check_daily_loss_limit(self) -> bool:
        """v0.9 FIX: Based on current capital, not initial."""
        max_loss = self.capital * config.MAX_DAILY_LOSS
        if self.daily_pnl <= -max_loss:
            self.stopped_for_day = True
            self.stats['daily_stops'] += 1
            return True
        return False
    
    def _get_dd_risk_scale(self) -> float:
        """
        Drawdown-based risk scaling:
          DD < 5%  → 1.0 (full size)
          DD 5-10% → 0.5 (half size)
          DD > 10% → 0.0 (stop trading)
        """
        self.peak_equity = max(self.peak_equity, self.capital)
        dd = (self.peak_equity - self.capital) / self.peak_equity
        
        dd_stop = getattr(config, 'DD_STOP_LEVEL', 0.10)
        dd_reduce = getattr(config, 'DD_REDUCE_LEVEL', 0.05)
        dd_buffer = getattr(config, 'DD_RECOVERY_BUFFER', 0.02)
        
        if dd >= dd_stop:
            if not self.dd_halted:
                self.dd_halted = True
                self.stats['dd_halts'] = self.stats.get('dd_halts', 0) + 1
                logger.warning("DD HALT: %.1f%% drawdown — trading paused", dd * 100)
            return 0.0
        
        if self.dd_halted:
            if dd < (dd_stop - dd_buffer):
                self.dd_halted = False
                logger.info("DD RESUME: DD=%.1f%% < %.1f%% — trading resumed", dd * 100, (dd_stop - dd_buffer) * 100)
            else:
                return 0.0
        
        if dd >= dd_reduce:
            self.stats['dd_reductions'] = self.stats.get('dd_reductions', 0) + 1
            return 0.5
        
        return 1.0
    
    def _calculate_position_size(self, atr: float) -> float:
        risk_amount = self.capital * config.RISK_PER_TRADE
        stop_distance = config.STOP_LOSS_ATR_MULT * atr
        if stop_distance <= 0:
            return 0
        return risk_amount / stop_distance
    
    def _check_entry_conditions(self, row: pd.Series, time=None):
        """
        Returns (Direction, probability) tuple.
        ML mode: ML decides direction, rules act as guardrails.
        """
        close = row['close']
        ema50 = row['ema50']
        rsi = row['rsi14']
        atr = row['atr14']
        atr_avg = row['atr_avg']
        adx = row['adx']
        ema_slope = row['ema_slope']
        
        # === GUARDRAIL: Minimum volatility ===
        if atr < config.ATR_FILTER_RATIO * atr_avg:
            return None, 0.0
        
        # === ML MODE ===
        if self.ml_predictor and self.ml_predictor.loaded:
            features = self._build_ml_features(row)
            if features is not None:
                direction_str, prob = self.ml_predictor.predict_direction(features)
                if direction_str is None:
                    return None, 0.0
                direction = Direction.LONG if direction_str == 'long' else Direction.SHORT
                self.stats['ml_signals'] = self.stats.get('ml_signals', 0) + 1
                return direction, prob
            return None, 0.0
        
        # === RULE-BASED FALLBACK ===
        if config.ADX_MIN_THRESHOLD > 0 and adx < config.ADX_MIN_THRESHOLD:
            return None, 0.0
        if config.EMA_SLOPE_MIN > 0 and ema_slope < config.EMA_SLOPE_MIN:
            return None, 0.0
        if close > ema50 and config.RSI_LONG_MIN <= rsi <= config.RSI_LONG_MAX:
            return Direction.LONG, 0.55
        if close < ema50 and config.RSI_SHORT_MIN <= rsi <= config.RSI_SHORT_MAX:
            return Direction.SHORT, 0.55
        return None, 0.0
    
    def _build_ml_features(self, row: pd.Series):
        """Build feature DataFrame from bar for ML prediction."""
        try:
            feature_data = {}
            for col in row.index:
                if col not in ['open', 'high', 'low', 'close', 'volume', 'datetime']:
                    feature_data[col] = [row[col]]
            return pd.DataFrame(feature_data).replace([np.inf, -np.inf], np.nan).fillna(0)
        except:
            return None
    
    def _check_exit_conditions(self, row: pd.Series) -> Tuple[bool, str, float]:
        if not self.position.is_open:
            return False, "", 0.0
        
        pos = self.position
        high, low, close = row['high'], row['low'], row['close']
        atr = row['atr14']
        pos.bars_held += 1
        
        # Check SL/TP hit
        if pos.direction == Direction.LONG:
            if low <= pos.current_sl:
                return True, "stop_loss", pos.current_sl
            if high >= pos.current_tp:
                return True, "take_profit", pos.current_tp
        else:
            if high >= pos.current_sl:
                return True, "stop_loss", pos.current_sl
            if low <= pos.current_tp:
                return True, "take_profit", pos.current_tp
        
        # === ATR TRAILING STOP (v0.9.1) ===
        if config.TRAILING_STOP_ENABLED and atr > 0:
            entry_price = pos.initial_entry_price
            
            if pos.direction == Direction.LONG:
                unrealized_profit = close - entry_price
                
                # Move to breakeven after 1×ATR profit
                if unrealized_profit >= config.BREAKEVEN_TRIGGER_ATR * pos.layers[0].atr_at_entry:
                    be_level = entry_price
                    if be_level > pos.current_sl:
                        pos.current_sl = be_level
                
                # Trail SL behind close
                trail_sl = close - config.TRAILING_STOP_ATR_MULT * atr
                if trail_sl > pos.current_sl:
                    pos.current_sl = trail_sl
            
            else:  # SHORT
                unrealized_profit = entry_price - close
                
                if unrealized_profit >= config.BREAKEVEN_TRIGGER_ATR * pos.layers[0].atr_at_entry:
                    be_level = entry_price
                    if be_level < pos.current_sl:
                        pos.current_sl = be_level
                
                trail_sl = close + config.TRAILING_STOP_ATR_MULT * atr
                if trail_sl < pos.current_sl:
                    pos.current_sl = trail_sl
        
        # Time exit
        if pos.bars_held >= config.MAX_BARS_IN_TRADE:
            return True, "time_exit", close
        
        return False, "", 0.0
    
    def run(self) -> Dict:
        """Run the backtest."""
        logger.info("Starting backtest v0.9.1 (Strategy Enhanced)...")
        logger.info("Data: %s to %s (%d bars)", self.data.index[0], self.data.index[-1], len(self.data))
        logger.info("ADX Filter: >%d | Trailing Stop: %s | Costs: %.1f pts/trade",
                     config.ADX_MIN_THRESHOLD,
                     "ON" if config.TRAILING_STOP_ENABLED else "OFF",
                     config.SPREAD_POINTS + config.SLIPPAGE_POINTS)
        
        for bar_idx, (time, row) in enumerate(self.data.iterrows()):
            self.stats['total_bars'] += 1
            self._check_daily_reset(time)
            
            # Track equity every bar
            self.equity_curve.append(self.capital)
            
            if self.stopped_for_day:
                if self.position.is_open:
                    should_exit, reason, exit_price = self._check_exit_conditions(row)
                    if should_exit:
                        record = self.position.close_position(exit_price, time, reason)
                        self._process_closed_trade(record)
                continue
            
            # Manage existing position
            if self.position.is_open:
                should_exit, reason, exit_price = self._check_exit_conditions(row)
                if should_exit:
                    record = self.position.close_position(exit_price, time, reason)
                    self._process_closed_trade(record)
            
            # Check for new entry
            if not self.position.is_open and not self.stopped_for_day:
                direction, ml_prob = self._check_entry_conditions(row, time)
                
                if direction:
                    self.stats['entry_signals'] += 1
                    
                    # News risk scoring
                    risk_score = self.news_manager.calculate_risk_score(time)
                    
                    if risk_score >= config.NEWS_RISK_THRESHOLD_HIGH:
                        self.stats['news_blocks'] += 1
                        continue
                    
                    # Position sizing: Kelly-based if ML active
                    base_size = self._calculate_position_size(row['atr14'])
                    
                    if self.ml_predictor and getattr(config, 'ML_CONFIDENCE_SIZING', False):
                        tp_sl = config.TAKE_PROFIT_ATR_MULT / config.STOP_LOSS_ATR_MULT
                        kelly = self.ml_predictor.kelly_size(ml_prob, tp_sl)
                        base_size *= max(kelly * 4, 0.3)  # Scale, keep minimum
                    
                    # Drawdown risk scaling
                    dd_scale = self._get_dd_risk_scale()
                    if dd_scale <= 0:
                        self.stats['dd_blocks'] = self.stats.get('dd_blocks', 0) + 1
                        continue
                    base_size *= dd_scale
                    
                    size_reduced = False
                    if risk_score >= config.NEWS_RISK_THRESHOLD_MED:
                        base_size *= 0.5
                        size_reduced = True
                        self.stats['size_reductions'] += 1
                    else:
                        self.stats['normal_entries'] += 1
                    
                    if base_size > 0:
                        self.position.open_position(
                            direction=direction, price=row['close'],
                            size=base_size, atr=row['atr14'],
                            time=time, bar_idx=bar_idx,
                            risk_score=risk_score, size_reduced=size_reduced
                        )
        
        # Close remaining position
        if self.position.is_open:
            last_row = self.data.iloc[-1]
            record = self.position.close_position(
                last_row['close'], self.data.index[-1], "end_of_data"
            )
            self._process_closed_trade(record)
        
        return self._calculate_metrics()
    
    def _process_closed_trade(self, record: TradeRecord):
        """Process and record a closed trade."""
        self.trade_records.append(record)
        self.capital += record.net_pnl
        self.daily_pnl += record.net_pnl
        self.stats['total_trading_costs'] += record.trading_costs
        self._check_daily_loss_limit()
    
    def _calculate_metrics(self) -> Dict:
        if not self.trade_records:
            return {"error": "No trades executed"}
        
        net_pnls = [t.net_pnl for t in self.trade_records]
        gross_pnls = [t.gross_pnl for t in self.trade_records]
        winning = [t for t in self.trade_records if t.net_pnl > 0]
        losing = [t for t in self.trade_records if t.net_pnl < 0]
        reduced = [t for t in self.trade_records if t.size_reduced]
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Bar-based max drawdown (more accurate than trade-based)
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown_pct = (peak - equity) / peak * 100
        max_drawdown = drawdown_pct.max() if len(drawdown_pct) > 0 else 0.0
        
        # Corrected Sharpe: annualize based on trades per year
        if len(net_pnls) > 1:
            data_years = (self.data.index[-1] - self.data.index[0]).days / 365.25
            trades_per_year = len(net_pnls) / data_years if data_years > 0 else 252
            sharpe = np.mean(net_pnls) / np.std(net_pnls) * np.sqrt(trades_per_year)
        else:
            sharpe = 0
        
        win_pnls = [t.net_pnl for t in winning]
        lose_pnls = [t.net_pnl for t in losing]
        
        return {
            "total_trades": len(self.trade_records),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.trade_records) * 100,
            "total_return_pct": total_return,
            "total_return_gross_pct": sum(gross_pnls) / self.initial_capital * 100,
            "final_capital": self.capital,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "avg_win": np.mean(win_pnls) if win_pnls else 0,
            "avg_loss": np.mean(lose_pnls) if lose_pnls else 0,
            "profit_factor": abs(sum(win_pnls) / sum(lose_pnls)) if lose_pnls else float('inf'),
            "avg_bars_in_trade": np.mean([t.bars_held for t in self.trade_records]),
            "total_trading_costs": self.stats['total_trading_costs'],
            "trades_with_reduced_size": len(reduced),
            "exit_reasons": self._count_exit_reasons(),
            **self.stats
        }
    
    def _count_exit_reasons(self) -> Dict[str, int]:
        reasons = {}
        for trade in self.trade_records:
            reasons[trade.exit_reason] = reasons.get(trade.exit_reason, 0) + 1
        return reasons
    
    def get_trade_log(self) -> pd.DataFrame:
        records = []
        for t in self.trade_records:
            records.append({
                "entry_time": t.entry_time, "exit_time": t.exit_time,
                "direction": t.direction, "entry_price": t.entry_price,
                "exit_price": t.exit_price, "size": t.initial_size,
                "gross_pnl": t.gross_pnl, "costs": t.trading_costs,
                "net_pnl": t.net_pnl, "exit_reason": t.exit_reason,
                "bars_held": t.bars_held, "news_risk": t.news_risk_score,
                "size_reduced": t.size_reduced
            })
        return pd.DataFrame(records)

# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward(data: pd.DataFrame) -> Dict:
    """
    Walk-forward validation: train on early data, test on last N years.
    Returns in-sample and out-of-sample metrics for comparison.
    """
    if not config.WALK_FORWARD_ENABLED:
        return {}
    
    data_start = data['datetime'].min() if 'datetime' in data.columns else data.index.min()
    data_end = data['datetime'].max() if 'datetime' in data.columns else data.index.max()
    
    total_years = (data_end - data_start).days / 365.25
    if total_years < config.WF_MIN_TRAIN_YEARS + config.WF_TEST_YEARS:
        logger.warning("Not enough data for walk-forward (%.1f years, need %d)",
                       total_years, config.WF_MIN_TRAIN_YEARS + config.WF_TEST_YEARS)
        return {}
    
    split_date = data_end - timedelta(days=config.WF_TEST_YEARS * 365)
    
    if 'datetime' in data.columns:
        train_data = data[data['datetime'] < split_date].copy()
        test_data = data[data['datetime'] >= split_date].copy()
    else:
        train_data = data[data.index < split_date].copy()
        test_data = data[data.index >= split_date].copy()
    
    logger.info("Walk-Forward Split: Train=%d bars | Test=%d bars | Split=%s",
                len(train_data), len(test_data), split_date.strftime('%Y-%m-%d'))
    
    # Reset singletons for clean runs
    reset_news_manager()
    reset_intelligence_layer()
    
    # In-sample
    logger.info("Running IN-SAMPLE backtest...")
    bt_train = Backtester(train_data)
    metrics_train = bt_train.run()
    
    # Reset singletons again
    reset_news_manager()
    reset_intelligence_layer()
    
    # Out-of-sample
    logger.info("Running OUT-OF-SAMPLE backtest...")
    bt_test = Backtester(test_data)
    metrics_test = bt_test.run()
    
    return {
        "split_date": split_date.strftime('%Y-%m-%d'),
        "in_sample": metrics_train,
        "out_of_sample": metrics_test,
        "test_trade_log": bt_test.get_trade_log()
    }

# =============================================================================
# CSV DATA LOADER
# =============================================================================

def load_csv_data(filepath: str = None) -> pd.DataFrame:
    filepath = filepath or config.DATA_FILE_PATH
    logger.info("Loading data from: %s", filepath)
    
    df = pd.read_csv(
        filepath, sep=r'\s+', header=None,
        names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
        dtype={'date': str, 'time': str, 'open': float, 'high': float,
               'low': float, 'close': float, 'volume': float},
        on_bad_lines='skip'
    )
    
    logger.info("  Raw rows: %d", len(df))
    
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M')
    df = df.drop(columns=['date', 'time'])
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    invalid = (
        (df['high'] < df['low']) | (df['high'] < df['open']) |
        (df['high'] < df['close']) | (df['low'] > df['open']) | (df['low'] > df['close'])
    )
    df = df[~invalid]
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
    
    logger.info("  Clean rows: %d | Range: %s to %s",
                len(df), df['datetime'].min(), df['datetime'].max())
    return df

def validate_data(df: pd.DataFrame) -> bool:
    min_required = max(config.EMA_PERIOD, config.ATR_PERIOD, config.RSI_PERIOD) + config.ATR_AVG_PERIOD + 100
    if len(df) < min_required:
        logger.error("Not enough data: %d rows (need %d)", len(df), min_required)
        return False
    logger.info("Data validation passed")
    return True

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run backtest with optional walk-forward validation."""
    setup_logging()
    
    print("=" * 70)
    print("NASDAQ H1 Trading System v0.9")
    print("Foundation Hardened — Realistic Cost Simulation")
    print("=" * 70)
    
    # Load data
    print("\n📂 LOADING DATA:")
    try:
        data = load_csv_data()
    except FileNotFoundError:
        print(f"❌ Error: {config.DATA_FILE_PATH} not found")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None
    
    print("\n🔍 VALIDATING:")
    if not validate_data(data):
        return None, None
    
    # === FULL BACKTEST ===
    print("\n🚀 FULL BACKTEST:")
    reset_news_manager()
    reset_intelligence_layer()
    backtester = Backtester(data)
    metrics = backtester.run()
    
    if "error" in metrics:
        print(f"❌ {metrics['error']}")
        return metrics, backtester
    
    _print_results(metrics, "FULL BACKTEST")
    
    # === WALK-FORWARD ===
    if config.WALK_FORWARD_ENABLED:
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION")
        print("=" * 70)
        wf = run_walk_forward(data)
        
        if wf and 'in_sample' in wf and 'out_of_sample' in wf:
            print(f"\n📅 Split Date: {wf['split_date']}")
            
            is_m = wf['in_sample']
            os_m = wf['out_of_sample']
            
            if 'error' not in is_m and 'error' not in os_m:
                print(f"\n{'Metric':<25} {'In-Sample':>12} {'Out-of-Sample':>14}")
                print("-" * 55)
                print(f"{'Trades':<25} {is_m['total_trades']:>12} {os_m['total_trades']:>14}")
                print(f"{'Win Rate':<25} {is_m['win_rate']:>11.1f}% {os_m['win_rate']:>13.1f}%")
                print(f"{'Net Return':<25} {is_m['total_return_pct']:>11.2f}% {os_m['total_return_pct']:>13.2f}%")
                print(f"{'Max Drawdown':<25} {is_m['max_drawdown_pct']:>11.2f}% {os_m['max_drawdown_pct']:>13.2f}%")
                print(f"{'Sharpe Ratio':<25} {is_m['sharpe_ratio']:>12.2f} {os_m['sharpe_ratio']:>14.2f}")
                print(f"{'Profit Factor':<25} {is_m['profit_factor']:>12.2f} {os_m['profit_factor']:>14.2f}")
                print(f"{'Trading Costs':<25} ${is_m['total_trading_costs']:>10,.0f} ${os_m['total_trading_costs']:>12,.0f}")
                
                # Overfitting check
                if is_m['total_return_pct'] > 0 and os_m['total_return_pct'] > 0:
                    ratio = os_m['total_return_pct'] / is_m['total_return_pct']
                    print(f"\n📊 OOS/IS Ratio: {ratio:.2f}", end="")
                    if ratio > 0.5:
                        print(" ✅ Robust (>0.5)")
                    else:
                        print(" ⚠️ Possible overfitting (<0.5)")
                elif os_m['total_return_pct'] <= 0:
                    print("\n⚠️ Out-of-sample return is NEGATIVE — strategy may be overfit")
                
                wf['test_trade_log'].to_csv('trade_log_oos.csv', index=False)
                print("✅ OOS trade log saved to trade_log_oos.csv")
    
    # Save full trade log
    trade_log = backtester.get_trade_log()
    trade_log.to_csv(config.TRADE_LOG_PATH, index=False)
    print(f"\n✅ Full trade log saved to {config.TRADE_LOG_PATH}")
    
    return metrics, backtester


def _print_results(metrics: Dict, title: str):
    """Print formatted backtest results."""
    print(f"\n{'=' * 70}")
    print(f"{title} RESULTS")
    print(f"{'=' * 70}")
    
    print(f"\n📊 TRADES:")
    print(f"   Total: {metrics['total_trades']} | Won: {metrics['winning_trades']} | Lost: {metrics['losing_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    
    print(f"\n💰 RETURNS:")
    print(f"   Gross Return: {metrics['total_return_gross_pct']:.2f}%")
    print(f"   Trading Costs: ${metrics['total_trading_costs']:,.0f}")
    print(f"   Net Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Final Capital: ${metrics['final_capital']:,.2f}")
    
    print(f"\n📉 RISK:")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    
    print(f"\n📰 NEWS FILTER:")
    print(f"   Signals: {metrics['entry_signals']} | Blocked: {metrics['news_blocks']} | Reduced: {metrics['size_reductions']}")
    
    print(f"\n🚪 EXITS: ", end="")
    print(" | ".join(f"{k}: {v}" for k, v in metrics['exit_reasons'].items()))
    
    print(f"\n⏱️ Avg Duration: {metrics['avg_bars_in_trade']:.1f} bars | Daily Stops: {metrics['daily_stops']}")


if __name__ == "__main__":
    metrics, backtester = main()
