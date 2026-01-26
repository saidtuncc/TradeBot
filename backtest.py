# backtest.py
# System v0.7: Dynamic Position Management Engine
# NASDAQ H1 Trend-Following with Pyramiding & Hedging
# Dependencies: pandas, numpy

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from enum import Enum
import config

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

@dataclass
class Layer:
    """Represents a single entry layer in a pyramided position."""
    entry_price: float
    size: float
    entry_time: datetime
    atr_at_entry: float
    layer_id: int

@dataclass 
class TradeRecord:
    """Complete record of a closed trade for logging."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    initial_entry_price: float
    avg_entry_price: float
    exit_price: float
    initial_size: float
    max_size: float
    final_size: float
    num_layers: int
    pnl: float
    exit_reason: str
    bars_held: int
    hedge_activated: bool
    scaled_in: bool

# =============================================================================
# POSITION MANAGER CLASS
# =============================================================================

class PositionManager:
    """
    State machine for dynamic position management.
    Handles pyramiding (scaling in) and hedging (risk neutralization).
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to flat state."""
        self.direction: Direction = Direction.FLAT
        self.layers: List[Layer] = []
        self.current_sl: float = 0.0
        self.current_tp: float = 0.0
        self.entry_bar_idx: int = 0
        self.bars_held: int = 0
        
        # Hedge state
        self.hedge_active: bool = False
        self.hedge_size: float = 0.0
        self.hedge_bars: int = 0
        
        # Tracking
        self.initial_size: float = 0.0
        self.max_size_reached: float = 0.0
        self.hedge_was_activated: bool = False
        self.scaled_in: bool = False
        self.initial_entry_price: float = 0.0
    
    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.direction != Direction.FLAT and len(self.layers) > 0
    
    @property
    def total_size(self) -> float:
        """Total position size across all layers."""
        return sum(layer.size for layer in self.layers)
    
    @property
    def net_exposure(self) -> float:
        """Net exposure after hedge."""
        gross = self.total_size
        if self.hedge_active:
            return max(0, gross - self.hedge_size)
        return gross
    
    @property
    def avg_entry_price(self) -> float:
        """Weighted average entry price across layers."""
        if not self.layers:
            return 0.0
        total_value = sum(layer.entry_price * layer.size for layer in self.layers)
        total_size = self.total_size
        return total_value / total_size if total_size > 0 else 0.0
    
    @property
    def num_layers(self) -> int:
        """Number of active layers."""
        return len(self.layers)
    
    def open_position(self, direction: Direction, price: float, size: float, 
                     atr: float, time: datetime, bar_idx: int) -> None:
        """Open a new position with initial layer."""
        self.reset()
        self.direction = direction
        self.entry_bar_idx = bar_idx
        self.initial_size = size
        self.initial_entry_price = price
        self.max_size_reached = size
        
        # Create initial layer
        layer = Layer(
            entry_price=price,
            size=size,
            entry_time=time,
            atr_at_entry=atr,
            layer_id=1
        )
        self.layers.append(layer)
        
        # Set SL and TP
        if direction == Direction.LONG:
            self.current_sl = price - config.STOP_LOSS_ATR_MULT * atr
            self.current_tp = price + config.TAKE_PROFIT_ATR_MULT * atr
        else:  # SHORT
            self.current_sl = price + config.STOP_LOSS_ATR_MULT * atr
            self.current_tp = price - config.TAKE_PROFIT_ATR_MULT * atr
    
    def add_layer(self, price: float, size: float, atr: float, time: datetime) -> bool:
        """
        Add a scaling layer (pyramiding).
        Returns True if layer was added, False if at max layers.
        """
        if not self.is_open:
            return False
        
        if self.num_layers >= config.MAX_LAYERS:
            return False
        
        layer = Layer(
            entry_price=price,
            size=size,
            entry_time=time,
            atr_at_entry=atr,
            layer_id=self.num_layers + 1
        )
        self.layers.append(layer)
        self.scaled_in = True
        self.max_size_reached = max(self.max_size_reached, self.total_size)
        
        # Move SL to break-even if configured
        if config.MOVE_SL_TO_BE_ON_SCALE:
            self.update_sl_to_breakeven()
        
        return True
    
    def update_sl_to_breakeven(self) -> None:
        """Move stop loss to break-even (average entry price)."""
        if self.direction == Direction.LONG:
            # For long, SL can only move up
            new_sl = self.avg_entry_price
            if new_sl > self.current_sl:
                self.current_sl = new_sl
        else:  # SHORT
            # For short, SL can only move down
            new_sl = self.avg_entry_price
            if new_sl < self.current_sl:
                self.current_sl = new_sl
    
    def activate_hedge(self) -> bool:
        """
        Activate partial hedge to reduce exposure during volatility.
        For netting simulation: reduces effective exposure.
        """
        if self.hedge_active:
            return False
        
        self.hedge_size = self.total_size * config.HEDGE_RATIO
        self.hedge_active = True
        self.hedge_bars = 0
        self.hedge_was_activated = True
        return True
    
    def deactivate_hedge(self) -> None:
        """Remove hedge, restore full exposure."""
        self.hedge_active = False
        self.hedge_size = 0.0
        self.hedge_bars = 0
    
    def check_hedge_timeout(self) -> bool:
        """Check if hedge has been active too long."""
        if self.hedge_active:
            self.hedge_bars += 1
            if self.hedge_bars >= config.HEDGE_MAX_BARS:
                return True
        return False
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL based on current price."""
        if not self.is_open:
            return 0.0
        
        if self.direction == Direction.LONG:
            price_change = current_price - self.avg_entry_price
        else:
            price_change = self.avg_entry_price - current_price
        
        return price_change * self.net_exposure
    
    def calculate_unrealized_pnl_in_atr(self, current_price: float, atr: float) -> float:
        """Calculate unrealized PnL normalized by ATR."""
        if not self.is_open or atr <= 0:
            return 0.0
        
        if self.direction == Direction.LONG:
            price_change = current_price - self.avg_entry_price
        else:
            price_change = self.avg_entry_price - current_price
        
        return price_change / atr
    
    def close_position(self, exit_price: float, exit_time: datetime, 
                       exit_reason: str) -> TradeRecord:
        """Close entire position and return trade record."""
        # Calculate final PnL (on net exposure, accounting for any hedge)
        if self.direction == Direction.LONG:
            price_change = exit_price - self.avg_entry_price
        else:
            price_change = self.avg_entry_price - exit_price
        
        # PnL is based on net exposure (hedged portion doesn't contribute to PnL)
        pnl = price_change * self.net_exposure
        
        record = TradeRecord(
            entry_time=self.layers[0].entry_time,
            exit_time=exit_time,
            direction=self.direction.name,
            initial_entry_price=self.initial_entry_price,
            avg_entry_price=self.avg_entry_price,
            exit_price=exit_price,
            initial_size=self.initial_size,
            max_size=self.max_size_reached,
            final_size=self.net_exposure,
            num_layers=self.num_layers,
            pnl=pnl,
            exit_reason=exit_reason,
            bars_held=self.bars_held,
            hedge_activated=self.hedge_was_activated,
            scaled_in=self.scaled_in
        )
        
        self.reset()
        return record

# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    for i in range(period, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index) and +DI/-DI.
    
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr
    
    # Calculate DX
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return adx, plus_di, minus_di

# =============================================================================
# VOLATILITY ANOMALY DETECTION
# =============================================================================

def detect_candle_anomaly(candle_range: float, atr: float) -> bool:
    """Detect if current candle is abnormally large."""
    threshold = atr * config.CANDLE_ANOMALY_MULT
    return candle_range > threshold

def detect_atr_spike(atr: float, atr_avg: float) -> bool:
    """Detect if ATR has spiked above normal levels."""
    threshold = atr_avg * config.ATR_SPIKE_MULT
    return atr > threshold

def is_volatility_anomaly(row: pd.Series) -> Tuple[bool, str]:
    """Combined volatility anomaly check."""
    candle_range = row['high'] - row['low']
    atr = row['atr14']
    atr_avg = row['atr_avg']
    
    if detect_candle_anomaly(candle_range, atr):
        return True, "candle_anomaly"
    if detect_atr_spike(atr, atr_avg):
        return True, "atr_spike"
    return False, ""

def should_trigger_hedge(candle_range: float, atr: float) -> bool:
    """Check if volatility spike should trigger hedge (while in trade)."""
    return candle_range > atr * config.HEDGE_TRIGGER_MULT

def should_exit_hedge(candle_range: float, atr: float) -> bool:
    """Check if volatility has subsided enough to exit hedge."""
    return candle_range < atr * config.HEDGE_EXIT_MULT

# =============================================================================
# BACKTESTING ENGINE v0.7
# =============================================================================

class Backtester:
    """
    Event-driven backtester with dynamic position management.
    Supports pyramiding (scaling in) and hedging (risk neutralization).
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
        
        # Statistics
        self.stats = {
            'anomaly_bars': 0,
            'entry_blocks': 0,
            'scale_ins': 0,
            'hedges_activated': 0,
            'hedge_timeouts': 0
        }
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Calculate all indicators (vectorized)."""
        df = self.data
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        # Core indicators
        df['ema50'] = calculate_ema(df['close'], config.EMA_PERIOD)
        df['atr14'] = calculate_atr(df['high'], df['low'], df['close'], config.ATR_PERIOD)
        df['atr_avg'] = df['atr14'].rolling(window=config.ATR_AVG_PERIOD).mean()
        df['rsi14'] = calculate_rsi(df['close'], config.RSI_PERIOD)
        
        # ADX (NEW)
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
            df['high'], df['low'], df['close'], config.ADX_PERIOD
        )
        df['adx_rising'] = df['adx'] > df['adx'].shift(1)
        
        # Candle range for volatility checks
        df['candle_range'] = df['high'] - df['low']
        
        self.data = df.dropna()
    
    def _check_daily_reset(self, current_time: datetime):
        """Reset daily PnL tracking at start of new day."""
        current_day = current_time.date()
        if self.current_date != current_day:
            self.current_date = current_day
            self.daily_pnl = 0.0
            self.stopped_for_day = False
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been hit."""
        max_loss = self.initial_capital * config.MAX_DAILY_LOSS
        if self.daily_pnl <= -max_loss:
            self.stopped_for_day = True
            return True
        return False
    
    def _calculate_position_size(self, atr: float) -> float:
        """Calculate base position size from risk parameters."""
        risk_amount = self.capital * config.RISK_PER_TRADE
        stop_distance = config.STOP_LOSS_ATR_MULT * atr
        if stop_distance <= 0:
            return 0
        return risk_amount / stop_distance
    
    def _check_entry_conditions(self, row: pd.Series) -> Optional[Direction]:
        """
        Check if entry conditions are met.
        Includes ADX chop filter.
        """
        close = row['close']
        ema50 = row['ema50']
        rsi = row['rsi14']
        atr = row['atr14']
        atr_avg = row['atr_avg']
        adx = row['adx']
        
        # Minimum volatility filter
        if atr < config.ATR_FILTER_RATIO * atr_avg:
            return None
        
        # ADX Chop Filter (NEW) - Do not enter choppy markets
        if adx < config.ADX_MIN_THRESHOLD:
            return None
        
        # Long conditions
        if close > ema50:
            if config.RSI_LONG_MIN <= rsi <= config.RSI_LONG_MAX:
                return Direction.LONG
        
        # Short conditions
        if close < ema50:
            if config.RSI_SHORT_MIN <= rsi <= config.RSI_SHORT_MAX:
                return Direction.SHORT
        
        return None
    
    def _check_scaling_conditions(self, row: pd.Series) -> bool:
        """
        Check if we should add a scaling layer (pyramid).
        
        Conditions:
        1. Position is open and profitable
        2. Unrealized profit > SCALE_IN_PROFIT_ATR × ATR
        3. ADX is rising (trend strengthening)
        4. Not at max layers
        5. Hedge is not active
        """
        if not config.PYRAMIDING_ENABLED:
            return False
        
        if not self.position.is_open:
            return False
        
        if self.position.hedge_active:
            return False
        
        if self.position.num_layers >= config.MAX_LAYERS:
            return False
        
        # Check profit threshold
        profit_atr = self.position.calculate_unrealized_pnl_in_atr(
            row['close'], row['atr14']
        )
        if profit_atr <= config.SCALE_IN_PROFIT_ATR:
            return False
        
        # Check ADX is rising
        if not row['adx_rising']:
            return False
        
        # ADX should be showing strong trend
        if row['adx'] < config.ADX_STRONG_TREND:
            return False
        
        return True
    
    def _check_exit_conditions(self, row: pd.Series) -> Tuple[bool, str, float]:
        """
        Check if position should be exited.
        Returns: (should_exit, reason, exit_price)
        """
        if not self.position.is_open:
            return False, "", 0.0
        
        pos = self.position
        high = row['high']
        low = row['low']
        close = row['close']
        
        pos.bars_held += 1
        
        # Check stop loss
        if pos.direction == Direction.LONG:
            if low <= pos.current_sl:
                return True, "stop_loss", pos.current_sl
            if high >= pos.current_tp:
                return True, "take_profit", pos.current_tp
        else:  # SHORT
            if high >= pos.current_sl:
                return True, "stop_loss", pos.current_sl
            if low <= pos.current_tp:
                return True, "take_profit", pos.current_tp
        
        # Time-based exit
        max_bars = config.HIGH_VOL_MAX_BARS if pos.hedge_active else config.MAX_BARS_IN_TRADE
        if pos.bars_held >= max_bars:
            return True, "time_exit", close
        
        return False, "", 0.0
    
    def _manage_position(self, row: pd.Series, time: datetime, bar_idx: int):
        """
        Main position management logic.
        Handles scaling, hedging, and exits.
        """
        if not self.position.is_open:
            return
        
        pos = self.position
        candle_range = row['candle_range']
        atr = row['atr14']
        
        # === HEDGE MANAGEMENT ===
        if config.HEDGING_ENABLED:
            # Check if we should activate hedge (volatility spike while in trade)
            if not pos.hedge_active and should_trigger_hedge(candle_range, atr):
                pos.activate_hedge()
                self.stats['hedges_activated'] += 1
            
            # Check if we should deactivate hedge (volatility subsided)
            elif pos.hedge_active:
                if should_exit_hedge(candle_range, atr):
                    pos.deactivate_hedge()
                elif pos.check_hedge_timeout():
                    pos.deactivate_hedge()
                    self.stats['hedge_timeouts'] += 1
        
        # === SCALING MANAGEMENT ===
        if self._check_scaling_conditions(row):
            scale_size = pos.initial_size * config.SCALE_IN_SIZE_RATIO
            if pos.add_layer(row['close'], scale_size, atr, time):
                self.stats['scale_ins'] += 1
    
    def run(self) -> Dict:
        """Run the backtest with dynamic position management."""
        print("Starting backtest v0.7 (Dynamic Position Management)...")
        print(f"Data range: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"Total bars: {len(self.data)}")
        print(f"Pyramiding: {'ENABLED' if config.PYRAMIDING_ENABLED else 'DISABLED'}")
        print(f"Hedging: {'ENABLED' if config.HEDGING_ENABLED else 'DISABLED'}")
        print("-" * 50)
        
        for bar_idx, (time, row) in enumerate(self.data.iterrows()):
            self._check_daily_reset(time)
            
            # Skip if stopped for day
            if self.stopped_for_day:
                if self.position.is_open:
                    should_exit, reason, exit_price = self._check_exit_conditions(row)
                    if should_exit:
                        record = self.position.close_position(exit_price, time, reason)
                        self.trade_records.append(record)
                        self.capital += record.pnl
                        self.daily_pnl += record.pnl
                continue
            
            # Check volatility anomaly
            is_anomaly, anomaly_reason = is_volatility_anomaly(row)
            if is_anomaly:
                self.stats['anomaly_bars'] += 1
            
            # === MANAGE EXISTING POSITION ===
            if self.position.is_open:
                # Position management (scaling, hedging)
                self._manage_position(row, time, bar_idx)
                
                # Check exits
                should_exit, reason, exit_price = self._check_exit_conditions(row)
                if should_exit:
                    record = self.position.close_position(exit_price, time, reason)
                    self.trade_records.append(record)
                    self.capital += record.pnl
                    self.daily_pnl += record.pnl
                    self._check_daily_loss_limit()
            
            # === CHECK FOR NEW ENTRY ===
            if not self.position.is_open and not self.stopped_for_day:
                # Block new entries during volatility anomaly (if filter enabled)
                if config.VOLATILITY_FILTER_ENABLED and is_anomaly:
                    self.stats['entry_blocks'] += 1
                    continue
                
                direction = self._check_entry_conditions(row)
                if direction:
                    size = self._calculate_position_size(row['atr14'])
                    self.position.open_position(
                        direction=direction,
                        price=row['close'],
                        size=size,
                        atr=row['atr14'],
                        time=time,
                        bar_idx=bar_idx
                    )
        
        # Close any remaining position
        if self.position.is_open:
            last_row = self.data.iloc[-1]
            record = self.position.close_position(
                last_row['close'], 
                self.data.index[-1], 
                "end_of_data"
            )
            self.trade_records.append(record)
            self.capital += record.pnl
        
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trade_records:
            return {"error": "No trades executed"}
        
        pnls = [t.pnl for t in self.trade_records]
        winning_trades = [t for t in self.trade_records if t.pnl > 0]
        losing_trades = [t for t in self.trade_records if t.pnl < 0]
        scaled_trades = [t for t in self.trade_records if t.scaled_in]
        hedged_trades = [t for t in self.trade_records if t.hedge_activated]
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Max drawdown
        equity_curve = [self.initial_capital]
        for trade in self.trade_records:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 24)
        else:
            sharpe = 0
        
        avg_bars = np.mean([t.bars_held for t in self.trade_records])
        avg_layers = np.mean([t.num_layers for t in self.trade_records])
        
        winning_pnls = [t.pnl for t in winning_trades]
        losing_pnls = [t.pnl for t in losing_trades]
        
        metrics = {
            "total_trades": len(self.trade_records),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trade_records) * 100,
            "total_return_pct": total_return,
            "final_capital": self.capital,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "avg_win": np.mean(winning_pnls) if winning_pnls else 0,
            "avg_loss": np.mean(losing_pnls) if losing_pnls else 0,
            "profit_factor": abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls else float('inf'),
            "avg_bars_in_trade": avg_bars,
            "avg_layers_per_trade": avg_layers,
            "trades_with_scaling": len(scaled_trades),
            "trades_with_hedge": len(hedged_trades),
            "exit_reasons": self._count_exit_reasons(),
            **self.stats
        }
        
        return metrics
    
    def _count_exit_reasons(self) -> Dict[str, int]:
        """Count trades by exit reason."""
        reasons = {}
        for trade in self.trade_records:
            reasons[trade.exit_reason] = reasons.get(trade.exit_reason, 0) + 1
        return reasons
    
    def get_trade_log(self) -> pd.DataFrame:
        """Return detailed trade log as DataFrame."""
        records = []
        for t in self.trade_records:
            records.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "initial_entry": t.initial_entry_price,
                "avg_entry": t.avg_entry_price,
                "exit_price": t.exit_price,
                "initial_size": t.initial_size,
                "max_size": t.max_size,
                "final_size": t.final_size,
                "num_layers": t.num_layers,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "scaled_in": t.scaled_in,
                "hedge_activated": t.hedge_activated
            })
        return pd.DataFrame(records)

# =============================================================================
# CSV DATA LOADER
# =============================================================================

def load_csv_data(filepath: str = "USATECHIDXUSD60.csv") -> pd.DataFrame:
    """Load NASDAQ H1 data from CSV file."""
    print(f"Loading data from: {filepath}")
    
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
        dtype={
            'date': str, 'time': str,
            'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
        },
        on_bad_lines='skip'
    )
    
    print(f"  Raw rows loaded: {len(df)}")
    
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M')
    df = df.drop(columns=['date', 'time'])
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    # Data cleaning
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    invalid_ohlc = (
        (df['high'] < df['low']) | (df['high'] < df['open']) | 
        (df['high'] < df['close']) | (df['low'] > df['open']) | (df['low'] > df['close'])
    )
    df = df[~invalid_ohlc]
    
    zero_price = (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)
    df = df[~zero_price]
    
    print(f"  Clean rows: {len(df)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df

def validate_data(df: pd.DataFrame) -> bool:
    """Validate data is suitable for backtesting."""
    min_required = max(config.EMA_PERIOD, config.ATR_PERIOD, config.RSI_PERIOD, config.ADX_PERIOD) + config.ATR_AVG_PERIOD + 100
    if len(df) < min_required:
        print(f"Not enough data: {len(df)} rows (need {min_required})")
        return False
    print("✅ Data validation passed")
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run backtest with dynamic position management."""
    print("=" * 70)
    print("NASDAQ H1 Trading System v0.7")
    print("Dynamic Position Management: Pyramiding + Hedging")
    print("=" * 70)
    
    print("\n📂 LOADING DATA:")
    try:
        data = load_csv_data("USATECHIDXUSD60.csv")
    except FileNotFoundError:
        print("❌ Error: USATECHIDXUSD60.csv not found")
        return None, None
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None, None
    
    print("\n🔍 VALIDATING DATA:")
    if not validate_data(data):
        return None, None
    
    print("\n🚀 RUNNING BACKTEST:")
    backtester = Backtester(data)
    metrics = backtester.run()
    
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    if "error" in metrics:
        print(f"❌ {metrics['error']}")
        return metrics, backtester
    
    print(f"\n📊 TRADE STATISTICS:")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Winning: {metrics['winning_trades']} | Losing: {metrics['losing_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    
    print(f"\n💰 RETURNS:")
    print(f"   Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"   Final Capital: ${metrics['final_capital']:,.2f}")
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
    
    print(f"\n📉 RISK METRICS:")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    
    print(f"\n📈 POSITION MANAGEMENT:")
    print(f"   Avg Bars in Trade: {metrics['avg_bars_in_trade']:.1f}")
    print(f"   Avg Layers per Trade: {metrics['avg_layers_per_trade']:.2f}")
    print(f"   Trades with Scaling: {metrics['trades_with_scaling']}")
    print(f"   Trades with Hedge: {metrics['trades_with_hedge']}")
    
    print(f"\n🔥 FILTER STATISTICS:")
    print(f"   Anomaly Bars: {metrics['anomaly_bars']}")
    print(f"   Entry Blocks: {metrics['entry_blocks']}")
    print(f"   Scale-Ins Executed: {metrics['scale_ins']}")
    print(f"   Hedges Activated: {metrics['hedges_activated']}")
    
    print(f"\n🚪 EXIT REASONS:")
    for reason, count in metrics['exit_reasons'].items():
        print(f"   {reason}: {count}")
    
    trade_log = backtester.get_trade_log()
    trade_log.to_csv('trade_log.csv', index=False)
    print(f"\n✅ Trade log saved to trade_log.csv")
    
    return metrics, backtester

if __name__ == "__main__":
    metrics, backtester = main()
