# live_trader.py
"""
MT5 Live Trader — Connects ML models to MetaTrader 5 for automated trading.
Runs on Windows VPS. Checks every H1 bar close for ML signals.

Usage:
    python live_trader.py              # Run with default settings
    python live_trader.py --dry-run    # Log signals without placing orders
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

# ─── Setup ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('live_trader.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger('live_trader')

# ─── Configuration ───────────────────────────────────────────────────────────

# MT5 connection (update these with your demo account details)
MT5_LOGIN = 0           # Your MT5 login number
MT5_PASSWORD = ""       # Your MT5 password
MT5_SERVER = ""         # Your MT5 server name
MT5_PATH = None         # Path to MT5 terminal.exe (auto-detect if None)

# Trading
SYMBOL = "USATECHIDXUSD"   # MT5 symbol name (check your broker's name)
TIMEFRAME_H1 = None         # Set after MT5 import
VOLUME = 0.10               # Minimum lot for NASDAQ CFD
CHECK_INTERVAL_SEC = 60     # Check every 60 seconds
LOOKBACK_BARS = 200         # Bars to fetch for feature calculation

# Mapping of config -> MT5 symbol alternatives
SYMBOL_ALTERNATIVES = [
    "US100Cash", "US100", "USATECHIDXUSD", "USTEC", "NAS100",
    "NASDAQ", "Nasdaq", "US100.cash", "#NAS100", "NAS100.cash",
    "GerTech30Cash", "US100-MAR26",
]


class MT5Connector:
    """Handles MetaTrader 5 connection and order execution."""

    def __init__(self, login=None, password=None, server=None, path=None):
        import MetaTrader5 as mt5
        self.mt5 = mt5
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.symbol = None

    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        kwargs = {}
        if self.path:
            kwargs['path'] = self.path
        if self.login:
            kwargs['login'] = self.login
            kwargs['password'] = self.password
            kwargs['server'] = self.server

        if not self.mt5.initialize(**kwargs):
            logger.error("MT5 initialize failed: %s", self.mt5.last_error())
            return False

        info = self.mt5.account_info()
        if info:
            logger.info("═══ MT5 Connected ═══")
            logger.info("  Account: %d (%s)", info.login, info.server)
            logger.info("  Balance: $%.2f", info.balance)
            logger.info("  Leverage: 1:%d", info.leverage)
            logger.info("  Trade mode: %s", "Demo" if info.trade_mode == 0 else "Real")
        self.connected = True
        return True

    def find_symbol(self) -> Optional[str]:
        """Find the correct symbol name on this broker."""
        for sym in SYMBOL_ALTERNATIVES:
            info = self.mt5.symbol_info(sym)
            if info is not None:
                if not info.visible:
                    self.mt5.symbol_select(sym, True)
                self.symbol = sym
                logger.info("  Symbol found: %s (digits=%d, spread=%d)",
                            sym, info.digits, info.spread)
                return sym
        logger.error("Could not find NASDAQ symbol. Tried: %s", SYMBOL_ALTERNATIVES)
        return None

    def get_bars(self, timeframe, count: int) -> Optional[pd.DataFrame]:
        """Fetch OHLCV bars from MT5."""
        rates = self.mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No bars returned for %s", self.symbol)
            return None

        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    def get_position(self) -> Optional[Dict]:
        """Get current open position for our symbol."""
        positions = self.mt5.positions_get(symbol=self.symbol)
        if positions and len(positions) > 0:
            pos = positions[0]
            return {
                'ticket': pos.ticket,
                'type': 'long' if pos.type == 0 else 'short',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'time': datetime.fromtimestamp(pos.time),
            }
        return None

    def place_order(self, direction: str, volume: float,
                    sl: float, tp: float) -> bool:
        """Place a market order with SL/TP."""
        symbol_info = self.mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error("Symbol info not available")
            return False

        if direction == 'long':
            order_type = self.mt5.ORDER_TYPE_BUY
            price = self.mt5.symbol_info_tick(self.symbol).ask
        else:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(self.symbol).bid

        # Round to symbol precision
        digits = symbol_info.digits
        price = round(price, digits)
        sl = round(sl, digits)
        tp = round(tp, digits)
        
        # Enforce broker's volume constraints
        vol_min = symbol_info.volume_min
        vol_step = symbol_info.volume_step
        volume = max(volume, vol_min)
        volume = round(round(volume / vol_step) * vol_step, 5)

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 20260312,
            "comment": f"TradeBot ML {direction.upper()}",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        result = self.mt5.order_send(request)
        if result is None:
            logger.error("Order send failed: %s", self.mt5.last_error())
            return False

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            logger.error("Order rejected: %s (code=%d)", result.comment, result.retcode)
            return False

        logger.info("✅ ORDER EXECUTED: %s %.2f lots @ %.2f | SL=%.2f TP=%.2f | ticket=%d",
                     direction.upper(), volume, price, sl, tp, result.order)
        return True

    def close_position(self, ticket: int) -> bool:
        """Close position by ticket."""
        pos = self.mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        pos = pos[0]

        if pos.type == 0:  # Long → sell to close
            order_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(self.symbol).bid
        else:  # Short → buy to close
            order_type = self.mt5.ORDER_TYPE_BUY
            price = self.mt5.symbol_info_tick(self.symbol).ask

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 20260312,
            "comment": "TradeBot close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        result = self.mt5.order_send(request)
        if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
            logger.info("✅ POSITION CLOSED: ticket=%d profit=%.2f", ticket, pos.profit)
            return True
        logger.error("Close failed: %s", result.comment if result else "None")
        return False

    def disconnect(self):
        self.mt5.shutdown()
        self.connected = False


class LiveTrader:
    """Main live trading loop: fetch bars → compute features → ML predict → trade."""

    def __init__(self, connector: MT5Connector, dry_run: bool = False):
        self.connector = connector
        self.dry_run = dry_run
        self.last_bar_time = None
        self.trade_count = 0
        self.daily_pnl = 0.0
        self.current_day = None

        # Load ML predictor (LightGBM only, skip LSTM for RAM)
        self.predictor = None
        try:
            from ml.predictor import get_predictor
            self.predictor = get_predictor()
            if self.predictor and self.predictor.loaded:
                logger.info("ML Predictor loaded: ✅")
            else:
                logger.warning("ML Predictor not loaded — will use rule-based only")
                self.predictor = None
        except Exception as e:
            logger.error("ML Predictor init error: %s", e)

        # Load config
        import config
        self.config = config

        # Load live news calendar
        self.news_mgr = None
        try:
            from live_news import get_live_news
            self.news_mgr = get_live_news()
            self.news_mgr.refresh_events(force=True)
            logger.info("Live Calendar: ✅")
        except Exception as e:
            logger.warning("Live Calendar not available: %s", e)

    def run(self):
        """Main loop: check every minute, trade on new H1 bar."""
        import MetaTrader5 as mt5

        logger.info("═══ Live Trader Started ═══")
        logger.info("  Symbol: %s", self.connector.symbol)
        logger.info("  Mode: %s", "DRY RUN (no real orders)" if self.dry_run else "LIVE")
        logger.info("  ML: %s", "Active" if self.predictor else "Disabled")
        logger.info("  Checking every %d seconds for new H1 bar...", CHECK_INTERVAL_SEC)

        try:
            while True:
                try:
                    self._tick(mt5)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error("Tick error: %s", e, exc_info=True)

                time.sleep(CHECK_INTERVAL_SEC)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.connector.disconnect()
            logger.info("═══ Live Trader Stopped ═══")

    def _tick(self, mt5):
        """Single tick: manage positions, check for new bar, evaluate, trade."""
        # Get latest bars
        bars = self.connector.get_bars(mt5.TIMEFRAME_H1, LOOKBACK_BARS)
        if bars is None or len(bars) < 50:
            return

        # ─── Position Management (every tick) ─────────────────────────────
        positions = self._get_all_positions()
        self._manage_positions(positions, bars, mt5)

        # ─── New Bar Check ────────────────────────────────────────────────
        completed_bar = bars.iloc[-2]
        completed_time = completed_bar['datetime']

        if self.last_bar_time is not None and completed_time <= self.last_bar_time:
            return  # No new bar yet

        self.last_bar_time = completed_time
        now = datetime.now()

        # Daily reset
        today = now.date()
        if self.current_day != today:
            self.current_day = today
            self.daily_pnl = 0.0
            self.trade_count = 0
            logger.info("── New day: %s ──", today)

        logger.info("New H1 bar: %s | C=%.2f H=%.2f L=%.2f",
                     completed_time, completed_bar['close'],
                     completed_bar['high'], completed_bar['low'])

        # Log open positions
        if positions:
            for pos in positions:
                logger.info("  📌 %s %.2f lots, P&L=$%.2f (ticket=%d)",
                             pos['type'].upper(), pos['volume'],
                             pos['profit'], pos['ticket'])

        # ─── Daily loss limit ────────────────────────────────────────────
        account = mt5.account_info()
        if account:
            daily_limit = account.balance * self.config.MAX_DAILY_LOSS
            if self.daily_pnl <= -daily_limit:
                logger.warning("  ⛔ Daily loss limit reached ($%.2f)", self.daily_pnl)
                return

        # ─── Check if we can open more positions ─────────────────────────
        max_pos = self.config.MAX_CONCURRENT_POSITIONS
        if len(positions) >= max_pos:
            logger.info("  Max positions reached (%d/%d)", len(positions), max_pos)
            return

        # ─── ML Signal Generation ────────────────────────────────────────
        direction, probability = self._get_ml_signal(bars)

        if direction is None:
            logger.info("  No ML signal")
            return

        logger.info("  📊 ML SIGNAL: %s (prob=%.3f)", direction.upper(), probability)

        # ─── News Check ──────────────────────────────────────────────────
        news_scale = 1.0
        if self.news_mgr:
            score, events = self.news_mgr.calculate_risk_score()
            logger.info("  %s", self.news_mgr.get_status_line())

            if score >= 7.0:
                # HIGH risk: only block if event is UPCOMING (not past)
                future_events = [e for e in events
                                if (e.timestamp - now).total_seconds() > 0]
                if future_events:
                    logger.warning("  ⏳ NEWS WAIT: %s in %.1fh — waiting for post-event",
                                    future_events[0].name,
                                    (future_events[0].timestamp - now).total_seconds() / 3600)
                    return
                else:
                    # Event just passed → volatility play! Trade with the move
                    logger.info("  🔥 POST-NEWS: Event passed, trading momentum")
                    news_scale = 1.0  # Full size on post-event momentum
            elif score >= 4.0:
                news_scale = 0.5
                logger.info("  ⚠️ NEWS NEARBY: score=%.1f — half volume", score)

        # ─── Check if this is a pyramid or new trade ─────────────────────
        same_direction_pos = [p for p in positions if p['type'] == direction]
        opposite_pos = [p for p in positions if p['type'] != direction]

        # Don't open opposite direction while positions exist
        if opposite_pos:
            logger.info("  Opposite position open — skipping %s signal", direction.upper())
            return

        # Pyramiding check
        is_pyramid = len(same_direction_pos) > 0
        if is_pyramid:
            if not self._should_pyramid(same_direction_pos, bars, direction):
                return

        # ─── Calculate SL/TP/Volume ──────────────────────────────────────
        atr = self._calculate_atr(bars)
        if atr <= 0:
            return

        current_price = float(completed_bar['close'])

        if direction == 'long':
            sl = current_price - self.config.STOP_LOSS_ATR_MULT * atr
            tp = current_price + self.config.TAKE_PROFIT_ATR_MULT * atr
        else:
            sl = current_price + self.config.STOP_LOSS_ATR_MULT * atr
            tp = current_price - self.config.TAKE_PROFIT_ATR_MULT * atr

        # Position sizing
        volume = VOLUME
        if self.predictor and self.config.ML_CONFIDENCE_SIZING:
            tp_sl = self.config.TAKE_PROFIT_ATR_MULT / self.config.STOP_LOSS_ATR_MULT
            kelly = self.predictor.kelly_size(probability, tp_sl)
            volume = max(round(VOLUME * max(kelly * 4, 0.5), 2), VOLUME)

        # Apply news scaling
        volume = max(round(volume * news_scale, 2), VOLUME)

        # Pyramid size decay
        if is_pyramid:
            layer = len(same_direction_pos)
            decay = self.config.PYRAMID_SIZE_DECAY ** layer
            volume = max(round(volume * decay, 2), VOLUME)
            logger.info("  📐 PYRAMID layer %d (%.0f%% size)", layer + 1, decay * 100)

        logger.info("  → SL=%.2f TP=%.2f ATR=%.2f Vol=%.2f", sl, tp, atr, volume)

        # ─── Execute ─────────────────────────────────────────────────────
        label = "PYRAMID" if is_pyramid else "NEW"
        if self.dry_run:
            logger.info("  🔸 DRY RUN: Would %s %s %.2f @ %.2f",
                         label, direction.upper(), volume, current_price)
            self._log_signal(direction, probability, current_price, sl, tp, volume, f"DRY_{label}")
        else:
            success = self.connector.place_order(direction, volume, sl, tp)
            if success:
                self.trade_count += 1
                self._log_signal(direction, probability, current_price, sl, tp, volume, label)
            else:
                self._log_signal(direction, probability, current_price, sl, tp, volume, "REJECTED")

    def _get_all_positions(self) -> list:
        """Get all open positions for our symbol."""
        try:
            positions = self.connector.mt5.positions_get(symbol=self.connector.symbol)
            if not positions:
                return []
            result = []
            for pos in positions:
                result.append({
                    'ticket': pos.ticket,
                    'type': 'long' if pos.type == 0 else 'short',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'time': datetime.fromtimestamp(pos.time),
                })
            return result
        except Exception:
            return []

    def _manage_positions(self, positions: list, bars, mt5):
        """Check open positions for time-exit (max 48h hold)."""
        if not positions:
            return

        now = datetime.now()
        max_hold = timedelta(hours=self.config.MAX_BARS_IN_TRADE)  # 48 H1 bars = 48h

        for pos in positions:
            hold_time = now - pos['time']

            # Time exit: close if held too long
            if hold_time > max_hold:
                logger.warning("  ⏰ TIME EXIT: ticket=%d held %.1fh (max=%dh), closing",
                                pos['ticket'], hold_time.total_seconds() / 3600,
                                self.config.MAX_BARS_IN_TRADE)
                if not self.dry_run:
                    self.connector.close_position(pos['ticket'])
                else:
                    logger.info("  🔸 DRY RUN: Would close ticket=%d (time exit)", pos['ticket'])

    def _should_pyramid(self, same_dir_positions: list, bars, direction: str) -> bool:
        """Check if we should add a pyramid layer."""
        if not self.config.PYRAMIDING_ENABLED:
            return False

        if len(same_dir_positions) > self.config.PYRAMID_MAX_LAYERS:
            logger.info("  Max pyramid layers reached (%d)", len(same_dir_positions))
            return False

        # Check if existing position is profitable enough
        atr = self._calculate_atr(bars)
        min_profit_points = self.config.PYRAMID_MIN_PROFIT_ATR * atr

        for pos in same_dir_positions:
            current_price = float(bars.iloc[-2]['close'])
            if pos['type'] == 'long':
                unrealized = current_price - pos['price_open']
            else:
                unrealized = pos['price_open'] - current_price

            if unrealized < min_profit_points:
                logger.info("  Pyramid skip: position not profitable enough (%.1f < %.1f ATR)",
                             unrealized / atr if atr > 0 else 0, self.config.PYRAMID_MIN_PROFIT_ATR)
                return False

        logger.info("  ✅ Pyramid condition met: existing positions in profit")
        return True

    def _get_ml_signal(self, bars: pd.DataFrame) -> Tuple[Optional[str], float]:
        """Compute ML features from bars and get prediction."""
        if not self.predictor:
            return None, 0.0

        try:
            # Build features from live bars
            features = self._build_live_features(bars)
            if features is None or features.empty:
                return None, 0.0

            direction, prob = self.predictor.predict_direction(features)
            return direction, prob

        except Exception as e:
            logger.warning("ML signal error: %s", e)
            return None, 0.0

    def _build_live_features(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Build ML feature row from live H1 bars + multi-TF, strictly matching model features."""
        try:
            import MetaTrader5 as mt5
            from ml.feature_engine import build_h1_features, add_higher_tf_features, add_m15_features

            df = bars.copy()
            df = build_h1_features(df)

            if df.empty:
                return None

            # Fetch higher timeframe bars from MT5
            h4_bars = self.connector.get_bars(mt5.TIMEFRAME_H4, 100)
            d1_bars = self.connector.get_bars(mt5.TIMEFRAME_D1, 100)
            m15_bars = self.connector.get_bars(mt5.TIMEFRAME_M15, 500)

            # Add multi-TF features
            if h4_bars is not None or d1_bars is not None:
                df = add_higher_tf_features(df, h4_df=h4_bars, d1_df=d1_bars)
            if m15_bars is not None:
                df = add_m15_features(df, m15_bars)

            # Take last row
            last_row = df.iloc[[-1]].copy()
            last_row = last_row.replace([np.inf, -np.inf], 0).fillna(0)

            # Get the exact feature list from the saved model
            for direction in ['long', 'short']:
                selected = self.predictor.models[direction].get('features', [])
                if selected:
                    result = pd.DataFrame(0.0, index=last_row.index, columns=selected)
                    matched = 0
                    for col in selected:
                        if col in last_row.columns:
                            result[col] = last_row[col].values
                            matched += 1
                    
                    logger.info("  Features: %d/%d matched", matched, len(selected))
                    return result

            logger.warning("No saved feature list found in model")
            return None

        except Exception as e:
            logger.warning("Feature build error: %s", e)
            return None

    def _calculate_atr(self, bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR from bars."""
        df = bars.tail(period + 1).copy()
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        return float(df['tr'].dropna().mean())

    def _log_signal(self, direction, probability, price, sl, tp, volume, status):
        """Log signal to CSV."""
        log_file = 'live_signals.csv'
        header = not os.path.exists(log_file)
        with open(log_file, 'a') as f:
            if header:
                f.write("timestamp,direction,probability,price,sl,tp,volume,status\n")
            f.write(f"{datetime.now().isoformat()},{direction},{probability:.4f},"
                    f"{price:.2f},{sl:.2f},{tp:.2f},{volume:.2f},{status}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TradeBot MT5 Live Trader")
    parser.add_argument('--dry-run', action='store_true',
                        help='Log signals without placing real orders')
    parser.add_argument('--login', type=int, default=MT5_LOGIN,
                        help='MT5 login number')
    parser.add_argument('--password', type=str, default=MT5_PASSWORD,
                        help='MT5 password')
    parser.add_argument('--server', type=str, default=MT5_SERVER,
                        help='MT5 server')
    args = parser.parse_args()

    print("═" * 60)
    print("  TradeBot ML Live Trader v1.0")
    print("═" * 60)

    # Connect to MT5
    connector = MT5Connector(
        login=args.login or None,
        password=args.password or None,
        server=args.server or None,
    )

    if not connector.connect():
        print("❌ Failed to connect to MT5. Make sure MT5 is running.")
        sys.exit(1)

    symbol = connector.find_symbol()
    if not symbol:
        print("❌ Could not find trading symbol.")
        connector.disconnect()
        sys.exit(1)

    # Start trading
    trader = LiveTrader(connector, dry_run=args.dry_run)
    trader.run()


if __name__ == '__main__':
    main()
