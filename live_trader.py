# live_trader.py
"""
MT5 Multi-Timeframe Live Trader v2.0
Real-time continuous analysis across H1/M15/M5 timeframes.

Architecture:
  H1  → Strategic direction (ML model, 32 features)
  M15 → Tactical entry/exit (RSI, momentum confirmation)
  M5  → Real-time position management (smart exits)

Usage:
    python live_trader.py              # Live trading
    python live_trader.py --dry-run    # Simulation mode
"""

import os
import sys
import time
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
MT5_LOGIN = 0
MT5_PASSWORD = ""
MT5_SERVER = ""
MT5_PATH = None

SYMBOL = "USATECHIDXUSD"
VOLUME = 0.10
LOOKBACK_BARS = 200

SYMBOL_ALTERNATIVES = [
    "US100Cash", "US100", "USATECHIDXUSD", "USTEC", "NAS100",
    "NASDAQ", "Nasdaq", "US100.cash", "#NAS100", "NAS100.cash",
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
        rates = self.mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    def place_order(self, direction: str, volume: float, sl: float, tp: float) -> bool:
        symbol_info = self.mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False

        if direction == 'long':
            order_type = self.mt5.ORDER_TYPE_BUY
            price = self.mt5.symbol_info_tick(self.symbol).ask
        else:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(self.symbol).bid

        digits = symbol_info.digits
        price = round(price, digits)
        sl = round(sl, digits)
        tp = round(tp, digits)

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
            "sl": sl, "tp": tp,
            "deviation": 20,
            "magic": 20260312,
            "comment": f"TradeBot {direction.upper()}",
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

        logger.info("✅ ORDER: %s %.2f lots @ %.2f | SL=%.2f TP=%.2f | ticket=%d",
                     direction.upper(), volume, price, sl, tp, result.order)
        return True

    def close_position(self, ticket: int) -> bool:
        pos = self.mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        pos = pos[0]

        if pos.type == 0:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(self.symbol).bid
        else:
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
            logger.info("✅ CLOSED: ticket=%d profit=%.2f", ticket, pos.profit)
            return True
        logger.error("Close failed: %s", result.comment if result else "None")
        return False

    def disconnect(self):
        self.mt5.shutdown()
        self.connected = False


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME LIVE TRADER
# ═══════════════════════════════════════════════════════════════════════════════

class LiveTrader:
    """
    Multi-TF Real-Time Trading Engine v3.0

    H1  → Strategic direction (ML ensemble)
    M15 → Tactical entry/exit (ML model confirmation)
    M5  → Scalp overlay + dynamic position management

    Features:
    - Smart kademe (DCA when very confident)
    - Kademeli çıkış (FIFO, oldest exits first)
    - M5 scalp overlay (mini trades, even opposite)
    - Dynamic exits (no SL/TP dependency)
    - Magic number isolation (ignores manual trades)
    """
    BOT_MAGIC = 20260312

    def __init__(self, connector: MT5Connector, dry_run: bool = False):
        self.connector = connector
        self.dry_run = dry_run
        self.trade_count = 0
        self.current_day = None

        # Track last bar times per timeframe
        self.last_bar_times = {}

        # H1 strategic state
        self.h1_direction = None
        self.h1_probability = 0.0
        self.h1_signal_time = None

        # Load ML predictor
        self.predictor = None
        try:
            from ml.predictor import get_predictor
            self.predictor = get_predictor()
            if self.predictor and self.predictor.loaded:
                logger.info("ML Predictor: ✅")
            else:
                self.predictor = None
        except Exception as e:
            logger.error("ML Predictor error: %s", e)

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
            logger.warning("Calendar unavailable: %s", e)

        # Telegram notifications
        self.telegram = None
        try:
            from telegram_bot import get_telegram
            self.telegram = get_telegram()
        except Exception as e:
            logger.warning("Telegram unavailable: %s", e)

    def run(self):
        """Continuous multi-TF analysis every 30 seconds."""
        import MetaTrader5 as mt5
        self.mt5 = mt5

        logger.info("═══ Multi-TF Live Trader v2.0 ═══")
        logger.info("  Symbol: %s", self.connector.symbol)
        logger.info("  Mode: %s", "DRY RUN" if self.dry_run else "🔴 LIVE")
        logger.info("  ML: %s", "Active" if self.predictor else "Disabled")
        logger.info("  Timeframes: H1 (strategy) + M15 (entry) + M5 (management)")
        logger.info("  Cycle: every 30 seconds")

        if self.telegram:
            self.telegram.set_trader(self)
            self.telegram.notify_startup(
                self.connector.symbol,
                "DRY RUN" if self.dry_run else "LIVE",
                bool(self.predictor)
            )
            self.telegram.start_polling()

        try:
            while True:
                try:
                    self._run_cycle()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error("Cycle error: %s", e, exc_info=True)
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.connector.disconnect()
            logger.info("═══ Trader Stopped ═══")

    def _run_cycle(self):
        """One analysis cycle across all timeframes."""
        mt5 = self.mt5
        now = datetime.now()

        # Daily reset
        today = now.date()
        if self.current_day != today:
            # Daily summary before reset
            if self.current_day is not None and self.telegram:
                account = self.mt5.account_info()
                bal = account.balance if account else 0
                self.telegram.notify_daily_summary(
                    self.trade_count, 0, bal, self._get_all_positions()
                )
            self.current_day = today
            self.trade_count = 0
            logger.info("═══ New day: %s ═══", today)

        # Fetch all timeframe data
        h1_bars = self.connector.get_bars(mt5.TIMEFRAME_H1, LOOKBACK_BARS)
        m15_bars = self.connector.get_bars(mt5.TIMEFRAME_M15, 100)
        m5_bars = self.connector.get_bars(mt5.TIMEFRAME_M5, 60)

        if h1_bars is None or len(h1_bars) < 50:
            return

        positions = self._get_all_positions()

        # 1. Position management (every cycle, M5-based)
        self._manage_positions(positions, h1_bars, m5_bars, now)

        # 2. H1 strategic analysis (on new H1 bar)
        h1_time = h1_bars.iloc[-2]['datetime']
        if self._is_new_bar('H1', h1_time):
            self._analyze_h1(h1_bars, now)

        # 3. M15 tactical analysis (on new M15 bar)
        if m15_bars is not None and len(m15_bars) > 20:
            m15_time = m15_bars.iloc[-2]['datetime']
            if self._is_new_bar('M15', m15_time):
                # Refresh positions (may have changed during management)
                positions = self._get_all_positions()
                self._analyze_m15(m15_bars, h1_bars, positions, now)

    def _is_new_bar(self, tf: str, bar_time) -> bool:
        last = self.last_bar_times.get(tf)
        if last is not None and bar_time <= last:
            return False
        self.last_bar_times[tf] = bar_time
        return True

    # ═══════════════════════════════════════════════════════════════════
    # H1 STRATEGIC ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    def _analyze_h1(self, h1_bars, now):
        """H1 bar closed → full ensemble analysis, set strategic direction."""
        bar = h1_bars.iloc[-2]
        logger.info("📊 H1 BAR: %s | C=%.2f H=%.2f L=%.2f",
                     bar['datetime'], bar['close'], bar['high'], bar['low'])

        # Try ensemble first (H1+M15+M5)
        direction, probability, details = self._get_ensemble_signal(h1_bars)

        if direction:
            self.h1_direction = direction
            self.h1_probability = probability
            self.h1_signal_time = now

            # Log agreement
            agree_count = sum(1 for k, v in details.items()
                            if k.endswith(f'_{direction}') and not k.startswith('ensemble') and v > 0.5)
            logger.info("  🎯 ENSEMBLE: %s (prob=%.3f, %d/3 agree)",
                         direction.upper(), probability, agree_count)
            for k, v in sorted(details.items()):
                logger.info("    %s: %.3f", k, v)
        else:
            self.h1_direction = None
            self.h1_probability = 0.0
            logger.info("  Ensemble: No clear direction")

    # ═══════════════════════════════════════════════════════════════════
    # M15 TACTICAL ENTRY/EXIT
    # ═══════════════════════════════════════════════════════════════════

    def _analyze_m15(self, m15_bars, h1_bars, positions, now):
        """M15 bar closed → entry/exit timing based on H1 strategy."""
        bar = m15_bars.iloc[-2]
        close = float(bar['close'])

        # M15 indicators
        m15_rsi = self._calc_rsi(m15_bars['close'], 14)
        m15_mom = (close / float(m15_bars.iloc[-6]['close']) - 1) * 100 if len(m15_bars) > 6 else 0

        logger.info("  M15: C=%.2f RSI=%.1f Mom=%.2f%%", close, m15_rsi, m15_mom)

        # News check
        news_scale = 1.0
        if self.news_mgr:
            score, events = self.news_mgr.calculate_risk_score()
            if score >= 6.0:
                future = [e for e in events if (e.timestamp - now).total_seconds() > 0]
                if future:
                    logger.info("  ⏳ News wait: %s in %.1fh",
                                 future[0].name,
                                 (future[0].timestamp - now).total_seconds() / 3600)
                    return
                else:
                    logger.info("  🔥 Post-news momentum")
            elif score >= 3.0:
                news_scale = 0.5

        # M15 model signal (replaces simple RSI heuristic)
        m15_direction = None
        m15_prob = 0.0
        if self.predictor and 'm15' in self.predictor.timeframes:
            m15_feats = self._build_tf_features(m15_bars, 'm15')
            if m15_feats is not None:
                m15_long = self.predictor.predict_single('m15', 'long', m15_feats)
                m15_short = self.predictor.predict_single('m15', 'short', m15_feats)
                if m15_long > 0.55:
                    m15_direction = 'long'
                    m15_prob = m15_long
                elif m15_short > 0.55:
                    m15_direction = 'short'
                    m15_prob = m15_short
                logger.info("  🧠 M15 ML: %s (L=%.3f S=%.3f)",
                             (m15_direction or 'neutral').upper(), m15_long, m15_short)

        # H1 staleness fix: if H1 is >2h old and M15 disagrees strongly
        if self.h1_signal_time and self.h1_direction:
            h1_age_hours = (now - self.h1_signal_time).total_seconds() / 3600
            if h1_age_hours > 2.0 and m15_direction and m15_direction != self.h1_direction and m15_prob > 0.60:
                logger.warning("  🔄 M15 OVERRIDE: H1=%s (%.1fh old) → M15=%s (prob=%.3f)",
                                self.h1_direction.upper(), h1_age_hours,
                                m15_direction.upper(), m15_prob)
                self.h1_direction = m15_direction
                self.h1_probability = m15_prob

        # Need direction (H1 or M15 override)
        if self.h1_direction is None:
            return

        # Max positions check
        if len(positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return

        direction = self.h1_direction
        probability = self.h1_probability

        # M15 confirmation: use model if available, fall back to RSI
        confirmed = False
        if m15_direction:
            # ML-based confirmation
            confirmed = (m15_direction == direction)
        else:
            # Fallback: RSI heuristic
            if direction == 'long':
                if 35 < m15_rsi < 75 and m15_mom > -0.1:
                    confirmed = True
            else:
                if 25 < m15_rsi < 65 and m15_mom < 0.1:
                    confirmed = True

        if not confirmed:
            logger.info("  M15 not confirming %s", direction.upper())
            return

        # Direction safety
        opposite = [p for p in positions if p['type'] != direction]
        if opposite:
            return

        # Pyramid check
        same = [p for p in positions if p['type'] == direction]
        is_pyramid = len(same) > 0
        if is_pyramid and not self._should_pyramid(same, h1_bars, direction, probability):
            return

        # Execute
        self._execute_trade(direction, probability, close, h1_bars,
                            is_pyramid, len(same), news_scale)

    def _execute_trade(self, direction, probability, price, h1_bars,
                       is_pyramid, layer_count, news_scale):
        """Build SL/TP and place order."""
        atr = self._calculate_atr(h1_bars)
        if atr <= 0:
            return

        if direction == 'long':
            sl = price - self.config.STOP_LOSS_ATR_MULT * atr
            tp = price + self.config.TAKE_PROFIT_ATR_MULT * atr
        else:
            sl = price + self.config.STOP_LOSS_ATR_MULT * atr
            tp = price - self.config.TAKE_PROFIT_ATR_MULT * atr

        # Position sizing
        volume = VOLUME
        if self.predictor and self.config.ML_CONFIDENCE_SIZING:
            tp_sl = self.config.TAKE_PROFIT_ATR_MULT / self.config.STOP_LOSS_ATR_MULT
            kelly = self.predictor.kelly_size(probability, tp_sl)
            volume = max(round(VOLUME * max(kelly * 4, 0.5), 2), VOLUME)

        volume = max(round(volume * news_scale, 2), VOLUME)

        if is_pyramid:
            decay = self.config.PYRAMID_SIZE_DECAY ** layer_count
            volume = max(round(volume * decay, 2), VOLUME)
            logger.info("  📐 PYRAMID layer %d (%.0f%%)", layer_count + 1, decay * 100)

        label = "PYRAMID" if is_pyramid else "NEW"
        logger.info("  → %s %s: SL=%.2f TP=%.2f ATR=%.2f Vol=%.2f",
                     label, direction.upper(), sl, tp, atr, volume)

        if self.dry_run:
            logger.info("  🔸 DRY: Would %s %s %.2f @ %.2f",
                         label, direction.upper(), volume, price)
            self._log_signal(direction, probability, price, sl, tp, volume, f"DRY_{label}")
            if self.telegram:
                self.telegram.notify_order(direction, volume, price, sl, tp, 0, f"DRY_{label}")
        else:
            if self.connector.place_order(direction, volume, sl, tp):
                self.trade_count += 1
                self._log_signal(direction, probability, price, sl, tp, volume, label)
                if self.telegram:
                    # Get last ticket
                    positions = self._get_all_positions()
                    ticket = positions[-1]['ticket'] if positions else 0
                    self.telegram.notify_order(direction, volume, price, sl, tp, ticket, label)

    # ═══════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT (M5-based, real-time)
    # ═══════════════════════════════════════════════════════════════════

    def _get_all_positions(self) -> list:
        """Get only BOT positions (magic number filtered)."""
        try:
            positions = self.connector.mt5.positions_get(symbol=self.connector.symbol)
            if not positions:
                return []
            return [{
                'ticket': p.ticket,
                'type': 'long' if p.type == 0 else 'short',
                'volume': p.volume,
                'price_open': p.price_open,
                'sl': p.sl, 'tp': p.tp,
                'profit': p.profit,
                'time': datetime.fromtimestamp(p.time),
                'magic': p.magic,
                'comment': p.comment,
            } for p in positions if p.magic == self.BOT_MAGIC]
        except Exception:
            return []

    def _manage_positions(self, positions, h1_bars, m5_bars, now):
        """Smart position management v3 — dynamic exits, kademe, scalp."""
        if not positions:
            # No positions → check for scalp opportunity
            self._m5_scalp(m5_bars, h1_bars, positions, now)
            return

        max_hold = timedelta(hours=self.config.MAX_BARS_IN_TRADE)
        atr = self._calculate_atr(h1_bars)
        price = float(h1_bars.iloc[-1]['close'])

        # M5 ML predictions for dynamic management
        m5_long_prob = 0.5
        m5_short_prob = 0.5
        if m5_bars is not None and len(m5_bars) > 5:
            if self.predictor and 'm5' in self.predictor.timeframes:
                m5_feats = self._build_tf_features(m5_bars, 'm5')
                if m5_feats is not None:
                    m5_long_prob = self.predictor.predict_single('m5', 'long', m5_feats)
                    m5_short_prob = self.predictor.predict_single('m5', 'short', m5_feats)

        # Sort by open time (oldest first for FIFO exit)
        positions_sorted = sorted(positions, key=lambda p: p['time'])

        for i, pos in enumerate(positions_sorted):
            hold_h = (now - pos['time']).total_seconds() / 3600
            pnl_pts = (price - pos['price_open']) if pos['type'] == 'long' \
                else (pos['price_open'] - price)
            pnl_atr = pnl_pts / atr if atr > 0 else 0
            is_scalp = 'SCALP' in (pos.get('comment', '') or '')

            # ══ Scalp positions: tight management ══
            if is_scalp:
                if hold_h > 2.0:
                    logger.info("  ⏰ SCALP TIME: ticket=%d held %.1fh", pos['ticket'], hold_h)
                    self._close_position(pos, 'SCALP_TIME')
                    continue
                if pnl_atr >= 0.8:
                    logger.info("  💰 SCALP PROFIT: ticket=%d +%.1f ATR", pos['ticket'], pnl_atr)
                    self._close_position(pos, 'SCALP_PROFIT')
                    continue
                if pnl_atr < -0.5:
                    logger.info("  ✂️ SCALP CUT: ticket=%d %.1f ATR", pos['ticket'], pnl_atr)
                    self._close_position(pos, 'SCALP_CUT')
                    continue
                continue  # Don't apply normal rules to scalps

            # ══ Normal positions: full management ══

            # 1. Time exit (48h max)
            if (now - pos['time']) > max_hold:
                logger.warning("  ⏰ TIME EXIT: ticket=%d held %.1fh", pos['ticket'], hold_h)
                self._close_position(pos, 'TIME')
                continue

            # 2. DYNAMIC PROFIT-TAKE (M5 ML model based)
            if pnl_atr >= 1.5:
                m5_against = (pos['type'] == 'long' and m5_short_prob > 0.60) or \
                             (pos['type'] == 'short' and m5_long_prob > 0.60)
                if m5_against:
                    logger.info("  💰 DYNAMIC EXIT: ticket=%d +%.1f ATR (M5 reverse prob=%.2f)",
                                 pos['ticket'], pnl_atr,
                                 m5_short_prob if pos['type'] == 'long' else m5_long_prob)
                    self._close_position(pos, 'DYNAMIC_PROFIT')
                    continue

            # 3. KADEMELI ÇIKIŞ: oldest position exits at lower target
            if pnl_atr >= 1.0 and len(positions_sorted) > 1:
                normal_positions = [p for p in positions_sorted
                                   if 'SCALP' not in (p.get('comment', '') or '')]
                if len(normal_positions) > 1 and i == 0:  # Oldest normal position
                    logger.info("  🎯 FIFO EXIT: oldest ticket=%d +%.1f ATR",
                                 pos['ticket'], pnl_atr)
                    self._close_position(pos, 'FIFO_EXIT')
                    continue

            # 4. TRAILING STOP
            if pnl_atr >= 1.0:
                if pos['type'] == 'long':
                    new_sl = price - 0.7 * atr
                    if new_sl > pos['sl']:
                        self._modify_sl(pos, new_sl)
                else:
                    new_sl = price + 0.7 * atr
                    if new_sl < pos['sl'] or pos['sl'] == 0:
                        self._modify_sl(pos, new_sl)

            # 5. BREAKEVEN: move SL to entry when profitable
            if pnl_atr >= 0.8:
                if pos['type'] == 'long' and pos['sl'] < pos['price_open']:
                    self._modify_sl(pos, pos['price_open'] + 0.1 * atr)
                    logger.info("  🛡️ BREAKEVEN: ticket=%d", pos['ticket'])
                elif pos['type'] == 'short' and (pos['sl'] > pos['price_open'] or pos['sl'] == 0):
                    self._modify_sl(pos, pos['price_open'] - 0.1 * atr)
                    logger.info("  🛡️ BREAKEVEN: ticket=%d", pos['ticket'])

            # 6. DYNAMIC TP EXTEND
            if pnl_atr >= 2.0 and self.h1_direction == pos['type'] and self.h1_probability > 0.60:
                new_tp = price + 1.5 * atr if pos['type'] == 'long' else price - 1.5 * atr
                if pos['type'] == 'long' and new_tp > pos['tp']:
                    self._modify_tp(pos, new_tp)
                elif pos['type'] == 'short' and new_tp < pos['tp']:
                    self._modify_tp(pos, new_tp)

            # 7. LOSING MANAGEMENT
            if pnl_atr < -0.3:
                # Ensemble yön değiştirdi → çık
                if self.h1_direction and self.h1_direction != pos['type']:
                    logger.warning("  🔄 SMART EXIT: ticket=%d %s->ML=%s (loss=%.1f ATR)",
                                    pos['ticket'], pos['type'].upper(),
                                    self.h1_direction.upper(), pnl_atr)
                    if self.telegram:
                        self.telegram.notify_smart_exit(
                            pos['ticket'], pos['type'], self.h1_direction, pnl_atr)
                    self._close_position(pos, 'SMART_EXIT')
                    continue

                # M5 strongly against + moderate loss
                m5_against = (pos['type'] == 'long' and m5_short_prob > 0.65) or \
                             (pos['type'] == 'short' and m5_long_prob > 0.65)
                if m5_against and pnl_atr < -0.8:
                    logger.warning("  📉 M5 EXIT: ticket=%d M5 reverse, loss=%.1f ATR",
                                    pos['ticket'], pnl_atr)
                    self._close_position(pos, 'M5_EXIT')
                    continue

                # Deep loss + no direction
                if pnl_atr < -1.5 and self.h1_direction is None:
                    logger.warning("  📉 DEEP LOSS: ticket=%d loss=%.1f ATR",
                                    pos['ticket'], pnl_atr)
                    self._close_position(pos, 'DEEP_LOSS')
                    continue

                # ML still agrees → hold
                if self.h1_direction == pos['type']:
                    logger.info("  🧠 HOLD: ticket=%d ML still %s (loss=%.1f ATR)",
                                 pos['ticket'], pos['type'].upper(), pnl_atr)

        # Smart kademe check
        self._smart_kademe(positions, h1_bars, m5_bars, price, atr, now)

        # M5 scalp overlay
        self._m5_scalp(m5_bars, h1_bars, positions, now)

    def _smart_kademe(self, positions, h1_bars, m5_bars, price, atr, now):
        """Analiz eminse zararda da kademe al. Max 3 layer."""
        if not self.h1_direction or not self.predictor:
            return

        direction = self.h1_direction
        same = [p for p in positions if p['type'] == direction
                and 'SCALP' not in (p.get('comment', '') or '')]

        if len(same) >= 3 or len(same) == 0:
            return

        # Cooldown: 30 min since last
        newest = max(same, key=lambda p: p['time'])
        if (now - newest['time']).total_seconds() < 1800:
            return

        # Need price dip of 0.5 ATR from average entry
        avg_entry = sum(p['price_open'] for p in same) / len(same)
        if direction == 'long' and price > avg_entry - 0.5 * atr:
            return
        if direction == 'short' and price < avg_entry + 0.5 * atr:
            return

        # Must be VERY confident (prob > 0.60)
        if self.h1_probability < 0.60:
            return

        # M15 must also agree
        m15_agrees = False
        if 'm15' in self.predictor.timeframes:
            try:
                import MetaTrader5 as mt5
                m15_bars = self.connector.get_bars(mt5.TIMEFRAME_M15, 500)
                if m15_bars is not None:
                    m15_feats = self._build_tf_features(m15_bars, 'm15')
                    if m15_feats is not None:
                        prob = self.predictor.predict_single('m15', direction, m15_feats)
                        m15_agrees = prob > 0.55
            except Exception:
                pass

        if not m15_agrees:
            return

        layer = len(same)
        volume = max(round(VOLUME * (0.8 ** layer), 2), VOLUME)
        logger.info("  🟢 KADEME %d: %s %.2f lots (prob=%.3f, dip=%.1f ATR)",
                     layer + 1, direction.upper(), volume,
                     self.h1_probability, abs(price - avg_entry) / atr)

        self._execute_trade(direction, self.h1_probability, price, h1_bars,
                            True, layer, 1.0)

    def _m5_scalp(self, m5_bars, h1_bars, positions, now):
        """M5 model based scalp: short-term mini trades."""
        if m5_bars is None or len(m5_bars) < 20:
            return
        if not self.predictor or 'm5' not in self.predictor.timeframes:
            return

        # Max 1 active scalp
        scalps = [p for p in positions if 'SCALP' in (p.get('comment', '') or '')]
        if len(scalps) >= 1:
            return

        if len(positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return

        m5_feats = self._build_tf_features(m5_bars, 'm5')
        if m5_feats is None:
            return

        m5_long = self.predictor.predict_single('m5', 'long', m5_feats)
        m5_short = self.predictor.predict_single('m5', 'short', m5_feats)

        scalp_dir = None
        scalp_prob = 0.0
        if m5_long > 0.65:
            scalp_dir = 'long'
            scalp_prob = m5_long
        elif m5_short > 0.65:
            scalp_dir = 'short'
            scalp_prob = m5_short

        if not scalp_dir:
            return

        price = float(m5_bars.iloc[-1]['close'])
        atr = self._calculate_atr(h1_bars)

        if scalp_dir == 'long':
            sl = price - 0.75 * atr
            tp = price + 1.0 * atr
        else:
            sl = price + 0.75 * atr
            tp = price - 1.0 * atr

        volume = max(round(VOLUME * 0.5, 2), VOLUME)

        logger.info("  ⚡ SCALP: %s %.2f lots (M5=%.3f) SL=%.2f TP=%.2f",
                     scalp_dir.upper(), volume, scalp_prob, sl, tp)

        if self.dry_run:
            logger.info("  🔸 DRY: Would SCALP %s %.2f @ %.2f",
                         scalp_dir.upper(), volume, price)
            self._log_signal(scalp_dir, scalp_prob, price, sl, tp, volume, 'DRY_SCALP')
        else:
            import MetaTrader5 as mt5
            order_type = mt5.ORDER_TYPE_BUY if scalp_dir == 'long' else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(self.connector.symbol)
            fill_price = tick.ask if scalp_dir == 'long' else tick.bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.connector.symbol,
                "volume": volume,
                "type": order_type,
                "price": fill_price,
                "sl": sl, "tp": tp,
                "deviation": 20,
                "magic": self.BOT_MAGIC,
                "comment": f"TradeBot SCALP {scalp_dir.upper()}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("  ✅ SCALP: %s %.2f @ %.2f ticket=%d",
                             scalp_dir.upper(), volume, fill_price, result.order)
                if self.telegram:
                    self.telegram.notify_order(scalp_dir, volume, fill_price, sl, tp,
                                               result.order, 'SCALP')






    def _modify_sl(self, pos, new_sl):
        """Move SL dynamically."""
        if self.dry_run:
            logger.info("  🔸 DRY: Trail SL to %.2f for ticket=%d", new_sl, pos['ticket'])
            return
        try:
            import MetaTrader5 as mt5
            request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'symbol': self.connector.symbol,
                'position': pos['ticket'],
                'sl': round(new_sl, 2),
                'tp': pos['tp'],
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("  📐 SL updated: ticket=%d SL=%.2f", pos['ticket'], new_sl)
                if self.telegram:
                    self.telegram._send(f"📐 SL: {pos['ticket']} -> {new_sl:.2f}")
        except Exception as e:
            logger.warning("  SL modify error: %s", e)

    def _modify_tp(self, pos, new_tp):
        """Extend TP when trend is strong."""
        if self.dry_run:
            logger.info("  🔸 DRY: Move TP to %.2f for ticket=%d", new_tp, pos['ticket'])
            return
        try:
            import MetaTrader5 as mt5
            request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'symbol': self.connector.symbol,
                'position': pos['ticket'],
                'sl': pos['sl'],
                'tp': round(new_tp, 2),
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("  🎯 TP extended: ticket=%d TP=%.2f", pos['ticket'], new_tp)
                if self.telegram:
                    self.telegram._send(f"🎯 TP: {pos['ticket']} -> {new_tp:.2f}")
        except Exception as e:
            logger.warning("  TP modify error: %s", e)

    def _close_position(self, pos, reason='MANUAL'):
        """Close position with Telegram notification."""
        if self.dry_run:
            logger.info("  🔸 DRY: Close ticket=%d (P&L=$%.2f) [%s]",
                         pos['ticket'], pos['profit'], reason)
        else:
            self.connector.close_position(pos['ticket'])
            if self.telegram:
                self.telegram.notify_close(
                    pos['type'], pos['ticket'], pos['profit'], reason)

    def _should_pyramid(self, same_pos, bars, direction, probability=0):
        """Placeholder — kademe logic moved to _smart_kademe."""
        if not self.config.PYRAMIDING_ENABLED:
            return False
        if len(same_pos) >= 3:
            return False
        if probability < 0.60:
            return False
        return True

    # ═══════════════════════════════════════════════════════════════════
    # ML + FEATURE ENGINE (ENSEMBLE)
    # ═══════════════════════════════════════════════════════════════════

    def _get_ensemble_signal(self, h1_bars):
        """Get ensemble signal from H1+M15+M5 models."""
        if not self.predictor:
            return None, 0.0, {}
        try:
            features_dict = self._build_all_tf_features(h1_bars)
            if not features_dict:
                return None, 0.0, {}
            return self.predictor.predict_ensemble(features_dict)
        except Exception as e:
            logger.warning("Ensemble signal error: %s", e)
            return None, 0.0, {}

    def _get_ml_signal(self, bars: pd.DataFrame) -> Tuple[Optional[str], float]:
        """Backward-compatible H1-only signal."""
        if not self.predictor:
            return None, 0.0
        try:
            features = self._build_tf_features(bars, 'h1')
            if features is None or features.empty:
                return None, 0.0
            return self.predictor.predict_direction(features)
        except Exception as e:
            logger.warning("ML signal error: %s", e)
            return None, 0.0

    def _build_all_tf_features(self, h1_bars) -> dict:
        """Build features for all available timeframes."""
        features_dict = {}

        # H1 features
        h1_feats = self._build_tf_features(h1_bars, 'h1')
        if h1_feats is not None:
            features_dict['h1'] = h1_feats

        # M15 + M5: fetch from MT5
        try:
            import MetaTrader5 as mt5
            m15_bars = self.connector.get_bars(mt5.TIMEFRAME_M15, 500)
            if m15_bars is not None:
                m15_feats = self._build_tf_features(m15_bars, 'm15')
                if m15_feats is not None:
                    features_dict['m15'] = m15_feats

            m5_bars = self.connector.get_bars(mt5.TIMEFRAME_M5, 500)
            if m5_bars is not None:
                m5_feats = self._build_tf_features(m5_bars, 'm5')
                if m5_feats is not None:
                    features_dict['m5'] = m5_feats
        except Exception as e:
            logger.warning("Multi-TF feature error: %s", e)

        return features_dict

    def _build_tf_features(self, bars, timeframe: str) -> Optional[pd.DataFrame]:
        """Build features for a specific timeframe and align to model's expected features."""
        try:
            df = bars.copy()

            if timeframe == 'h1':
                from ml.feature_engine import build_h1_features, add_higher_tf_features, add_m15_features
                df = build_h1_features(df)
                try:
                    import MetaTrader5 as mt5
                    h4_bars = self.connector.get_bars(mt5.TIMEFRAME_H4, 100)
                    d1_bars = self.connector.get_bars(mt5.TIMEFRAME_D1, 100)
                    m15_bars = self.connector.get_bars(mt5.TIMEFRAME_M15, 500)
                    if h4_bars is not None or d1_bars is not None:
                        df = add_higher_tf_features(df, h4_df=h4_bars, d1_df=d1_bars)
                    if m15_bars is not None:
                        df = add_m15_features(df, m15_bars)
                except Exception:
                    pass

            elif timeframe == 'm15':
                from ml.feature_engine import build_m15_features, add_m15_higher_tf
                df = build_m15_features(df)
                try:
                    import MetaTrader5 as mt5
                    h1_bars = self.connector.get_bars(mt5.TIMEFRAME_H1, 200)
                    h4_bars = self.connector.get_bars(mt5.TIMEFRAME_H4, 100)
                    d1_bars = self.connector.get_bars(mt5.TIMEFRAME_D1, 100)
                    df = add_m15_higher_tf(df, h1_df=h1_bars, h4_df=h4_bars, d1_df=d1_bars)
                except Exception:
                    pass

            elif timeframe == 'm5':
                from ml.feature_engine import build_m5_features, add_m5_higher_tf
                df = build_m5_features(df)
                try:
                    import MetaTrader5 as mt5
                    h1_bars = self.connector.get_bars(mt5.TIMEFRAME_H1, 200)
                    m15_bars = self.connector.get_bars(mt5.TIMEFRAME_M15, 500)
                    df = add_m5_higher_tf(df, h1_df=h1_bars, m15_df=m15_bars)
                except Exception:
                    pass

            if df.empty:
                return None

            last_row = df.iloc[[-1]].copy()
            last_row = last_row.replace([np.inf, -np.inf], 0).fillna(0)

            # Align to model features
            tf_models = self.predictor.timeframes.get(timeframe, {})
            for direction in ['long', 'short']:
                bundle = tf_models.get(direction)
                if bundle and bundle.get('features'):
                    selected = bundle['features']
                    result = pd.DataFrame(0.0, index=last_row.index, columns=selected)
                    matched = 0
                    for col in selected:
                        if col in last_row.columns:
                            result[col] = last_row[col].values
                            matched += 1
                    return result

            return last_row

        except Exception as e:
            logger.warning("Feature build error (%s): %s", timeframe, e)
            return None

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _calculate_atr(self, bars, period=14):
        df = bars.tail(period + 1).copy()
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        return float(df['tr'].dropna().mean())

    def _calc_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    def _log_signal(self, direction, probability, price, sl, tp, volume, status):
        log_file = 'live_signals.csv'
        header = not os.path.exists(log_file)
        with open(log_file, 'a') as f:
            if header:
                f.write("timestamp,direction,probability,price,sl,tp,volume,status\n")
            f.write(f"{datetime.now().isoformat()},{direction},{probability:.4f},"
                    f"{price:.2f},{sl:.2f},{tp:.2f},{volume:.2f},{status}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TradeBot Multi-TF Live Trader v2.0")
    parser.add_argument('--dry-run', action='store_true',
                        help='Log signals without placing real orders')
    parser.add_argument('--login', type=int, default=MT5_LOGIN)
    parser.add_argument('--password', type=str, default=MT5_PASSWORD)
    parser.add_argument('--server', type=str, default=MT5_SERVER)
    args = parser.parse_args()

    print("═" * 60)
    print("  TradeBot Multi-TF Live Trader v2.0")
    print("═" * 60)

    connector = MT5Connector(
        login=args.login or None,
        password=args.password or None,
        server=args.server or None,
    )

    if not connector.connect():
        print("❌ Failed to connect to MT5.")
        sys.exit(1)

    symbol = connector.find_symbol()
    if not symbol:
        print("❌ Could not find trading symbol.")
        connector.disconnect()
        sys.exit(1)

    trader = LiveTrader(connector, dry_run=args.dry_run)
    trader.run()


if __name__ == '__main__':
    main()
