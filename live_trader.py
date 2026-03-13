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
    Multi-TF Real-Time Trading Engine v2.0

    H1  → Strategic direction (ML model)
    M15 → Tactical entry/exit (RSI + momentum confirmation)
    M5  → Position management (real-time smart exits)

    Checks every 30 seconds. No daily trade limit.
    """

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

        # Anti-repeat cooldown: don't reopen same direction after SL
        self.last_close_time = {}    # {'long': datetime, 'short': datetime}
        self.last_close_reason = {}  # {'long': 'SL', 'short': 'TP'}
        self.COOLDOWN_MINUTES = 30   # Wait 30 min after SL before same direction

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
        """H1 bar closed → full ML analysis, set strategic direction."""
        bar = h1_bars.iloc[-2]
        logger.info("📊 H1 BAR: %s | C=%.2f H=%.2f L=%.2f",
                     bar['datetime'], bar['close'], bar['high'], bar['low'])

        direction, probability = self._get_ml_signal(h1_bars)

        if direction:
            self.h1_direction = direction
            self.h1_probability = probability
            self.h1_signal_time = now
            logger.info("  🎯 H1 DIRECTION: %s (prob=%.3f)", direction.upper(), probability)
        else:
            self.h1_direction = None
            self.h1_probability = 0.0
            logger.info("  H1: No clear direction")

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

        # Need H1 direction
        if self.h1_direction is None:
            return

        # Max positions check
        if len(positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return

        direction = self.h1_direction
        probability = self.h1_probability

        # Anti-repeat cooldown check
        if direction in self.last_close_time:
            elapsed = (now - self.last_close_time[direction]).total_seconds() / 60
            reason = self.last_close_reason.get(direction, '')
            if elapsed < self.COOLDOWN_MINUTES and reason == 'SL':
                logger.info("  ⏳ Cooldown: %s SL'd %.0f min ago (wait %d min)",
                             direction.upper(), elapsed, self.COOLDOWN_MINUTES)
                return

        # M15 confirmation
        confirmed = False
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
            } for p in positions]
        except Exception:
            return []

    def _manage_positions(self, positions, h1_bars, m5_bars, now):
        """Active position management — trailing stop, profit-taking, smart exits."""
        if not positions:
            # Check if any positions were closed by SL/TP (detect externally)
            self._detect_sl_tp_closes(now)
            return

        max_hold = timedelta(hours=self.config.MAX_BARS_IN_TRADE)
        atr = self._calculate_atr(h1_bars)
        price = float(h1_bars.iloc[-1]['close'])

        # M5 momentum + RSI
        m5_mom = 0
        m5_rsi = 50
        if m5_bars is not None and len(m5_bars) > 5:
            m5_mom = (float(m5_bars.iloc[-1]['close']) / float(m5_bars.iloc[-4]['close']) - 1) * 100
            m5_rsi = self._calc_rsi(m5_bars['close'], 14)

        for pos in positions:
            hold_h = (now - pos['time']).total_seconds() / 3600
            pnl_pts = (price - pos['price_open']) if pos['type'] == 'long' \
                else (pos['price_open'] - price)
            pnl_atr = pnl_pts / atr if atr > 0 else 0

            # ── Rule 1: Time exit (48h max) ──
            if (now - pos['time']) > max_hold:
                logger.warning("  ⏰ TIME EXIT: ticket=%d held %.1fh", pos['ticket'], hold_h)
                self._close_position(pos, 'TIME')
                continue

            # ── Rule 2: TRAILING STOP — lock in profit ──
            if pnl_atr >= 1.0:
                # Move SL to breakeven + 0.3 ATR
                if pos['type'] == 'long':
                    new_sl = pos['price_open'] + 0.3 * atr
                    if pos['sl'] < new_sl:
                        self._modify_sl(pos, new_sl)
                else:
                    new_sl = pos['price_open'] - 0.3 * atr
                    if pos['sl'] > new_sl or pos['sl'] == 0:
                        self._modify_sl(pos, new_sl)

            if pnl_atr >= 2.0:
                # Move SL to lock 1.0 ATR profit
                if pos['type'] == 'long':
                    new_sl = pos['price_open'] + 1.0 * atr
                    if pos['sl'] < new_sl:
                        self._modify_sl(pos, new_sl)
                else:
                    new_sl = pos['price_open'] - 1.0 * atr
                    if pos['sl'] > new_sl or pos['sl'] == 0:
                        self._modify_sl(pos, new_sl)

            # ── Rule 3: ACTIVE PROFIT-TAKING (momentum fading) ──
            if pnl_atr >= 1.5:
                # Take profit when M5 momentum reverses
                against = (pos['type'] == 'long' and m5_mom < -0.2 and m5_rsi > 65) or \
                          (pos['type'] == 'short' and m5_mom > 0.2 and m5_rsi < 35)
                if against:
                    logger.info("  💰 PROFIT-TAKE: ticket=%d +%.1f ATR (M5 fading)",
                                 pos['ticket'], pnl_atr)
                    self._close_position(pos, 'PROFIT_TAKE')
                    continue

            # ── Rule 4: ML says opposite + losing → smart exit ──
            if pnl_atr < -0.5:
                if self.h1_direction and self.h1_direction != pos['type']:
                    logger.warning("  🔄 SMART EXIT: ticket=%d %s->ML=%s (loss=%.1f ATR)",
                                    pos['ticket'], pos['type'].upper(),
                                    self.h1_direction.upper(), pnl_atr)
                    if self.telegram:
                        self.telegram.notify_smart_exit(
                            pos['ticket'], pos['type'], self.h1_direction, pnl_atr)
                    self._close_position(pos, 'SL')
                    continue

                # M5 momentum strongly against + loss
                against = (pos['type'] == 'long' and m5_mom < -0.3) or \
                          (pos['type'] == 'short' and m5_mom > 0.3)
                if against and pnl_atr < -1.0:
                    logger.warning("  📉 M5 EXIT: ticket=%d M5=%.2f%% loss=%.1f ATR",
                                    pos['ticket'], m5_mom, pnl_atr)
                    self._close_position(pos, 'SL')
                    continue

                # Deep loss with no signal
                if pnl_atr < -1.5 and self.h1_direction is None:
                    logger.warning("  📉 DEEP LOSS EXIT: ticket=%d loss=%.1f ATR",
                                    pos['ticket'], pnl_atr)
                    self._close_position(pos, 'SL')
                    continue

    def _detect_sl_tp_closes(self, now):
        """Detect if a position was closed by broker SL/TP (not by us)."""
        # If we had positions tracked but now have 0, something closed externally
        if not hasattr(self, '_last_position_count'):
            self._last_position_count = 0
            return
        # Will be set by the cycle

    def _modify_sl(self, pos, new_sl):
        """Move SL to new level (trailing stop)."""
        if self.dry_run:
            logger.info("  🔸 DRY: Would trail SL to %.2f for ticket=%d", new_sl, pos['ticket'])
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
                logger.info("  📐 TRAIL SL: ticket=%d SL=%.2f", pos['ticket'], new_sl)
                if self.telegram:
                    self.telegram._send(
                        f"📐 TRAILING SL\n"
                        f"Ticket: {pos['ticket']}\n"
                        f"New SL: {new_sl:.2f}")
        except Exception as e:
            logger.warning("  Trail SL error: %s", e)

    def _close_position(self, pos, reason='MANUAL'):
        """Close position and track for cooldown."""
        # Record cooldown
        self.last_close_time[pos['type']] = datetime.now()
        self.last_close_reason[pos['type']] = reason

        if self.dry_run:
            logger.info("  🔸 DRY: Would close ticket=%d (P&L=$%.2f) [%s]",
                         pos['ticket'], pos['profit'], reason)
        else:
            self.connector.close_position(pos['ticket'])
            if self.telegram:
                self.telegram.notify_close(
                    pos['type'], pos['ticket'], pos['profit'], reason)

    def _should_pyramid(self, same_pos, bars, direction, probability=0):
        if not self.config.PYRAMIDING_ENABLED:
            return False
        if len(same_pos) > self.config.PYRAMID_MAX_LAYERS:
            return False

        import config
        threshold = getattr(config, 'ML_LONG_THRESHOLD', 0.22) if direction == 'long' \
            else getattr(config, 'ML_SHORT_THRESHOLD', 0.35)

        if probability < threshold + 0.10:
            logger.info("  Pyramid skip: prob=%.3f < %.3f", probability, threshold + 0.10)
            return False

        atr = self._calculate_atr(bars)
        price = float(bars.iloc[-2]['close'])
        for pos in same_pos:
            u = (price - pos['price_open']) if pos['type'] == 'long' \
                else (pos['price_open'] - price)
            if u < -1.5 * atr:
                return False

        logger.info("  ✅ Pyramid OK (prob=%.3f)", probability)
        return True

    # ═══════════════════════════════════════════════════════════════════
    # ML + FEATURE ENGINE
    # ═══════════════════════════════════════════════════════════════════

    def _get_ml_signal(self, bars: pd.DataFrame) -> Tuple[Optional[str], float]:
        if not self.predictor:
            return None, 0.0
        try:
            features = self._build_live_features(bars)
            if features is None or features.empty:
                return None, 0.0
            return self.predictor.predict_direction(features)
        except Exception as e:
            logger.warning("ML signal error: %s", e)
            return None, 0.0

    def _build_live_features(self, bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Build ML features from H1 bars + multi-TF."""
        try:
            import MetaTrader5 as mt5
            from ml.feature_engine import build_h1_features, add_higher_tf_features, add_m15_features

            df = bars.copy()
            df = build_h1_features(df)
            if df.empty:
                return None

            h4_bars = self.connector.get_bars(mt5.TIMEFRAME_H4, 100)
            d1_bars = self.connector.get_bars(mt5.TIMEFRAME_D1, 100)
            m15_bars = self.connector.get_bars(mt5.TIMEFRAME_M15, 500)

            if h4_bars is not None or d1_bars is not None:
                df = add_higher_tf_features(df, h4_df=h4_bars, d1_df=d1_bars)
            if m15_bars is not None:
                df = add_m15_features(df, m15_bars)

            last_row = df.iloc[[-1]].copy()
            last_row = last_row.replace([np.inf, -np.inf], 0).fillna(0)

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
            return None
        except Exception as e:
            logger.warning("Feature error: %s", e)
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
