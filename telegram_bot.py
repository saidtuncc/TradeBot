# telegram_bot.py
"""
Telegram Notifications + Commands for TradeBot.

Commands:
  /status    - Bot status, ML direction, positions
  /positions - Open positions detail
  /balance   - Account balance
  /trades    - Today's trade count
  /help      - Show commands
"""

import logging
import urllib.request
import urllib.parse
import ssl
import json
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger('telegram')


class TelegramNotifier:
    """Telegram Bot: send alerts + receive commands."""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = bool(token and chat_id)
        self.last_update_id = 0

        # Reference to trader (set later)
        self.trader = None

        try:
            self.ctx = ssl._create_unverified_context()
        except Exception:
            self.ctx = None

        if self.enabled:
            logger.info("Telegram: ✅ (chat_id=%s)", chat_id)

    def set_trader(self, trader):
        """Link to LiveTrader for status queries."""
        self.trader = trader

    def start_polling(self):
        """Start background thread to listen for commands."""
        if not self.enabled:
            return
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        logger.info("Telegram command listener started")

    def _poll_loop(self):
        """Poll for incoming Telegram messages."""
        import time
        while True:
            try:
                self._check_messages()
            except Exception as e:
                logger.debug("Telegram poll error: %s", e)
            time.sleep(5)

    def _check_messages(self):
        """Check for new messages and handle commands."""
        url = f"{self.base_url}/getUpdates?offset={self.last_update_id + 1}&timeout=3"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10, context=self.ctx) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return

        if not data.get('ok'):
            return

        for update in data.get('result', []):
            self.last_update_id = update['update_id']
            msg = update.get('message', {})
            text = msg.get('text', '').strip()
            chat_id = str(msg.get('chat', {}).get('id', ''))

            # Only respond to our chat
            if chat_id != self.chat_id:
                continue

            if text.startswith('/'):
                self._handle_command(text.lower())

    def _handle_command(self, cmd: str):
        """Handle a bot command."""
        if cmd == '/status':
            self._cmd_status()
        elif cmd == '/positions' or cmd == '/pos':
            self._cmd_positions()
        elif cmd == '/balance' or cmd == '/bal':
            self._cmd_balance()
        elif cmd == '/trades':
            self._cmd_trades()
        elif cmd == '/help' or cmd == '/start':
            self._cmd_help()
        else:
            self._send(f"Unknown command: {cmd}\nType /help for commands")

    def _cmd_status(self):
        """Bot status overview."""
        if not self.trader:
            self._send("Bot not linked yet")
            return

        t = self.trader
        positions = t._get_all_positions()
        h1_dir = t.h1_direction or "None"
        h1_prob = t.h1_probability

        # News status
        news_line = "No calendar"
        if t.news_mgr:
            news_line = t.news_mgr.get_status_line()

        pos_count = len(positions)
        total_pnl = sum(p['profit'] for p in positions)

        self._send(
            f"📊 TRADEBOT STATUS\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Mode: {'DRY RUN' if t.dry_run else 'LIVE'}\n"
            f"H1 Direction: {h1_dir.upper()} ({h1_prob:.1%})\n"
            f"Open Positions: {pos_count}\n"
            f"Total PnL: ${total_pnl:.2f}\n"
            f"Trades Today: {t.trade_count}\n"
            f"{news_line}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

    def _cmd_positions(self):
        """Show open positions."""
        if not self.trader:
            self._send("Bot not linked")
            return

        positions = self.trader._get_all_positions()
        if not positions:
            self._send("📭 No open positions")
            return

        lines = [f"📌 OPEN POSITIONS ({len(positions)})"]
        for p in positions:
            emoji = "🟢" if p['type'] == 'long' else "🔴"
            hold_h = (datetime.now() - p['time']).total_seconds() / 3600
            lines.append(
                f"\n{emoji} {p['type'].upper()} {p['volume']:.2f} lots\n"
                f"  Entry: {p['price_open']:.2f}\n"
                f"  PnL: ${p['profit']:.2f}\n"
                f"  Hold: {hold_h:.1f}h\n"
                f"  SL: {p['sl']:.2f} | TP: {p['tp']:.2f}"
            )
        self._send("\n".join(lines))

    def _cmd_balance(self):
        """Show account balance."""
        if not self.trader:
            self._send("Bot not linked")
            return
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info:
                self._send(
                    f"💰 ACCOUNT\n"
                    f"Balance: ${info.balance:.2f}\n"
                    f"Equity: ${info.equity:.2f}\n"
                    f"Margin: ${info.margin:.2f}\n"
                    f"Free: ${info.margin_free:.2f}\n"
                    f"Profit: ${info.profit:.2f}"
                )
            else:
                self._send("Could not get account info")
        except Exception as e:
            self._send(f"Error: {e}")

    def _cmd_trades(self):
        """Today's trade count."""
        if not self.trader:
            self._send("Bot not linked")
            return
        self._send(f"📈 Trades today: {self.trader.trade_count}")

    def _cmd_help(self):
        """Show available commands."""
        self._send(
            "🤖 TRADEBOT COMMANDS\n"
            "━━━━━━━━━━━━━━━\n"
            "/status - Bot durumu\n"
            "/positions - Acik pozisyonlar\n"
            "/balance - Hesap bakiyesi\n"
            "/trades - Bugunun islem sayisi\n"
            "/help - Bu mesaj"
        )

    # ─── Notification Methods ─────────────────────────────────────

    def _send(self, text: str):
        if not self.enabled:
            return
        try:
            params = urllib.parse.urlencode({
                'chat_id': self.chat_id,
                'text': text,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/sendMessage",
                data=params,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            urllib.request.urlopen(req, timeout=10, context=self.ctx)
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)

    def notify_order(self, direction, volume, price, sl, tp, ticket, label="NEW"):
        emoji = "🟢" if direction == "long" else "🔴"
        self._send(
            f"{emoji} {label} {direction.upper()}\n"
            f"Lot: {volume:.2f} @ {price:.2f}\n"
            f"SL: {sl:.2f} | TP: {tp:.2f}\n"
            f"Ticket: {ticket}"
        )

    def notify_close(self, direction, ticket, profit, reason=""):
        emoji = "💰" if profit >= 0 else "💸"
        sign = "+" if profit >= 0 else ""
        self._send(
            f"{emoji} CLOSED {direction.upper()}\n"
            f"PnL: {sign}${profit:.2f}\n"
            f"Reason: {reason}\n"
            f"Ticket: {ticket}"
        )

    def notify_smart_exit(self, ticket, pos_type, ml_direction, loss_atr):
        self._send(
            f"🔄 SMART EXIT\n"
            f"{pos_type.upper()} -> ML says {ml_direction.upper()}\n"
            f"Loss: {loss_atr:.1f} ATR\n"
            f"Ticket: {ticket}"
        )

    def notify_daily_summary(self, trade_count, total_pnl, balance, positions):
        sign = "+" if total_pnl >= 0 else ""
        pos_text = f"{len(positions)} open" if positions else "No positions"
        self._send(
            f"📊 DAILY SUMMARY\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Trades: {trade_count}\n"
            f"PnL: {sign}${total_pnl:.2f}\n"
            f"Balance: ${balance:.2f}\n"
            f"Positions: {pos_text}"
        )

    def notify_startup(self, symbol, mode, ml):
        self._send(
            f"🚀 TradeBot v2.0 Started\n"
            f"Symbol: {symbol}\n"
            f"Mode: {mode}\n"
            f"ML: {'Active' if ml else 'Off'}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )


_notifier: Optional[TelegramNotifier] = None

def get_telegram() -> Optional[TelegramNotifier]:
    global _notifier
    if _notifier is None:
        try:
            import config
            token = getattr(config, 'TELEGRAM_TOKEN', '')
            chat_id = getattr(config, 'TELEGRAM_CHAT_ID', '')
            if token and chat_id:
                _notifier = TelegramNotifier(token, chat_id)
        except Exception as e:
            logger.warning("Telegram init error: %s", e)
    return _notifier
