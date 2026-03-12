# telegram_bot.py
"""
Telegram Notifications for TradeBot.
Sends trade alerts, position updates, and daily summaries.
"""

import logging
import urllib.request
import urllib.parse
import ssl
from datetime import datetime
from typing import Optional

logger = logging.getLogger('telegram')


class TelegramNotifier:
    """Send trading notifications via Telegram Bot API."""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = bool(token and chat_id)

        try:
            self.ctx = ssl._create_unverified_context()
        except Exception:
            self.ctx = None

        if self.enabled:
            logger.info("Telegram: ✅ (chat_id=%s)", chat_id)

    def _send(self, text: str):
        """Send plain text message via Telegram API."""
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

    def notify_news(self, events, score):
        lines = [f"📅 NEWS ALERT (score={score:.1f})"]
        for e in events[:5]:
            hours = (e.timestamp - datetime.now()).total_seconds() / 3600
            lines.append(f"  [{e.impact}] {e.name} in {hours:.1f}h")
        self._send("\n".join(lines))

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
