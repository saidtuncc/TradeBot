# telegram_bot.py
"""
Telegram Notifications for TradeBot.
Sends trade alerts, position updates, and daily summaries.

Setup:
  1. Talk to @BotFather on Telegram → /newbot → get TOKEN
  2. Talk to @userinfobot → get your CHAT_ID
  3. Set in config.py:
     TELEGRAM_TOKEN = "your-bot-token"
     TELEGRAM_CHAT_ID = "your-chat-id"
"""

import logging
import urllib.request
import urllib.parse
import ssl
import json
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

        # SSL context (VPS compatibility)
        try:
            self.ctx = ssl._create_unverified_context()
        except Exception:
            self.ctx = None

        if self.enabled:
            logger.info("Telegram: ✅ (chat_id=%s)", chat_id)
        else:
            logger.info("Telegram: ❌ (no token/chat_id)")

    def _send(self, text: str, parse_mode: str = "HTML"):
        """Send message via Telegram API."""
        if not self.enabled:
            return

        try:
            params = urllib.parse.urlencode({
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode,
            }).encode()

            req = urllib.request.Request(
                f"{self.base_url}/sendMessage",
                data=params,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            urllib.request.urlopen(req, timeout=10, context=self.ctx)

        except Exception as e:
            logger.warning("Telegram send failed: %s", e)

    # ─── Trade Notifications ─────────────────────────────────────────

    def notify_order(self, direction: str, volume: float, price: float,
                     sl: float, tp: float, ticket: int, label: str = "NEW"):
        """Notify: new order opened."""
        emoji = "🟢" if direction == "long" else "🔴"
        self._send(
            f"{emoji} <b>{label} {direction.upper()}</b>\n"
            f"📊 {volume:.2f} lots @ {price:.2f}\n"
            f"🛑 SL: {sl:.2f}\n"
            f"🎯 TP: {tp:.2f}\n"
            f"🎫 Ticket: {ticket}"
        )

    def notify_close(self, direction: str, ticket: int, profit: float,
                     reason: str = ""):
        """Notify: position closed."""
        emoji = "💰" if profit >= 0 else "💸"
        sign = "+" if profit >= 0 else ""
        self._send(
            f"{emoji} <b>CLOSED {direction.upper()}</b>\n"
            f"P&L: <b>{sign}${profit:.2f}</b>\n"
            f"Reason: {reason}\n"
            f"🎫 Ticket: {ticket}"
        )

    def notify_signal(self, h1_direction: str, h1_prob: float,
                      m15_rsi: float, m15_mom: float, confirmed: bool):
        """Notify: ML signal generated."""
        status = "✅ Confirmed" if confirmed else "⏳ Waiting M15"
        emoji = "📈" if h1_direction == "long" else "📉"
        self._send(
            f"{emoji} <b>ML Signal: {h1_direction.upper()}</b>\n"
            f"Prob: {h1_prob:.1%}\n"
            f"M15 RSI: {m15_rsi:.1f} | Mom: {m15_mom:.2f}%\n"
            f"Status: {status}"
        )

    def notify_smart_exit(self, ticket: int, pos_type: str,
                          ml_direction: str, loss_atr: float):
        """Notify: smart exit triggered."""
        self._send(
            f"🔄 <b>SMART EXIT</b>\n"
            f"Position: {pos_type.upper()} → ML says {ml_direction.upper()}\n"
            f"Loss: {loss_atr:.1f} ATR\n"
            f"🎫 Ticket: {ticket}"
        )

    def notify_news(self, events: list, score: float):
        """Notify: upcoming high-impact news."""
        lines = [f"📅 <b>NEWS ALERT</b> (score={score:.1f})"]
        for e in events[:5]:
            hours = (e.timestamp - datetime.now()).total_seconds() / 3600
            lines.append(f"  [{e.impact}] {e.name} in {hours:.1f}h")
        self._send("\n".join(lines))

    def notify_daily_summary(self, trade_count: int, total_pnl: float,
                             balance: float, positions: list):
        """Daily summary at midnight."""
        emoji = "📊" if total_pnl >= 0 else "⚠️"
        sign = "+" if total_pnl >= 0 else ""
        pos_text = f"{len(positions)} open" if positions else "No open positions"

        self._send(
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Trades: {trade_count}\n"
            f"P&L: <b>{sign}${total_pnl:.2f}</b>\n"
            f"Balance: ${balance:.2f}\n"
            f"Positions: {pos_text}"
        )

    def notify_startup(self, symbol: str, mode: str, ml: bool):
        """Bot started notification."""
        self._send(
            f"🚀 <b>TradeBot v2.0 Started</b>\n"
            f"Symbol: {symbol}\n"
            f"Mode: {mode}\n"
            f"ML: {'✅' if ml else '❌'}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )


# Singleton
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
            else:
                logger.info("Telegram not configured (set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID in config.py)")
        except Exception as e:
            logger.warning("Telegram init error: %s", e)
    return _notifier
