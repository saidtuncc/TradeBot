# telegram_bot.py
"""
Telegram Bot v3.0 — Rich notifications + Management commands.

Commands:
  /status      - Bot durumu, ensemble, pozisyonlar
  /positions   - Açık pozisyon detayları
  /balance     - Hesap bakiyesi
  /trades      - Bugünün işlem sayısı
  /performance - Haftalık/aylık performans
  /close <ID>  - Belirli pozisyonu kapat
  /closeall    - Tüm bot pozisyonlarını kapat
  /help        - Komut listesi
"""

import logging
import urllib.request
import urllib.parse
import ssl
import json
import threading
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger('telegram')


class TelegramNotifier:
    """Telegram Bot v3: Rich notifications + Management commands."""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = bool(token and chat_id)
        self.last_update_id = 0
        self.trader = None

        # Trade log for performance tracking
        self.trade_log = []
        self._load_trade_log()

        try:
            self.ctx = ssl._create_unverified_context()
        except Exception:
            self.ctx = None

        if self.enabled:
            logger.info("Telegram v3: ✅ (chat_id=%s)", chat_id)

    def set_trader(self, trader):
        self.trader = trader

    # ═══════════════════════════════════════════════════════════════
    # POLLING (Background command listener)
    # ═══════════════════════════════════════════════════════════════

    def start_polling(self):
        if not self.enabled:
            return
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        logger.info("Telegram command listener started")

    def _poll_loop(self):
        import time
        while True:
            try:
                self._check_messages()
            except Exception as e:
                logger.debug("Telegram poll error: %s", e)
            time.sleep(5)

    def _check_messages(self):
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

            if chat_id != self.chat_id:
                continue
            if text.startswith('/'):
                self._handle_command(text)

    def _handle_command(self, cmd: str):
        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        handlers = {
            '/status': self._cmd_status,
            '/positions': self._cmd_positions,
            '/pos': self._cmd_positions,
            '/balance': self._cmd_balance,
            '/bal': self._cmd_balance,
            '/trades': self._cmd_trades,
            '/performance': self._cmd_performance,
            '/perf': self._cmd_performance,
            '/close': lambda: self._cmd_close(args),
            '/closeall': self._cmd_closeall,
            '/help': self._cmd_help,
            '/start': self._cmd_help,
        }

        handler = handlers.get(command)
        if handler:
            try:
                handler()
            except Exception as e:
                self._send(f"❌ Hata: {e}")
        else:
            self._send(f"❓ Bilinmeyen komut: {command}\n/help yazın")

    # ═══════════════════════════════════════════════════════════════
    # COMMANDS
    # ═══════════════════════════════════════════════════════════════

    def _cmd_status(self):
        if not self.trader:
            self._send("⏳ Bot henüz bağlanmadı")
            return

        t = self.trader
        positions = t._get_all_positions()
        h1_dir = t.h1_direction or "Belirsiz"
        h1_prob = t.h1_probability
        pos_count = len(positions)
        total_pnl = sum(p['profit'] for p in positions)
        scalps = sum(1 for p in positions if 'SCALP' in (p.get('comment', '') or ''))
        normals = pos_count - scalps

        # Risk emoji
        if total_pnl > 0:
            risk = "🟢"
        elif total_pnl > -50:
            risk = "🟡"
        else:
            risk = "🔴"

        news_line = ""
        if t.news_mgr:
            news_line = f"\n📰 {t.news_mgr.get_status_line()}"

        self._send(
            f"📊 TRADEBOT v3.0\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"{'🔴 LIVE' if not t.dry_run else '🔸 DRY RUN'}\n\n"
            f"🧠 Ensemble: {h1_dir.upper()} ({h1_prob:.0%})\n"
            f"{risk} P&L: ${total_pnl:+.2f}\n"
            f"📌 Pozisyon: {normals} normal + {scalps} scalp\n"
            f"📈 İşlem: {t.trade_count} bugün"
            f"{news_line}\n\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )

    def _cmd_positions(self):
        if not self.trader:
            self._send("⏳ Bot henüz bağlanmadı")
            return

        positions = self.trader._get_all_positions()
        if not positions:
            self._send("📭 Açık pozisyon yok")
            return

        lines = [f"📌 AÇIK POZİSYONLAR ({len(positions)})"]
        lines.append("━━━━━━━━━━━━━━━━━")

        for p in sorted(positions, key=lambda x: x['time']):
            is_scalp = 'SCALP' in (p.get('comment', '') or '')
            emoji = "⚡" if is_scalp else ("🟢" if p['type'] == 'long' else "🔴")
            tag = " SCALP" if is_scalp else ""
            hold_h = (datetime.now() - p['time']).total_seconds() / 3600
            pnl_emoji = "✅" if p['profit'] >= 0 else "❌"

            lines.append(
                f"\n{emoji} #{p['ticket']}{tag}\n"
                f"  {p['type'].upper()} {p['volume']:.2f} lots\n"
                f"  Giriş: {p['price_open']:.2f}\n"
                f"  {pnl_emoji} P&L: ${p['profit']:+.2f}\n"
                f"  SL: {p['sl']:.2f} | TP: {p['tp']:.2f}\n"
                f"  ⏱ {hold_h:.1f} saat"
            )

        self._send("\n".join(lines))

    def _cmd_balance(self):
        if not self.trader:
            self._send("⏳ Bot henüz bağlanmadı")
            return
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info:
                margin_pct = (info.margin / info.equity * 100) if info.equity > 0 else 0
                equity_emoji = "🟢" if info.profit >= 0 else "🔴"
                self._send(
                    f"💰 HESAP BİLGİSİ\n"
                    f"━━━━━━━━━━━━━━━━━\n"
                    f"💵 Bakiye:  ${info.balance:.2f}\n"
                    f"{equity_emoji} Equity:  ${info.equity:.2f}\n"
                    f"📊 Marjin:  ${info.margin:.2f} ({margin_pct:.1f}%)\n"
                    f"🆓 Serbest: ${info.margin_free:.2f}\n"
                    f"📈 Kâr:     ${info.profit:+.2f}"
                )
            else:
                self._send("❌ Hesap bilgisi alınamadı")
        except Exception as e:
            self._send(f"❌ Hata: {e}")

    def _cmd_trades(self):
        if not self.trader:
            self._send("⏳ Bot henüz bağlanmadı")
            return

        today = [t for t in self.trade_log
                 if t.get('time', '').startswith(datetime.now().strftime('%Y-%m-%d'))]
        wins = sum(1 for t in today if t.get('pnl', 0) > 0)
        losses = sum(1 for t in today if t.get('pnl', 0) <= 0)
        total_pnl = sum(t.get('pnl', 0) for t in today)

        self._send(
            f"📈 BUGÜNKÜ İŞLEMLER\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Toplam: {self.trader.trade_count}\n"
            f"✅ Kazanç: {wins}\n"
            f"❌ Kayıp: {losses}\n"
            f"💰 Net P&L: ${total_pnl:+.2f}"
        )

    def _cmd_performance(self):
        """Weekly + monthly performance breakdown."""
        if not self.trade_log:
            self._send("📊 Henüz performans verisi yok")
            return

        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        week_trades = [t for t in self.trade_log
                       if t.get('time', '') >= week_ago.strftime('%Y-%m-%d')]
        month_trades = [t for t in self.trade_log
                        if t.get('time', '') >= month_ago.strftime('%Y-%m-%d')]

        def stats(trades):
            if not trades:
                return 0, 0, 0, 0
            pnls = [t.get('pnl', 0) for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100 if pnls else 0
            return len(trades), sum(pnls), wr, max(pnls) if pnls else 0

        wc, wp, wwr, wb = stats(week_trades)
        mc, mp, mwr, mb = stats(month_trades)

        self._send(
            f"📊 PERFORMANS\n"
            f"━━━━━━━━━━━━━━━━━\n\n"
            f"📅 Son 7 gün:\n"
            f"  İşlem: {wc}\n"
            f"  P&L: ${wp:+.2f}\n"
            f"  Win Rate: {wwr:.0f}%\n"
            f"  En iyi: ${wb:+.2f}\n\n"
            f"📅 Son 30 gün:\n"
            f"  İşlem: {mc}\n"
            f"  P&L: ${mp:+.2f}\n"
            f"  Win Rate: {mwr:.0f}%\n"
            f"  En iyi: ${mb:+.2f}"
        )

    def _cmd_close(self, args):
        """Close a specific position by ticket."""
        if not self.trader:
            self._send("⏳ Bot henüz bağlanmadı")
            return
        if not args:
            self._send("❓ Kullanım: /close <ticket>\nÖrnek: /close 12345")
            return
        if self.trader.dry_run:
            self._send("🔸 DRY RUN modunda pozisyon kapatılamaz")
            return

        try:
            ticket = int(args[0])
            positions = self.trader._get_all_positions()
            pos = next((p for p in positions if p['ticket'] == ticket), None)

            if not pos:
                self._send(f"❌ Ticket #{ticket} bulunamadı")
                return

            success = self.trader.connector.close_position(ticket)
            if success:
                self._send(
                    f"✅ KAPATILDI\n"
                    f"Ticket: #{ticket}\n"
                    f"Yön: {pos['type'].upper()}\n"
                    f"P&L: ${pos['profit']:+.2f}"
                )
            else:
                self._send(f"❌ Kapatma başarısız: #{ticket}")
        except ValueError:
            self._send("❌ Geçersiz ticket numarası")
        except Exception as e:
            self._send(f"❌ Hata: {e}")

    def _cmd_closeall(self):
        """Close all bot positions."""
        if not self.trader:
            self._send("⏳ Bot henüz bağlanmadı")
            return
        if self.trader.dry_run:
            self._send("🔸 DRY RUN modunda pozisyon kapatılamaz")
            return

        positions = self.trader._get_all_positions()
        if not positions:
            self._send("📭 Kapatılacak pozisyon yok")
            return

        closed = 0
        total_pnl = 0
        for pos in positions:
            try:
                if self.trader.connector.close_position(pos['ticket']):
                    closed += 1
                    total_pnl += pos['profit']
            except Exception:
                pass

        self._send(
            f"🔒 TOPLU KAPANIŞ\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Kapatılan: {closed}/{len(positions)}\n"
            f"Net P&L: ${total_pnl:+.2f}"
        )

    def _cmd_help(self):
        self._send(
            "🤖 TRADEBOT v3.0\n"
            "━━━━━━━━━━━━━━━━━\n\n"
            "📊 Bilgi:\n"
            "  /status - Bot durumu\n"
            "  /positions - Açık pozisyonlar\n"
            "  /balance - Hesap bakiyesi\n"
            "  /trades - Bugünkü işlemler\n"
            "  /performance - Haftalık/aylık\n\n"
            "⚙️ Yönetim:\n"
            "  /close <ticket> - Pozisyon kapat\n"
            "  /closeall - Tümünü kapat\n\n"
            "  /help - Bu mesaj"
        )

    # ═══════════════════════════════════════════════════════════════
    # RICH NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════

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
        """Rich order notification with type-specific formatting."""
        if label == 'SCALP' or 'SCALP' in label:
            emoji = "⚡"
            type_tag = "SCALP"
        elif 'PYRAMID' in label or 'DRY_PYRAMID' in label:
            emoji = "🟢"
            type_tag = "KADEME"
        else:
            emoji = "🟢" if direction == "long" else "🔴"
            type_tag = "YENİ İŞLEM"

        arrow = "📈" if direction == "long" else "📉"
        risk_pts = abs(price - sl)

        self._send(
            f"{emoji} {type_tag} {arrow}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Yön: {direction.upper()}\n"
            f"Lot: {volume:.2f} @ {price:.2f}\n"
            f"🛡 SL: {sl:.2f} ({risk_pts:.1f} puan)\n"
            f"🎯 TP: {tp:.2f}\n"
            f"🎫 #{ticket}"
        )

    def notify_ensemble(self, direction, probability, details):
        """Show ensemble breakdown when new signal arrives."""
        if not details:
            return

        lines = [
            f"🧠 ENSEMBLE SİNYALİ",
            f"━━━━━━━━━━━━━━━━━",
            f"Yön: {'📈' if direction == 'long' else '📉'} {direction.upper()} ({probability:.0%})",
            f""
        ]

        # Model agreement
        agree = 0
        total = 0
        for tf in ['h1', 'm15', 'm5']:
            key = f'{tf}_{direction}'
            if key in details:
                total += 1
                prob = details[key]
                check = "✅" if prob > 0.5 else "❌"
                bar = "█" * int(prob * 10)
                lines.append(f"  {check} {tf.upper():4s} {prob:.0%} {bar}")
                if prob > 0.5:
                    agree += 1

        strength = "💪 GÜÇLÜ" if agree == 3 else ("👍 NORMAL" if agree == 2 else "⚠️ ZAYIF")
        lines.append(f"\n{strength} ({agree}/{total} model onay)")
        self._send("\n".join(lines))

    def notify_kademe(self, direction, layer, volume, probability, dip_atr):
        """Kademe (DCA) notification."""
        self._send(
            f"🟢 KADEME {layer}/3\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Yön: {direction.upper()}\n"
            f"Lot: {volume:.2f}\n"
            f"Güven: {probability:.0%}\n"
            f"Düşüş: {dip_atr:.1f} ATR\n"
            f"⚠️ Analiz hala aynı yönde emin"
        )

    def notify_close(self, direction, ticket, profit, reason=""):
        """Rich close notification with reason explanation."""
        emoji = "💰" if profit >= 0 else "💸"

        # Turkish reason explanations
        reason_map = {
            'TP_HIT': '🎯 Take Profit hedefi',
            'SL_HIT': '🛡 Stop Loss tetiklendi',
            'DYNAMIC_PROFIT': '📈 M5 model kâr al sinyali',
            'FIFO_EXIT': '🔄 Kademeli çıkış (en eski)',
            'SMART_EXIT': '🧠 ML yön değiştirdi',
            'M5_EXIT': '📉 M5 ters sinyal + zarar',
            'DEEP_LOSS': '⛔ Derin zarar kesimi',
            'TIME': '⏰ Zaman aşımı (48h)',
            'SCALP_PROFIT': '⚡ Scalp kâr',
            'SCALP_TIME': '⚡ Scalp süre doldu',
            'SCALP_CUT': '⚡ Scalp zarar kesimi',
            'PROFIT_TAKE': '💰 Kâr realizasyonu',
        }
        reason_text = reason_map.get(reason, reason)

        self._send(
            f"{emoji} POZİSYON KAPANDI\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🎫 #{ticket} {direction.upper()}\n"
            f"{'✅' if profit >= 0 else '❌'} P&L: ${profit:+.2f}\n"
            f"📋 {reason_text}"
        )

        # Log for performance tracking
        self._log_trade(profit, reason)

    def notify_smart_exit(self, ticket, pos_type, ml_direction, loss_atr):
        self._send(
            f"🧠 AKILLI ÇIKIŞ\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🎫 #{ticket}\n"
            f"Eski: {pos_type.upper()} → ML: {ml_direction.upper()}\n"
            f"Zarar: {loss_atr:.1f} ATR\n"
            f"💡 Model yön değiştirdi, pozisyon kapatıldı"
        )

    def notify_daily_summary(self, trade_count, total_pnl, balance, positions):
        """End-of-day summary with emoji performance."""
        sign = "+" if total_pnl >= 0 else ""
        perf_emoji = "🟢" if total_pnl >= 0 else "🔴"

        # Calculate today's stats from trade log
        today_str = datetime.now().strftime('%Y-%m-%d')
        today = [t for t in self.trade_log if t.get('time', '').startswith(today_str)]
        wins = sum(1 for t in today if t.get('pnl', 0) > 0)
        losses = len(today) - wins
        wr = (wins / len(today) * 100) if today else 0

        pos_text = f"{len(positions)} açık" if positions else "Pozisyon yok"

        self._send(
            f"📊 GÜNLÜK ÖZET\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"{perf_emoji} {datetime.now().strftime('%d %B %Y')}\n\n"
            f"📈 İşlem: {trade_count}\n"
            f"✅ Kazanç: {wins} | ❌ Kayıp: {losses}\n"
            f"🎯 Win Rate: {wr:.0f}%\n"
            f"💰 Net P&L: {sign}${total_pnl:.2f}\n"
            f"🏦 Bakiye: ${balance:.2f}\n"
            f"📌 {pos_text}"
        )

    def notify_startup(self, symbol, mode, ml):
        model_info = ""
        if ml and self.trader and hasattr(self.trader, 'predictor') and self.trader.predictor:
            tfs = list(self.trader.predictor.timeframes.keys())
            model_info = f"\n🧠 Modeller: {', '.join(tf.upper() for tf in tfs)}"

        self._send(
            f"🚀 TradeBot v3.0 Başladı!\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol}\n"
            f"{'🔴 LIVE' if mode != 'DRY RUN' else '🔸 DRY RUN'}\n"
            f"{'🧠 ML: Aktif' if ml else '⚠️ ML: Kapalı'}"
            f"{model_info}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"✨ Ensemble + Scalp + Kademe aktif"
        )

    # ═══════════════════════════════════════════════════════════════
    # TRADE LOG (for performance tracking)
    # ═══════════════════════════════════════════════════════════════

    def _log_trade(self, pnl, reason):
        """Log trade for performance tracking."""
        entry = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'pnl': round(pnl, 2),
            'reason': reason,
        }
        self.trade_log.append(entry)
        self._save_trade_log()

    def _load_trade_log(self):
        """Load trade log from file."""
        path = 'telegram_trades.json'
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.trade_log = json.load(f)
            except Exception:
                self.trade_log = []

    def _save_trade_log(self):
        """Save trade log to file."""
        try:
            with open('telegram_trades.json', 'w') as f:
                json.dump(self.trade_log[-500:], f)  # Keep last 500
        except Exception:
            pass


# ─── Singleton ──────────────────────────────────────────────────

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
