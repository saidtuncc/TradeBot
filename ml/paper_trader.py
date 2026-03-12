# ml/paper_trader.py
"""
Paper Trading Module — Real-time ML signal logger + simulator.
Run this alongside your live bot to monitor signals without real money.
"""

import logging
import os
import json
import csv
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)

SIGNAL_LOG = 'ml/paper_trades.csv'
SUMMARY_LOG = 'ml/paper_summary.json'


class PaperTrader:
    """
    Monitors ML signals in real-time, logs them, and tracks
    hypothetical P&L without making real trades.
    """

    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.open_position = None
        self.trades = []
        self.signal_count = 0

        # Initialize predictor
        from ml.predictor import get_predictor
        self.predictor = get_predictor()

        # Ensure CSV header exists
        if not os.path.exists(SIGNAL_LOG):
            with open(SIGNAL_LOG, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'direction', 'probability', 'entry_price',
                    'exit_price', 'exit_reason', 'pnl', 'bars_held',
                    'atr', 'threshold_used'
                ])

        logger.info("PaperTrader initialized: capital=%.0f", initial_capital)

    def evaluate_bar(self, bar_data: Dict, features=None) -> Optional[Dict]:
        """
        Evaluate a single bar for ML signal.

        Args:
            bar_data: dict with keys: datetime, open, high, low, close, volume, atr14
            features: pd.DataFrame with ML features (if pre-computed)

        Returns:
            Signal dict or None
        """
        import config
        import pandas as pd
        import numpy as np

        if not self.predictor or not self.predictor.loaded:
            return None

        # Build features if not provided
        if features is None:
            features = pd.DataFrame({k: [v] for k, v in bar_data.items()
                                      if k not in ['datetime', 'open', 'high', 'low',
                                                    'close', 'volume']})
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Check for exit first
        if self.open_position:
            exit_signal = self._check_exit(bar_data)
            if exit_signal:
                self._close_position(exit_signal, bar_data)

        # Check for entry
        if not self.open_position:
            direction, prob = self.predictor.predict_direction(features)
            if direction:
                self.signal_count += 1
                signal = {
                    'timestamp': bar_data.get('datetime', datetime.now()),
                    'direction': direction,
                    'probability': prob,
                    'entry_price': bar_data['close'],
                    'atr': bar_data.get('atr14', 0),
                }

                # Calculate SL/TP
                atr = signal['atr']
                if direction == 'long':
                    signal['sl'] = signal['entry_price'] - config.STOP_LOSS_ATR_MULT * atr
                    signal['tp'] = signal['entry_price'] + config.TAKE_PROFIT_ATR_MULT * atr
                else:
                    signal['sl'] = signal['entry_price'] + config.STOP_LOSS_ATR_MULT * atr
                    signal['tp'] = signal['entry_price'] - config.TAKE_PROFIT_ATR_MULT * atr

                # Position size (Kelly)
                tp_sl = config.TAKE_PROFIT_ATR_MULT / config.STOP_LOSS_ATR_MULT
                kelly = self.predictor.kelly_size(prob, tp_sl)
                risk_amount = self.capital * config.RISK_PER_TRADE
                sl_dist = config.STOP_LOSS_ATR_MULT * atr
                base_size = risk_amount / sl_dist if sl_dist > 0 else 0
                signal['size'] = base_size * max(kelly * 4, 0.3)

                self.open_position = signal
                self.open_position['bars_held'] = 0

                logger.info("📊 PAPER SIGNAL: %s @ %.2f (prob=%.2f, size=%.4f)",
                            direction.upper(), signal['entry_price'], prob, signal['size'])

                return signal

        return None

    def _check_exit(self, bar_data: Dict) -> Optional[str]:
        """Check if open position should be exited."""
        import config

        pos = self.open_position
        pos['bars_held'] += 1
        high, low, close = bar_data['high'], bar_data['low'], bar_data['close']

        if pos['direction'] == 'long':
            if low <= pos['sl']:
                return 'stop_loss'
            if high >= pos['tp']:
                return 'take_profit'
        else:
            if high >= pos['sl']:
                return 'stop_loss'
            if low <= pos['tp']:
                return 'take_profit'

        if pos['bars_held'] >= config.MAX_BARS_IN_TRADE:
            return 'time_exit'

        return None

    def _close_position(self, reason: str, bar_data: Dict):
        """Close the position and log it."""
        pos = self.open_position

        if reason == 'stop_loss':
            exit_price = pos['sl']
        elif reason == 'take_profit':
            exit_price = pos['tp']
        else:
            exit_price = bar_data['close']

        if pos['direction'] == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']

        # Trading costs
        from config import SPREAD_POINTS, SLIPPAGE_POINTS
        costs = (SPREAD_POINTS + SLIPPAGE_POINTS) * pos['size']
        net_pnl = pnl - costs

        self.capital += net_pnl

        trade_record = {
            'timestamp': str(pos['timestamp']),
            'direction': pos['direction'],
            'probability': pos['probability'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'exit_reason': reason,
            'pnl': net_pnl,
            'bars_held': pos['bars_held'],
            'atr': pos['atr'],
            'threshold_used': 0,
        }
        self.trades.append(trade_record)

        # Log to CSV
        with open(SIGNAL_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(trade_record.values())

        logger.info("📊 PAPER CLOSE: %s → %s, PnL=%.2f, Bars=%d, Capital=%.0f",
                     pos['direction'].upper(), reason, net_pnl,
                     pos['bars_held'], self.capital)

        self.open_position = None

    def get_summary(self) -> Dict:
        """Get current paper trading summary."""
        if not self.trades:
            return {'status': 'no_trades', 'capital': self.capital}

        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        gp = sum(t['pnl'] for t in wins)
        gl = abs(sum(t['pnl'] for t in losses))

        summary = {
            'total_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades) * 100,
            'profit_factor': gp / gl if gl > 0 else float('inf'),
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'capital': self.capital,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'long_trades': len([t for t in self.trades if t['direction'] == 'long']),
            'short_trades': len([t for t in self.trades if t['direction'] == 'short']),
            'signals_total': self.signal_count,
        }

        # Save summary
        with open(SUMMARY_LOG, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def print_summary(self):
        """Print formatted summary."""
        s = self.get_summary()
        print(f"\n{'='*50}")
        print(f"PAPER TRADING SUMMARY")
        print(f"{'='*50}")
        print(f"  Trades:   {s.get('total_trades', 0)}")
        print(f"  Win Rate: {s.get('win_rate', 0):.1f}%")
        print(f"  PF:       {s.get('profit_factor', 0):.2f}")
        print(f"  Return:   {s.get('return_pct', 0):.2f}%")
        print(f"  Capital:  {s.get('capital', 0):,.0f}")
        print(f"  L/S:      {s.get('long_trades', 0)}/{s.get('short_trades', 0)}")
