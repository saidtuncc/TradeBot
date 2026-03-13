#!/usr/bin/env python
"""
Ensemble Backtester — Tests multi-TF strategy with kademe, scalp, dynamic exits.
Usage: python backtest_ensemble.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from dataclasses import dataclass, field
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────
INITIAL_CAPITAL = 10000.0
VOLUME = 0.10       # Base lot size
SPREAD_PTS = 5.0    # Typical spread in points
COMMISSION = 0.50   # Per lot per side
MAX_HOLD_H = 48     # Max hold hours
ATR_PERIOD = 14
SL_ATR = 2.0        # Normal SL
TP_ATR = 3.0        # Normal TP
SCALP_SL_ATR = 0.75
SCALP_TP_ATR = 1.0
MAX_POSITIONS = 5
KADEME_PROB_MIN = 0.60
KADEME_DIP_ATR = 0.5
SCALP_PROB_MIN = 0.65

# Ensemble weights
WEIGHTS = {'h1': 0.50, 'm15': 0.30, 'm5': 0.20}


@dataclass
class Position:
    ticket: int
    direction: str       # 'long' or 'short'
    entry_price: float
    volume: float
    sl: float
    tp: float
    entry_time: pd.Timestamp
    is_scalp: bool = False
    exit_price: float = 0.0
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: str = ''
    pnl: float = 0.0


@dataclass
class BacktestResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    scalp_trades: int = 0
    kademe_trades: int = 0
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)


def load_models():
    """Load all 6 models."""
    models = {}
    for name in ['h1_long', 'h1_short', 'm15_long', 'm15_short', 'm5_long', 'm5_short']:
        path = f'ml/models/{name}.pkl'
        if os.path.exists(path):
            models[name] = joblib.load(path)
            logger.info("Loaded %s (features=%d)", name, len(models[name]['features']))
    return models


def predict(models, tf, direction, features_df):
    """Predict probability for a single TF+direction."""
    key = f'{tf}_{direction}'
    if key not in models:
        return 0.5
    bundle = models[key]
    selected = bundle['features']
    X = pd.DataFrame(0.0, index=features_df.index, columns=selected)
    for col in selected:
        if col in features_df.columns:
            X[col] = features_df[col].values
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    raw = bundle['model'].predict(X)
    prob = float(raw[-1]) if len(raw) > 0 else 0.5
    # Calibrate
    cal = bundle.get('calibrator')
    scl = bundle.get('scaler')
    if cal and scl:
        prob = float(cal.predict_proba(scl.transform([[prob]]))[0][1])
    return np.clip(prob, 0, 1)


def ensemble_predict(models, h1_feats, m15_feats, m5_feats):
    """Weighted ensemble prediction."""
    probs = {}
    total_w = 0
    long_w = 0
    short_w = 0

    for tf, feats in [('h1', h1_feats), ('m15', m15_feats), ('m5', m5_feats)]:
        if feats is None:
            continue
        w = WEIGHTS[tf]
        lp = predict(models, tf, 'long', feats)
        sp = predict(models, tf, 'short', feats)
        probs[f'{tf}_long'] = lp
        probs[f'{tf}_short'] = sp
        long_w += lp * w
        short_w += sp * w
        total_w += w

    if total_w == 0:
        return None, 0.0, probs

    long_f = long_w / total_w
    short_f = short_w / total_w
    probs['ens_long'] = long_f
    probs['ens_short'] = short_f

    if long_f > 0.55:
        return 'long', long_f, probs
    elif short_f > 0.55:
        return 'short', short_f, probs
    return None, 0.0, probs


def calc_atr(df, period=ATR_PERIOD):
    """Calculate ATR from OHLC dataframe."""
    if len(df) < period + 1:
        return 50.0  # Default
    tr = np.maximum(
        df['high'].values[-period-1:] - df['low'].values[-period-1:],
        np.maximum(
            np.abs(df['high'].values[-period-1:] - np.roll(df['close'].values[-period-1:], 1)),
            np.abs(df['low'].values[-period-1:] - np.roll(df['close'].values[-period-1:], 1))
        )
    )
    return float(np.mean(tr[1:]))


def run_backtest():
    """Main backtest loop."""
    from ml.feature_engine import (build_h1_features, add_higher_tf_features,
                                    build_m15_features, add_m15_higher_tf,
                                    build_m5_features, add_m5_higher_tf)

    logger.info("=" * 60)
    logger.info("ENSEMBLE BACKTEST")
    logger.info("=" * 60)

    # Load models
    models = load_models()
    if len(models) < 4:
        logger.error("Need at least 4 models loaded")
        return

    # Load data
    col_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    def load_csv(path):
        df = pd.read_csv(path, sep='\t', header=None, names=col_names)
        df['datetime'] = pd.to_datetime(df['datetime'])
        for c in ['open', 'high', 'low', 'close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        return df.dropna(subset=['close'])

    h1_raw = load_csv('data/USATECHIDXUSD60.csv')
    m15_raw = load_csv('data/USATECHIDXUSD15.csv')
    m5_raw = load_csv('data/USATECHIDXUSD5.csv')

    logger.info("H1:  %d bars (%s to %s)", len(h1_raw), h1_raw['datetime'].min(), h1_raw['datetime'].max())
    logger.info("M15: %d bars (%s to %s)", len(m15_raw), m15_raw['datetime'].min(), m15_raw['datetime'].max())
    logger.info("M5:  %d bars (%s to %s)", len(m5_raw), m5_raw['datetime'].min(), m5_raw['datetime'].max())

    # Build features for all TFs
    logger.info("Building H1 features...")
    h1_df = build_h1_features(h1_raw.copy())

    # H4/D1 for cross-TF
    h4_path = 'data/USATECHIDXUSD240.csv'
    d1_path = 'data/USATECHIDXUSD1440.csv'
    h4_df_raw = load_csv(h4_path) if os.path.exists(h4_path) else None
    d1_df_raw = load_csv(d1_path) if os.path.exists(d1_path) else None

    if h4_df_raw is not None or d1_df_raw is not None:
        h1_df = add_higher_tf_features(h1_df, h4_df=h4_df_raw, d1_df=d1_df_raw)

    logger.info("Building M15 features...")
    m15_df = build_m15_features(m15_raw.copy())
    m15_df = add_m15_higher_tf(m15_df, h1_df=h1_raw.copy(), h4_df=h4_df_raw, d1_df=d1_df_raw)

    logger.info("Building M5 features...")
    m5_df = build_m5_features(m5_raw.copy())
    m5_df = add_m5_higher_tf(m5_df, h1_df=h1_raw.copy(), m15_df=m15_raw.copy())

    # Clean
    for df in [h1_df, m15_df, m5_df]:
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

    # Use last 20% as test period
    h1_test_start = int(len(h1_df) * 0.8)
    test_start_time = h1_df.iloc[h1_test_start]['datetime']
    logger.info("Test period starts: %s", test_start_time)

    # State
    capital = INITIAL_CAPITAL
    positions: List[Position] = []
    closed_trades: List[Position] = []
    equity_curve = []
    ticket_counter = 0
    h1_direction = None
    h1_probability = 0.0
    h1_signal_time = None

    # Simulate bar-by-bar on H1 timeframe
    logger.info("Running simulation...")
    for h1_idx in range(h1_test_start, len(h1_df)):
        h1_row = h1_df.iloc[h1_idx]
        now = h1_row['datetime']
        price = float(h1_row['close'])
        high = float(h1_row['high'])
        low = float(h1_row['low'])

        # ATR
        atr = calc_atr(h1_df.iloc[max(0, h1_idx-ATR_PERIOD-1):h1_idx+1])

        # Get H1 features
        h1_feats = h1_df.iloc[[h1_idx]].copy()

        # Find matching M15 features (closest before H1 bar)
        m15_mask = m15_df['datetime'] <= now
        m15_feats = m15_df[m15_mask].iloc[[-1]].copy() if m15_mask.any() else None

        # Find matching M5 features
        m5_mask = m5_df['datetime'] <= now
        m5_feats = m5_df[m5_mask].iloc[[-1]].copy() if m5_mask.any() else None

        # ── Check SL/TP hits for existing positions ──
        to_close = []
        for pos in positions:
            if pos.direction == 'long':
                if low <= pos.sl:
                    pos.exit_price = pos.sl
                    pos.exit_reason = 'SL_HIT'
                    to_close.append(pos)
                elif high >= pos.tp:
                    pos.exit_price = pos.tp
                    pos.exit_reason = 'TP_HIT'
                    to_close.append(pos)
            else:
                if high >= pos.sl:
                    pos.exit_price = pos.sl
                    pos.exit_reason = 'SL_HIT'
                    to_close.append(pos)
                elif low <= pos.tp:
                    pos.exit_price = pos.tp
                    pos.exit_reason = 'TP_HIT'
                    to_close.append(pos)

        for pos in to_close:
            pos.exit_time = now
            if pos.direction == 'long':
                pos.pnl = (pos.exit_price - pos.entry_price) * pos.volume * 100 - COMMISSION
            else:
                pos.pnl = (pos.entry_price - pos.exit_price) * pos.volume * 100 - COMMISSION
            capital += pos.pnl
            closed_trades.append(pos)
            positions.remove(pos)

        # ── Dynamic position management ──
        if m5_feats is not None:
            m5_long_p = predict(models, 'm5', 'long', m5_feats)
            m5_short_p = predict(models, 'm5', 'short', m5_feats)
        else:
            m5_long_p = m5_short_p = 0.5

        positions_sorted = sorted(positions, key=lambda p: p.entry_time)
        to_close_dynamic = []
        for i, pos in enumerate(positions_sorted):
            hold_h = (now - pos.entry_time).total_seconds() / 3600
            pnl_pts = (price - pos.entry_price) if pos.direction == 'long' \
                else (pos.entry_price - price)
            pnl_atr = pnl_pts / atr if atr > 0 else 0

            # Scalp management
            if pos.is_scalp:
                if hold_h > 2.0 or pnl_atr >= 0.8 or pnl_atr < -0.5:
                    pos.exit_price = price
                    pos.exit_reason = 'SCALP_EXIT'
                    to_close_dynamic.append(pos)
                continue

            # Time exit
            if hold_h > MAX_HOLD_H:
                pos.exit_price = price
                pos.exit_reason = 'TIME'
                to_close_dynamic.append(pos)
                continue

            # Dynamic profit-take (M5 model)
            if pnl_atr >= 1.5:
                m5_against = (pos.direction == 'long' and m5_short_p > 0.60) or \
                             (pos.direction == 'short' and m5_long_p > 0.60)
                if m5_against:
                    pos.exit_price = price
                    pos.exit_reason = 'DYNAMIC_PROFIT'
                    to_close_dynamic.append(pos)
                    continue

            # FIFO exit (oldest first at +1.0 ATR)
            normal_pos = [p for p in positions_sorted if not p.is_scalp]
            if pnl_atr >= 1.0 and len(normal_pos) > 1 and i == 0:
                pos.exit_price = price
                pos.exit_reason = 'FIFO_EXIT'
                to_close_dynamic.append(pos)
                continue

            # Smart exit (direction changed)
            if pnl_atr < -0.3 and h1_direction and h1_direction != pos.direction:
                pos.exit_price = price
                pos.exit_reason = 'SMART_EXIT'
                to_close_dynamic.append(pos)
                continue

            # M5 exit (strongly against + moderate loss)
            if pnl_atr < -0.8:
                m5_against = (pos.direction == 'long' and m5_short_p > 0.65) or \
                             (pos.direction == 'short' and m5_long_p > 0.65)
                if m5_against:
                    pos.exit_price = price
                    pos.exit_reason = 'M5_EXIT'
                    to_close_dynamic.append(pos)
                    continue

            # Trailing stop
            if pnl_atr >= 1.0:
                if pos.direction == 'long':
                    new_sl = price - 0.7 * atr
                    if new_sl > pos.sl:
                        pos.sl = new_sl
                else:
                    new_sl = price + 0.7 * atr
                    if new_sl < pos.sl:
                        pos.sl = new_sl

            # Breakeven
            if pnl_atr >= 0.8:
                if pos.direction == 'long' and pos.sl < pos.entry_price:
                    pos.sl = pos.entry_price + 0.1 * atr
                elif pos.direction == 'short' and pos.sl > pos.entry_price:
                    pos.sl = pos.entry_price - 0.1 * atr

        for pos in to_close_dynamic:
            pos.exit_time = now
            if pos.direction == 'long':
                pos.pnl = (pos.exit_price - pos.entry_price) * pos.volume * 100 - COMMISSION
            else:
                pos.pnl = (pos.entry_price - pos.exit_price) * pos.volume * 100 - COMMISSION
            pos.pnl -= SPREAD_PTS * pos.volume  # Spread cost
            capital += pos.pnl
            closed_trades.append(pos)
            if pos in positions:
                positions.remove(pos)

        # ── Ensemble signal ──
        direction, prob, details = ensemble_predict(models, h1_feats, m15_feats, m5_feats)
        if direction:
            h1_direction = direction
            h1_probability = prob
            h1_signal_time = now

        # ── M15 confirmation → Open normal trade ──
        if h1_direction and len(positions) < MAX_POSITIONS:
            m15_prob = predict(models, 'm15', h1_direction, m15_feats) if m15_feats is not None else 0.5
            if m15_prob > 0.55:
                # Check no duplicate recent entry
                same = [p for p in positions if p.direction == h1_direction and not p.is_scalp]
                recent = any((now - p.entry_time).total_seconds() < 3600 for p in same)
                if not recent and len(same) < 3:
                    ticket_counter += 1
                    if h1_direction == 'long':
                        sl = price - SL_ATR * atr
                        tp = price + TP_ATR * atr
                        entry_cost = SPREAD_PTS * VOLUME
                    else:
                        sl = price + SL_ATR * atr
                        tp = price - TP_ATR * atr
                        entry_cost = SPREAD_PTS * VOLUME

                    capital -= entry_cost + COMMISSION
                    positions.append(Position(
                        ticket=ticket_counter, direction=h1_direction,
                        entry_price=price, volume=VOLUME,
                        sl=sl, tp=tp, entry_time=now, is_scalp=False
                    ))

        # ── Smart kademe ──
        if h1_direction and h1_probability >= KADEME_PROB_MIN:
            same = [p for p in positions if p.direction == h1_direction and not p.is_scalp]
            if 0 < len(same) < 3:
                avg_entry = sum(p.entry_price for p in same) / len(same)
                dip = (avg_entry - price) / atr if h1_direction == 'long' else (price - avg_entry) / atr
                newest = max(same, key=lambda p: p.entry_time)
                cooldown_ok = (now - newest.entry_time).total_seconds() >= 3600  # 1h cooldown in backtest

                if dip >= KADEME_DIP_ATR and cooldown_ok:
                    m15_prob = predict(models, 'm15', h1_direction, m15_feats) if m15_feats is not None else 0.5
                    if m15_prob > 0.55:
                        ticket_counter += 1
                        vol = max(round(VOLUME * (0.8 ** len(same)), 2), VOLUME)
                        if h1_direction == 'long':
                            sl = price - SL_ATR * atr
                            tp = price + TP_ATR * atr
                        else:
                            sl = price + SL_ATR * atr
                            tp = price - TP_ATR * atr
                        capital -= SPREAD_PTS * vol + COMMISSION
                        positions.append(Position(
                            ticket=ticket_counter, direction=h1_direction,
                            entry_price=price, volume=vol,
                            sl=sl, tp=tp, entry_time=now, is_scalp=False
                        ))

        # ── M5 scalp ──
        if m5_feats is not None and len(positions) < MAX_POSITIONS:
            scalps = [p for p in positions if p.is_scalp]
            if len(scalps) < 1:
                if m5_long_p > SCALP_PROB_MIN:
                    ticket_counter += 1
                    vol = max(round(VOLUME * 0.5, 2), VOLUME)
                    sl = price - SCALP_SL_ATR * atr
                    tp = price + SCALP_TP_ATR * atr
                    capital -= SPREAD_PTS * vol + COMMISSION
                    positions.append(Position(
                        ticket=ticket_counter, direction='long',
                        entry_price=price, volume=vol,
                        sl=sl, tp=tp, entry_time=now, is_scalp=True
                    ))
                elif m5_short_p > SCALP_PROB_MIN:
                    ticket_counter += 1
                    vol = max(round(VOLUME * 0.5, 2), VOLUME)
                    sl = price + SCALP_SL_ATR * atr
                    tp = price - SCALP_TP_ATR * atr
                    capital -= SPREAD_PTS * vol + COMMISSION
                    positions.append(Position(
                        ticket=ticket_counter, direction='short',
                        entry_price=price, volume=vol,
                        sl=sl, tp=tp, entry_time=now, is_scalp=True
                    ))

        # Equity
        unrealized = 0
        for p in positions:
            if p.direction == 'long':
                unrealized += (price - p.entry_price) * p.volume * 100
            else:
                unrealized += (p.entry_price - price) * p.volume * 100
        equity_curve.append({'time': now, 'equity': capital + unrealized, 'capital': capital})

    # Close remaining positions
    for pos in positions:
        pos.exit_price = float(h1_df.iloc[-1]['close'])
        pos.exit_time = h1_df.iloc[-1]['datetime']
        pos.exit_reason = 'END'
        if pos.direction == 'long':
            pos.pnl = (pos.exit_price - pos.entry_price) * pos.volume * 100 - COMMISSION
        else:
            pos.pnl = (pos.entry_price - pos.exit_price) * pos.volume * 100 - COMMISSION
        capital += pos.pnl
        closed_trades.append(pos)

    # ── Report ──
    result = analyze_results(closed_trades, equity_curve)
    print_report(result)
    return result


def analyze_results(trades, equity_curve):
    """Analyze backtest results."""
    r = BacktestResult()
    r.trades = trades
    r.equity_curve = equity_curve
    r.total_trades = len(trades)

    if not trades:
        return r

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    r.winners = len(wins)
    r.losers = len(losses)
    r.total_pnl = sum(pnls)
    r.win_rate = r.winners / r.total_trades if r.total_trades > 0 else 0
    r.avg_win = np.mean(wins) if wins else 0
    r.avg_loss = np.mean(losses) if losses else 0
    r.profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999

    r.scalp_trades = sum(1 for t in trades if t.is_scalp)
    r.kademe_trades = sum(1 for t in trades if t.exit_reason in ['FIFO_EXIT'])

    # Drawdown
    if equity_curve:
        eq = [e['equity'] for e in equity_curve]
        peak = eq[0]
        max_dd = 0
        for v in eq:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100
            max_dd = max(max_dd, dd)
        r.max_drawdown = max_dd

    # Sharpe
    if len(pnls) > 1:
        daily_returns = pd.Series(pnls)
        r.sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    return r


def print_report(r):
    """Print formatted backtest report."""
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info("  Total Trades:    %d", r.total_trades)
    logger.info("  Winners:         %d (%.1f%%)", r.winners, r.win_rate * 100)
    logger.info("  Losers:          %d", r.losers)
    logger.info("  Total P&L:       $%.2f", r.total_pnl)
    logger.info("  Avg Win:         $%.2f", r.avg_win)
    logger.info("  Avg Loss:        $%.2f", r.avg_loss)
    logger.info("  Profit Factor:   %.2f", r.profit_factor)
    logger.info("  Max Drawdown:    %.1f%%", r.max_drawdown)
    logger.info("  Sharpe Ratio:    %.2f", r.sharpe)
    logger.info("  Scalp Trades:    %d", r.scalp_trades)
    logger.info("  FIFO Exits:      %d", r.kademe_trades)
    logger.info("")

    # Exit reasons breakdown
    reasons = {}
    for t in r.trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    logger.info("Exit Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pnl = sum(t.pnl for t in r.trades if t.exit_reason == reason)
        logger.info("  %-18s %4d trades  $%+.2f", reason, count, pnl)

    # Monthly breakdown
    if r.trades:
        logger.info("\nMonthly P&L:")
        monthly = {}
        for t in r.trades:
            if t.exit_time:
                key = t.exit_time.strftime('%Y-%m')
                monthly[key] = monthly.get(key, 0) + t.pnl
        for month, pnl in sorted(monthly.items()):
            bar = "█" * max(1, int(abs(pnl) / 10))
            sign = "+" if pnl > 0 else ""
            logger.info("  %s: $%s%.2f %s", month, sign, pnl, bar if pnl > 0 else "")


if __name__ == '__main__':
    run_backtest()
