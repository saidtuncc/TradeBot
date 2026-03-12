# ml/feature_engine.py
"""
Feature Engineering V2 — ML Trading Model
Major improvements over V1:
  1. M15 features are OPTIONAL (fillna instead of drop) → 67K rows vs 25K
  2. Triple-barrier labeling (SL/TP/time → realistic trade outcome)
  3. Added: Stochastic, OBV, lag features, session encoding
  4. Both LONG and SHORT targets
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# INDICATORS (vectorized)
# =========================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calculate_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger(close: pd.Series, period=20, std_mult=2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma * 100
    pct_b = (close - lower) / (upper - lower)
    return width, pct_b


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    obv = (volume * direction).cumsum()
    # Normalize to rolling z-score
    obv_zscore = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()
    return obv_zscore


# =========================================================================
# FEATURE BUILDER V2
# =========================================================================

def build_h1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all H1-based features. Returns ~35 features."""
    out = df.copy()
    c, h, l, o, v = out['close'], out['high'], out['low'], out['open'], out['volume']

    # --- Returns (6) ---
    for hours in [1, 2, 4, 8, 12, 24]:
        out[f'return_{hours}h'] = c.pct_change(hours) * 100

    # --- Lag returns (3) — previous bar returns for sequence context ---
    out['return_lag1'] = out['return_1h'].shift(1)
    out['return_lag2'] = out['return_1h'].shift(2)
    out['return_lag3'] = out['return_1h'].shift(3)

    # --- Trend (6) ---
    out['ema10'] = calculate_ema(c, 10)
    out['ema20'] = calculate_ema(c, 20)
    out['ema50'] = calculate_ema(c, 50)
    out['ema_spread'] = (out['ema10'] - out['ema50']) / out['ema50'] * 100
    out['ema_slope_5'] = out['ema50'].pct_change(5) * 100
    out['ema10_20_cross'] = ((out['ema10'] > out['ema20']).astype(int) -
                              (out['ema10'] < out['ema20']).astype(int))

    # --- Momentum (6) ---
    out['rsi14'] = calculate_rsi(c, 14)
    macd, macd_sig, macd_hist = calculate_macd(c)
    out['macd'] = macd
    out['macd_signal'] = macd_sig
    out['macd_hist'] = macd_hist
    stoch_k, stoch_d = calculate_stochastic(h, l, c)
    out['stoch_k'] = stoch_k
    out['stoch_d'] = stoch_d

    # --- Volatility (5) ---
    out['atr14'] = calculate_atr(h, l, c, 14)
    out['atr_avg50'] = out['atr14'].rolling(50).mean()
    out['atr_ratio'] = out['atr14'] / out['atr_avg50']
    bb_width, bb_pctb = calculate_bollinger(c)
    out['bb_width'] = bb_width
    out['bb_pctb'] = bb_pctb

    # --- Candle (4) ---
    body = (c - o).abs()
    full_range = (h - l).replace(0, np.nan)
    out['body_ratio'] = body / full_range
    out['upper_wick_pct'] = (h - pd.concat([c, o], axis=1).max(axis=1)) / full_range
    out['lower_wick_pct'] = (pd.concat([c, o], axis=1).min(axis=1) - l) / full_range
    out['gap'] = (o - c.shift(1)) / c.shift(1) * 100

    # --- ADX (4) ---
    adx, plus_di, minus_di = calculate_adx(h, l, c, 14)
    out['adx'] = adx
    out['plus_di'] = plus_di
    out['minus_di'] = minus_di
    out['adx_trend'] = adx.diff(3)

    # --- Volume (2) ---
    out['obv_zscore'] = calculate_obv(c, v)
    out['volume_ratio'] = v / v.rolling(20).mean()

    # --- Time (4) — cyclical encoding ---
    if 'datetime' in out.columns:
        dt = pd.to_datetime(out['datetime'])
        out['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        out['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
        out['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 5)
        out['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 5)

    n_features = len([c for c in out.columns
                      if c not in ['datetime', 'open', 'high', 'low', 'close', 'volume']])
    logger.info("Generated %d H1 features", n_features)
    return out


def add_higher_tf_features(h1_df: pd.DataFrame, h4_df=None, d1_df=None) -> pd.DataFrame:
    """Add higher timeframe features via merge_asof."""
    from ml.data_loader import merge_higher_tf_features
    result = h1_df.copy()

    if h4_df is not None:
        h4 = h4_df.copy()
        h4_c = h4.set_index('datetime')['close']
        h4['h4_ema20'] = calculate_ema(h4_c, 20).values
        h4['h4_rsi14'] = calculate_rsi(h4_c, 14).values
        h4_atr = calculate_atr(h4.set_index('datetime')['high'],
                               h4.set_index('datetime')['low'], h4_c, 14)
        h4['h4_atr14'] = h4_atr.values
        h4['h4_trend'] = (h4_c > h4['h4_ema20'].values).astype(int).values
        h4['h4_momentum'] = h4_c.pct_change(5).values * 100
        h4_features = h4[['datetime', 'h4_rsi14', 'h4_trend', 'h4_atr14', 'h4_momentum']].dropna()
        result = merge_higher_tf_features(result, h4_features)
        logger.info("Added 4 H4 features")

    if d1_df is not None:
        d1 = d1_df.copy()
        d1_c = d1.set_index('datetime')['close']
        d1['d1_ema20'] = calculate_ema(d1_c, 20).values
        d1['d1_rsi14'] = calculate_rsi(d1_c, 14).values
        d1_atr = calculate_atr(d1.set_index('datetime')['high'],
                               d1.set_index('datetime')['low'], d1_c, 14)
        d1['d1_atr14'] = d1_atr.values
        d1['d1_trend'] = (d1_c > d1['d1_ema20'].values).astype(int).values
        d1['d1_vol_ratio'] = (d1['d1_atr14'] / d1['d1_atr14'].rolling(50).mean()).values
        d1['d1_return_5d'] = d1_c.pct_change(5).values * 100
        d1_features = d1[['datetime', 'd1_rsi14', 'd1_atr14', 'd1_trend',
                          'd1_vol_ratio', 'd1_return_5d']].dropna()
        result = merge_higher_tf_features(result, d1_features)
        logger.info("Added 5 D1 features")

    return result


def add_m15_features(h1_df: pd.DataFrame, m15_df: pd.DataFrame) -> pd.DataFrame:
    """Add M15 features. Returns NaN for bars where M15 data unavailable."""
    m15 = m15_df.copy()
    m15_c = m15.set_index('datetime')['close']
    m15['m15_rsi14'] = calculate_rsi(m15_c, 14).values

    m15['m15_hour'] = m15['datetime'].dt.floor('h')
    m15_agg = m15.groupby('m15_hour').agg(
        m15_momentum=('close', lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else 0),
        m15_intra_range=('high', lambda x: x.max() - m15.loc[x.index, 'low'].min()),
        m15_rsi_last=('m15_rsi14', 'last'),
    ).reset_index().rename(columns={'m15_hour': 'datetime'})

    result = pd.merge(h1_df, m15_agg, on='datetime', how='left')
    logger.info("Added 3 M15 features (NaN-filled where unavailable)")
    return result


def add_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add news risk score features."""
    from news_manager import get_news_manager
    nm = get_news_manager()
    if not nm.enabled:
        df['news_risk_score'] = 0.0
        df['news_high_count'] = 0
        return df

    scores = []
    high_counts = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row['datetime']) if isinstance(row['datetime'], str) else row['datetime']
        score = nm.calculate_risk_score(dt)
        events = nm.get_upcoming_events(dt, hours=4)
        high_count = sum(1 for e in events if e.impact == 'HIGH')
        scores.append(score)
        high_counts.append(high_count)

    df['news_risk_score'] = scores
    df['news_high_count'] = high_counts
    logger.info("Added 2 news features")
    return df


# =========================================================================
# TRIPLE-BARRIER LABELING (Industry Standard)
# =========================================================================

def triple_barrier_label(
    df: pd.DataFrame,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 2.0,
    max_bars: int = 10,
) -> pd.DataFrame:
    """
    Triple-barrier labeling for BOTH directions:
      target_long:  1 if LONG trade would be profitable
      target_short: 1 if SHORT trade would be profitable
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr14'].values
    n = len(df)

    long_labels = np.zeros(n, dtype=int)
    short_labels = np.zeros(n, dtype=int)
    long_reasons = [''] * n
    short_reasons = [''] * n

    for i in range(n - max_bars):
        entry = closes[i]
        atr = atrs[i]
        if atr <= 0 or np.isnan(atr):
            continue

        # === LONG barriers ===
        long_tp = entry + tp_atr_mult * atr
        long_sl = entry - sl_atr_mult * atr
        hit = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if highs[j] >= long_tp:
                long_labels[i] = 1
                long_reasons[i] = 'tp'
                hit = True
                break
            if lows[j] <= long_sl:
                long_labels[i] = 0
                long_reasons[i] = 'sl'
                hit = True
                break
        if not hit:
            final = closes[min(i + max_bars, n - 1)]
            long_labels[i] = 1 if final > entry else 0
            long_reasons[i] = 'time'

        # === SHORT barriers (reversed) ===
        short_tp = entry - tp_atr_mult * atr
        short_sl = entry + sl_atr_mult * atr
        hit = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if lows[j] <= short_tp:
                short_labels[i] = 1
                short_reasons[i] = 'tp'
                hit = True
                break
            if highs[j] >= short_sl:
                short_labels[i] = 0
                short_reasons[i] = 'sl'
                hit = True
                break
        if not hit:
            final = closes[min(i + max_bars, n - 1)]
            short_labels[i] = 1 if final < entry else 0
            short_reasons[i] = 'time'

    df['target_long'] = long_labels
    df['target_short'] = short_labels
    df['target'] = long_labels  # Backward compat
    df['target_reason'] = long_reasons

    df = df.iloc[:-max_bars].copy()

    long_rate = df['target_long'].mean() * 100
    short_rate = df['target_short'].mean() * 100
    logger.info("Triple-barrier: LONG+=%.1f%%, SHORT+=%.1f%%", long_rate, short_rate)
    return df


# =========================================================================
# FULL PIPELINE V2
# =========================================================================

def build_full_feature_set(base_dir: str = '.', add_news: bool = True) -> pd.DataFrame:
    """
    V2 Feature Pipeline:
    - Does NOT drop rows with missing M15 (fills with 0 instead)
    - Uses triple-barrier labeling
    - ~50 features
    """
    from ml.data_loader import load_all_timeframes

    logger.info("=== Building Feature Set V2 ===")
    data = load_all_timeframes(base_dir)

    # H1 base
    h1 = build_h1_features(data['H1'])

    # Cross-TF
    h1 = add_higher_tf_features(h1, h4_df=data.get('H4'), d1_df=data.get('D1'))

    if 'M15' in data:
        h1 = add_m15_features(h1, data['M15'])

    # News
    if add_news:
        h1 = add_news_features(h1)

    # Triple-barrier target (instead of simple threshold)
    h1 = triple_barrier_label(h1, tp_atr_mult=2.0, sl_atr_mult=2.0, max_bars=8)

    # Fill M15 NaN with 0 instead of dropping
    m15_cols = [c for c in h1.columns if c.startswith('m15_')]
    for col in m15_cols:
        h1[col] = h1[col].fillna(0)

    # Drop remaining NaN from indicator warm-up only
    before = len(h1)
    h1 = h1.dropna(subset=[c for c in h1.columns if c not in
                           ['target_reason', 'target_pnl_ratio'] + m15_cols])
    logger.info("Dropped %d warm-up rows, final: %d rows (V1 had 25K)", before - len(h1), len(h1))

    return h1
