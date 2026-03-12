# ml/data_loader.py
"""
Multi-timeframe data loader for ML pipeline.
Loads H1, M15, H4, D1 data and merges cross-timeframe features.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)

DEFAULT_PATHS = {
    'H1': 'data/USATECHIDXUSD60.csv',
    'M15': 'data/USATECHIDXUSD15.csv',
    'H4': 'data/USATECHIDXUSD240.csv',
    'D1': 'data/USATECHIDXUSD1440.csv',
}


def load_ohlcv(filepath: str) -> pd.DataFrame:
    """Load a single CSV file into a datetime-indexed DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(
        filepath, sep=r'\s+', header=None,
        names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
        dtype={'date': str, 'time': str, 'open': float, 'high': float,
               'low': float, 'close': float, 'volume': float},
        on_bad_lines='skip'
    )

    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M')
    df = df.drop(columns=['date', 'time'])
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    df = df.sort_values('datetime').reset_index(drop=True)

    invalid = (
        (df['high'] < df['low']) | (df['high'] < df['open']) |
        (df['high'] < df['close']) | (df['low'] > df['open']) | (df['low'] > df['close'])
    )
    df = df[~invalid]
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

    logger.info("Loaded %s: %d bars (%s to %s)", filepath, len(df),
                df['datetime'].min(), df['datetime'].max())
    return df


def load_all_timeframes(base_dir: str = '.') -> Dict[str, pd.DataFrame]:
    """Load all available timeframe data."""
    data = {}
    for tf, filename in DEFAULT_PATHS.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            data[tf] = load_ohlcv(path)
        else:
            logger.warning("Timeframe %s not available: %s", tf, path)
    return data


def merge_higher_tf_features(
    h1_df: pd.DataFrame,
    htf_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge higher-timeframe features into H1 data via merge_asof (no look-ahead).
    htf_features must have 'datetime' column.
    """
    h1 = h1_df.sort_values('datetime').reset_index(drop=True)
    htf = htf_features.sort_values('datetime').reset_index(drop=True)

    merged = pd.merge_asof(
        h1, htf,
        on='datetime',
        direction='backward'
    )
    return merged
