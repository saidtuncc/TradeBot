# tests/test_indicators.py
"""Unit tests for indicator calculations and core logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestEMA(unittest.TestCase):
    """Test EMA calculation."""
    
    def test_ema_basic(self):
        from backtest import calculate_ema
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ema = calculate_ema(data, 3)
        # EMA should be close to recent values
        self.assertAlmostEqual(ema.iloc[-1], 9.0, delta=1.5)
    
    def test_ema_constant_series(self):
        from backtest import calculate_ema
        data = pd.Series([5.0] * 20)
        ema = calculate_ema(data, 10)
        # EMA of constant should be constant
        self.assertAlmostEqual(ema.iloc[-1], 5.0, places=5)
    
    def test_ema_length_matches(self):
        from backtest import calculate_ema
        data = pd.Series(range(100))
        ema = calculate_ema(data, 50)
        self.assertEqual(len(ema), 100)


class TestATR(unittest.TestCase):
    """Test ATR calculation."""
    
    def test_atr_basic(self):
        from backtest import calculate_atr
        high = pd.Series([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
        low = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        close = pd.Series([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        atr = calculate_atr(high, low, close, 5)
        # ATR should be around 2 (high-low range)
        self.assertAlmostEqual(atr.iloc[-1], 2.0, delta=0.5)
    
    def test_atr_positive(self):
        from backtest import calculate_atr
        np.random.seed(42)
        n = 100
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        high = close + abs(np.random.randn(n))
        low = close - abs(np.random.randn(n))
        atr = calculate_atr(high, low, close, 14)
        # ATR should always be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all())


class TestRSI(unittest.TestCase):
    """Test RSI calculation (vectorized ewm version)."""
    
    def test_rsi_range(self):
        from backtest import calculate_rsi
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        rsi = calculate_rsi(data, 14)
        valid_rsi = rsi.dropna()
        # RSI must be between 0 and 100
        self.assertTrue((valid_rsi >= 0).all(), f"RSI below 0: {valid_rsi.min()}")
        self.assertTrue((valid_rsi <= 100).all(), f"RSI above 100: {valid_rsi.max()}")
    
    def test_rsi_uptrend(self):
        from backtest import calculate_rsi
        # Strong uptrend → RSI should be high
        data = pd.Series(range(100, 200))
        rsi = calculate_rsi(data, 14)
        self.assertGreater(rsi.iloc[-1], 70)
    
    def test_rsi_downtrend(self):
        from backtest import calculate_rsi
        # Strong downtrend → RSI should be low
        data = pd.Series(range(200, 100, -1))
        rsi = calculate_rsi(data, 14)
        self.assertLess(rsi.iloc[-1], 30)


class TestNewsManager(unittest.TestCase):
    """Test news risk scoring."""
    
    def test_disabled_returns_zero(self):
        import config
        original = config.ENABLE_NEWS_FILTER
        config.ENABLE_NEWS_FILTER = False
        
        from news_manager import NewsManager
        nm = NewsManager()
        score = nm.calculate_risk_score(datetime.now())
        self.assertEqual(score, 0.0)
        
        config.ENABLE_NEWS_FILTER = original
    
    def test_no_events_returns_zero(self):
        from news_manager import NewsManager
        nm = NewsManager.__new__(NewsManager)
        nm.events = []
        nm._timestamps = []
        nm.enabled = True
        nm._file_loaded = True
        
        score = nm.calculate_risk_score(datetime(2020, 1, 1))
        self.assertEqual(score, 0.0)
    
    def test_risk_level_labels(self):
        from news_manager import NewsManager
        nm = NewsManager.__new__(NewsManager)
        nm.events = []
        nm._timestamps = []
        nm.enabled = True
        
        self.assertIn("BLOCKED", nm.get_risk_level(8.0))
        self.assertIn("REDUCED", nm.get_risk_level(5.0))
        self.assertIn("NORMAL", nm.get_risk_level(2.0))


class TestPositionManager(unittest.TestCase):
    """Test position manager logic."""
    
    def test_open_close_long(self):
        from backtest import PositionManager, Direction
        pm = PositionManager()
        
        pm.open_position(Direction.LONG, 100.0, 1.0, 5.0, datetime.now(), 0)
        self.assertTrue(pm.is_open)
        self.assertEqual(pm.direction, Direction.LONG)
        
        record = pm.close_position(110.0, datetime.now(), "take_profit")
        self.assertGreater(record.gross_pnl, 0)
        self.assertFalse(pm.is_open)
    
    def test_open_close_short(self):
        from backtest import PositionManager, Direction
        pm = PositionManager()
        
        pm.open_position(Direction.SHORT, 100.0, 1.0, 5.0, datetime.now(), 0)
        record = pm.close_position(90.0, datetime.now(), "take_profit")
        self.assertGreater(record.gross_pnl, 0)
    
    def test_trading_costs_deducted(self):
        from backtest import PositionManager, Direction
        pm = PositionManager()
        
        pm.open_position(Direction.LONG, 100.0, 1.0, 5.0, datetime.now(), 0)
        record = pm.close_position(100.0, datetime.now(), "time_exit")
        
        # Zero price change → gross_pnl = 0, but costs > 0
        self.assertEqual(record.gross_pnl, 0.0)
        self.assertGreater(record.trading_costs, 0.0)
        self.assertLess(record.net_pnl, 0.0)  # Net loss due to costs
    
    def test_pyramiding_locked(self):
        from backtest import PositionManager, Direction
        pm = PositionManager()
        pm.open_position(Direction.LONG, 100.0, 1.0, 5.0, datetime.now(), 0)
        
        # Should be rejected (PYRAMIDING_ENABLED = False)
        result = pm.add_layer(105.0, 0.5, 5.0, datetime.now())
        self.assertFalse(result)
        self.assertEqual(pm.num_layers, 1)
    
    def test_hedging_locked(self):
        from backtest import PositionManager, Direction
        pm = PositionManager()
        pm.open_position(Direction.LONG, 100.0, 1.0, 5.0, datetime.now(), 0)
        
        # Should be rejected (HEDGING_ENABLED = False)
        result = pm.activate_hedge()
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
