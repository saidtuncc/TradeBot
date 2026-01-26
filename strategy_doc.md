# strategy_doc.md
# NASDAQ H1 Trend-Following Trading System

## Overview

A robust, rule-based, low-risk trend-following system designed for the NASDAQ index on the H1 timeframe. Prioritizes clarity and simplicity over complexity to avoid overfitting.

---

## Strategy Logic (Plain English)

**Core Philosophy:** Trade with the established trend, protect capital aggressively, and avoid uncertain conditions.

The system waits for a clear trend to establish (price above/below EMA 50), confirms momentum is present but not exhausted (RSI filter), ensures the market is active enough to trade (ATR volatility filter), and avoids high-impact news events.

### Indicators Used (3 Total)

| # | Category | Indicator | Purpose |
|---|----------|-----------|---------|
| 1 | Trend | EMA 50 | Direction filter - only trade in trend direction |
| 2 | Volatility | ATR 14 | Dynamic stop sizing + chop filter |
| 3 | Momentum | RSI 14 | Prevents exhaustion entries |

---

## Entry Conditions

### Long Entry (ALL must be true)
1. ✅ Close > EMA(50)
2. ✅ 40 ≤ RSI(14) ≤ 70
3. ✅ ATR(14) > 0.5 × ATR_Average(50)
4. ✅ Not in news blackout window
5. ✅ Daily loss < 1%

### Short Entry (ALL must be true)
1. ✅ Close < EMA(50)
2. ✅ 30 ≤ RSI(14) ≤ 60
3. ✅ ATR(14) > 0.5 × ATR_Average(50)
4. ✅ Not in news blackout window
5. ✅ Daily loss < 1%

---

## Exit Conditions

| Exit Type | Condition |
|-----------|-----------|
| **Stop Loss** | 2 × ATR from entry |
| **Take Profit** | 3 × ATR from entry (1.5:1 R:R) |
| **Time Exit** | Close after 10 bars if no SL/TP hit |
| **Emergency** | Close all if daily loss reaches 1% |

---

## Risk Management

| Parameter | Value |
|-----------|-------|
| Risk per trade | 0.5% of equity |
| Max daily loss | 1% of equity |
| Position sizing | `(Equity × 0.005) / (2 × ATR)` |
| Max positions | 1 concurrent |

---

## News Blackout Windows (UTC)

- **FOMC days:** 14:00 - 20:00
- **NFP (1st Friday):** 12:00 - 15:00
- **CPI/PPI days:** 12:00 - 15:00

---

## Weaknesses & Failure Scenarios

### Known Weaknesses

1. **Sideways Markets** - Trend-following fails in prolonged ranges
2. **Whipsaws** - Price oscillating around EMA 50 causes false signals
3. **Slippage** - Hourly gaps (especially overnight) may impact fills
4. **News Spikes** - Static filter may miss unscheduled events

### Critical Failure Scenarios

1. Extended bear market with violent counter-trend rallies
2. Flash crashes where stops don't fill at expected price
3. Prolonged low-volatility periods generating no signals

### Mitigations Built-In

- ATR filter reduces trading in dead markets
- RSI exhaustion filter prevents chasing
- Time-based exit limits drawdown from stuck trades
- Daily loss limit prevents catastrophic days

---

## File Structure

```
TradeBot/
├── config.py          # All tunable parameters
├── news_calendar.py   # News blackout logic
├── backtest.py        # Main backtesting engine
├── strategy_doc.md    # This document
└── trade_log.csv      # Generated after backtest
```

---

## Usage

```bash
cd /path/to/TradeBot
python backtest.py
```

This runs the backtest with synthetic data and outputs:
- Performance metrics to console
- Trade log to `trade_log.csv`

For production, replace `generate_sample_data()` with real OHLCV data.
