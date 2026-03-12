# TradeBot — ML-Enhanced NASDAQ Trading System v1.0

## Overview
ML-driven NASDAQ H1 trend-following system with dual LONG/SHORT LightGBM models, 
Kelly criterion position sizing, news risk scoring, and drawdown control.

---

## Architecture

```
ML Models (Offense)     →  Signal + Probability
Backend Rules (Defense) →  Guardrails + Risk Management
```

### ML Layer
- **Dual Models:** Separate LONG/SHORT LightGBM + LSTM stacking
- **Features:** 33 selected from 50+ (multi-TF: H1, M15, H4, D1)
- **Sizing:** Kelly criterion (quarter-Kelly, confidence-based)
- **Retrain:** Rolling 3-year window, Optuna tuned

### Rule-Based Layer
| Rule | Purpose |
|------|---------|
| EMA 50 | Trend direction filter |
| ADX > 25 | Trend strength requirement |
| ATR chop filter | Skip low-volatility markets |
| RSI range | Prevent exhaustion entries |
| News risk score | Block/reduce around events |
| DD control | 5% reduce, 10% halt |
| Daily loss limit | 1% daily max |

### Exit Rules
| Type | Value |
|------|-------|
| Stop Loss | 2 × ATR |
| Take Profit | 3 × ATR (3:2 R:R) |
| Trailing Stop | 2.5 × ATR |
| Time Exit | 10 bars max |

---

## Performance (OOS 2022-2026)

| Metric | Value |
|--------|-------|
| Win Rate | 58.7% |
| Profit Factor | 3.34 |
| Max Drawdown | -2.31% |
| Avg trades/day | 2.3 |
| LONG WR | 62% |
| SHORT WR | 56% |

---

## File Structure

```
TradeBot/
├── config.py              # All parameters
├── backtest.py            # Backtesting engine
├── news_manager.py        # News risk scoring
├── ai_interface.py        # AI interface (future)
├── data/                  # Market data + news events
├── ml/
│   ├── data_loader.py     # Multi-TF data loader
│   ├── feature_engine.py  # 33-feature engineering
│   ├── trainer.py         # V3 dual model trainer
│   ├── predictor.py       # Inference + Kelly sizing
│   ├── retrain.py         # Rolling retrain + bias test
│   └── paper_trader.py    # Paper trading logger
├── tests/
└── .gitignore
```
