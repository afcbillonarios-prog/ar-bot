# Scalping Bot — XAU/USD & BTC/USD

> Institutional-grade scalping bot combining **Smart Money Concepts (SMC)**, **Machine Learning (XGBoost)**, real-time **Kraken WebSocket** data, and a professional **Plotly Dash** dashboard.

---

## Architecture

```
bot/
├── config/          # Settings, environment, trading pairs
├── data/            # Real-time data manager & cache
├── websocket/       # Kraken WebSocket client (async)
├── indicators/      # Technical indicators (EMA, RSI, ATR, VWAP, BB, MACD)
├── smart_money/     # SMC: BOS, CHOCH, FVG, Order Blocks, Liquidity, Imbalance
├── ml/              # XGBoost model, feature engineering, predictor
├── risk/            # Position sizing, SL/TP, break-even, trailing stop
├── signals/         # Signal generator & multi-filter system
├── execution/       # Trade execution & order management
├── dashboard/       # Plotly Dash real-time dashboard
├── backtesting/     # Backtest engine, metrics, Monte Carlo simulation
├── alerts/          # Telegram notifications
├── utils/           # Logging, helpers
├── main.py          # Entry point
├── Dockerfile       # Containerization
└── docker-compose.yml
```

---

## Features

### Smart Money Concepts
- **Break of Structure (BOS)** — Detects bullish/bearish structure breaks
- **Change of Character (CHOCH)** — Identifies trend reversal signals
- **Fair Value Gaps (FVG)** — Detects unfilled price gaps
- **Order Blocks** — Institutional order flow zones with strength scoring
- **Liquidity Sweeps** — Buy-side / sell-side liquidity grabs
- **Equal Highs/Lows** — Double top/bottom detection
- **Imbalances** — Volume + body ratio anomaly detection
- **Market Structure** — Swing high/low pivot analysis

### Entry Conditions (LONG)
| Condition | Weight |
|-----------|--------|
| Bullish trend (EMA20 > EMA50) | +0.15 |
| Bearish liquidity sweep | +0.20 |
| Bullish BOS confirmation | +0.15 |
| Price at Order Block | +0.15 |
| Bullish FVG confirmation | +0.10 |
| Strong volume (ratio > 1.2) | +0.05 |
| RSI between 30-70 | +0.05 |
| ML confidence > 75% | Filter |

### Entry Conditions (SHORT)
| Condition | Weight |
|-----------|--------|
| Bearish trend (EMA20 < EMA50) | +0.15 |
| Bullish liquidity sweep | +0.20 |
| CHOCH bearish confirmation | +0.15 |
| Bearish BOS | +0.10 |
| Price at Order Block | +0.15 |
| Bearish FVG | +0.10 |
| Volume confirmation | +0.05 |
| ML confidence > 75% | Filter |

### Machine Learning (XGBoost)
- **Features**: RSI, ATR, VWAP distance, EMA spread, momentum, BB width, volume ratio, volatility, MACD, price position, FVG size, BOS direction, OB proximity
- **Classification**: TP hit (1) vs SL hit (-1) vs neutral (0)
- **Confidence threshold**: 75%
- **Auto-retrain**: Every 24 hours
- **Feature importance tracking**

### Risk Management
| Parameter | Value |
|-----------|-------|
| Max risk per trade | 1% |
| Max daily risk | 5% |
| Max open trades | 3 |
| Min risk/reward | 1.5 |
| SL multiplier (ATR) | 1.5x |
| TP multiplier (ATR) | 3.0x |
| Break-even trigger | 1.0 ATR |
| Trailing activation | 1.5 ATR |

### Trailing Stop
- Follows swing highs/lows
- Dynamic distance based on ATR
- Automatic break-even
- Partial profit taking (30%/30%/40%)

### Dashboard (Plotly Dash)
- Real-time candlestick chart with EMAs
- Order Blocks, FVGs, liquidity zones overlaid
- ML confidence bar chart
- Volume profile
- Equity curve + drawdown
- Smart Money Analysis panel
- Recent trades table
- Live stats (PnL, win rate, active trades)

### Backtesting
- Vectorized backtest engine
- Metrics: Win Rate, Profit Factor, Sharpe, Sortino, Calmar, Max DD
- Monte Carlo simulation (1000 runs)
- VaR 95%/99% calculation
- Risk of ruin analysis

---

## Quick Start

### 1. Clone & Setup
```bash
git clone <repo> scalping-bot
cd scalping-bot
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure
```bash
cp config/.env.example config/.env
# Edit config/.env with your credentials
```

### 3. Run
```bash
python main.py
```

### 4. Dashboard
Open `http://localhost:8050`

### Docker
```bash
docker-compose up -d --build
```

---

## Configuration

Edit `config/.env`:
```env
KRAKEN_API_KEY=your_key
KRAKEN_SECRET_KEY=your_secret
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
LOG_LEVEL=INFO
MAX_RISK_PER_TRADE=0.01
ML_ENABLED=true
ML_CONFIDENCE_THRESHOLD=0.75
```

---

## Performance Metrics

After backtesting, analyze:
| Metric | Target |
|--------|--------|
| Win Rate | > 55% |
| Profit Factor | > 1.5 |
| Sharpe Ratio | > 1.5 |
| Max Drawdown | < 15% |
| Expectancy | Positive |

---

## Roadmap

1. **Phase 1** — Core architecture, WebSocket, indicators, candle processing
2. **Phase 2** — Smart Money Concepts (BOS, CHOCH, FVG, OB, liquidity)
3. **Phase 3** — ML pipeline (features, XGBoost training, prediction)
4. **Phase 4** — Risk management, position sizing, SL/TP, trailing
5. **Phase 5** — Signal generation, entry/exit logic, filters
6. **Phase 6** — Dashboard, real-time charts, SMC visualization
7. **Phase 7** — Backtesting, Monte Carlo, performance analytics
8. **Phase 8** — Telegram alerts, Docker, production hardening

---

## License
MIT
