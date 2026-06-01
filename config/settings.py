import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class TradingPair(Enum):
    XAUUSD = "XAU/USD"
    XBTUSD = "XBT/USD"
    XAUTUSD = "XAUT/USD"

    @property
    def kraken_pair(self) -> str:
        return self.value

    @property
    def kraken_ws_name(self) -> str:
        pair_map = {
            "XAU/USD": "XAUT/USD",
            "XBT/USD": "XBT/USD",
            "XAUT/USD": "XAUT/USD",
        }
        return pair_map.get(self.value, self.value)


@dataclass
class KrakenConfig:
    api_key: str = os.getenv("KRAKEN_API_KEY", "")
    api_secret: str = os.getenv("KRAKEN_API_SECRET", "")
    ws_url: str = "wss://ws.kraken.com/v2"
    rest_url: str = "https://api.kraken.com"
    pair: str = "XAUT/USD"
    timeframe: str = "15m"
    candle_limit: int = 500
    ws_heartbeat: int = 30


@dataclass
class MT5Config:
    login: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "")
    symbol: str = "XAUTUSD"
    magic_number: int = 2026
    deviation: int = 20
    fill_type: str = "IOC"
    enabled: bool = os.getenv("MT5_ENABLED", "false").lower() == "true"


@dataclass
class TradingConfig:
    symbol: str = "XAUT/USD"
    timeframe: str = "15m"
    risk_per_trade: float = 0.01
    max_drawdown: float = 0.10
    max_open_trades: int = 3
    default_sl_atr_mult: float = 1.5
    default_tp_atr_mult: float = 3.0
    trailing_stop_atr: float = 1.0
    min_rr_ratio: float = 2.0
    capital: float = float(os.getenv("TRADING_CAPITAL", "100000"))
    trading_hours: dict = field(default_factory=lambda: {
        "london_open": 8,
        "london_close": 16,
        "ny_open": 13,
        "ny_close": 21,
    })
    blocked_hours_utc: list = field(default_factory=lambda: [22, 23, 0, 1, 2, 3, 4, 5])


@dataclass
class StrategyConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_buy_threshold: float = 55.0
    rsi_sell_threshold: float = 45.0
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_spike_mult: float = 1.5
    breakout_lookback: int = 10
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    vwap_std_mult: float = 1.0


@dataclass
class MultiStrategyConfig:
    enabled: bool = True
    confidence_threshold: float = 0.55
    trend_atr_sl: float = 1.5
    trend_atr_tp: float = 3.0
    mr_atr_sl: float = 1.2
    mr_atr_tp: float = 2.0
    breakout_atr_sl: float = 1.5
    breakout_atr_tp: float = 3.0
    min_strategies_agree: int = 1


@dataclass
class MLConfig:
    model_path: str = "models/xgboost_xaut.json"
    feature_columns: list = field(default_factory=lambda: [
        "rsi", "ema_distance", "atr_ratio", "volume_ratio",
        "bb_position", "macd_hist", "candle_body_ratio",
        "wick_ratio", "adx", "trend_strength", "momentum_5",
        "momentum_10", "volatility_20", "vwap_deviation",
        "session_london", "session_ny", "hour_sin", "hour_cos",
    ])
    min_confidence: float = 0.70
    retrain_interval_hours: int = 24
    lookback_training: int = 5000
    walk_forward_windows: int = 5
    enabled: bool = True
    use_gpu: bool = False
    feature_window: int = 50


@dataclass
class TelegramConfig:
    bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    alert_chat_id: str = os.getenv("TELEGRAM_ALERT_CHAT_ID", "")
    send_signals: bool = True
    send_errors: bool = True
    send_daily_report: bool = True
    enabled: bool = False

    def __post_init__(self):
        if self.bot_token and self.chat_id:
            self.enabled = True


@dataclass
class DatabaseConfig:
    postgres_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://trading:trading123@localhost:5432/trading_bot"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    candle_table: str = "candles_xaut_usd"
    trades_table: str = "trades"
    signals_table: str = "signals"


@dataclass
class BacktestConfig:
    start_date: str = "2025-01-01"
    end_date: str = "2026-05-31"
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0001
    mc_simulations: int = 1000


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01
    max_daily_risk: float = 0.05
    max_open_trades: int = 3
    min_risk_reward: float = 1.5
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 3.0
    break_even_trigger: float = 1.0
    trailing_activation: float = 1.5
    partial_take_profits: list = field(default_factory=lambda: [0.3, 0.3, 0.4])
    partial_tp_levels: list = field(default_factory=lambda: [1.0, 2.0, 3.0])


@dataclass
class WebSocketConfig:
    url: str = "wss://ws.kraken.com/v2"
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10


@dataclass
class DashboardConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8050
    streamlit_port: int = 8501
    streamlit_enabled: bool = True


@dataclass
class Settings:
    kraken: KrakenConfig = field(default_factory=KrakenConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    multi_strategy: MultiStrategyConfig = field(default_factory=MultiStrategyConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ws: WebSocketConfig = field(default_factory=WebSocketConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    log_level: str = "INFO"
    log_file: str = "logs/trading_bot.log"
    paper_trading: bool = True

    @property
    def pairs(self):
        return [TradingPair(self.trading.symbol)]

    @property
    def timeframe(self):
        return self._TimeframeProxy(self.trading.timeframe)

    class _TimeframeProxy:
        def __init__(self, primary: str):
            self.primary = primary
            self.confirmation = "1m"
            self.higher = "15m"


config = Settings()
