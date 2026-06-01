import numpy as np
import pandas as pd
import ta


class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_all(self, cfg=None) -> pd.DataFrame:
        if cfg is None:
            from config.settings import config
            cfg = config.strategy

        df = self.df

        df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=cfg.ema_fast)
        df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=cfg.ema_slow)
        df["rsi"] = ta.momentum.rsi(df["close"], window=cfg.rsi_period)

        macd = ta.trend.MACD(df["close"], window_slow=cfg.macd_slow,
                              window_fast=cfg.macd_fast, window_sign=cfg.macd_signal)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(df["close"], window=cfg.bb_period, window_dev=cfg.bb_std)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"],
                                                      window=cfg.atr_period)
        df["atr_ratio"] = df["atr"] / df["close"]

        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=cfg.adx_period)
        df["di_plus"] = ta.trend.adx_pos(df["high"], df["low"], df["close"], window=cfg.adx_period)
        df["di_minus"] = ta.trend.adx_neg(df["high"], df["low"], df["close"], window=cfg.atr_period)

        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        df["volume_spike"] = df["volume_ratio"] > cfg.volume_spike_mult

        df["vwap"] = self._compute_vwap(df)
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"]

        df["ema_distance"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]
        df["trend_strength"] = df["ema_distance"].abs() * df["adx"] / 100

        df["candle_body"] = df["close"] - df["open"]
        df["candle_range"] = df["high"] - df["low"]
        df["candle_body_ratio"] = np.where(
            df["candle_range"] > 0,
            np.abs(df["candle_body"]) / df["candle_range"],
            0,
        )
        df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["wick_ratio"] = np.where(
            df["candle_range"] > 0,
            (df["wick_upper"] + df["wick_lower"]) / df["candle_range"],
            0,
        )

        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)
        df["momentum_20"] = df["close"].pct_change(20)
        df["volatility_20"] = df["close"].pct_change().rolling(20).std()

        df["highest_10"] = df["high"].rolling(10).max()
        df["lowest_10"] = df["low"].rolling(10).min()
        df["breakout_up"] = df["close"] > df["highest_10"].shift(1)
        df["breakout_down"] = df["close"] < df["lowest_10"].shift(1)

        df["stoch_rsi"] = ta.momentum.stochrsi(df["close"], window=cfg.rsi_period)

        if "time" in df.columns:
            try:
                df["hour"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.hour
            except Exception:
                df["hour"] = 12
        else:
            df["hour"] = 12
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["session_london"] = df["hour"].between(8, 16).astype(int)
        df["session_ny"] = df["hour"].between(13, 21).astype(int)

        df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

        self.df = df
        return df

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        return cumulative_tp_vol / cumulative_vol

    def get_feature_columns(self) -> list:
        from config.settings import config
        return config.ml.feature_columns

    def prepare_ml_features(self) -> pd.DataFrame:
        features = self.get_feature_columns()
        available = [f for f in features if f in self.df.columns]
        return self.df[available + ["target"]].dropna()

    @staticmethod
    def apply_all(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 5:
            return df

        df = df.copy()

        df["ema_8"] = ta.trend.ema_indicator(df["close"], window=8)
        df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
        df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)

        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

        macd_obj = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()

        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

        df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
        df["volume_spike"] = df["volume_ratio"] > 1.5

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum().replace(0, np.nan)
        df["vwap"] = cum_tp_vol / cum_vol

        df["ema_spread_20_50"] = ((df["ema_20"] - df["ema_50"]) / df["close"]) * 100

        df["momentum_10"] = df["close"].pct_change(10)

        df["candle_body"] = df["close"] - df["open"]
        df["candle_range"] = df["high"] - df["low"]
        df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]

        df["highest_10"] = df["high"].rolling(10).max()
        df["lowest_10"] = df["low"].rolling(10).min()
        df["breakout_up"] = df["close"] > df["highest_10"].shift(1)
        df["breakout_down"] = df["close"] < df["lowest_10"].shift(1)

        df["highest_20"] = df["high"].rolling(20).max()
        df["lowest_20"] = df["low"].rolling(20).min()

        if "time" in df.columns:
            try:
                dt_col = pd.to_datetime(df["time"], unit="s", utc=True)
                df["hour"] = dt_col.dt.hour
            except Exception:
                df["hour"] = 12
        else:
            df["hour"] = 12

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["session_london"] = df["hour"].between(8, 16).astype(int)
        df["session_ny"] = df["hour"].between(13, 21).astype(int)

        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
        df["di_plus"] = ta.trend.adx_pos(df["high"], df["low"], df["close"], window=14)
        df["di_minus"] = ta.trend.adx_neg(df["high"], df["low"], df["close"], window=14)

        df["stoch_rsi"] = ta.momentum.stochrsi(df["close"], window=14)

        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_20"] = df["close"].pct_change(20)
        df["volatility_20"] = df["close"].pct_change().rolling(20).std()

        df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

        return df
