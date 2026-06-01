import numpy as np
import pandas as pd
from typing import Tuple


class VolatilityIndicators:
    @staticmethod
    def historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
        log_returns = np.log(series / series.shift(1))
        return log_returns.rolling(window=period).std() * np.sqrt(252)

    @staticmethod
    def keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
        from .technical import TechnicalIndicators
        middle = TechnicalIndicators.ema(df["close"], period)
        atr = TechnicalIndicators.atr(df, period)
        upper = middle + atr_mult * atr
        lower = middle - atr_mult * atr
        return upper, middle, lower

    @staticmethod
    def volatility_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        atr_short = df["close"].diff().abs().rolling(window=5).mean()
        atr_long = df["close"].diff().abs().rolling(window=period).mean()
        return atr_short / atr_long.replace(0, np.nan)

    @staticmethod
    def volatility_regime(df: pd.DataFrame, period: int = 20) -> pd.Series:
        atr = df["close"].diff().abs().rolling(window=period).mean()
        atr_ma = atr.rolling(window=period * 3).mean()
        regime = pd.Series(index=df.index, data="normal")
        regime[atr > atr_ma * 1.5] = "high"
        regime[atr < atr_ma * 0.5] = "low"
        return regime

    @staticmethod
    def z_score_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        log_returns = np.log(df["close"] / df["close"].shift(1))
        rolling_std = log_returns.rolling(window=period).std()
        mean_std = rolling_std.mean()
        std_std = rolling_std.std()
        return (rolling_std - mean_std) / std_std.replace(0, np.nan)

    @staticmethod
    def classify_volatility(z_score: float) -> str:
        if z_score > 2.0:
            return "extremely_high"
        elif z_score > 1.5:
            return "very_high"
        elif z_score > 1.0:
            return "elevated"
        elif z_score < -1.0:
            return "low"
        elif z_score < -1.5:
            return "very_low"
        return "normal"
