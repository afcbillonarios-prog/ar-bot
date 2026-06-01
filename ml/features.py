import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from indicators import TechnicalIndicators as TI
from utils import logger


class FeatureEngine:
    def __init__(self, window: int = 50):
        self.window = window
        self.log = logger
        self._feature_names: List[str] = []
        self._last_features: Optional[pd.DataFrame] = None

    def compute_features(self, df: pd.DataFrame, smc_analysis=None) -> pd.DataFrame:
        if df.empty or len(df) < self.window:
            return pd.DataFrame()

        df = TI.apply_all(df)
        features = pd.DataFrame(index=df.index)

        features["rsi"] = df["rsi_14"]
        features["atr"] = df["atr_14"]
        features["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
        features["ema_spread"] = df["ema_spread_20_50"]
        features["momentum"] = df["momentum_10"]
        features["bb_width"] = df["bb_width"]
        features["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        features["volume_ratio"] = df["volume_ratio"]

        features["volatility_5"] = df["close"].pct_change().rolling(5).std()
        features["volatility_10"] = df["close"].pct_change().rolling(10).std()
        features["volatility_ratio"] = features["volatility_5"] / features["volatility_10"].replace(0, np.nan)

        features["close_ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
        features["close_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
        features["ema_cross"] = ((df["ema_20"] - df["ema_50"]) / df["ema_50"]) * 100

        features["macd_value"] = df["macd"]
        features["macd_signal"] = df["macd_signal"]
        features["macd_histogram"] = df["macd_hist"]
        features["macd_cross"] = ((df["macd"] - df["macd_signal"]) / df["macd_signal"].replace(0, np.nan)).clip(-5, 5)

        features["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        features["close_open_range"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
        features["body_to_wick"] = (df["close"] - df["open"]).abs() / ((df["high"] - df["low"]) - (df["close"] - df["open"]).abs() + 1e-10)

        features["volume_momentum"] = df["volume"].pct_change(3)
        features["dollar_volume"] = df["volume"] * df["close"]
        features["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(10).mean().replace(0, np.nan)

        features["price_position_20"] = (df["close"] - df["low"].rolling(20).min()) / (df["high"].rolling(20).max() - df["low"].rolling(20).min() + 1e-10)
        features["price_position_50"] = (df["close"] - df["low"].rolling(50).min()) / (df["high"].rolling(50).max() - df["low"].rolling(50).min() + 1e-10)

        if smc_analysis is not None:
            features["adx"] = df["adx"]
            features["trend_strength"] = df.get("trend_strength", pd.Series(0.0, index=df.index))
            features["fvg_size"] = self._extract_fvg_size(smc_analysis, df)
            features["bos_direction"] = self._extract_bos_direction(smc_analysis)
            features["ob_proximity"] = self._extract_ob_proximity(smc_analysis, df)
        else:
            features["adx"] = df["adx"]
            features["trend_strength"] = df.get("trend_strength", pd.Series(0.0, index=df.index))
            features["fvg_size"] = 0.0
            features["bos_direction"] = 0
            features["ob_proximity"] = 0.0

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)

        self._feature_names = features.columns.tolist()
        self._last_features = features

        return features

    def _extract_fvg_size(self, smc_analysis, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(0.0, index=df.index)
        if hasattr(smc_analysis, "active_fvgs") and smc_analysis.active_fvgs:
            last_idx = df.index[-1]
            for fvg in smc_analysis.active_fvgs:
                if hasattr(fvg, "size"):
                    result.iloc[-1] = max(result.iloc[-1], fvg.size)
        return result

    def _extract_bos_direction(self, smc_analysis) -> pd.Series:
        result = pd.Series(0, index=range(1))
        if hasattr(smc_analysis, "bos"):
            bos_val = smc_analysis.bos
            if hasattr(bos_val, "value"):
                if bos_val.value == "BOS":
                    result.iloc[-1] = 1
                elif bos_val.value == "CHOCH":
                    result.iloc[-1] = -1
        return result

    def _extract_ob_proximity(self, smc_analysis, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(0.0, index=df.index)
        if hasattr(smc_analysis, "nearest_bullish_ob") and smc_analysis.nearest_bullish_ob:
            price = df["close"].iloc[-1]
            ob = smc_analysis.nearest_bullish_ob
            if hasattr(ob, "low") and hasattr(ob, "high"):
                if ob.low <= price <= ob.high:
                    result.iloc[-1] = 1.0
        if hasattr(smc_analysis, "nearest_bearish_ob") and smc_analysis.nearest_bearish_ob:
            price = df["close"].iloc[-1]
            ob = smc_analysis.nearest_bearish_ob
            if hasattr(ob, "low") and hasattr(ob, "high"):
                if ob.low <= price <= ob.high:
                    result.iloc[-1] = -1.0
        return result

    def get_feature_vector(self, df: pd.DataFrame, smc_analysis=None) -> np.ndarray:
        features = self.compute_features(df, smc_analysis)
        if features.empty:
            return np.array([])
        return features.iloc[-1].values.reshape(1, -1)

    def create_labels(self, df: pd.DataFrame, tp_multiplier: float = 2.0, sl_multiplier: float = 1.0, lookahead_bars: int = 20) -> pd.Series:
        if df.empty or "atr_14" not in df.columns:
            return pd.Series(dtype=int)

        labels = pd.Series(0, index=df.index, dtype=int)
        atr = df["atr_14"].ffill()
        close = df["close"]

        for i in range(len(df) - 1):
            tp_distance = atr.iloc[i] * tp_multiplier
            sl_distance = atr.iloc[i] * sl_multiplier
            tp_long = close.iloc[i] + tp_distance
            sl_long = close.iloc[i] - sl_distance
            tp_short = close.iloc[i] - tp_distance
            sl_short = close.iloc[i] + sl_distance

            lookahead = min(i + lookahead_bars, len(df))
            future_prices = close.iloc[i + 1: lookahead + 1]

            if len(future_prices) > 0:
                high_reached = future_prices.max()
                low_reached = future_prices.min()

                if high_reached >= tp_long and low_reached <= sl_long:
                    first_tp_idx = (future_prices >= tp_long).idxmax()
                    first_sl_idx = (future_prices <= sl_long).idxmax()
                    labels.iloc[i] = 1 if first_tp_idx < first_sl_idx else -1
                elif high_reached >= tp_long:
                    labels.iloc[i] = 1
                elif low_reached <= sl_long:
                    labels.iloc[i] = -1

        return labels

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)
