import numpy as np
import pandas as pd
from typing import Optional

from utils import logger
from .trend_following import StrategySignal, MarketRegime


class MeanReversionStrategy:
    NAME = "mean_reversion"

    def __init__(self, atr_sl_mult: float = 1.2, atr_tp_mult: float = 2.0):
        self.log = logger
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

    def analyze(self, df: pd.DataFrame) -> Optional[StrategySignal]:
        if df.empty or len(df) < 50:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(latest["close"])
        atr = float(latest["atr_14"]) if "atr_14" in df.columns and not np.isnan(latest["atr_14"]) else 0

        if atr <= 0:
            return None

        bb_upper = float(latest["bb_upper"]) if "bb_upper" in df.columns and not np.isnan(latest["bb_upper"]) else None
        bb_lower = float(latest["bb_lower"]) if "bb_lower" in df.columns and not np.isnan(latest["bb_lower"]) else None
        bb_middle = float(latest["bb_middle"]) if "bb_middle" in df.columns and not np.isnan(latest["bb_middle"]) else None
        bb_width = float(latest["bb_width"]) if "bb_width" in df.columns and not np.isnan(latest["bb_width"]) else None
        rsi = float(latest["rsi_14"]) if "rsi_14" in df.columns and not np.isnan(latest["rsi_14"]) else 50
        vwap = float(latest["vwap"]) if "vwap" in df.columns and not np.isnan(latest["vwap"]) else price
        adx = float(latest["adx"]) if "adx" in df.columns and not np.isnan(latest["adx"]) else 0
        vol_ratio = float(latest["volume_ratio"]) if "volume_ratio" in df.columns and not np.isnan(latest["volume_ratio"]) else 1.0
        ema_20 = float(latest["ema_20"]) if "ema_20" in df.columns and not np.isnan(latest["ema_20"]) else price
        ema_50 = float(latest["ema_50"]) if "ema_50" in df.columns and not np.isnan(latest["ema_50"]) else price

        if bb_upper is None or bb_lower is None or bb_middle is None:
            return None

        prev_bb_upper = float(prev["bb_upper"]) if "bb_upper" in prev.index and not np.isnan(prev["bb_upper"]) else bb_upper
        prev_bb_lower = float(prev["bb_lower"]) if "bb_lower" in prev.index and not np.isnan(prev["bb_lower"]) else bb_lower

        avg_bb_width = float(df["bb_width"].tail(20).mean()) if "bb_width" in df.columns else None

        is_ranging = adx < 25 if adx > 0 else True

        bb_position = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        long_score = 0.0
        long_reasons = []

        if price <= bb_lower * 1.002:
            long_score += 0.20
            long_reasons.append("at_lower_band")
        elif bb_position < 0.15:
            long_score += 0.15
            long_reasons.append("near_lower_band")

        if rsi < 30:
            long_score += 0.20
            long_reasons.append(f"rsi_oversold_{rsi:.0f}")
        elif rsi < 40:
            long_score += 0.10
            long_reasons.append(f"rsi_low_{rsi:.0f}")

        if price < vwap:
            long_score += 0.10
            long_reasons.append("below_vwap")

        prev_close = float(prev["close"]) if "close" in prev.index else price
        if price > prev_close and prev_close <= bb_lower * 1.005:
            long_score += 0.15
            long_reasons.append("bounce_from_lower")

        if is_ranging:
            long_score += 0.10
            long_reasons.append("market_ranging")

        if avg_bb_width and bb_width and bb_width < avg_bb_width * 0.7:
            long_score += 0.05
            long_reasons.append("bb_squeeze")

        body = float(latest["close"] - latest["open"])
        lower_wick = float(latest.get("wick_lower", 0))
        if body > 0 and lower_wick > abs(body) * 1.5:
            long_score += 0.10
            long_reasons.append("hammer_rejection")

        if vol_ratio > 1.0:
            long_score += 0.05
            long_reasons.append("volume_support")

        short_score = 0.0
        short_reasons = []

        if price >= bb_upper * 0.998:
            short_score += 0.20
            short_reasons.append("at_upper_band")
        elif bb_position > 0.85:
            short_score += 0.15
            short_reasons.append("near_upper_band")

        if rsi > 70:
            short_score += 0.20
            short_reasons.append(f"rsi_overbought_{rsi:.0f}")
        elif rsi > 60:
            short_score += 0.10
            short_reasons.append(f"rsi_high_{rsi:.0f}")

        if price > vwap:
            short_score += 0.10
            short_reasons.append("above_vwap")

        if price < prev_close and prev_close >= bb_upper * 0.995:
            short_score += 0.15
            short_reasons.append("rejection_from_upper")

        if is_ranging:
            short_score += 0.10
            short_reasons.append("market_ranging")

        if avg_bb_width and bb_width and bb_width < avg_bb_width * 0.7:
            short_score += 0.05
            short_reasons.append("bb_squeeze")

        upper_wick = float(latest.get("wick_upper", 0))
        if body < 0 and upper_wick > abs(body) * 1.5:
            short_score += 0.10
            short_reasons.append("shooting_star_rejection")

        if vol_ratio > 1.0:
            short_score += 0.05
            short_reasons.append("volume_support")

        if long_score > short_score and long_score >= 0.5:
            sl = price - atr * self.atr_sl_mult
            tp = bb_middle if bb_middle > price else price + atr * self.atr_tp_mult
            return StrategySignal(
                strategy_name=self.NAME,
                direction="long",
                confidence=min(long_score, 1.0),
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                atr=atr,
                reasons=long_reasons,
                regime=MarketRegime.RANGING,
            )

        if short_score > long_score and short_score >= 0.5:
            sl = price + atr * self.atr_sl_mult
            tp = bb_middle if bb_middle < price else price - atr * self.atr_tp_mult
            return StrategySignal(
                strategy_name=self.NAME,
                direction="short",
                confidence=min(short_score, 1.0),
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                atr=atr,
                reasons=short_reasons,
                regime=MarketRegime.RANGING,
            )

        return None
