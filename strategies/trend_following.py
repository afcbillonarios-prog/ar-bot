from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

import pandas as pd
import numpy as np

from utils import logger


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass
class StrategySignal:
    strategy_name: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    reasons: List[str] = field(default_factory=list)
    regime: MarketRegime = MarketRegime.UNKNOWN

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    @property
    def is_short(self) -> bool:
        return self.direction == "short"

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0


class TrendFollowingStrategy:
    NAME = "trend_following"

    def __init__(self, atr_sl_mult: float = 1.5, atr_tp_mult: float = 3.0):
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

        ema_20 = float(latest["ema_20"]) if "ema_20" in df.columns and not np.isnan(latest["ema_20"]) else None
        ema_50 = float(latest["ema_50"]) if "ema_50" in df.columns and not np.isnan(latest["ema_50"]) else None
        rsi = float(latest["rsi_14"]) if "rsi_14" in df.columns and not np.isnan(latest["rsi_14"]) else 50
        adx = float(latest["adx"]) if "adx" in df.columns and not np.isnan(latest["adx"]) else 0
        di_plus = float(latest["di_plus"]) if "di_plus" in df.columns and not np.isnan(latest["di_plus"]) else 0
        di_minus = float(latest["di_minus"]) if "di_minus" in df.columns and not np.isnan(latest["di_minus"]) else 0
        vol_ratio = float(latest["volume_ratio"]) if "volume_ratio" in df.columns and not np.isnan(latest["volume_ratio"]) else 1.0

        prev_ema_20 = float(prev["ema_20"]) if "ema_20" in prev.index and not np.isnan(prev["ema_20"]) else None
        prev_ema_50 = float(prev["ema_50"]) if "ema_50" in prev.index and not np.isnan(prev["ema_50"]) else None

        if ema_20 is None or ema_50 is None:
            return None

        bb_upper = float(latest["bb_upper"]) if "bb_upper" in df.columns and not np.isnan(latest["bb_upper"]) else price
        bb_lower = float(latest["bb_lower"]) if "bb_lower" in df.columns and not np.isnan(latest["bb_lower"]) else price

        long_score = 0.0
        long_reasons = []

        if ema_20 > ema_50:
            long_score += 0.15
            long_reasons.append("ema20_above_ema50")

        if adx > 25:
            long_score += 0.15
            long_reasons.append(f"adx_strong_{adx:.0f}")

        if di_plus > di_minus:
            long_score += 0.10
            long_reasons.append("di_plus_dominant")

        if rsi > 50 and rsi < 75:
            long_score += 0.10
            long_reasons.append(f"rsi_bullish_{rsi:.0f}")
        elif rsi > 45:
            long_score += 0.05
            long_reasons.append(f"rsi_neutral_{rsi:.0f}")

        if prev_ema_20 is not None and prev_ema_50 is not None:
            if prev_ema_20 <= prev_ema_50 and ema_20 > ema_50:
                long_score += 0.15
                long_reasons.append("ema_crossover")

        pullback_to_ema = (price - ema_20) / ema_20 * 100
        if -0.5 <= pullback_to_ema <= 0.3 and ema_20 > ema_50:
            long_score += 0.10
            long_reasons.append("pullback_to_ema20")

        if vol_ratio > 1.2:
            long_score += 0.05
            long_reasons.append("volume_confirm")

        price_above_vwap = price > float(latest.get("vwap", price)) if "vwap" in df.columns else False
        if price_above_vwap:
            long_score += 0.05
            long_reasons.append("above_vwap")

        long_body = float(latest["close"] - latest["open"]) if "close" in latest.index and "open" in latest.index else 0
        if long_body > 0 and long_body > atr * 0.3:
            long_score += 0.05
            long_reasons.append("strong_bullish_candle")

        short_score = 0.0
        short_reasons = []

        if ema_20 < ema_50:
            short_score += 0.15
            short_reasons.append("ema20_below_ema50")

        if adx > 25:
            short_score += 0.15
            short_reasons.append(f"adx_strong_{adx:.0f}")

        if di_minus > di_plus:
            short_score += 0.10
            short_reasons.append("di_minus_dominant")

        if rsi < 50 and rsi > 25:
            short_score += 0.10
            short_reasons.append(f"rsi_bearish_{rsi:.0f}")
        elif rsi < 55:
            short_score += 0.05
            short_reasons.append(f"rsi_neutral_{rsi:.0f}")

        if prev_ema_20 is not None and prev_ema_50 is not None:
            if prev_ema_20 >= prev_ema_50 and ema_20 < ema_50:
                short_score += 0.15
                short_reasons.append("ema_crossunder")

        pullback_to_ema_short = (ema_20 - price) / ema_20 * 100
        if -0.5 <= pullback_to_ema_short <= 0.3 and ema_20 < ema_50:
            short_score += 0.10
            short_reasons.append("pullback_to_ema20_short")

        if vol_ratio > 1.2:
            short_score += 0.05
            short_reasons.append("volume_confirm")

        if not price_above_vwap:
            short_score += 0.05
            short_reasons.append("below_vwap")

        short_body = float(latest["close"] - latest["open"]) if "close" in latest.index and "open" in latest.index else 0
        if short_body < 0 and abs(short_body) > atr * 0.3:
            short_score += 0.05
            short_reasons.append("strong_bearish_candle")

        if long_score > short_score and long_score >= 0.5:
            sl = price - atr * self.atr_sl_mult
            tp = price + atr * self.atr_tp_mult
            return StrategySignal(
                strategy_name=self.NAME,
                direction="long",
                confidence=min(long_score, 1.0),
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                atr=atr,
                reasons=long_reasons,
                regime=MarketRegime.TRENDING_UP,
            )

        if short_score > long_score and short_score >= 0.5:
            sl = price + atr * self.atr_sl_mult
            tp = price - atr * self.atr_tp_mult
            return StrategySignal(
                strategy_name=self.NAME,
                direction="short",
                confidence=min(short_score, 1.0),
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                atr=atr,
                reasons=short_reasons,
                regime=MarketRegime.TRENDING_DOWN,
            )

        return None
