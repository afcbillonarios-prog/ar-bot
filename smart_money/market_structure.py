import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StructureBreak(Enum):
    BOS = "break_of_structure"
    CHOCH = "change_of_character"
    NONE = "none"


@dataclass
class MarketStructurePoint:
    index: int
    price: float
    timestamp: float
    point_type: str

    @property
    def is_high(self) -> bool:
        return self.point_type == "high"

    @property
    def is_low(self) -> bool:
        return self.point_type == "low"


class MarketStructure:
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self._swing_highs: List[MarketStructurePoint] = []
        self._swing_lows: List[MarketStructurePoint] = []
        self._last_bos: Optional[StructureBreak] = None
        self._last_choch: Optional[StructureBreak] = None
        self._trend: TrendDirection = TrendDirection.NEUTRAL

    def _find_pivot_points(self, df: pd.DataFrame) -> Tuple[List[MarketStructurePoint], List[MarketStructurePoint]]:
        highs = df["high"].values
        lows = df["low"].values
        timestamps = df["timestamp"].values if "timestamp" in df.columns else df.index.values

        swing_highs, swing_lows = [], []
        window = 2

        for i in range(window, len(df) - window):
            if all(highs[i] > highs[i - j] and highs[i] > highs[i + j] for j in range(1, window + 1)):
                swing_highs.append(MarketStructurePoint(
                    index=i, price=highs[i], timestamp=timestamps[i], point_type="high"
                ))

            if all(lows[i] < lows[i - j] and lows[i] < lows[i + j] for j in range(1, window + 1)):
                swing_lows.append(MarketStructurePoint(
                    index=i, price=lows[i], timestamp=timestamps[i], point_type="low"
                ))

        return swing_highs, swing_lows

    def detect_bos(self, df: pd.DataFrame) -> Tuple[Optional[StructureBreak], Optional[float]]:
        swing_highs, swing_lows = self._find_pivot_points(df)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return StructureBreak.NONE, None

        last_close = df["close"].iloc[-1]
        result = StructureBreak.NONE
        target_price = None

        if self._trend in (TrendDirection.BULLISH, TrendDirection.NEUTRAL):
            prev_high = swing_highs[-2].price if len(swing_highs) >= 2 else None
            if prev_high and last_close > prev_high:
                result = StructureBreak.BOS
                target_price = prev_high

        if self._trend in (TrendDirection.BEARISH, TrendDirection.NEUTRAL):
            prev_low = swing_lows[-2].price if len(swing_lows) >= 2 else None
            if prev_low and last_close < prev_low:
                result = StructureBreak.BOS
                target_price = prev_low

        self._last_bos = result
        return result, target_price

    def detect_choch(self, df: pd.DataFrame) -> Tuple[Optional[StructureBreak], Optional[float]]:
        swing_highs, swing_lows = self._find_pivot_points(df)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return StructureBreak.NONE, None

        last_close = df["close"].iloc[-1]
        result = StructureBreak.NONE
        target_price = None

        if self._trend == TrendDirection.BULLISH:
            if swing_lows and swing_lows[-1].price < swing_lows[-2].price:
                if last_close < swing_lows[-1].price:
                    result = StructureBreak.CHOCH
                    target_price = swing_lows[-2].price

        elif self._trend == TrendDirection.BEARISH:
            if swing_highs and swing_highs[-1].price > swing_highs[-2].price:
                if last_close > swing_highs[-1].price:
                    result = StructureBreak.CHOCH
                    target_price = swing_highs[-2].price

        self._last_choch = result
        return result, target_price

    def update_trend(self, df: pd.DataFrame):
        if len(df) < 50:
            return

        ema_20 = df["close"].rolling(20).mean().iloc[-1]
        ema_50 = df["close"].rolling(50).mean().iloc[-1]
        ema_200 = df["close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else None

        if ema_20 > ema_50:
            self._trend = TrendDirection.BULLISH
        elif ema_20 < ema_50:
            self._trend = TrendDirection.BEARISH
        else:
            self._trend = TrendDirection.NEUTRAL

    def get_higher_highs_lows(self) -> bool:
        if len(self._swing_highs) < 2 or len(self._swing_lows) < 2:
            return False
        return (self._swing_highs[-1].price > self._swing_highs[-2].price and
                self._swing_lows[-1].price > self._swing_lows[-2].price)

    def get_lower_highs_lows(self) -> bool:
        if len(self._swing_highs) < 2 or len(self._swing_lows) < 2:
            return False
        return (self._swing_highs[-1].price < self._swing_highs[-2].price and
                self._swing_lows[-1].price < self._swing_lows[-2].price)

    @property
    def trend(self) -> TrendDirection:
        return self._trend

    @property
    def last_bos(self) -> Optional[StructureBreak]:
        return self._last_bos

    @property
    def last_choch(self) -> Optional[StructureBreak]:
        return self._last_choch

    @property
    def swing_highs(self) -> List[MarketStructurePoint]:
        return self._swing_highs

    @property
    def swing_lows(self) -> List[MarketStructurePoint]:
        return self._swing_lows
