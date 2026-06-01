import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class OrderBlock:
    start_idx: int
    end_idx: int
    high: float
    low: float
    direction: str
    strength: int
    timestamp: float
    mitigated: bool = False

    def __contains__(self, price: float) -> bool:
        return self.low <= price <= self.high

    def mitigation_ratio(self, price: float) -> float:
        if self.high == self.low:
            return 0.0
        if self.direction == "bullish":
            return (price - self.low) / (self.high - self.low)
        else:
            return (self.high - price) / (self.high - self.low)


class OrderBlockDetector:
    def __init__(self, min_strength: int = 2, lookback: int = 100):
        self.min_strength = min_strength
        self.lookback = lookback
        self._order_blocks: List[OrderBlock] = []

    def detect(self, df: pd.DataFrame) -> List[OrderBlock]:
        self._order_blocks = []
        data = df.tail(self.lookback).reset_index(drop=True)

        if len(data) < 10:
            return self._order_blocks

        for i in range(3, len(data) - 1):
            current = data.iloc[i]
            prev = data.iloc[i - 1]

            if current["close"] > current["open"] and prev["close"] < prev["open"]:
                ob = OrderBlock(
                    start_idx=i - 1,
                    end_idx=i,
                    high=prev["high"],
                    low=prev["low"],
                    direction="bullish",
                    strength=self._calculate_strength(data, i, "bullish"),
                    timestamp=current.get("timestamp", i),
                )
                if ob.strength >= self.min_strength:
                    self._order_blocks.append(ob)

            elif current["close"] < current["open"] and prev["close"] > prev["open"]:
                ob = OrderBlock(
                    start_idx=i - 1,
                    end_idx=i,
                    high=prev["high"],
                    low=prev["low"],
                    direction="bearish",
                    strength=self._calculate_strength(data, i, "bearish"),
                    timestamp=current.get("timestamp", i),
                )
                if ob.strength >= self.min_strength:
                    self._order_blocks.append(ob)

        self._order_blocks = self._order_blocks[-10:]
        return self._order_blocks

    def _calculate_strength(self, df: pd.DataFrame, idx: int, direction: str) -> int:
        strength = 1
        data = df.iloc[max(0, idx - 5): idx + 3]

        volume_ratio = data["volume"].iloc[-1] / data["volume"].iloc[:-1].mean() if data["volume"].iloc[:-1].mean() > 0 else 0
        if volume_ratio > 1.5:
            strength += 1
        if volume_ratio > 2.0:
            strength += 1

        body_size = abs(data["close"].iloc[-1] - data["open"].iloc[-1])
        avg_body = abs(data["close"] - data["open"]).mean()
        if avg_body > 0 and body_size / avg_body > 1.5:
            strength += 1

        wick_ratio = abs(data["high"].iloc[-1] - data["low"].iloc[-1])
        if wick_ratio > 0 and body_size / wick_ratio > 0.6:
            strength += 1

        return strength

    def find_nearest_ob(self, price: float, direction: Optional[str] = None) -> Optional[OrderBlock]:
        candidates = [ob for ob in self._order_blocks if not ob.mitigated]
        if direction:
            candidates = [ob for ob in candidates if ob.direction == direction]
        if not candidates:
            return None
        return min(candidates, key=lambda ob: abs(price - ob.high if ob.direction == "bearish" else price - ob.low))

    def find_ob_in_zone(self, price: float, tolerance: float = 0.001) -> List[OrderBlock]:
        return [
            ob for ob in self._order_blocks
            if not ob.mitigated and abs(price - ob.low) <= tolerance * price
        ]

    def mark_mitigated(self, price: float, tolerance: float = 0.0005):
        for ob in self._order_blocks:
            if not ob.mitigated and ob.low * (1 - tolerance) <= price <= ob.high * (1 + tolerance):
                ob.mitigated = True

    @property
    def order_blocks(self) -> List[OrderBlock]:
        return [ob for ob in self._order_blocks if not ob.mitigated]

    def clear(self):
        self._order_blocks = []
