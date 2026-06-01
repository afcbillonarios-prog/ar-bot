import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LiquiditySweep:
    price: float
    side: str
    timestamp: float
    index: int
    volume: float = 0.0
    type: str = ""

    def __str__(self):
        return f"{self.side.upper()} Sweep at {self.price:.2f} ({self.type})"


class LiquidityDetector:
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self._sweeps: List[LiquiditySweep] = []
        self._equal_highs: List[Tuple[float, int]] = []
        self._equal_lows: List[Tuple[float, int]] = []

    def detect_sweeps(self, df: pd.DataFrame) -> List[LiquiditySweep]:
        data = df.tail(self.lookback * 2).reset_index(drop=True)
        if len(data) < self.lookback:
            return []

        new_sweeps = []
        highs = data["high"].values
        lows = data["low"].values
        closes = data["close"].values
        timestamps = data["timestamp"].values if "timestamp" in data.columns else data.index.values

        recent_highs = highs[-self.lookback:-1]
        recent_lows = lows[-self.lookback:-1]

        if len(recent_highs) > 0:
            zone_high = np.max(recent_highs)
            if closes[-1] > zone_high:
                sweep = LiquiditySweep(
                    price=zone_high,
                    side="short",
                    timestamp=timestamps[-1],
                    index=len(data) - 1,
                    volume=float(data["volume"].iloc[-1]),
                    type="break_above"
                )
                new_sweeps.append(sweep)

        if len(recent_lows) > 0:
            zone_low = np.min(recent_lows)
            if closes[-1] < zone_low:
                sweep = LiquiditySweep(
                    price=zone_low,
                    side="long",
                    timestamp=timestamps[-1],
                    index=len(data) - 1,
                    volume=float(data["volume"].iloc[-1]),
                    type="break_below"
                )
                new_sweeps.append(sweep)

        self._sweeps.extend(new_sweeps)
        self._sweeps = self._sweeps[-50:]
        return new_sweeps

    def detect_equal_highs_lows(self, df: pd.DataFrame) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        data = df.tail(self.lookback * 2).reset_index(drop=True)
        if len(data) < 10:
            return [], []

        highs = data["high"].values
        lows = data["low"].values
        equal_highs, equal_lows = [], []
        tolerance = 0.0003

        for i in range(len(highs) - 5):
            for j in range(i + 1, min(i + 10, len(highs))):
                if abs(highs[i] - highs[j]) / highs[i] < tolerance:
                    equal_highs.append((highs[i], j))

        for i in range(len(lows) - 5):
            for j in range(i + 1, min(i + 10, len(lows))):
                if abs(lows[i] - lows[j]) / lows[i] < tolerance:
                    equal_lows.append((lows[i], j))

        self._equal_highs = list(set(equal_highs[-20:]))
        self._equal_lows = list(set(equal_lows[-20:]))
        return self._equal_highs, self._equal_lows

    def find_buy_liquidity(self, df: pd.DataFrame) -> Optional[float]:
        lows = df["low"].tail(self.lookback).values
        if len(lows) < 3:
            return None
        return np.min(lows)

    def find_sell_liquidity(self, df: pd.DataFrame) -> Optional[float]:
        highs = df["high"].tail(self.lookback).values
        if len(highs) < 3:
            return None
        return np.max(highs)

    def detect_double_top(self, df: pd.DataFrame, tolerance: float = 0.001) -> Optional[Tuple[float, int]]:
        swing_highs = []
        data = df.tail(30).reset_index(drop=True)
        for i in range(1, len(data) - 1):
            if data["high"].iloc[i] > data["high"].iloc[i - 1] and data["high"].iloc[i] > data["high"].iloc[i + 1]:
                swing_highs.append((data["high"].iloc[i], i))

        for i in range(len(swing_highs)):
            for j in range(i + 1, len(swing_highs)):
                if abs(swing_highs[i][0] - swing_highs[j][0]) / swing_highs[i][0] < tolerance:
                    return (swing_highs[i][0], swing_highs[j][1])
        return None

    def detect_double_bottom(self, df: pd.DataFrame, tolerance: float = 0.001) -> Optional[Tuple[float, int]]:
        swing_lows = []
        data = df.tail(30).reset_index(drop=True)
        for i in range(1, len(data) - 1):
            if data["low"].iloc[i] < data["low"].iloc[i - 1] and data["low"].iloc[i] < data["low"].iloc[i + 1]:
                swing_lows.append((data["low"].iloc[i], i))

        for i in range(len(swing_lows)):
            for j in range(i + 1, len(swing_lows)):
                if abs(swing_lows[i][0] - swing_lows[j][0]) / swing_lows[i][0] < tolerance:
                    return (swing_lows[i][0], swing_lows[j][1])
        return None

    @property
    def sweeps(self) -> List[LiquiditySweep]:
        return self._sweeps[-20:]

    def clear(self):
        self._sweeps = []
        self._equal_highs = []
        self._equal_lows = []
