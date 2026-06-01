import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FVG:
    start_idx: int
    end_idx: int
    upper: float
    lower: float
    direction: str
    size: float
    timestamp: float
    mitigated: bool = False

    def __contains__(self, price: float) -> bool:
        return self.lower <= price <= self.upper


class FVGDetector:
    def __init__(self, min_size_pips: float = 0.5, lookback: int = 100):
        self.min_size_pips = min_size_pips
        self.lookback = lookback
        self._fvgs: List[FVG] = []

    def detect(self, df: pd.DataFrame) -> List[FVG]:
        self._fvgs = []
        data = df.tail(self.lookback).reset_index(drop=True)

        if len(data) < 5:
            return self._fvgs

        for i in range(2, len(data) - 1):
            candle_2 = data.iloc[i - 2]
            candle_1 = data.iloc[i - 1]
            current = data.iloc[i]

            bullish_gap = candle_2["high"] < candle_1["low"]
            bearish_gap = candle_2["low"] > candle_1["high"]

            if bullish_gap:
                fvg_size = candle_1["low"] - candle_2["high"]
                fvg_size_pips = fvg_size * 10000 if fvg_size < 1 else fvg_size

                if fvg_size_pips >= self.min_size_pips:
                    fvg = FVG(
                        start_idx=i - 2,
                        end_idx=i,
                        upper=candle_1["low"],
                        lower=candle_2["high"],
                        direction="bullish",
                        size=fvg_size,
                        timestamp=current.get("timestamp", i),
                    )
                    self._fvgs.append(fvg)

            elif bearish_gap:
                fvg_size = candle_2["low"] - candle_1["high"]
                fvg_size_pips = fvg_size * 10000 if fvg_size < 1 else fvg_size

                if fvg_size_pips >= self.min_size_pips:
                    fvg = FVG(
                        start_idx=i - 2,
                        end_idx=i,
                        upper=candle_2["low"],
                        lower=candle_1["high"],
                        direction="bearish",
                        size=fvg_size,
                        timestamp=current.get("timestamp", i),
                    )
                    self._fvgs.append(fvg)

        self._fvgs = self._fvgs[-20:]
        return self._fvgs

    def check_mitigation(self, price: float, tolerance: float = 0.0001) -> List[FVG]:
        mitigated = []
        for fvg in self._fvgs:
            if not fvg.mitigated and fvg.lower * (1 - tolerance) <= price <= fvg.upper * (1 + tolerance):
                fvg.mitigated = True
                mitigated.append(fvg)
        return mitigated

    def get_unmitigated(self, direction: Optional[str] = None) -> List[FVG]:
        fvgs = [f for f in self._fvgs if not f.mitigated]
        if direction:
            fvgs = [f for f in fvgs if f.direction == direction]
        return fvgs

    def get_active_fvgs(self, current_price: float, max_distance_pct: float = 0.01) -> List[FVG]:
        active = []
        for fvg in self.get_unmitigated():
            distance = min(abs(current_price - fvg.upper), abs(current_price - fvg.lower))
            if distance / current_price <= max_distance_pct:
                active.append(fvg)
        return active

    @property
    def fvgs(self) -> List[FVG]:
        return self._fvgs

    def clear(self):
        self._fvgs = []
