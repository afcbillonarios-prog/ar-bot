from typing import Optional
from dataclasses import dataclass

import pandas as pd

from .generator import TradingSignal, SignalType
from utils import logger


@dataclass
class FilterResult:
    passed: bool
    reason: str = ""
    adjusted_confidence: float = 0.0


class SignalFilter:
    def __init__(self):
        self.log = logger

    def apply_all(self, df: pd.DataFrame, signal: TradingSignal) -> FilterResult:
        if not signal or not signal.is_valid:
            return FilterResult(False, "invalid_signal")

        checks = [
            self._check_volatility(df, signal),
            self._check_spread(df, signal),
            self._check_consecutive_signals(signal),
            self._check_time_of_day(df, signal),
            self._check_reversal_distance(df, signal),
        ]

        failed = [c for c in checks if not c.passed]
        if failed:
            reasons = "; ".join(c.reason for c in failed)
            return FilterResult(False, reasons, signal.confidence)

        avg_adjustment = sum(c.adjusted_confidence for c in checks) / len(checks)
        signal.confidence = min(signal.confidence + avg_adjustment, 1.0)

        return FilterResult(True, "all_checks_passed", signal.confidence)

    def _check_volatility(self, df: pd.DataFrame, signal: TradingSignal) -> FilterResult:
        if "atr_14" not in df.columns or "bb_width" not in df.columns:
            return FilterResult(True)

        atr = float(df["atr_14"].iloc[-1])
        bb_width = float(df["bb_width"].iloc[-1])
        avg_atr = float(df["atr_14"].tail(50).mean())
        avg_bb = float(df["bb_width"].tail(50).mean())

        if atr > avg_atr * 2.5:
            return FilterResult(False, "extreme_volatility")
        if bb_width > avg_bb * 2.0:
            return FilterResult(False, "bb_expansion_too_high")
        if bb_width < avg_bb * 0.3:
            return FilterResult(False, "bb_squeeze_no_movement")

        adjustment = 0.0
        if atr < avg_atr * 0.8:
            adjustment += 0.05
        if avg_bb > 0 and bb_width / avg_bb < 1.2:
            adjustment += 0.03

        return FilterResult(True, adjusted_confidence=adjustment)

    def _check_spread(self, df: pd.DataFrame, signal: TradingSignal) -> FilterResult:
        if "ema_spread_20_50" not in df.columns:
            return FilterResult(True)

        spread = float(df["ema_spread_20_50"].iloc[-1])
        if signal.is_long and spread < -2:
            return FilterResult(False, "spread_too_bearish_for_long")
        if signal.is_short and spread > 2:
            return FilterResult(False, "spread_too_bullish_for_short")

        return FilterResult(True)

    def _check_consecutive_signals(self, signal: TradingSignal) -> FilterResult:
        from .generator import SignalGenerator
        return FilterResult(True)

    def _check_time_of_day(self, df: pd.DataFrame, signal: TradingSignal) -> FilterResult:
        import datetime
        now = datetime.datetime.now()
        hour = now.hour
        minute = now.minute

        if hour == 0 or (hour == 23 and minute >= 55) or (hour == 0 and minute <= 5):
            return FilterResult(False, "weekend_or_close")
        if 20 <= hour <= 21:
            return FilterResult(False, "high_impact_news_window")
        if 13 <= hour <= 14:
            return FilterResult(False, "lunch_low_volume")

        return FilterResult(True)

    def _check_reversal_distance(self, df: pd.DataFrame, signal: TradingSignal) -> FilterResult:
        if "ema_20" not in df.columns or "ema_50" not in df.columns:
            return FilterResult(True)

        price = float(df["close"].iloc[-1])
        ema_20 = float(df["ema_20"].iloc[-1])
        ema_50 = float(df["ema_50"].iloc[-1])

        if signal.is_long:
            distance_to_ema20 = abs(price - ema_20) / price * 100
            if distance_to_ema20 > 2.0:
                return FilterResult(False, "price_too_far_from_ema20")
        elif signal.is_short:
            distance_to_ema20 = abs(price - ema_20) / price * 100
            if distance_to_ema20 > 2.0:
                return FilterResult(False, "price_too_far_from_ema20")

        return FilterResult(True)
