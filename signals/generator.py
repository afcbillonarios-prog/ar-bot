from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

import pandas as pd

from utils import logger
from smart_money import SmartMoneyConcepts
from smart_money.concepts import SMCAnalysis
from smart_money.market_structure import TrendDirection, StructureBreak
from indicators import TechnicalIndicators as TI


class SignalType(Enum):
    STRONG_LONG = "strong_long"
    LONG = "long"
    NEUTRAL = "neutral"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


@dataclass
class TradingSignal:
    signal_type: SignalType
    direction: str
    price: float
    confidence: float
    timestamp: float
    reason: List[str] = field(default_factory=list)
    ml_confidence: float = 0.0
    atr: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_price: Optional[float] = None

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    @property
    def is_short(self) -> bool:
        return self.direction == "short"

    @property
    def is_valid(self) -> bool:
        return self.confidence >= 0.6


class SignalGenerator:
    def __init__(self):
        self.log = logger
        self.smc = SmartMoneyConcepts()
        self._last_signal: Optional[TradingSignal] = None
        self._signals_history: List[TradingSignal] = []

    def generate(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if df.empty or len(df) < 50:
            return None

        analysis = self.smc.analyze(df)
        latest = df.iloc[-1]
        price = float(latest["close"])
        timestamp = float(latest["timestamp"]) if "timestamp" in df.columns else 0

        long_signal = self._check_long_conditions(df, analysis, price)
        short_signal = self._check_short_conditions(df, analysis, price)

        if long_signal and short_signal:
            signal = self._resolve_conflict(long_signal, short_signal, analysis)
        elif long_signal:
            signal = long_signal
        elif short_signal:
            signal = short_signal
        else:
            signal = TradingSignal(
                signal_type=SignalType.NEUTRAL,
                direction="neutral",
                price=price,
                confidence=0.0,
                timestamp=timestamp,
            )

        if signal.signal_type != SignalType.NEUTRAL:
            signal.atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else 0
            self._signals_history.append(signal)
            self._last_signal = signal

        return signal

    def _check_long_conditions(self, df: pd.DataFrame, analysis: SMCAnalysis, price: float) -> Optional[TradingSignal]:
        reasons = []
        confidence = 0.0

        if analysis.trend == TrendDirection.BULLISH:
            confidence += 0.15
            reasons.append("trend_bullish")

        ema_20 = df["ema_20"].iloc[-1] if "ema_20" in df.columns else None
        ema_50 = df["ema_50"].iloc[-1] if "ema_50" in df.columns else None
        if ema_20 and ema_50 and ema_20 > ema_50:
            confidence += 0.1
            reasons.append("ema_alignment")

        sweeps = analysis.sweeps
        bearish_sweeps = [s for s in sweeps if s.side == "long"]
        if bearish_sweeps:
            confidence += 0.2
            reasons.append(f"liquidity_sweep_{len(bearish_sweeps)}")

        if analysis.bos == StructureBreak.BOS:
            confidence += 0.15
            reasons.append("bos_confirmed")

        if analysis.nearest_bullish_ob:
            if price >= analysis.nearest_bullish_ob.low * 0.999 and price <= analysis.nearest_bullish_ob.high * 1.001:
                confidence += 0.15
                reasons.append("ob_reaction")

        active_bullish_fvgs = [f for f in analysis.active_fvgs if f.direction == "bullish"]
        if active_bullish_fvgs:
            confidence += 0.1
            reasons.append("fvg_confirmation")

        if "volume_ratio" in df.columns:
            vol_ratio = float(df["volume_ratio"].iloc[-1])
            if vol_ratio > 1.2:
                confidence += 0.05
                reasons.append("volume_confirmation")

        if "rsi_14" in df.columns:
            rsi = float(df["rsi_14"].iloc[-1])
            if 30 <= rsi <= 70:
                confidence += 0.05
                reasons.append("rsi_aligned")

        imbalance_bullish = any(i["type"] == "bullish" for i in analysis.imbalances)
        if imbalance_bullish:
            confidence += 0.05
            reasons.append("imbalance_bullish")

        if confidence >= 0.6:
            signal_type = SignalType.STRONG_LONG if confidence >= 0.8 else SignalType.LONG
            return TradingSignal(
                signal_type=signal_type,
                direction="long",
                price=price,
                confidence=min(confidence, 1.0),
                timestamp=df.iloc[-1].get("timestamp", 0),
                reason=reasons,
            )
        return None

    def _check_short_conditions(self, df: pd.DataFrame, analysis: SMCAnalysis, price: float) -> Optional[TradingSignal]:
        reasons = []
        confidence = 0.0

        if analysis.trend == TrendDirection.BEARISH:
            confidence += 0.15
            reasons.append("trend_bearish")

        ema_20 = df["ema_20"].iloc[-1] if "ema_20" in df.columns else None
        ema_50 = df["ema_50"].iloc[-1] if "ema_50" in df.columns else None
        if ema_20 and ema_50 and ema_20 < ema_50:
            confidence += 0.1
            reasons.append("ema_alignment")

        sweeps = analysis.sweeps
        bullish_sweeps = [s for s in sweeps if s.side == "short"]
        if bullish_sweeps:
            confidence += 0.2
            reasons.append(f"liquidity_sweep_{len(bullish_sweeps)}")

        if analysis.choch == StructureBreak.CHOCH:
            confidence += 0.15
            reasons.append("choch_confirmed")

        if analysis.bos == StructureBreak.BOS and analysis.trend == TrendDirection.BEARISH:
            confidence += 0.1
            reasons.append("bos_bearish")

        if analysis.nearest_bearish_ob:
            if price >= analysis.nearest_bearish_ob.low * 0.999 and price <= analysis.nearest_bearish_ob.high * 1.001:
                confidence += 0.15
                reasons.append("ob_reaction")

        active_bearish_fvgs = [f for f in analysis.active_fvgs if f.direction == "bearish"]
        if active_bearish_fvgs:
            confidence += 0.1
            reasons.append("fvg_confirmation")

        if "volume_ratio" in df.columns:
            vol_ratio = float(df["volume_ratio"].iloc[-1])
            if vol_ratio > 1.2:
                confidence += 0.05
                reasons.append("volume_confirmation")

        if "rsi_14" in df.columns:
            rsi = float(df["rsi_14"].iloc[-1])
            if 30 <= rsi <= 70:
                confidence += 0.05
                reasons.append("rsi_aligned")

        imbalance_bearish = any(i["type"] == "bearish" for i in analysis.imbalances)
        if imbalance_bearish:
            confidence += 0.05
            reasons.append("imbalance_bearish")

        if confidence >= 0.6:
            signal_type = SignalType.STRONG_SHORT if confidence >= 0.8 else SignalType.SHORT
            return TradingSignal(
                signal_type=signal_type,
                direction="short",
                price=price,
                confidence=min(confidence, 1.0),
                timestamp=df.iloc[-1].get("timestamp", 0),
                reason=reasons,
            )
        return None

    def _resolve_conflict(self, long: TradingSignal, short: TradingSignal, analysis: SMCAnalysis) -> TradingSignal:
        if long.confidence > short.confidence + 0.1:
            return long
        elif short.confidence > long.confidence + 0.1:
            return short
        return TradingSignal(
            signal_type=SignalType.NEUTRAL,
            direction="neutral",
            price=long.price,
            confidence=0.0,
            timestamp=long.timestamp,
            reason=["conflict_no_resolution"],
        )

    def get_last_signal(self) -> Optional[TradingSignal]:
        return self._last_signal

    def get_recent_signals(self, n: int = 10) -> List[TradingSignal]:
        return self._signals_history[-n:]

    def clear_history(self):
        self._signals_history = []
        self._last_signal = None
