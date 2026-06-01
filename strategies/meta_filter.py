import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from utils import logger
from .trend_following import TrendFollowingStrategy, StrategySignal, MarketRegime
from .mean_reversion import MeanReversionStrategy
from .breakout_momentum import BreakoutMomentumStrategy


@dataclass
class MetaFilterResult:
    active_strategy: str
    signal: Optional[StrategySignal]
    regime: MarketRegime
    confidence: float
    all_signals: List[StrategySignal] = field(default_factory=list)
    regime_scores: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class AIMetaFilter:
    def __init__(self, confidence_threshold: float = 0.55):
        self.log = logger
        self.confidence_threshold = confidence_threshold
        self.trend_strategy = TrendFollowingStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()
        self.breakout_strategy = BreakoutMomentumStrategy()
        self._regime_history: List[MarketRegime] = []
        self._strategy_performance: Dict[str, Dict] = {
            "trend_following": {"wins": 0, "losses": 0, "total_confidence": 0.0},
            "mean_reversion": {"wins": 0, "losses": 0, "total_confidence": 0.0},
            "breakout_momentum": {"wins": 0, "losses": 0, "total_confidence": 0.0},
        }

    def analyze(self, df: pd.DataFrame) -> MetaFilterResult:
        if df.empty or len(df) < 50:
            return MetaFilterResult(
                active_strategy="none",
                signal=None,
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                reason="insufficient_data",
            )

        regime = self._detect_regime(df)
        regime_scores = self._score_regime(df)

        signals: List[StrategySignal] = []

        trend_signal = self.trend_strategy.analyze(df)
        if trend_signal:
            signals.append(trend_signal)

        mr_signal = self.mean_reversion_strategy.analyze(df)
        if mr_signal:
            signals.append(mr_signal)

        breakout_signal = self.breakout_strategy.analyze(df)
        if breakout_signal:
            signals.append(breakout_signal)

        if not signals:
            return MetaFilterResult(
                active_strategy="none",
                signal=None,
                regime=regime,
                confidence=0.0,
                regime_scores=regime_scores,
                reason="no_signals_generated",
            )

        best = self._select_best(signals, regime, regime_scores)

        if best and best.confidence >= self.confidence_threshold:
            return MetaFilterResult(
                active_strategy=best.strategy_name,
                signal=best,
                regime=regime,
                confidence=best.confidence,
                all_signals=signals,
                regime_scores=regime_scores,
                reason=f"selected_{best.strategy_name}",
            )

        return MetaFilterResult(
            active_strategy="none",
            signal=None,
            regime=regime,
            confidence=best.confidence if best else 0.0,
            all_signals=signals,
            regime_scores=regime_scores,
            reason="below_threshold",
        )

    def _detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        latest = df.iloc[-1]

        adx = float(latest["adx"]) if "adx" in df.columns and not np.isnan(latest["adx"]) else 0
        bb_width = float(latest["bb_width"]) if "bb_width" in df.columns and not np.isnan(latest["bb_width"]) else 0
        atr = float(latest["atr_14"]) if "atr_14" in df.columns and not np.isnan(latest["atr_14"]) else 0
        avg_atr = float(df["atr_14"].tail(50).mean()) if "atr_14" in df.columns and len(df) >= 50 else atr

        ema_20 = float(latest["ema_20"]) if "ema_20" in df.columns and not np.isnan(latest["ema_20"]) else 0
        ema_50 = float(latest["ema_50"]) if "ema_50" in df.columns and not np.isnan(latest["ema_50"]) else 0

        if atr > avg_atr * 2.0:
            regime = MarketRegime.HIGH_VOLATILITY
        elif adx > 25 and ema_20 != ema_50:
            if ema_20 > ema_50:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
        elif adx < 20:
            regime = MarketRegime.RANGING
        else:
            regime = MarketRegime.RANGING

        self._regime_history.append(regime)
        if len(self._regime_history) > 20:
            self._regime_history = self._regime_history[-20:]

        return regime

    def _score_regime(self, df: pd.DataFrame) -> Dict[str, float]:
        latest = df.iloc[-1]

        adx = float(latest["adx"]) if "adx" in df.columns and not np.isnan(latest["adx"]) else 0
        bb_width = float(latest["bb_width"]) if "bb_width" in df.columns and not np.isnan(latest["bb_width"]) else 0
        rsi = float(latest["rsi_14"]) if "rsi_14" in df.columns and not np.isnan(latest["rsi_14"]) else 50
        vol_ratio = float(latest["volume_ratio"]) if "volume_ratio" in df.columns and not np.isnan(latest["volume_ratio"]) else 1.0
        atr = float(latest["atr_14"]) if "atr_14" in df.columns and not np.isnan(latest["atr_14"]) else 0
        avg_atr = float(df["atr_14"].tail(20).mean()) if "atr_14" in df.columns else atr

        trend_score = 0.0
        trend_score += min(adx / 50, 1.0) * 0.3
        ema_20 = float(latest["ema_20"]) if "ema_20" in df.columns and not np.isnan(latest["ema_20"]) else 0
        ema_50 = float(latest["ema_50"]) if "ema_50" in df.columns and not np.isnan(latest["ema_50"]) else 0
        ema_diff = abs(ema_20 - ema_50) / ema_50 if ema_50 > 0 else 0
        trend_score += min(ema_diff * 100, 1.0) * 0.3
        if rsi > 55 or rsi < 45:
            trend_score += 0.2
        trend_score += min(vol_ratio / 2, 1.0) * 0.2

        mr_score = 0.0
        bb_range = 1.0 - min(adx / 50, 1.0)
        mr_score += bb_range * 0.3
        if rsi < 35 or rsi > 65:
            mr_score += 0.3
        if bb_width < 0.03:
            mr_score += 0.2
        if vol_ratio < 1.2:
            mr_score += 0.2

        breakout_score = 0.0
        atr_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        breakout_score += min(atr_ratio / 2, 1.0) * 0.3
        breakout_score += min(vol_ratio / 2, 1.0) * 0.3
        if adx > 20:
            breakout_score += 0.2
        if rsi > 55 or rsi < 45:
            breakout_score += 0.2

        total = trend_score + mr_score + breakout_score
        if total > 0:
            trend_score /= total
            mr_score /= total
            breakout_score /= total

        return {
            "trend_following": trend_score,
            "mean_reversion": mr_score,
            "breakout_momentum": breakout_score,
        }

    def _select_best(
        self,
        signals: List[StrategySignal],
        regime: MarketRegime,
        regime_scores: Dict[str, float],
    ) -> Optional[StrategySignal]:
        if not signals:
            return None

        scored = []
        for sig in signals:
            regime_bonus = regime_scores.get(sig.strategy_name, 0.0) * 0.3
            perf = self._strategy_performance.get(sig.strategy_name, {"wins": 0, "losses": 0})
            total = perf["wins"] + perf["losses"]
            win_rate = perf["wins"] / total if total > 0 else 0.5
            perf_bonus = win_rate * 0.1

            adjusted_confidence = sig.confidence * 0.6 + regime_bonus + perf_bonus
            scored.append((sig, adjusted_confidence))

        scored.sort(key=lambda x: x[1], reverse=True)

        best_sig, best_score = scored[0]

        best_sig.confidence = min(best_score, 1.0)

        return best_sig

    def record_outcome(self, strategy_name: str, won: bool, confidence: float):
        if strategy_name in self._strategy_performance:
            if won:
                self._strategy_performance[strategy_name]["wins"] += 1
            else:
                self._strategy_performance[strategy_name]["losses"] += 1
            self._strategy_performance[strategy_name]["total_confidence"] += confidence

    @property
    def regime_history(self) -> List[MarketRegime]:
        return self._regime_history

    @property
    def strategy_performance(self) -> Dict[str, Dict]:
        return self._strategy_performance
