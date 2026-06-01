import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from config.settings import config

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    regime: str
    confidence: float
    best_strategy: str


class AIMetaFilter:
    def __init__(self, ml_engine=None):
        self.ml_engine = ml_engine
        self.name = "AIMetaFilter"

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        if len(df) < 60:
            return MarketRegime("UNKNOWN", 0.0, "TrendFollowing")

        latest = df.iloc[-1]

        adx = latest.get("adx", 20)
        bb_width = latest.get("bb_width", 0.02)
        atr_ratio = latest.get("atr_ratio", 0.005)
        volume_ratio = latest.get("volume_ratio", 1.0)
        rsi = latest.get("rsi", 50)
        ema_distance = abs(latest.get("ema_distance", 0))

        trend_score = 0.0
        if adx > 25:
            trend_score += 0.4
        if ema_distance > 0.002:
            trend_score += 0.3
        if bb_width > 0.02:
            trend_score += 0.3

        volatility_score = 0.0
        if atr_ratio > 0.008:
            volatility_score += 0.4
        if bb_width > 0.03:
            volatility_score += 0.3
        if volume_ratio > 1.5:
            volatility_score += 0.3

        mean_reversion_score = 0.0
        if adx < 20:
            mean_reversion_score += 0.4
        if bb_width < 0.015:
            mean_reversion_score += 0.3
        if 30 < rsi < 70:
            mean_reversion_score += 0.3

        scores = {
            "TrendFollowing": trend_score,
            "MeanReversion": mean_reversion_score,
            "BreakoutMomentum": volatility_score,
        }

        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]

        if best_score < 0.3:
            return MarketRegime("UNCLEAR", best_score, "NONE")

        return MarketRegime(
            regime=best_strategy,
            confidence=best_score,
            best_strategy=best_strategy,
        )

    def filter_signal(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        signal_side: str,
        ml_confidence: float = 0.0,
    ) -> dict:
        regime = self.detect_regime(df)

        strategy_match = regime.best_strategy == strategy_name

        if regime.regime == "UNCLEAR":
            return {
                "approved": False,
                "reason": f"Mercado unclear ({regime.confidence:.2f})",
                "regime": regime.regime,
                "regime_confidence": regime.confidence,
            }

        if not strategy_match:
            return {
                "approved": False,
                "reason": f"Estrategia {strategy_name} no óptima para regime {regime.regime}",
                "regime": regime.regime,
                "regime_confidence": regime.confidence,
            }

        latest = df.iloc[-1]
        session_london = latest.get("session_london", 0)
        session_ny = latest.get("session_ny", 0)
        in_session = session_london or session_ny

        if not in_session:
            hour = latest.get("hour", 12)
            if hour in config.trading.blocked_hours_utc:
                return {
                    "approved": False,
                    "reason": f"Hora bloqueada: {hour}:00 UTC",
                    "regime": regime.regime,
                    "regime_confidence": regime.confidence,
                }

        open_trades = 0
        if hasattr(self, '_db') and self._db:
            open_trades = len(self._db.get_open_trades())

        if open_trades >= config.trading.max_open_trades:
            return {
                "approved": False,
                "reason": f"Max trades abiertos: {open_trades}/{config.trading.max_open_trades}",
                "regime": regime.regime,
                "regime_confidence": regime.confidence,
            }

        ml_approved = True
        if self.ml_engine and ml_confidence < config.ml.min_confidence:
            ml_approved = False

        if not ml_approved:
            return {
                "approved": False,
                "reason": f"ML confidence {ml_confidence:.2f} < {config.ml.min_confidence}",
                "regime": regime.regime,
                "regime_confidence": regime.confidence,
            }

        return {
            "approved": True,
            "reason": f"OK: {regime.regime} regime ({regime.confidence:.2f})",
            "regime": regime.regime,
            "regime_confidence": regime.confidence,
        }

    def set_db(self, db):
        self._db = db
