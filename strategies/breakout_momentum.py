import numpy as np
import pandas as pd
from typing import Optional

from utils import logger
from .trend_following import StrategySignal, MarketRegime


class BreakoutMomentumStrategy:
    NAME = "breakout_momentum"

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
        high = float(latest["high"])
        low = float(latest["low"])
        atr = float(latest["atr_14"]) if "atr_14" in df.columns and not np.isnan(latest["atr_14"]) else 0

        if atr <= 0:
            return None

        rsi = float(latest["rsi_14"]) if "rsi_14" in df.columns and not np.isnan(latest["rsi_14"]) else 50
        vol_ratio = float(latest["volume_ratio"]) if "volume_ratio" in df.columns and not np.isnan(latest["volume_ratio"]) else 1.0
        adx = float(latest["adx"]) if "adx" in df.columns and not np.isnan(latest["adx"]) else 0
        macd_hist = float(latest["macd_hist"]) if "macd_hist" in df.columns and not np.isnan(latest["macd_hist"]) else 0
        bb_width = float(latest["bb_width"]) if "bb_width" in df.columns and not np.isnan(latest["bb_width"]) else None
        ema_20 = float(latest["ema_20"]) if "ema_20" in df.columns and not np.isnan(latest["ema_20"]) else price
        ema_50 = float(latest["ema_50"]) if "ema_50" in df.columns and not np.isnan(latest["ema_50"]) else price

        highest_20 = float(latest["highest_20"]) if "highest_20" in df.columns and not np.isnan(latest["highest_20"]) else high
        lowest_20 = float(latest["lowest_20"]) if "lowest_20" in df.columns and not np.isnan(latest["lowest_20"]) else low

        highest_10 = float(latest["highest_10"]) if "highest_10" in df.columns and not np.isnan(latest["highest_10"]) else high
        lowest_10 = float(latest["lowest_10"]) if "lowest_10" in df.columns and not np.isnan(latest["lowest_10"]) else low

        prev_highest_10 = float(prev["highest_10"]) if "highest_10" in prev.index and not np.isnan(prev["highest_10"]) else highest_10
        prev_lowest_10 = float(prev["lowest_10"]) if "lowest_10" in prev.index and not np.isnan(prev["lowest_10"]) else lowest_10

        avg_atr = float(df["atr_14"].tail(20).mean()) if "atr_14" in df.columns else atr
        atr_expanding = atr > avg_atr * 1.1 if avg_atr > 0 else False

        avg_vol = float(df["volume"].tail(20).mean()) if "volume" in df.columns else 1
        volume_spike = float(latest["volume"]) > avg_vol * 1.5 if avg_vol > 0 else False

        breakout_up = price > prev_highest_10
        breakout_down = price < prev_lowest_10

        long_score = 0.0
        long_reasons = []

        if breakout_up:
            long_score += 0.25
            long_reasons.append("breakout_high")

        if volume_spike:
            long_score += 0.20
            long_reasons.append(f"volume_spike_{vol_ratio:.1f}")

        if atr_expanding:
            long_score += 0.10
            long_reasons.append("atr_expanding")

        if rsi > 55 and rsi < 80:
            long_score += 0.10
            long_reasons.append(f"rsi_momentum_{rsi:.0f}")

        if macd_hist > 0:
            long_score += 0.10
            long_reasons.append("macd_bullish")

        if ema_20 > ema_50:
            long_score += 0.10
            long_reasons.append("trend_aligned")

        if adx > 20:
            long_score += 0.05
            long_reasons.append("trending")

        body = float(latest["close"] - latest["open"])
        if body > 0 and body > atr * 0.5:
            long_score += 0.05
            long_reasons.append("strong_bullish_body")

        prev_close = float(prev["close"]) if "close" in prev.index else price
        momentum = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0
        if momentum > 0.1:
            long_score += 0.05
            long_reasons.append("positive_momentum")

        short_score = 0.0
        short_reasons = []

        if breakout_down:
            short_score += 0.25
            short_reasons.append("breakout_low")

        if volume_spike:
            short_score += 0.20
            short_reasons.append(f"volume_spike_{vol_ratio:.1f}")

        if atr_expanding:
            short_score += 0.10
            short_reasons.append("atr_expanding")

        if rsi < 45 and rsi > 20:
            short_score += 0.10
            short_reasons.append(f"rsi_momentum_{rsi:.0f}")

        if macd_hist < 0:
            short_score += 0.10
            short_reasons.append("macd_bearish")

        if ema_20 < ema_50:
            short_score += 0.10
            short_reasons.append("trend_aligned_short")

        if adx > 20:
            short_score += 0.05
            short_reasons.append("trending")

        if body < 0 and abs(body) > atr * 0.5:
            short_score += 0.05
            short_reasons.append("strong_bearish_body")

        if momentum < -0.1:
            short_score += 0.05
            short_reasons.append("negative_momentum")

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
                regime=MarketRegime.HIGH_VOLATILITY,
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
                regime=MarketRegime.HIGH_VOLATILITY,
            )

        return None
