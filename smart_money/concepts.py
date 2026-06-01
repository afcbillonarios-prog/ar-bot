import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .market_structure import MarketStructure, TrendDirection, StructureBreak
from .order_blocks import OrderBlockDetector, OrderBlock
from .fvg import FVGDetector, FVG
from .liquidity import LiquidityDetector, LiquiditySweep


@dataclass
class SMCAnalysis:
    trend: TrendDirection
    bos: Optional[StructureBreak]
    choch: Optional[StructureBreak]
    order_blocks: List[OrderBlock]
    fvgs: List[FVG]
    sweeps: List[LiquiditySweep]
    buy_liquidity: Optional[float]
    sell_liquidity: Optional[float]
    double_top: Optional[Tuple[float, int]]
    double_bottom: Optional[Tuple[float, int]]
    higher_highs_lows: bool
    lower_highs_lows: bool
    nearest_bullish_ob: Optional[OrderBlock]
    nearest_bearish_ob: Optional[OrderBlock]
    active_fvgs: List[FVG]
    imbalances: List[dict]


class SmartMoneyConcepts:
    def __init__(self):
        self.market_structure = MarketStructure()
        self.order_block_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.liquidity_detector = LiquidityDetector()

    def analyze(self, df: pd.DataFrame) -> SMCAnalysis:
        if df.empty or len(df) < 20:
            return self._empty_analysis()

        self.market_structure.update_trend(df)
        bos, bos_price = self.market_structure.detect_bos(df)
        choch, choch_price = self.market_structure.detect_choch(df)
        order_blocks = self.order_block_detector.detect(df)
        fvgs = self.fvg_detector.detect(df)
        sweeps = self.liquidity_detector.detect_sweeps(df)
        equal_highs, equal_lows = self.liquidity_detector.detect_equal_highs_lows(df)

        latest_price = df["close"].iloc[-1]
        buy_liq = self.liquidity_detector.find_buy_liquidity(df)
        sell_liq = self.liquidity_detector.find_sell_liquidity(df)
        double_top = self.liquidity_detector.detect_double_top(df)
        double_bottom = self.liquidity_detector.detect_double_bottom(df)

        self.fvg_detector.check_mitigation(latest_price)
        self.order_block_detector.mark_mitigated(latest_price)

        nearest_bullish_ob = self.order_block_detector.find_nearest_ob(latest_price, "bullish")
        nearest_bearish_ob = self.order_block_detector.find_nearest_ob(latest_price, "bearish")
        active_fvgs = self.fvg_detector.get_active_fvgs(latest_price)

        imbalances = self._detect_imbalances(df)

        return SMCAnalysis(
            trend=self.market_structure.trend,
            bos=bos,
            choch=choch,
            order_blocks=order_blocks[-5:],
            fvgs=fvgs[-10:],
            sweeps=sweeps[-5:],
            buy_liquidity=buy_liq,
            sell_liquidity=sell_liq,
            double_top=double_top,
            double_bottom=double_bottom,
            higher_highs_lows=self.market_structure.get_higher_highs_lows(),
            lower_highs_lows=self.market_structure.get_lower_highs_lows(),
            nearest_bullish_ob=nearest_bullish_ob,
            nearest_bearish_ob=nearest_bearish_ob,
            active_fvgs=active_fvgs,
            imbalances=imbalances,
        )

    def _detect_imbalances(self, df: pd.DataFrame) -> List[dict]:
        imbalances = []
        data = df.tail(50).reset_index(drop=True)
        if len(data) < 10:
            return imbalances

        volume_sma = data["volume"].rolling(10).mean()
        body_sizes = (data["close"] - data["open"]).abs()
        avg_body = body_sizes.rolling(10).mean()

        for i in range(2, len(data)):
            if volume_sma.iloc[i] > 0 and volume_sma.iloc[i - 1] > 0:
                vol_spike = data["volume"].iloc[i] / volume_sma.iloc[i]
                body_ratio = body_sizes.iloc[i] / avg_body.iloc[i] if avg_body.iloc[i] > 0 else 0

                if vol_spike > 1.8 and body_ratio > 1.5:
                    candle_type = "bullish" if data["close"].iloc[i] > data["open"].iloc[i] else "bearish"
                    imbalances.append({
                        "index": i,
                        "type": candle_type,
                        "volume_ratio": float(vol_spike),
                        "body_ratio": float(body_ratio),
                        "price": float(data["close"].iloc[i]),
                    })

        return imbalances

    def _empty_analysis(self) -> SMCAnalysis:
        return SMCAnalysis(
            trend=TrendDirection.NEUTRAL,
            bos=StructureBreak.NONE,
            choch=StructureBreak.NONE,
            order_blocks=[],
            fvgs=[],
            sweeps=[],
            buy_liquidity=None,
            sell_liquidity=None,
            double_top=None,
            double_bottom=None,
            higher_highs_lows=False,
            lower_highs_lows=False,
            nearest_bullish_ob=None,
            nearest_bearish_ob=None,
            active_fvgs=[],
            imbalances=[],
        )
