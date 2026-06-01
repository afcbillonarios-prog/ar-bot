#!/usr/bin/env python3
"""
Multi-Strategy Signal Analyzer
Runs all 3 strategies + AI meta-filter on current data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ccxt
import pandas as pd
from config.settings import Settings
from indicators import TechnicalIndicators as TI
from strategies import AIMetaFilter, TrendFollowingStrategy, MeanReversionStrategy, BreakoutMomentumStrategy
from smart_money import SmartMoneyConcepts
from ml import MLPredictor


def fetch_data(pair: str = "XAUT/USD", timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
    exchange = ccxt.kraken()
    try:
        bars = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def main():
    settings = Settings()
    symbol = settings.trading.symbol
    timeframe = settings.trading.timeframe

    print(f"\n{'='*60}")
    print(f"  MULTI-STRATEGY ANALYSIS: {symbol} | {timeframe}")
    print(f"{'='*60}\n")

    df = fetch_data(symbol, timeframe)
    if df.empty:
        print("Failed to fetch data")
        return

    df = TI.apply_all(df)
    latest = df.iloc[-1]
    price = float(latest["close"])
    atr = float(latest.get("atr_14", 0))

    print(f"Current Price: ${price:.2f}")
    print(f"ATR(14): ${atr:.2f}")
    print(f"RSI(14): {latest.get('rsi_14', 0):.1f}")
    print(f"ADX: {latest.get('adx', 0):.1f}")
    print(f"EMA20: ${latest.get('ema_20', 0):.2f}")
    print(f"EMA50: ${latest.get('ema_50', 0):.2f}")
    print(f"Volume Ratio: {latest.get('volume_ratio', 0):.2f}")
    print()

    trend = TrendFollowingStrategy()
    mr = MeanReversionStrategy()
    bo = BreakoutMomentumStrategy()
    meta = AIMetaFilter(confidence_threshold=settings.multi_strategy.confidence_threshold)

    print("--- TREND FOLLOWING ---")
    trend_sig = trend.analyze(df)
    if trend_sig:
        print(f"  Signal: {trend_sig.direction.upper()}")
        print(f"  Confidence: {trend_sig.confidence:.1%}")
        print(f"  Entry: ${trend_sig.entry_price:.2f}")
        print(f"  SL: ${trend_sig.stop_loss:.2f}")
        print(f"  TP: ${trend_sig.take_profit:.2f}")
        print(f"  R:R = {trend_sig.risk_reward:.1f}")
        print(f"  Reasons: {', '.join(trend_sig.reasons)}")
    else:
        print("  No signal")
    print()

    print("--- MEAN REVERSION ---")
    mr_sig = mr.analyze(df)
    if mr_sig:
        print(f"  Signal: {mr_sig.direction.upper()}")
        print(f"  Confidence: {mr_sig.confidence:.1%}")
        print(f"  Entry: ${mr_sig.entry_price:.2f}")
        print(f"  SL: ${mr_sig.stop_loss:.2f}")
        print(f"  TP: ${mr_sig.take_profit:.2f}")
        print(f"  R:R = {mr_sig.risk_reward:.1f}")
        print(f"  Reasons: {', '.join(mr_sig.reasons)}")
    else:
        print("  No signal")
    print()

    print("--- BREAKOUT MOMENTUM ---")
    bo_sig = bo.analyze(df)
    if bo_sig:
        print(f"  Signal: {bo_sig.direction.upper()}")
        print(f"  Confidence: {bo_sig.confidence:.1%}")
        print(f"  Entry: ${bo_sig.entry_price:.2f}")
        print(f"  SL: ${bo_sig.stop_loss:.2f}")
        print(f"  TP: ${bo_sig.take_profit:.2f}")
        print(f"  R:R = {bo_sig.risk_reward:.1f}")
        print(f"  Reasons: {', '.join(bo_sig.reasons)}")
    else:
        print("  No signal")
    print()

    print("--- AI META-FILTER ---")
    meta_result = meta.analyze(df)
    print(f"  Market Regime: {meta_result.regime.value}")
    print(f"  Active Strategy: {meta_result.active_strategy}")
    print(f"  Final Confidence: {meta_result.confidence:.1%}")
    print(f"  Regime Scores:")
    for strat, score in meta_result.regime_scores.items():
        print(f"    {strat}: {score:.2f}")
    if meta_result.signal:
        s = meta_result.signal
        print(f"\n  FINAL SIGNAL: {s.direction.upper()}")
        print(f"  Strategy: {s.strategy_name}")
        print(f"  Entry: ${s.entry_price:.2f}")
        print(f"  SL: ${s.stop_loss:.2f}")
        print(f"  TP: ${s.take_profit:.2f}")
        print(f"  R:R = {s.risk_reward:.1f}")
        print(f"  Confidence: {s.confidence:.1%}")
    else:
        print(f"\n  No final signal ({meta_result.reason})")
    print()

    smc = SmartMoneyConcepts()
    analysis = smc.analyze(df)
    print("--- SMART MONEY CONCEPTS ---")
    print(f"  Trend: {analysis.trend.value}")
    print(f"  BOS: {analysis.bos.value}")
    print(f"  CHOCH: {analysis.choch.value}")
    print(f"  Order Blocks: {len(analysis.order_blocks)}")
    print(f"  FVGs: {len(analysis.active_fvgs)}")
    print(f"  Liquidity Sweeps: {len(analysis.sweeps)}")
    print()

    print("="*60)


if __name__ == "__main__":
    main()
