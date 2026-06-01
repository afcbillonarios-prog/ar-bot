#!/usr/bin/env python3
"""
ML Model Training & Multi-Strategy Backtesting
XAUT/USD 15-minute timeframe
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ccxt
import pandas as pd
import numpy as np

from config.settings import Settings, TradingPair
from indicators import TechnicalIndicators as TI
from ml import MLModel, FeatureEngine
from strategies import TrendFollowingStrategy, MeanReversionStrategy, BreakoutMomentumStrategy, AIMetaFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log"),
    ],
)
logger = logging.getLogger(__name__)


def fetch_historical(pair: str = "XAUT/USD", timeframe: str = "15m", limit: int = 5000) -> pd.DataFrame:
    exchange = ccxt.kraken()
    all_bars = []
    since = None

    while len(all_bars) < limit:
        batch_size = min(500, limit - len(all_bars))
        try:
            bars = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=batch_size, since=since)
            if not bars:
                break
            all_bars = bars + all_bars
            since = bars[0][0] - 1
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            break

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def train(force: bool = False, symbol: str = "XAUT/USD", timeframe: str = "15m"):
    logger.info("=" * 60)
    logger.info("ML MODEL TRAINING - XAUT/USD")
    logger.info("=" * 60)

    logger.info("Downloading historical data...")
    df = fetch_historical(pair=symbol, timeframe=timeframe, limit=5000)
    logger.info(f"Data downloaded: {len(df)} candles")

    if len(df) < 200:
        logger.error(f"Insufficient data: {len(df)}. Need >= 200")
        return

    df = TI.apply_all(df)

    feature_engine = FeatureEngine(window=50)
    features = feature_engine.compute_features(df)

    if features.empty or len(features) < 50:
        logger.error("Insufficient features after computation")
        return

    labels = feature_engine.create_labels(df)
    valid_mask = labels != 0
    X = features[valid_mask].values
    y = np.where(labels[valid_mask].values == 1, 1, 0)

    if len(X) < 10:
        logger.error(f"Insufficient labeled samples: {len(X)}")
        return

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = MLModel(model_path="models/xgboost_xaut.json")
    model.build(X.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    model.save()

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_acc = np.mean(train_pred == y_train)
    val_acc = np.mean(val_pred == y_val)

    logger.info(f"Training accuracy: {train_acc:.2%}")
    logger.info(f"Validation accuracy: {val_acc:.2%}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Features: {X.shape[1]}")

    fi = model.get_feature_importance()
    if fi:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        logger.info("\nTop 10 Features:")
        for name, imp in sorted_fi[:10]:
            logger.info(f"  {name}: {imp:.4f}")

    logger.info("Training completed.")


def backtest(symbol: str = "XAUT/USD", timeframe: str = "15m"):
    settings = Settings()

    logger.info("=" * 60)
    logger.info("MULTI-STRATEGY BACKTEST - XAUT/USD 15m")
    logger.info("=" * 60)

    logger.info("Downloading data...")
    df = fetch_historical(pair=symbol, timeframe=timeframe, limit=2000)
    logger.info(f"Data: {len(df)} candles")

    if len(df) < 200:
        logger.error("Insufficient data for backtest")
        return

    df = TI.apply_all(df)

    tf_strat = TrendFollowingStrategy()
    mr_strat = MeanReversionStrategy()
    bm_strat = BreakoutMomentumStrategy()
    meta = AIMetaFilter(confidence_threshold=settings.multi_strategy.confidence_threshold)

    initial_capital = settings.backtest.initial_capital
    capital = initial_capital
    equity_curve = [capital]
    peak = capital
    trades = []
    active_trade = None
    risk_per_trade = settings.risk.max_risk_per_trade
    atr_sl_mult = settings.risk.atr_multiplier_sl
    atr_tp_mult = settings.risk.atr_multiplier_tp

    for i in range(60, len(df)):
        window = df.iloc[:i + 1]
        current = df.iloc[i]
        price = float(current["close"])
        high = float(current["high"])
        low = float(current["low"])
        atr = float(current["atr_14"]) if "atr_14" in current.index and not np.isnan(current["atr_14"]) else 0

        if active_trade:
            hit_sl = False
            hit_tp = False

            if active_trade["side"] == "long":
                if low <= active_trade["sl"]:
                    hit_sl = True
                elif high >= active_trade["tp"]:
                    hit_tp = True
            else:
                if high >= active_trade["sl"]:
                    hit_sl = True
                elif low <= active_trade["tp"]:
                    hit_tp = True

            if hit_sl or hit_tp or i == len(df) - 1:
                exit_price = active_trade["sl"] if hit_sl else active_trade["tp"] if hit_tp else price
                if active_trade["side"] == "long":
                    pnl = (exit_price - active_trade["entry"]) * active_trade["size"]
                else:
                    pnl = (active_trade["entry"] - exit_price) * active_trade["size"]

                capital += pnl
                trades.append({
                    "side": active_trade["side"],
                    "strategy": active_trade["strategy"],
                    "entry": active_trade["entry"],
                    "exit": exit_price,
                    "pnl": pnl,
                    "result": "WIN" if pnl > 0 else "LOSS",
                })
                active_trade = None

        elif atr > 0:
            meta_result = meta.analyze(window)

            if meta_result.signal is not None:
                sig = meta_result.signal
                size = (capital * risk_per_trade) / (atr * atr_sl_mult)
                if size > 0:
                    active_trade = {
                        "side": sig.direction,
                        "entry": price,
                        "sl": sig.stop_loss,
                        "tp": sig.take_profit,
                        "size": size,
                        "strategy": sig.strategy_name,
                        "confidence": sig.confidence,
                    }

        equity_curve.append(capital)
        peak = max(peak, capital)

    total_trades = len(trades)
    if total_trades > 0:
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = sum(1 for t in trades if t["pnl"] <= 0)
        win_rate = wins / total_trades
        total_pnl = capital - initial_capital
        max_dd = 0
        peak_eq = equity_curve[0]
        for eq in equity_curve:
            if eq > peak_eq:
                peak_eq = eq
            dd = (peak_eq - eq) / peak_eq
            max_dd = max(max_dd, dd)

        strategy_stats = {}
        for t in trades:
            s = t["strategy"]
            if s not in strategy_stats:
                strategy_stats[s] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if t["pnl"] > 0:
                strategy_stats[s]["wins"] += 1
            else:
                strategy_stats[s]["losses"] += 1
            strategy_stats[s]["pnl"] += t["pnl"]

        logger.info(f"\nTotal trades: {total_trades}")
        logger.info(f"Wins: {wins} | Losses: {losses}")
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Return: {(total_pnl / initial_capital) * 100:.2f}%")
        logger.info(f"Max Drawdown: {max_dd:.2%}")

        logger.info("\nStrategy Breakdown:")
        for strat, stats in strategy_stats.items():
            strat_total = stats["wins"] + stats["losses"]
            strat_wr = stats["wins"] / strat_total if strat_total > 0 else 0
            logger.info(f"  {strat}: {strat_total} trades, WR={strat_wr:.1%}, PnL=${stats['pnl']:.2f}")
    else:
        logger.info("No trades generated in backtest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Training & Backtesting")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--force", action="store_true", help="Force retraining")
    args = parser.parse_args()

    if not args.train and not args.backtest:
        args.train = True
        args.backtest = True

    if args.train:
        train(force=args.force)
    if args.backtest:
        backtest()
