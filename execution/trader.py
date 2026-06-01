import asyncio
from typing import Dict, Optional
from datetime import datetime

import pandas as pd

from utils import logger
from config.settings import Settings, TradingPair
from data import DataManager, PriceCache
from indicators import TechnicalIndicators as TI
from strategies import AIMetaFilter, MetaFilterResult, StrategySignal
from ml import MLPredictor, PredictionResult
from risk import RiskManager, TrailingStopManager, TradeRecord
from alerts.telegram import TelegramNotifier
from execution.mt5_bridge import MT5Bridge


class Trader:
    def __init__(self, settings: Settings):
        self.log = logger
        self.settings = settings
        self.data = DataManager()
        self.cache = PriceCache()
        self.meta_filter = AIMetaFilter(
            confidence_threshold=settings.multi_strategy.confidence_threshold
        )
        self.ml_predictor = MLPredictor(settings)
        self.risk_manager = RiskManager(settings)
        self.trailing_stop = TrailingStopManager(settings.risk)
        self.telegram = TelegramNotifier(settings)
        self.mt5 = MT5Bridge(settings)
        self._running = False
        self._last_train_time: Optional[datetime] = None
        self._signals_log: list = []

    async def start(self):
        self._running = True
        await self.ml_predictor.initialize()

        if self.settings.mt5.enabled:
            connected = await self.mt5.connect()
            if connected:
                self.log.info("MT5 bridge connected")
            else:
                self.log.warning("MT5 bridge failed to connect. Running in signal-only mode.")

        if self.settings.telegram.enabled:
            await self.telegram.send_message(
                "Trading Bot started\n"
                f"Strategies: Trend, Mean Reversion, Breakout\n"
                f"Timeframe: {self.settings.trading.timeframe}\n"
                f"Symbol: {self.settings.trading.symbol}"
            )
        self.log.info("Trader started with multi-strategy engine")

    async def stop(self):
        self._running = False
        if self.settings.mt5.enabled:
            await self.mt5.disconnect()
        if self.settings.telegram.enabled:
            await self.telegram.send_message("Trading bot stopped")
        self.log.info("Trader stopped")

    async def process_candle(self, pair: TradingPair):
        if not self._running:
            return

        df = self.data.get_candles(pair, 100)
        if df.empty or len(df) < 50:
            return

        df = TI.apply_all(df)

        await self._manage_active_trades(pair, df)

        if len(self.risk_manager.active_trades) >= self.settings.risk.max_open_trades:
            return

        meta_result = self.meta_filter.analyze(df)

        if meta_result.signal is None:
            return

        signal = meta_result.signal

        ml_result = await self.ml_predictor.predict(df)
        if ml_result.is_valid:
            signal.confidence = (signal.confidence + ml_result.confidence) / 2
            if signal.is_long and ml_result.signal != 1:
                self.log.debug(
                    f"ML rejected {signal.direction.upper()} "
                    f"(pred={ml_result.signal}, conf={ml_result.confidence:.2f})"
                )
                return
            if signal.is_short and ml_result.signal != -1:
                self.log.debug(
                    f"ML rejected {signal.direction.upper()} "
                    f"(pred={ml_result.signal}, conf={ml_result.confidence:.2f})"
                )
                return

        if signal.confidence < self.settings.multi_strategy.confidence_threshold:
            self.log.debug(
                f"Signal below threshold: {signal.confidence:.2f} < "
                f"{self.settings.multi_strategy.confidence_threshold}"
            )
            return

        await self._execute_signal(pair, df, signal, meta_result)

        await self._retrain_ml_if_needed(df)

    async def _execute_signal(
        self, pair: TradingPair, df: pd.DataFrame,
        signal: StrategySignal, meta_result: MetaFilterResult
    ):
        price = float(df["close"].iloc[-1])
        atr = signal.atr
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
        capital = self.settings.trading.capital

        trade = self.risk_manager.open_trade(
            symbol=pair.kraken_pair,
            side=signal.direction,
            price=price,
            atr=atr,
            capital=capital,
            high=high,
            low=low,
        )

        if trade:
            self.trailing_stop.initialize(trade.id, price, atr, signal.direction)

            await self._notify_trade(trade, signal, meta_result)

            if self.settings.mt5.enabled and not self.settings.paper_trading:
                await self._execute_mt5_order(trade, signal, pair)

            self._signals_log.append({
                "time": datetime.now(),
                "pair": pair.value,
                "direction": signal.direction,
                "strategy": signal.strategy_name,
                "confidence": signal.confidence,
                "regime": meta_result.regime.value,
                "entry": price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
            })

    async def _execute_mt5_order(self, trade: TradeRecord, signal: StrategySignal, pair: TradingPair):
        try:
            result = await self.mt5.send_order(
                pair=pair.kraken_pair,
                direction=signal.direction,
                volume=trade.quantity,
                price=trade.entry_price,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                comment=f"AI_{signal.strategy_name[:10]}",
            )
            if result.success:
                self.log.info(f"MT5 order executed: ticket={result.order_id}")
            else:
                self.log.error(f"MT5 order failed: {result.error}")
        except Exception as e:
            self.log.error(f"MT5 execution error: {e}")

    async def _manage_active_trades(self, pair: TradingPair, df: pd.DataFrame):
        for trade in self.risk_manager.active_trades:
            if trade.symbol != pair.kraken_pair:
                continue

            price = float(df["close"].iloc[-1])
            high = float(df["high"].iloc[-1])
            low = float(df["low"].iloc[-1])

            updated = self.risk_manager.update_trade(trade.id, price, high, low)
            if updated and updated.status == "closed":
                self.trailing_stop.remove(trade.id)
                await self._notify_close(updated)

                won = updated.pnl > 0
                self.meta_filter.record_outcome(
                    "unknown", won, trade.pnl_pct
                )

                if self.settings.mt5.enabled and not self.settings.paper_trading:
                    await self._close_mt5_positions(trade)
                continue

            self.risk_manager.move_to_break_even(trade.id, price)

            new_sl = self.trailing_stop.update(
                trade.id, trade.side, price, high, low, trade.entry_price
            )
            if new_sl:
                if (trade.side == "long" and new_sl > trade.stop_loss) or \
                   (trade.side == "short" and new_sl < trade.stop_loss):
                    trade.stop_loss = new_sl

    async def _close_mt5_positions(self, trade: TradeRecord):
        try:
            positions = await self.mt5.get_positions()
            for pos in positions:
                if pos["comment"] and trade.id in pos["comment"]:
                    await self.mt5.close_position(pos["ticket"])
        except Exception as e:
            self.log.error(f"MT5 close error: {e}")

    async def _retrain_ml_if_needed(self, df: pd.DataFrame):
        if not self._last_train_time:
            self._last_train_time = datetime.now()
            return

        hours_since = (datetime.now() - self._last_train_time).total_seconds() / 3600
        if hours_since >= self.settings.ml.retrain_interval_hours:
            await self.ml_predictor.train(df)
            self._last_train_time = datetime.now()

    async def _notify_trade(self, trade: TradeRecord, signal: StrategySignal, meta_result: MetaFilterResult):
        strategy_emoji = {
            "trend_following": "TREND",
            "mean_reversion": "REVERSION",
            "breakout_momentum": "BREAKOUT",
        }
        strat_label = strategy_emoji.get(signal.strategy_name, signal.strategy_name.upper())
        regime_label = meta_result.regime.value.replace("_", " ").upper()

        msg = (
            f"NEW SIGNAL [{strat_label}]\n"
            f"Pair: {trade.symbol}\n"
            f"Direction: {trade.side.upper()}\n"
            f"Entry: ${trade.entry_price:.2f}\n"
            f"SL: ${trade.stop_loss:.2f}\n"
            f"TP: ${trade.take_profit:.2f}\n"
            f"R:R = {signal.risk_reward:.1f}\n"
            f"Confidence: {signal.confidence:.1%}\n"
            f"Regime: {regime_label}\n"
            f"Reasons: {', '.join(signal.reasons[:5])}"
        )
        self.log.info(f"Signal: {msg}")
        await self.telegram.send_message(msg)

    async def _notify_close(self, trade: TradeRecord):
        direction_emoji = "WIN" if trade.pnl > 0 else "LOSS"
        msg = (
            f"{direction_emoji} TRADE CLOSED\n"
            f"Pair: {trade.symbol}\n"
            f"Direction: {trade.side.upper()}\n"
            f"Entry: ${trade.entry_price:.2f}\n"
            f"Exit: ${trade.exit_price:.2f}\n"
            f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:+.1f}%)\n"
            f"Duration: {(trade.exit_time - trade.entry_time).seconds // 60}m"
        )
        self.log.info(f"Trade closed: {msg}")
        await self.telegram.send_message(msg)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "active_trades": len(self.risk_manager.active_trades),
            "total_trades": len(self.risk_manager.closed_trades),
            "daily_pnl": self.risk_manager.daily_pnl,
            "total_pnl": self.risk_manager.total_pnl,
            "win_rate": self.risk_manager.win_rate,
            "ml_ready": self.ml_predictor.is_ready,
            "mt5_connected": self.mt5.is_connected,
            "strategy_performance": self.meta_filter.strategy_performance,
            "recent_signals": self._signals_log[-10:],
        }
