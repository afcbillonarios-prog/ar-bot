#!/usr/bin/env python3
"""
Multi-Strategy Trading Bot - XAUT/USD
Trend Following + Mean Reversion + Breakout Momentum
AI Meta-Filter + Kraken WebSocket + MT5 Bridge
"""

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import logger
from config.settings import Settings, TradingPair
from websocket import KrakenWebSocket, MessageHandler
from data import DataManager, PriceCache
from execution.trader import Trader
from dashboard import DashboardApp
from alerts.telegram import TelegramNotifier


class TradingBot:
    def __init__(self):
        self.log = logger
        self.settings = Settings()
        self.data_mgr = DataManager()
        self.cache = PriceCache()
        self.handler = MessageHandler()
        self.ws = KrakenWebSocket(self.settings)
        self.trader = Trader(self.settings)
        self.dashboard = DashboardApp(self.settings)
        self.telegram = TelegramNotifier(self.settings)
        self._running = False
        self._tasks = []

    async def _handle_ohlc(self, pair: TradingPair, data: list):
        if isinstance(data, list) and len(data) >= 8:
            await self.data_mgr.update_candle(pair, data)
            await self.trader.process_candle(pair)

    def _make_ohlc_handler(self, pair: TradingPair):
        async def handler(data):
            await self._handle_ohlc(pair, data)
        return handler

    async def start(self):
        self._running = True
        self.log.info("Starting Multi-Strategy Trading Bot...")

        await self.telegram.send_message(
            "Multi-Strategy Bot starting\n"
            "Strategies: Trend + Mean Reversion + Breakout\n"
            f"Symbol: {self.settings.trading.symbol}\n"
            f"Timeframe: {self.settings.trading.timeframe}"
        )
        await self.trader.start()

        async def run_dashboard():
            await self.dashboard.start(self.trader)
            self.dashboard.run()

        for pair in self.settings.pairs:
            handler = self._make_ohlc_handler(pair)
            await self.ws.subscribe_ohlc(pair, self.settings.timeframe.primary, handler)
            await self.ws.subscribe_ticker(pair)

        self._tasks = [
            asyncio.create_task(self.ws.connect()),
            asyncio.create_task(run_dashboard()),
        ]

        self.log.info(
            f"Bot running. Pairs: {[p.value for p in self.settings.pairs]} "
            f"Timeframe: {self.settings.trading.timeframe} "
            f"MT5: {'enabled' if self.settings.mt5.enabled else 'disabled'}"
        )

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        self._running = False
        self.log.info("Stopping Trading Bot...")

        for task in self._tasks:
            task.cancel()

        await self.trader.stop()
        await self.ws.disconnect()
        await self.telegram.send_message("Trading Bot stopped")
        await self.telegram.close()

        self.log.info("Trading Bot stopped gracefully")

    def run(self):
        async def main():
            bot = self

            def shutdown():
                asyncio.create_task(bot.stop())

            try:
                if sys.platform != "win32":
                    loop = asyncio.get_event_loop()
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        loop.add_signal_handler(sig, shutdown)
                await bot.start()
            except (KeyboardInterrupt, asyncio.CancelledError):
                await bot.stop()
            except Exception as e:
                self.log.critical(f"Fatal error: {e}", exc_info=True)
                await bot.stop()

        asyncio.run(main())


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
