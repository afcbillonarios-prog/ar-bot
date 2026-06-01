import asyncio
import json
import time
from typing import Callable, Dict, List, Optional
import websockets

from utils import logger
from config.settings import Settings, TradingPair


class KrakenWebSocket:
    def __init__(self, settings: Settings):
        self.log = logger
        self.settings = settings
        self.url = settings.ws.url
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._subscriptions: Dict[str, List[str]] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._reconnect_attempts = 0
        self._ping_task: Optional[asyncio.Task] = None
        self._reconnect_delay = settings.ws.reconnect_delay
        self._max_attempts = settings.ws.max_reconnect_attempts

    async def connect(self):
        while self._reconnect_attempts < self._max_attempts:
            try:
                self._ws = await websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10_485_760,
                )
                self._reconnect_attempts = 0
                self._running = True
                self.log.info(f"Connected to Kraken WebSocket at {self.url}")
                self._ping_task = asyncio.create_task(self._ping_loop())
                await self._resubscribe()
                await self._message_loop()
                return
            except Exception as e:
                self._reconnect_attempts += 1
                self.log.error(f"WebSocket connection failed ({self._reconnect_attempts}/{self._max_attempts}): {e}")
                if self._reconnect_attempts >= self._max_attempts:
                    self.log.critical("Max reconnection attempts reached")
                    raise
                await asyncio.sleep(self._reconnect_delay * min(self._reconnect_attempts, 5))

    async def subscribe_ohlc(self, pair: TradingPair, interval: int = 5, callback: Optional[Callable] = None):
        symbol = pair.kraken_pair
        channel = "ohlc"
        sub = {"event": "subscribe", "pair": [symbol], "subscription": {"name": channel, "interval": interval}}

        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        self._subscriptions[channel].append(symbol)

        if callback:
            key = f"{channel}:{symbol}"
            if key not in self._callbacks:
                self._callbacks[key] = []
            self._callbacks[key].append(callback)

        if self._ws:
            await self._ws.send(json.dumps(sub))
            self.log.info(f"Subscribed to {channel} for {symbol}")

    async def subscribe_ticker(self, pair: TradingPair, callback: Optional[Callable] = None):
        symbol = pair.kraken_pair
        channel = "ticker"
        sub = {"event": "subscribe", "pair": [symbol], "subscription": {"name": channel}}

        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        self._subscriptions[channel].append(symbol)

        if callback:
            key = f"{channel}:{symbol}"
            if key not in self._callbacks:
                self._callbacks[key] = []
            self._callbacks[key].append(callback)

        if self._ws:
            await self._ws.send(json.dumps(sub))

    async def _message_loop(self):
        while self._running and self._ws:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=30)
                await self._handle_message(json.loads(msg))
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed as e:
                self.log.warning(f"WebSocket connection closed: {e.code}")
                break
            except Exception as e:
                self.log.error(f"Message handling error: {e}")

        if self._running:
            await self._reconnect()

    async def _handle_message(self, msg: dict):
        if isinstance(msg, list) and len(msg) >= 4:
            channel_data = msg[2] if len(msg) > 2 else ""
            pair_symbol = msg[3] if len(msg) > 3 else ""

            if channel_data == "ohlc-5":
                key = f"ohlc:{pair_symbol}"
                if key in self._callbacks:
                    for cb in self._callbacks[key]:
                        try:
                            if asyncio.iscoroutinefunction(cb):
                                await cb(msg[1])
                            else:
                                cb(msg[1])
                        except Exception as e:
                            self.log.error(f"Callback error for {key}: {e}")

            elif isinstance(msg[2], dict) and "a" in msg[2]:
                key = f"ticker:{pair_symbol}"
                if key in self._callbacks:
                    for cb in self._callbacks[key]:
                        try:
                            if asyncio.iscoroutinefunction(cb):
                                await cb(msg[1])
                            else:
                                cb(msg[1])
                        except Exception as e:
                            self.log.error(f"Callback error for {key}: {e}")

        elif isinstance(msg, dict):
            if msg.get("event") == "heartbeat":
                return
            elif msg.get("event") == "subscriptionStatus":
                self.log.info(f"Subscription: {msg.get('status')} - {msg.get('pair', '')}")

    async def _ping_loop(self):
        while self._running:
            await asyncio.sleep(30)
            try:
                if self._ws:
                    await self._ws.send(json.dumps({"event": "ping"}))
            except Exception:
                break

    async def _resubscribe(self):
        for channel, symbols in self._subscriptions.items():
            for symbol in symbols:
                sub = {"event": "subscribe", "pair": [symbol], "subscription": {"name": channel}}
                try:
                    await self._ws.send(json.dumps(sub))
                except Exception as e:
                    self.log.error(f"Resubscribe failed for {channel}:{symbol}: {e}")

    async def _reconnect(self):
        self.log.info("Attempting reconnection...")
        await asyncio.sleep(self._reconnect_delay)
        await self.connect()

    async def disconnect(self):
        self._running = False
        if self._ping_task:
            self._ping_task.cancel()
        if self._ws:
            await self._ws.close()
        self.log.info("WebSocket disconnected")
