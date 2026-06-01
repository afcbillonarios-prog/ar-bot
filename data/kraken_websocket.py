import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import websockets
import pandas as pd

from config.settings import config

logger = logging.getLogger(__name__)


class KrakenWebSocket:
    def __init__(self, on_candle: Optional[Callable] = None, on_tick: Optional[Callable] = None):
        self.ws_url = config.kraken.ws_url
        self.pair = config.kraken.pair
        self.timeframe = config.kraken.timeframe
        self.on_candle = on_candle
        self.on_tick = on_tick
        self._running = False
        self._ws = None
        self._last_candle_time = None
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60

    def _build_subscribe_msg(self, channel: str, req_id: int = 1) -> dict:
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        interval = interval_map.get(self.timeframe, 15)
        return {
            "method": "subscribe",
            "req_id": req_id,
            "params": {
                "channel": channel,
                "symbol": [self.pair],
                "interval": interval,
            },
        }

    def _build_candle_key(self, ts: float) -> int:
        interval_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        interval = interval_map.get(self.timeframe, 900)
        return int(ts // interval) * interval

    async def connect(self):
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=config.kraken.ws_heartbeat,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1
                    logger.info(f"Conectado a Kraken WebSocket: {self.ws_url}")

                    sub_ohlcv = self._build_subscribe_msg("ohlc", req_id=1)
                    sub_ticker = self._build_subscribe_msg("ticker", req_id=2)
                    await ws.send(json.dumps(sub_ohlcv))
                    await ws.send(json.dumps(sub_ticker))
                    logger.info(f"Suscrito a {self.pair} {self.timeframe}")

                    async for message in ws:
                        await self._handle_message(json.loads(message))

            except websockets.ConnectionClosed as e:
                logger.warning(f"Conexión cerrada: {e}. Reconectando en {self._reconnect_delay}s...")
            except ConnectionRefusedError:
                logger.warning(f"Conexión rechazada. Reintentando en {self._reconnect_delay}s...")
            except Exception as e:
                logger.error(f"Error WebSocket: {e}", exc_info=True)

            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _handle_message(self, data: dict):
        channel = data.get("channel")
        msg_type = data.get("type")

        if msg_type in ("heartbeat", "status", "subscribe"):
            if msg_type == "subscribe":
                logger.info(f"Suscripción confirmada: {data}")
            return

        if channel == "ohlc" and msg_type in ("update", "snapshot"):
            await self._handle_ohlcv(data)
        elif channel == "ticker" and msg_type == "update":
            await self._handle_ticker(data)

    async def _handle_ohlcv(self, data: dict):
        try:
            for candle in data.get("data", []):
                candle_time = candle.get("time", candle.get("t", 0))
                open_price = float(candle.get("open", candle.get("o", 0)))
                high_price = float(candle.get("high", candle.get("h", 0)))
                low_price = float(candle.get("low", candle.get("l", 0)))
                close_price = float(candle.get("close", candle.get("c", 0)))
                volume = float(candle.get("volume", candle.get("v", 0)))

                candle_key = self._build_candle_key(candle_time)

                candle_df = pd.DataFrame([{
                    "time": candle_key,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "timestamp": datetime.fromtimestamp(candle_key, tz=timezone.utc),
                }])

                if self._last_candle_time is not None and candle_key > self._last_candle_time:
                    logger.info(
                        f"New candle: {self.pair} {self.timeframe} | "
                        f"O={open_price:.2f} H={high_price:.2f} L={low_price:.2f} "
                        f"C={close_price:.2f} V={volume:.4f}"
                    )

                self._last_candle_time = candle_key

                if self.on_candle:
                    await self.on_candle(candle_df)

        except Exception as e:
            logger.error(f"Error procesando OHLCV: {e}", exc_info=True)

    async def _handle_ticker(self, data: dict):
        try:
            for ticker in data.get("data", []):
                if self.on_tick:
                    await self.on_tick(ticker)
        except Exception as e:
            logger.error(f"Error procesando ticker: {e}", exc_info=True)

    async def disconnect(self):
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("Desconectado de Kraken WebSocket")

    async def fetch_historical(self, pair: str = None, timeframe: str = None, limit: int = None) -> pd.DataFrame:
        import ccxt
        pair = pair or self.pair
        tf = timeframe or self.timeframe
        limit = limit or config.kraken.candle_limit

        exchange = ccxt.kraken({
            "apiKey": config.kraken.api_key,
            "secret": config.kraken.api_secret,
        })

        bars = exchange.fetch_ohlcv(pair, timeframe=tf, limit=limit)
        df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        logger.info(f"Historical data cargado: {len(df)} velas {pair} {tf}")
        return df
