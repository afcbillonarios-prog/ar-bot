import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from utils import logger
from config.settings import TradingPair


class DataManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.log = logger
        self._candles: Dict[str, pd.DataFrame] = {}
        self._orderbook: Dict[str, Dict] = {}
        self._trades: Dict[str, List] = {}
        self._lock = asyncio.Lock()

    async def update_candle(self, pair: TradingPair, candle: dict):
        async with self._lock:
            symbol = pair.kraken_pair
            if symbol not in self._candles:
                columns = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
                self._candles[symbol] = pd.DataFrame(columns=columns)

            new_row = {
                "timestamp": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "vwap": float(candle[5]),
                "volume": float(candle[6]),
                "count": int(candle[7]),
                "datetime": datetime.fromtimestamp(candle[0]),
            }

            df = self._candles[symbol]
            ts = int(candle[0])

            if not df.empty and df.iloc[-1]["timestamp"] == ts:
                df.iloc[-1] = new_row
            else:
                new_df = pd.DataFrame([new_row])
                self._candles[symbol] = pd.concat([df, new_df], ignore_index=True).tail(500)

    def get_candles(self, pair: TradingPair, n: int = 100) -> pd.DataFrame:
        symbol = pair.kraken_pair
        df = self._candles.get(symbol, pd.DataFrame())
        if df.empty:
            return df
        return df.tail(n).reset_index(drop=True)

    def get_latest_price(self, pair: TradingPair) -> Optional[float]:
        df = self.get_candles(pair, 1)
        if df.empty:
            return None
        return float(df.iloc[-1]["close"])

    def get_highs_lows(self, pair: TradingPair, n: int = 20):
        df = self.get_candles(pair, n)
        if df.empty or len(df) < n:
            return [], []
        return df["high"].tolist(), df["low"].tolist()

    async def update_orderbook(self, pair: TradingPair, data: dict):
        async with self._lock:
            self._orderbook[pair.kraken_pair] = data

    def get_orderbook(self, pair: TradingPair) -> Dict:
        return self._orderbook.get(pair.kraken_pair, {})

    @property
    def all_symbols(self) -> List[str]:
        return list(self._candles.keys())
