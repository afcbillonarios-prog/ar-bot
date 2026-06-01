import asyncio
from typing import Callable, Dict, List, Optional
from config.settings import TradingPair
from utils import logger


class MessageHandler:
    def __init__(self):
        self.log = logger
        self._handlers: Dict[str, List[Callable]] = {}

    def register(self, key: str, callback: Callable):
        if key not in self._handlers:
            self._handlers[key] = []
        self._handlers[key].append(callback)
        self.log.debug(f"Handler registered for {key}")

    def unregister(self, key: str, callback: Optional[Callable] = None):
        if key in self._handlers:
            if callback:
                self._handlers[key].remove(callback)
            else:
                del self._handlers[key]

    async def dispatch(self, key: str, data: any):
        handlers = self._handlers.get(key, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.log.error(f"Handler dispatch error for {key}: {e}")
