import asyncio
import socket
from typing import Optional

import aiohttp
from utils import logger
from config.settings import Settings


class _AsyncResolver:
    """Custom resolver that uses asyncio loop.getaddrinfo (avoids aiodns issues on Windows)."""

    async def resolve(self, host: str, port: int = 0, family: int = 0):
        loop = asyncio.get_event_loop()
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM, family=family)
        return [
            {
                "hostname": host,
                "host": info[-1][0],
                "port": port,
                "family": info[0],
                "proto": info[1],
                "flags": info[2],
            }
            for info in infos
        ]

    async def close(self):
        pass


class TelegramNotifier:
    def __init__(self, settings: Settings):
        self.log = logger
        self.config = settings.telegram
        self._session = None
        self._base_url = f"https://api.telegram.org/bot{self.config.bot_token}"

    async def _ensure_session(self):
        if self._session is None and self.config.enabled:
            try:
                connector = aiohttp.TCPConnector(resolver=_AsyncResolver(), use_dns_cache=False)
                self._session = aiohttp.ClientSession(connector=connector)
            except ImportError:
                self.log.warning("aiohttp not installed. Install with: pip install aiohttp")
                return False
        return self._session is not None

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.config.enabled or not self.config.bot_token or not self.config.chat_id:
            return False

        if not await self._ensure_session():
            return False

        try:
            url = f"{self._base_url}/sendMessage"
            payload = {
                "chat_id": self.config.chat_id,
                "text": text[:4096],
                "parse_mode": parse_mode,
            }
            async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                result = await resp.json()
                if result.get("ok"):
                    self.log.debug("Telegram message sent")
                    return True
                else:
                    self.log.error(f"Telegram API error: {result}")
                    return False
        except Exception as e:
            self.log.error(f"Telegram send failed: {e}")
            return False

    async def send_photo(self, photo_path: str, caption: str = "") -> bool:
        if not self.config.enabled:
            return False

        if not await self._ensure_session():
            return False

        try:
            url = f"{self._base_url}/sendPhoto"
            form = aiohttp.FormData()
            form.add_field("chat_id", self.config.chat_id)
            form.add_field("photo", open(photo_path, "rb"))
            if caption:
                form.add_field("caption", caption[:1024])

            async with self._session.post(url, data=form) as resp:
                result = await resp.json()
                return result.get("ok", False)
        except Exception as e:
            self.log.error(f"Telegram photo send failed: {e}")
            return False

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
