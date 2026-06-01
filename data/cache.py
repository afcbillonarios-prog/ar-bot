import json
import asyncio
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta

from utils import logger


class PriceCache:
    def __init__(self, cache_dir: str = "data/cache", ttl_minutes: int = 60):
        self.log = logger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(minutes=ttl_minutes)
        self._memory_cache: dict = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if datetime.now() - entry["timestamp"] < self.ttl:
                    return entry["data"]
                del self._memory_cache[key]

            file_path = self.cache_dir / f"{key}.json"
            if file_path.exists():
                with open(file_path) as f:
                    entry = json.load(f)
                stored_ts = datetime.fromisoformat(entry["timestamp"])
                if datetime.now() - stored_ts < self.ttl:
                    self._memory_cache[key] = {"data": entry["data"], "timestamp": stored_ts}
                    return entry["data"]
                file_path.unlink()

        return None

    async def set(self, key: str, data: Any):
        async with self._lock:
            entry = {"data": data, "timestamp": datetime.now().isoformat()}
            self._memory_cache[key] = {"data": data, "timestamp": datetime.now()}

            file_path = self.cache_dir / f"{key}.json"
            with open(file_path, "w") as f:
                json.dump(entry, f)

    async def clear(self, key: Optional[str] = None):
        async with self._lock:
            if key:
                self._memory_cache.pop(key, None)
                file_path = self.cache_dir / f"{key}.json"
                if file_path.exists():
                    file_path.unlink()
            else:
                self._memory_cache.clear()
                for f in self.cache_dir.glob("*.json"):
                    f.unlink()
