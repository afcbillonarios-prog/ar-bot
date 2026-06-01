import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from utils import logger
from config.settings import Settings


@dataclass
class MT5OrderResult:
    success: bool
    order_id: int = 0
    price: float = 0.0
    volume: float = 0.0
    error: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MT5Bridge:
    def __init__(self, settings: Settings):
        self.log = logger
        self.settings = settings
        self.config = settings.mt5
        self._connected = False
        self._mt5 = None

    async def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5

            if not mt5.initialize():
                error = mt5.last_error()
                self.log.error(f"MT5 initialization failed: {error}")
                return False

            if self.config.login > 0:
                authorized = mt5.login(
                    self.config.login,
                    password=self.config.password,
                    server=self.config.server,
                )
                if not authorized:
                    error = mt5.last_error()
                    self.log.error(f"MT5 login failed: {error}")
                    mt5.shutdown()
                    return False

            account = mt5.account_info()
            if account:
                self.log.info(
                    f"MT5 connected: {account.login} | "
                    f"Balance: ${account.balance:.2f} | "
                    f"Server: {account.server}"
                )

            self._connected = True
            return True

        except ImportError:
            self.log.warning("MetaTrader5 package not installed. Install with: pip install MetaTrader5")
            return False
        except Exception as e:
            self.log.error(f"MT5 connection error: {e}")
            return False

    async def disconnect(self):
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            self.log.info("MT5 disconnected")

    def _get_mt5_symbol(self, pair: str) -> str:
        symbol_map = {
            "XAU/USD": self.config.symbol or "XAUTUSD",
            "XAUT/USD": self.config.symbol or "XAUTUSD",
            "XBT/USD": "XBTUSD",
            "BTC/USD": "XBTUSD",
        }
        return symbol_map.get(pair, pair.replace("/", ""))

    async def send_order(
        self,
        pair: str,
        direction: str,
        volume: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        magic: int = None,
        comment: str = "AI_BOT",
    ) -> MT5OrderResult:
        if not self._connected or not self._mt5:
            return MT5OrderResult(success=False, error="MT5 not connected")

        symbol = self._get_mt5_symbol(pair)
        mt5 = self._mt5

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            mt5.symbol_select(symbol, True)
            symbol_info = mt5.symbol_info(symbol)

        if symbol_info is None:
            return MT5OrderResult(success=False, error=f"Symbol {symbol} not found")

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return MT5OrderResult(success=False, error=f"No tick data for {symbol}")

        if direction == "long":
            order_type = mt5.ORDER_TYPE_BUY
            execution_price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            execution_price = tick.bid

        if magic is None:
            magic = self.config.magic_number

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(volume, 2),
            "type": order_type,
            "price": execution_price,
            "sl": round(stop_loss, symbol_info.digits),
            "tp": round(take_profit, symbol_info.digits),
            "deviation": self.config.deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        try:
            result = mt5.order_send(request)

            if result is None:
                return MT5OrderResult(success=False, error="order_send returned None")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return MT5OrderResult(
                    success=False,
                    error=f"Order failed: {result.comment} (code={result.retcode})",
                )

            self.log.info(
                f"MT5 ORDER OK: {direction.upper()} {symbol} "
                f"vol={volume:.2f} price={result.price:.2f} "
                f"SL={stop_loss:.2f} TP={take_profit:.2f} "
                f"ticket={result.order}"
            )

            return MT5OrderResult(
                success=True,
                order_id=result.order,
                price=result.price,
                volume=volume,
            )

        except Exception as e:
            return MT5OrderResult(success=False, error=str(e))

    async def close_position(self, ticket: int, symbol: str = None) -> MT5OrderResult:
        if not self._connected or not self._mt5:
            return MT5OrderResult(success=False, error="MT5 not connected")

        mt5 = self._mt5

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return MT5OrderResult(success=False, error=f"Position {ticket} not found")

        pos = positions[0]

        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": "AI_BOT_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        try:
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return MT5OrderResult(success=True, order_id=result.order, price=result.price, volume=pos.volume)
            else:
                error = result.comment if result else "Unknown error"
                return MT5OrderResult(success=False, error=error)
        except Exception as e:
            return MT5OrderResult(success=False, error=str(e))

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        if not self._connected or not self._mt5:
            return None

        info = self._mt5.account_info()
        if info is None:
            return None

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "leverage": info.leverage,
            "currency": info.currency,
        }

    async def get_positions(self) -> list:
        if not self._connected or not self._mt5:
            return []

        positions = self._mt5.positions_get(magic=self.config.magic_number)
        if positions is None:
            return []

        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "long" if p.type == 0 else "short",
                "volume": p.volume,
                "open_price": p.price_open,
                "current_price": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "swap": p.swap,
                "time": p.time,
                "comment": p.comment,
            }
            for p in positions
        ]

    async def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        if not self._connected or not self._mt5:
            return False

        mt5 = self._mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]
        symbol_info = mt5.symbol_info(pos.symbol)
        if symbol_info is None:
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": round(sl if sl is not None else pos.sl, symbol_info.digits),
            "tp": round(tp if tp is not None else pos.tp, symbol_info.digits),
        }

        try:
            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected
