from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import asyncio
import time

from utils import logger


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    id: str
    symbol: str
    side: str
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    reduce_only: bool = False
    post_only: bool = False
    client_id: Optional[str] = None


class OrderManager:
    def __init__(self):
        self.log = logger
        self._orders: Dict[str, Order] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    async def create_order(self, order: Order) -> str:
        order.created_at = time.time()
        order.updated_at = time.time()
        self._orders[order.id] = order
        self.log.info(f"Order created: {order.id} {order.side} {order.quantity} {order.symbol}")
        return order.id

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order:
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = time.time()
        self._dispatch(order_id, "cancelled")
        self.log.info(f"Order cancelled: {order_id}")
        return True

    async def update_order_status(self, order_id: str, status: OrderStatus, **kwargs):
        order = self._orders.get(order_id)
        if not order:
            return
        order.status = status
        if "filled_qty" in kwargs:
            order.filled_qty = kwargs["filled_qty"]
        if "avg_fill_price" in kwargs:
            order.avg_fill_price = kwargs["avg_fill_price"]
        order.updated_at = time.time()
        self._dispatch(order_id, status.value)

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_orders(self, symbol: Optional[str] = None, status: Optional[OrderStatus] = None) -> List[Order]:
        result = list(self._orders.values())
        if symbol:
            result = [o for o in result if o.symbol == symbol]
        if status:
            result = [o for o in result if o.status == status]
        return result

    def register_callback(self, order_id: str, callback: Callable):
        if order_id not in self._callbacks:
            self._callbacks[order_id] = []
        self._callbacks[order_id].append(callback)

    def _dispatch(self, order_id: str, event: str):
        callbacks = self._callbacks.get(order_id, [])
        for cb in callbacks:
            try:
                cb(order_id, event)
            except Exception as e:
                self.log.error(f"Callback error for {order_id}: {e}")

    async def cancel_all(self, symbol: Optional[str] = None):
        if symbol:
            orders = [o for o in self._orders.values() if o.symbol == symbol and o.status in (OrderStatus.PENDING, OrderStatus.OPEN)]
        else:
            orders = [o for o in self._orders.values() if o.status in (OrderStatus.PENDING, OrderStatus.OPEN)]

        for order in orders:
            await self.cancel_order(order.id)

        self.log.info(f"Cancelled {len(orders)} orders")

    @property
    def active_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status in (OrderStatus.PENDING, OrderStatus.OPEN)]

    @property
    def total_orders(self) -> int:
        return len(self._orders)
