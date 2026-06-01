from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from utils import logger
from config.settings import Settings, RiskConfig


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_amount: float
    entry_time: datetime
    status: str = "open"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trailing_activated: bool = False
    partial_tp_hit: List[bool] = field(default_factory=list)
    id: str = ""


class RiskManager:
    def __init__(self, settings: Settings):
        self.log = logger
        self.config = settings.risk
        self.settings = settings
        self._trades: Dict[str, TradeRecord] = {}
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset: datetime = datetime.now()
        self._current_id: int = 0

    def calculate_position_size(self, price: float, stop_distance: float, capital: float) -> Tuple[float, float]:
        risk_amount = capital * self.config.max_risk_per_trade
        if stop_distance <= 0:
            return 0.0, 0.0
        quantity = risk_amount / stop_distance
        position_value = quantity * price
        max_position = capital * 0.3
        if position_value > max_position:
            quantity = max_position / price
            risk_amount = quantity * stop_distance
        return quantity, risk_amount

    def calculate_sl_tp(
        self, price: float, atr: float, direction: str, high: float = None, low: float = None
    ) -> Tuple[float, float]:
        atr_sl = atr * self.config.atr_multiplier_sl
        atr_tp = atr * self.config.atr_multiplier_tp

        if direction == "long":
            stop_loss = price - atr_sl
            take_profit = price + atr_tp
            if low and stop_loss > low:
                stop_loss = low - atr * 0.5
        else:
            stop_loss = price + atr_sl
            take_profit = price - atr_tp
            if high and stop_loss < high:
                stop_loss = high + atr * 0.5

        return stop_loss, take_profit

    def open_trade(
        self, symbol: str, side: str, price: float, atr: float,
        capital: float, high: float = None, low: float = None
    ) -> Optional[TradeRecord]:
        if self._should_block_trade():
            return None

        if len(self.active_trades) >= self.config.max_open_trades:
            self.log.warning(f"Max open trades reached ({self.config.max_open_trades})")
            return None

        sl, tp = self.calculate_sl_tp(price, atr, side, high, low)
        stop_distance = abs(price - sl)
        quantity, risk_amount = self.calculate_position_size(price, stop_distance, capital)

        if quantity <= 0:
            self.log.warning("Invalid position size (<= 0)")
            return None

        rr_ratio = abs(tp - price) / abs(sl - price)
        if rr_ratio < self.config.min_risk_reward:
            self.log.warning(f"Risk/Reward too low: {rr_ratio:.2f} < {self.config.min_risk_reward}")
            return None

        self._current_id += 1
        trade = TradeRecord(
            id=f"{symbol}_{self._current_id}",
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            quantity=quantity,
            risk_amount=risk_amount,
            entry_time=datetime.now(),
            partial_tp_hit=[False] * len(self.config.partial_take_profits),
        )

        self._trades[trade.id] = trade
        self._daily_trades += 1
        self._reset_daily_if_needed()

        self.log.info(
            f"NEW TRADE [{trade.id}] {side.upper()} {symbol} "
            f"Entry={price:.2f} SL={sl:.2f} TP={tp:.2f} "
            f"Qty={quantity:.4f} Risk=${risk_amount:.2f} RR={rr_ratio:.2f}"
        )

        return trade

    def update_trade(self, trade_id: str, current_price: float, high: float, low: float) -> Optional[TradeRecord]:
        trade = self._trades.get(trade_id)
        if not trade or trade.status != "open":
            return None

        self._check_partial_tps(trade, current_price, high, low)
        self._check_stop_loss(trade, current_price, high, low)
        self._check_take_profit(trade, current_price, high, low)

        return trade

    def close_trade(self, trade_id: str, price: float, reason: str = "manual"):
        trade = self._trades.get(trade_id)
        if not trade:
            return

        trade.status = "closed"
        trade.exit_price = price
        trade.exit_time = datetime.now()

        if trade.side == "long":
            trade.pnl = (price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - price) * trade.quantity

        trade.pnl_pct = (trade.pnl / trade.risk_amount) * 100
        self._daily_pnl += trade.pnl

        self.log.info(
            f"CLOSE [{trade.id}] {reason} "
            f"Exit={price:.2f} PnL=${trade.pnl:.2f} ({trade.pnl_pct:+.1f}%)"
        )

    def _check_stop_loss(self, trade: TradeRecord, price: float, high: float, low: float):
        if trade.side == "long" and low <= trade.stop_loss:
            self.close_trade(trade.id, trade.stop_loss, "stop_loss")
        elif trade.side == "short" and high >= trade.stop_loss:
            self.close_trade(trade.id, trade.stop_loss, "stop_loss")

    def _check_take_profit(self, trade: TradeRecord, price: float, high: float, low: float):
        if trade.side == "long" and high >= trade.take_profit:
            self.close_trade(trade.id, trade.take_profit, "take_profit")
        elif trade.side == "short" and low <= trade.take_profit:
            self.close_trade(trade.id, trade.take_profit, "take_profit")

    def _check_partial_tps(self, trade: TradeRecord, price: float, high: float, low: float):
        for i, (tp_frac, tp_level) in enumerate(zip(
            self.config.partial_take_profits, self.config.partial_tp_levels
        )):
            if trade.partial_tp_hit[i]:
                continue

            tp_price = None
            if trade.side == "long":
                tp_price = trade.entry_price + (trade.take_profit - trade.entry_price) * tp_level / self.config.atr_multiplier_tp
                if high >= tp_price:
                    trade.partial_tp_hit[i] = True
                    self.log.info(f"PARTIAL TP {i+1} [{trade.id}] at {tp_price:.2f} ({tp_frac*100:.0f}%)")
            else:
                tp_price = trade.entry_price - (trade.entry_price - trade.take_profit) * tp_level / self.config.atr_multiplier_tp
                if low <= tp_price:
                    trade.partial_tp_hit[i] = True
                    self.log.info(f"PARTIAL TP {i+1} [{trade.id}] at {tp_price:.2f} ({tp_frac*100:.0f}%)")

    def move_to_break_even(self, trade_id: str, current_price: float):
        trade = self._trades.get(trade_id)
        if not trade or trade.status != "open":
            return

        entry = trade.entry_price
        distance = abs(current_price - entry)
        atr_approx = abs(trade.take_profit - trade.stop_loss) / (self.config.atr_multiplier_sl + self.config.atr_multiplier_tp)

        if distance >= atr_approx * self.config.break_even_trigger:
            trade.stop_loss = entry
            self.log.info(f"Break-even activated for [{trade.id}]")

    def _should_block_trade(self) -> bool:
        self._reset_daily_if_needed()
        if abs(self._daily_pnl) >= self.settings.risk.max_daily_risk * 100000:
            self.log.warning("Daily risk limit reached. Blocking new trades.")
            return True
        return False

    def _reset_daily_if_needed(self):
        now = datetime.now()
        if now.date() > self._last_reset.date():
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset = now

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        return self._trades.get(trade_id)

    @property
    def active_trades(self) -> List[TradeRecord]:
        return [t for t in self._trades.values() if t.status == "open"]

    @property
    def closed_trades(self) -> List[TradeRecord]:
        return [t for t in self._trades.values() if t.status == "closed"]

    @property
    def daily_pnl(self) -> float:
        self._reset_daily_if_needed()
        return self._daily_pnl

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self._trades.values() if t.status == "closed")

    @property
    def win_rate(self) -> float:
        closed = [t for t in self._trades.values() if t.status == "closed"]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed)
