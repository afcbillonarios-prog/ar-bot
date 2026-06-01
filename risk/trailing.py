from typing import Optional
from dataclasses import dataclass

from utils import logger
from config.settings import RiskConfig


@dataclass
class TrailingState:
    activated: bool = False
    highest_high: float = 0.0
    lowest_low: float = float("inf")
    swing_high: float = 0.0
    swing_low: float = float("inf")
    activation_price: float = 0.0
    trailing_distance: float = 0.0


class TrailingStopManager:
    def __init__(self, config: RiskConfig):
        self.log = logger
        self.config = config
        self._states: dict = {}

    def initialize(self, trade_id: str, entry_price: float, atr: float, side: str = "long"):
        if side == "long":
            activation_price = entry_price + atr * self.config.trailing_activation
        else:
            activation_price = entry_price - atr * self.config.trailing_activation
        self._states[trade_id] = TrailingState(
            highest_high=entry_price,
            lowest_low=entry_price,
            swing_high=entry_price,
            swing_low=entry_price,
            activation_price=activation_price,
            trailing_distance=atr * self.config.atr_multiplier_sl * 0.5,
        )

    def update(
        self, trade_id: str, side: str, current_price: float,
        high: float, low: float, entry_price: float,
    ) -> Optional[float]:
        state = self._states.get(trade_id)
        if not state:
            return None

        state.highest_high = max(state.highest_high, high)
        state.lowest_low = min(state.lowest_low, low)

        if side == "long":
            return self._update_long(state, current_price, high, entry_price)
        else:
            return self._update_short(state, current_price, low, entry_price)

    def _update_long(self, state: TrailingState, price: float, high: float, entry: float) -> Optional[float]:
        if high > state.swing_high:
            state.swing_high = high

        if not state.activated:
            if price >= state.activation_price:
                state.activated = True
                self.log.info("Trailing stop activated (LONG)")
                return entry

        if state.activated:
            new_sl = state.swing_high - state.trailing_distance
            if high > state.highest_high:
                state.highest_high = high
                new_sl = high - state.trailing_distance
            return max(new_sl, entry)

        return None

    def _update_short(self, state: TrailingState, price: float, low: float, entry: float) -> Optional[float]:
        if low < state.swing_low:
            state.swing_low = low

        if not state.activated:
            if price <= state.activation_price:
                state.activated = True
                self.log.info("Trailing stop activated (SHORT)")
                return entry

        if state.activated:
            new_sl = state.swing_low + state.trailing_distance
            if low < state.lowest_low:
                state.lowest_low = low
                new_sl = low + state.trailing_distance
            return min(new_sl, entry)

        return None

    def get_state(self, trade_id: str) -> Optional[TrailingState]:
        return self._states.get(trade_id)

    def remove(self, trade_id: str):
        self._states.pop(trade_id, None)

    def clear(self):
        self._states.clear()
