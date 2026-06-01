import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.settings import config

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.cfg = config.trading
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.daily_loss_limit = self.cfg.max_drawdown
        self._last_reset_day = datetime.now(timezone.utc).date()

    def _check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset_day:
            logger.info(f"Reset diario: PnL anterior={self.daily_pnl:.2f}, trades={self.daily_trades}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self._last_reset_day = today

    def can_trade(self, account_balance: float = 10000.0) -> dict:
        self._check_daily_reset()

        reasons = []

        if abs(self.daily_pnl) >= account_balance * self.daily_loss_limit:
            reasons.append(f"Loss diario límite alcanzado: {self.daily_pnl:.2f}")

        if self.consecutive_losses >= self.max_consecutive_losses:
            reasons.append(f"Pérdidas consecutivas: {self.consecutive_losses}/{self.max_consecutive_losses}")

        if self.daily_trades >= self.cfg.max_open_trades * 3:
            reasons.append(f"Demasiados trades hoy: {self.daily_trades}")

        hour = datetime.now(timezone.utc).hour
        if hour in self.cfg.blocked_hours_utc:
            reasons.append(f"Hora bloqueada: {hour}:00 UTC")

        return {
            "can_trade": len(reasons) == 0,
            "reasons": reasons,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
        }

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        sl_price: float,
        risk_pct: float = None,
    ) -> float:
        risk_pct = risk_pct or self.cfg.risk_per_trade
        risk_amount = account_balance * risk_pct
        sl_distance = abs(entry_price - sl_price)

        if sl_distance == 0:
            return 0.01

        position_value = risk_amount / sl_distance
        lots = position_value / entry_price

        lots = max(0.01, round(lots, 2))

        max_lots = account_balance * 0.02 / sl_distance
        lots = min(lots, max(0.01, round(max_lots, 2)))

        return lots

    def calculate_dynamic_sl_tp(
        self,
        entry_price: float,
        atr: float,
        side: str,
        regime: str = "TrendFollowing",
    ) -> dict:
        regime_params = {
            "TrendFollowing": {"sl_mult": 1.5, "tp_mult": 3.0},
            "MeanReversion": {"sl_mult": 1.2, "tp_mult": 2.0},
            "BreakoutMomentum": {"sl_mult": 2.0, "tp_mult": 3.5},
        }

        params = regime_params.get(regime, {"sl_mult": 1.5, "tp_mult": 3.0})

        if side == "BUY":
            sl = entry_price - (atr * params["sl_mult"])
            tp = entry_price + (atr * params["tp_mult"])
        else:
            sl = entry_price + (atr * params["sl_mult"])
            tp = entry_price - (atr * params["tp_mult"])

        sl_distance = abs(entry_price - sl)
        tp_distance = abs(tp - entry_price)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        return {
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "sl_distance": round(sl_distance, 2),
            "tp_distance": round(tp_distance, 2),
            "rr_ratio": round(rr_ratio, 2),
            "risk_params": params,
        }

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_sl: float,
        atr: float,
        side: str,
    ) -> Optional[float]:
        trail_distance = atr * config.trading.trailing_stop_atr

        if side == "BUY":
            new_sl = current_price - trail_distance
            if new_sl > current_sl and new_sl > entry_price:
                return round(new_sl, 2)
        else:
            new_sl = current_price + trail_distance
            if new_sl < current_sl and new_sl < entry_price:
                return round(new_sl, 2)

        return None

    def record_trade_result(self, pnl: float, is_win: bool):
        self._check_daily_reset()
        self.daily_pnl += pnl
        self.daily_trades += 1

        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def should_reduce_risk(self) -> bool:
        if self.consecutive_losses >= 3:
            return True
        if self.daily_pnl < 0:
            return True
        return False

    def get_risk_adjusted_pct(self) -> float:
        base = self.cfg.risk_per_trade

        if self.consecutive_losses >= 3:
            return base * 0.5

        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl) / 10000
            return max(base * 0.25, base * (1 - loss_pct))

        return base

    def get_status(self) -> dict:
        self._check_daily_reset()
        return {
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "risk_pct": self.get_risk_adjusted_pct(),
            "can_trade": self.can_trade()["can_trade"],
        }
