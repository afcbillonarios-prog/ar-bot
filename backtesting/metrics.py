import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_time: float = 0.0
    total_trades: int = 0
    total_fees: float = 0.0
    recovery_factor: float = 0.0
    risk_of_ruin: float = 0.0

    @classmethod
    def calculate(cls, equity_curve: List[float], trades: List, initial_capital: float) -> "PerformanceMetrics":
        metrics = cls()
        equity = np.array(equity_curve)
        returns = pd.Series(equity).pct_change().dropna().values

        if len(returns) == 0:
            return metrics

        metrics.total_return = (equity[-1] - initial_capital) / initial_capital
        metrics.volatility = float(np.std(returns) * np.sqrt(252))
        metrics.annualized_return = float(np.mean(returns) * 252 * 100)

        if metrics.volatility > 0:
            metrics.sharpe_ratio = (np.mean(returns) * 252) / metrics.volatility

        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns) * np.sqrt(252)
            if downside_std > 0:
                metrics.sortino_ratio = (np.mean(returns) * 252) / downside_std

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        metrics.max_drawdown = float(np.min(drawdown))

        if abs(metrics.max_drawdown) > 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown * 100)

        if trades:
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl <= 0]
            metrics.total_trades = len(trades)
            metrics.win_rate = len(wins) / len(trades) if trades else 0
            metrics.avg_win = float(np.mean(wins)) if wins else 0
            metrics.avg_loss = float(np.mean(losses)) if losses else 0
            metrics.profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float("inf")
            metrics.expectancy = (metrics.win_rate * metrics.avg_win) - ((1 - metrics.win_rate) * abs(metrics.avg_loss))

        recovery = abs(metrics.total_return) / abs(metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
        metrics.recovery_factor = float(recovery)

        if metrics.win_rate > 0 and metrics.avg_loss != 0:
            win_loss_ratio = abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else 1
            p = metrics.win_rate
            q = 1 - p
            b = win_loss_ratio
            if b > 0:
                edge = (p * b - q) / b
                if edge > 0:
                    metrics.risk_of_ruin = float(((q / p) ** (initial_capital / (metrics.avg_loss * (1 / edge + 1)))) * 100)

        return metrics
