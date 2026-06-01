import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from utils import logger
from config.settings import Settings, TradingPair, BacktestConfig
from indicators import TechnicalIndicators as TI
from strategies import AIMetaFilter


@dataclass
class BacktestTrade:
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    side: str = ""
    size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    strategy: str = ""
    confidence: float = 0.0
    regime: str = ""


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_bars: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    strategy_breakdown: Dict[str, Dict] = field(default_factory=dict)


class BacktestEngine:
    def __init__(self, settings: Settings):
        self.log = logger
        self.settings = settings
        self.config = settings.backtest
        self.meta_filter = AIMetaFilter(
            confidence_threshold=settings.multi_strategy.confidence_threshold
        )

    def run(self, df: pd.DataFrame, pair: TradingPair) -> BacktestResult:
        if df.empty or len(df) < 100:
            return BacktestResult()

        self.log.info(f"Running multi-strategy backtest for {pair.value} on {len(df)} candles")
        df = TI.apply_all(df)

        result = BacktestResult()
        capital = self.config.initial_capital
        equity = [capital]
        peak = capital
        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None

        strategy_stats = {}

        for i in range(50, len(df)):
            window = df.iloc[:i + 1]
            current = df.iloc[i]

            if active_trade:
                is_exit, reason = self._check_exit(active_trade, df, i)
                if is_exit:
                    active_trade.exit_time = current.get("datetime", datetime.now())
                    active_trade.exit_price = float(current["close"])
                    active_trade.pnl = self._calculate_pnl(active_trade)
                    active_trade.pnl_pct = (active_trade.pnl / (active_trade.entry_price * active_trade.size)) * 100 if active_trade.size > 0 else 0
                    active_trade.exit_reason = reason

                    trades.append(active_trade)
                    capital += active_trade.pnl

                    strat = active_trade.strategy
                    if strat not in strategy_stats:
                        strategy_stats[strat] = {"wins": 0, "losses": 0, "pnl": 0.0}
                    if active_trade.pnl > 0:
                        strategy_stats[strat]["wins"] += 1
                    else:
                        strategy_stats[strat]["losses"] += 1
                    strategy_stats[strat]["pnl"] += active_trade.pnl

                    active_trade = None

            elif len(self.risk_manager_trades(trades)) < self.settings.risk.max_open_trades:
                meta_result = self.meta_filter.analyze(window)

                if meta_result.signal is not None:
                    signal = meta_result.signal
                    price = float(current["close"])
                    atr = signal.atr

                    size = (capital * self.settings.risk.max_risk_per_trade) / (atr * self.settings.risk.atr_multiplier_sl) if atr > 0 else 0

                    if size > 0:
                        active_trade = BacktestTrade(
                            entry_time=current.get("datetime", datetime.now()),
                            entry_price=price,
                            side=signal.direction,
                            size=size,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            strategy=signal.strategy_name,
                            confidence=signal.confidence,
                            regime=meta_result.regime.value,
                        )

            equity.append(capital)
            peak = max(peak, capital)

        dd = [(p - peak) / peak * 100 for p in equity]
        result.trades = trades
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.pnl > 0)
        result.losing_trades = sum(1 for t in trades if t.pnl <= 0)
        result.total_pnl = capital - self.config.initial_capital
        result.equity_curve = equity
        result.drawdown_curve = dd
        result.max_drawdown = min(dd) if dd else 0
        result.strategy_breakdown = strategy_stats

        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl <= 0]
            result.avg_win = np.mean(wins) if wins else 0
            result.avg_loss = np.mean(losses) if losses else 0
            result.profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float("inf")
            result.expectancy = (result.win_rate * result.avg_win) - ((1 - result.win_rate) * abs(result.avg_loss))
            result.avg_holding_bars = sum(
                (t.exit_time - t.entry_time).total_seconds() / 900 for t in trades if t.exit_time
            ) / len(trades) if trades else 0

        returns = pd.Series(equity).pct_change().dropna()
        result.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        self.log.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"Win Rate: {result.win_rate:.1%}, "
            f"PnL: ${result.total_pnl:.2f}, "
            f"Sharpe: {result.sharpe_ratio:.2f}, "
            f"Max DD: {result.max_drawdown:.1%}"
        )
        for strat, stats in strategy_stats.items():
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total if total > 0 else 0
            self.log.info(
                f"  {strat}: {total} trades, WR={wr:.1%}, PnL=${stats['pnl']:.2f}"
            )

        return result

    def _check_exit(self, trade: BacktestTrade, df: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        candle = df.iloc[idx]
        high = float(candle["high"])
        low = float(candle["low"])

        if trade.side == "long":
            if low <= trade.stop_loss:
                return True, "stop_loss"
            if high >= trade.take_profit:
                return True, "take_profit"
        else:
            if high >= trade.stop_loss:
                return True, "stop_loss"
            if low <= trade.take_profit:
                return True, "take_profit"

        if idx >= len(df) - 1:
            return True, "end_of_data"

        return False, ""

    def _calculate_pnl(self, trade: BacktestTrade) -> float:
        if trade.side == "long":
            return (trade.exit_price - trade.entry_price) * trade.size
        return (trade.entry_price - trade.exit_price) * trade.size

    def risk_manager_trades(self, trades: List[BacktestTrade]) -> List[BacktestTrade]:
        return [t for t in trades if not t.exit_time]

    def run_monte_carlo(self, trades: List[BacktestTrade], n_simulations: int = 1000) -> Dict:
        if not trades:
            return {}

        pnl_values = np.array([t.pnl for t in trades])
        results = []

        for _ in range(n_simulations):
            sampled = np.random.choice(pnl_values, size=len(pnl_values), replace=True)
            results.append(np.sum(sampled))

        results = np.array(results)
        return {
            "mean_return": float(np.mean(results)),
            "median_return": float(np.median(results)),
            "std_return": float(np.std(results)),
            "var_95": float(np.percentile(results, 5)),
            "var_99": float(np.percentile(results, 1)),
            "prob_positive": float(np.mean(results > 0)),
            "max_return": float(np.max(results)),
            "min_return": float(np.min(results)),
        }
