import numpy as np
import pandas as pd
from dash import Input, Output, State, html
import plotly.graph_objs as go
from datetime import datetime
from typing import Optional

from data import DataManager
from config.settings import Settings, TradingPair
from smart_money.concepts import TrendDirection, StructureBreak


def create_candlestick(df: pd.DataFrame, trades: list = None, analysis=None, pair_name: str = ""):
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    colors = ["#00ff88" if row["close"] >= row["open"] else "#ff4444" for _, row in df.iterrows()]

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
        name="Price",
    ))

    for col, color, width, name in [
        ("ema_20", "#58a6ff", 1, "EMA 20"),
        ("ema_50", "#d29922", 1, "EMA 50"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                line=dict(color=color, width=width),
                name=name,
            ))

    if analysis:
        for ob in analysis.order_blocks:
            if ob.direction == "bullish":
                fig.add_hrect(
                    y0=ob.low, y1=ob.high,
                    fillcolor="green", opacity=0.15,
                    line_width=0, name=f"OB {ob.direction}",
                )
            else:
                fig.add_hrect(
                    y0=ob.low, y1=ob.high,
                    fillcolor="red", opacity=0.15,
                    line_width=0, name=f"OB {ob.direction}",
                )

        for fvg in analysis.active_fvgs:
            fillcolor = "green" if fvg.direction == "bullish" else "red"
            fig.add_hrect(
                y0=fvg.lower, y1=fvg.upper,
                fillcolor=fillcolor, opacity=0.1,
                line_width=0, name=f"FVG {fvg.direction}",
            )

        if analysis.buy_liquidity:
            fig.add_hline(
                y=analysis.buy_liquidity,
                line_dash="dash", line_color="#00ff88", opacity=0.3,
                name="Buy Liq",
            )
        if analysis.sell_liquidity:
            fig.add_hline(
                y=analysis.sell_liquidity,
                line_dash="dash", line_color="#ff4444", opacity=0.3,
                name="Sell Liq",
            )

    if trades:
        for t in trades[-20:]:
            marker_color = "#00ff88" if t.pnl > 0 else "#ff4444"
            symbol = "triangle-up" if t.side == "long" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[t.entry_time],
                y=[t.entry_price],
                mode="markers",
                marker=dict(size=10, color=marker_color, symbol=symbol),
                name=f"{t.side.upper()} {t.pnl:+.1f}",
                hovertemplate=f"{t.side.upper()}<br>Entry: {t.entry_price:.2f}<br>PnL: {t.pnl:.2f}<br>",
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(showgrid=True, gridcolor="#21262d", rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
    )

    return fig


def create_equity_curve(equity_curve: list):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve,
        fill="tozeroy",
        line=dict(color="#58a6ff", width=2),
        name="Equity",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def create_drawdown_chart(drawdown: list):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=drawdown,
        fill="tozeroy",
        line=dict(color="#ff4444", width=1),
        name="Drawdown",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", tickformat=".1%"),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def create_ml_confidence_chart(predictions: list):
    fig = go.Figure()
    if predictions:
        confidences = [p["confidence"] for p in predictions[-50:]]
        signals = [p["signal"] for p in predictions[-50:]]
        colors = ["#00ff88" if s == 1 else "#ff4444" if s == -1 else "#888" for s in signals]

        fig.add_trace(go.Bar(
            y=confidences,
            marker_color=colors,
            name="ML Confidence",
        ))
        fig.add_hline(y=0.75, line_dash="dash", line_color="#d29922", opacity=0.5)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", range=[0, 1]),
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def create_volume_heatmap(df: pd.DataFrame):
    fig = go.Figure()
    if len(df) > 20:
        volume = df["volume"].values[-50:]
        colors = ["#00ff88" if df["close"].iloc[-50:].values[i] >= df["open"].iloc[-50:].values[i] else "#ff4444" for i in range(len(volume))]

        fig.add_trace(go.Bar(
            y=volume,
            marker_color=colors,
            name="Volume",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def register_callbacks(app, trader, settings):
    data_mgr = DataManager()

    @app.callback(
        [
            Output("price-chart", "figure"),
            Output("ml-confidence", "figure"),
            Output("volume-heatmap", "figure"),
            Output("equity-curve", "figure"),
            Output("drawdown-chart", "figure"),
            Output("smc-analysis", "children"),
            Output("trades-table", "data"),
            Output("stat-active_trades", "children"),
            Output("stat-win_rate", "children"),
            Output("stat-daily_pnl", "children"),
            Output("stat-total_pnl", "children"),
            Output("stat-ml_ready", "children"),
            Output("stat-sharpe", "children"),
        ],
        [Input("interval-update", "n_intervals")],
    )
    def update_dashboard(n):
        df = data_mgr.get_candles(TradingPair.XAUUSD, 100)
        if df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            return [empty_fig] * 5 + ["No data"] + [[]] + ["0"] * 6

        from indicators import TechnicalIndicators as TI
        df_ind = TI.apply_all(df)

        from smart_money.concepts import SmartMoneyConcepts
        smc = SmartMoneyConcepts()
        analysis = smc.analyze(df_ind)

        trades_data = []
        if trader:
            all_trades = trader.risk_manager.closed_trades[-20:] + trader.risk_manager.active_trades
            for t in all_trades:
                rr = abs(t.take_profit - t.entry_price) / abs(t.stop_loss - t.entry_price) if t.stop_loss != t.entry_price else 0
                trades_data.append({
                    "time": t.entry_time.strftime("%H:%M:%S"),
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry": f"${t.entry_price:.2f}",
                    "exit": f"${t.exit_price:.2f}" if t.exit_price else "Open",
                    "pnl": round(t.pnl, 2),
                    "rr": round(rr, 2),
                })

        price_chart = create_candlestick(df_ind, trades_data if trader else None, analysis)
        equity_fig = create_equity_curve([100000])
        dd_fig = create_drawdown_chart([0])
        ml_fig = create_ml_confidence_chart([])
        vol_fig = create_volume_heatmap(df_ind)

        trend_str = analysis.trend.value if analysis.trend else "N/A"
        bos_str = analysis.bos.value if analysis.bos else "N/A"
        choch_str = analysis.choch.value if analysis.choch else "N/A"
        smc_html = html.Div([
            html.P(f"Trend: {trend_str}", style={"margin": "4px 0"}),
            html.P(f"BOS: {bos_str} | CHOCH: {choch_str}", style={"margin": "4px 0"}),
            html.P(f"Order Blocks: {len(analysis.order_blocks)}", style={"margin": "4px 0"}),
            html.P(f"Active FVGs: {len(analysis.active_fvgs)}", style={"margin": "4px 0"}),
            html.P(f"Sweeps: {len(analysis.sweeps)}", style={"margin": "4px 0"}),
            html.P(f"Buy Liq: {analysis.buy_liquidity:.2f}" if analysis.buy_liquidity else "Buy Liq: None", style={"margin": "4px 0"}),
            html.P(f"Sell Liq: {analysis.sell_liquidity:.2f}" if analysis.sell_liquidity else "Sell Liq: None", style={"margin": "4px 0"}),
            html.P(f"Imbalances: {len(analysis.imbalances)}", style={"margin": "4px 0"}),
        ])

        stats = trader.stats if trader else {}
        return [
            price_chart, ml_fig, vol_fig, equity_fig, dd_fig,
            smc_html, trades_data,
            str(stats.get("active_trades", 0)),
            f"{stats.get('win_rate', 0):.1%}",
            f"${stats.get('daily_pnl', 0):.2f}",
            f"${stats.get('total_pnl', 0):.2f}",
            "Yes" if stats.get("ml_ready") else "No",
            "0.00",
        ]
