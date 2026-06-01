from dash import dcc, html, dash_table
import plotly.graph_objs as go


class DashboardLayout:
    DARK_THEME = {
        "bg": "#0d1117",
        "card": "#161b22",
        "border": "#30363d",
        "text": "#c9d1d9",
        "green": "#00ff88",
        "red": "#ff4444",
        "blue": "#58a6ff",
        "yellow": "#d29922",
        "grid": "#21262d",
    }

    def build(self):
        T = self.DARK_THEME
        return html.Div(
            style={
                "backgroundColor": T["bg"],
                "color": T["text"],
                "fontFamily": "'Inter', 'Segoe UI', sans-serif",
                "minHeight": "100vh",
                "padding": "20px",
            },
            children=[
                dcc.Interval(id="interval-update", interval=1000),
                dcc.Store(id="store-data"),
                html.H1(
                    "Scalping Bot Dashboard",
                    style={"textAlign": "center", "color": T["blue"], "marginBottom": "30px", "fontSize": "2.2rem"},
                ),
                html.Div(
                    className="stats-grid",
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "16px", "marginBottom": "24px"},
                    children=[
                        self._stat_card("Active Trades", "0", T["blue"], "active_trades"),
                        self._stat_card("Win Rate", "0%", T["green"], "win_rate"),
                        self._stat_card("Daily PnL", "$0.00", T["green"], "daily_pnl"),
                        self._stat_card("Total PnL", "$0.00", T["yellow"], "total_pnl"),
                        self._stat_card("ML Ready", "No", T["blue"], "ml_ready"),
                        self._stat_card("Sharpe", "0.00", T["text"], "sharpe"),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px", "marginBottom": "24px"},
                    children=[
                        self._chart_card("Price Chart", dcc.Graph(id="price-chart", style={"height": "500px"})),
                        html.Div(
                            style={"display": "flex", "flexDirection": "column", "gap": "16px"},
                            children=[
                                self._chart_card("ML Confidence", dcc.Graph(id="ml-confidence", style={"height": "240px"})),
                                self._chart_card("Volume Profile", dcc.Graph(id="volume-heatmap", style={"height": "240px"})),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "24px"},
                    children=[
                        self._chart_card("Equity Curve", dcc.Graph(id="equity-curve", style={"height": "300px"})),
                        self._chart_card("Drawdown", dcc.Graph(id="drawdown-chart", style={"height": "300px"})),
                    ],
                ),
                html.Div(
                    className="smc-panel",
                    style={
                        "backgroundColor": T["card"],
                        "border": f"1px solid {T['border']}",
                        "borderRadius": "12px",
                        "padding": "20px",
                        "marginBottom": "24px",
                    },
                    children=[
                        html.H3("Smart Money Analysis", style={"color": T["blue"], "marginBottom": "16px"}),
                        html.Div(id="smc-analysis", style={"color": T["text"]}),
                    ],
                ),
                html.Div(
                    className="trades-table",
                    style={
                        "backgroundColor": T["card"],
                        "border": f"1px solid {T['border']}",
                        "borderRadius": "12px",
                        "padding": "20px",
                    },
                    children=[
                        html.H3("Recent Trades", style={"color": T["blue"], "marginBottom": "16px"}),
                        dash_table.DataTable(
                            id="trades-table",
                            columns=[
                                {"name": "Time", "id": "time"},
                                {"name": "Symbol", "id": "symbol"},
                                {"name": "Side", "id": "side"},
                                {"name": "Entry", "id": "entry"},
                                {"name": "Exit", "id": "exit"},
                                {"name": "PnL", "id": "pnl"},
                                {"name": "RR", "id": "rr"},
                            ],
                            style_header={
                                "backgroundColor": T["bg"],
                                "color": T["text"],
                                "fontWeight": "bold",
                                "border": f"1px solid {T['border']}",
                            },
                            style_cell={
                                "backgroundColor": T["card"],
                                "color": T["text"],
                                "border": f"1px solid {T['border']}",
                                "padding": "8px 12px",
                            },
                            style_data_conditional=[
                                {"if": {"filter_query": "{pnl} > 0", "column_id": "pnl"}, "color": T["green"]},
                                {"if": {"filter_query": "{pnl} < 0", "column_id": "pnl"}, "color": T["red"]},
                                {"if": {"filter_query": '{side} = "long"', "column_id": "side"}, "color": T["green"]},
                                {"if": {"filter_query": '{side} = "short"', "column_id": "side"}, "color": T["red"]},
                            ],
                            page_size=10,
                        ),
                    ],
                ),
            ],
        )

    def _stat_card(self, title, value, color, id_suffix):
        T = self.DARK_THEME
        return html.Div(
            style={
                "backgroundColor": T["card"],
                "border": f"1px solid {T['border']}",
                "borderRadius": "12px",
                "padding": "20px",
                "textAlign": "center",
                "transition": "transform 0.2s",
            },
            children=[
                html.Div(title, style={"fontSize": "0.85rem", "color": T["text"], "marginBottom": "8px", "opacity": "0.7"}),
                html.Div(
                    id=f"stat-{id_suffix}",
                    children=value,
                    style={"fontSize": "1.8rem", "fontWeight": "bold", "color": color},
                ),
            ],
        )

    def _chart_card(self, title, graph):
        T = self.DARK_THEME
        return html.Div(
            style={
                "backgroundColor": T["card"],
                "border": f"1px solid {T['border']}",
                "borderRadius": "12px",
                "padding": "16px",
            },
            children=[
                html.H4(title, style={"color": T["blue"], "marginBottom": "12px", "fontSize": "1rem"}),
                graph,
            ],
        )
