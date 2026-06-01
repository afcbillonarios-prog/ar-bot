import asyncio
from typing import Optional

from utils import logger
from config.settings import Settings


class DashboardApp:
    def __init__(self, settings: Settings):
        self.log = logger
        self.settings = settings
        self._app = None
        self._running = False

    async def start(self, trader=None):
        if not self.settings.dashboard.enabled:
            self.log.info("Dashboard disabled")
            return

        try:
            import dash
            from dash import dcc, html
            import plotly.graph_objs as go
            import plotly.express as px

            self._app = dash.Dash(
                __name__,
                title="Scalping Bot Dashboard",
                update_title="Updating...",
                assets_ignore=".*",
            )

            from .layouts import DashboardLayout
            from .callbacks import register_callbacks

            layout_builder = DashboardLayout()
            self._app.layout = layout_builder.build()

            register_callbacks(self._app, trader, self.settings)

            self._running = True
            self.log.info(f"Dashboard starting on {self.settings.dashboard.host}:{self.settings.dashboard.port}")

        except ImportError as e:
            self.log.warning(f"Dash not available: {e}. Install: pip install dash plotly")
            return

    def run(self):
        if self._app and self._running:
            self._app.run_server(
                host=self.settings.dashboard.host,
                port=self.settings.dashboard.port,
                debug=False,
            )
