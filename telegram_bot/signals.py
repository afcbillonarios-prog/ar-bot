import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from config.settings import config

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self):
        self.token = config.telegram.bot_token
        self.chat_id = config.telegram.chat_id
        self.alert_chat_id = config.telegram.alert_chat_id or self.chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = bool(self.token and self.chat_id)

        if not self.enabled:
            logger.warning("Telegram no configurado. Señales no se enviarán.")

    def _send(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        try:
            url = f"{self.base_url}/sendMessage"
            resp = requests.post(url, data={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Error Telegram: {resp.status_code} - {resp.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error enviando Telegram: {e}")
            return False

    def send_signal(self, signal: dict) -> bool:
        side_emoji = "🟢" if signal.get("side") == "BUY" else "🔴"
        side_text = signal.get("side", "N/A")
        strategy = signal.get("strategy", "N/A")
        price = signal.get("price", 0)
        sl = signal.get("sl", 0)
        tp = signal.get("tp", 0)
        confidence = signal.get("confidence", 0)
        regime = signal.get("regime", "N/A")
        reason = signal.get("reason", "")

        risk = abs(price - sl) if price and sl else 0
        reward = abs(tp - price) if tp and price else 0
        rr = reward / risk if risk > 0 else 0

        msg = (
            f"{side_emoji} <b>SEÑAL {side_text} XAUT/USD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Estrategia: <b>{strategy}</b>\n"
            f"🎯 Regime: <b>{regime}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Precio: <b>{price:.2f}</b>\n"
            f"🛑 Stop Loss: <b>{sl:.2f}</b>\n"
            f"✅ Take Profit: <b>{tp:.2f}</b>\n"
            f"📐 R:R = <b>1:{rr:.1f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 Confianza ML: <b>{confidence:.0%}</b>\n"
            f"📝 {reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )

        return self._send(self.chat_id, msg)

    def send_order_executed(self, order: dict, signal: dict) -> bool:
        side = signal.get("side", "N/A")
        emoji = "🟢" if side == "BUY" else "🔴"
        mode = order.get("mode", "unknown").upper()

        msg = (
            f"{emoji} <b>ORDEN EJECUTADA [{mode}]</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 {side} XAUT/USD\n"
            f"💰 Precio: <b>{order.get('price', 0):.2f}</b>\n"
            f"📦 Volumen: <b>{order.get('volume', 0)}</b>\n"
            f"🛑 SL: {signal.get('sl', 0):.2f}\n"
            f"✅ TP: {signal.get('tp', 0):.2f}\n"
            f"🆔 ID: {order.get('order_id', 'N/A')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )

        return self._send(self.chat_id, msg)

    def send_trade_closed(self, trade: dict) -> bool:
        pnl = trade.get("pnl", 0)
        is_winner = trade.get("is_winner", False)
        emoji = "💰" if is_winner else "💸"

        msg = (
            f"{emoji} <b>TRADE CERRADO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 {trade.get('side', 'N/A')} XAUT/USD\n"
            f"💰 Entry: {trade.get('entry_price', 0):.2f}\n"
            f"💰 Exit: {trade.get('exit_price', 0):.2f}\n"
            f"{'📈' if is_winner else '📉'} PnL: <b>{pnl:+.2f}</b>\n"
            f"🎯 Estrategia: {trade.get('strategy', 'N/A')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )

        return self._send(self.chat_id, msg)

    def send_error(self, error_msg: str) -> bool:
        msg = (
            f"🚨 <b>ERROR BOT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"❌ {error_msg}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        return self._send(self.alert_chat_id, msg)

    def send_daily_report(self, stats: dict) -> bool:
        msg = (
            f"📊 <b>REPORTE DIARIO XAUT/USD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 Trades: <b>{stats.get('total_trades', 0)}</b>\n"
            f"🎯 Win Rate: <b>{stats.get('win_rate', 0):.1%}</b>\n"
            f"💰 PnL Total: <b>{stats.get('total_pnl', 0):+.2f}</b>\n"
            f"📊 PnL Promedio: <b>{stats.get('avg_pnl', 0):+.2f}</b>\n"
            f"🏆 Mejor: {stats.get('best_trade', 0):+.2f}\n"
            f"💔 Peor: {stats.get('worst_trade', 0):+.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return self._send(self.chat_id, msg)

    def send_status(self, status_msg: str) -> bool:
        msg = (
            f"🤖 <b>BOT STATUS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"{status_msg}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        return self._send(self.chat_id, msg)

    def send_startup(self) -> bool:
        msg = (
            f"🚀 <b>BOT INICIADO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Par: XAUT/USD\n"
            f"⏱ Temporalidad: 15m\n"
            f"🧠 ML: XGBoost\n"
            f"📋 3 Estrategias activas\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        return self._send(self.chat_id, msg)
