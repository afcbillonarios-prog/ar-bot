import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Boolean, Text,
    Index, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import config

logger = logging.getLogger(__name__)

Base = declarative_base()


class Candle(Base):
    __tablename__ = config.database.candle_table
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(Integer, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        Index("idx_candle_time", "time", unique=True),
    )


class Trade(Base):
    __tablename__ = config.database.trades_table
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    strategy = Column(String(50), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    sl = Column(Float, nullable=False)
    tp = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    ml_confidence = Column(Float, nullable=True)
    ml_prediction = Column(String(10), nullable=True)
    is_winner = Column(Boolean, nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default="open")
    __table_args__ = (
        Index("idx_trade_status", "status"),
        Index("idx_trade_timestamp", "timestamp"),
    )


class Signal(Base):
    __tablename__ = config.database.signals_table
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    strategy = Column(String(50), nullable=False)
    price = Column(Float, nullable=False)
    sl = Column(Float, nullable=False)
    tp = Column(Float, nullable=False)
    ml_confidence = Column(Float, nullable=True)
    ml_prediction = Column(String(10), nullable=True)
    executed = Column(Boolean, default=False)
    reason = Column(Text, nullable=True)


class Database:
    def __init__(self):
        try:
            self.engine = create_engine(
                config.database.postgres_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to PostgreSQL database.")
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL ({e}). Falling back to SQLite...")
            sqlite_path = "sqlite:///trading_bot.db"
            self.engine = create_engine(
                sqlite_path,
                connect_args={"check_same_thread": False},
            )
            logger.info("SQLite database initialized at: trading_bot.db")

        self.SessionLocal = sessionmaker(bind=self.engine)
        self._ensure_tables()

    def _ensure_tables(self):
        Base.metadata.create_all(self.engine)
        logger.info("Tablas de base de datos verificadas/creadas")

    def get_session(self) -> Session:
        return self.SessionLocal()

    def save_candle(self, candle_data: dict) -> bool:
        try:
            if "sqlite" in self.engine.url.drivername:
                with self.get_session() as session:
                    existing = session.query(Candle).filter(Candle.time == candle_data["time"]).first()
                    if existing:
                        existing.open = candle_data["open"]
                        existing.high = candle_data["high"]
                        existing.low = candle_data["low"]
                        existing.close = candle_data["close"]
                        existing.volume = candle_data["volume"]
                        if "timestamp" in candle_data:
                            existing.timestamp = candle_data["timestamp"]
                    else:
                        candle = Candle(**candle_data)
                        session.add(candle)
                    session.commit()
                return True

            with self.get_session() as session:
                stmt = pg_insert(Candle).values(**candle_data).on_conflict_do_update(
                    index_elements=["time"],
                    set_={
                        "open": candle_data["open"],
                        "high": candle_data["high"],
                        "low": candle_data["low"],
                        "close": candle_data["close"],
                        "volume": candle_data["volume"],
                    },
                )
                session.execute(stmt)
                session.commit()
            return True
        except Exception as e:
            logger.error(f"Error guardando vela: {e}")
            return False

    def save_candles_bulk(self, df: pd.DataFrame) -> int:
        try:
            records = df.to_dict("records")
            if "sqlite" in self.engine.url.drivername:
                with self.get_session() as session:
                    for r in records:
                        existing = session.query(Candle).filter(Candle.time == r["time"]).first()
                        if existing:
                            existing.open = r["open"]
                            existing.high = r["high"]
                            existing.low = r["low"]
                            existing.close = r["close"]
                            existing.volume = r["volume"]
                            if "timestamp" in r:
                                existing.timestamp = r["timestamp"]
                        else:
                            candle = Candle(**r)
                            session.add(candle)
                    session.commit()
                return len(records)

            with self.get_session() as session:
                stmt = pg_insert(Candle).values(records).on_conflict_do_update(
                    index_elements=["time"],
                    set_={"open": Candle.open, "high": Candle.high, "low": Candle.low,
                           "close": Candle.close, "volume": Candle.volume},
                )
                session.execute(stmt)
                session.commit()
            return len(records)
        except Exception as e:
            logger.error(f"Error guardando velas bulk: {e}")
            return 0

    def get_candles(self, limit: int = 500) -> pd.DataFrame:
        try:
            with self.get_session() as session:
                result = session.execute(
                    text(f"SELECT time, open, high, low, close, volume, timestamp "
                         f"FROM {config.database.candle_table} "
                         f"ORDER BY time DESC LIMIT :limit"),
                    {"limit": limit},
                )
                rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "timestamp"])
            return df.sort_values("time").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error obteniendo velas: {e}")
            return pd.DataFrame()

    def save_signal(self, signal_data: dict) -> bool:
        try:
            with self.get_session() as session:
                signal = Signal(**signal_data)
                session.add(signal)
                session.commit()
            return True
        except Exception as e:
            logger.error(f"Error guardando señal: {e}")
            return False

    def save_trade(self, trade_data: dict) -> bool:
        try:
            with self.get_session() as session:
                trade = Trade(**trade_data)
                session.add(trade)
                session.commit()
            return True
        except Exception as e:
            logger.error(f"Error guardando trade: {e}")
            return False

    def get_open_trades(self) -> list:
        try:
            with self.get_session() as session:
                trades = session.query(Trade).filter(Trade.status == "open").all()
                return [
                    {
                        "id": t.id, "symbol": t.symbol, "side": t.side,
                        "entry_price": t.entry_price, "sl": t.sl, "tp": t.tp,
                        "volume": t.volume, "strategy": t.strategy,
                        "timestamp": t.timestamp.isoformat(),
                    }
                    for t in trades
                ]
        except Exception as e:
            logger.error(f"Error obteniendo trades abiertos: {e}")
            return []

    def get_closed_trades(self, days: int = 30) -> pd.DataFrame:
        try:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            with self.get_session() as session:
                if "sqlite" in self.engine.url.drivername:
                    result = session.execute(
                        text(f"SELECT * FROM {config.database.trades_table} "
                             f"WHERE status = 'closed' AND closed_at >= :cutoff "
                             f"ORDER BY closed_at DESC"),
                        {"cutoff": cutoff.strftime("%Y-%m-%d %H:%M:%S")},
                    )
                else:
                    result = session.execute(
                        text(f"SELECT * FROM {config.database.trades_table} "
                             f"WHERE status = 'closed' AND closed_at >= :cutoff "
                             f"ORDER BY closed_at DESC"),
                        {"cutoff": cutoff},
                    )
                rows = result.fetchall()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows, columns=result.keys())
        except Exception as e:
            logger.error(f"Error obteniendo trades cerrados: {e}")
            return pd.DataFrame()

    def get_performance_stats(self, days: int = 30) -> dict:
        df = self.get_closed_trades(days)
        if df.empty:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}

        total = len(df)
        winners = df["is_winner"].sum()
        return {
            "total_trades": total,
            "win_rate": winners / total if total > 0 else 0,
            "total_pnl": df["pnl"].sum(),
            "avg_pnl": df["pnl"].mean(),
            "best_trade": df["pnl"].max(),
            "worst_trade": df["pnl"].min(),
            "avg_rr": (df["pnl"].abs() / df["entry_price"] - 1).mean() if not df.empty else 0,
        }
