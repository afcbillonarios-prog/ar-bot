import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "trading_bot",
    level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=Path(log_dir) / f"{name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
