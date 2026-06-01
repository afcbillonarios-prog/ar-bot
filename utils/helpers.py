import math
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Union


def calculate_pips(price_diff: float, pair_type: str = "forex") -> float:
    if pair_type == "forex":
        return price_diff * 10000
    return price_diff


def pips_to_price(pips: float, pair_type: str = "forex") -> float:
    if pair_type == "forex":
        return pips / 10000
    return pips


def round_price(price: float, tick_size: float = 0.01) -> float:
    return round(price / tick_size) * tick_size


def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def timestamp_to_dt(timestamp: Union[int, float]) -> datetime:
    return datetime.fromtimestamp(timestamp)


def dt_to_timestamp(dt: datetime) -> float:
    return dt.timestamp()


def truncate_decimal(value: float, decimals: int = 2) -> float:
    d = Decimal(str(value))
    return float(d.quantize(Decimal(10) ** -decimals, rounding=ROUND_DOWN))


def normalize_angle(angle: float) -> float:
    return (angle + 360) % 360


def weighted_average(values: list, weights: list) -> float:
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default
