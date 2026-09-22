from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketState:
    """
    Состояние рынка в конкретный момент времени.

    Первичная версия состояния Quantum Engine:
    цена, скорость, ускорение, волатильность и объём.
    """

    timestamp: pd.Timestamp
    price: float
    velocity: float
    acceleration: float
    volatility: float
    volume: float


REQUIRED_COLUMNS = {
    "timestamp",
    "close",
    "log_return",
    "acceleration",
    "volatility",
    "volume",
}


def build_market_state(row: pd.Series) -> MarketState:
    """
    Преобразует строку DataFrame в строго типизированное состояние рынка.
    """

    missing_columns = REQUIRED_COLUMNS - set(row.index)

    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные поля состояния: "
            f"{sorted(missing_columns)}"
        )

    values = [
        row["close"],
        row["log_return"],
        row["acceleration"],
        row["volatility"],
        row["volume"],
    ]

    if any(pd.isna(value) for value in values):
        raise ValueError(
            "Невозможно построить MarketState: обнаружены NaN."
        )

    return MarketState(
        timestamp=pd.Timestamp(row["timestamp"]),
        price=float(row["close"]),
        velocity=float(row["log_return"]),
        acceleration=float(row["acceleration"]),
        volatility=float(row["volatility"]),
        volume=float(row["volume"]),
    )