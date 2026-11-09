from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class AuditReport:
    rows: int
    start_time: Optional[pd.Timestamp]
    end_time: Optional[pd.Timestamp]

    duplicate_timestamps: int
    missing_intervals: int

    null_values: int
    invalid_prices: int
    invalid_ohlc: int
    invalid_volume: int

    @property
    def is_valid(self) -> bool:
        return (
            self.duplicate_timestamps == 0
            and self.missing_intervals == 0
            and self.null_values == 0
            and self.invalid_prices == 0
            and self.invalid_ohlc == 0
            and self.invalid_volume == 0
        )


def audit_ohlcv(
    df: pd.DataFrame,
    expected_frequency: str = "1min",
) -> AuditReport:
    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        return AuditReport(
            rows=0,
            start_time=None,
            end_time=None,
            duplicate_timestamps=0,
            missing_intervals=0,
            null_values=0,
            invalid_prices=0,
            invalid_ohlc=0,
            invalid_volume=0,
        )

    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    duplicate_timestamps = int(timestamps.duplicated().sum())

    sorted_timestamps = timestamps.sort_values()

    expected = pd.date_range(
        start=sorted_timestamps.iloc[0],
        end=sorted_timestamps.iloc[-1],
        freq=expected_frequency,
        tz="UTC",
    )

    actual = pd.DatetimeIndex(sorted_timestamps.unique())

    missing_intervals = int(len(expected.difference(actual)))

    price_columns = ["open", "high", "low", "close"]

    null_values = int(df[list(required_columns)].isna().sum().sum())

    invalid_prices = int(
        (
            df[price_columns] <= 0
        ).any(axis=1).sum()
    )

    invalid_ohlc = int(
        (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ).sum()
    )

    invalid_volume = int(
        (df["volume"] < 0).sum()
    )

    return AuditReport(
        rows=len(df),
        start_time=sorted_timestamps.iloc[0],
        end_time=sorted_timestamps.iloc[-1],
        duplicate_timestamps=duplicate_timestamps,
        missing_intervals=missing_intervals,
        null_values=null_values,
        invalid_prices=invalid_prices,
        invalid_ohlc=invalid_ohlc,
        invalid_volume=invalid_volume,
    )