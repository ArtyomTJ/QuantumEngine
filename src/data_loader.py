from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}

BINANCE_COLUMNS = {
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает CSV с рыночными данными и приводит его
    к единому формату Quantum Engine.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path)

    # Binance использует open_time как timestamp в миллисекундах.
    if "open_time" in df.columns:
        missing_columns = BINANCE_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Отсутствуют обязательные Binance-столбцы: "
                f"{sorted(missing_columns)}"
            )

        df["timestamp"] = pd.to_datetime(
            df["open_time"],
            unit="ms",
            utc=True,
        )

        # timestamp ставим первым столбцом.
        columns = ["timestamp"] + [
            column for column in df.columns
            if column != "timestamp"
        ]

        df = df[columns]

    else:
        # Поддержка нашего внутреннего упрощённого формата.
        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Отсутствуют обязательные столбцы: "
                f"{sorted(missing_columns)}"
            )

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
            )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if "high" in df.columns and "low" in df.columns:
        if (df["high"] < df["low"]).any():
            raise ValueError("Обнаружены свечи, где high < low.")

    if (df["volume"] < 0).any():
        raise ValueError("Обнаружен отрицательный объём.")

    if df["timestamp"].duplicated().any():
        raise ValueError("Обнаружены дублирующиеся timestamp.")

    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df