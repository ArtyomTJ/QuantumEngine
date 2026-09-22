from pathlib import Path

import numpy as np

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


def describe(name, series):
    values = series.dropna()

    print()
    print(f"=== {name} ===")
    print(f"Среднее:      {values.mean():.6f}")
    print(f"Std:          {values.std():.6f}")
    print(f"Медиана:      {values.median():.6f}")
    print(f"P95:          {values.quantile(0.95):.6f}")
    print(f"P99:          {values.quantile(0.99):.6f}")
    print(f"Максимум:     {values.max():.6f}")


def main():
    print("Загрузка данных...")

    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт динамики...")

    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    df["velocity_squared"] = df["log_return"] ** 2
    df["acceleration_squared"] = df["acceleration"] ** 2
    df["volatility_volume"] = (
        df["volatility"] * df["volume"]
    )

    scales = {
        "velocity_squared": df["velocity_squared"].quantile(0.95),
        "acceleration_squared": df["acceleration_squared"].quantile(0.95),
        "volatility_volume": df["volatility_volume"].quantile(0.95),
    }

    df["velocity_normalized"] = (
        df["velocity_squared"] / scales["velocity_squared"]
    )

    df["acceleration_normalized"] = (
        df["acceleration_squared"]
        / scales["acceleration_squared"]
    )

    df["volatility_volume_normalized"] = (
        df["volatility_volume"]
        / scales["volatility_volume"]
    )

    print()
    print("=== Energy Scales ===")
    print(
        f"Velocity² P95:       "
        f"{scales['velocity_squared']:.12g}"
    )
    print(
        f"Acceleration² P95:   "
        f"{scales['acceleration_squared']:.12g}"
    )
    print(
        f"Volatility×Volume P95:"
        f" {scales['volatility_volume']:.12g}"
    )

    describe(
        "Normalized Velocity²",
        df["velocity_normalized"],
    )

    describe(
        "Normalized Acceleration²",
        df["acceleration_normalized"],
    )

    describe(
        "Normalized Volatility×Volume",
        df["volatility_volume_normalized"],
    )


if __name__ == "__main__":
    main()