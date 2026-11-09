from pathlib import Path

import numpy as np

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


def describe(name, series):
    values = series.dropna()

    print()
    print(f"=== {name} ===")
    print(f"Среднее:      {values.mean():.12g}")
    print(f"Std:          {values.std():.12g}")
    print(f"Медиана:      {values.median():.12g}")
    print(f"P95:          {values.quantile(0.95):.12g}")
    print(f"P99:          {values.quantile(0.99):.12g}")
    print(f"Максимум:     {values.max():.12g}")


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
    df["volatility_volume"] = df["volatility"] * df["volume"]

    print()
    print("=== Quantum Engine: Energy Components ===")

    describe(
        "Velocity²",
        df["velocity_squared"],
    )

    describe(
        "Acceleration²",
        df["acceleration_squared"],
    )

    describe(
        "Volatility × Volume",
        df["volatility_volume"],
    )


if __name__ == "__main__":
    main()