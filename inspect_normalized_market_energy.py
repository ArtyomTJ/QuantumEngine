from pathlib import Path

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics
from src.energy_normalization import (
    fit_energy_scales,
    add_normalized_energy,
)


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


def describe(name, series):
    values = series.dropna()

    print()
    print(f"=== {name} ===")
    print(f"Количество:   {len(values):,}")
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

    df = df.dropna(
        subset=[
            "log_return",
            "acceleration",
            "volatility",
        ]
    ).reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    validation_df = df.iloc[split_index:].copy()

    print()
    print("=== Temporal Split ===")
    print(f"Всего состояний:    {len(df):,}")
    print(f"Train:              {len(train_df):,}")
    print(f"Validation:         {len(validation_df):,}")

    print()
    print("Train start:")
    print(train_df["timestamp"].iloc[0])

    print("Train end:")
    print(train_df["timestamp"].iloc[-1])

    print()
    print("Validation start:")
    print(validation_df["timestamp"].iloc[0])

    print("Validation end:")
    print(validation_df["timestamp"].iloc[-1])

    print()
    print("Расчёт energy scales только на TRAIN...")

    scales = fit_energy_scales(
        train_df,
        quantile=0.95,
    )

    print()
    print("=== Energy Scales ===")
    print(f"Velocity² P95:         {scales.velocity_p95:.12g}")
    print(
        "Acceleration² P95:     "
        f"{scales.acceleration_p95:.12g}"
    )
    print(
        "Volatility×Volume P95: "
        f"{scales.volatility_volume_p95:.12g}"
    )

    print()
    print("Применение scales ко всему временному ряду...")

    normalized_df = add_normalized_energy(
        df,
        scales,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    normalized_train = normalized_df.iloc[:split_index]

    normalized_validation = normalized_df.iloc[split_index:]

    describe(
        "TRAIN normalized energy",
        normalized_train["normalized_energy"],
    )

    describe(
        "VALIDATION normalized energy",
        normalized_validation["normalized_energy"],
    )

    print()
    print("=== Last 5 Validation States ===")

    print(
        normalized_validation[
            [
                "timestamp",
                "close",
                "log_return",
                "acceleration",
                "volatility",
                "volume",
                "normalized_energy",
            ]
        ].tail(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()