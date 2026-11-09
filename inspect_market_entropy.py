from pathlib import Path

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics
from src.market_entropy import add_entropy
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
    print(f"P05:          {values.quantile(0.05):.6f}")
    print(f"P95:          {values.quantile(0.95):.6f}")
    print(f"Минимум:      {values.min():.6f}")
    print(f"Максимум:     {values.max():.6f}")


def main():
    print("Загрузка данных...")

    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт динамики...")

    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    print("Расчёт энтропии...")

    df = add_entropy(
        df,
        window=60,
    )

    # Для энергии используем только train-часть
    # при расчёте масштабов.
    df = df.dropna(
        subset=[
            "log_return",
            "acceleration",
            "volatility",
            "entropy",
        ]
    ).reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index]
    validation_df = df.iloc[split_index:]

    scales = fit_energy_scales(
        train_df,
        quantile=0.95,
    )

    df = add_normalized_energy(
        df,
        scales,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    train_df = df.iloc[:split_index]
    validation_df = df.iloc[split_index:]

    describe(
        "TRAIN Entropy",
        train_df["entropy"],
    )

    describe(
        "VALIDATION Entropy",
        validation_df["entropy"],
    )

    describe(
        "ALL Entropy",
        df["entropy"],
    )

    print()
    print("=== Energy / Entropy Relationship ===")

    correlation = df[
        [
            "normalized_energy",
            "entropy",
        ]
    ].corr().iloc[0, 1]

    print(
        "Pearson correlation "
        f"(energy, entropy): {correlation:.6f}"
    )

    print()
    print("=== Market Regimes ===")

    low_entropy = df["entropy"] < 0.25
    medium_entropy = (
        (df["entropy"] >= 0.25)
        & (df["entropy"] < 0.75)
    )
    high_entropy = df["entropy"] >= 0.75

    print(
        f"Low entropy (<0.25):       "
        f"{low_entropy.sum():,}"
    )

    print(
        f"Medium entropy (0.25-0.75): "
        f"{medium_entropy.sum():,}"
    )

    print(
        f"High entropy (>=0.75):     "
        f"{high_entropy.sum():,}"
    )

    print()
    print("=== Last 10 States ===")

    print(
        df[
            [
                "timestamp",
                "close",
                "normalized_energy",
                "probability_up",
                "probability_down",
                "entropy",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()