from pathlib import Path

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics
from src.market_entropy import add_entropy
from src.state_entropy import add_state_entropy


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
    print(f"P99:          {values.quantile(0.99):.6f}")
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

    print("Расчёт бинарной энтропии...")

    df = add_entropy(
        df,
        window=60,
    )

    print("Расчёт многосостоянийной энтропии...")

    df = add_state_entropy(
        df,
        window=60,
    )

    print()

    describe(
        "Directional Entropy",
        df["entropy"],
    )

    describe(
        "State Entropy",
        df["state_entropy"],
    )

    valid = df[
        [
            "entropy",
            "state_entropy",
        ]
    ].dropna()

    print()
    print("=== Entropy Correlation ===")

    correlation = valid.corr().iloc[0, 1]

    print(
        f"Pearson correlation: {correlation:.6f}"
    )

    print()
    print("=== Directional Entropy Regimes ===")

    directional = valid["entropy"]

    print(
        f"Low (<0.25):       "
        f"{(directional < 0.25).sum():,}"
    )

    print(
        f"Medium (0.25-0.75): "
        f"{((directional >= 0.25) & (directional < 0.75)).sum():,}"
    )

    print(
        f"High (>=0.75):     "
        f"{(directional >= 0.75).sum():,}"
    )

    print()
    print("=== State Entropy Regimes ===")

    state_entropy = valid["state_entropy"]

    print(
        f"Low (<0.25):       "
        f"{(state_entropy < 0.25).sum():,}"
    )

    print(
        f"Medium (0.25-0.75): "
        f"{((state_entropy >= 0.25) & (state_entropy < 0.75)).sum():,}"
    )

    print(
        f"High (>=0.75):     "
        f"{(state_entropy >= 0.75).sum():,}"
    )

    print()
    print("=== Last 10 States ===")

    print(
        df[
            [
                "timestamp",
                "close",
                "log_return",
                "return_zscore",
                "market_direction_state",
                "entropy",
                "state_entropy",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()