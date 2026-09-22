from pathlib import Path

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics
from src.market_energy import add_energy


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


def main() -> None:
    print("Загрузка данных...")
    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт динамики...")
    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    print("Расчёт энергии...")
    df = add_energy(
        df,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    valid = df.dropna(
        subset=[
            "log_return",
            "acceleration",
            "volatility",
            "energy",
        ]
    )

    print()
    print("=== Quantum Engine: Market Energy ===")
    print(f"Количество состояний: {len(valid):,}")

    print()
    print("=== Energy ===")
    print(f"Среднее:   {valid['energy'].mean():.10f}")
    print(f"Std:       {valid['energy'].std():.10f}")
    print(f"Минимум:   {valid['energy'].min():.10f}")
    print(f"Максимум:  {valid['energy'].max():.10f}")

    print()
    print("=== Last 5 states ===")

    print(
        valid[
            [
                "timestamp",
                "close",
                "log_return",
                "acceleration",
                "volatility",
                "volume",
                "energy",
            ]
        ].tail(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()