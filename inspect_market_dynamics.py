from pathlib import Path

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


def main() -> None:
    print("Загрузка данных...")

    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт динамики...")

    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    print()
    print("=== Quantum Engine: Market Dynamics ===")

    print(f"Количество свечей: {len(df):,}")
    print(f"Начало:            {df['timestamp'].iloc[0]}")
    print(f"Конец:             {df['timestamp'].iloc[-1]}")

    valid_returns = df["log_return"].dropna()
    valid_acceleration = df["acceleration"].dropna()
    valid_volatility = df["volatility"].dropna()

    print()
    print("=== Log Return ===")
    print(f"Среднее:            {valid_returns.mean():.10f}")
    print(f"Std:                {valid_returns.std():.10f}")
    print(f"Минимум:            {valid_returns.min():.10f}")
    print(f"Максимум:           {valid_returns.max():.10f}")

    print()
    print("=== Acceleration ===")
    print(f"Среднее:            {valid_acceleration.mean():.10f}")
    print(f"Std:                {valid_acceleration.std():.10f}")
    print(f"Минимум:            {valid_acceleration.min():.10f}")
    print(f"Максимум:           {valid_acceleration.max():.10f}")

    print()
    print("=== Volatility (60 min) ===")
    print(f"Среднее:            {valid_volatility.mean():.10f}")
    print(f"Std:                {valid_volatility.std():.10f}")
    print(f"Минимум:            {valid_volatility.min():.10f}")
    print(f"Максимум:           {valid_volatility.max():.10f}")

    print()
    print("=== Последние 5 состояний ===")

    print(
        df[
            [
                "timestamp",
                "close",
                "log_return",
                "acceleration",
                "volatility",
            ]
        ].tail(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()