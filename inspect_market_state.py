from pathlib import Path

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics
from src.market_state import build_market_state


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


def main() -> None:
    print("Загрузка данных...")
    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт рыночной динамики...")
    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    # Берём последнюю строку, для которой все признаки уже рассчитаны.
    row = df.dropna(
        subset=[
            "log_return",
            "acceleration",
            "volatility",
        ]
    ).iloc[-1]

    state = build_market_state(row)

    print()
    print("=== Quantum Engine: Real Market State ===")
    print(f"Timestamp:     {state.timestamp}")
    print(f"Price:         {state.price:.2f}")
    print(f"Velocity:      {state.velocity:.10f}")
    print(f"Acceleration:  {state.acceleration:.10f}")
    print(f"Volatility:    {state.volatility:.10f}")
    print(f"Volume:        {state.volume:.6f}")


if __name__ == "__main__":
    main()