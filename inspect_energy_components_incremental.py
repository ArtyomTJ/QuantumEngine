from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics
from src.energy_normalization import (
    fit_energy_scales,
    add_normalized_energy,
)
from src.forecast_target import add_future_return


DATA_PATH = Path(
    "data/raw/BTCUSDT-1m-2026-08.csv"
)


MODELS = {
    "A: volatility": [
        "volatility",
    ],
    "B: volatility + velocity²": [
        "volatility",
        "velocity_squared",
    ],
    "C: volatility + acceleration²": [
        "volatility",
        "acceleration_squared",
    ],
    "D: volatility + volatility×volume": [
        "volatility",
        "volatility_volume",
    ],
    "E: volatility + normalized_energy": [
        "volatility",
        "normalized_energy",
    ],
}


def evaluate_model(
    name: str,
    features: list[str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    model = LinearRegression()

    model.fit(
        train_df[features],
        train_df["future_abs_return"],
    )

    prediction = model.predict(
        validation_df[features]
    )

    r2 = r2_score(
        validation_df["future_abs_return"],
        prediction,
    )

    mae = mean_absolute_error(
        validation_df["future_abs_return"],
        prediction,
    )

    return {
        "name": name,
        "r2": r2,
        "mae": mae,
        "model": model,
    }


def main():
    print("Загрузка данных...")

    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт динамики...")

    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    print("Расчёт будущего движения...")

    df = add_future_return(
        df,
        horizon=5,
    )

    df["future_abs_return"] = (
        df["future_return"].abs()
    )

    print("Подготовка компонентов...")

    df["velocity_squared"] = (
        df["log_return"] ** 2
    )

    df["acceleration_squared"] = (
        df["acceleration"] ** 2
    )

    df["volatility_volume"] = (
        df["volatility"]
        * df["volume"]
    )

    df = df.dropna(
        subset=[
            "volatility",
            "velocity_squared",
            "acceleration_squared",
            "volatility_volume",
            "future_abs_return",
        ]
    ).reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    validation_df = df.iloc[split_index:].copy()

    print()
    print("=== Temporal Split ===")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")

    print()
    print("Расчёт energy scales только на train...")

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

    train_df = df.iloc[:split_index].copy()
    validation_df = df.iloc[split_index:].copy()

    print()
    print("=== Incremental Model Comparison ===")

    results = []

    for name, features in MODELS.items():
        result = evaluate_model(
            name,
            features,
            train_df,
            validation_df,
        )

        results.append(result)

        print()
        print(name)
        print(f"Features: {', '.join(features)}")
        print(f"R²:       {result['r2']:.8f}")
        print(f"MAE:      {result['mae']:.10f}")

    baseline_r2 = results[0]["r2"]
    baseline_mae = results[0]["mae"]

    print()
    print("=== Improvement Relative To Model A ===")

    for result in results:
        delta_r2 = result["r2"] - baseline_r2
        delta_mae = result["mae"] - baseline_mae

        print()
        print(result["name"])
        print(f"ΔR²:  {delta_r2:+.8f}")
        print(f"ΔMAE: {delta_mae:+.10f}")

    print()
    print("=== Model Coefficients ===")

    for result in results:
        model = result["model"]

        print()
        print(result["name"])

        for feature, coefficient in zip(
            MODELS[result["name"]],
            model.coef_,
        ):
            print(
                f"{feature}: "
                f"{coefficient:.10f}"
            )

        print(
            f"Intercept: "
            f"{model.intercept_:.10f}"
        )


if __name__ == "__main__":
    main()