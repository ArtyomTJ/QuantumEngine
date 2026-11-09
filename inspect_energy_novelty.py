from pathlib import Path

import numpy as np
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

    "B: volatility + volatility²": [
        "volatility",
        "volatility_squared",
    ],

    "C: volatility + log(volatility)": [
        "volatility",
        "volatility_log",
    ],

    "D: volatility + velocity component": [
        "volatility",
        "velocity_component",
    ],

    "E: volatility + acceleration component": [
        "volatility",
        "acceleration_component",
    ],

    "F: volatility + vol×volume component": [
        "volatility",
        "volatility_volume_component",
    ],

    "G: volatility + full energy": [
        "volatility",
        "normalized_energy",
    ],

    "H: volatility + energy without velocity": [
        "volatility",
        "energy_without_velocity",
    ],

    "I: volatility + energy without acceleration": [
        "volatility",
        "energy_without_acceleration",
    ],

    "J: volatility + energy without vol×volume": [
        "volatility",
        "energy_without_volatility_volume",
    ],
}


def evaluate_model(
    name,
    features,
    train_df,
    validation_df,
):
    model = LinearRegression()

    model.fit(
        train_df[features],
        train_df["future_abs_return"],
    )

    prediction = model.predict(
        validation_df[features]
    )

    return {
        "name": name,
        "features": features,
        "model": model,
        "r2": r2_score(
            validation_df["future_abs_return"],
            prediction,
        ),
        "mae": mean_absolute_error(
            validation_df["future_abs_return"],
            prediction,
        ),
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

    # Базовые компоненты.
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

    # =========================================================
    # TRAIN-ONLY SCALING
    # =========================================================

    scales = fit_energy_scales(
        train_df,
        quantile=0.95,
    )

    # P95 для самой volatility.
    volatility_p95 = (
        train_df["volatility"]
        .dropna()
        .quantile(0.95)
    )

    # =========================================================
    # NORMALIZED ENERGY COMPONENTS
    # =========================================================

    df = add_normalized_energy(
        df,
        scales,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    # Базовые компоненты energy уже создаются
    # функцией add_normalized_energy:
    #
    # velocity_normalized
    # acceleration_normalized
    # volatility_volume_normalized
    #
    # Переименуем их для читаемости эксперимента.

    df["velocity_component"] = (
        df["velocity_normalized"]
    )

    df["acceleration_component"] = (
        df["acceleration_normalized"]
    )

    df["volatility_volume_component"] = (
        df["volatility_volume_normalized"]
    )

    # =========================================================
    # NON-LINEAR VOLATILITY BASELINES
    # =========================================================

    df["volatility_squared"] = (
        df["volatility"] ** 2
    )

    df["volatility_log"] = np.log1p(
        df["volatility"]
        / volatility_p95
    )

    # =========================================================
    # ENERGY ABLATIONS
    # =========================================================

    df["energy_without_velocity"] = (
        df["acceleration_component"]
        + df["volatility_volume_component"]
    )

    df["energy_without_acceleration"] = (
        df["velocity_component"]
        + df["volatility_volume_component"]
    )

    df["energy_without_volatility_volume"] = (
        df["velocity_component"]
        + df["acceleration_component"]
    )

    # Снова разделяем после добавления признаков.
    train_df = df.iloc[:split_index].copy()
    validation_df = df.iloc[split_index:].copy()

    # =========================================================
    # EVALUATION
    # =========================================================

    print()
    print("=== Energy Novelty Experiment ===")

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
        print(
            f"Features: {', '.join(features)}"
        )
        print(
            f"R²:  {result['r2']:.8f}"
        )
        print(
            f"MAE: {result['mae']:.10f}"
        )

    # =========================================================
    # BASELINE COMPARISON
    # =========================================================

    baseline_r2 = results[0]["r2"]
    baseline_mae = results[0]["mae"]

    print()
    print("=== Relative To Model A ===")

    for result in results:

        delta_r2 = (
            result["r2"]
            - baseline_r2
        )

        delta_mae = (
            result["mae"]
            - baseline_mae
        )

        print()
        print(result["name"])

        print(
            f"ΔR²:  {delta_r2:+.8f}"
        )

        print(
            f"ΔMAE: {delta_mae:+.10f}"
        )

    # =========================================================
    # ENERGY COMPONENT ABLATION
    # =========================================================

    full_energy_r2 = (
        results[6]["r2"]
    )

    print()
    print("=== Energy Component Contributions ===")

    for result in results[7:]:

        delta_from_full = (
            result["r2"]
            - full_energy_r2
        )

        print()
        print(result["name"])

        print(
            "ΔR² relative to full Energy: "
            f"{delta_from_full:+.8f}"
        )

    # =========================================================
    # COEFFICIENTS
    # =========================================================

    print()
    print("=== Model Coefficients ===")

    for result in results:

        print()
        print(result["name"])

        for feature, coefficient in zip(
            result["features"],
            result["model"].coef_,
        ):
            print(
                f"{feature}: "
                f"{coefficient:.10f}"
            )

        print(
            "Intercept: "
            f"{result['model'].intercept_:.10f}"
        )


if __name__ == "__main__":
    main()