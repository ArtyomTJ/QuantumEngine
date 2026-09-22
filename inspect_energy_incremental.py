from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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


def main():
    print("Загрузка данных...")

    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт динамики...")

    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    print("Расчёт 5m future return...")

    df = add_future_return(
        df,
        horizon=5,
    )

    df["future_abs_return"] = (
        df["future_return"].abs()
    )

    df = df.dropna(
        subset=[
            "volatility",
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

    target = "future_abs_return"

    x_train_baseline = train_df[
        ["volatility"]
    ]

    x_validation_baseline = validation_df[
        ["volatility"]
    ]

    x_train_extended = train_df[
        [
            "volatility",
            "normalized_energy",
        ]
    ]

    x_validation_extended = validation_df[
        [
            "volatility",
            "normalized_energy",
        ]
    ]

    y_train = train_df[target]
    y_validation = validation_df[target]

    print()
    print("=== Model A: Volatility only ===")

    model_a = LinearRegression()

    model_a.fit(
        x_train_baseline,
        y_train,
    )

    prediction_a = model_a.predict(
        x_validation_baseline
    )

    r2_a = r2_score(
        y_validation,
        prediction_a,
    )

    mae_a = mean_absolute_error(
        y_validation,
        prediction_a,
    )

    print(f"R²:  {r2_a:.8f}")
    print(f"MAE: {mae_a:.10f}")

    print()
    print("=== Model B: Volatility + Energy ===")

    model_b = LinearRegression()

    model_b.fit(
        x_train_extended,
        y_train,
    )

    prediction_b = model_b.predict(
        x_validation_extended
    )

    r2_b = r2_score(
        y_validation,
        prediction_b,
    )

    mae_b = mean_absolute_error(
        y_validation,
        prediction_b,
    )

    print(f"R²:  {r2_b:.8f}")
    print(f"MAE: {mae_b:.10f}")

    print()
    print("=== Incremental Information ===")

    print(
        f"ΔR²:  {r2_b - r2_a:.8f}"
    )

    print(
        f"ΔMAE: {mae_b - mae_a:.10f}"
    )

    print()
    print("=== Model B coefficients ===")

    for name, coefficient in zip(
        x_train_extended.columns,
        model_b.coef_,
    ):
        print(
            f"{name}: {coefficient:.10f}"
        )

    print()
    print("=== Energy residual correlation ===")

    volatility_model = LinearRegression()

    volatility_model.fit(
        train_df[["volatility"]],
        y_train,
    )

    train_pred = volatility_model.predict(
        train_df[["volatility"]]
    )

    validation_pred = volatility_model.predict(
        validation_df[["volatility"]]
    )

    train_residual = (
        y_train.to_numpy() - train_pred
    )

    validation_residual = (
        y_validation.to_numpy()
        - validation_pred
    )

    validation_energy = (
        validation_df["normalized_energy"]
        .to_numpy()
    )

    correlation = pearsonr(
        validation_energy,
        validation_residual,
    ).statistic

    print(
        "Correlation energy / "
        f"volatility residual: {correlation:.8f}"
    )


if __name__ == "__main__":
    main()