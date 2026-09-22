from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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

HORIZON = 5


CANDIDATES = {
    "A: volatility": [
        "volatility",
    ],
    "B: volatility + log(volatility)": [
        "volatility",
        "volatility_log",
    ],
    "C: volatility + full energy": [
        "volatility",
        "normalized_energy",
    ],
    "D: volatility + energy without velocity": [
        "volatility",
        "energy_without_velocity",
    ],
}


def add_experimental_features(
    df: pd.DataFrame,
    scales,
) -> pd.DataFrame:
    """
    Добавляет признаки для эксперимента.

    Все параметры масштабирования уже должны быть
    рассчитаны на соответствующей train-части.
    """

    result = add_normalized_energy(
        df,
        scales,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    # Масштаб volatility для log(volatility).
    volatility_p95 = (
        result["volatility"]
        .dropna()
        .quantile(0.95)
    )

    if volatility_p95 <= 0:
        raise ValueError(
            "Volatility P95 должен быть положительным."
        )

    result["volatility_log"] = np.log1p(
        result["volatility"] / volatility_p95
    )

    result["energy_without_velocity"] = (
        result["acceleration_normalized"]
        + result["volatility_volume_normalized"]
    )

    return result


def fit_and_evaluate(
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

    r2 = r2_score(
        validation_df["future_abs_return"],
        prediction,
    )

    mae = mean_absolute_error(
        validation_df["future_abs_return"],
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            validation_df["future_abs_return"],
            prediction,
        )
    )

    return {
        "model": model,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
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
        horizon=HORIZON,
    )

    df["future_abs_return"] = (
        df["future_return"].abs()
    )

    # Убираем строки, где базовые признаки ещё
    # не рассчитаны или отсутствует будущее.
    df = df.dropna(
        subset=[
            "volatility",
            "future_return",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # TEMPORAL SPLIT
    # ---------------------------------------------------------

    n = len(df)

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    # Не позволяем target последних HORIZON
    # строк train/validation пересекать границу.
    train_end_safe = train_end - HORIZON
    validation_end_safe = validation_end - HORIZON

    train_raw = df.iloc[:train_end_safe].copy()

    validation_raw = df.iloc[
        train_end:validation_end_safe
    ].copy()

    test_raw = df.iloc[
        validation_end:
    ].copy()

    print()
    print("=== Temporal Split ===")

    print(
        f"Train:      {len(train_raw):,}"
    )

    print(
        f"Validation: {len(validation_raw):,}"
    )

    print(
        f"Test:       {len(test_raw):,}"
    )

    print()
    print("=== Date Ranges ===")

    print(
        f"Train: "
        f"{train_raw['timestamp'].iloc[0]} "
        f"→ "
        f"{train_raw['timestamp'].iloc[-1]}"
    )

    print(
        f"Validation: "
        f"{validation_raw['timestamp'].iloc[0]} "
        f"→ "
        f"{validation_raw['timestamp'].iloc[-1]}"
    )

    print(
        f"Test: "
        f"{test_raw['timestamp'].iloc[0]} "
        f"→ "
        f"{test_raw['timestamp'].iloc[-1]}"
    )

    # ---------------------------------------------------------
    # TRAIN-ONLY SCALING
    # ---------------------------------------------------------

    print()
    print(
        "Расчёт energy scales только на TRAIN..."
    )

    train_scales = fit_energy_scales(
        train_raw,
        quantile=0.95,
    )

    train_df = add_experimental_features(
        train_raw,
        train_scales,
    )

    validation_df = add_experimental_features(
        validation_raw,
        train_scales,
    )

    test_df = add_experimental_features(
        test_raw,
        train_scales,
    )

    # ---------------------------------------------------------
    # VALIDATION MODEL SELECTION
    # ---------------------------------------------------------

    print()
    print(
        "=== Validation Model Selection ==="
    )

    validation_results = {}

    for name, features in CANDIDATES.items():
        result = fit_and_evaluate(
            features,
            train_df,
            validation_df,
        )

        validation_results[name] = result

        print()
        print(name)
        print(
            f"R²:   {result['r2']:.8f}"
        )
        print(
            f"MAE:  {result['mae']:.10f}"
        )
        print(
            f"RMSE: {result['rmse']:.10f}"
        )

    # Главный критерий выбора — минимальный RMSE.
    # При равенстве используем максимальный R².
    selected_name = min(
        validation_results,
        key=lambda name: (
            validation_results[name]["rmse"],
            -validation_results[name]["r2"],
        ),
    )

    selected_features = CANDIDATES[
        selected_name
    ]

    print()
    print("=== SELECTED MODEL ===")
    print(selected_name)
    print(
        "Features: "
        + ", ".join(selected_features)
    )

    # ---------------------------------------------------------
    # FINAL TRAINING SET
    # ---------------------------------------------------------

    print()
    print(
        "Переобучение выбранной модели "
        "на TRAIN + VALIDATION..."
    )

    combined_raw = pd.concat(
        [
            train_raw,
            validation_raw,
        ],
        ignore_index=True,
    )

    # Теперь, когда выбор модели уже закончен,
    # разрешено рассчитать масштабы на TRAIN+VALIDATION.
    final_scales = fit_energy_scales(
        combined_raw,
        quantile=0.95,
    )

    combined_df = add_experimental_features(
        combined_raw,
        final_scales,
    )

    final_test_df = add_experimental_features(
        test_raw,
        final_scales,
    )

    final_model = LinearRegression()

    final_model.fit(
        combined_df[selected_features],
        combined_df["future_abs_return"],
    )

    # ---------------------------------------------------------
    # FINAL TEST
    # ---------------------------------------------------------

    test_prediction = final_model.predict(
        final_test_df[selected_features]
    )

    test_actual = (
        final_test_df["future_abs_return"]
    )

    test_r2 = r2_score(
        test_actual,
        test_prediction,
    )

    test_mae = mean_absolute_error(
        test_actual,
        test_prediction,
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            test_actual,
            test_prediction,
        )
    )

    print()
    print("=== FINAL TEST RESULT ===")

    print(
        f"Model: {selected_name}"
    )

    print(
        f"R²:   {test_r2:.8f}"
    )

    print(
        f"MAE:  {test_mae:.10f}"
    )

    print(
        f"RMSE: {test_rmse:.10f}"
    )

    # ---------------------------------------------------------
    # TEST CORRELATION
    # ---------------------------------------------------------

    test_frame = pd.DataFrame(
        {
            "actual": test_actual.to_numpy(),
            "prediction": test_prediction,
        }
    )

    pearson = test_frame.corr(
        method="pearson"
    ).iloc[0, 1]

    spearman = test_frame.corr(
        method="spearman"
    ).iloc[0, 1]

    print()
    print("=== TEST PREDICTION CORRELATION ===")

    print(
        f"Pearson:  {pearson:.8f}"
    )

    print(
        f"Spearman: {spearman:.8f}"
    )

    # ---------------------------------------------------------
    # COEFFICIENTS
    # ---------------------------------------------------------

    print()
    print("=== FINAL MODEL COEFFICIENTS ===")

    for feature, coefficient in zip(
        selected_features,
        final_model.coef_,
    ):
        print(
            f"{feature}: "
            f"{coefficient:.10f}"
        )

    print(
        "Intercept: "
        f"{final_model.intercept_:.10f}"
    )


if __name__ == "__main__":
    main()