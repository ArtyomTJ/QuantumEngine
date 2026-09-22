from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyScales:
    velocity_p95: float
    acceleration_p95: float
    volatility_volume_p95: float


REQUIRED_COLUMNS = {
    "log_return",
    "acceleration",
    "volatility",
    "volume",
}


def _validate_columns(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: "
            f"{sorted(missing_columns)}"
        )


def fit_energy_scales(
    df: pd.DataFrame,
    quantile: float = 0.95,
) -> EnergyScales:
    """
    Рассчитывает масштабы нормализации на переданном наборе данных.

    В дальнейшем этот набор должен быть только train-частью.
    """

    if not 0 < quantile < 1:
        raise ValueError("quantile должен находиться между 0 и 1.")

    _validate_columns(df)

    velocity_squared = df["log_return"] ** 2
    acceleration_squared = df["acceleration"] ** 2
    volatility_volume = df["volatility"] * df["volume"]

    scales = EnergyScales(
        velocity_p95=float(
            velocity_squared.dropna().quantile(quantile)
        ),
        acceleration_p95=float(
            acceleration_squared.dropna().quantile(quantile)
        ),
        volatility_volume_p95=float(
            volatility_volume.dropna().quantile(quantile)
        ),
    )

    if (
        scales.velocity_p95 <= 0
        or scales.acceleration_p95 <= 0
        or scales.volatility_volume_p95 <= 0
    ):
        raise ValueError(
            "Масштабы энергии должны быть положительными."
        )

    return scales


def add_normalized_energy(
    df: pd.DataFrame,
    scales: EnergyScales,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> pd.DataFrame:
    """
    Добавляет нормализованные компоненты и normalized_energy.

    Нормализация:
        x_normalized = x / P95(x)

    Сжатие хвоста:
        log(1 + x_normalized)
    """

    _validate_columns(df)

    parameters = [alpha, beta, gamma]

    if not all(np.isfinite(value) for value in parameters):
        raise ValueError("Коэффициенты энергии должны быть конечными.")

    result = df.copy()

    velocity_squared = result["log_return"] ** 2
    acceleration_squared = result["acceleration"] ** 2
    volatility_volume = (
        result["volatility"] * result["volume"]
    )

    result["velocity_normalized"] = np.log1p(
        velocity_squared / scales.velocity_p95
    )

    result["acceleration_normalized"] = np.log1p(
        acceleration_squared / scales.acceleration_p95
    )

    result["volatility_volume_normalized"] = np.log1p(
        volatility_volume / scales.volatility_volume_p95
    )

    result["normalized_energy"] = (
        alpha * result["velocity_normalized"]
        + beta * result["acceleration_normalized"]
        + gamma * result["volatility_volume_normalized"]
    )

    return result