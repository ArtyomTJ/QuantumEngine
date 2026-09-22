import numpy as np
import pandas as pd


def calculate_energy(
    velocity: float,
    acceleration: float,
    volatility: float,
    volume: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """
    Рассчитывает экспериментальную энергию рынка.

    E = alpha * v^2 + beta * a^2 + gamma * sigma * V

    Это модельная гипотеза Quantum Engine,
    а не физический закон.
    """

    values = [
        velocity,
        acceleration,
        volatility,
        volume,
        alpha,
        beta,
        gamma,
    ]

    if not all(np.isfinite(value) for value in values):
        raise ValueError("Параметры энергии должны быть конечными числами.")

    if volatility < 0:
        raise ValueError("Volatility не может быть отрицательной.")

    if volume < 0:
        raise ValueError("Volume не может быть отрицательным.")

    return (
        alpha * velocity**2
        + beta * acceleration**2
        + gamma * volatility * volume
    )


def add_energy(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> pd.DataFrame:
    """
    Добавляет столбец energy к DataFrame.
    """

    required_columns = {
        "log_return",
        "acceleration",
        "volatility",
        "volume",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy()

    result["energy"] = np.nan

    valid_mask = result[
        [
            "log_return",
            "acceleration",
            "volatility",
            "volume",
        ]
    ].notna().all(axis=1)

    result.loc[valid_mask, "energy"] = (
        result.loc[valid_mask]
        .apply(
            lambda row: calculate_energy(
                velocity=row["log_return"],
                acceleration=row["acceleration"],
                volatility=row["volatility"],
                volume=row["volume"],
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            ),
            axis=1,
        )
    )

    return result