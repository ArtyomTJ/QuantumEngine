import numpy as np
import pandas as pd


def calculate_binary_entropy(
    probability_up: float,
) -> float:
    """
    Рассчитывает нормализованную бинарную энтропию.

    H = -p_up * ln(p_up) - p_down * ln(p_down)

    Возвращаемое значение находится в диапазоне [0, 1].
    """

    if not np.isfinite(probability_up):
        raise ValueError(
            "Вероятность должна быть конечным числом."
        )

    if not 0.0 <= probability_up <= 1.0:
        raise ValueError(
            "Вероятность должна находиться в диапазоне [0, 1]."
        )

    probability_down = 1.0 - probability_up

    entropy = 0.0

    if probability_up > 0.0:
        entropy -= probability_up * np.log(probability_up)

    if probability_down > 0.0:
        entropy -= probability_down * np.log(probability_down)

    return float(entropy / np.log(2.0))


def add_entropy(
    df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Добавляет вероятности движения вверх/вниз
    и нормализованную энтропию рынка.

    Вероятности рассчитываются только по прошлым
    наблюдениям текущего rolling-окна.
    """

    if window < 2:
        raise ValueError("window должен быть >= 2.")

    if "log_return" not in df.columns:
        raise ValueError(
            "Отсутствует обязательный столбец log_return."
        )

    result = df.copy()

    up = (
        result["log_return"] > 0
    ).astype(float)

    valid_returns = result["log_return"].notna()

    up_probability = (
        up.where(valid_returns)
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )

    result["probability_up"] = up_probability
    result["probability_down"] = 1.0 - up_probability

    result["entropy"] = (
        result["probability_up"]
        .apply(
            lambda probability: (
                calculate_binary_entropy(probability)
                if pd.notna(probability)
                else np.nan
            )
        )
    )

    return result