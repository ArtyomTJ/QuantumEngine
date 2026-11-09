import numpy as np
import pandas as pd


def add_future_return(
    df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Добавляет фактическую будущую log-return
    на заданном горизонте.

    future_return = log(close[t+h] / close[t])

    Последние horizon строк не имеют будущего
    наблюдения и получают NaN.
    """

    if horizon < 1:
        raise ValueError("horizon должен быть >= 1.")

    required_columns = {"timestamp", "close"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: "
            f"{sorted(missing_columns)}"
        )

    if (df["close"] <= 0).any():
        raise ValueError(
            "Цена close должна быть положительной."
        )

    result = df.copy()

    result["future_return"] = np.log(
        result["close"].shift(-horizon)
        / result["close"]
    )

    result["future_direction"] = np.select(
        [
            result["future_return"] > 0,
            result["future_return"] < 0,
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    return result