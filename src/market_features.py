import numpy as np
import pandas as pd


def build_price_dynamics(
    df: pd.DataFrame,
    volatility_window: int = 60,
) -> pd.DataFrame:
    """
    Рассчитывает базовые динамические характеристики цены.

    log_return:
        логарифмическая доходность.

    acceleration:
        изменение логарифмической доходности.

    volatility:
        скользящее стандартное отклонение логарифмической доходности.
    """

    if volatility_window < 2:
        raise ValueError("volatility_window должен быть >= 2.")

    required_columns = {"timestamp", "close"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: "
            f"{sorted(missing_columns)}"
        )

    if (df["close"] <= 0).any():
        raise ValueError("Цена close должна быть положительной.")

    result = df.copy()

    result["log_return"] = np.log(
        result["close"] / result["close"].shift(1)
    )

    result["acceleration"] = result["log_return"].diff()

    result["volatility"] = (
        result["log_return"]
        .rolling(
            window=volatility_window,
            min_periods=volatility_window,
        )
        .std()
    )

    return result