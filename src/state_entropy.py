import numpy as np
import pandas as pd


STATE_NAMES = (
    "strong_down",
    "down",
    "neutral",
    "up",
    "strong_up",
)


def calculate_multistate_entropy(
    probabilities: list[float],
) -> float:
    """
    Нормализованная Shannon entropy
    для произвольного распределения состояний.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probabilities должны быть одномерным массивом."
        )

    if len(probabilities) < 2:
        raise ValueError(
            "Необходимо минимум два состояния."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Вероятности должны быть конечными числами."
        )

    if np.any(probabilities < 0):
        raise ValueError(
            "Вероятности не могут быть отрицательными."
        )

    total = probabilities.sum()

    if not np.isclose(total, 1.0):
        raise ValueError(
            "Сумма вероятностей должна быть равна 1."
        )

    entropy = 0.0

    for probability in probabilities:
        if probability > 0:
            entropy -= probability * np.log(probability)

    max_entropy = np.log(len(probabilities))

    return float(entropy / max_entropy)


def classify_return(
    z_score: float,
    strong_threshold: float = 1.0,
    neutral_threshold: float = 0.25,
) -> str:
    """
    Классифицирует нормализованную доходность
    в одно из пяти состояний.
    """

    if not np.isfinite(z_score):
        raise ValueError(
            "z_score должен быть конечным числом."
        )

    if strong_threshold <= neutral_threshold:
        raise ValueError(
            "strong_threshold должен быть больше "
            "neutral_threshold."
        )

    if z_score <= -strong_threshold:
        return "strong_down"

    if z_score < -neutral_threshold:
        return "down"

    if z_score <= neutral_threshold:
        return "neutral"

    if z_score < strong_threshold:
        return "up"

    return "strong_up"


def add_state_entropy(
    df: pd.DataFrame,
    window: int = 60,
    strong_threshold: float = 1.0,
    neutral_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Добавляет z-score доходности, состояние рынка
    и многосостоянийную энтропию.
    """

    if window < 2:
        raise ValueError(
            "window должен быть >= 2."
        )

    required_columns = {
        "log_return",
        "volatility",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy()

    result["return_zscore"] = (
        result["log_return"]
        / result["volatility"]
    )

    result["market_direction_state"] = (
        result["return_zscore"].apply(
            lambda value: (
                classify_return(
                    value,
                    strong_threshold=strong_threshold,
                    neutral_threshold=neutral_threshold,
                )
                if pd.notna(value)
                else np.nan
            )
        )
    )

    for state_name in STATE_NAMES:
        state_indicator = (
            result["market_direction_state"]
            == state_name
        ).astype(float)

        state_indicator = state_indicator.where(
            result["market_direction_state"].notna()
        )

        result[f"state_prob_{state_name}"] = (
            state_indicator
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

    probability_columns = [
        f"state_prob_{state}"
        for state in STATE_NAMES
    ]

    def entropy_from_row(row):
        probabilities = row.values

        if np.isnan(probabilities).any():
            return np.nan

        return calculate_multistate_entropy(
            probabilities.tolist()
        )

    result["state_entropy"] = (
        result[probability_columns]
        .apply(
            entropy_from_row,
            axis=1,
        )
    )

    return result