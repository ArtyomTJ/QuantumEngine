"""Контейнеры данных для feature/target pipeline."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    """Готовые для ML признаки, target и временная привязка."""

    timestamps: pd.Series
    features: pd.DataFrame
    target_return: pd.Series | None
    target_direction: pd.Series | None
    feature_columns: tuple[str, ...]
