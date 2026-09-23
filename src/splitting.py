"""Детерминированные временные разбиения без перемешивания."""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int
    validation_size: int
    test_size: int | None
    step_size: int
    embargo: int

    def __post_init__(self) -> None:
        for name in (
            "train_size",
            "validation_size",
            "step_size",
            "embargo",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} должен быть целым числом.")
        if self.test_size is not None and (
            isinstance(self.test_size, bool)
            or not isinstance(self.test_size, Integral)
        ):
            raise TypeError("test_size должен быть целым числом или None.")

        if self.train_size < 1:
            raise ValueError("train_size должен быть >= 1.")
        if self.validation_size < 1:
            raise ValueError("validation_size должен быть >= 1.")
        if self.test_size is not None and self.test_size < 1:
            raise ValueError("test_size должен быть >= 1 или None.")
        if self.step_size < 1:
            raise ValueError("step_size должен быть >= 1.")
        if self.embargo < 0:
            raise ValueError("embargo не может быть отрицательным.")


@dataclass(frozen=True)
class TemporalSplit:
    """Один упорядоченный fold с явными индексами выборок."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...] | None

    @property
    def train_start(self) -> int:
        return self.train_indices[0]

    @property
    def train_end(self) -> int:
        return self.train_indices[-1] + 1

    @property
    def validation_start(self) -> int:
        return self.validation_indices[0]

    @property
    def validation_end(self) -> int:
        return self.validation_indices[-1] + 1

    @property
    def test_start(self) -> int | None:
        if self.test_indices is None:
            return None
        return self.test_indices[0]

    @property
    def test_end(self) -> int | None:
        if self.test_indices is None:
            return None
        return self.test_indices[-1] + 1


class WalkForwardSplitter:
    """Создаёт rolling временные folds с embargo между выборками."""

    def __init__(
        self,
        config: WalkForwardConfig,
        target_horizon: int = 5,
    ) -> None:
        if target_horizon < 1:
            raise ValueError("target_horizon должен быть >= 1.")
        if config.embargo < target_horizon:
            raise ValueError(
                "embargo должен быть не меньше target_horizon."
            )

        self.config = config
        self.target_horizon = target_horizon

    def split(self, timestamps: pd.Series) -> Iterator[TemporalSplit]:
        """Возвращает только полностью помещающиеся во входные данные folds."""

        values = pd.Series(timestamps).reset_index(drop=True)

        if values.empty:
            return

        converted = pd.to_datetime(values, utc=True)

        if converted.duplicated().any():
            raise ValueError("timestamps не должны содержать дубликаты.")
        if not converted.is_monotonic_increasing:
            raise ValueError("timestamps должны быть отсортированы по возрастанию.")

        start = 0
        total = len(converted)

        while True:
            train_start = start
            train_end = train_start + self.config.train_size
            validation_start = train_end + self.config.embargo
            validation_end = validation_start + self.config.validation_size

            if validation_end > total:
                return

            test_indices: tuple[int, ...] | None = None

            if self.config.test_size is not None:
                test_start = validation_end + self.config.embargo
                test_end = test_start + self.config.test_size

                if test_end > total:
                    return

                test_indices = tuple(range(test_start, test_end))

            yield TemporalSplit(
                train_indices=tuple(range(train_start, train_end)),
                validation_indices=tuple(
                    range(validation_start, validation_end)
                ),
                test_indices=test_indices,
            )

            start += self.config.step_size
