"""Единый детерминированный feature/target pipeline QuantumEngine."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_audit import audit_ohlcv
from src.datasets import PreparedDataset
from src.energy_normalization import (
    EnergyScales,
    add_normalized_energy,
    fit_energy_scales,
)
from src.forecast_target import add_future_return
from src.market_energy import add_energy
from src.market_entropy import add_entropy
from src.market_features import build_price_dynamics
from src.state_entropy import STATE_NAMES, add_state_entropy


@dataclass(frozen=True)
class PipelineConfig:
    volatility_window: int = 60
    entropy_window: int = 60
    state_entropy_window: int = 60
    energy_quantile: float = 0.95
    energy_alpha: float = 1.0
    energy_beta: float = 1.0
    energy_gamma: float = 1.0
    target_horizon: int = 5
    include_binary_entropy: bool = True
    include_state_entropy: bool = True
    feature_columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "volatility_window",
            "entropy_window",
            "state_entropy_window",
        ):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} должен быть >= 2.")

        if self.target_horizon < 1:
            raise ValueError("target_horizon должен быть >= 1.")
        if not 0.0 < self.energy_quantile < 1.0:
            raise ValueError("energy_quantile должен находиться между 0 и 1.")

        for name in ("energy_alpha", "energy_beta", "energy_gamma"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} должен быть конечным числом.")

        if not isinstance(self.include_binary_entropy, bool):
            raise ValueError("include_binary_entropy должен быть bool.")
        if not isinstance(self.include_state_entropy, bool):
            raise ValueError("include_state_entropy должен быть bool.")

        if self.feature_columns is not None:
            if not self.feature_columns:
                raise ValueError("feature_columns не может быть пустым.")
            if any(
                not isinstance(column, str) or not column
                for column in self.feature_columns
            ):
                raise ValueError(
                    "feature_columns должен содержать непустые имена колонок."
                )
            if len(set(self.feature_columns)) != len(self.feature_columns):
                raise ValueError("feature_columns не должен содержать дубликаты.")


class FeatureTargetPipeline:
    """Строит causal-features и train-only нормализованную энергию."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._energy_scales: EnergyScales | None = None
        self._feature_columns: tuple[str, ...] | None = None

    @property
    def energy_scales(self) -> EnergyScales:
        if self._energy_scales is None:
            raise RuntimeError("Pipeline ещё не fitted.")
        return self._energy_scales

    @property
    def feature_columns(self) -> tuple[str, ...]:
        if self._feature_columns is None:
            raise RuntimeError("Pipeline ещё не fitted.")
        return self._feature_columns

    @property
    def required_history_rows(self) -> int:
        entropy_history = self.config.entropy_window + 1
        state_history = self.config.volatility_window + 1

        if self.config.include_state_entropy:
            state_history += self.config.state_entropy_window

        return max(entropy_history, state_history)

    def fit(self, train_ohlcv: pd.DataFrame) -> "FeatureTargetPipeline":
        """Fit только train-derived state: EnergyScales и schema признаков."""

        train = self._canonical_ohlcv(train_ohlcv)
        engineered = self._engineer(train)

        scales = fit_energy_scales(
            engineered,
            quantile=self.config.energy_quantile,
        )
        transformed = self._add_normalized_energy(engineered, scales)
        columns = self._resolve_feature_columns(transformed)

        self._energy_scales = scales
        self._feature_columns = columns
        return self

    def transform_features(
        self,
        ohlcv: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Возвращает признаки текущего блока, используя только прошлую history."""

        frame, history_rows = self._transform_full(ohlcv, history)
        columns = ["timestamp", *self.feature_columns]
        return frame.iloc[history_rows:][columns].reset_index(drop=True)

    def make_training_dataset(
        self,
        ohlcv: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> PreparedDataset:
        """Создаёт X/y и удаляет строки без фактически доступного future target."""

        frame, history_rows = self._transform_full(ohlcv, history)
        labelled = add_future_return(
            frame,
            horizon=self.config.target_horizon,
        ).iloc[history_rows:].reset_index(drop=True)

        valid = labelled["future_return"].notna()
        valid &= labelled.loc[:, self.feature_columns].notna().all(axis=1)
        selected = labelled.loc[valid].reset_index(drop=True)

        return PreparedDataset(
            timestamps=selected["timestamp"].copy(),
            features=selected.loc[:, self.feature_columns].copy(),
            target_return=selected["future_return"].copy(),
            target_direction=selected["future_direction"].copy(),
            feature_columns=self.feature_columns,
        )

    def make_inference_dataset(
        self,
        ohlcv: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> PreparedDataset:
        """Создаёт только признаки: future target при inference не рассчитывается."""

        features = self.transform_features(ohlcv, history)
        return PreparedDataset(
            timestamps=features["timestamp"].copy(),
            features=features.loc[:, self.feature_columns].copy(),
            target_return=None,
            target_direction=None,
            feature_columns=self.feature_columns,
        )

    def _transform_full(
        self,
        ohlcv: pd.DataFrame,
        history: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, int]:
        scales = self.energy_scales
        current = self._canonical_ohlcv(ohlcv)

        if history is None:
            combined = current
            history_rows = 0
        else:
            previous = self._canonical_ohlcv(history)
            if previous["timestamp"].iloc[-1] >= current["timestamp"].iloc[0]:
                raise ValueError("history должна целиком предшествовать ohlcv.")

            combined = pd.concat([previous, current], ignore_index=True)
            report = audit_ohlcv(combined)
            if not report.is_valid:
                raise ValueError("history и ohlcv должны образовывать валидный ряд.")
            history_rows = len(previous)

        engineered = self._engineer(combined)
        return self._add_normalized_energy(engineered, scales), history_rows

    def _canonical_ohlcv(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "open", "high", "low", "close", "volume"}

        if not isinstance(ohlcv, pd.DataFrame):
            raise TypeError("ohlcv должен быть pandas DataFrame.")

        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(
                f"Отсутствуют обязательные столбцы: {sorted(missing)}"
            )
        if ohlcv.empty:
            raise ValueError("ohlcv не может быть пустым.")

        result = ohlcv.copy(deep=True).reset_index(drop=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

        if not result["timestamp"].is_monotonic_increasing:
            raise ValueError("timestamps должны быть отсортированы по возрастанию.")

        report = audit_ohlcv(result)
        if not report.is_valid:
            raise ValueError("OHLCV не прошёл audit_ohlcv.")

        return result

    def _engineer(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        result = build_price_dynamics(
            ohlcv,
            volatility_window=self.config.volatility_window,
        )

        if self.config.include_binary_entropy:
            result = add_entropy(
                result,
                window=self.config.entropy_window,
            )

        if self.config.include_state_entropy:
            result = add_state_entropy(
                result,
                window=self.config.state_entropy_window,
            )

        return add_energy(
            result,
            alpha=self.config.energy_alpha,
            beta=self.config.energy_beta,
            gamma=self.config.energy_gamma,
        )

    def _add_normalized_energy(
        self,
        frame: pd.DataFrame,
        scales: EnergyScales,
    ) -> pd.DataFrame:
        return add_normalized_energy(
            frame,
            scales,
            alpha=self.config.energy_alpha,
            beta=self.config.energy_beta,
            gamma=self.config.energy_gamma,
        )

    def _resolve_feature_columns(
        self,
        frame: pd.DataFrame,
    ) -> tuple[str, ...]:
        if self.config.feature_columns is not None:
            columns = self.config.feature_columns
        else:
            columns = list(self._default_feature_columns())

        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Отсутствуют feature columns: {sorted(missing)}"
            )

        return tuple(columns)

    def _default_feature_columns(self) -> Iterable[str]:
        columns = [
            "log_return",
            "acceleration",
            "volatility",
            "volume",
            "energy",
            "velocity_normalized",
            "acceleration_normalized",
            "volatility_volume_normalized",
            "normalized_energy",
        ]

        if self.config.include_binary_entropy:
            columns.extend(
                ["probability_up", "probability_down", "entropy"]
            )

        if self.config.include_state_entropy:
            columns.append("return_zscore")
            columns.extend(
                f"state_prob_{state}" for state in STATE_NAMES
            )
            columns.append("state_entropy")

        return tuple(columns)
