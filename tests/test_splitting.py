import pandas as pd
import pytest

from src.splitting import WalkForwardConfig, WalkForwardSplitter


def timestamps(rows: int = 80) -> pd.Series:
    return pd.Series(
        pd.date_range(
            "2026-01-01",
            periods=rows,
            freq="1min",
            tz="UTC",
        )
    )


def test_splitter_preserves_temporal_order_and_no_overlap():
    splitter = WalkForwardSplitter(
        WalkForwardConfig(
            train_size=20,
            validation_size=10,
            test_size=10,
            step_size=10,
            embargo=5,
        ),
        target_horizon=5,
    )

    split = next(splitter.split(timestamps()))

    assert split.train_end <= split.validation_start - 5
    assert split.validation_end <= split.test_start - 5
    assert set(split.train_indices).isdisjoint(split.validation_indices)
    assert set(split.validation_indices).isdisjoint(split.test_indices)
    assert max(split.train_indices) < min(split.validation_indices)
    assert max(split.validation_indices) < min(split.test_indices)


def test_splitter_requires_sufficient_embargo_for_target_integrity():
    with pytest.raises(ValueError, match="embargo"):
        WalkForwardSplitter(
            WalkForwardConfig(
                train_size=20,
                validation_size=10,
                test_size=None,
                step_size=10,
                embargo=4,
            ),
            target_horizon=5,
        )


def test_splitter_is_deterministic():
    config = WalkForwardConfig(
        train_size=20,
        validation_size=10,
        test_size=None,
        step_size=10,
        embargo=5,
    )

    first = list(WalkForwardSplitter(config).split(timestamps()))
    second = list(WalkForwardSplitter(config).split(timestamps()))

    assert first == second


def test_train_targets_cannot_reach_validation_start():
    horizon = 5
    splitter = WalkForwardSplitter(
        WalkForwardConfig(
            train_size=20,
            validation_size=10,
            test_size=None,
            step_size=10,
            embargo=horizon,
        ),
        target_horizon=horizon,
    )

    split = next(splitter.split(timestamps()))
    last_train_index = split.train_indices[-1]

    assert last_train_index + horizon < split.validation_start


def test_splitter_rejects_non_monotonic_timestamps():
    values = timestamps()
    values.iloc[10], values.iloc[11] = values.iloc[11], values.iloc[10]
    splitter = WalkForwardSplitter(
        WalkForwardConfig(
            train_size=20,
            validation_size=10,
            test_size=None,
            step_size=10,
            embargo=5,
        )
    )

    with pytest.raises(ValueError, match="отсортированы"):
        list(splitter.split(values))
