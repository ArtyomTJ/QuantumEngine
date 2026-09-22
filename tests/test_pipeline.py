import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.pipeline import FeatureTargetPipeline, PipelineConfig


def create_ohlcv(rows: int = 180) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="1min",
        tz="UTC",
    )
    steps = np.arange(rows, dtype=float)
    close = 100.0 + steps * 0.03 + np.sin(steps / 3.0)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.05,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": 10.0 + (steps % 11.0),
        }
    )


def test_pipeline_is_deterministic():
    df = create_ohlcv()
    config = PipelineConfig(
        volatility_window=10,
        entropy_window=10,
        state_entropy_window=10,
    )

    first = FeatureTargetPipeline(config).fit(df).make_training_dataset(df)
    second = FeatureTargetPipeline(config).fit(df).make_training_dataset(df)

    pdt.assert_frame_equal(first.features, second.features)
    pdt.assert_series_equal(first.target_return, second.target_return)
    pdt.assert_series_equal(first.target_direction, second.target_direction)
    assert first.feature_columns == second.feature_columns


def test_pipeline_does_not_mutate_input():
    df = create_ohlcv()
    original = df.copy(deep=True)
    pipeline = FeatureTargetPipeline(
        PipelineConfig(
            volatility_window=10,
            entropy_window=10,
            state_entropy_window=10,
        )
    ).fit(df)

    pipeline.make_training_dataset(df)

    pdt.assert_frame_equal(df, original)


def test_energy_scales_are_fit_only_on_train():
    train = create_ohlcv(120)
    validation = create_ohlcv(60)
    validation["timestamp"] = pd.date_range(
        train["timestamp"].iloc[-1] + pd.Timedelta(minutes=1),
        periods=len(validation),
        freq="1min",
        tz="UTC",
    )
    altered_validation = validation.copy()
    altered_validation.loc[:, "volume"] *= 1_000_000

    config = PipelineConfig(
        volatility_window=10,
        entropy_window=10,
        state_entropy_window=10,
    )
    pipeline = FeatureTargetPipeline(config).fit(train)
    scales = pipeline.energy_scales

    pipeline.transform_features(validation, history=train.tail(20))
    pipeline.transform_features(altered_validation, history=train.tail(20))

    assert pipeline.energy_scales == scales


def test_changing_future_rows_does_not_change_past_features():
    df = create_ohlcv()
    config = PipelineConfig(
        volatility_window=10,
        entropy_window=10,
        state_entropy_window=10,
    )
    pipeline = FeatureTargetPipeline(config).fit(df.iloc[:120])
    baseline = pipeline.transform_features(df)

    changed = df.copy()
    changed.loc[121:, ["open", "high", "low", "close"]] *= 1.5
    changed.loc[121:, "volume"] *= 5.0
    transformed = pipeline.transform_features(changed)

    pdt.assert_frame_equal(
        baseline.iloc[:121].reset_index(drop=True),
        transformed.iloc[:121].reset_index(drop=True),
    )


def test_training_dataset_excludes_rows_without_future_target():
    df = create_ohlcv()
    horizon = 5
    pipeline = FeatureTargetPipeline(
        PipelineConfig(
            volatility_window=10,
            entropy_window=10,
            state_entropy_window=10,
            target_horizon=horizon,
        )
    ).fit(df)

    dataset = pipeline.make_training_dataset(df)

    assert len(dataset.features) < len(df) - horizon
    assert dataset.target_return.notna().all()
    assert not (dataset.target_direction == 0).any()


def test_inference_dataset_has_no_target():
    df = create_ohlcv()
    pipeline = FeatureTargetPipeline(
        PipelineConfig(
            volatility_window=10,
            entropy_window=10,
            state_entropy_window=10,
        )
    ).fit(df)

    dataset = pipeline.make_inference_dataset(df)

    assert dataset.target_return is None
    assert dataset.target_direction is None
    assert len(dataset.features) == len(df)


def test_history_transform_matches_full_series():
    df = create_ohlcv()
    split = 120
    config = PipelineConfig(
        volatility_window=10,
        entropy_window=10,
        state_entropy_window=10,
    )
    pipeline = FeatureTargetPipeline(config).fit(df.iloc[:split])

    full = pipeline.transform_features(df)
    validation = pipeline.transform_features(
        df.iloc[split:],
        history=df.iloc[:split].tail(pipeline.required_history_rows),
    )

    expected = full.iloc[split:].reset_index(drop=True)
    pdt.assert_frame_equal(validation, expected)


def test_custom_feature_schema_is_fixed_and_ordered():
    df = create_ohlcv()
    columns = ("normalized_energy", "entropy", "volatility")
    pipeline = FeatureTargetPipeline(
        PipelineConfig(
            volatility_window=10,
            entropy_window=10,
            state_entropy_window=10,
            feature_columns=columns,
        )
    ).fit(df)

    dataset = pipeline.make_inference_dataset(df)

    assert pipeline.feature_columns == columns
    assert dataset.feature_columns == columns
    assert tuple(dataset.features.columns) == columns


def test_unknown_custom_feature_column_is_rejected():
    with pytest.raises(ValueError, match="feature columns"):
        FeatureTargetPipeline(
            PipelineConfig(feature_columns=("does_not_exist",))
        ).fit(create_ohlcv())
