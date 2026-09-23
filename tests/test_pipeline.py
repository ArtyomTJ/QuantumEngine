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

    first = FeatureTargetPipeline(config).fit(df, train_only=True).make_training_dataset(df)
    second = FeatureTargetPipeline(config).fit(df, train_only=True).make_training_dataset(df)

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
    ).fit(df, train_only=True)

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
    pipeline = FeatureTargetPipeline(config).fit(train, train_only=True)
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
    pipeline = FeatureTargetPipeline(config).fit(df.iloc[:120], train_only=True)
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
    ).fit(df, train_only=True)

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
    ).fit(df, train_only=True)

    dataset = pipeline.make_inference_dataset(df)

    assert dataset.target_return is None
    assert dataset.target_direction is None
    assert len(dataset.features) < len(df)
    assert np.isfinite(dataset.features.to_numpy(dtype=float)).all()


def test_history_transform_matches_full_series():
    df = create_ohlcv()
    split = 120
    config = PipelineConfig(
        volatility_window=10,
        entropy_window=10,
        state_entropy_window=10,
    )
    pipeline = FeatureTargetPipeline(config).fit(df.iloc[:split], train_only=True)

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
    ).fit(df, train_only=True)

    dataset = pipeline.make_inference_dataset(df)

    assert pipeline.feature_columns == columns
    assert dataset.feature_columns == columns
    assert tuple(dataset.features.columns) == columns


def test_unknown_custom_feature_column_is_rejected():
    with pytest.raises(ValueError, match="feature_columns"):
        FeatureTargetPipeline(
            PipelineConfig(feature_columns=("does_not_exist",))
        ).fit(create_ohlcv(), train_only=True)


@pytest.mark.parametrize(
    "column",
    ("timestamp", "future_return", "future_direction"),
)
def test_service_and_target_columns_are_rejected_from_feature_schema(column):
    with pytest.raises(ValueError, match="служебные/target"):
        PipelineConfig(feature_columns=(column,))


@pytest.mark.parametrize("column", ("open", "high", "low", "close"))
def test_raw_ohlcv_columns_are_rejected_from_feature_schema(column):
    with pytest.raises(ValueError, match="engineered features"):
        FeatureTargetPipeline(
            PipelineConfig(feature_columns=(column,))
        ).fit(create_ohlcv(), train_only=True)


def test_fit_requires_explicit_train_only_confirmation():
    pipeline = FeatureTargetPipeline()

    with pytest.raises(TypeError):
        pipeline.fit(create_ohlcv())
    with pytest.raises(ValueError, match="train_only=True"):
        pipeline.fit(create_ohlcv(), train_only=False)


def test_inference_with_sufficient_history_has_only_finite_features():
    df = create_ohlcv()
    pipeline = FeatureTargetPipeline(
        PipelineConfig(
            volatility_window=10,
            entropy_window=10,
            state_entropy_window=10,
        )
    ).fit(df.iloc[:120], train_only=True)

    dataset = pipeline.make_inference_dataset(
        df.iloc[120:],
        history=df.iloc[120 - pipeline.required_history_rows : 120],
    )

    assert len(dataset.features) == len(df) - 120
    assert np.isfinite(dataset.features.to_numpy(dtype=float)).all()


def test_inference_rejects_insufficient_history():
    df = create_ohlcv()
    pipeline = FeatureTargetPipeline(
        PipelineConfig(
            volatility_window=10,
            entropy_window=10,
            state_entropy_window=10,
        )
    ).fit(df.iloc[:120], train_only=True)

    with pytest.raises(ValueError, match="Недостаточно history"):
        pipeline.make_inference_dataset(
            df.iloc[120:],
            history=df.iloc[120 - pipeline.required_history_rows + 1 : 120],
        )


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_non_finite_ohlcv_values_are_rejected(value):
    df = create_ohlcv()
    df.loc[10, "close"] = value

    with pytest.raises(ValueError, match="конечные значения"):
        FeatureTargetPipeline().fit(df, train_only=True)
