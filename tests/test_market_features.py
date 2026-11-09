import numpy as np
import pandas as pd
import pytest

from src.market_features import build_price_dynamics


def create_price_dataframe():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-08-01 00:00:00",
                periods=5,
                freq="1min",
                tz="UTC",
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
            ],
        }
    )


def test_log_return():
    df = create_price_dataframe()

    result = build_price_dynamics(
        df,
        volatility_window=3,
    )

    expected = np.log(101.0 / 100.0)

    assert np.isnan(result["log_return"].iloc[0])
    assert result["log_return"].iloc[1] == pytest.approx(expected)


def test_acceleration():
    df = create_price_dataframe()

    result = build_price_dynamics(
        df,
        volatility_window=3,
    )

    expected_first = np.log(101.0 / 100.0)
    expected_second = np.log(102.0 / 101.0)

    expected_acceleration = expected_second - expected_first

    assert np.isnan(result["acceleration"].iloc[0])
    assert np.isnan(result["acceleration"].iloc[1])
    assert result["acceleration"].iloc[2] == pytest.approx(
        expected_acceleration
    )


def test_volatility():
    df = create_price_dataframe()

    result = build_price_dynamics(
        df,
        volatility_window=3,
    )

    returns = np.log(
        np.array([101.0, 102.0, 103.0])
        / np.array([100.0, 101.0, 102.0])
    )

    expected_volatility = np.std(returns, ddof=1)

    assert result["volatility"].iloc[3] == pytest.approx(
        expected_volatility
    )


def test_original_dataframe_is_not_modified():
    df = create_price_dataframe()
    original_columns = list(df.columns)

    build_price_dynamics(
        df,
        volatility_window=3,
    )

    assert list(df.columns) == original_columns


def test_invalid_price_window():
    df = create_price_dataframe()

    with pytest.raises(ValueError):
        build_price_dynamics(
            df,
            volatility_window=1,
        )


def test_negative_close():
    df = create_price_dataframe()
    df.loc[2, "close"] = -10

    with pytest.raises(ValueError):
        build_price_dynamics(df)