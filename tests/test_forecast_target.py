import numpy as np
import pandas as pd
import pytest

from src.forecast_target import add_future_return


def create_dataframe():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-08-01 00:00:00",
                periods=6,
                freq="1min",
                tz="UTC",
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
            ],
        }
    )


def test_future_return():
    df = create_dataframe()

    result = add_future_return(
        df,
        horizon=2,
    )

    expected = np.log(102.0 / 100.0)

    assert result["future_return"].iloc[0] == pytest.approx(
        expected
    )


def test_future_direction():
    df = create_dataframe()

    result = add_future_return(
        df,
        horizon=2,
    )

    assert result["future_direction"].iloc[0] == 1


def test_last_rows_are_nan():
    df = create_dataframe()

    result = add_future_return(
        df,
        horizon=2,
    )

    assert pd.isna(result["future_return"].iloc[-1])
    assert pd.isna(result["future_return"].iloc[-2])


def test_invalid_horizon():
    df = create_dataframe()

    with pytest.raises(ValueError):
        add_future_return(df, horizon=0)


def test_invalid_price():
    df = create_dataframe()
    df.loc[2, "close"] = 0.0

    with pytest.raises(ValueError):
        add_future_return(df)


def test_original_dataframe_is_not_modified():
    df = create_dataframe()
    original_columns = list(df.columns)

    add_future_return(df)

    assert list(df.columns) == original_columns