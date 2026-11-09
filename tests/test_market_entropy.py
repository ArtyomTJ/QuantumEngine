import numpy as np
import pandas as pd
import pytest

from src.market_entropy import (
    add_entropy,
    calculate_binary_entropy,
)


def test_zero_entropy_for_certain_direction():
    assert calculate_binary_entropy(1.0) == pytest.approx(0.0)
    assert calculate_binary_entropy(0.0) == pytest.approx(0.0)


def test_maximum_entropy():
    assert calculate_binary_entropy(0.5) == pytest.approx(1.0)


def test_entropy_is_bounded():
    for probability in np.linspace(0.0, 1.0, 11):
        entropy = calculate_binary_entropy(probability)

        assert 0.0 <= entropy <= 1.0


def test_invalid_probability():
    with pytest.raises(ValueError):
        calculate_binary_entropy(-0.1)

    with pytest.raises(ValueError):
        calculate_binary_entropy(1.1)


def test_add_entropy():
    df = pd.DataFrame(
        {
            "log_return": [
                0.01,
                -0.01,
                0.02,
                -0.02,
            ]
        }
    )

    result = add_entropy(
        df,
        window=4,
    )

    assert result["probability_up"].iloc[3] == pytest.approx(0.5)
    assert result["probability_down"].iloc[3] == pytest.approx(0.5)
    assert result["entropy"].iloc[3] == pytest.approx(1.0)


def test_one_direction():
    df = pd.DataFrame(
        {
            "log_return": [
                0.01,
                0.02,
                0.03,
                0.04,
            ]
        }
    )

    result = add_entropy(
        df,
        window=4,
    )

    assert result["probability_up"].iloc[3] == pytest.approx(1.0)
    assert result["probability_down"].iloc[3] == pytest.approx(0.0)
    assert result["entropy"].iloc[3] == pytest.approx(0.0)


def test_invalid_window():
    df = pd.DataFrame(
        {
            "log_return": [0.01, 0.02]
        }
    )

    with pytest.raises(ValueError):
        add_entropy(df, window=1)


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame(
        {
            "log_return": [0.01, -0.01]
        }
    )

    original_columns = list(df.columns)

    add_entropy(df, window=2)

    assert list(df.columns) == original_columns