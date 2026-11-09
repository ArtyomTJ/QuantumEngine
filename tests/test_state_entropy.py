import numpy as np
import pandas as pd
import pytest

from src.state_entropy import (
    add_state_entropy,
    calculate_multistate_entropy,
    classify_return,
)


def test_uniform_distribution_has_max_entropy():
    probabilities = [
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
    ]

    assert calculate_multistate_entropy(
        probabilities
    ) == pytest.approx(1.0)


def test_certain_state_has_zero_entropy():
    probabilities = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    assert calculate_multistate_entropy(
        probabilities
    ) == pytest.approx(0.0)


def test_invalid_probabilities():
    with pytest.raises(ValueError):
        calculate_multistate_entropy(
            [0.5, 0.5, 0.5, 0.5, 0.5]
        )

    with pytest.raises(ValueError):
        calculate_multistate_entropy(
            [-0.1, 0.3, 0.3, 0.3, 0.2]
        )


def test_classify_return():
    assert classify_return(-2.0) == "strong_down"
    assert classify_return(-0.5) == "down"
    assert classify_return(0.0) == "neutral"
    assert classify_return(0.5) == "up"
    assert classify_return(2.0) == "strong_up"


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        classify_return(
            0.5,
            strong_threshold=0.2,
            neutral_threshold=0.5,
        )


def test_add_state_entropy():
    df = pd.DataFrame(
        {
            "log_return": [
                -0.003,
                -0.001,
                0.0,
                0.001,
                0.003,
            ],
            "volatility": [
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
            ],
        }
    )

    result = add_state_entropy(
        df,
        window=5,
    )

    assert "return_zscore" in result.columns
    assert "market_direction_state" in result.columns
    assert "state_entropy" in result.columns

    assert result["state_entropy"].iloc[-1] >= 0.0
    assert result["state_entropy"].iloc[-1] <= 1.0


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame(
        {
            "log_return": [0.01, -0.01],
            "volatility": [0.01, 0.01],
        }
    )

    original_columns = list(df.columns)

    add_state_entropy(
        df,
        window=2,
    )

    assert list(df.columns) == original_columns