import pandas as pd
import pytest

from src.market_state import MarketState, build_market_state


def create_valid_row():
    return pd.Series(
        {
            "timestamp": pd.Timestamp(
                "2026-08-31 23:59:00+00:00"
            ),
            "close": 78549.6,
            "log_return": 0.000276,
            "acceleration": 0.000256,
            "volatility": 0.000291,
            "volume": 85.5,
        }
    )


def test_build_market_state():
    row = create_valid_row()

    state = build_market_state(row)

    assert isinstance(state, MarketState)

    assert state.timestamp == pd.Timestamp(
        "2026-08-31 23:59:00+00:00"
    )

    assert state.price == 78549.6
    assert state.velocity == 0.000276
    assert state.acceleration == 0.000256
    assert state.volatility == 0.000291
    assert state.volume == 85.5


def test_missing_field():
    row = create_valid_row()

    row = row.drop("volume")

    with pytest.raises(ValueError):
        build_market_state(row)


def test_nan_field():
    row = create_valid_row()

    row["volatility"] = float("nan")

    with pytest.raises(ValueError):
        build_market_state(row)


def test_numeric_conversion():
    row = create_valid_row()

    row["close"] = "78549.6"
    row["log_return"] = "0.000276"
    row["acceleration"] = "0.000256"
    row["volatility"] = "0.000291"
    row["volume"] = "85.5"

    state = build_market_state(row)

    assert isinstance(state.price, float)
    assert isinstance(state.velocity, float)
    assert isinstance(state.acceleration, float)
    assert isinstance(state.volatility, float)
    assert isinstance(state.volume, float)