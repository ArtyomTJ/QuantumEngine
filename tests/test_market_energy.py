import numpy as np
import pandas as pd
import pytest

from src.market_energy import add_energy, calculate_energy


def test_calculate_energy():
    energy = calculate_energy(
        velocity=0.1,
        acceleration=0.2,
        volatility=0.3,
        volume=4.0,
        alpha=2.0,
        beta=3.0,
        gamma=5.0,
    )

    expected = (
        2.0 * 0.1**2
        + 3.0 * 0.2**2
        + 5.0 * 0.3 * 4.0
    )

    assert energy == pytest.approx(expected)


def test_negative_volume():
    with pytest.raises(ValueError):
        calculate_energy(
            velocity=0.1,
            acceleration=0.2,
            volatility=0.3,
            volume=-1.0,
        )


def test_negative_volatility():
    with pytest.raises(ValueError):
        calculate_energy(
            velocity=0.1,
            acceleration=0.2,
            volatility=-0.3,
            volume=1.0,
        )


def test_non_finite_values():
    with pytest.raises(ValueError):
        calculate_energy(
            velocity=np.nan,
            acceleration=0.2,
            volatility=0.3,
            volume=1.0,
        )


def test_add_energy():
    df = pd.DataFrame(
        {
            "log_return": [0.1, 0.2],
            "acceleration": [0.01, 0.02],
            "volatility": [0.3, 0.4],
            "volume": [2.0, 3.0],
        }
    )

    result = add_energy(
        df,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    assert "energy" in result.columns
    assert result["energy"].notna().all()

    assert result["energy"].iloc[0] == pytest.approx(
        0.1**2 + 0.01**2 + 0.3 * 2.0
    )


def test_add_energy_does_not_modify_original():
    df = pd.DataFrame(
        {
            "log_return": [0.1],
            "acceleration": [0.01],
            "volatility": [0.3],
            "volume": [2.0],
        }
    )

    original_columns = list(df.columns)

    add_energy(df)

    assert list(df.columns) == original_columns