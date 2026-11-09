import numpy as np
import pandas as pd
import pytest

from src.energy_normalization import (
    EnergyScales,
    add_normalized_energy,
    fit_energy_scales,
)


def create_dataframe():
    return pd.DataFrame(
        {
            "log_return": [
                0.01,
                0.02,
                0.03,
                0.04,
            ],
            "acceleration": [
                0.02,
                0.03,
                0.04,
                0.05,
            ],
            "volatility": [
                0.01,
                0.02,
                0.03,
                0.04,
            ],
            "volume": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    )


def test_fit_energy_scales():
    df = create_dataframe()

    scales = fit_energy_scales(df)

    assert isinstance(scales, EnergyScales)
    assert scales.velocity_p95 > 0
    assert scales.acceleration_p95 > 0
    assert scales.volatility_volume_p95 > 0


def test_invalid_quantile():
    df = create_dataframe()

    with pytest.raises(ValueError):
        fit_energy_scales(df, quantile=1.0)


def test_add_normalized_energy():
    df = create_dataframe()

    scales = fit_energy_scales(df)

    result = add_normalized_energy(
        df,
        scales,
    )

    assert "velocity_normalized" in result.columns
    assert "acceleration_normalized" in result.columns
    assert "volatility_volume_normalized" in result.columns
    assert "normalized_energy" in result.columns

    assert result["normalized_energy"].notna().all()


def test_custom_weights():
    df = create_dataframe()

    scales = fit_energy_scales(df)

    result = add_normalized_energy(
        df,
        scales,
        alpha=2.0,
        beta=3.0,
        gamma=4.0,
    )

    expected = (
        2.0 * result["velocity_normalized"]
        + 3.0 * result["acceleration_normalized"]
        + 4.0 * result["volatility_volume_normalized"]
    )

    pd.testing.assert_series_equal(
        result["normalized_energy"],
        expected,
        check_names=False,
    )


def test_original_dataframe_is_not_modified():
    df = create_dataframe()
    original_columns = list(df.columns)

    scales = fit_energy_scales(df)

    add_normalized_energy(
        df,
        scales,
    )

    assert list(df.columns) == original_columns


def test_invalid_coefficients():
    df = create_dataframe()

    scales = fit_energy_scales(df)

    with pytest.raises(ValueError):
        add_normalized_energy(
            df,
            scales,
            alpha=np.nan,
        )