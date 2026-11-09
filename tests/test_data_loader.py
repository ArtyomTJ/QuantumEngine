import pandas as pd
import pytest

from src.data_loader import load_ohlcv_csv


def create_sample_csv(path):
    data = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
                "2026-01-01 00:02:00",
            ],
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [10, 12, 15],
        }
    )

    data.to_csv(path, index=False)


def test_load_ohlcv_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"

    create_sample_csv(csv_path)

    df = load_ohlcv_csv(csv_path)

    assert len(df) == 3
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["close"].iloc[-1] == 103


def test_missing_column(tmp_path):
    csv_path = tmp_path / "invalid.csv"

    data = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00"],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
        }
    )

    data.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_ohlcv_csv(csv_path)


def test_negative_volume(tmp_path):
    csv_path = tmp_path / "invalid_volume.csv"

    data = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00"],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
            "volume": [-5],
        }
    )

    data.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_ohlcv_csv(csv_path)
def test_load_binance_csv(tmp_path):
    csv_path = tmp_path / "binance.csv"

    data = pd.DataFrame(
        {
            "open_time": [
                1785542400000,
                1785542460000,
            ],
            "open": [62859.90, 62863.80],
            "high": [62872.40, 62902.40],
            "low": [62859.80, 62902.40],
            "close": [62863.80, 62902.40],
            "volume": [70.188, 108.845],
            "close_time": [
                1785542459999,
                1785542519999,
            ],
            "quote_volume": [
                4412340.02750,
                6844078.90420,
            ],
            "count": [
                1293,
                2122,
            ],
            "taker_buy_volume": [
                29.359,
                67.267,
            ],
            "taker_buy_quote_volume": [
                1845583.23570,
                4229677.08940,
            ],
            "ignore": [0, 0],
        }
    )

    data.to_csv(csv_path, index=False)

    df = load_ohlcv_csv(csv_path)

    assert len(df) == 2

    assert df["timestamp"].iloc[0] == pd.Timestamp(
        "2026-08-01 00:00:00+00:00"
    )

    assert df["open"].iloc[0] == 62859.90
    assert df["close"].iloc[1] == 62902.40
    assert df["volume"].iloc[0] == 70.188

    assert "quote_volume" in df.columns
    assert "count" in df.columns
    assert "taker_buy_volume" in df.columns