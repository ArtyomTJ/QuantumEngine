import pandas as pd

from src.data_audit import audit_ohlcv


def create_valid_dataframe():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-08-01 00:00:00",
                    "2026-08-01 00:01:00",
                    "2026-08-01 00:02:00",
                ],
                utc=True,
            ),
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [10, 12, 15],
        }
    )


def test_valid_dataframe():
    df = create_valid_dataframe()

    report = audit_ohlcv(df)

    assert report.rows == 3
    assert report.duplicate_timestamps == 0
    assert report.missing_intervals == 0
    assert report.null_values == 0
    assert report.invalid_prices == 0
    assert report.invalid_ohlc == 0
    assert report.invalid_volume == 0
    assert report.is_valid is True


def test_missing_interval():
    df = create_valid_dataframe()

    df = df.drop(index=1).reset_index(drop=True)

    report = audit_ohlcv(df)

    assert report.missing_intervals == 1
    assert report.is_valid is False


def test_duplicate_timestamp():
    df = create_valid_dataframe()

    duplicate_row = df.iloc[[1]]
    df = pd.concat([df, duplicate_row], ignore_index=True)

    report = audit_ohlcv(df)

    assert report.duplicate_timestamps == 1
    assert report.is_valid is False


def test_invalid_ohlc():
    df = create_valid_dataframe()

    df.loc[1, "high"] = 90

    report = audit_ohlcv(df)

    assert report.invalid_ohlc == 1
    assert report.is_valid is False


def test_negative_volume():
    df = create_valid_dataframe()

    df.loc[1, "volume"] = -10

    report = audit_ohlcv(df)

    assert report.invalid_volume == 1
    assert report.is_valid is False