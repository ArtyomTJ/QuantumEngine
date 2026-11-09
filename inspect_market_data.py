from pathlib import Path

from src.data_loader import load_ohlcv_csv
from src.data_audit import audit_ohlcv


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


def main() -> None:
    print("Загрузка данных...")
    df = load_ohlcv_csv(DATA_PATH)

    print("Аудит данных...")
    report = audit_ohlcv(df)

    print()
    print("=== Quantum Engine: Data Audit ===")
    print(f"Строк:                 {report.rows:,}")
    print(f"Начало:                {report.start_time}")
    print(f"Конец:                 {report.end_time}")
    print(f"Дубликаты timestamp:   {report.duplicate_timestamps:,}")
    print(f"Пропущенные интервалы: {report.missing_intervals:,}")
    print(f"Пустые значения:       {report.null_values:,}")
    print(f"Некорректные цены:     {report.invalid_prices:,}")
    print(f"Некорректный OHLC:     {report.invalid_ohlc:,}")
    print(f"Некорректный объём:    {report.invalid_volume:,}")
    print(f"Данные валидны:        {report.is_valid}")

    print()
    print("=== Price / Volume ===")
    print(f"Минимум цены:          {df['low'].min():,.2f}")
    print(f"Максимум цены:         {df['high'].max():,.2f}")
    print(f"Средний объём:         {df['volume'].mean():,.3f}")
    print(f"Общий объём:           {df['volume'].sum():,.3f}")


if __name__ == "__main__":
    main()