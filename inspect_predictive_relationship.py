from pathlib import Path

import pandas as pd

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics
from src.energy_normalization import (
    fit_energy_scales,
    add_normalized_energy,
)
from src.state_entropy import add_state_entropy
from src.forecast_target import add_future_return


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


def print_group_analysis(
    name: str,
    train_series: pd.Series,
    validation_df: pd.DataFrame,
    target_column: str = "future_return",
):
    print()
    print(f"=== {name} ===")

    # Границы квантилей вычисляем ТОЛЬКО на train.
    quantile_edges = train_series.quantile(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ).to_numpy()

    # Убираем возможные одинаковые границы.
    quantile_edges = pd.unique(quantile_edges)

    if len(quantile_edges) < 3:
        print("Недостаточно уникальных значений для разбиения.")
        return

    labels = [
        f"Q{i}"
        for i in range(1, len(quantile_edges))
    ]

    grouped = validation_df.copy()

    grouped["bucket"] = pd.cut(
        grouped[name],
        bins=quantile_edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    result = (
        grouped
        .dropna(subset=["bucket", target_column])
        .groupby("bucket", observed=True)
        .agg(
            count=(target_column, "size"),
            mean_return=(target_column, "mean"),
            median_return=(target_column, "median"),
            mean_abs_return=(
                target_column,
                lambda x: x.abs().mean(),
            ),
            up_rate=(
                target_column,
                lambda x: (x > 0).mean(),
            ),
        )
    )

    print(result.to_string())


def main():
    print("Загрузка данных...")

    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт динамики...")

    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    print("Расчёт многосостоянийной энтропии...")

    df = add_state_entropy(
        df,
        window=60,
    )

    print("Расчёт будущей доходности...")

    df = add_future_return(
        df,
        horizon=5,
    )

    # Для анализа нужны все признаки состояния.
    df = df.dropna(
        subset=[
            "log_return",
            "acceleration",
            "volatility",
            "state_entropy",
            "future_return",
        ]
    ).reset_index(drop=True)

    # Временное разделение.
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    validation_df = df.iloc[split_index:].copy()

    print()
    print("=== Temporal Split ===")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")

    # Энергетические масштабы только train.
    scales = fit_energy_scales(
        train_df,
        quantile=0.95,
    )

    df = add_normalized_energy(
        df,
        scales,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    train_df = df.iloc[:split_index].copy()
    validation_df = df.iloc[split_index:].copy()

    print()
    print("=== Validation Correlations ===")

    pearson_energy = validation_df[
        [
            "normalized_energy",
            "future_return",
        ]
    ].corr().iloc[0, 1]

    spearman_energy = validation_df[
        [
            "normalized_energy",
            "future_return",
        ]
    ].corr(method="spearman").iloc[0, 1]

    pearson_entropy = validation_df[
        [
            "state_entropy",
            "future_return",
        ]
    ].corr().iloc[0, 1]

    spearman_entropy = validation_df[
        [
            "state_entropy",
            "future_return",
        ]
    ].corr(method="spearman").iloc[0, 1]

    print(
        f"Energy / future return — Pearson:  "
        f"{pearson_energy:.6f}"
    )

    print(
        f"Energy / future return — Spearman: "
        f"{spearman_energy:.6f}"
    )

    print(
        f"Entropy / future return — Pearson: "
        f"{pearson_entropy:.6f}"
    )

    print(
        f"Entropy / future return — Spearman:"
        f" {spearman_entropy:.6f}"
    )

    # Корреляция с абсолютным будущим движением.
    validation_df["future_abs_return"] = (
        validation_df["future_return"].abs()
    )

    energy_abs_corr = validation_df[
        [
            "normalized_energy",
            "future_abs_return",
        ]
    ].corr().iloc[0, 1]

    entropy_abs_corr = validation_df[
        [
            "state_entropy",
            "future_abs_return",
        ]
    ].corr().iloc[0, 1]

    print()
    print("=== Relationship With Future Move Size ===")

    print(
        f"Energy / |future return|:  "
        f"{energy_abs_corr:.6f}"
    )

    print(
        f"Entropy / |future return|: "
        f"{entropy_abs_corr:.6f}"
    )

    # Анализ по квантилям энергии.
    train_energy = train_df["normalized_energy"]

    print_group_analysis(
        "normalized_energy",
        train_energy,
        validation_df,
    )

    # Анализ по квантилям энтропии.
    train_entropy = train_df["state_entropy"]

    print_group_analysis(
        "state_entropy",
        train_entropy,
        validation_df,
    )


if __name__ == "__main__":
    main()