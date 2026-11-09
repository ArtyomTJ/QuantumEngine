from pathlib import Path

from src.data_loader import load_ohlcv_csv
from src.market_features import build_price_dynamics
from src.energy_normalization import (
    fit_energy_scales,
    add_normalized_energy,
)
from src.state_entropy import add_state_entropy
from src.forecast_target import add_future_return


DATA_PATH = Path("data/raw/BTCUSDT-1m-2026-08.csv")


FEATURES = [
    "volatility",
    "volume",
    "volatility_volume",
    "velocity_squared",
    "acceleration_squared",
    "normalized_energy",
]


def describe_correlation(
    df,
    feature: str,
    target: str = "future_abs_return",
):
    pearson = df[[feature, target]].corr(
        method="pearson"
    ).iloc[0, 1]

    spearman = df[[feature, target]].corr(
        method="spearman"
    ).iloc[0, 1]

    return pearson, spearman


def print_quantile_analysis(
    feature: str,
    train_df,
    validation_df,
):
    print()
    print(f"=== {feature}: validation quantiles ===")

    train_values = train_df[feature].dropna()

    edges = train_values.quantile(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ).to_numpy()

    edges = sorted(set(edges))

    if len(edges) < 3:
        print("Недостаточно уникальных границ.")
        return

    labels = [
        f"Q{i}"
        for i in range(1, len(edges))
    ]

    data = validation_df.copy()

    data["bucket"] = pd.cut(
        data[feature],
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    grouped = (
        data
        .dropna(
            subset=[
                "bucket",
                "future_abs_return",
                "future_return",
            ]
        )
        .groupby(
            "bucket",
            observed=True,
        )
        .agg(
            count=("future_abs_return", "size"),
            mean_abs_return=(
                "future_abs_return",
                "mean",
            ),
            median_abs_return=(
                "future_abs_return",
                "median",
            ),
            mean_return=(
                "future_return",
                "mean",
            ),
            up_rate=(
                "future_return",
                lambda x: (x > 0).mean(),
            ),
        )
    )

    print(grouped.to_string())


def main():
    print("Загрузка данных...")

    df = load_ohlcv_csv(DATA_PATH)

    print("Расчёт динамики...")

    df = build_price_dynamics(
        df,
        volatility_window=60,
    )

    print("Расчёт state entropy...")

    df = add_state_entropy(
        df,
        window=60,
    )

    print("Расчёт будущего движения...")

    df = add_future_return(
        df,
        horizon=5,
    )

    df["future_abs_return"] = (
        df["future_return"].abs()
    )

    # Базовые компоненты энергии.
    df["velocity_squared"] = (
        df["log_return"] ** 2
    )

    df["acceleration_squared"] = (
        df["acceleration"] ** 2
    )

    df["volatility_volume"] = (
        df["volatility"]
        * df["volume"]
    )

    # Убираем строки, где невозможно рассчитать признаки.
    df = df.dropna(
        subset=[
            "log_return",
            "acceleration",
            "volatility",
            "future_return",
        ]
    ).reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    validation_df = df.iloc[split_index:].copy()

    print()
    print("=== Temporal Split ===")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")

    # Масштабы энергии вычисляются только на train.
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
    print("=== Correlations with |future 5m return| ===")
    print()

    print(
        f"{'Feature':<28}"
        f"{'Pearson':>12}"
        f"{'Spearman':>12}"
    )

    print("-" * 52)

    for feature in FEATURES:
        pearson, spearman = describe_correlation(
            validation_df,
            feature,
        )

        print(
            f"{feature:<28}"
            f"{pearson:>12.6f}"
            f"{spearman:>12.6f}"
        )

    print()
    print("=== Correlations with future 5m return ===")
    print()

    print(
        f"{'Feature':<28}"
        f"{'Pearson':>12}"
        f"{'Spearman':>12}"
    )

    print("-" * 52)

    for feature in FEATURES:
        pearson = validation_df[
            [feature, "future_return"]
        ].corr(
            method="pearson"
        ).iloc[0, 1]

        spearman = validation_df[
            [feature, "future_return"]
        ].corr(
            method="spearman"
        ).iloc[0, 1]

        print(
            f"{feature:<28}"
            f"{pearson:>12.6f}"
            f"{spearman:>12.6f}"
        )

    for feature in FEATURES:
        print_quantile_analysis(
            feature,
            train_df,
            validation_df,
        )

    print()
    print("=== Energy vs Volatility ===")

    energy_volatility_corr = validation_df[
        [
            "normalized_energy",
            "volatility",
        ]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        "Spearman correlation: "
        f"{energy_volatility_corr:.6f}"
    )


if __name__ == "__main__":
    import pandas as pd

    main()