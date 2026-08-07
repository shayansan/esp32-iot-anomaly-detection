#!/usr/bin/env python3

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

# This file is located at:
# project/ml/anomaly_detection/analysis/analyze_datasets.py
#
# parents[0] -> analysis
# parents[1] -> anomaly_detection
# parents[2] -> ml
# parents[3] -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data/anomaly_detection/raw"
)

ANALYSIS_RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts/anomaly_detection/host_results/dataset_analysis"
)

ANALYSIS_PLOTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts/anomaly_detection/plots/dataset_analysis"
)

DATASET_PATHS = {
    "train": RAW_DATA_DIRECTORY / "train_seed42.csv",
    "validation": (
        RAW_DATA_DIRECTORY / "validation_seed123.csv"
    ),
    "test": RAW_DATA_DIRECTORY / "test_seed999.csv",
}


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

EXPECTED_COLUMNS = [
    "timestamp_ms",
    "temperature_c",
    "humidity_percent",
    "light_raw",
    "anomaly",
]

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_percent",
    "light_raw",
]

FEATURE_LABELS = {
    "temperature_c": "Temperature (°C)",
    "humidity_percent": "Humidity (%)",
    "light_raw": "Light level",
}

# We plot only the first 400 rows in time-series plots.
#
# This range contains:
# - temperature anomaly at samples 100–104
# - humidity anomaly at samples 200–204
# - light anomaly at samples 300–304
TIME_SERIES_SAMPLE_LIMIT = 400


def load_dataset(
    name: str,
    path: Path,
) -> pd.DataFrame:
    """
    Load one dataset and add useful analysis columns.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset was not found: {path}"
        )

    data = pd.read_csv(path)

    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"{name} has incorrect columns.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {data.columns.tolist()}"
        )

    # Copy the DataFrame before adding analysis-only columns.
    data = data.copy()

    # A row number that begins from zero.
    data["sample_index"] = np.arange(len(data))

    # Convert the timestamps into elapsed seconds.
    #
    # The first timestamp is subtracted so the graph starts
    # approximately from zero seconds.
    data["time_seconds"] = (
        data["timestamp_ms"]
        - data["timestamp_ms"].iloc[0]
    ) / 1000.0

    data["dataset"] = name

    data["anomaly_type"] = data.apply(
        identify_anomaly_type,
        axis=1,
    )

    return data


def identify_anomaly_type(row: pd.Series) -> str:
    """
    Identify the known synthetic anomaly type.

    These rules are used only for analysis and visualization.
    They will not be given to the neural network.
    """

    if row["anomaly"] == 0:
        return "normal"

    if row["temperature_c"] > 30.0:
        return "temperature_spike"

    if row["humidity_percent"] < 30.0:
        return "humidity_drop"

    if np.isclose(
        row["light_raw"],
        1023.0,
        atol=0.01,
    ):
        return "light_saturation"

    return "unknown_anomaly"


def create_dataset_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Create one summary row for each dataset.
    """

    summary_records = []

    for name, data in datasets.items():
        normal_rows = int(
            (data["anomaly"] == 0).sum()
        )

        anomaly_rows = int(
            (data["anomaly"] == 1).sum()
        )

        anomaly_percentage = (
            anomaly_rows / len(data) * 100.0
        )

        median_interval_ms = float(
            data["timestamp_ms"]
            .diff()
            .dropna()
            .median()
        )

        summary_records.append({
            "dataset": name,
            "total_rows": len(data),
            "normal_rows": normal_rows,
            "anomaly_rows": anomaly_rows,
            "anomaly_percentage": anomaly_percentage,
            "median_interval_ms": median_interval_ms,
        })

    return pd.DataFrame(summary_records)


def create_anomaly_type_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Count each anomaly type in every dataset.
    """

    combined = pd.concat(
        datasets.values(),
        ignore_index=True,
    )

    summary = (
        combined
        .groupby(
            ["dataset", "anomaly_type"],
            observed=True,
        )
        .size()
        .reset_index(name="row_count")
    )

    return summary


def create_feature_statistics(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate feature statistics separately for normal
    and anomalous samples.
    """

    records = []

    for dataset_name, data in datasets.items():
        for label_value, label_name in (
            (0, "normal"),
            (1, "anomaly"),
        ):
            selected_rows = data[
                data["anomaly"] == label_value
            ]

            for feature in FEATURE_COLUMNS:
                records.append({
                    "dataset": dataset_name,
                    "class": label_name,
                    "feature": feature,
                    "count": len(selected_rows),
                    "mean": selected_rows[feature].mean(),
                    "standard_deviation": (
                        selected_rows[feature].std()
                    ),
                    "minimum": selected_rows[feature].min(),
                    "maximum": selected_rows[feature].max(),
                })

    return pd.DataFrame(records)


def plot_class_distribution(
    summary: pd.DataFrame,
) -> None:
    """
    Create a normal/anomaly count plot for all datasets.
    """

    figure, axis = plt.subplots(figsize=(9, 5))

    dataset_positions = np.arange(len(summary))
    bar_width = 0.35

    axis.bar(
        dataset_positions - bar_width / 2,
        summary["normal_rows"],
        width=bar_width,
        label="Normal",
    )

    axis.bar(
        dataset_positions + bar_width / 2,
        summary["anomaly_rows"],
        width=bar_width,
        label="Anomaly",
    )

    axis.set_title("Class Distribution by Dataset")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Number of rows")

    axis.set_xticks(dataset_positions)
    axis.set_xticklabels(summary["dataset"])

    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    output_path = (
        ANALYSIS_PLOTS_DIRECTORY / "class_distribution.png"
    )

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_feature_time_series(
    dataset_name: str,
    data: pd.DataFrame,
) -> None:
    """
    Plot the beginning of each feature and mark anomalies.
    """

    sample_limit = min(
        TIME_SERIES_SAMPLE_LIMIT,
        len(data),
    )

    selected_data = data.iloc[:sample_limit]
    anomaly_rows = selected_data[
        selected_data["anomaly"] == 1
    ]

    for feature in FEATURE_COLUMNS:
        figure, axis = plt.subplots(figsize=(12, 5))

        axis.plot(
            selected_data["sample_index"],
            selected_data[feature],
            linewidth=1.2,
            label="Sensor value",
        )

        axis.scatter(
            anomaly_rows["sample_index"],
            anomaly_rows[feature],
            marker="x",
            s=55,
            label="Labeled anomaly",
        )

        axis.set_title(
            f"{dataset_name.capitalize()} Dataset — "
            f"{FEATURE_LABELS[feature]}"
        )

        axis.set_xlabel("Sample index")
        axis.set_ylabel(FEATURE_LABELS[feature])

        axis.legend()
        axis.grid(alpha=0.3)

        figure.tight_layout()

        output_path = ANALYSIS_PLOTS_DIRECTORY / (
            f"{dataset_name}_{feature}_time_series.png"
        )

        figure.savefig(output_path, dpi=150)
        plt.close(figure)


def plot_feature_histograms(
    training_data: pd.DataFrame,
) -> None:
    """
    Compare normal and anomalous feature distributions.
    """

    normal_rows = training_data[
        training_data["anomaly"] == 0
    ]

    anomaly_rows = training_data[
        training_data["anomaly"] == 1
    ]

    for feature in FEATURE_COLUMNS:
        figure, axis = plt.subplots(figsize=(9, 5))

        axis.hist(
            normal_rows[feature],
            bins=50,
            alpha=0.65,
            label="Normal",
        )

        axis.hist(
            anomaly_rows[feature],
            bins=50,
            alpha=0.65,
            label="Anomaly",
        )

        axis.set_title(
            f"Training Distribution — "
            f"{FEATURE_LABELS[feature]}"
        )

        axis.set_xlabel(FEATURE_LABELS[feature])
        axis.set_ylabel("Frequency")

        axis.legend()
        axis.grid(axis="y", alpha=0.3)

        figure.tight_layout()

        output_path = ANALYSIS_PLOTS_DIRECTORY / (
            f"train_{feature}_histogram.png"
        )

        figure.savefig(output_path, dpi=150)
        plt.close(figure)


def plot_feature_relationships(
    training_data: pd.DataFrame,
) -> None:
    """
    Create pairwise feature scatter plots.

    These plots show whether anomalies form visibly
    different regions from normal samples.
    """

    normal_rows = training_data[
        training_data["anomaly"] == 0
    ]

    anomaly_rows = training_data[
        training_data["anomaly"] == 1
    ]

    for feature_x, feature_y in combinations(
        FEATURE_COLUMNS,
        2,
    ):
        figure, axis = plt.subplots(figsize=(8, 6))

        axis.scatter(
            normal_rows[feature_x],
            normal_rows[feature_y],
            s=10,
            alpha=0.35,
            label="Normal",
        )

        axis.scatter(
            anomaly_rows[feature_x],
            anomaly_rows[feature_y],
            s=28,
            alpha=0.8,
            marker="x",
            label="Anomaly",
        )

        axis.set_title(
            "Training Feature Relationship"
        )

        axis.set_xlabel(FEATURE_LABELS[feature_x])
        axis.set_ylabel(FEATURE_LABELS[feature_y])

        axis.legend()
        axis.grid(alpha=0.3)

        figure.tight_layout()

        output_path = ANALYSIS_PLOTS_DIRECTORY / (
            f"train_scatter_{feature_x}_vs_"
            f"{feature_y}.png"
        )

        figure.savefig(output_path, dpi=150)
        plt.close(figure)


def check_distribution_similarity(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compare normal feature means and standard deviations.

    Training, validation, and test datasets should have
    similar normal distributions because they simulate
    the same system.
    """

    records = []

    for dataset_name, data in datasets.items():
        normal_rows = data[data["anomaly"] == 0]

        for feature in FEATURE_COLUMNS:
            records.append({
                "dataset": dataset_name,
                "feature": feature,
                "normal_mean": normal_rows[feature].mean(),
                "normal_standard_deviation": (
                    normal_rows[feature].std()
                ),
            })

    return pd.DataFrame(records)


def main() -> None:
    for directory in (
        ANALYSIS_RESULTS_DIRECTORY,
        ANALYSIS_PLOTS_DIRECTORY,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    datasets = {}

    for name, path in DATASET_PATHS.items():
        datasets[name] = load_dataset(
            name=name,
            path=path,
        )

    dataset_summary = create_dataset_summary(
        datasets
    )

    anomaly_type_summary = (
        create_anomaly_type_summary(datasets)
    )

    feature_statistics = create_feature_statistics(
        datasets
    )

    distribution_comparison = (
        check_distribution_similarity(datasets)
    )

    # Save analysis results as CSV files.
    dataset_summary.to_csv(
        ANALYSIS_RESULTS_DIRECTORY / "dataset_summary.csv",
        index=False,
    )

    anomaly_type_summary.to_csv(
        ANALYSIS_RESULTS_DIRECTORY / "anomaly_type_summary.csv",
        index=False,
    )

    feature_statistics.to_csv(
        ANALYSIS_RESULTS_DIRECTORY / "feature_statistics.csv",
        index=False,
    )

    distribution_comparison.to_csv(
        ANALYSIS_RESULTS_DIRECTORY
        / "normal_distribution_comparison.csv",
        index=False,
    )

    # Create plots.
    plot_class_distribution(dataset_summary)

    for name, data in datasets.items():
        plot_feature_time_series(
            dataset_name=name,
            data=data,
        )

    plot_feature_histograms(datasets["train"])
    plot_feature_relationships(datasets["train"])

    # Print the most important results.
    print("\nDataset summary")
    print("=" * 80)

    print(
        dataset_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nAnomaly type summary")
    print("=" * 80)

    print(
        anomaly_type_summary.to_string(index=False)
    )

    print("\nNormal distribution comparison")
    print("=" * 80)

    print(
        distribution_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    unknown_anomalies = anomaly_type_summary[
        anomaly_type_summary["anomaly_type"]
        == "unknown_anomaly"
    ]

    if unknown_anomalies.empty:
        print(
            "\n[PASS] Every anomalous row matches "
            "a known synthetic anomaly type."
        )
    else:
        print(
            "\n[WARNING] Some anomalous rows could not "
            "be classified."
        )

        print(unknown_anomalies.to_string(index=False))

    print("\nAnalysis outputs saved to:")
    print(
        f"Results: {ANALYSIS_RESULTS_DIRECTORY}"
    )
    print(
        f"Plots: {ANALYSIS_PLOTS_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
