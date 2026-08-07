#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data/anomaly_detection/raw"
)

PROCESSED_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data/anomaly_detection/processed"
)

TRAIN_FILE = RAW_DATA_DIRECTORY / "train_seed42.csv"

VALIDATION_FILE = (
    RAW_DATA_DIRECTORY / "validation_seed123.csv"
)

TEST_FILE = RAW_DATA_DIRECTORY / "test_seed999.csv"


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_percent",
    "light_raw",
]

TARGET_COLUMN = "anomaly"

EXCLUDED_COLUMNS = [
    "timestamp_ms",
]

EXPECTED_COLUMNS = [
    "timestamp_ms",
    "temperature_c",
    "humidity_percent",
    "light_raw",
    "anomaly",
]


def load_dataset(
    name: str,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one CSV dataset and separate it into:

    features:
        temperature, humidity, light

    labels:
        anomaly
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset does not exist: {path}"
        )

    data = pd.read_csv(path)

    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"{name} dataset has incorrect columns.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {data.columns.tolist()}"
        )

    if data.isna().any().any():
        raise ValueError(
            f"{name} dataset contains missing values"
        )

    labels_are_valid = data[
        TARGET_COLUMN
    ].isin([0, 1]).all()

    if not labels_are_valid:
        raise ValueError(
            f"{name} contains invalid anomaly labels"
        )

    features = data[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    labels = data[
        TARGET_COLUMN
    ].to_numpy(dtype=np.int8)

    if not np.isfinite(features).all():
        raise ValueError(
            f"{name} contains NaN or infinite features"
        )

    return features, labels

def calculate_training_statistics(
    training_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and standard deviation using
    only the training dataset.
    """

    feature_mean = training_features.mean(axis=0)

    feature_standard_deviation = (
        training_features.std(
            axis=0,
            ddof=0,
        )
    )

    if np.any(feature_standard_deviation < 1e-8):
        raise ValueError(
            "At least one feature has almost zero "
            "standard deviation"
        )

    return feature_mean, feature_standard_deviation


def normalize_features(
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_standard_deviation: np.ndarray,
) -> np.ndarray:
    """
    Apply z-score normalization:

        normalized = (value - mean) / standard deviation
    """

    normalized_features = (
        features - feature_mean
    ) / feature_standard_deviation

    return normalized_features.astype(np.float32)


def save_processed_dataset(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    """
    Save model-ready features and labels in one
    compressed NumPy archive.
    """

    np.savez_compressed(
        path,
        features=features,
        labels=labels,
    )


def count_classes(
    labels: np.ndarray,
) -> dict[str, int]:
    """
    Count normal and anomalous labels.
    """

    normal_count = int(
        np.sum(labels == 0)
    )

    anomaly_count = int(
        np.sum(labels == 1)
    )

    return {
        "normal": normal_count,
        "anomaly": anomaly_count,
    }


def print_dataset_information(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    """
    Print the most important information about
    one processed dataset.
    """

    class_counts = count_classes(labels)

    print(f"\n{name.capitalize()} dataset")
    print("-" * 60)

    print(f"Feature shape: {features.shape}")
    print(f"Label shape:   {labels.shape}")
    print(f"Feature dtype: {features.dtype}")
    print(f"Label dtype:   {labels.dtype}")

    print(
        f"Normal rows:   "
        f"{class_counts['normal']}"
    )

    print(
        f"Anomaly rows:  "
        f"{class_counts['anomaly']}"
    )

    print(
        "Feature minimums:",
        np.round(
            features.min(axis=0),
            4,
        ),
    )

    print(
        "Feature maximums:",
        np.round(
            features.max(axis=0),
            4,
        ),
    )


def main() -> None:
    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load raw datasets
    # --------------------------------------------------------

    train_features_raw, train_labels = (
        load_dataset(
            name="training",
            path=TRAIN_FILE,
        )
    )

    validation_features_raw, validation_labels = (
        load_dataset(
            name="validation",
            path=VALIDATION_FILE,
        )
    )

    test_features_raw, test_labels = (
        load_dataset(
            name="test",
            path=TEST_FILE,
        )
    )

    # --------------------------------------------------------
    # 2. Calculate statistics from training data only
    # --------------------------------------------------------

    feature_mean, feature_standard_deviation = (
        calculate_training_statistics(
            train_features_raw
        )
    )

    # --------------------------------------------------------
    # 3. Apply the same normalization to all datasets
    # --------------------------------------------------------

    train_features = normalize_features(
        train_features_raw,
        feature_mean,
        feature_standard_deviation,
    )

    validation_features = normalize_features(
        validation_features_raw,
        feature_mean,
        feature_standard_deviation,
    )

    test_features = normalize_features(
        test_features_raw,
        feature_mean,
        feature_standard_deviation,
    )

    # --------------------------------------------------------
    # 4. Validate the processed arrays
    # --------------------------------------------------------

    for name, features in (
        ("training", train_features),
        ("validation", validation_features),
        ("test", test_features),
    ):
        if not np.isfinite(features).all():
            raise ValueError(
                f"{name} normalized data contains "
                "NaN or infinity"
            )

    # --------------------------------------------------------
    # 5. Save processed arrays
    # --------------------------------------------------------

    save_processed_dataset(
        PROCESSED_DATA_DIRECTORY / "train.npz",
        train_features,
        train_labels,
    )

    save_processed_dataset(
        PROCESSED_DATA_DIRECTORY
        / "validation.npz",
        validation_features,
        validation_labels,
    )

    save_processed_dataset(
        PROCESSED_DATA_DIRECTORY / "test.npz",
        test_features,
        test_labels,
    )

    # --------------------------------------------------------
    # 6. Save preprocessing metadata
    # --------------------------------------------------------

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_order": {
            str(index): feature_name
            for index, feature_name in enumerate(
                FEATURE_COLUMNS
            )
        },
        "target_column": TARGET_COLUMN,
        "excluded_columns": EXCLUDED_COLUMNS,
        "normalization": {
            "method": "z_score",
            "formula": (
                "(value - mean) / "
                "standard_deviation"
            ),
            "statistics_source": (
                "training dataset only"
            ),
            "feature_mean": {
                feature: float(mean)
                for feature, mean in zip(
                    FEATURE_COLUMNS,
                    feature_mean,
                )
            },
            "feature_standard_deviation": {
                feature: float(std)
                for feature, std in zip(
                    FEATURE_COLUMNS,
                    feature_standard_deviation,
                )
            },
        },
        "datasets": {
            "train": {
                "rows": int(len(train_labels)),
                "classes": count_classes(
                    train_labels
                ),
            },
            "validation": {
                "rows": int(
                    len(validation_labels)
                ),
                "classes": count_classes(
                    validation_labels
                ),
            },
            "test": {
                "rows": int(len(test_labels)),
                "classes": count_classes(
                    test_labels
                ),
            },
        },
        "model_input": {
            "number_of_features": len(
                FEATURE_COLUMNS
            ),
            "shape_per_sample": [
                len(FEATURE_COLUMNS)
            ],
            "dtype": "float32",
        },
    }

    metadata_path = (
        PROCESSED_DATA_DIRECTORY
        / "preprocessing_metadata.json"
    )

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 7. Print results
    # --------------------------------------------------------

    print("\nRaw training feature statistics")
    print("=" * 60)

    for index, feature in enumerate(
        FEATURE_COLUMNS
    ):
        print(
            f"{feature}: "
            f"mean={feature_mean[index]:.6f}, "
            f"std="
            f"{feature_standard_deviation[index]:.6f}"
        )

    print_dataset_information(
        name="training",
        features=train_features,
        labels=train_labels,
    )

    print_dataset_information(
        name="validation",
        features=validation_features,
        labels=validation_labels,
    )

    print_dataset_information(
        name="test",
        features=test_features,
        labels=test_labels,
    )

    print("\nNormalized training checks")
    print("=" * 60)

    normalized_training_mean = (
        train_features.mean(axis=0)
    )

    normalized_training_std = (
        train_features.std(axis=0)
    )

    print(
        "Training means:",
        np.round(
            normalized_training_mean,
            6,
        ),
    )

    print(
        "Training standard deviations:",
        np.round(
            normalized_training_std,
            6,
        ),
    )

    print("\nExample transformation")
    print("=" * 60)

    print(
        "Raw first training sample:",
        train_features_raw[0],
    )

    print(
        "Normalized first training sample:",
        train_features[0],
    )

    print(
        "First training label:",
        train_labels[0],
    )

    print(
        "\nProcessed files saved in:\n"
        f"{PROCESSED_DATA_DIRECTORY}"
    )


if __name__ == "__main__":
    main()