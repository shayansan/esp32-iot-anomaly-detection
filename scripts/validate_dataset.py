#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "timestamp_ms",
    "temperature_c",
    "humidity_percent",
    "light_raw",
    "anomaly",
]


def parse_arguments() -> argparse.Namespace:
    """Read the dataset path and expected row count."""

    parser = argparse.ArgumentParser(
        description="Validate an ESP32 synthetic sensor dataset."
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the CSV dataset",
    )

    parser.add_argument(
        "--expected-rows",
        type=int,
        required=True,
        help="Expected number of data rows",
    )

    return parser.parse_args()


def expected_anomaly_count(number_of_rows: int) -> int:
    """
    Calculate the expected number of anomaly rows.

    This reproduces the ESP32 condition:

        sampleNumber >= 100
        sampleNumber % 100 < 5
    """

    count = 0

    for sample_number in range(number_of_rows):
        if (
            sample_number >= 100
            and sample_number % 100 < 5
        ):
            count += 1

    return count


def print_check(name: str, passed: bool) -> None:
    """Print a readable PASS or FAIL result."""

    result = "PASS" if passed else "FAIL"
    print(f"[{result}] {name}")


def main() -> None:
    args = parse_arguments()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {args.dataset}"
        )

    data = pd.read_csv(args.dataset)

    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Basic structure
    # ---------------------------------------------------------

    columns_are_correct = (
        data.columns.tolist() == EXPECTED_COLUMNS
    )

    row_count_is_correct = (
        len(data) == args.expected_rows
    )

    print_check(
        "CSV columns are correct",
        columns_are_correct,
    )

    print_check(
        f"Row count is {args.expected_rows}",
        row_count_is_correct,
    )

    # ---------------------------------------------------------
    # 2. Missing and duplicate values
    # ---------------------------------------------------------

    has_no_missing_values = not data.isna().any().any()

    duplicate_rows = data.duplicated().sum()

    print_check(
        "Dataset contains no missing values",
        has_no_missing_values,
    )

    print_check(
        "Dataset contains no duplicate rows",
        duplicate_rows == 0,
    )

    # ---------------------------------------------------------
    # 3. Label validation
    # ---------------------------------------------------------

    unique_labels = set(
        data["anomaly"].unique().tolist()
    )

    labels_are_binary = unique_labels.issubset({0, 1})

    actual_anomalies = int(data["anomaly"].sum())

    expected_anomalies = expected_anomaly_count(
        args.expected_rows
    )

    print_check(
        "Anomaly labels contain only 0 and 1",
        labels_are_binary,
    )

    print_check(
        (
            f"Anomaly count is {expected_anomalies} "
            f"(actual: {actual_anomalies})"
        ),
        actual_anomalies == expected_anomalies,
    )

    # ---------------------------------------------------------
    # 4. Timestamp validation
    # ---------------------------------------------------------

    timestamp_differences = (
        data["timestamp_ms"]
        .diff()
        .dropna()
    )

    timestamps_increase = (
        timestamp_differences > 0
    ).all()

    median_interval = float(
        timestamp_differences.median()
    )

    interval_is_reasonable = (
        95.0 <= median_interval <= 105.0
    )

    print_check(
        "Timestamps are strictly increasing",
        timestamps_increase,
    )

    print_check(
        (
            "Median sample interval is approximately "
            f"100 ms (actual: {median_interval:.2f} ms)"
        ),
        interval_is_reasonable,
    )

    # ---------------------------------------------------------
    # 5. Sensor limits
    # ---------------------------------------------------------

    temperature_is_valid = data[
        "temperature_c"
    ].between(-20.0, 80.0).all()

    humidity_is_valid = data[
        "humidity_percent"
    ].between(0.0, 100.0).all()

    light_is_valid = data[
        "light_raw"
    ].between(0.0, 1023.0).all()

    print_check(
        "Temperature values are within valid limits",
        temperature_is_valid,
    )

    print_check(
        "Humidity values are within valid limits",
        humidity_is_valid,
    )

    print_check(
        "Light values are within valid limits",
        light_is_valid,
    )

    # ---------------------------------------------------------
    # 6. Identify the three anomaly types
    # ---------------------------------------------------------

    anomaly_rows = data[data["anomaly"] == 1]

    temperature_spikes = anomaly_rows[
        anomaly_rows["temperature_c"] > 30.0
    ]

    humidity_drops = anomaly_rows[
        anomaly_rows["humidity_percent"] < 30.0
    ]

    light_saturation = anomaly_rows[
        np.isclose(
            anomaly_rows["light_raw"],
            1023.0,
            atol=0.01,
        )
    ]

    recognized_anomalies = (
        len(temperature_spikes)
        + len(humidity_drops)
        + len(light_saturation)
    )

    all_anomalies_recognized = (
        recognized_anomalies == actual_anomalies
    )

    all_types_are_present = all(
        count > 0
        for count in (
            len(temperature_spikes),
            len(humidity_drops),
            len(light_saturation),
        )
    )

    print_check(
        "Every anomaly matches a known anomaly type",
        all_anomalies_recognized,
    )

    print_check(
        "All three anomaly types are present",
        all_types_are_present,
    )

    # ---------------------------------------------------------
    # 7. Check anomaly event lengths
    # ---------------------------------------------------------

    anomaly_mask = data["anomaly"].eq(1)

    group_identifiers = (
        anomaly_mask.ne(anomaly_mask.shift())
        .cumsum()
    )

    event_lengths = (
        data[anomaly_mask]
        .groupby(group_identifiers[anomaly_mask])
        .size()
        .tolist()
    )

    events_have_five_samples = (
        len(event_lengths) > 0
        and all(length == 5 for length in event_lengths)
    )

    print_check(
        "Every anomaly event lasts exactly five samples",
        events_have_five_samples,
    )

    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------

    normal_count = int(
        (data["anomaly"] == 0).sum()
    )

    anomaly_percentage = (
        actual_anomalies / len(data) * 100.0
    )

    print("\nSummary")
    print("-" * 70)

    print(f"Rows:                {len(data)}")
    print(f"Normal rows:         {normal_count}")
    print(f"Anomaly rows:        {actual_anomalies}")
    print(
        f"Anomaly percentage:  "
        f"{anomaly_percentage:.2f}%"
    )

    print(
        f"Temperature spikes:  "
        f"{len(temperature_spikes)}"
    )

    print(
        f"Humidity drops:      "
        f"{len(humidity_drops)}"
    )

    print(
        f"Light saturation:    "
        f"{len(light_saturation)}"
    )

    print(
        f"Anomaly events:      "
        f"{len(event_lengths)}"
    )

    print("\nNumeric statistics")
    print("-" * 70)

    print(
        data[
            [
                "temperature_c",
                "humidity_percent",
                "light_raw",
                "anomaly",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()