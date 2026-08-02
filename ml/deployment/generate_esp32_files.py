#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml/models/baseline_anomaly_int8.tflite"
)

PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "ml/data/processed/preprocessing_metadata.json"
)

CONVERSION_RESULTS_PATH = (
    PROJECT_ROOT
    / "ml/evaluation/conversion/"
    "conversion_results.json"
)

TEST_RESULTS_PATH = (
    PROJECT_ROOT
    / "ml/evaluation/model_test/test_predictions.csv"
)

PROBABILITY_COMPARISON_PATH = (
    PROJECT_ROOT
    / "ml/evaluation/conversion/probability_comparison.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "firmware/tinyml_inference/src"
)


def cpp_float(value: float) -> str:
    text = f"{value:.9g}"

    if "." not in text and "e" not in text.lower():
        text += ".0"

    return f"{text}f"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def write_model_data(model_bytes: bytes) -> None:
    header_path = OUTPUT_DIRECTORY / "model_data.h"
    source_path = OUTPUT_DIRECTORY / "model_data.cc"

    header_path.write_text(
        """#pragma once

extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;
""",
        encoding="utf-8",
    )

    lines = []

    bytes_per_line = 12

    for start in range(
        0,
        len(model_bytes),
        bytes_per_line,
    ):
        section = model_bytes[
            start:start + bytes_per_line
        ]

        formatted = ", ".join(
            f"0x{value:02x}"
            for value in section
        )

        lines.append(f"    {formatted},")

    source = """#include "model_data.h"

alignas(16) const unsigned char g_model_data[] = {
"""

    source += "\n".join(lines)

    source += """
};

const unsigned int g_model_data_len =
    sizeof(g_model_data);
"""

    source_path.write_text(
        source,
        encoding="utf-8",
    )


def write_model_config(
    feature_columns: list[str],
    feature_means: list[float],
    feature_standard_deviations: list[float],
    threshold: float,
) -> None:
    path = OUTPUT_DIRECTORY / "model_config.h"

    means = ",\n    ".join(
        cpp_float(value)
        for value in feature_means
    )

    standard_deviations = ",\n    ".join(
        cpp_float(value)
        for value in feature_standard_deviations
    )

    feature_names = ",\n    ".join(
        f'"{name}"'
        for name in feature_columns
    )

    content = f"""#pragma once

#include <cstddef>

inline constexpr std::size_t kFeatureCount =
    {len(feature_columns)};

inline constexpr const char* kFeatureNames[
    kFeatureCount
] = {{
    {feature_names}
}};

inline constexpr float kFeatureMean[
    kFeatureCount
] = {{
    {means}
}};

inline constexpr float kFeatureStandardDeviation[
    kFeatureCount
] = {{
    {standard_deviations}
}};

inline constexpr float kClassificationThreshold =
    {cpp_float(threshold)};
"""

    path.write_text(
        content,
        encoding="utf-8",
    )


def select_deployment_samples(
    threshold: float,
    feature_columns: list[str],
) -> list[dict]:
    test_results = pd.read_csv(
        TEST_RESULTS_PATH
    )

    probability_comparison = pd.read_csv(
        PROBABILITY_COMPARISON_PATH
    )

    if len(test_results) != len(
        probability_comparison
    ):
        raise ValueError(
            "Prediction files have different lengths"
        )

    test_results = test_results.copy()

    test_results["int8_probability"] = (
        probability_comparison[
            "int8_tflite_probability"
        ].to_numpy()
    )

    test_results["int8_prediction"] = (
        test_results["int8_probability"]
        >= threshold
    ).astype(int)

    correct_results = test_results[
        test_results["anomaly"]
        == test_results["int8_prediction"]
    ]

    print("\n===== DEBUG =====")

    print("\nThreshold:", threshold)

    print("\nAll anomaly types:")
    print(test_results["anomaly_type"].value_counts())

    print("\nCorrectly classified:")
    print(correct_results["anomaly_type"].value_counts())

    print("\nTemperature spike rows:")
    temperature_rows = test_results[
        test_results["anomaly_type"] == "temperature_spike"
    ]

    print(temperature_rows[[
        "anomaly",
        "int8_probability",
        "int8_prediction",
    ]])

    print("=================\n")

    selected_samples = []

    normal_rows = correct_results[
        correct_results["anomaly_type"]
        == "normal"
    ]

    if normal_rows.empty:
        raise ValueError(
            "No correctly classified normal sample found"
        )

    normal_sample = normal_rows.sort_values(
        "int8_probability",
        ascending=True,
    ).iloc[0]

    selected_samples.append({
        "name": "normal",
        "features": [
            float(normal_sample[column])
            for column in feature_columns
        ],
        "expected_label": 0,
    })

    anomaly_types = [
        "temperature_spike",
        "humidity_drop",
        "light_saturation",
    ]

    for anomaly_type in anomaly_types:
        rows = correct_results[
            correct_results["anomaly_type"]
            == anomaly_type
        ]

        if rows.empty:
            raise ValueError(
                "No correctly classified sample found "
                f"for {anomaly_type}"
            )

        sample = rows.sort_values(
            "int8_probability",
            ascending=False,
        ).iloc[0]

        selected_samples.append({
            "name": anomaly_type,
            "features": [
                float(sample[column])
                for column in feature_columns
            ],
            "expected_label": 1,
        })

    return selected_samples


def write_test_samples(
    samples: list[dict],
) -> None:
    path = (
        OUTPUT_DIRECTORY
        / "deployment_test_samples.h"
    )

    rows = []

    for sample in samples:
        feature_values = ", ".join(
            cpp_float(value)
            for value in sample["features"]
        )

        rows.append(
            "    {\n"
            f'        "{sample["name"]}",\n'
            f"        {{{feature_values}}},\n"
            f'        {sample["expected_label"]}\n'
            "    },"
        )

    content = f"""#pragma once

#include <cstddef>

#include "model_config.h"

struct DeploymentTestSample
{{
    const char* name;
    float features[kFeatureCount];
    int expected_label;
}};

inline constexpr DeploymentTestSample
    kDeploymentTestSamples[] = {{
{chr(10).join(rows)}
}};

inline constexpr std::size_t
    kDeploymentTestSampleCount =
        sizeof(kDeploymentTestSamples)
        / sizeof(kDeploymentTestSamples[0]);
"""

    path.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    required_files = [
        MODEL_PATH,
        PREPROCESSING_PATH,
        CONVERSION_RESULTS_PATH,
        TEST_RESULTS_PATH,
        PROBABILITY_COMPARISON_PATH,
    ]

    for path in required_files:
        require_file(path)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bytes = MODEL_PATH.read_bytes()

    with PREPROCESSING_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        preprocessing = json.load(file)

    with CONVERSION_RESULTS_PATH.open(
    mode="r",
    encoding="utf-8",
    ) as file:
        conversion_results = json.load(file)

    feature_columns = preprocessing[
        "feature_columns"
    ]

    means_dictionary = preprocessing[
        "normalization"
    ]["feature_mean"]

    standard_deviation_dictionary = preprocessing[
        "normalization"
    ]["feature_standard_deviation"]

    feature_means = [
        float(means_dictionary[feature])
        for feature in feature_columns
    ]

    feature_standard_deviations = [
        float(
            standard_deviation_dictionary[feature]
        )
        for feature in feature_columns
    ]

    threshold = float(
    conversion_results[
        "int8_tflite_model"
    ]["deployment_threshold"]
    )

    deployment_samples = (
        select_deployment_samples(
            threshold,
            feature_columns,
        )
    )

    write_model_data(model_bytes)

    write_model_config(
        feature_columns,
        feature_means,
        feature_standard_deviations,
        threshold,
    )

    write_test_samples(
        deployment_samples
    )

    print(
        f"Model bytes generated: {len(model_bytes)}"
    )

    print(
        f"Classification threshold: {threshold:.9f}"
    )

    print("\nGenerated deployment samples:")

    for sample in deployment_samples:
        print(sample)

    print(
        "\nGenerated files saved in:\n"
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()