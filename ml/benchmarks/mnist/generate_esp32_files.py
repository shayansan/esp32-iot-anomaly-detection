from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocess_dataset import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "mnist"
    / "models"
    / "mnist_int8.tflite"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "firmware"
    / "mnist_tflm"
    / "main"
)

MODEL_HEADER_PATH = OUTPUT_DIRECTORY / "model_data.h"
MODEL_SOURCE_PATH = OUTPUT_DIRECTORY / "model_data.cc"

CONFIG_HEADER_PATH = OUTPUT_DIRECTORY / "model_config.h"

TEST_HEADER_PATH = OUTPUT_DIRECTORY / "test_data.h"
TEST_SOURCE_PATH = OUTPUT_DIRECTORY / "test_data.cc"


def format_model_bytes(
    data: bytes,
    values_per_line: int = 12,
) -> str:
    """
    Format binary model bytes as a C++ hexadecimal array.
    """

    values = [
        f"0x{value:02x}"
        for value in data
    ]

    lines = []

    for index in range(
        0,
        len(values),
        values_per_line,
    ):
        chunk = values[
            index:index + values_per_line
        ]

        lines.append(
            "    " + ", ".join(chunk)
        )

    return ",\n".join(lines)


def format_int8_values(
    values: np.ndarray,
    values_per_line: int = 16,
) -> str:
    """
    Format INT8 values as a C++ array.
    """

    flat_values = values.flatten()

    lines = []

    for index in range(
        0,
        len(flat_values),
        values_per_line,
    ):
        chunk = flat_values[
            index:index + values_per_line
        ]

        formatted = ", ".join(
            str(int(value))
            for value in chunk
        )

        lines.append(
            "    " + formatted
        )

    return ",\n".join(lines)


def quantize_sample(
    sample: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """
    Convert normalized float32 MNIST input to INT8.
    """

    quantized = np.round(
        sample / scale + zero_point
    )

    quantized = np.clip(
        quantized,
        -128,
        127,
    )

    return quantized.astype(np.int8)


def write_model_files(
    model_data: bytes,
) -> None:
    """
    Generate model_data.h and model_data.cc.
    """

    model_header = """\
#pragma once

extern const unsigned char g_mnist_model[];
extern const unsigned int g_mnist_model_len;
"""

    model_source = f"""\
#include "model_data.h"

alignas(16) const unsigned char g_mnist_model[] = {{
{format_model_bytes(model_data)}
}};

const unsigned int g_mnist_model_len =
    sizeof(g_mnist_model);
"""

    MODEL_HEADER_PATH.write_text(
        model_header,
        encoding="utf-8",
    )

    MODEL_SOURCE_PATH.write_text(
        model_source,
        encoding="utf-8",
    )


def write_model_config(
    input_details: dict,
    output_details: dict,
) -> None:
    """
    Generate model_config.h from real TFLite tensor metadata.
    """

    input_shape = input_details["shape"]

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    output_scale, output_zero_point = (
        output_details["quantization"]
    )

    config_header = f"""\
#pragma once

namespace mnist {{

constexpr int kInputBatch = {int(input_shape[0])};
constexpr int kInputHeight = {int(input_shape[1])};
constexpr int kInputWidth = {int(input_shape[2])};
constexpr int kInputChannels = {int(input_shape[3])};

constexpr int kInputElementCount =
    kInputHeight * kInputWidth * kInputChannels;

constexpr int kOutputClassCount = 10;

constexpr float kInputScale =
    {float(input_scale):.17g}f;

constexpr int kInputZeroPoint =
    {int(input_zero_point)};

constexpr float kOutputScale =
    {float(output_scale):.17g}f;

constexpr int kOutputZeroPoint =
    {int(output_zero_point)};

}}  // namespace mnist
"""

    CONFIG_HEADER_PATH.write_text(
        config_header,
        encoding="utf-8",
    )


def write_test_sample(
    input_scale: float,
    input_zero_point: int,
) -> None:
    """
    Generate one known MNIST test sample for
    initial ESP32 inference validation.
    """

    _, _, x_test, y_test = (
        load_and_preprocess_data()
    )

    sample = x_test[0]
    expected_label = int(y_test[0])

    quantized_sample = quantize_sample(
        sample,
        input_scale,
        input_zero_point,
    )

    test_header = """\
#pragma once

#include <cstdint>

extern const std::int8_t g_mnist_test_input[];
extern const int g_mnist_test_expected_label;
"""

    test_source = f"""\
#include "test_data.h"

const std::int8_t g_mnist_test_input[] = {{
{format_int8_values(quantized_sample)}
}};

const int g_mnist_test_expected_label =
    {expected_label};
"""

    TEST_HEADER_PATH.write_text(
        test_header,
        encoding="utf-8",
    )

    TEST_SOURCE_PATH.write_text(
        test_source,
        encoding="utf-8",
    )

    print(
        "Generated test sample with expected label:",
        expected_label,
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data = MODEL_PATH.read_bytes()

    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL_PATH)
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()[0]
    )

    output_details = (
        interpreter.get_output_details()[0]
    )

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    write_model_files(
        model_data
    )

    write_model_config(
        input_details,
        output_details,
    )

    write_test_sample(
        input_scale,
        input_zero_point,
    )

    print()
    print("Generated ESP32 files:")

    for path in [
        MODEL_HEADER_PATH,
        MODEL_SOURCE_PATH,
        CONFIG_HEADER_PATH,
        TEST_HEADER_PATH,
        TEST_SOURCE_PATH,
    ]:
        print(path)

    print()
    print(
        "Embedded model size:",
        len(model_data),
        "bytes",
    )


if __name__ == "__main__":
    main()
