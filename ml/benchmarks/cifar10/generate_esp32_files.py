from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocess_dataset import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cifar10"
    / "models"
    / "cifar10_int8.tflite"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "firmware"
    / "cifar10_tflm"
    / "main"
)


def format_byte_array(
    data: bytes,
    values_per_line: int = 12,
) -> str:
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
        line = ", ".join(
            values[
                index:
                index + values_per_line
            ]
        )

        lines.append(
            "    " + line
        )

    return ",\n".join(lines)


def format_int8_array(
    data: np.ndarray,
    values_per_line: int = 16,
) -> str:
    flat_data = data.flatten()

    values = [
        str(int(value))
        for value in flat_data
    ]

    lines = []

    for index in range(
        0,
        len(values),
        values_per_line,
    ):
        line = ", ".join(
            values[
                index:
                index + values_per_line
            ]
        )

        lines.append(
            "    " + line
        )

    return ",\n".join(lines)


def quantize_input(
    sample: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    quantized = np.round(
        sample / scale
        + zero_point
    )

    quantized = np.clip(
        quantized,
        -128,
        127,
    )

    return quantized.astype(
        np.int8
    )


def write_model_data(
    model_bytes: bytes,
) -> None:
    header_path = (
        OUTPUT_DIRECTORY
        / "model_data.h"
    )

    source_path = (
        OUTPUT_DIRECTORY
        / "model_data.cc"
    )

    header_path.write_text(
        """#pragma once

#include <cstddef>
#include <cstdint>

extern const unsigned char g_cifar10_model_data[];
extern const std::size_t g_cifar10_model_data_len;
""",
        encoding="utf-8",
    )

    source_content = f"""#include "model_data.h"

alignas(16) const unsigned char g_cifar10_model_data[] = {{
{format_byte_array(model_bytes)}
}};

const std::size_t g_cifar10_model_data_len =
    sizeof(g_cifar10_model_data);
"""

    source_path.write_text(
        source_content,
        encoding="utf-8",
    )


def write_model_config(
    input_details: dict,
    output_details: dict,
) -> None:
    input_shape = input_details["shape"]

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    output_scale, output_zero_point = (
        output_details["quantization"]
    )

    config_path = (
        OUTPUT_DIRECTORY
        / "model_config.h"
    )

    config_content = f"""#pragma once

namespace cifar10
{{

constexpr int kInputBatch =
    {int(input_shape[0])};

constexpr int kInputHeight =
    {int(input_shape[1])};

constexpr int kInputWidth =
    {int(input_shape[2])};

constexpr int kInputChannels =
    {int(input_shape[3])};

constexpr int kInputElementCount =
    kInputHeight
    * kInputWidth
    * kInputChannels;

constexpr int kOutputClassCount =
    {int(output_details["shape"][-1])};

constexpr float kInputScale =
    {float(input_scale):.17g}f;

constexpr int kInputZeroPoint =
    {int(input_zero_point)};

constexpr float kOutputScale =
    {float(output_scale):.17g}f;

constexpr int kOutputZeroPoint =
    {int(output_zero_point)};

}}  // namespace cifar10
"""

    config_path.write_text(
        config_content,
        encoding="utf-8",
    )


def write_test_data(
    quantized_sample: np.ndarray,
    expected_label: int,
) -> None:
    header_path = (
        OUTPUT_DIRECTORY
        / "test_data.h"
    )

    source_path = (
        OUTPUT_DIRECTORY
        / "test_data.cc"
    )

    header_path.write_text(
        """#pragma once

#include <cstddef>
#include <cstdint>

extern const std::int8_t g_cifar10_test_input[];
extern const std::size_t g_cifar10_test_input_len;
extern const int g_cifar10_test_expected_label;
""",
        encoding="utf-8",
    )

    source_content = f"""#include "test_data.h"

const std::int8_t g_cifar10_test_input[] = {{
{format_int8_array(quantized_sample)}
}};

const std::size_t g_cifar10_test_input_len =
    sizeof(g_cifar10_test_input)
    / sizeof(g_cifar10_test_input[0]);

const int g_cifar10_test_expected_label =
    {expected_label};
"""

    source_path.write_text(
        source_content,
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bytes = MODEL_PATH.read_bytes()

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

    (
        _,
        _,
        x_test,
        y_test,
    ) = load_and_preprocess_data()

    test_sample = x_test[0:1]

    expected_label = int(
        y_test[0]
    )

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    quantized_sample = quantize_input(
        test_sample,
        input_scale,
        input_zero_point,
    )

    write_model_data(
        model_bytes
    )

    write_model_config(
        input_details,
        output_details,
    )

    write_test_data(
        quantized_sample,
        expected_label,
    )

    print("CIFAR-10 ESP32 files generated")
    print("=" * 40)

    print(
        "Model size:",
        len(model_bytes),
        "bytes",
    )

    print(
        "Input shape:",
        input_details["shape"],
    )

    print(
        "Input dtype:",
        input_details["dtype"],
    )

    print(
        "Input scale:",
        input_scale,
    )

    print(
        "Input zero point:",
        input_zero_point,
    )

    print(
        "Output shape:",
        output_details["shape"],
    )

    print(
        "Output dtype:",
        output_details["dtype"],
    )

    print(
        "Output scale:",
        output_details["quantization"][0],
    )

    print(
        "Output zero point:",
        output_details["quantization"][1],
    )

    print(
        "Test sample index: 0"
    )

    print(
        "Expected label:",
        expected_label,
    )

    print()
    print("Generated files:")

    for file_name in [
        "model_data.h",
        "model_data.cc",
        "model_config.h",
        "test_data.h",
        "test_data.cc",
    ]:
        path = (
            OUTPUT_DIRECTORY
            / file_name
        )

        print(
            f"  {file_name}: "
            f"{path.stat().st_size} bytes"
        )


if __name__ == "__main__":
    main()
