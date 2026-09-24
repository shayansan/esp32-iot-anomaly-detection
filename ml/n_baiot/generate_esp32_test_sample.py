from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "preprocessed"
    / "reduced_60"
    / "test.npz"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "models"
    / "reduced_60"
    / "model_int8.tflite"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "firmware"
    / "n_baiot_tflm"
    / "main"
    / "test_sample.h"
)

SAMPLE_INDEX = 2


def main():
    data = np.load(TEST_PATH)

    x_test = data["x"]
    y_test = data["y"]

    sample = x_test[
        SAMPLE_INDEX:SAMPLE_INDEX + 1
    ].astype(np.float32)

    expected_label = int(
        y_test[SAMPLE_INDEX]
    )

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

    output_scale, output_zero_point = (
        output_details["quantization"]
    )

    quantized_input = np.round(
        sample / input_scale
        + input_zero_point
    )

    quantized_input = np.clip(
        quantized_input,
        -128,
        127,
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized_input,
    )

    interpreter.invoke()

    quantized_output = int(
        interpreter.get_tensor(
            output_details["index"]
        ).reshape(-1)[0]
    )

    probability = (
        quantized_output
        - output_zero_point
    ) * output_scale

    predicted_label = int(
        probability >= 0.5
    )

    values = ", ".join(
        str(int(v))
        for v in quantized_input.reshape(-1)
    )

    content = f"""#pragma once

#include <cstdint>

static const int8_t g_test_input[60] = {{
    {values}
}};

static const int g_expected_label = {expected_label};
static const int g_host_predicted_label = {predicted_label};
"""

    OUTPUT_PATH.write_text(
        content,
        encoding="utf-8",
    )

    print("Sample index:", SAMPLE_INDEX)
    print("Expected label:", expected_label)
    print("Host predicted label:", predicted_label)
    print("Quantized output:", quantized_output)
    print("Probability:", probability)
    print("Input scale:", input_scale)
    print("Input zero point:", input_zero_point)
    print("Output scale:", output_scale)
    print("Output zero point:", output_zero_point)
    print("\nGenerated:", OUTPUT_PATH)


if __name__ == "__main__":
    main()