import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocess_dataset import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "cifar10"
    / "models"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cifar10"
    / "host_results"
    / "tflite_evaluation.json"
)

FLOAT32_MODEL_PATH = (
    MODEL_DIRECTORY
    / "cifar10_float32.tflite"
)

INT8_MODEL_PATH = (
    MODEL_DIRECTORY
    / "cifar10_int8.tflite"
)

KERAS_MODEL_PATH = (
    MODEL_DIRECTORY
    / "cifar10_cnn.keras"
)


def evaluate_float32(
    model_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    correct = 0

    for index in range(len(x_test)):
        sample = x_test[index:index + 1]

        interpreter.set_tensor(
            input_details["index"],
            sample.astype(np.float32),
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        predicted_class = int(
            np.argmax(output[0])
        )

        if predicted_class == int(y_test[index]):
            correct += 1

    return correct / len(y_test)


def evaluate_int8(
    model_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    correct = 0

    for index in range(len(x_test)):
        sample = x_test[index:index + 1]

        quantized_sample = np.round(
            sample / input_scale
            + input_zero_point
        )

        quantized_sample = np.clip(
            quantized_sample,
            -128,
            127,
        ).astype(np.int8)

        interpreter.set_tensor(
            input_details["index"],
            quantized_sample,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        predicted_class = int(
            np.argmax(output[0])
        )

        if predicted_class == int(y_test[index]):
            correct += 1

    return correct / len(y_test)


def main() -> None:
    (
        _,
        _,
        x_test,
        y_test,
    ) = load_and_preprocess_data()

    keras_model = tf.keras.models.load_model(
        KERAS_MODEL_PATH
    )

    _, keras_accuracy = keras_model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print("Evaluating Float32 TFLite...")

    float32_accuracy = evaluate_float32(
        FLOAT32_MODEL_PATH,
        x_test,
        y_test,
    )

    print("Evaluating Full INT8 TFLite...")

    int8_accuracy = evaluate_int8(
        INT8_MODEL_PATH,
        x_test,
        y_test,
    )

    keras_size = KERAS_MODEL_PATH.stat().st_size
    float32_size = FLOAT32_MODEL_PATH.stat().st_size
    int8_size = INT8_MODEL_PATH.stat().st_size

    results = {
        "keras": {
            "accuracy": float(keras_accuracy),
            "size_bytes": keras_size,
        },
        "float32_tflite": {
            "accuracy": float(float32_accuracy),
            "size_bytes": float32_size,
        },
        "int8_tflite": {
            "accuracy": float(int8_accuracy),
            "size_bytes": int8_size,
        },
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("CIFAR-10 TFLite evaluation")
    print("=" * 40)

    print(
        "Keras accuracy:",
        keras_accuracy,
    )

    print(
        "Float32 TFLite accuracy:",
        float32_accuracy,
    )

    print(
        "INT8 TFLite accuracy:",
        int8_accuracy,
    )

    print()
    print("Model sizes:")

    print(
        "Keras:",
        keras_size,
        "bytes",
    )

    print(
        "Float32 TFLite:",
        float32_size,
        "bytes",
    )

    print(
        "INT8 TFLite:",
        int8_size,
        "bytes",
    )

    print()
    print(
        "Results saved to:",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()
