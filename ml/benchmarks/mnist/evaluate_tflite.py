import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocess_dataset import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mnist"
    / "models"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mnist"
    / "host_results"
)

FLOAT32_MODEL_PATH = (
    MODEL_DIRECTORY
    / "mnist_float32.tflite"
)

INT8_MODEL_PATH = (
    MODEL_DIRECTORY
    / "mnist_int8.tflite"
)

KERAS_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "keras_evaluation.json"
)

OUTPUT_PATH = (
    RESULTS_DIRECTORY
    / "tflite_evaluation.json"
)


def quantize_input(
    sample: np.ndarray,
    scale: float,
    zero_point: int,
    dtype: type,
) -> np.ndarray:
    """
    Quantize a float32 input sample using the
    TFLite tensor quantization parameters.
    """

    quantized = np.round(
        sample / scale + zero_point
    )

    limits = np.iinfo(dtype)

    quantized = np.clip(
        quantized,
        limits.min,
        limits.max,
    )

    return quantized.astype(dtype)


def evaluate_model(
    model_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate one TFLite model using batch-size-one inference.
    """

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details["index"]
    output_index = output_details["index"]

    input_dtype = input_details["dtype"]

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    output_scale, output_zero_point = (
        output_details["quantization"]
    )

    predictions = []

    for sample in x_test:
        sample = np.expand_dims(
            sample,
            axis=0,
        )

        if input_dtype == np.float32:
            model_input = sample.astype(
                np.float32
            )

        else:
            if input_scale == 0:
                raise ValueError(
                    "Invalid INT8 input scale."
                )

            model_input = quantize_input(
                sample=sample,
                scale=input_scale,
                zero_point=input_zero_point,
                dtype=input_dtype,
            )

        interpreter.set_tensor(
            input_index,
            model_input,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )[0]

        predicted_digit = int(
            np.argmax(output)
        )

        predictions.append(
            predicted_digit
        )

    predictions = np.array(
        predictions,
        dtype=np.uint8,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    return {
        "model": model_path.name,
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "model_size_bytes": int(
            model_path.stat().st_size
        ),
        "input_shape": (
            input_details["shape"].tolist()
        ),
        "input_dtype": str(
            input_details["dtype"]
        ),
        "input_scale": float(
            input_scale
        ),
        "input_zero_point": int(
            input_zero_point
        ),
        "output_shape": (
            output_details["shape"].tolist()
        ),
        "output_dtype": str(
            output_details["dtype"]
        ),
        "output_scale": float(
            output_scale
        ),
        "output_zero_point": int(
            output_zero_point
        ),
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    _, _, x_test, y_test = (
        load_and_preprocess_data()
    )

    print("Evaluating Float32 TFLite...")

    float32_results = evaluate_model(
        FLOAT32_MODEL_PATH,
        x_test,
        y_test,
    )

    print(
        "Float32 accuracy:",
        float32_results["accuracy"],
    )

    print()
    print("Evaluating INT8 TFLite...")

    int8_results = evaluate_model(
        INT8_MODEL_PATH,
        x_test,
        y_test,
    )

    print(
        "INT8 accuracy:",
        int8_results["accuracy"],
    )

    keras_accuracy = None

    if KERAS_RESULTS_PATH.exists():
        keras_results = json.loads(
            KERAS_RESULTS_PATH.read_text(
                encoding="utf-8"
            )
        )

        keras_accuracy = keras_results[
            "accuracy"
        ]

    results = {
        "keras_baseline_accuracy": keras_accuracy,
        "float32_tflite": float32_results,
        "int8_tflite": int8_results,
    }

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Comparison:")

    print(
        "Keras:",
        keras_accuracy,
    )

    print(
        "Float32 TFLite:",
        float32_results["accuracy"],
    )

    print(
        "INT8 TFLite:",
        int8_results["accuracy"],
    )

    print()
    print("Results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
