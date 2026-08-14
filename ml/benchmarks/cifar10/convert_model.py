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

KERAS_MODEL_PATH = (
    MODEL_DIRECTORY
    / "cifar10_cnn.keras"
)

FLOAT32_MODEL_PATH = (
    MODEL_DIRECTORY
    / "cifar10_float32.tflite"
)

INT8_MODEL_PATH = (
    MODEL_DIRECTORY
    / "cifar10_int8.tflite"
)


def representative_dataset(
    x_train: np.ndarray,
):
    """
    Representative samples are used by the
    TFLite converter to determine suitable
    quantization ranges.

    Only training data is used here.
    """
    sample_count = min(
        1000,
        len(x_train),
    )

    for index in range(sample_count):
        sample = x_train[
            index:index + 1
        ]

        yield [
            sample.astype(
                np.float32
            )
        ]


def print_tensor_details(
    model_path: Path,
    model_name: str,
) -> None:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()[0]
    )

    output_details = (
        interpreter.get_output_details()[0]
    )

    print()
    print(model_name)
    print("-" * 40)

    print("Input:")
    print(
        "  shape:",
        input_details["shape"],
    )
    print(
        "  dtype:",
        input_details["dtype"],
    )
    print(
        "  quantization:",
        input_details["quantization"],
    )

    print("Output:")
    print(
        "  shape:",
        output_details["shape"],
    )
    print(
        "  dtype:",
        output_details["dtype"],
    )
    print(
        "  quantization:",
        output_details["quantization"],
    )


def main() -> None:
    (
        x_train,
        _,
        _,
        _,
    ) = load_and_preprocess_data()

    print("Loading Keras model:")

    print(
        KERAS_MODEL_PATH
    )

    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Float32 TFLite
    # -----------------------------------------

    print()
    print(
        "Converting Float32 TFLite model..."
    )

    float32_converter = (
        tf.lite.TFLiteConverter.from_keras_model(
            model
        )
    )

    float32_model = (
        float32_converter.convert()
    )

    FLOAT32_MODEL_PATH.write_bytes(
        float32_model
    )

    print(
        "Saved:",
        FLOAT32_MODEL_PATH,
    )

    # -----------------------------------------
    # Full INT8 TFLite
    # -----------------------------------------

    print()
    print(
        "Converting Full INT8 TFLite model..."
    )

    int8_converter = (
        tf.lite.TFLiteConverter.from_keras_model(
            model
        )
    )

    int8_converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    int8_converter.representative_dataset = (
        lambda: representative_dataset(
            x_train
        )
    )

    int8_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    int8_converter.inference_input_type = (
        tf.int8
    )

    int8_converter.inference_output_type = (
        tf.int8
    )

    int8_model = int8_converter.convert()

    INT8_MODEL_PATH.write_bytes(
        int8_model
    )

    print(
        "Saved:",
        INT8_MODEL_PATH,
    )

    # -----------------------------------------
    # Model sizes
    # -----------------------------------------

    keras_size = (
        KERAS_MODEL_PATH.stat().st_size
    )

    float32_size = (
        FLOAT32_MODEL_PATH.stat().st_size
    )

    int8_size = (
        INT8_MODEL_PATH.stat().st_size
    )

    print()
    print("Model sizes")
    print("=" * 40)

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
        "Full INT8 TFLite:",
        int8_size,
        "bytes",
    )

    # -----------------------------------------
    # Tensor information
    # -----------------------------------------

    print_tensor_details(
        FLOAT32_MODEL_PATH,
        "Float32 TFLite",
    )

    print_tensor_details(
        INT8_MODEL_PATH,
        "Full INT8 TFLite",
    )


if __name__ == "__main__":
    main()
