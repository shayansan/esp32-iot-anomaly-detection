from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from preprocess_dataset import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mnist"
    / "models"
)

KERAS_MODEL_PATH = (
    MODEL_DIRECTORY
    / "mnist_cnn.keras"
)

FLOAT32_MODEL_PATH = (
    MODEL_DIRECTORY
    / "mnist_float32.tflite"
)

INT8_MODEL_PATH = (
    MODEL_DIRECTORY
    / "mnist_int8.tflite"
)


def convert_float32_model(
    model: keras.Model,
) -> bytes:
    """
    Convert the Keras model to standard Float32 TFLite.
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    return tflite_model


def representative_dataset():
    """
    Provide representative samples for INT8 calibration.
    """

    x_train, _, _, _ = load_and_preprocess_data()

    # A subset is enough for quantization calibration.
    calibration_samples = x_train[:1000]

    for sample in calibration_samples:
        sample = np.expand_dims(
            sample,
            axis=0,
        ).astype(np.float32)

        yield [sample]


def convert_int8_model(
    model: keras.Model,
) -> bytes:
    """
    Convert the Keras model to a fully quantized INT8 TFLite model.
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    return tflite_model


def inspect_tflite_model(
    model_path: Path,
) -> None:
    """
    Print input/output tensor information for a TFLite model.
    """

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print()
    print("Model:", model_path.name)

    print(
        "Input shape:",
        input_details["shape"],
    )

    print(
        "Input dtype:",
        input_details["dtype"],
    )

    print(
        "Input quantization:",
        input_details["quantization"],
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
        "Output quantization:",
        output_details["quantization"],
    )

    print(
        "Model size:",
        model_path.stat().st_size,
        "bytes",
    )


def main() -> None:
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Keras model:")
    print(KERAS_MODEL_PATH)

    model = keras.models.load_model(
        KERAS_MODEL_PATH
    )

    print()
    print("Converting Float32 TFLite model...")

    float32_model = convert_float32_model(
        model
    )

    FLOAT32_MODEL_PATH.write_bytes(
        float32_model
    )

    print(
        "Saved:",
        FLOAT32_MODEL_PATH,
    )

    print()
    print("Converting full INT8 TFLite model...")

    int8_model = convert_int8_model(
        model
    )

    INT8_MODEL_PATH.write_bytes(
        int8_model
    )

    print(
        "Saved:",
        INT8_MODEL_PATH,
    )

    inspect_tflite_model(
        FLOAT32_MODEL_PATH
    )

    inspect_tflite_model(
        INT8_MODEL_PATH
    )


if __name__ == "__main__":
    main()
