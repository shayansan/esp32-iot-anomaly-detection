from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "preprocessed"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "models"
)

CONFIGS = [
    "full_115",
    "reduced_60",
]

REPRESENTATIVE_SAMPLES = 1000


def representative_dataset(config):
    data = np.load(
        DATA_ROOT
        / config
        / "train.npz"
    )

    x_train = data["x"]

    indices = np.linspace(
        0,
        len(x_train) - 1,
        REPRESENTATIVE_SAMPLES,
        dtype=int,
    )

    for index in indices:
        sample = x_train[
            index:index + 1
        ].astype(np.float32)

        yield [sample]


def convert_float32(model):
    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    return converter.convert()


def convert_int8(model, config):
    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        lambda:
        representative_dataset(config)
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def main():
    for config in CONFIGS:

        print("\n" + "=" * 60)
        print(f"Converting: {config}")

        model_dir = MODEL_ROOT / config

        model = tf.keras.models.load_model(
            model_dir / "model.keras"
        )

        float32_model = convert_float32(
            model
        )

        float32_path = (
            model_dir
            / "model_float32.tflite"
        )

        float32_path.write_bytes(
            float32_model
        )

        int8_model = convert_int8(
            model,
            config,
        )

        int8_path = (
            model_dir
            / "model_int8.tflite"
        )

        int8_path.write_bytes(
            int8_model
        )

        print(
            f"Float32 size: "
            f"{len(float32_model):,} bytes"
        )

        print(
            f"INT8 size: "
            f"{len(int8_model):,} bytes"
        )

    print(
        "\nConversion completed successfully."
    )


if __name__ == "__main__":
    main()