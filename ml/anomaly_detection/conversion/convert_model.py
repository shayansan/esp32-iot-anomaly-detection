#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[3]

KERAS_MODEL_PATH = (
    PROJECT_ROOT
    / "ml/models/baseline_anomaly_model.keras"
)

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "ml/data/processed/train.npz"
)

VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "ml/data/processed/validation.npz"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "ml/data/processed/test.npz"
)

EVALUATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "artifacts/anomaly_detection/host_results/"
    "model_test/evaluation_results.json"
)

FLOAT_MODEL_PATH = (
    PROJECT_ROOT
    / "ml/models/baseline_anomaly_float32.tflite"
)

INT8_MODEL_PATH = (
    PROJECT_ROOT
    / "ml/models/baseline_anomaly_int8.tflite"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts/anomaly_detection/host_results/conversion"
)

REPRESENTATIVE_SAMPLE_COUNT = 500


def load_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        features = archive[
            "features"
        ].astype(np.float32)

        labels = archive[
            "labels"
        ].astype(np.int8)

    return features, labels

def choose_best_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            probabilities,
        )
    )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_scores = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])

def load_threshold() -> float:
    if not EVALUATION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Evaluation results not found: "
            f"{EVALUATION_RESULTS_PATH}"
        )

    with EVALUATION_RESULTS_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    return float(results["selected_threshold"])


def create_representative_samples(
    training_features: np.ndarray,
) -> np.ndarray:
    sample_count = min(
        REPRESENTATIVE_SAMPLE_COUNT,
        len(training_features),
    )

    indexes = np.linspace(
        0,
        len(training_features) - 1,
        num=sample_count,
        dtype=np.int32,
    )

    return training_features[indexes]


def representative_dataset_generator(
    representative_samples: np.ndarray,
):
    def generator():
        for sample in representative_samples:
            yield [
                sample.reshape(
                    1,
                    -1,
                ).astype(np.float32)
            ]

    return generator


def convert_float_model(
    model: tf.keras.Model,
) -> bytes:
    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    return converter.convert()


def convert_int8_model(
    model: tf.keras.Model,
    representative_samples: np.ndarray,
) -> bytes:
    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        representative_dataset_generator(
            representative_samples
        )
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def quantize_input(
    input_values: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError(
            "Input quantization scale must be positive"
        )

    quantized = np.round(
        input_values / scale + zero_point
    )

    quantized = np.clip(
        quantized,
        -128,
        127,
    )

    return quantized.astype(np.int8)


def dequantize_output(
    quantized_values: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError(
            "Output quantization scale must be positive"
        )

    return (
        quantized_values.astype(np.float32)
        - zero_point
    ) * scale


def run_tflite_model(
    model_path: Path,
    features: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
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

    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    output_scale, output_zero_point = (
        output_details["quantization"]
    )

    probabilities: list[float] = []

    for sample in features:
        input_sample = sample.reshape(
            input_details["shape"]
        )

        if input_dtype == np.int8:
            input_sample = quantize_input(
                input_sample,
                float(input_scale),
                int(input_zero_point),
            )
        else:
            input_sample = input_sample.astype(
                input_dtype
            )

        interpreter.set_tensor(
            input_details["index"],
            input_sample,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        if output_dtype == np.int8:
            output = dequantize_output(
                output,
                float(output_scale),
                int(output_zero_point),
            )

        probabilities.append(
            float(output.reshape(-1)[0])
        )

    model_information = {
        "input_shape": (
            input_details["shape"].tolist()
        ),
        "input_dtype": str(input_dtype),
        "input_scale": float(input_scale),
        "input_zero_point": int(
            input_zero_point
        ),
        "output_shape": (
            output_details["shape"].tolist()
        ),
        "output_dtype": str(output_dtype),
        "output_scale": float(output_scale),
        "output_zero_point": int(
            output_zero_point
        ),
    }

    return (
        np.asarray(
            probabilities,
            dtype=np.float32,
        ),
        model_information,
    )


def calculate_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    true_positive = int(
        np.sum(
            (labels == 1)
            & (predictions == 1)
        )
    )

    true_negative = int(
        np.sum(
            (labels == 0)
            & (predictions == 0)
        )
    )

    false_positive = int(
        np.sum(
            (labels == 0)
            & (predictions == 1)
        )
    )

    false_negative = int(
        np.sum(
            (labels == 1)
            & (predictions == 0)
        )
    )

    accuracy = (
        true_positive + true_negative
    ) / len(labels)

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1_score = (
        2.0 * precision * recall
        / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def compare_probabilities(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    differences = np.abs(
        reference - candidate
    )

    return {
        "mean_absolute_difference": float(
            differences.mean()
        ),
        "maximum_absolute_difference": float(
            differences.max()
        ),
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Keras model not found: "
            f"{KERAS_MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH
    )

    train_features, _ = load_dataset(
    TRAIN_DATA_PATH
)

    validation_features, validation_labels = (
        load_dataset(
            VALIDATION_DATA_PATH
        )
    )

    test_features, test_labels = load_dataset(
        TEST_DATA_PATH
    )

    selected_threshold = load_threshold()

    representative_samples = (
        create_representative_samples(
            train_features
        )
    )

    print("Converting float32 model...")

    float_model = convert_float_model(
        model
    )

    FLOAT_MODEL_PATH.write_bytes(
        float_model
    )

    print("Converting fully quantized int8 model...")

    int8_model = convert_int8_model(
        model,
        representative_samples,
    )

    INT8_MODEL_PATH.write_bytes(
        int8_model
    )

    print("Running Keras test inference...")

    keras_probabilities = (
        model.predict(
            test_features,
            verbose=0,
        )
        .reshape(-1)
        .astype(np.float32)
    )

    print("Running float32 LiteRT inference...")

    (
        float_probabilities,
        float_model_information,
    ) = run_tflite_model(
        FLOAT_MODEL_PATH,
        test_features,
    )

    print("Running int8 LiteRT inference...")

    (
        int8_probabilities,
        int8_model_information,
    ) = run_tflite_model(
        INT8_MODEL_PATH,
        test_features,
    )

    print(
        "Running int8 LiteRT validation inference..."
    )

    (
        int8_validation_probabilities,
        _,
    ) = run_tflite_model(
        INT8_MODEL_PATH,
        validation_features,
    )

    int8_deployment_threshold = (
        choose_best_threshold(
            validation_labels,
            int8_validation_probabilities,
        )
    )

    print(
        "Int8 deployment threshold:",
        int8_deployment_threshold,
    )

    keras_metrics = calculate_binary_metrics(
        test_labels,
        keras_probabilities,
        selected_threshold,
    )

    float_metrics = calculate_binary_metrics(
        test_labels,
        float_probabilities,
        selected_threshold,
    )

    int8_metrics = calculate_binary_metrics(
    test_labels,
    int8_probabilities,
    int8_deployment_threshold,
    )

    float_comparison = compare_probabilities(
        keras_probabilities,
        float_probabilities,
    )

    int8_comparison = compare_probabilities(
        keras_probabilities,
        int8_probabilities,
    )

    results = {
        "tensorflow_version": tf.__version__,
        "selected_threshold": (
            selected_threshold
        ),
        "representative_samples": int(
            len(representative_samples)
        ),
        "keras_model": {
            "path": str(KERAS_MODEL_PATH),
            "size_bytes": int(
                KERAS_MODEL_PATH.stat().st_size
            ),
            "test_metrics": keras_metrics,
        },
        "float32_tflite_model": {
            "path": str(FLOAT_MODEL_PATH),
            "size_bytes": len(float_model),
            "tensor_information": (
                float_model_information
            ),
            "test_metrics": float_metrics,
            "comparison_with_keras": (
                float_comparison
            ),
        },
        "int8_tflite_model": {
    "path": str(INT8_MODEL_PATH),
    "size_bytes": len(int8_model),

    "deployment_threshold": (
        int8_deployment_threshold
    ),

    "threshold_selected_from": (
        "int8 validation probabilities"
    ),

    "tensor_information": (
        int8_model_information
    ),

    "test_metrics": int8_metrics,

    "comparison_with_keras": (
        int8_comparison
    ),
    },
    }

    results_path = (
        OUTPUT_DIRECTORY
        / "conversion_results.json"
    )

    with results_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    probability_table = np.column_stack(
        [
            test_labels,
            keras_probabilities,
            float_probabilities,
            int8_probabilities,
        ]
    )

    np.savetxt(
        OUTPUT_DIRECTORY
        / "probability_comparison.csv",
        probability_table,
        delimiter=",",
        header=(
            "label,"
            "keras_probability,"
            "float32_tflite_probability,"
            "int8_tflite_probability"
        ),
        comments="",
    )

    print("\nConversion complete")
    print("=" * 60)

    print(
        f"Keras model size: "
        f"{KERAS_MODEL_PATH.stat().st_size} bytes"
    )

    print(
        f"Float32 model size: "
        f"{len(float_model)} bytes"
    )

    print(
        f"Int8 model size: "
        f"{len(int8_model)} bytes"
    )

    print("\nFloat32 tensor information:")
    print(
        json.dumps(
            float_model_information,
            indent=4,
        )
    )

    print("\nInt8 tensor information:")
    print(
        json.dumps(
            int8_model_information,
            indent=4,
        )
    )

    print("\nKeras metrics:")
    print(
        json.dumps(
            keras_metrics,
            indent=4,
        )
    )

    print("\nFloat32 LiteRT metrics:")
    print(
        json.dumps(
            float_metrics,
            indent=4,
        )
    )

    print("\nInt8 LiteRT metrics:")
    print(
        json.dumps(
            int8_metrics,
            indent=4,
        )
    )

    print(
        "\nResults saved to:\n"
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()