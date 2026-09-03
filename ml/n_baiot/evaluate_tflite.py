from pathlib import Path
import json

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


def load_test_data(config):
    data = np.load(
        DATA_ROOT
        / config
        / "test.npz"
    )

    return data["x"], data["y"]


def evaluate_model(
    model_path,
    x_test,
    y_test,
):
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]

    input_scale, input_zero_point = (
        input_details["quantization"]
    )

    output_scale, output_zero_point = (
        output_details["quantization"]
    )

    probabilities = []

    for sample in x_test:
        sample = sample.reshape(1, -1)

        if input_dtype == np.int8:
            sample = np.round(
                sample / input_scale
                + input_zero_point
            )

            sample = np.clip(
                sample,
                -128,
                127,
            ).astype(np.int8)

        else:
            sample = sample.astype(
                input_dtype
            )

        interpreter.set_tensor(
            input_details["index"],
            sample,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        ).reshape(-1)[0]

        if output_dtype == np.int8:
            output = (
                float(output)
                - output_zero_point
            ) * output_scale

        probabilities.append(
            float(output)
        )

    probabilities = np.array(
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.uint8)

    tp = int(np.sum(
        (y_test == 1)
        & (predictions == 1)
    ))

    tn = int(np.sum(
        (y_test == 0)
        & (predictions == 0)
    ))

    fp = int(np.sum(
        (y_test == 0)
        & (predictions == 1)
    ))

    fn = int(np.sum(
        (y_test == 1)
        & (predictions == 0)
    ))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

    accuracy = (
        (tp + tn)
        / len(y_test)
    )

    return {
        "input_dtype": str(input_dtype),
        "output_dtype": str(output_dtype),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    for config in CONFIGS:
        x_test, y_test = load_test_data(
            config
        )

        print("\n" + "=" * 60)
        print(config)

        results = {}

        for model_name in [
            "model_float32.tflite",
            "model_int8.tflite",
        ]:
            model_path = (
                MODEL_ROOT
                / config
                / model_name
            )

            result = evaluate_model(
                model_path,
                x_test,
                y_test,
            )

            results[model_name] = result

            print(f"\n{model_name}")

            print(
                f"Input dtype: "
                f"{result['input_dtype']}"
            )

            print(
                f"Output dtype: "
                f"{result['output_dtype']}"
            )

            print(
                f"Accuracy: "
                f"{result['accuracy']:.6f}"
            )

            print(
                f"Precision: "
                f"{result['precision']:.6f}"
            )

            print(
                f"Recall: "
                f"{result['recall']:.6f}"
            )

            print(
                f"F1: "
                f"{result['f1']:.6f}"
            )

            print(
                f"TP={result['tp']} "
                f"TN={result['tn']} "
                f"FP={result['fp']} "
                f"FN={result['fn']}"
            )

        output_path = (
            MODEL_ROOT
            / config
            / "tflite_evaluation.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=2,
            )
            file.write("\n")


if __name__ == "__main__":
    main()