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


def evaluate(config):
    data = np.load(
        DATA_ROOT
        / config
        / "test.npz"
    )

    x_test = data["x"]
    y_test = data["y"]

    model = tf.keras.models.load_model(
        MODEL_ROOT
        / config
        / "model.keras"
    )

    probabilities = model.predict(
        x_test,
        verbose=0,
    ).reshape(-1)

    predictions = (
        probabilities >= 0.5
    ).astype(np.uint8)

    tp = int(
        np.sum(
            (y_test == 1)
            & (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (y_test == 0)
            & (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (y_test == 0)
            & (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (y_test == 1)
            & (predictions == 0)
        )
    )

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

    result = {
        "configuration": config,
        "threshold": 0.5,
        "samples": int(len(y_test)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    output_path = (
        MODEL_ROOT
        / config
        / "evaluation_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )
        file.write("\n")

    return result


def main():
    for config in CONFIGS:
        result = evaluate(config)

        print("\n" + "=" * 50)
        print(config)

        print(
            f"TP: {result['tp']}"
        )
        print(
            f"TN: {result['tn']}"
        )
        print(
            f"FP: {result['fp']}"
        )
        print(
            f"FN: {result['fn']}"
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


if __name__ == "__main__":
    main()