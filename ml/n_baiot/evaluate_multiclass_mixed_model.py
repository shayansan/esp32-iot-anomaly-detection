from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VARIANT = "full_115"

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_mixed_preprocessed"
    / VARIANT
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "multiclass_mixed_models"
    / VARIANT
    / "model.keras"
)

CLASS_NAMES = [
    "benign",
    "gafgyt_combo",
    "gafgyt_junk",
    "gafgyt_scan",
    "gafgyt_tcp",
    "gafgyt_udp",
    "mirai_ack",
    "mirai_scan",
    "mirai_syn",
    "mirai_udp",
    "mirai_udpplain",
]


def load_split(name):
    data = np.load(
        DATA_ROOT / f"{name}.npz"
    )

    return (
        data["x"].astype(np.float32),
        data["y"].astype(np.int32),
    )


def evaluate_split(
    model,
    split_name,
):
    x, y = load_split(split_name)

    probabilities = model.predict(
        x,
        batch_size=256,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    print(
        f"\n=== {split_name.upper()} ==="
    )

    print(
        "Accuracy:",
        accuracy_score(
            y,
            predictions,
        ),
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y,
            predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )


def main():
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    evaluate_split(
        model,
        "validation",
    )

    evaluate_split(
        model,
        "test",
    )


if __name__ == "__main__":
    main()