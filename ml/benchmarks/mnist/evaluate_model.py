import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras

from preprocess_dataset import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "mnist"
    / "models"
    / "mnist_cnn.keras"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mnist"
    / "host_results"
)

PLOTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mnist"
    / "plots"
)

RESULTS_PATH = RESULTS_DIRECTORY / "keras_evaluation.json"

CONFUSION_MATRIX_PATH = (
    PLOTS_DIRECTORY
    / "keras_confusion_matrix.png"
)


def calculate_per_class_accuracy(
    matrix: np.ndarray,
) -> dict[str, float]:
    """
    Calculate classification accuracy for each MNIST digit.
    """

    per_class_accuracy = {}

    for digit in range(10):
        total = matrix[digit].sum()
        correct = matrix[digit, digit]

        accuracy = (
            float(correct / total)
            if total > 0
            else 0.0
        )

        per_class_accuracy[str(digit)] = accuracy

    return per_class_accuracy


def save_confusion_matrix(
    matrix: np.ndarray,
) -> None:
    """
    Save the confusion matrix as an image.
    """

    figure, axis = plt.subplots(figsize=(8, 8))

    image = axis.imshow(matrix)

    figure.colorbar(image, ax=axis)

    axis.set_title("MNIST Keras Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")

    axis.set_xticks(range(10))
    axis.set_yticks(range(10))

    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150,
    )

    plt.close(figure)


def main() -> None:
    _, _, x_test, y_test = load_and_preprocess_data()

    print("Loading model:")
    print(MODEL_PATH)

    model = keras.models.load_model(MODEL_PATH)

    probabilities = model.predict(
        x_test,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    per_class_accuracy = calculate_per_class_accuracy(
        matrix,
    )

    model_size_bytes = MODEL_PATH.stat().st_size

    results = {
        "model": "mnist_cnn.keras",
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "model_size_bytes": int(model_size_bytes),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": matrix.tolist(),
    }

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIRECTORY.mkdir(
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

    save_confusion_matrix(matrix)

    print()
    print("Test accuracy:", accuracy)
    print("Keras model size:", model_size_bytes, "bytes")

    print()
    print("Per-class accuracy:")

    for digit, class_accuracy in per_class_accuracy.items():
        print(
            f"  {digit}: "
            f"{class_accuracy:.4f}"
        )

    print()
    print("Results saved to:")
    print(RESULTS_PATH)

    print("Confusion matrix saved to:")
    print(CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
