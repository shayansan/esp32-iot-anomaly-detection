import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from preprocess_dataset import load_and_preprocess_data


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cifar10"
    / "models"
    / "cifar10_cnn.keras"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cifar10"
    / "host_results"
    / "keras_evaluation.json"
)

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cifar10"
    / "plots"
    / "keras_confusion_matrix.png"
)


def main() -> None:
    (
        _,
        _,
        x_test,
        y_test,
    ) = load_and_preprocess_data()

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    probabilities = model.predict(
        x_test,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    correct_predictions = int(
        np.sum(predictions == y_test)
    )

    incorrect_predictions = int(
        len(y_test) - correct_predictions
    )

    per_class_accuracy = {}

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        class_total = int(
            matrix[class_id].sum()
        )

        class_correct = int(
            matrix[class_id, class_id]
        )

        accuracy = (
            class_correct / class_total
        )

        per_class_accuracy[class_name] = accuracy

    model_size = MODEL_PATH.stat().st_size

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "correct_predictions": (
            correct_predictions
        ),
        "incorrect_predictions": (
            incorrect_predictions
        ),
        "model_size_bytes": model_size,
        "per_class_accuracy": (
            per_class_accuracy
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
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

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(matrix)

    plt.title(
        "CIFAR-10 Keras Confusion Matrix"
    )

    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "True class"
    )

    plt.xticks(
        range(10),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(10),
        CLASS_NAMES,
    )

    plt.colorbar()

    plt.tight_layout()

    CONFUSION_MATRIX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150,
    )

    plt.close()

    print("CIFAR-10 Keras evaluation")
    print("=" * 40)

    print(
        "Test accuracy:",
        test_accuracy,
    )

    print(
        "Test loss:",
        test_loss,
    )

    print(
        "Correct predictions:",
        correct_predictions,
    )

    print(
        "Incorrect predictions:",
        incorrect_predictions,
    )

    print(
        "Model size:",
        model_size,
        "bytes",
    )

    print(
        "\nPer-class accuracy:"
    )

    for class_name, accuracy in (
        per_class_accuracy.items()
    ):
        print(
            f"  {class_name:10s}: "
            f"{accuracy:.4f}"
        )

    print(
        "\nResults saved to:",
        RESULTS_PATH,
    )

    print(
        "Confusion matrix saved to:",
        CONFUSION_MATRIX_PATH,
    )


if __name__ == "__main__":
    main()
