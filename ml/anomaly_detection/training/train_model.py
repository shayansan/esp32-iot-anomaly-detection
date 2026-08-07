#!/usr/bin/env python3

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATA_DIRECTORY = (
    PROJECT_ROOT / "ml/data/processed"
)

MODEL_DIRECTORY = PROJECT_ROOT / "ml/models"

TRAINING_RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts/anomaly_detection/host_results/training"
)

TRAINING_PLOTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts/anomaly_detection/plots/training"
)

TRAIN_FILE = (
    PROCESSED_DATA_DIRECTORY / "train.npz"
)

VALIDATION_FILE = (
    PROCESSED_DATA_DIRECTORY / "validation.npz"
)

BEST_MODEL_PATH = (
    MODEL_DIRECTORY / "baseline_anomaly_model.keras"
)


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

NUMBER_OF_FEATURES = 3
BATCH_SIZE = 64
MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 0.001


def load_processed_dataset(
    name: str,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one processed .npz dataset.

    The returned arrays are:

    features:
        Shape: (number_of_samples, 3)
        Type: float32

    labels:
        Shape: (number_of_samples,)
        Type: float32
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset does not exist: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        if "features" not in archive:
            raise KeyError(
                f"{name} dataset does not contain "
                "'features'"
            )

        if "labels" not in archive:
            raise KeyError(
                f"{name} dataset does not contain "
                "'labels'"
            )

        features = archive[
            "features"
        ].astype(np.float32)

        labels = archive[
            "labels"
        ].astype(np.float32)

    if features.ndim != 2:
        raise ValueError(
            f"{name} features must be a 2D array"
        )

    if features.shape[1] != NUMBER_OF_FEATURES:
        raise ValueError(
            f"{name} must contain "
            f"{NUMBER_OF_FEATURES} features per sample, "
            f"but its shape is {features.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"{name} labels must be a 1D array"
        )

    if len(features) != len(labels):
        raise ValueError(
            f"{name} feature and label counts differ"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"{name} features contain NaN or infinity"
        )

    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError(
            f"{name} labels must contain only 0 and 1"
        )

    return features, labels


def calculate_class_weights(
    labels: np.ndarray,
) -> dict[int, float]:
    """
    Give more importance to the minority anomaly class.

    Formula:

        class weight =
            total samples /
            (number of classes × samples in class)
    """

    normal_count = int(
        np.sum(labels == 0)
    )

    anomaly_count = int(
        np.sum(labels == 1)
    )

    if normal_count == 0 or anomaly_count == 0:
        raise ValueError(
            "Training data must contain both classes"
        )

    total_count = len(labels)
    number_of_classes = 2

    normal_weight = (
        total_count
        / (number_of_classes * normal_count)
    )

    anomaly_weight = (
        total_count
        / (number_of_classes * anomaly_count)
    )

    return {
        0: float(normal_weight),
        1: float(anomaly_weight),
    }


def build_model() -> keras.Model:
    """
    Create a small binary-classification neural network.
    """

    model = keras.Sequential(
        [
            keras.Input(
                shape=(NUMBER_OF_FEATURES,),
                name="sensor_features",
            ),

            keras.layers.Dense(
                units=8,
                activation="relu",
                name="hidden_layer_1",
            ),

            keras.layers.Dense(
                units=4,
                activation="relu",
                name="hidden_layer_2",
            ),

            keras.layers.Dense(
                units=1,
                activation="sigmoid",
                name="anomaly_probability",
            ),
        ],
        name="esp32_anomaly_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),

        loss=keras.losses.BinaryCrossentropy(),

        metrics=[
            keras.metrics.BinaryAccuracy(
                name="accuracy",
                threshold=0.5,
            ),

            keras.metrics.Precision(
                name="precision",
                thresholds=0.5,
            ),

            keras.metrics.Recall(
                name="recall",
                thresholds=0.5,
            ),

            keras.metrics.AUC(
                name="roc_auc",
                curve="ROC",
            ),

            keras.metrics.AUC(
                name="pr_auc",
                curve="PR",
            ),
        ],
    )

    return model


def save_model_summary(
    model: keras.Model,
    output_path: Path,
) -> None:
    """
    Save model.summary() into a text file.
    """

    summary_lines: list[str] = []

    def capture_line(
        line: str,
        line_break: bool = True,
    ) -> None:
        del line_break
        summary_lines.append(line)

    model.summary(print_fn=capture_line)

    output_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def save_history(
    history: keras.callbacks.History,
    output_path: Path,
) -> None:
    """
    Save every recorded training metric as JSON.
    """

    serializable_history = {
        metric_name: [
            float(value)
            for value in metric_values
        ]
        for metric_name, metric_values
        in history.history.items()
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_history,
            file,
            indent=4,
        )


def plot_single_metric(
    history: keras.callbacks.History,
    training_metric: str,
    validation_metric: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """
    Plot one training metric and its validation equivalent.
    """

    epochs = range(
        1,
        len(history.history[training_metric]) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        epochs,
        history.history[training_metric],
        label="Training",
    )

    axis.plot(
        epochs,
        history.history[validation_metric],
        label="Validation",
    )

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)

    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)


def plot_precision_and_recall(
    history: keras.callbacks.History,
    output_path: Path,
) -> None:
    """
    Plot precision and recall together.
    """

    epochs = range(
        1,
        len(history.history["precision"]) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        epochs,
        history.history["precision"],
        label="Training precision",
    )

    axis.plot(
        epochs,
        history.history["val_precision"],
        label="Validation precision",
    )

    axis.plot(
        epochs,
        history.history["recall"],
        label="Training recall",
    )

    axis.plot(
        epochs,
        history.history["val_recall"],
        label="Validation recall",
    )

    axis.set_title(
        "Precision and Recall During Training"
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric value")

    axis.set_ylim(0.0, 1.05)
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)


def convert_results_to_float(
    results: dict[str, float],
) -> dict[str, float]:
    """
    Convert TensorFlow or NumPy values into normal floats.
    """

    return {
        name: float(value)
        for name, value in results.items()
    }


def main() -> None:
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for directory in (
        TRAINING_RESULTS_DIRECTORY,
        TRAINING_PLOTS_DIRECTORY,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # 1. Load only training and validation data
    # --------------------------------------------------------

    train_features, train_labels = (
        load_processed_dataset(
            name="training",
            path=TRAIN_FILE,
        )
    )

    validation_features, validation_labels = (
        load_processed_dataset(
            name="validation",
            path=VALIDATION_FILE,
        )
    )

    print("Training feature shape:")
    print(train_features.shape)

    print("\nValidation feature shape:")
    print(validation_features.shape)

    print("\nTraining class counts:")
    print(
        "Normal:",
        int(np.sum(train_labels == 0)),
    )
    print(
        "Anomaly:",
        int(np.sum(train_labels == 1)),
    )

    # --------------------------------------------------------
    # 2. Calculate class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_labels
    )

    print("\nClass weights:")
    print(
        f"Normal class weight:  "
        f"{class_weights[0]:.6f}"
    )
    print(
        f"Anomaly class weight: "
        f"{class_weights[1]:.6f}"
    )

    # --------------------------------------------------------
    # 3. Create the model
    # --------------------------------------------------------

    model = build_model()

    print("\nModel architecture:")
    model.summary()

    save_model_summary(
        model=model,
        output_path=(
            TRAINING_RESULTS_DIRECTORY
            / "model_summary.txt"
        ),
    )

    # --------------------------------------------------------
    # 4. Configure callbacks
    # --------------------------------------------------------

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.CSVLogger(
            filename=(
                TRAINING_RESULTS_DIRECTORY
                / "training_log.csv"
            ),
            separator=",",
            append=False,
        ),
    ]

    # --------------------------------------------------------
    # 5. Train the model
    # --------------------------------------------------------

    history = model.fit(
        train_features,
        train_labels,

        validation_data=(
            validation_features,
            validation_labels,
        ),

        epochs=MAXIMUM_EPOCHS,
        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        shuffle=True,
        verbose=2,
    )

    # --------------------------------------------------------
    # 6. Load the best saved model
    # --------------------------------------------------------

    best_model = keras.models.load_model(
        BEST_MODEL_PATH
    )

    # --------------------------------------------------------
    # 7. Basic training-stage evaluation
    # --------------------------------------------------------

    train_results = best_model.evaluate(
        train_features,
        train_labels,
        verbose=0,
        return_dict=True,
    )

    validation_results = best_model.evaluate(
        validation_features,
        validation_labels,
        verbose=0,
        return_dict=True,
    )

    train_results = convert_results_to_float(
        train_results
    )

    validation_results = convert_results_to_float(
        validation_results
    )

    # --------------------------------------------------------
    # 8. Save training history and metrics
    # --------------------------------------------------------

    save_history(
        history=history,
        output_path=(
            TRAINING_RESULTS_DIRECTORY
            / "training_history.json"
        ),
    )

    best_epoch = int(
        np.argmin(history.history["val_loss"]) + 1
    )

    training_metadata = {
        "tensorflow_version": tf.__version__,
        "random_seed": RANDOM_SEED,
        "input_features": 3,
        "architecture": [
            {
                "layer": "Dense",
                "units": 8,
                "activation": "relu",
            },
            {
                "layer": "Dense",
                "units": 4,
                "activation": "relu",
            },
            {
                "layer": "Dense",
                "units": 1,
                "activation": "sigmoid",
            },
        ],
        "parameter_count": int(
            best_model.count_params()
        ),
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "epochs_completed": len(
            history.history["loss"]
        ),
        "best_epoch": best_epoch,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "learning_rate": LEARNING_RATE,
        "class_weights": {
            str(class_id): weight
            for class_id, weight
            in class_weights.items()
        },
        "training_metrics": train_results,
        "validation_metrics": validation_results,
        "test_dataset_used": False,
    }

    metadata_path = (
        TRAINING_RESULTS_DIRECTORY
        / "training_metadata.json"
    )

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 9. Create training plots
    # --------------------------------------------------------

    plot_single_metric(
        history=history,
        training_metric="loss",
        validation_metric="val_loss",
        title="Training and Validation Loss",
        y_label="Binary cross-entropy loss",
        output_path=(
            TRAINING_PLOTS_DIRECTORY
            / "loss_curve.png"
        ),
    )

    plot_single_metric(
        history=history,
        training_metric="accuracy",
        validation_metric="val_accuracy",
        title="Training and Validation Accuracy",
        y_label="Accuracy",
        output_path=(
            TRAINING_PLOTS_DIRECTORY
            / "accuracy_curve.png"
        ),
    )

    plot_single_metric(
        history=history,
        training_metric="pr_auc",
        validation_metric="val_pr_auc",
        title="Training and Validation PR AUC",
        y_label="PR AUC",
        output_path=(
            TRAINING_PLOTS_DIRECTORY
            / "pr_auc_curve.png"
        ),
    )

    plot_precision_and_recall(
        history=history,
        output_path=(
            TRAINING_PLOTS_DIRECTORY
            / "precision_recall_curve.png"
        ),
    )

    # --------------------------------------------------------
    # 10. Print final results
    # --------------------------------------------------------

    print("\nTraining completed")
    print("=" * 60)

    print(
        f"Epochs completed: "
        f"{len(history.history['loss'])}"
    )

    print(f"Best epoch: {best_epoch}")

    print(
        f"Model parameters: "
        f"{best_model.count_params()}"
    )

    print("\nTraining metrics:")

    for name, value in train_results.items():
        print(f"{name}: {value:.6f}")

    print("\nValidation metrics:")

    for name, value in validation_results.items():
        print(f"{name}: {value:.6f}")

    print(
        "\nBest model saved to:\n"
        f"{BEST_MODEL_PATH}"
    )

    print("\nTraining outputs saved to:")
    print(
        f"Results: {TRAINING_RESULTS_DIRECTORY}"
    )
    print(
        f"Plots: {TRAINING_PLOTS_DIRECTORY}"
    )


if __name__ == "__main__":
    main()