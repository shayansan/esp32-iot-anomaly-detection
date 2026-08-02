#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml/models/baseline_anomaly_model.keras"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "ml/data/processed/validation.npz"
)

TEST_FILE = (
    PROJECT_ROOT
    / "ml/data/processed/test.npz"
)

RAW_TEST_FILE = (
    PROJECT_ROOT
    / "ml/data/raw/test_seed999.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "ml/evaluation/model_test"
)


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
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            probabilities,
        )
    )

    usable_precision = precision[:-1]
    usable_recall = recall[:-1]

    denominator = (
        usable_precision + usable_recall
    )

    f1_scores = np.divide(
        2.0
        * usable_precision
        * usable_recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(
        np.argmax(f1_scores)
    )

    best_threshold = float(
        thresholds[best_index]
    )

    return (
        best_threshold,
        thresholds,
        usable_precision,
        usable_recall,
        f1_scores,
    )


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def identify_anomaly_type(
    row: pd.Series,
) -> str:
    if row["anomaly"] == 0:
        return "normal"

    if row["temperature_c"] > 30.0:
        return "temperature_spike"

    if row["humidity_percent"] < 30.0:
        return "humidity_drop"

    if np.isclose(
        row["light_raw"],
        1023.0,
        atol=0.01,
    ):
        return "light_saturation"

    return "unknown_anomaly"


def create_per_type_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    anomaly_rows = results[
        results["anomaly"] == 1
    ]

    records = []

    for anomaly_type, group in anomaly_rows.groupby(
        "anomaly_type"
    ):
        total = len(group)

        detected = int(
            group["predicted_anomaly"].sum()
        )

        missed = total - detected

        recall = (
            detected / total
            if total > 0
            else 0.0
        )

        records.append({
            "anomaly_type": anomaly_type,
            "total_samples": total,
            "detected_samples": detected,
            "missed_samples": missed,
            "recall": recall,
        })

    return pd.DataFrame(records)


def plot_threshold_results(
    threshold_table: pd.DataFrame,
    selected_threshold: float,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        threshold_table["threshold"],
        threshold_table["precision"],
        label="Precision",
    )

    axis.plot(
        threshold_table["threshold"],
        threshold_table["recall"],
        label="Recall",
    )

    axis.plot(
        threshold_table["threshold"],
        threshold_table["f1_score"],
        label="F1-score",
    )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            f"Selected threshold "
            f"{selected_threshold:.4f}"
        ),
    )

    axis.set_title(
        "Validation Metrics by Threshold"
    )

    axis.set_xlabel("Threshold")
    axis.set_ylabel("Metric value")

    axis.set_ylim(0.0, 1.05)
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "validation_threshold_metrics.png",
        dpi=150,
    )

    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
) -> None:
    figure, axis = plt.subplots(
        figsize=(6, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Normal",
            "Anomaly",
        ],
    )

    display.plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title(
        "Test Dataset Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "test_confusion_matrix.png",
        dpi=150,
    )

    plt.close(figure)


def plot_probability_distribution(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.hist(
        probabilities[labels == 0],
        bins=50,
        alpha=0.7,
        label="Normal",
    )

    axis.hist(
        probabilities[labels == 1],
        bins=50,
        alpha=0.7,
        label="Anomaly",
    )

    axis.axvline(
        threshold,
        linestyle="--",
        label=(
            f"Threshold {threshold:.4f}"
        ),
    )

    axis.set_title(
        "Test Prediction Probabilities"
    )

    axis.set_xlabel(
        "Predicted anomaly probability"
    )

    axis.set_ylabel("Number of samples")

    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "test_probability_distribution.png",
        dpi=150,
    )

    plt.close(figure)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    validation_features, validation_labels = (
        load_dataset(
            VALIDATION_FILE
        )
    )

    test_features, test_labels = (
        load_dataset(
            TEST_FILE
        )
    )

    validation_probabilities = (
        model.predict(
            validation_features,
            verbose=0,
        )
        .reshape(-1)
    )

    (
        selected_threshold,
        thresholds,
        precision_values,
        recall_values,
        f1_values,
    ) = choose_best_threshold(
        validation_labels,
        validation_probabilities,
    )

    validation_predictions = (
        validation_probabilities
        >= selected_threshold
    ).astype(np.int8)

    test_probabilities = (
        model.predict(
            test_features,
            verbose=0,
        )
        .reshape(-1)
    )

    test_predictions = (
        test_probabilities
        >= selected_threshold
    ).astype(np.int8)

    validation_metrics = calculate_metrics(
        validation_labels,
        validation_predictions,
        validation_probabilities,
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_predictions,
        test_probabilities,
    )

    test_matrix = confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1],
    )

    report_text = classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "normal",
            "anomaly",
        ],
        digits=6,
        zero_division=0,
    )

    report_dictionary = classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "normal",
            "anomaly",
        ],
        output_dict=True,
        zero_division=0,
    )

    threshold_table = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision_values,
        "recall": recall_values,
        "f1_score": f1_values,
    })

    threshold_table.to_csv(
        OUTPUT_DIRECTORY
        / "validation_threshold_analysis.csv",
        index=False,
    )

    raw_test_data = pd.read_csv(
        RAW_TEST_FILE
    )

    if len(raw_test_data) != len(
        test_predictions
    ):
        raise ValueError(
            "Raw and processed test datasets "
            "have different lengths"
        )

    test_results = raw_test_data.copy()

    test_results["anomaly_type"] = (
        test_results.apply(
            identify_anomaly_type,
            axis=1,
        )
    )

    test_results[
        "anomaly_probability"
    ] = test_probabilities

    test_results[
        "predicted_anomaly"
    ] = test_predictions

    test_results[
        "prediction_correct"
    ] = (
        test_results["anomaly"]
        == test_results[
            "predicted_anomaly"
        ]
    )

    test_results.to_csv(
        OUTPUT_DIRECTORY
        / "test_predictions.csv",
        index=False,
    )

    per_type_summary = (
        create_per_type_summary(
            test_results
        )
    )

    per_type_summary.to_csv(
        OUTPUT_DIRECTORY
        / "per_anomaly_type_metrics.csv",
        index=False,
    )

    evaluation_results = {
        "model": str(MODEL_PATH),
        "selected_threshold": (
            selected_threshold
        ),
        "threshold_selected_from": (
            "validation dataset"
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "test_metrics": test_metrics,
        "test_confusion_matrix": (
            test_matrix.tolist()
        ),
        "classification_report": (
            report_dictionary
        ),
        "test_dataset_used_for_threshold": (
            False
        ),
    }

    with (
        OUTPUT_DIRECTORY
        / "evaluation_results.json"
    ).open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_results,
            file,
            indent=4,
        )

    (
        OUTPUT_DIRECTORY
        / "classification_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    plot_threshold_results(
        threshold_table,
        selected_threshold,
    )

    plot_confusion_matrix(
        test_matrix
    )

    plot_probability_distribution(
        test_labels,
        test_probabilities,
        selected_threshold,
    )

    print("\nSelected threshold")
    print("=" * 60)
    print(f"{selected_threshold:.6f}")

    print("\nValidation metrics")
    print("=" * 60)

    for name, value in validation_metrics.items():
        print(f"{name}: {value}")

    print("\nTest metrics")
    print("=" * 60)

    for name, value in test_metrics.items():
        print(f"{name}: {value}")

    print("\nClassification report")
    print("=" * 60)
    print(report_text)

    print("\nPer-anomaly-type results")
    print("=" * 60)
    print(
        per_type_summary.to_string(
            index=False
        )
    )

    print(
        "\nResults saved in:\n"
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()