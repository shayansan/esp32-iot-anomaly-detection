from pathlib import Path
import csv
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_binary"
    / "train.npz"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_binary"
    / "subset_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_binary"
)

CORRELATION_THRESHOLD = 0.99


def main():
    data = np.load(TRAIN_PATH)

    x = data["x"].astype(np.float64)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    feature_names = metadata["feature_names"]

    print(f"Samples: {x.shape[0]}")
    print(f"Features: {x.shape[1]}")
    print(
        f"Correlation threshold: "
        f"{CORRELATION_THRESHOLD}"
    )

    correlation_matrix = np.corrcoef(
        x,
        rowvar=False,
    )

    high_pairs = []

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):

            correlation = correlation_matrix[i, j]

            if abs(correlation) >= CORRELATION_THRESHOLD:
                high_pairs.append(
                    (
                        feature_names[i],
                        feature_names[j],
                        float(correlation),
                    )
                )

    high_pairs.sort(
        key=lambda item: abs(item[2]),
        reverse=True,
    )

    print(
        f"Highly correlated pairs: "
        f"{len(high_pairs)}"
    )

    print("\nTop correlated pairs:")

    for feature_a, feature_b, correlation in high_pairs[:30]:
        print(
            f"{feature_a} <-> {feature_b}: "
            f"{correlation:.6f}"
        )

    csv_path = (
        OUTPUT_DIR
        / "high_correlation_pairs.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "feature_a",
                "feature_b",
                "correlation",
            ]
        )

        writer.writerows(high_pairs)

    summary = {
        "samples_analyzed": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "highly_correlated_pair_count": len(high_pairs),
    }

    summary_path = (
        OUTPUT_DIR
        / "correlation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )
        file.write("\n")

    print("\nSaved:")
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()