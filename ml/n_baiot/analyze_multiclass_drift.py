from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_preprocessed"
    / "full_115"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_multiclass"
    / "subset_metadata.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "multiclass_drift"
)

TOP_FEATURES = 8


def load_split(name):
    data = np.load(
        DATA_ROOT / f"{name}.npz"
    )

    return (
        data["x"].astype(np.float64),
        data["y"].astype(np.int32),
    )


def main():
    train_x, train_y = load_split(
        "train"
    )

    validation_x, validation_y = (
        load_split(
            "validation"
        )
    )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    feature_names = metadata[
        "feature_names"
    ]

    class_mapping = {
        int(key): value
        for key, value
        in metadata[
            "class_mapping"
        ].items()
    }

    class_ids = sorted(
        class_mapping.keys()
    )

    train_centroids = []

    for class_id in class_ids:
        samples = train_x[
            train_y == class_id
        ]

        centroid = np.mean(
            samples,
            axis=0,
        )

        train_centroids.append(
            centroid
        )

    train_centroids = np.stack(
        train_centroids
    )

    print(
        "=== TRAIN CLASS CENTROID NEIGHBORS ==="
    )

    train_neighbor_results = []

    for class_id in class_ids:
        centroid = train_centroids[
            class_id
        ]

        distances = np.linalg.norm(
            train_centroids - centroid,
            axis=1,
        )

        distances[class_id] = np.inf

        nearest_id = int(
            np.argmin(distances)
        )

        result = {
            "class_id": class_id,
            "class_name":
                class_mapping[class_id],
            "nearest_other_class_id":
                nearest_id,
            "nearest_other_class":
                class_mapping[nearest_id],
            "distance":
                float(
                    distances[nearest_id]
                ),
        }

        train_neighbor_results.append(
            result
        )

        print(
            f"{class_mapping[class_id]:16s} "
            f"-> "
            f"{class_mapping[nearest_id]:16s} "
            f"distance="
            f"{distances[nearest_id]:.4f}"
        )

    print(
        "\n=== VALIDATION DRIFT ==="
    )

    drift_results = []

    for class_id in class_ids:
        train_centroid = (
            train_centroids[class_id]
        )

        validation_samples = (
            validation_x[
                validation_y == class_id
            ]
        )

        validation_centroid = np.mean(
            validation_samples,
            axis=0,
        )

        shift = (
            validation_centroid
            - train_centroid
        )

        absolute_shift = np.abs(
            shift
        )

        distances = np.linalg.norm(
            train_centroids
            - validation_centroid,
            axis=1,
        )

        nearest_id = int(
            np.argmin(distances)
        )

        same_class_distance = float(
            distances[class_id]
        )

        nearest_distance = float(
            distances[nearest_id]
        )

        top_indices = np.argsort(
            absolute_shift
        )[-TOP_FEATURES:][::-1]

        top_features = []

        for index in top_indices:
            top_features.append(
                {
                    "index":
                        int(index),
                    "name":
                        feature_names[index],
                    "signed_shift":
                        float(
                            shift[index]
                        ),
                    "absolute_shift":
                        float(
                            absolute_shift[
                                index
                            ]
                        ),
                }
            )

        result = {
            "class_id": class_id,
            "class_name":
                class_mapping[class_id],
            "mean_absolute_shift":
                float(
                    np.mean(
                        absolute_shift
                    )
                ),
            "l2_shift":
                float(
                    np.linalg.norm(
                        shift
                    )
                ),
            "nearest_train_class_id":
                nearest_id,
            "nearest_train_class":
                class_mapping[
                    nearest_id
                ],
            "nearest_train_distance":
                nearest_distance,
            "same_class_train_distance":
                same_class_distance,
            "top_shifted_features":
                top_features,
        }

        drift_results.append(
            result
        )

        print()
        print(
            class_mapping[class_id]
        )

        print(
            "  mean |shift|:",
            f"{result['mean_absolute_shift']:.4f}",
        )

        print(
            "  L2 shift:",
            f"{result['l2_shift']:.4f}",
        )

        print(
            "  nearest train class:",
            result[
                "nearest_train_class"
            ],
        )

        print(
            "  distance to own train class:",
            f"{same_class_distance:.4f}",
        )

        print(
            "  nearest distance:",
            f"{nearest_distance:.4f}",
        )

        print(
            "  top shifted features:"
        )

        for feature in top_features:
            print(
                "   ",
                feature["name"],
                f"{feature['signed_shift']:+.4f}",
            )

    output = {
        "variant": "full_115",
        "comparison":
            "train_vs_validation",
        "feature_space":
            "training-standardized",
        "train_class_neighbors":
            train_neighbor_results,
        "validation_drift":
            drift_results,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / "full_115_train_validation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\nSaved:",
        output_path,
    )


if __name__ == "__main__":
    main()