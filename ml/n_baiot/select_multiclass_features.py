from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_multiclass"
    / "train.npz"
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
    / "multiclass_feature_selection"
)

CORRELATION_THRESHOLD = 0.99


def main():
    data = np.load(TRAIN_PATH)

    x = data["x"].astype(
        np.float64
    )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    feature_names = metadata[
        "feature_names"
    ]

    if x.shape[1] != len(feature_names):
        raise ValueError(
            "Feature count mismatch"
        )

    print(
        "Calculating correlation matrix..."
    )

    correlation_matrix = np.corrcoef(
        x,
        rowvar=False,
    )

    selected_indices = []
    removed_features = []

    for current_index, current_name in enumerate(
        feature_names
    ):
        conflict = None

        for kept_index in selected_indices:
            correlation = correlation_matrix[
                current_index,
                kept_index,
            ]

            if (
                np.isfinite(correlation)
                and abs(correlation)
                >= CORRELATION_THRESHOLD
            ):
                conflict = {
                    "index": current_index,
                    "name": current_name,
                    "correlated_with_index":
                        kept_index,
                    "correlated_with":
                        feature_names[
                            kept_index
                        ],
                    "correlation":
                        float(correlation),
                }

                break

        if conflict is None:
            selected_indices.append(
                current_index
            )
        else:
            removed_features.append(
                conflict
            )

    selected_features = [
        feature_names[index]
        for index in selected_indices
    ]

    # Final safety check:
    # no two selected features should still
    # have |r| >= threshold.
    selected_corr = correlation_matrix[
        np.ix_(
            selected_indices,
            selected_indices,
        )
    ]

    upper_triangle = np.triu(
        np.ones(
            selected_corr.shape,
            dtype=bool,
        ),
        k=1,
    )

    remaining_high_corr = (
        np.abs(
            selected_corr[
                upper_triangle
            ]
        )
        >= CORRELATION_THRESHOLD
    )

    if np.any(remaining_high_corr):
        raise ValueError(
            "Selected feature set still "
            "contains highly correlated pairs"
        )

    output = {
        "selection_split": "train",
        "method":
            "greedy_original_feature_order",
        "correlation_threshold":
            CORRELATION_THRESHOLD,
        "original_feature_count":
            len(feature_names),
        "selected_feature_count":
            len(selected_features),
        "removed_feature_count":
            len(removed_features),
        "selected_indices":
            selected_indices,
        "selected_features":
            selected_features,
        "removed_features":
            removed_features,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / "selected_features.json"
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

    print()
    print(
        "Original features:",
        len(feature_names),
    )
    print(
        "Selected features:",
        len(selected_features),
    )
    print(
        "Removed features:",
        len(removed_features),
    )

    print(
        "\nFirst 20 selected features:"
    )

    for feature in selected_features[:20]:
        print(
            "-",
            feature,
        )

    print(
        "\nSaved:",
        output_path,
    )


if __name__ == "__main__":
    main()