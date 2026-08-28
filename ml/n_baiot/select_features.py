from pathlib import Path
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_binary"
    / "selected_features.json"
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

    correlation_matrix = np.corrcoef(
        x,
        rowvar=False,
    )

    kept_indices = []
    removed_features = []

    for candidate_index, candidate_name in enumerate(
        feature_names
    ):
        redundant_with = None

        for kept_index in kept_indices:
            correlation = correlation_matrix[
                candidate_index,
                kept_index,
            ]

            if abs(correlation) >= CORRELATION_THRESHOLD:
                redundant_with = {
                    "feature": feature_names[kept_index],
                    "correlation": float(correlation),
                }
                break

        if redundant_with is None:
            kept_indices.append(candidate_index)

        else:
            removed_features.append(
                {
                    "feature": candidate_name,
                    "redundant_with": redundant_with["feature"],
                    "correlation": redundant_with["correlation"],
                }
            )

    selected_features = [
        feature_names[index]
        for index in kept_indices
    ]

    result = {
        "method": "correlation_redundancy_removal",
        "fitted_on": "training_set_only",
        "correlation_threshold": CORRELATION_THRESHOLD,
        "original_feature_count": len(feature_names),
        "selected_feature_count": len(selected_features),
        "removed_feature_count": len(removed_features),
        "selected_features": selected_features,
        "removed_features": removed_features,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )
        file.write("\n")

    print(
        f"Original features: "
        f"{len(feature_names)}"
    )

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    print(
        f"Removed features: "
        f"{len(removed_features)}"
    )

    print("\nRemoved features:")

    for item in removed_features:
        print(
            f"{item['feature']} "
            f"-> {item['redundant_with']} "
            f"(r={item['correlation']:.6f})"
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()