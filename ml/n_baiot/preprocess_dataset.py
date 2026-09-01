from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_binary"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "preprocessed"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "preprocessing"
)

SUBSET_METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_binary"
    / "subset_metadata.json"
)

SELECTED_FEATURES_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_binary"
    / "selected_features.json"
)


def load_split(name):
    data = np.load(INPUT_DIR / f"{name}.npz")

    return {
        "x": data["x"],
        "y": data["y"],
        "source_id": data["source_id"],
    }


def fit_standardizer(x_train):
    x64 = x_train.astype(np.float64)

    mean = np.mean(
        x64,
        axis=0,
    )

    std = np.std(
        x64,
        axis=0,
    )

    std[std == 0] = 1.0

    return mean, std


def transform(x, mean, std):
    x_scaled = (
        x.astype(np.float64) - mean
    ) / std

    return x_scaled.astype(np.float32)


def save_configuration(
    name,
    feature_names,
    feature_indices,
    splits,
):
    output_dir = OUTPUT_ROOT / name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_x = splits["train"]["x"][
        :,
        feature_indices,
    ]

    mean, std = fit_standardizer(
        train_x
    )

    for split_name, split in splits.items():
        x_selected = split["x"][
            :,
            feature_indices,
        ]

        x_scaled = transform(
            x_selected,
            mean,
            std,
        )

        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            x=x_scaled,
            y=split["y"],
            source_id=split["source_id"],
        )

        print(
            f"{name} / {split_name}: "
            f"{x_scaled.shape}"
        )

        print(
            f"  mean(abs): "
            f"{np.abs(x_scaled.mean(axis=0)).mean():.6f}"
        )

        print(
            f"  std(mean): "
            f"{x_scaled.std(axis=0).mean():.6f}"
        )

    configuration = {
        "configuration": name,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_indices": feature_indices,
        "scaler": "standardization",
        "fit_on": "training_set_only",
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        ARTIFACT_DIR / f"{name}_scaler.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
        )
        file.write("\n")


def main():
    with open(
        SUBSET_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        subset_metadata = json.load(file)

    with open(
        SELECTED_FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        selected_metadata = json.load(file)

    all_features = subset_metadata[
        "feature_names"
    ]

    selected_features = selected_metadata[
        "selected_features"
    ]

    selected_indices = [
        all_features.index(name)
        for name in selected_features
    ]

    splits = {
        "train": load_split("train"),
        "validation": load_split(
            "validation"
        ),
        "test": load_split("test"),
    }

    print(
        "Creating full 115-feature version"
    )

    save_configuration(
        name="full_115",
        feature_names=all_features,
        feature_indices=list(
            range(len(all_features))
        ),
        splits=splits,
    )

    print(
        "\nCreating reduced 60-feature version"
    )

    save_configuration(
        name="reduced_60",
        feature_names=selected_features,
        feature_indices=selected_indices,
        splits=splits,
    )

    print(
        "\nPreprocessing completed successfully."
    )


if __name__ == "__main__":
    main()