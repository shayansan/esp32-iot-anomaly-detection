from pathlib import Path
import json

import numpy as np
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "multiclass_feature_selection"
    / "selected_features.json"
)

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_multiclass"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_preprocessed"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "multiclass_preprocessing"
)
def load_split(split_name):
    path = (
        INPUT_ROOT
        / f"{split_name}.npz"
    )

    data = np.load(path)

    return {
        "x": data["x"],
        "y": data["y"],
        "source_id": data["source_id"],
    }
def validate_array(
    split_name,
    x,
):
    if np.isnan(x).any():
        raise ValueError(
            f"{split_name}: NaN detected"
        )

    if np.isinf(x).any():
        raise ValueError(
            f"{split_name}: Inf detected"
        )

    if x.shape[1] != 115:
        raise ValueError(
            f"{split_name}: expected "
            f"115 features, got "
            f"{x.shape[1]}"
        )

def load_selected_indices():
    with FEATURE_SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    indices = selection[
        "selected_indices"
    ]

    if len(indices) != 54:
        raise ValueError(
            f"Expected 54 selected features, "
            f"got {len(indices)}"
        )

    return np.array(
        indices,
        dtype=np.int64,
    )

def save_split(
    variant_name,
    split_name,
    x,
    y,
    source_id,
):
    variant_root = (
        OUTPUT_ROOT
        / variant_name
    )

    variant_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        variant_root
        / f"{split_name}.npz"
    )

    np.savez_compressed(
        path,
        x=x.astype(np.float32),
        y=y,
        source_id=source_id,
    )

    print(
        f"Saved {variant_name}/"
        f"{split_name}: "
        f"{x.shape}"
    )

def preprocess_variant(
    variant_name,
    train,
    validation,
    test,
    feature_indices=None,
):
    if feature_indices is None:
        train_x = train["x"]
        validation_x = validation["x"]
        test_x = test["x"]
    else:
        train_x = train["x"][
            :,
            feature_indices,
        ]

        validation_x = validation["x"][
            :,
            feature_indices,
        ]

        test_x = test["x"][
            :,
            feature_indices,
        ]

    scaler = StandardScaler()

    scaler.fit(
        train_x.astype(
            np.float64
        )
    )

    train_scaled = scaler.transform(
        train_x
    )

    validation_scaled = scaler.transform(
        validation_x
    )

    test_scaled = scaler.transform(
        test_x
    )

    for name, x in [
        ("train", train_scaled),
        (
            "validation",
            validation_scaled,
        ),
        ("test", test_scaled),
    ]:
        if np.isnan(x).any():
            raise ValueError(
                f"{variant_name}/"
                f"{name}: NaN detected"
            )

        if np.isinf(x).any():
            raise ValueError(
                f"{variant_name}/"
                f"{name}: Inf detected"
            )

    save_split(
        variant_name,
        "train",
        train_scaled,
        train["y"],
        train["source_id"],
    )

    save_split(
        variant_name,
        "validation",
        validation_scaled,
        validation["y"],
        validation["source_id"],
    )

    save_split(
        variant_name,
        "test",
        test_scaled,
        test["y"],
        test["source_id"],
    )

    return {
        "scaler": scaler,
        "train_x": train_scaled,
        "validation_x":
            validation_scaled,
        "test_x": test_scaled,
    }

def save_scaler(
    variant_name,
    scaler,
    feature_count,
):
    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        ARTIFACT_ROOT
        / f"{variant_name}_scaler.json"
    )

    scaler_data = {
        "variant": variant_name,
        "feature_count":
            feature_count,
        "fit_split": "train",
        "mean":
            scaler.mean_.tolist(),
        "scale":
            scaler.scale_.tolist(),
        "variance":
            scaler.var_.tolist(),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scaler_data,
            file,
            indent=2,
        )

    print(
        f"Saved scaler: {path}"
    )

def print_variant_statistics(
    variant_name,
    result,
):
    print(
        f"\n=== {variant_name} ==="
    )

    for name, x in [
        (
            "train",
            result["train_x"],
        ),
        (
            "validation",
            result[
                "validation_x"
            ],
        ),
        (
            "test",
            result["test_x"],
        ),
    ]:
        print(
            f"\n{name}:"
        )

        print(
            "shape:",
            x.shape,
        )

        print(
            "mean(abs):",
            float(
                np.mean(
                    np.abs(
                        np.mean(
                            x,
                            axis=0,
                        )
                    )
                )
            ),
        )

        print(
            "std(mean):",
            float(
                np.mean(
                    np.std(
                        x,
                        axis=0,
                    )
                )
            ),
        )

        print(
            "NaN:",
            int(np.isnan(x).sum()),
        )

        print(
            "Inf:",
            int(np.isinf(x).sum()),
        )

def main():
    train = load_split(
        "train"
    )

    validation = load_split(
        "validation"
    )

    test = load_split(
        "test"
    )

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        validate_array(
            name,
            split["x"],
        )

    # Full 115-feature variant
    full_result = preprocess_variant(
        "full_115",
        train,
        validation,
        test,
    )

    save_scaler(
        "full_115",
        full_result["scaler"],
        115,
    )

    print_variant_statistics(
        "full_115",
        full_result,
    )

    # Reduced 54-feature variant
    selected_indices = (
        load_selected_indices()
    )

    reduced_result = (
        preprocess_variant(
            "reduced_54",
            train,
            validation,
            test,
            selected_indices,
        )
    )

    save_scaler(
        "reduced_54",
        reduced_result["scaler"],
        len(selected_indices),
    )

    print_variant_statistics(
        "reduced_54",
        reduced_result,
    )


if __name__ == "__main__":
    main()